import os
import csv
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.optim import AdamW

from torch.optim.lr_scheduler import StepLR

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


Config = {"batch_size": 8,
          "cuda_index": 0,
          "max_length": 512
          }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file_path):
        # combine input: original text with masked sbn
        #self.data = pd.read_csv(input_file_path)
        #print(f"Working on {len(self.data)} data")
        # Access the first and second columns as input and output
        #self.input_col = self.data.iloc[:, 0]
        #self.output_col = self.data.iloc[:, 1]
        chunks = []

        # Read the file in chunks
        for chunk in pd.read_csv(input_file_path, chunksize=1000):
            # Keep only the first two columns
            chunk = chunk.iloc[:, :2]
            chunks.append(chunk)

        # Concatenate all chunks into one DataFrame
        self.data = pd.concat(chunks, ignore_index=True)

        # Show the number of data points loaded
        print(f"Working on {len(self.data)} data")

        # Access the first and second columns as input and output
        self.input_col = self.data.iloc[:, 0]
        self.output_col = self.data.iloc[:, 1]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_val = self.input_col[idx]
        output_val = self.output_col[idx]

        # 检查并打印空值
        if pd.isnull(input_val) or pd.isnull(output_val):
            print(f"第{idx}行有空值: input={input_val}, output={output_val}")

        text = str(input_val).strip() if pd.notnull(input_val) else ""
        sbn = str(output_val).strip() if pd.notnull(output_val) else ""
        return text, sbn


def get_dataloader(input_file_path, batch_size=Config["batch_size"]):
    dataset = Dataset(input_file_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


class Generator:

    def __init__(self, lang, load_path=""):
        """
        :param train: train or test
        """
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', max_length=Config["max_length"])

        if len(load_path) == 0:
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', max_length=512)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(load_path)

        self.model.to(self.device)

    def evaluate(self, val_loader, save_path):
        # Initialize a list to store all data in the form of dictionaries
        data = []

        self.model.eval()
        with torch.no_grad():
            for i, (text, target) in enumerate(tqdm(val_loader)):
                x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)[
                    'input_ids'].to(
                    self.device)
                out_put = self.model.generate(x)
                for j in range(len(out_put)):
                    o = out_put[j]
                    pred_text = self.tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    # Use input text as a key and prediction as a value in the dictionary
                    data.append({'commits': text[j], 'release': pred_text})

        # Write all data to the CSV file at once
        with open(save_path, 'w', newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=['commits', 'release'])
            writer.writeheader()  # Write the header (column names)
            writer.writerows(data)  # Write all rows at once

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch, (text, target) in enumerate(val_loader):
                x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids'].to(self.device)
                y = self.tokenizer(target, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids'].to(self.device)

                output = self.model(x, labels=y)
                total_loss += output.loss.item()

        average_loss = total_loss / len(val_loader)
        return average_loss


    def train(self, train_loader, val_loader, lr, epoch_number, patience=5, step_size=10, gamma=0.2, save_path="", min_epoch=1, min_delta=0.001):
        optimizer = AdamW(self.model.parameters(), lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epoch_number):
            self.model.train()
            pbar = tqdm(train_loader)
            for batch, (text, target) in enumerate(pbar):
                x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids'].to(self.device)
                y = self.tokenizer(target, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids'].to(self.device)

                optimizer.zero_grad()
                output = self.model(x, labels=y)
                loss = output.loss
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Loss: {loss.item():.3f}")

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}, Current Learning Rate: {current_lr}")

            # Validation phase
            val_loss = self.validate(val_loader)
            print(f"val loss: {val_loss}")

            # Check if validation loss improved significantly
            loss_improvement = best_val_loss - val_loss
            if loss_improvement > min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if len(save_path) != 0:
                    self.model.save_pretrained(save_path)
            else:
                epochs_no_improve += 1

            # Adjust learning rate
            scheduler.step()

            # Early stopping check
            if epochs_no_improve == patience and epoch > min_epoch:
                print("Early stopping triggered")
                break
