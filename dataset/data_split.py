import json
import os
from tqdm import tqdm
import time
import ijson
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_split_commit_urls(file_path):
    """Load split commit URLs and convert the list to a set for O(1) lookup time."""
    with open(file_path, 'r', encoding='utf-8') as f:
        split_urls = json.load(f)
        # Convert lists to sets for faster lookup
        split_urls['train'] = set(split_urls['train'])
        split_urls['test'] = set(split_urls['test'])
        split_urls['val'] = set(split_urls['val'])
    return split_urls

def process_entry(entry, split_urls):
    """Determine the set membership of an entry and return the result."""
    commit_url = entry.get('commit_url', '')
    if not commit_url:
        return None, None

    json_entry = json.dumps(entry, ensure_ascii=False)
    if commit_url in split_urls['train']:
        return 'train', json_entry
    elif commit_url in split_urls['test']:
        return 'test', json_entry
    elif commit_url in split_urls['val']:
        return 'val', json_entry
    else:
        return None, None

def write_chunk_to_file(file, data_chunk):
    """Write a chunk of data to the specified file."""
    if data_chunk:
        file.write(',\n'.join(data_chunk) + ',\n')

def split_data_based_on_commit_urls(split_urls, input_file_path, output_dir):
    """Split data into training, test, and validation sets based on commit URLs using multithreading."""
    os.makedirs(output_dir, exist_ok=True)
    train_file_path = os.path.join(output_dir, 'train.json')
    test_file_path = os.path.join(output_dir, 'test.json')
    val_file_path = os.path.join(output_dir, 'val.json')

    # Open files for writing
    train_file = open(train_file_path, 'w', encoding='utf-8')
    test_file = open(test_file_path, 'w', encoding='utf-8')
    val_file = open(val_file_path, 'w', encoding='utf-8')

    # Write the beginning bracket
    train_file.write('[')
    test_file.write('[')
    val_file.write('[')

    # Calculate the total number of entries
    total_entries = len(split_urls['train']) + len(split_urls['test']) + len(split_urls['val'])
    tqdm.write(f"Total entries to process: {total_entries}")

    start_time = time.time()

    # Data buffers and batch size
    train_data, test_data, val_data = [], [], []
    batch_size = 10  # Adjust based on system memory

    with open(input_file_path, 'r', encoding='utf-8') as file:
        items = ijson.items(file, 'item')
        progress_bar = tqdm(desc=f"Processing {os.path.basename(input_file_path)}", total=total_entries, unit=" entries")
        executor = ThreadPoolExecutor(max_workers=8)  # Adjust the number of threads as needed
        futures = []

        for entry in items:
            future = executor.submit(process_entry, entry, split_urls)
            futures.append(future)

            if len(futures) >= batch_size:
                for future in as_completed(futures):
                    split_type, json_entry = future.result()
                    if split_type == 'train':
                        train_data.append(json_entry)
                    elif split_type == 'test':
                        test_data.append(json_entry)
                    elif split_type == 'val':
                        val_data.append(json_entry)
                    progress_bar.update(1)

                # Write data to files and reset buffers
                write_chunk_to_file(train_file, train_data)
                write_chunk_to_file(test_file, test_data)
                write_chunk_to_file(val_file, val_data)
                train_data, test_data, val_data = [], [], []
                futures = []

        # Process remaining futures
        for future in as_completed(futures):
            split_type, json_entry = future.result()
            if split_type == 'train':
                train_data.append(json_entry)
            elif split_type == 'test':
                test_data.append(json_entry)
            elif split_type == 'val':
                val_data.append(json_entry)
            progress_bar.update(1)

        # Write remaining data
        write_chunk_to_file(train_file, train_data)
        write_chunk_to_file(test_file, test_data)
        write_chunk_to_file(val_file, val_data)

        progress_bar.close()
        executor.shutdown()

    # Remove the trailing comma and close the files
    for f in [train_file, test_file, val_file]:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if file_size > 1:
            f.seek(f.tell() - 2, os.SEEK_SET)  # Remove the last comma
        f.write(']')
        f.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f"Data splitting completed successfully in {elapsed_time:.2f} seconds.")

# Example usage
split_commit_urls_path = 'split_commit_urls.json'  # Path to the file containing split commit URLs
diff_file_path = 'diff_rn.json'  # Path to the file to be split

# Load and convert commit URLs to a set
split_urls = load_split_commit_urls(split_commit_urls_path)

# Use multithreading to split data
tqdm.write("Splitting diff_rn.json")
split_data_based_on_commit_urls(split_urls, diff_file_path, 'split_data/diff_rn')

