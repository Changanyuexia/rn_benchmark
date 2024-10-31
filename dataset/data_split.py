import json
import os
from tqdm import tqdm
import time
import ijson
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_split_commit_urls(file_path):
    """加载拆分后的提交URL，并将列表转换为集合以加速查找。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        split_urls = json.load(f)
        # 将列表转换为集合以实现O(1)的查找时间
        split_urls['train'] = set(split_urls['train'])
        split_urls['test'] = set(split_urls['test'])
        split_urls['val'] = set(split_urls['val'])
    return split_urls

def process_entry(entry, split_urls):
    """确定条目所属的集合并返回结果。"""
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
    """将数据块写入指定文件。"""
    if data_chunk:
        file.write(',\n'.join(data_chunk) + ',\n')

def split_data_based_on_commit_urls(split_urls, input_file_path, output_dir):
    """使用多线程根据提交URL将数据拆分为训练、测试和验证集。"""
    os.makedirs(output_dir, exist_ok=True)
    train_file_path = os.path.join(output_dir, 'train.json')
    test_file_path = os.path.join(output_dir, 'test.json')
    val_file_path = os.path.join(output_dir, 'val.json')

    # 打开文件以进行写入
    train_file = open(train_file_path, 'w', encoding='utf-8')
    test_file = open(test_file_path, 'w', encoding='utf-8')
    val_file = open(val_file_path, 'w', encoding='utf-8')

    # 写入开头的方括号
    train_file.write('[')
    test_file.write('[')
    val_file.write('[')

    # 计算总条目数
    total_entries = len(split_urls['train']) + len(split_urls['test']) + len(split_urls['val'])
    tqdm.write(f"Total entries to process: {total_entries}")

    start_time = time.time()

    # 数据缓冲区和批处理大小
    train_data, test_data, val_data = [], [], []
    batch_size = 10  # 可根据系统内存进行调整

    with open(input_file_path, 'r', encoding='utf-8') as file:
        items = ijson.items(file, 'item')
        progress_bar = tqdm(desc=f"Processing {os.path.basename(input_file_path)}", total=total_entries, unit=" entries")
        executor = ThreadPoolExecutor(max_workers=8)  # 根据需要调整线程数
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

                # 将数据写入文件并重置缓冲区
                write_chunk_to_file(train_file, train_data)
                write_chunk_to_file(test_file, test_data)
                write_chunk_to_file(val_file, val_data)
                train_data, test_data, val_data = [], [], []
                futures = []

        # 处理剩余的未来对象
        for future in as_completed(futures):
            split_type, json_entry = future.result()
            if split_type == 'train':
                train_data.append(json_entry)
            elif split_type == 'test':
                test_data.append(json_entry)
            elif split_type == 'val':
                val_data.append(json_entry)
            progress_bar.update(1)

        # 写入剩余的数据
        write_chunk_to_file(train_file, train_data)
        write_chunk_to_file(test_file, test_data)
        write_chunk_to_file(val_file, val_data)

        progress_bar.close()
        executor.shutdown()

    # 移除尾部逗号并关闭文件
    for f in [train_file, test_file, val_file]:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if file_size > 1:
            f.seek(f.tell() - 2, os.SEEK_SET)  # 移除最后的逗号
        f.write(']')
        f.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f"Data splitting completed successfully in {elapsed_time:.2f} seconds.")

# 示例用法
split_commit_urls_path = 'split_commit_urls.json'  # 拆分后的提交URL文件路径
diff_file_path = 'diff_rn.json'  # 要拆分的文件路径

# 加载并转换提交URL为集合
split_urls = load_split_commit_urls(split_commit_urls_path)

# 使用多线程拆分数据
tqdm.write("Splitting diff_rn.json")
split_data_based_on_commit_urls(split_urls, diff_file_path, 'split_data/diff_rn')

