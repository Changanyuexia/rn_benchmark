import json
import re
import nltk
from nltk.corpus import words

# 确保已下载 'words' 词典
nltk.download('words', quiet=True)

# 将词典转换为集合，便于快速查找
english_words = set(words.words())

def load_json(file_path):
    """从文件中加载 JSON 数据。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"从 {file_path} 加载 JSON 数据时出错：{e}")
        return []

def remove_emojis(text):
    """移除字符串中的表情符号。"""
    emoji_pattern = re.compile(
        "["
        u"\U0001F1E6-\U0001F1FF"  # 国旗（地区标志符号）
        u"\U0001F300-\U0001F5FF"  # 符号和象形文字
        u"\U0001F600-\U0001F64F"  # 表情符号
        u"\U0001F680-\U0001F6FF"  # 交通和地图符号
        u"\U0001F700-\U0001F77F"  # 炼金术符号
        u"\U0001F780-\U0001F7FF"  # 几何图形扩展
        u"\U0001F800-\U0001F8FF"  # 补充箭头-C
        u"\U0001F900-\U0001F9FF"  # 补充符号和象形文字
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols（象棋符号）
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A（符号和象形文字扩展-A）
        u"\U00002702-\U000027B0"  # Dingbats（杂项符号）
        u"\U000024C2-\U0001F251"
        u"\U0001F100-\U0001F1FF"  # Enclosed Alphanumeric Supplement
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def is_technical_line(line):
    """检查单行文本是否被视为技术性条目。"""
    line_stripped = line.strip('- ').strip()

    # 如果行为空，视为技术性条目
    if not line_stripped:
        return True

    # 提取单词
    words_in_line = re.findall(r'\b[a-zA-Z]+\b', line_stripped)

    # 如果只有一个单词，且是有效英文单词，保留这一行
    if len(words_in_line) == 1 and words_in_line[0].lower() in english_words:
        return False  # 这个单词行不会被删除

    # 检查是否包含长的十六进制字符串（长度超过32）
    if re.search(r'\b[0-9a-fA-F]{32,}\b', line_stripped):
        return True

    # 检查是否存在异常模式
    if contains_unusual_pattern(line_stripped):
        return True

    # 统计总字符数、字母字符数和数字字符数
    total_chars = len(line_stripped)
    letter_chars = len(re.findall(r'[a-zA-Z]', line_stripped))
    digit_chars = len(re.findall(r'\d', line_stripped))

    # 避免除以零
    if total_chars == 0:
        return True

    # 计算字母字符和数字字符占总字符的比例
    letter_ratio = letter_chars / total_chars
    digit_ratio = digit_chars / total_chars

    # 如果字母字符比例低于 30%，视为技术性条目
    if letter_ratio < 0.3:
        return True

    # 如果数字字符比例高于 50%，视为技术性条目
    if digit_ratio > 0.5:
        return True

    # 如果未找到任何单词，视为技术性条目
    if not words_in_line:
        return True

    return False

def contains_unusual_pattern(entry):
    """检查条目是否包含异常模式，例如缺少空格或过多的特殊字符。"""
    # 检查是否缺少空格且长度较长
    if ' ' not in entry and len(entry) > 50:
        return True

    # 计算特殊字符的比例
    special_chars = re.findall(r'[^\w\s]', entry)
    special_char_ratio = len(special_chars) / len(entry) if len(entry) > 0 else 0

    # 如果特殊字符比例高于 40%，视为技术性条目
    if special_char_ratio > 0.4:
        return True

    return False

def clean_release_note_text(note_text):
    """
    清理 release note 文本：
    - 移除表情符号。
    - 移除技术性行。
    - 移除问题引用（例如，#1234）。
    - 修剪空行。
    """
    lines = note_text.split('\n')  # 按换行符拆分
    cleaned_lines = []
    previous_line_empty = False  # 用于避免连续的空行

    for line in lines:
        # 移除表情符号
        line_no_emoji = remove_emojis(line)

        # 移除 # 后跟数字的内容（以及周围的逗号和空格）
        cleaned_line = re.sub(r',?\s*#\d+', '', line_no_emoji).strip()

        # 如果该行是技术性条目，则跳过
        if is_technical_line(cleaned_line):
            continue

        if cleaned_line:
            cleaned_lines.append(cleaned_line)
            previous_line_empty = False
        else:
            if not previous_line_empty:
                cleaned_lines.append('')
                previous_line_empty = True

    # 移除开头和结尾的空行
    while cleaned_lines and cleaned_lines[0] == '':
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == '':
        cleaned_lines.pop()

    return '\n'.join(cleaned_lines)  # 用换行符重新连接

def remove_duplicate_release_notes(data, filtered_out_entries):
    """根据 'release_note' 字段移除重复的 release notes。"""
    unique_entries = []
    seen_notes = set()

    for entry in data:
        release_note = entry.get('release_note')
        if release_note:
            if release_note not in seen_notes:
                unique_entries.append(entry)
                seen_notes.add(release_note)
            else:
                # 将被过滤的条目添加到 filtered_out_entries，原因是 'duplication'
                filtered_out_entries.append({'release_note': release_note, 'reason': 'duplication'})
        else:
            # 处理没有 release_note 的条目（如果需要）
            filtered_out_entries.append({'release_note': '', 'reason': 'no_release_note'})

    return unique_entries

def filter_release_notes_with_bot_keyword(data, filtered_out_entries, bot_keyword="ci: Workflow Bot"):
    """移除包含特定 bot 关键字的整个 release_note 条目。"""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if bot_keyword.lower() in release_note.lower():
            # 将被过滤的条目添加到 filtered_out_entries，原因是 'bot_clean'
            filtered_out_entries.append({'release_note': release_note, 'reason': 'bot_clean'})
        else:
            filtered_entries.append(entry)
    return filtered_entries

def filter_release_notes_with_more_than_40_sentences(data, filtered_out_entries):
    """过滤掉包含超过 40 句话的 release notes。"""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if count_sentences(release_note) > 40:
            # 将被过滤的条目添加到 filtered_out_entries，原因是 'too_many_sentences'
            filtered_out_entries.append({'release_note': release_note, 'reason': 'too_many_sentences'})
        else:
            filtered_entries.append(entry)
    return filtered_entries

def filter_single_sentence_entries(data, filtered_out_entries):
    """过滤掉只有一个句子的 release notes。"""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if count_sentences(release_note) > 1:
            filtered_entries.append(entry)
        else:
            # 将被过滤的条目添加到 filtered_out_entries，原因是 'single_sentence'
            filtered_out_entries.append({'release_note': release_note, 'reason': 'single_sentence'})
    return filtered_entries

def count_sentences(note_text):
    """
    统计文本中的句子数量。
    根据换行符 '\n' 分割文本，忽略空行、只包含 3 个单词以内的行，以及仅由特殊符号构成的行。
    """
    lines = note_text.split('\n')
    # 过滤有效的句子行
    valid_sentences = [
        line for line in lines
        if line.strip()                              # 非空行
        and not re.fullmatch(r'[^a-zA-Z0-9]+', line.strip())  # 不仅由特殊符号组成
    ]
    return len(valid_sentences)

def filter_commit_style_release_notes(data, filtered_out_entries):
    """过滤掉符合提交信息格式（连续 3 行提交格式）的 release notes。"""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if is_commit_style_release(release_note):
            # 将被过滤的条目添加到 filtered_out_entries，原因是 'commit_style'
            filtered_out_entries.append({'release_note': release_note, 'reason': 'commit_style'})
        else:
            filtered_entries.append(entry)
    return filtered_entries

def is_commit_style_release(note_text):
    """
    检测 release note 是否包含连续 3 行符合提交信息格式的条目。
    格式示例：哈希值（7到40位的十六进制字符） + "-" + 提交类型（如 chore, fix, docs, refactor, feat）。
    如果连续 3 行匹配此格式，则返回 True。
    """
    lines = note_text.split('\n')  # 按换行符分割文本
    # 正则表达式：匹配哈希值后跟 "-" 的提交信息格式
    pattern = re.compile(r'^[0-9a-f]{7,40} - (chore|fix|docs|refactor|feat|style|test|perf|ci|build).*', re.IGNORECASE)

    # 用于记录连续匹配的行数
    consecutive_matches = 0

    for line in lines:
        if pattern.match(line.strip()):
            consecutive_matches += 1  # 如果匹配，则增加计数
            if consecutive_matches == 3:
                return True  # 如果连续匹配 3 行，则返回 True
        else:
            consecutive_matches = 0  # 如果不匹配，重置计数

    return False  # 如果没有连续 3 行匹配，则返回 False

def filter_long_release_notes(data, filtered_out_entries, max_word_count=1000):
    """过滤掉单词数量超过指定数量的 release notes。"""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if count_words(release_note) <= max_word_count:
            filtered_entries.append(entry)
        else:
            # 将被过滤的条目添加到 filtered_out_entries，原因是 'too_many_words'
            filtered_out_entries.append({'release_note': release_note, 'reason': 'too_many_words'})
    return filtered_entries

def count_words(text):
    """统计文本中的单词数量。"""
    return len(text.split())

def filter_single_line_entries(data, filtered_out_entries):
    """过滤掉只有一行的 release notes。"""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if len(release_note.strip().split('\n')) > 1:
            filtered_entries.append(entry)
        else:
            # 将被过滤的条目添加到 filtered_out_entries，原因是 'single_line'
            filtered_out_entries.append({'release_note': release_note, 'reason': 'single_line'})
    return filtered_entries

def filter_by_word_count(data, min_word_count, filtered_out_entries):
    """过滤掉单词数量少于指定数量的 release notes。"""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if count_words(release_note) >= min_word_count:
            filtered_entries.append(entry)
        else:
            # 将被过滤的条目添加到 filtered_out_entries，原因是 'less_than_10_words'
            filtered_out_entries.append({'release_note': release_note, 'reason': 'less_than_10_words'})
    return filtered_entries

def clean_release_notes(data):
    """
    通过移除重复项、过滤包含 bot 关键字的 release notes、过滤过长的条目、
    过滤只有一个句子的条目、过滤提交信息格式的条目、过滤单行条目、
    按单词数过滤，并清理 release note 文本来处理数据。
    """
    filtered_out_entries = []  # 存储被过滤的条目及原因的列表

    # 移除重复的 release notes
    unique_entries = remove_duplicate_release_notes(data, filtered_out_entries)

    # 移除包含 bot 关键字的整个 release_note 条目
    bot_filtered_entries = filter_release_notes_with_bot_keyword(unique_entries, filtered_out_entries)

    # 过滤掉包含超过 40 句话的 release notes
    sentence_filtered_entries = filter_release_notes_with_more_than_40_sentences(bot_filtered_entries, filtered_out_entries)

    # 过滤掉只有一个句子的 release notes
    multi_sentence_entries = filter_single_sentence_entries(sentence_filtered_entries, filtered_out_entries)

    # 过滤掉符合提交信息格式的 release notes
    commit_style_filtered_entries = filter_commit_style_release_notes(multi_sentence_entries, filtered_out_entries)

    # 过滤掉单词数量超过 1000 的 release notes
    long_filtered_entries = filter_long_release_notes(commit_style_filtered_entries, filtered_out_entries, max_word_count=1000)

    # 过滤掉只有一行的 release notes
    multi_line_entries = filter_single_line_entries(long_filtered_entries, filtered_out_entries)

    # 过滤掉单词数量少于 10 的条目
    sufficient_word_entries = filter_by_word_count(
        multi_line_entries, min_word_count=10, filtered_out_entries=filtered_out_entries
    )

    # 对于每个清洗后的条目，清理 release note 文本并添加到 'cleaned_release_note'
    final_entries = []
    for entry in sufficient_word_entries:
        cleaned_note = clean_release_note_text(entry['release_note'])
        if cleaned_note.strip():  # 如果清洗后的内容不为空
            entry['cleaned_release_note'] = cleaned_note
            final_entries.append(entry)
        else:
            # 如果清洗后内容为空，则认为该条目无效
            filtered_out_entries.append({'release_note': entry['release_note'], 'reason': 'empty_after_cleaning'})

    return final_entries, filtered_out_entries

def save_cleaned_data(data, output_file_path):
    """将清洗后的 release notes 保存到 JSON 文件。"""
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_filtered_out_entries(filtered_entries, output_file_path):
    """将被过滤的 release notes 及原因保存到 JSON 文件。"""
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_entries, f, indent=4, ensure_ascii=False)

# 示例用法：
# 加载数据
data = load_json('rn.json')

# 清洗 release notes
cleaned_data, filtered_out_entries = clean_release_notes(data)

# 保存清洗后的数据，包含 'cleaned_release_note' 键
save_cleaned_data(cleaned_data, 'cleaned_rn.json')

# 保存被过滤的条目及原因
save_filtered_out_entries(filtered_out_entries, 'filtered_out_release_notes.json')

# 输出清洗后的条目数和被过滤的条目数
print(f"清洗后的 release notes 数量：{len(cleaned_data)}")
print(f"被过滤的 release notes 数量：{len(filtered_out_entries)}")

