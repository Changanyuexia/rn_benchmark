import json
import re
import nltk
from nltk.corpus import words

# Ensure 'words' corpus is downloaded
nltk.download('words', quiet=True)

# Convert word list to set for fast lookup
english_words = set(words.words())

def load_jsonl(file_path):
    """Load JSONL data from file, each line is a JSON object."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {e}")
    return data

def remove_emojis(text):
    """Remove emojis from a string."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F1E6-\U0001F1FF"  # flags (iOS)
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"
        u"\U0001F100-\U0001F1FF"  # Enclosed Alphanumeric Supplement
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def is_technical_line(line):
    """Check if a line is considered technical."""
    line_stripped = line.strip('- ').strip()
    if not line_stripped:
        return True
    words_in_line = re.findall(r'\b[a-zA-Z]+\b', line_stripped)
    if len(words_in_line) == 1 and words_in_line[0].lower() in english_words:
        return False
    if re.search(r'\b[0-9a-fA-F]{32,}\b', line_stripped):
        return True
    if contains_unusual_pattern(line_stripped):
        return True
    total_chars = len(line_stripped)
    letter_chars = len(re.findall(r'[a-zA-Z]', line_stripped))
    digit_chars = len(re.findall(r'\d', line_stripped))
    if total_chars == 0:
        return True
    letter_ratio = letter_chars / total_chars
    digit_ratio = digit_chars / total_chars
    if letter_ratio < 0.3:
        return True
    if digit_ratio > 0.5:
        return True
    if not words_in_line:
        return True
    return False

def contains_unusual_pattern(entry):
    """Check for unusual patterns, e.g., no spaces or too many special chars."""
    if ' ' not in entry and len(entry) > 50:
        return True
    special_chars = re.findall(r'[^\w\s]', entry)
    special_char_ratio = len(special_chars) / len(entry) if len(entry) > 0 else 0
    if special_char_ratio > 0.4:
        return True
    return False

def clean_release_note_text(note_text):
    """
    Clean release note text:
    - Remove emojis
    - Remove technical lines
    - Remove issue references (e.g., #1234)
    - Trim empty lines
    """
    lines = note_text.split('\n')
    cleaned_lines = []
    previous_line_empty = False
    for line in lines:
        line_no_emoji = remove_emojis(line)
        cleaned_line = re.sub(r',?\s*#\d+', '', line_no_emoji).strip()
        if is_technical_line(cleaned_line):
            continue
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
            previous_line_empty = False
        else:
            if not previous_line_empty:
                cleaned_lines.append('')
                previous_line_empty = True
    while cleaned_lines and cleaned_lines[0] == '':
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == '':
        cleaned_lines.pop()
    return '\n'.join(cleaned_lines)

def remove_duplicate_release_notes(data, filtered_out_entries):
    """Remove duplicate release notes by 'release_note' field."""
    unique_entries = []
    seen_notes = set()
    for entry in data:
        release_note = entry.get('release_note')
        if release_note:
            if release_note not in seen_notes:
                unique_entries.append(entry)
                seen_notes.add(release_note)
            else:
                filtered_out_entries.append({'release_note': release_note, 'reason': 'duplication'})
        else:
            filtered_out_entries.append({'release_note': '', 'reason': 'no_release_note'})
    return unique_entries

def filter_release_notes_with_bot_keyword(data, filtered_out_entries, bot_keyword="ci: Workflow Bot"):
    """Remove entries containing a specific bot keyword in release_note."""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if bot_keyword.lower() in release_note.lower():
            filtered_out_entries.append({'release_note': release_note, 'reason': 'bot_clean'})
        else:
            filtered_entries.append(entry)
    return filtered_entries

def filter_release_notes_with_more_than_40_sentences(data, filtered_out_entries):
    """Filter out release notes with more than 40 sentences."""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if count_sentences(release_note) > 40:
            filtered_out_entries.append({'release_note': release_note, 'reason': 'too_many_sentences'})
        else:
            filtered_entries.append(entry)
    return filtered_entries

def filter_single_sentence_entries(data, filtered_out_entries):
    """Filter out release notes with only one sentence."""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if count_sentences(release_note) > 1:
            filtered_entries.append(entry)
        else:
            filtered_out_entries.append({'release_note': release_note, 'reason': 'single_sentence'})
    return filtered_entries

def count_sentences(note_text):
    """
    Count the number of sentences in the text.
    Split by '\n', ignore empty lines and lines with only special chars.
    """
    lines = note_text.split('\n')
    valid_sentences = [
        line for line in lines
        if line.strip() and not re.fullmatch(r'[^a-zA-Z0-9]+', line.strip())
    ]
    return len(valid_sentences)

def filter_commit_style_release_notes(data, filtered_out_entries):
    """Filter out release notes with 3 consecutive commit-style lines."""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if is_commit_style_release(release_note):
            filtered_out_entries.append({'release_note': release_note, 'reason': 'commit_style'})
        else:
            filtered_entries.append(entry)
    return filtered_entries

def is_commit_style_release(note_text):
    """
    Detect if release note contains 3 consecutive commit-style lines.
    Example: hash (7-40 hex) + '-' + type (chore, fix, docs, etc.)
    """
    lines = note_text.split('\n')
    pattern = re.compile(r'^[0-9a-f]{7,40} - (chore|fix|docs|refactor|feat|style|test|perf|ci|build).*', re.IGNORECASE)
    consecutive_matches = 0
    for line in lines:
        if pattern.match(line.strip()):
            consecutive_matches += 1
            if consecutive_matches == 3:
                return True
        else:
            consecutive_matches = 0
    return False

def filter_long_release_notes(data, filtered_out_entries, max_word_count=1000):
    """Filter out release notes with more than max_word_count words."""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if count_words(release_note) <= max_word_count:
            filtered_entries.append(entry)
        else:
            filtered_out_entries.append({'release_note': release_note, 'reason': 'too_many_words'})
    return filtered_entries

def count_words(text):
    """Count the number of words in the text."""
    return len(text.split())

def filter_single_line_entries(data, filtered_out_entries):
    """Filter out release notes with only one line."""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if len(release_note.strip().split('\n')) > 1:
            filtered_entries.append(entry)
        else:
            filtered_out_entries.append({'release_note': release_note, 'reason': 'single_line'})
    return filtered_entries

def filter_by_word_count(data, min_word_count, filtered_out_entries):
    """Filter out release notes with fewer than min_word_count words."""
    filtered_entries = []
    for entry in data:
        release_note = entry.get('release_note', '')
        if count_words(release_note) >= min_word_count:
            filtered_entries.append(entry)
        else:
            filtered_out_entries.append({'release_note': release_note, 'reason': 'less_than_10_words'})
    return filtered_entries

def clean_release_notes(data):
    """
    Remove duplicates, filter by bot keyword, length, sentence count, commit style, single line, word count, and clean text.
    """
    filtered_out_entries = []
    unique_entries = remove_duplicate_release_notes(data, filtered_out_entries)
    bot_filtered_entries = filter_release_notes_with_bot_keyword(unique_entries, filtered_out_entries)
    sentence_filtered_entries = filter_release_notes_with_more_than_40_sentences(bot_filtered_entries, filtered_out_entries)
    multi_sentence_entries = filter_single_sentence_entries(sentence_filtered_entries, filtered_out_entries)
    commit_style_filtered_entries = filter_commit_style_release_notes(multi_sentence_entries, filtered_out_entries)
    long_filtered_entries = filter_long_release_notes(commit_style_filtered_entries, filtered_out_entries, max_word_count=1000)
    multi_line_entries = filter_single_line_entries(long_filtered_entries, filtered_out_entries)
    sufficient_word_entries = filter_by_word_count(
        multi_line_entries, min_word_count=10, filtered_out_entries=filtered_out_entries
    )
    final_entries = []
    for entry in sufficient_word_entries:
        cleaned_note = clean_release_note_text(entry['release_note'])
        if cleaned_note.strip():
            entry['cleaned_release_note'] = cleaned_note
            final_entries.append(entry)
        else:
            filtered_out_entries.append({'release_note': entry['release_note'], 'reason': 'empty_after_cleaning'})
    return final_entries, filtered_out_entries

def save_cleaned_data(data, output_file_path):
    """Save cleaned release notes to a JSON file."""
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_filtered_out_entries(filtered_entries, output_file_path):
    """Save filtered out release notes and reasons to a JSON file."""
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_entries, f, indent=4, ensure_ascii=False)

# Example usage:
# Load data from final_data.jsonl
all_data = load_jsonl('final_data.jsonl')

# Extract release note entries
rn_entries = [
    {"commit_url": entry["commit_url"], "release_note": entry.get("release_note", "")}
    for entry in all_data if entry.get("release_note", "").strip()
]

# Clean release notes
cleaned_data, filtered_out_entries = clean_release_notes(rn_entries)

# Save cleaned data and filtered out entries
save_cleaned_data(cleaned_data, 'cleaned_rn.json')
save_filtered_out_entries(filtered_out_entries, 'filtered_out_release_notes.json')

print(f"Number of cleaned release notes: {len(cleaned_data)}")
print(f"Number of filtered out release notes: {len(filtered_out_entries)}")

