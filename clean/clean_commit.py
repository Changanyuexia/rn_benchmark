import json
import re
import os
import random
from collections import defaultdict
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_commit_msg(msg):
    """Clean commit messages by removing '#number' tags."""
    return re.sub(r',?\s*#\d+', '', msg).strip()

def process_commit(commit):
    """Process and clean each commit's messages, but do not clean diffs."""
    original_msg = commit.get('msg', '')
    cleaned_msg = clean_commit_msg(original_msg)
    # Only keep the original diff, do not clean diff
    new_diffs = []
    for diff in commit.get('diff', []):
        new_diffs.append({
            "old_path": diff.get('old_path', ''),
            "new_path": diff.get('new_path', ''),
            "change_type": diff.get('change_type', ''),
            "diff": diff.get('diff', '')
        })
    commit['diff'] = new_diffs
    commit['cleaned_msg'] = cleaned_msg
    return commit

def process_and_save_data(commits_file, diffs_file):
    """
    Merge commits.json and diffs.jsonl by commit_url, clean data, and save as a JSONL file.
    Also output statistics and split_commit_urls.json.
    """
    try:
        # Build diff_map from diffs.jsonl
        diff_map = {}
        with open(diffs_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                diff_map[entry['commit_url']] = entry

        output_file = open('final_data.jsonl', 'w', encoding='utf-8')
        line_count = 0
        language_stats = defaultdict(lambda: {'release_note_count': 0, 'commit_count': 0, 'msg_count': 0, 'diff_count': 0, 'total_cleaned_release_note_tokens': 0, 'total_cleaned_msg_tokens': 0, 'total_cleaned_diff_tokens': 0, 'total_graph_commits_tokens': 0})
        with open(commits_file, 'r', encoding='utf-8') as file:
            commit_urls = []
            lines = file.readlines()
            total_lines = len(lines)
            for line in tqdm(lines, total=total_lines, desc="Processing JSON lines"):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                line_count += 1
                logging.info(f"Processing line {line_count}")
                try:
                    entry = json.loads(line)
                    commit_url = entry['commit_url']
                    diff_info = diff_map.get(commit_url, {})
                    if len(entry.get('commits', [])) > 250:
                        continue  # Skip entries with more than 250 commits
                    processed_commits = [process_commit(commit) for commit in entry['commits']]
                    entry['commits'] = processed_commits
                    commit_urls.append(commit_url)
                    # Merge fields for output
                    merged_data = {
                        'commit_url': commit_url,
                        'release_note': entry.get('release_note', ''),
                        'cleaned_release_note': entry.get('cleaned_release_note', ''),
                        'commits': [{'msg': commit['msg'], 'cleaned_msg': commit['cleaned_msg'], 'diff': commit['diff']} for commit in processed_commits],
                        'diff': diff_info.get('diff', ''),
                        'mod': diff_info.get('mod', []),
                        'diff_tokens': diff_info.get('diff_tokens', 0),
                        'graph_commits': entry.get('graph_commits', '')
                    }
                    output_file.write(json.dumps(merged_data, ensure_ascii=False) + '\n')
                    # Update statistics
                    language = entry.get('language', 'Unknown')
                    language_stats[language]['release_note_count'] += 1
                    language_stats[language]['commit_count'] += len(processed_commits)
                    language_stats[language]['msg_count'] += len([commit for commit in processed_commits if commit.get('msg', '').strip()])
                    language_stats[language]['diff_count'] += len([diff for commit in processed_commits for diff in commit.get('diff', []) if diff.get('diff', '').strip()])
                    language_stats[language]['total_cleaned_release_note_tokens'] += len(entry.get('cleaned_release_note', '').split())
                    language_stats[language]['total_graph_commits_tokens'] += len(entry.get('graph_commits', '').split())
                    for commit in processed_commits:
                        language_stats[language]['total_cleaned_msg_tokens'] += len(commit.get('cleaned_msg', '').split())
                        for diff in commit.get('diff', []):
                            language_stats[language]['total_cleaned_diff_tokens'] += len(diff.get('diff', '').split())
                except json.JSONDecodeError as e:
                    logging.error(f"JSON Decode Error: {e}")
        output_file.close()
        # Calculate average token counts and save statistics
        stats = {}
        for language, data in language_stats.items():
            stats[language] = {
                'total_release_notes': data['release_note_count'],
                'total_commits': data['commit_count'],
                'total_msgs': data['msg_count'],
                'total_diffs': data['diff_count']
            }
        average_stats = {
            'average_cleaned_release_note_tokens': sum(data['total_cleaned_release_note_tokens'] for data in language_stats.values()) / sum(data['release_note_count'] for data in language_stats.values()) if sum(data['release_note_count'] for data in language_stats.values()) > 0 else 0,
            'average_cleaned_msg_tokens': sum(data['total_cleaned_msg_tokens'] for data in language_stats.values()) / sum(data['msg_count'] for data in language_stats.values()) if sum(data['msg_count'] for data in language_stats.values()) > 0 else 0,
            'average_cleaned_diff_tokens': sum(data['total_cleaned_diff_tokens'] for data in language_stats.values()) / sum(data['diff_count'] for data in language_stats.values()) if sum(data['diff_count'] for data in language_stats.values()) > 0 else 0,
            'average_graph_commits_tokens': sum(data['total_graph_commits_tokens'] for data in language_stats.values()) / sum(data['release_note_count'] for data in language_stats.values()) if sum(data['release_note_count'] for data in language_stats.values()) > 0 else 0,
        }
        with open('language_statistics.json', 'w', encoding='utf-8') as stats_file:
            json.dump({'language_stats': stats, 'average_stats': average_stats}, stats_file, indent=4, ensure_ascii=False)
        logging.info("Data processing, file generation, and statistics calculation completed successfully.")
        # Split commit URLs into train, test, and validation sets
        random.shuffle(commit_urls)
        total = len(commit_urls)
        train_split = int(total * 0.8)
        test_split = int(total * 0.9)
        split_data = {
            'train': commit_urls[:train_split],
            'test': commit_urls[train_split:test_split],
            'val': commit_urls[test_split:]
        }
        with open('split_commit_urls.json', 'w', encoding='utf-8') as split_file:
            json.dump(split_data, split_file, indent=4, ensure_ascii=False)
        logging.info("Data splitting completed successfully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

# Example usage
commits_file = 'commits.json'
diffs_file = 'diffs.jsonl'
process_and_save_data(commits_file, diffs_file)

