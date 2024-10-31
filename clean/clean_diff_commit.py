import json
import re
import os
import random
from collections import defaultdict
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_diff(diff_text):
    """Keep lines containing '+' or '-' and ignore blocks that are purely additions or deletions."""
    if isinstance(diff_text, str):
        lines = diff_text.split('\n')
        filtered_lines = [line for line in lines if line.startswith('+') or line.startswith('-')]
        has_addition = any(line.startswith('+') for line in filtered_lines)
        has_deletion = any(line.startswith('-') for line in filtered_lines)
        return '\n'.join(filtered_lines) if has_addition and has_deletion else ""
    return ""  # Return empty string if not a string input

def clean_msg(msg):
    """Clean commit messages by removing '#number' tags."""
    return re.sub(r',?\s*#\d+', '', msg).strip()

def process_commit(commit):
    """Process and clean each commit's diffs while retaining both original and cleaned metadata and messages."""
    original_msg = commit.get('msg', '')
    cleaned_msg = clean_msg(original_msg)
    new_diffs = []
    for diff in commit.get('diff', []):
        original_diff = diff['diff']
        cleaned_diff = clean_diff(original_diff)
        new_diffs.append({
            "old_path": diff.get('old_path', ''),
            "new_path": diff.get('new_path', ''),
            "change_type": diff.get('change_type', ''),
            "diff": original_diff,  # Preserve the original diff
            "cleaned_diff": cleaned_diff  # Store the cleaned diff
        })
    commit['diff'] = new_diffs
    commit['cleaned_msg'] = cleaned_msg
    return commit

def process_and_save_data(file_path):
    """Read and process JSON file line by line, clean data, split it into different views, and save the results."""
    try:
        final_data_file = open('final_data.json', 'w', encoding='utf-8')
        graph_rn_commits_file = open('graph_rn_commits.json', 'w', encoding='utf-8')
        rn_commits_file = open('rn_commits.json', 'w', encoding='utf-8')
        diff_rn_file = open('diff_rn.json', 'w', encoding='utf-8')

        final_data_file.write('[')
        graph_rn_commits_file.write('[')
        rn_commits_file.write('[')
        diff_rn_file.write('[')

        line_count = 0
        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))  # Calculate total lines for progress bar
        language_stats = defaultdict(lambda: {'release_note_count': 0, 'commit_count': 0, 'msg_count': 0, 'diff_count': 0, 'total_cleaned_release_note_tokens': 0, 'total_cleaned_msg_tokens': 0, 'total_cleaned_diff_tokens': 0, 'total_graph_commits_tokens': 0})

        with open(file_path, 'r', encoding='utf-8') as file:
            commit_urls = []
            for line in tqdm(file, total=total_lines, desc="Processing JSON lines"):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                line_count += 1
                logging.info(f"Processing line {line_count}")

                try:
                    entry = json.loads(line)  # Parse JSON line by line
                    if len(entry.get('commits', [])) > 250:
                        continue  # Skip entries with more than 250 commits

                    processed_commits = [process_commit(commit) for commit in entry['commits']]
                    entry['commits'] = processed_commits

                    # Save commit URL for splitting later
                    commit_urls.append(entry['commit_url'])

                    # Save to final_data.json
                    final_data = {
                        'commit_url': entry['commit_url'],
                        'release_note': entry['release_note'],
                        'cleaned_release_note': entry['cleaned_release_note'],
                        'commits': [{'msg': commit['msg'], 'cleaned_msg': commit['cleaned_msg'], 'diff': commit['diff']} for commit in processed_commits]
                    }
                    final_data_file.write(json.dumps(final_data, ensure_ascii=False) + ',\n')

                    # Save to graph_rn_commits.json
                    graph_rn_commits_data = {
                        'commit_url': entry['commit_url'],
                        'release_note': entry['release_note'],
                        'cleaned_release_note': entry['cleaned_release_note'],
                        'graph_commits': entry['graph_commits']
                    }
                    graph_rn_commits_file.write(json.dumps(graph_rn_commits_data, ensure_ascii=False) + ',\n')

                    # Save to rn_commits.json
                    rn_commits_data = {
                        'commit_url': entry['commit_url'],
                        'release_note': entry['release_note'],
                        'cleaned_release_note': entry['cleaned_release_note'],
                        'commits': [{'msg': commit['msg'], 'cleaned_msg': commit['cleaned_msg']} for commit in processed_commits]
                    }
                    rn_commits_file.write(json.dumps(rn_commits_data, ensure_ascii=False) + ',\n')

                    # Save to diff_rn.json
                    diff_rn_data = {
                        'commit_url': entry['commit_url'],
                        'release_note': entry['release_note'],
                        'cleaned_release_note': entry['cleaned_release_note'],
                        'diffs': [{'msg': commit['msg'], 'cleaned_msg': commit['cleaned_msg'], 'diff': commit['diff']} for commit in processed_commits]
                    }
                    diff_rn_file.write(json.dumps(diff_rn_data, ensure_ascii=False) + ',\n')

                    # Update statistics
                    #language = entry.get('language', 'Unknown')
                    #language_stats[language]['release_note_count'] += 1
                    #language_stats[language]['commit_count'] += len(processed_commits)
                    #language_stats[language]['msg_count'] += len([commit for commit in processed_commits if commit.get('msg', '')])
                    #language_stats[language]['diff_count'] += len([commit for commit in processed_commits if commit.get('diff', '')])
                    #language_stats[language]['total_cleaned_release_note_tokens'] += len(entry.get('cleaned_release_note', '').split())
                    #language_stats[language]['total_graph_commits_tokens'] += len(entry.get('graph_commits', '').split())
                    #for commit in processed_commits:
                    #    language_stats[language]['total_cleaned_msg_tokens'] += len(commit.get('cleaned_msg', '').split())
                    #    for diff in commit.get('diff', []):
                    #        language_stats[language]['total_cleaned_diff_tokens'] += len(diff.get('cleaned_diff', '').split())
                    # Update statistics
                    language = entry.get('language', 'Unknown')
                    language_stats[language]['release_note_count'] += 1
                    language_stats[language]['commit_count'] += len(processed_commits)
                    language_stats[language]['msg_count'] += len([commit for commit in processed_commits if commit.get('msg', '').strip()])
                    language_stats[language]['diff_count'] += len([diff for commit in processed_commits for diff in commit.get('diff', []) if diff.get('cleaned_diff', '').strip()])
                    language_stats[language]['total_cleaned_release_note_tokens'] += len(entry.get('cleaned_release_note', '').split())
                    language_stats[language]['total_graph_commits_tokens'] += len(entry.get('graph_commits', '').split())
                    for commit in processed_commits:
                        language_stats[language]['total_cleaned_msg_tokens'] += len(commit.get('cleaned_msg', '').split())
                        for diff in commit.get('diff', []):
                            language_stats[language]['total_cleaned_diff_tokens'] += len(diff.get('cleaned_diff', '').split())

                except json.JSONDecodeError as e:
                    logging.error(f"JSON Decode Error: {e}")

        # Remove trailing commas and close files
        final_data_file.seek(final_data_file.tell() - 2, 0)
        graph_rn_commits_file.seek(graph_rn_commits_file.tell() - 2, 0)
        rn_commits_file.seek(rn_commits_file.tell() - 2, 0)
        diff_rn_file.seek(diff_rn_file.tell() - 2, 0)

        final_data_file.write(']')
        graph_rn_commits_file.write(']')
        rn_commits_file.write(']')
        diff_rn_file.write(']')

        final_data_file.close()
        graph_rn_commits_file.close()
        rn_commits_file.close()
        diff_rn_file.close()

        # Calculate average token counts and save statistics
        stats = {}
        for language, data in language_stats.items():
            stats[language] = {
                'total_release_notes': data['release_note_count'],
                'total_commits': data['commit_count'],
                'total_msgs': data['msg_count'],
                'total_diffs': data['diff_count']
            }

        # Calculate overall average token counts
        average_stats = {
            'average_cleaned_release_note_tokens': sum(data['total_cleaned_release_note_tokens'] for data in language_stats.values()) / sum(data['release_note_count'] for data in language_stats.values()) if sum(data['release_note_count'] for data in language_stats.values()) > 0 else 0,
            'average_cleaned_msg_tokens': sum(data['total_cleaned_msg_tokens'] for data in language_stats.values()) / sum(data['msg_count'] for data in language_stats.values()) if sum(data['msg_count'] for data in language_stats.values()) > 0 else 0,
            'average_cleaned_diff_tokens': sum(data['total_cleaned_diff_tokens'] for data in language_stats.values()) / sum(data['diff_count'] for data in language_stats.values()) if sum(data['diff_count'] for data in language_stats.values()) > 0 else 0,
            'average_graph_commits_tokens': sum(data['total_graph_commits_tokens'] for data in language_stats.values()) / sum(data['release_note_count'] for data in language_stats.values()) if sum(data['release_note_count'] for data in language_stats.values()) > 0 else 0,
        }

        # Save language-wise statistics and overall statistics
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
        # Save the split commit URLs
        with open('split_commit_urls.json', 'w', encoding='utf-8') as split_file:
            json.dump(split_data, split_file, indent=4, ensure_ascii=False)

        logging.info("Data splitting completed successfully.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

# Example usage
file_path = 'merged_results.json'

process_and_save_data(file_path)

