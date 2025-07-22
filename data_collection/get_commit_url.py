import json
import re

def extract_version(tag):
    """
    Extracts a version from a tag string, e.g., 'v10.1.1' -> (10, 1, 1).
    :param tag: Tag string
    :return: Extracted version tuple (major, minor, micro) or None if invalid
    """
    match = re.search(r'v?(\d+)\.(\d+)\.(\d+)', tag)
    if match:
        return tuple(map(int, match.groups()))
    return None

def generate_all_commit_urls(processed_file_path, output_file_path):
    with open(processed_file_path, 'r') as f:
        processed_data = json.load(f)
    commit_urls = []
    for repo_url, tags in processed_data.items():
        for i in range(1, len(tags)):
            prev_tag = tags[i-1]
            curr_tag = tags[i]
            prev_version = extract_version(prev_tag)
            curr_version = extract_version(curr_tag)
            # 只保留严格连续的tag对
            if (curr_version and prev_version and
                curr_version[0] == prev_version[0] and
                curr_version[1] == prev_version[1] and
                curr_version[2] == prev_version[2] + 1):
                commit_url = f"{repo_url}/compare/{prev_tag}...{curr_tag}"
                commit_urls.append({
                    "repo": repo_url,
                    "from_tag": prev_tag,
                    "to_tag": curr_tag,
                    "commit_url": commit_url
                })
    with open(output_file_path, 'w') as f:
        json.dump(commit_urls, f, indent=4, ensure_ascii=False)
    print(f"已输出所有连续tag对的commit_url到 {output_file_path}")

if __name__ == '__main__':
    processed_file_path = 'processed_repo_tags.json'  # 只用get_tags.py输出
    output_file_path = 'commit_urls.json'
    generate_all_commit_urls(processed_file_path, output_file_path)

