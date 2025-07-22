import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

# GitHub API 设置
TOKEN = "TOKEN"
HEADERS = {'Authorization': f'token {TOKEN}'}

def parse_url(commit_url):
    """
    从commit_url中解析出tag1和tag2
    """
    # 例如 https://github.com/xxx/yyy/compare/v1.1.1...v1.1.2
    m = re.search(r'/compare/(.+)\.\.\.(.+)$', commit_url)
    if m:
        return m.group(1), m.group(2)
    return None, None

def fetch_release_note(repo, tag):
    # repo: https://github.com/xxx/yyy
    # tag: v1.1.2
    api_url = repo.replace('https://github.com', 'https://api.github.com/repos') + f'/releases/tags/{tag}'
    try:
        response = requests.get(api_url, headers=HEADERS)
        response.raise_for_status()
        release_data = response.json()
        return tag, release_data.get('body', '').strip()
    except requests.exceptions.HTTPError as err:
        return tag, f"HTTP error occurred: {err} - {api_url}"
    except Exception as e:
        return tag, f"An error occurred: {e} - {api_url}"

def save_to_file(output_file, repo, tag, note):
    with open(output_file, 'a') as file:
        json.dump({repo: {tag: note}}, file)
        file.write('\n')

def main(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for entry in data:
            repo = entry['repo']
            commit_url = entry['commit_url']
            _, to_tag = parse_url(commit_url)
            if to_tag:
                futures[executor.submit(fetch_release_note, repo, to_tag)] = (repo, to_tag)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching release notes"):
            repo, tag = futures[future]
            tag, note = future.result()
            save_to_file(output_file, repo, tag, note)
            results.append({repo: {tag: note}})

if __name__ == "__main__":
    input_file = 'commit_urls.json'  # 只包含repo和commit_url
    output_file = 'release_notes.json'  
    main(input_file, output_file)

