import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# GitHub API 设置
TOKEN = "TOKEN"
# GitHub API 设置
HEADERS = {'Authorization': f'token {TOKEN}'}

def fetch_release_notes(repo, tag):
    url = f"{repo}/releases/tags/{tag}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        release_data = response.json()
        return tag, release_data.get('body', '').strip()
    except requests.exceptions.HTTPError as err:
        return tag, f"HTTP error occurred: {err} - {url}"
    except Exception as e:
        return tag, f"An error occurred: {e} - {url}"

def save_to_file(output_file, repo, tag, note):
    with open(output_file, 'a') as file:
        json.dump({repo: {tag: note}}, file)
        file.write('\n')

def main(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_release_notes, repo, tag): (repo, tag) for repo, tags in data.items() for tag in tags}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching release notes"):
            repo, tag = futures[future]
            tag, note = future.result()
            save_to_file(output_file, repo, tag, note)
            results.append({repo: {tag: note}})

if __name__ == "__main__":
    input_file = 'processed_repo_tags.json'  
    output_file = 'release_notes.json'  
    main(input_file, output_file)

