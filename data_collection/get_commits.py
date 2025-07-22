import os
import shutil
import json
import uuid
from git import Repo
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from pydriller import Repository
from filelock import FileLock


def clone_repo(repo_url, repo_cache, lock_dir="./locks"):
    """Clone the repository or reuse the cloned version from the cache with file lock."""
    if repo_url in repo_cache:
        return repo_cache[repo_url]
    else:
        # Use file locking to avoid concurrent access issues
        repo_path = f"./temp_repo_{uuid.uuid4()}"
        os.makedirs(repo_path, exist_ok=True)

        # Create a lock file for each repo to ensure it's cloned only once
        lock_file = os.path.join(lock_dir, f"{uuid.uuid5(uuid.NAMESPACE_URL, repo_url)}.lock")
        os.makedirs(lock_dir, exist_ok=True)

        with FileLock(lock_file):
            if repo_url in repo_cache:
                return repo_cache[repo_url]
            Repo.clone_from(repo_url, repo_path)
            repo_cache[repo_url] = repo_path
            remove_lock_files(repo_path)

        return repo_path

def remove_lock_files(repo_path):
    """Remove any Git lock files to avoid conflicts."""
    lock_files = ['config.lock', 'index.lock']
    for lock_file in lock_files:
        lock_path = os.path.join(repo_path, '.git', lock_file)
        if os.path.exists(lock_path):
            print(f"Removing lock file: {lock_path}")
            os.remove(lock_path)

def parse_url(commit_url):
    """Extracts tag1 and tag2 from the commit URL."""
    parts = commit_url.split('/')
    try:
        from_tag, to_tag = parts[-1].split('...')
        return from_tag, to_tag
    except ValueError:
        raise ValueError("Commit URL format is incorrect, unable to extract tags.")

def get_commits(repo_path, tag1, tag2):
    """Get the list of commits between two tags using PyDriller, including diff information."""
    commits_list = []
    
    try:
        for commit in Repository(path_to_repo=repo_path, from_tag=tag1, to_tag=tag2).traverse_commits():
            commit_time = commit.committer_date.strftime('%Y-%m-%d %H:%M:%S')

            if not commit.modified_files:
                print(f"Commit {commit.hash} has no modified files, skipping.")
                continue
            
            diff_info = []
            for mod in commit.modified_files:
                diff_data = {
                    'old_path': mod.old_path,
                    'new_path': mod.new_path,
                    'change_type': mod.change_type.name,
                    'diff': mod.diff
                }
                diff_info.append(diff_data)

            commit_info = {
                'hash': commit.hash,
                'msg': commit.msg,
                'author': commit.author.name,
                'date': commit_time,
                'diff': diff_info
            }
            commits_list.append(commit_info)

            print(f"Commit: {commit.hash} by {commit.author.name} on {commit_time}")
            print(f"Message: {commit.msg}")

    except Exception as e:
        print(f"Error getting commits between {tag1} and {tag2}: {e}")
    
    return commits_list

def get_graph(repo_path, tag1, tag2):
    """Get the graphical commit log between two tags."""
    repo = Repo(repo_path)
    try:
        graph_log = repo.git.log('--graph', '--oneline', f'{tag1}..{tag2}')
        print(f"Graphical commit log for {tag1}..{tag2}:\n{graph_log}")
        return graph_log
    except Exception as e:
        print(f"Error getting graph commits between {tag1} and {tag2}: {e}")
        return ""

def get_commit(commit_url, release_note, local_repo_path):
    """Process a single commit URL for a given repo."""
    tag1, tag2 = parse_url(commit_url)
    
    commits = get_commits(local_repo_path, tag1, tag2)
    graph_commits = get_graph(local_repo_path, tag1, tag2)

    output_data = {
        'commit_url': commit_url,
        'commits': commits,
        'graph_commits': graph_commits,
        'release_note': release_note
    }
    return output_data

def get_repo(repo_url, entries, repo_cache, max_workers=5):
    """Process all entries related to a single repository."""
    current_repo_path = clone_repo(repo_url, repo_cache)

    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_commit, entry['commit_url'], entry.get('release_note', ''), current_repo_path): entry['commit_url'] 
            for entry in entries
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing commits for {repo_url}"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing commit URL: {e}")

    return results

def group_by_repo(data):
    """Group entries by repository URL."""
    repo_groups = defaultdict(list)
    for entry in data:
        repo_url = entry['repo']
        repo_groups[repo_url].append(entry)
    return repo_groups

def save_progress(output_file, results, batch_size=100):  # Increased batch size to reduce I/O operations
    """Batch save results to reduce I/O operations."""
    if len(results) >= batch_size:
        with open(output_file, 'a') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        results.clear()

def update_log(progress_file, repo_url):
    """Update the progress log with the latest processed repository URL."""
    with open(progress_file, 'a') as pf:
        pf.write(f"{repo_url}\n")

def load_log(progress_file):
    """Loads the progress log and returns a set of processed repo URLs."""
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, 'r') as pf:
        processed_repos = set(line.strip() for line in pf)
    return processed_repos

def load_commits_from_json(json_file):
    """Load commit URLs and related info from updated_rn_commits.json."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Group by repo URL
    repo_commit_map = defaultdict(list)
    for entry in data:
        repo_url = entry['repo']
        commit_url = entry['commit_url']
        release_note = entry.get('release_note', '')
        repo_commit_map[repo_url].append({'commit_url': commit_url, 'release_note': release_note})
    
    return repo_commit_map

def get_notes_from_commit_urls(json_file, output_file, progress_file, max_workers=10, batch_size=100):  # Increased batch size
    """Process all entries from the repo.txt and updated_rn_commits.json using multi-threading."""
    # Load repos from repo.txt
    repo_commit_map = load_commits_from_json(json_file)

    processed_repos = load_log(progress_file)
    repo_cache = {}
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for repo_url, entries in repo_commit_map.items():
            if repo_url in processed_repos:
                continue

            futures[executor.submit(get_repo, repo_url, entries, repo_cache)] = repo_url

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing repositories"):
            try:
                result = future.result()
                results.extend(result)

                if len(results) >= batch_size:
                    save_progress(output_file, results)
                    results = []

                repo_url = futures[future]
                update_log(progress_file, repo_url)

            except Exception as e:
                print(f"Error processing repository: {e}")

    if results:
        save_progress(output_file, results)

# Example call to process commit_urls.json and save progress in real-time
json_file = 'commit_urls.json'  # Only input file
output_file = 'commits.json'  # Path to save the output
progress_file = 'progress_log.txt'  # Progress log

get_notes_from_commit_urls(json_file, output_file, progress_file, max_workers=7, batch_size=10)  # Increased max_workers and batch size
