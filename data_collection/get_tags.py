import json
import os
import shutil
import threading
import logging
import uuid
import re
from git import Repo
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from packaging.version import Version, InvalidVersion
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

repos_info = {}  # Stores the list of cleaned, time-sorted tags for each repository
progress_file = 'progress.json'
write_lock = threading.Lock()

def is_valid_release_tag(tag):
    """
    Check if a tag is a valid release version.
    1. Exclude tags containing 'alpha', 'beta', 'rc' (case-insensitive).
    2. Only keep tags like v1.2.3 or 1.2.3 (semantic versioning).
    3. Validate with packaging.version.Version.
    """
    if re.search(r'(alpha|beta|rc)', tag, re.IGNORECASE):
        return False
    if not re.match(r'^v?\\d+\\.\\d+\\.\\d+$', tag):
        return False
    try:
        Version(tag.lstrip('v'))
        return True
    except InvalidVersion:
        return False

def get_tag_time(repo, tag_name):
    """
    Get the commit time of a tag.
    Return as datetime object.
    """
    tag_ref = next((t for t in repo.tags if t.name == tag_name), None)
    if tag_ref is None:
        return None
    if tag_ref.tag:
        # Annotated tag
        return datetime.fromtimestamp(tag_ref.tag.tagged_date)
    else:
        # Lightweight tag
        return tag_ref.commit.committed_datetime

def load_progress():
    if not os.path.exists(progress_file):
        return {}
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return {}

def save_progress(progress):
    with write_lock:
        with open(progress_file, 'w') as f:
            json.dump(progress, f)

def read_repos(file_path):
    progress = load_progress()
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Reading repos from repo.txt"):
            repo = line.strip()
            if repo:
                normalized_repo = f"https://github.com/{repo.strip('/')}" if not repo.startswith("https://github.com/") else repo.strip('/')
                if normalized_repo not in repos_info:
                    repos_info[normalized_repo] = []

def get_tags_with_time(repo_url):
    """
    Clone the repository and retrieve all tags with their creation time.
    Returns a list of (tag_name, tag_time) tuples.
    """
    local_path = f"./temp_repo_{uuid.uuid4()}"
    try:
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        repo = Repo.clone_from(repo_url, local_path)
        tags = []
        for tag in repo.tags:
            tag_name = tag.name
            tag_time = get_tag_time(repo, tag_name)
            if tag_time:
                tags.append((tag_name, tag_time))
        shutil.rmtree(local_path)
        return tags
    except Exception as e:
        logging.error(f"Error fetching tags for {repo_url}: {e}")
        return []

def process_repo(repo_url):
    """
    For each repository:
    1. Fetch all tags and their creation times.
    2. Filter out invalid tags (pre-release, test, etc.).
    3. Validate tags with packaging.version.
    4. Sort valid tags chronologically.
    5. Save the sorted tag list.
    """
    tags_with_time = get_tags_with_time(repo_url)
    # Step 1: Filter and validate tags
    valid_tags = [(t, t_time) for t, t_time in tags_with_time if is_valid_release_tag(t)]
    # Step 2: Remove duplicates (by tag name)
    seen = set()
    unique_tags = []
    for t, t_time in valid_tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append((t, t_time))
    # Step 3: Sort by tag creation time (chronologically)
    unique_tags.sort(key=lambda x: x[1])
    # Step 4: Save sorted tag list
    sorted_tag_names = [t for t, _ in unique_tags]
    repos_info[repo_url] = sorted_tag_names
    if sorted_tag_names:
        logging.info(f"Tags for {repo_url}: {sorted_tag_names}")
    else:
        logging.warning(f"No valid tags for {repo_url}")
    save_progress({repo_url: 'processed'})

def save_results():
    """Save the tags of all processed repositories to a JSON file."""
    with open('processed_repo_tags.json', 'w') as f:
        json.dump(repos_info, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    file_path = 'repo.txt'  # Ensure this path correctly points to your repo.txt file
    read_repos(file_path)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_repo, url) for url in repos_info.keys()]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing repos"):
            pass
    save_results()
    print("All data processed and saved to processed_repo_tags.json.")
