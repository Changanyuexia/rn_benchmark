import json
import os
import shutil
import threading
import logging
import uuid
from git import Repo
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
repos_info = {}  # Stores the list of tags for each repository
progress_file = 'progress.json'
write_lock = threading.Lock()

def load_progress():
    """Load progress from the progress file."""
    if not os.path.exists(progress_file):
        return {}
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return {}

def save_progress(progress):
    """Save progress to the progress file."""
    with write_lock:
        with open(progress_file, 'w') as f:
            json.dump(progress, f)

def read_repos(file_path):
    """Read repositories from the given file and store in repos_info."""
    progress = load_progress()
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Reading repos from repo.txt"):
            repo = line.strip()
            if repo:
                normalized_repo = f"https://github.com/{repo.strip('/')}" if not repo.startswith("https://github.com/") else repo.strip('/')
                if normalized_repo not in repos_info:
                    repos_info[normalized_repo] = []

def get_tags(repo_url):
    """Clone the repository and retrieve its tags."""
    local_path = f"./temp_repo_{uuid.uuid4()}"
    try:
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        repo = Repo.clone_from(repo_url, local_path)
        tags = [tag.name for tag in repo.tags]
        shutil.rmtree(local_path)
        return tags
    except Exception as e:
        logging.error(f"Error fetching tags for {repo_url}: {e}")
        return []

def process_repo(repo_url):
    """Process each repository by fetching its tags and saving progress."""
    tags = get_tags(repo_url)
    if tags:
        repos_info[repo_url] = tags
        logging.info(f"Tags successfully fetched and stored for {repo_url}: {tags}") 
    else:
        logging.warning(f"No tags found for {repo_url}")
    save_progress({repo_url: 'processed'})  # Mark the repository as processed

def save_results():
    """Save the tags of all processed repositories to a JSON file."""
    with open('all_repo_tags.json', 'w') as f:
        json.dump(repos_info, f, indent=4)

if __name__ == "__main__":
    file_path = 'repo.txt'  # Ensure this path correctly points to your repo.txt file
    read_repos(file_path)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_repo, url) for url in repos_info.keys()]
        # Add tqdm progress bar to monitor the processing
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing repos"):
            pass
    save_results()
    print("All data processed and saved.")

