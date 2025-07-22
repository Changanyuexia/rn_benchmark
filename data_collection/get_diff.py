# ... previous code ...
# ====== Configuration Parameters ======
REPO_ROOT = "./repos"
LOCK_DIR = "./locks"
TEMP_DIR = "./temp_repo"
INPUT_JSON = "commit_urls.json"  # New input file name
OUTPUT_JSONL = "diffs.jsonl"
SEEN_LOG = "seen_urls.txt"
ERROR_LOG = "error_detailed.log"  # Detailed error log file
MAX_WORKERS = 2
BATCH_SIZE = 10
SAVE_INTERVAL = 5  # Save seen log and output every 5 entries
GITHUB_TOKEN = "xxxxx"

os.makedirs(REPO_ROOT, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def log_error(url, error_msg, error_type=None, stack_trace=None):
    """Log detailed error information to the error log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(ERROR_LOG, "a", encoding="utf-8") as ferr:
        ferr.write(f"\n{'='*50}\n")
        ferr.write(f"Time: {timestamp}\n")
        ferr.write(f"URL: {url}\n")
        ferr.write(f"Error Type: {error_type if error_type else 'Unknown'}\n")
        ferr.write(f"Error Message: {error_msg}\n")
        if stack_trace:
            ferr.write(f"Stack Trace:\n{stack_trace}\n")
        ferr.write(f"{'='*50}\n")

def parse_url(commit_url):
    parts = commit_url.split('/')
    try:
        owner, repo_name = parts[3], parts[4]
        from_tag, to_tag = parts[-1].split('...')
        clone_url = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{owner}/{repo_name}.git"
        return clone_url, repo_name, from_tag, to_tag
    except ValueError:
        raise ValueError("Commit URL format is incorrect, cannot extract tags")

def tag_exists(repo_path, tag):
    result = subprocess.run(["git", "-C", repo_path, "rev-parse", "--verify", tag], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0

def clone_or_get_repo(clone_url, repo_cache):
    name = clone_url.rstrip(".git").split("/")[-1]
    local = os.path.join(REPO_ROOT, name)
    if clone_url in repo_cache:
        return repo_cache[clone_url]
    if os.path.exists(local):
        try:
            repo_cache[clone_url] = Repo(local)
            return repo_cache[clone_url]
        except Exception as e:
            log_error(clone_url, str(e), "Repo load error", traceback.format_exc())
            shutil.rmtree(local, ignore_errors=True)
    lock_file = os.path.join(LOCK_DIR, f"{uuid.uuid5(uuid.NAMESPACE_URL, clone_url)}.lock")
    with FileLock(lock_file):
        if clone_url in repo_cache:
            return repo_cache[clone_url]
        log(f"🔽 Cloning {clone_url}")
        try:
            repo = Repo.clone_from(clone_url, local)
            repo.git.fetch('--all', '--tags', '--prune')
            repo_cache[clone_url] = repo
        except Exception as e:
            log_error(clone_url, str(e), "Git clone error", traceback.format_exc())
            raise
    return repo_cache[clone_url]

def remove_repo(clone_url):
    name = clone_url.rstrip(".git").split("/")[-1]
    path = os.path.join(REPO_ROOT, name)
    shutil.rmtree(path, ignore_errors=True)
    log(f"🧹 Removed repo: {path}")
    lock_file = os.path.join(LOCK_DIR, f"{uuid.uuid5(uuid.NAMESPACE_URL, clone_url)}.lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)
        log(f"🧹 Removed lock file: {lock_file}")

def get_version_diff_and_tokens(repo_path, tag1, tag2):
    try:
        # Get changed file list
        cmd_files = ["git", "-C", repo_path, "diff", "--name-status", tag1, tag2]
        files_result = subprocess.run(cmd_files, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        if files_result.returncode != 0:
            error_msg = f"Git diff --name-status failed: {files_result.stderr}"
            log_error(f"{tag1}...{tag2}", error_msg, "Git diff error")
            raise RuntimeError(error_msg)
        # Parse file status info
        changed_files = []
        for line in files_result.stdout.strip().split('\n'):
            if line:
                status, filename = line.split('\t', 1)
                changed_files.append({'status': status, 'filename': filename})
        # Get full diff content
        cmd_diff = ["git", "-C", repo_path, "diff", tag1, tag2]
        diff_result = subprocess.run(cmd_diff, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        if diff_result.returncode != 0:
            error_msg = f"Git diff failed: {diff_result.stderr}"
            log_error(f"{tag1}...{tag2}", error_msg, "Git diff error")
            raise RuntimeError(error_msg)
        diff_text = diff_result.stdout
        tokens = nltk.word_tokenize(diff_text)
        diff_tokens = len(tokens)
        return changed_files, diff_text, diff_tokens
    except Exception as e:
        log_error(f"{tag1}...{tag2}", str(e), "Diff processing error", traceback.format_exc())
        return None, None, str(e)

def load_seen_log():
    if not os.path.exists(SEEN_LOG):
        return set()
    with open(SEEN_LOG, encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def update_seen_log_batch(urls):
    with open(SEEN_LOG, 'a', encoding='utf-8') as f:
        for url in urls:
            f.write(url + '\n')

def process_repo_batch(repo_urls, repo_cache):
    """Process a batch of URLs for the same repo."""
    results = []
    if not repo_urls:
        return results
    # Use the first URL to clone repo
    first_url = repo_urls[0]
    clone_url, repo_name, _, _ = parse_url(first_url)
    repo = None
    repo_path = None
    try:
        repo = clone_or_get_repo(clone_url, repo_cache)
        repo_path = repo.working_tree_dir
        # Process all URLs for this repo
        for url in repo_urls:
            try:
                _, _, tag1, tag2 = parse_url(url)
                if not tag_exists(repo_path, tag1) or not tag_exists(repo_path, tag2):
                    error_msg = f"Tag {tag1} or {tag2} does not exist in repo (after fetch)"
                    log_error(url, error_msg, "Tag not found error")
                    results.append({
                        "commit_url": url,
                        "diff": "",
                        "mod": [],
                        "release_note": "",
                        "error": error_msg,
                        "error_type": "Tag not found error"
                    })
                    continue
                mod, diff, diff_tokens = get_version_diff_and_tokens(repo_path, tag1, tag2)
                if diff is None:
                    error_msg = f"Failed to get diff: {diff_tokens}"
                    log_error(url, error_msg, "Diff fetch error")
                    results.append({
                        "commit_url": url,
                        "diff": "",
                        "mod": [],
                        "release_note": "",
                        "error": error_msg,
                        "error_type": "Diff fetch error"
                    })
                    continue
                results.append({
                    "commit_url": url,
                    "diff": diff,
                    "mod": mod,
                    "release_note": "",
                    "diff_tokens": diff_tokens
                })
            except Exception as e:
                log_error(url, str(e), "Processing error", traceback.format_exc())
                results.append({
                    "commit_url": url,
                    "diff": "",
                    "mod": [],
                    "release_note": "",
                    "error": str(e),
                    "error_type": "Processing error"
                })
    except Exception as e:
        log_error(clone_url, str(e), "Repo processing error", traceback.format_exc())
        # If repo processing fails, mark all URLs as error
        for url in repo_urls:
            results.append({
                "commit_url": url,
                "diff": "",
                "mod": [],
                "release_note": "",
                "error": f"Repo processing failed: {str(e)}",
                "error_type": "Repo processing error"
            })
    finally:
        # Clean up repo after all URLs are processed
        if repo and repo_path:
            try:
                if clone_url in repo_cache:
                    del repo_cache[clone_url]
                subprocess.run(["rm", "-rf", repo_path], check=True)
                log(f"🧹 Cleaned repo: {repo_name}")
            except Exception as e:
                log(f"⚠️ Failed to clean repo {repo_name}: {e}")
    return results

def main():
    log("🚀 Starting version diff extraction (two-stage: clone then diff)...")
    with open(INPUT_JSON, encoding='utf-8') as f:
        rows = json.load(f)
    log(f"📊 Total entries to process: {len(rows)}")
    seen_urls = load_seen_log()
    log(f"🔄 Loaded {len(seen_urls)} seen URLs.")
    log(f"📝 New entries to process: {len(rows) - len(seen_urls)}")
    # Group URLs by repo
    repo_urls_map = defaultdict(list)
    for row in rows:
        url = row.get("commit_url", "")
        if url in seen_urls:
            continue
        repo = row.get("repo", None)
        if repo:
            repo_urls_map[repo].append(url)
        else:
            # For old format, try to parse repo from url
            clone_url, _, _, _ = parse_url(url)
            repo_urls_map[clone_url].append(url)
    log(f"📦 Number of repos to process: {len(repo_urls_map)}")
    repo_cache = {}
    results = []
    success_count = 0
    error_count = 0
    batch_error_count = 0
    error_types = defaultdict(int)
    seen_update_batch = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(OUTPUT_JSONL, 'a', encoding='utf-8') as fout:
        # Submit repo batch tasks
        futures = {
            executor.submit(process_repo_batch, urls, repo_cache): urls 
            for urls in repo_urls_map.values()
        }
        pbar = tqdm(total=len(rows) - len(seen_urls), desc="Processing commit URLs")
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                for result in batch_results:
                    results.append(result)
                    seen_update_batch.append(result["commit_url"])
                    if "error" in result:
                        error_count += 1
                        batch_error_count += 1
                        error_type = result.get("error_type", "Unknown")
                        error_types[error_type] += 1
                    else:
                        success_count += 1
                    # Update progress bar
                    pbar.set_description(f"Processing [Success:{success_count} Fail:{error_count}]")
                    pbar.update(1)
                # Report batch statistics after each batch
                if len(results) >= BATCH_SIZE:
                    for item in results:
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()
                    log(f"📊 Batch statistics:")
                    log(f"   ✅ Success: {BATCH_SIZE - batch_error_count}")
                    log(f"   ❌ Fail: {batch_error_count}")
                    if batch_error_count > 0:
                        log(f"   🔍 Error type statistics:")
                        for err_type, count in error_types.items():
                            log(f"      - {err_type}: {count}")
                    results = []
                    batch_error_count = 0
                    error_types.clear()
                if len(seen_update_batch) >= SAVE_INTERVAL:
                    update_seen_log_batch(seen_update_batch)
                    seen_update_batch = []
            except Exception as e:
                log(f"⚠️ Future error: {e}")
                error_count += 1
                batch_error_count += 1
                error_types["Future processing error"] += 1
                pbar.update(1)
        pbar.close()
        # Process remaining results
        if results:
            for item in results:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()
            log(f"📊 Final batch statistics:")
            log(f"   ✅ Success: {len(results) - batch_error_count}")
            log(f"   ❌ Fail: {batch_error_count}")
            if batch_error_count > 0:
                log(f"   🔍 Error type statistics:")
                for err_type, count in error_types.items():
                    log(f"      - {err_type}: {count}")
        if seen_update_batch:
            update_seen_log_batch(seen_update_batch)
    # Final statistics report
    log(f"✅ Finished writing output to {OUTPUT_JSONL}")
    log(f"📊 Final statistics report:")
    log(f"   ✅ Total success: {success_count}")
    log(f"   ❌ Total fail: {error_count}")
    log(f"   📦 Number of repos processed: {len(repo_urls_map)}")
    log(f"   📝 Total URLs processed: {success_count + error_count}")
    log(f"   💾 Results saved in: {OUTPUT_JSONL}")
    if error_count > 0:
        log(f"   🔍 Error type statistics:")
        for err_type, count in error_types.items():
            log(f"      - {err_type}: {count}")

if __name__ == "__main__":
    def safe_exit(sig, frame):
        log("⏹️ Caught interrupt signal, exiting safely...")
        exit(0)
    signal.signal(signal.SIGINT, safe_exit)
    signal.signal(signal.SIGTERM, safe_exit)
    main() 

