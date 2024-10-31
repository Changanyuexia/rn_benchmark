import json
import re

def extract_version(tag):
    """
    Extracts a version from a tag string, e.g., 'v10.1.1' -> (10, 1, 1).

    :param tag: Tag string
    :return: Extracted version tuple (major, minor, micro) or None if invalid
    """
    # Regex to extract version numbers from strings with possible prefixes
    match = re.search(r'v?(\d+)\.(\d+)\.(\d+)', tag)
    if match:
        return tuple(map(int, match.groups()))
    return None

def add_commit_urls(data, processed_data):
    """
    Add commit_url for each entry in the data.

    :param data: JSON data list
    :param processed_data: Data from processed_repo_tags.json
    :return: Updated data list
    """
    for entry in data:
        repo_url = entry['repo']
        tag = entry['tag']
        
        # Retrieve tags for this repository from processed_data
        all_tags = processed_data.get(repo_url, [])  # Ensure we use repo_url to access tags

        # Check if the current tag is in all_tags
        if tag in all_tags:
            current_index = all_tags.index(tag)
            if current_index > 0:
                # Previous tag in the list
                previous_tag = all_tags[current_index - 1]
                # Extract version numbers
                current_version = extract_version(tag)
                previous_version = extract_version(previous_tag)

                # Check adjacency
                if (current_version and previous_version and 
                    current_version[0] == previous_version[0] and 
                    current_version[1] == previous_version[1] and 
                    current_version[2] == previous_version[2] + 1):
                    commit_url = f"{repo_url}/compare/{previous_tag}...{tag}"
                else:
                    commit_url = None
            else:
                commit_url = None
        else:
            commit_url = None

        # Update the entry with commit_url
        entry['commit_url'] = commit_url
    
    # Filter out entries with commit_url == None
    filtered_data = [entry for entry in data if entry['commit_url'] is not None]

    return filtered_data

def write_json(data, file_path):
    """
    Save data to a JSON file.

    :param data: Data to be saved
    :param file_path: File path to save the data
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    sorted_file_path = 'non_empty_release_notes.json'
    processed_file_path = 'processed_repo_tags.json'
    output_file_path = 'updated_sorted_merged_release_notes.json'

    # Load data
    with open(sorted_file_path, 'r') as f:
        sorted_data = json.load(f)
    
    with open(processed_file_path, 'r') as f:
        processed_data = json.load(f)

    # Add commit_url for each entry
    updated_data = add_commit_urls(sorted_data, processed_data)

    # Save updated data
    write_json(updated_data, output_file_path)

    # Print the total number of entries in the updated data
    print(f"Data has been updated and saved to {output_file_path}.")
    print(f"Total entries with valid commit URLs: {len(updated_data)}")

