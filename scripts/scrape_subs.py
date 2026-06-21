import requests
from bs4 import BeautifulSoup
import os
import time

# Configuration
BASE_URL = "https://my-subs.co"
SHOW_URL = f"{BASE_URL}/showlistsubtitles-2075-rick-and-morty"
SAVE_DIR = "rick_and_morty_subtitles"
LANGUAGE = "English"

# Using a standard user agent to prevent basic blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

def setup_directory():
    """Creates the save directory if it doesn't exist."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

def get_episode_links():
    """Fetches the main page and extracts links to individual episodes."""
    print("Fetching main show page...")
    response = requests.get(SHOW_URL, headers=HEADERS)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    episode_links = set()
    
    # UPDATED: Using the exact class and href pattern you found
    for a_tag in soup.find_all('a', class_='list-group-item', href=True):
        href = a_tag['href']
        if '/versions-2075-' in href: 
            full_url = BASE_URL + href if href.startswith('/') else href
            episode_links.add(full_url)
            
    print(f"Found {len(episode_links)} episode links.")
    return list(episode_links)

def download_subtitles_for_episode(episode_url):
    """Scrapes the episode page for the download link and downloads the file."""
    response = requests.get(episode_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # We don't know the exact row HTML, so we will look at all list items/divs/rows
    # that might contain both the word "English" and the download link.
    # A standard approach is to find all elements that have text, filter for English, 
    # then look for the nearest download link.
    
    # Find all download links first based on your snippet
    download_tags = soup.find_all('a', href=lambda href: href and '/downloads/' in href)
    
    downloaded = False
    for a_tag in download_tags:
        # Navigate up the HTML tree to see if this specific download button is for 'English'
        # We check the parent elements (like the row or list item containing this button)
        parent_container = a_tag.find_parent(['li', 'tr', 'div', 'ul']) 
        
        if parent_container and LANGUAGE.lower() in parent_container.text.lower():
            dl_href = a_tag['href']
            dl_url = BASE_URL + dl_href if dl_href.startswith('/') else dl_href
            
            # Fetch the actual ZIP/SRT file
            file_response = requests.get(dl_url, headers=HEADERS)
            
            # Create a clean filename based on the episode URL (e.g., versions-2075-4-5-rick-and-morty-subtitles.zip)
            raw_name = episode_url.split('/')[-1]
            filename = f"{raw_name}.zip" 
            
            filepath = os.path.join(SAVE_DIR, filename)
            
            # Save the file
            with open(filepath, 'wb') as f:
                f.write(file_response.content)
            print(f"  -> Successfully Downloaded: {filename}")
            
            downloaded = True
            break # Stop looking after finding the first English subtitle for this episode
            
    if not downloaded:
        print(f"  -> No {LANGUAGE} subtitles found for this episode.")

def main():
    setup_directory()
    episode_links = get_episode_links()
    
    if not episode_links:
        print("No episode links found. The website structure may have changed.")
        return

    # Process each episode
    for i, link in enumerate(episode_links):
        print(f"\nProcessing ({i+1}/{len(episode_links)}): {link}")
        try:
            download_subtitles_for_episode(link)
        except Exception as e:
            print(f"  -> Failed to process {link}: {e}")
            
        # Polite delay to prevent getting IP banned by my-subs.co
        time.sleep(2) 

if __name__ == "__main__":
    main()