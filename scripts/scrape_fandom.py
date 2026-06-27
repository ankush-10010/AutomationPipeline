"""
scrape_fandom.py - Scrapes Fandom Wiki using the official API to gather lore and theories.
It is respectful of server load (uses the API, adds delays, has a User-Agent).
"""
import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from config_loader import get_project_path, load_pipeline_config, setup_logging

log = setup_logging("scrape_fandom")

import argparse

API_URL = "https://rickandmorty.fandom.com/api.php"  # Default, overwritten by args
HEADERS = {
    "User-Agent": "AIExplainerBot/1.0 (Contact: local-dev) Python/3.x"
}


def search_fandom(query: str, limit: int = 20) -> list:
    """Searches the Fandom wiki using the MediaWiki API."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "utf8": "",
        "format": "json",
        "srlimit": limit
    }
    log.info(f"Searching API for: {query}")
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        return [item['title'] for item in data.get('query', {}).get('search', [])]
    except Exception as e:
        log.error(f"Search failed: {e}")
        return []

def get_all_pages(limit: str = "max") -> list:
    """Fetches every single page title on the entire wiki (Main Namespace)."""
    titles = []
    apcontinue = None
    log.info("Fetching ALL pages from the entire wiki... This will be massive!")
    
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": 0,  # Only main articles, ignore user/talk pages
            "aplimit": limit,
            "format": "json"
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
            
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()
            
            pages = data.get("query", {}).get("allpages", [])
            for p in pages:
                titles.append(p["title"])
                
            apcontinue = data.get("continue", {}).get("apcontinue")
            if not apcontinue:
                break
                
            time.sleep(0.5)
        except Exception as e:
            log.error(f"Allpages fetch failed: {e}")
            break
            
    log.info(f"Found a total of {len(titles)} pages on the entire wiki!")
    return titles

def get_category_members(category: str, limit: str = "max") -> list:
    """Fetches all page titles within a specific category."""
    titles = []
    cmcontinue = None
    log.info(f"Fetching members of {category}...")
    
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": limit,
            "format": "json"
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
            
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()
            
            members = data.get("query", {}).get("categorymembers", [])
            for member in members:
                if member["ns"] == 0:  # Only main namespace pages
                    titles.append(member["title"])
                    
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break
                
            time.sleep(0.5) # Slight delay between pagination requests
        except Exception as e:
            log.error(f"Category fetch failed: {e}")
            break
            
    log.info(f"Found {len(titles)} pages in {category}")
    return titles

def get_page_content(title: str) -> str:
    """Fetches the parsed HTML of a specific page via the API."""
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "disabletoc": 1
    }
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        
        if 'error' in data:
            log.warning(f"Error fetching {title}: {data['error'].get('info')}")
            return ""
            
        html = data.get('parse', {}).get('text', {}).get('*', '')
        return html
    except Exception as e:
        log.error(f"Failed to fetch {title}: {e}")
        return ""

def extract_info_from_html(title: str, html: str) -> tuple[str, dict]:
    """Parses HTML to extract the introductory paragraphs and any theories/trivia."""
    if not html:
        return "", {}
        
    soup = BeautifulSoup(html, "html.parser")
    theories = {}
    intro_paragraphs = []
    
    # Extract intro (all paragraphs before the first header)
    # The structure might be nested, so we find the first h2 and gather preceding p tags
    first_h2 = soup.find('h2')
    if first_h2:
        for sibling in first_h2.find_all_previous('p'):
            text = sibling.get_text(strip=True)
            if text:
                intro_paragraphs.insert(0, text) # prepend since find_all_previous goes backwards
    else:
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                intro_paragraphs.append(text)
                
    intro_text = " ".join(intro_paragraphs[:3]) # Limit to top 3 paragraphs
    
    # Check if the whole page is a theory page (e.g. Evil Morty/Theories)
    if "theory" in title.lower() or "theories" in title.lower():
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        if paragraphs:
            theories[title] = " ".join(paragraphs)
            return intro_text, theories

    # Otherwise, look for specific sections
    for header in soup.find_all(['h2', 'h3']):
        header_text = header.get_text(strip=True).replace("[edit]", "").lower()
        
        # We target Theories, Trivia, or Speculation
        if any(keyword in header_text for keyword in ['theor', 'speculat', 'trivia', 'lore']):
            content = []
            sibling = header.find_next_sibling()
            while sibling and sibling.name not in ['h2', 'h3']:
                if sibling.name == 'p':
                    text = sibling.get_text(strip=True)
                    if text:
                        content.append(text)
                elif sibling.name == 'ul':
                    for li in sibling.find_all('li'):
                        text = li.get_text(strip=True)
                        if text:
                            # Keep it clean
                            content.append(f"- {text}")
                sibling = sibling.find_next_sibling()
                
            if content:
                section_title = f"{title} - {header.get_text(strip=True).replace('[edit]', '')}"
                theories[section_title] = " ".join(content)
                
    return intro_text, theories

def scrape_all_fandom_data(config: dict, custom_url: str = None, core_pages: list = None, scrape_all: bool = False):
    global API_URL
    if custom_url:
        API_URL = custom_url
        
    theories_path = get_project_path("theories_db", config)
    wiki_path = get_project_path("wiki_db", config)
    
    titles = set()
    
    if scrape_all:
        log.info("SCRAPE ALL MODE ACTIVATED: Fetching the entire wiki...")
        titles.update(get_all_pages(limit="max"))
    else:
        # 1. Search for theory-related pages
        titles.update(search_fandom("Theories", limit=15))
        titles.update(search_fandom("Theory", limit=15))
        
        # 2. Add some known high-lore pages that might contain Trivia/Theories
        core_lore_pages = core_pages if core_pages else [
            "Rick Sanchez", "Morty Smith", "Evil Morty", "Rick Prime",
            "Central Finite Curve", "Citadel of Ricks", "Mr. Poopybutthole",
            "Space Cruiser", "Portal Gun"
        ]
        titles.update(core_lore_pages)
        
        # 3. Add ALL characters from the wiki Category
        # We fetch all characters to populate wiki.json
        character_pages = get_category_members("Category:Characters", limit="max")
        titles.update(character_pages)
    
    log.info(f"Found {len(titles)} unique pages to scan. This might take a few minutes!")
    
    # Load existing state to allow resuming!
    existing_theories = {}
    if theories_path.exists():
        with open(theories_path, 'r', encoding='utf-8') as f:
            try:
                existing_theories = json.load(f)
            except json.JSONDecodeError:
                pass
                
    existing_wiki = {}
    if wiki_path.exists():
        with open(wiki_path, 'r', encoding='utf-8') as f:
            try:
                existing_wiki = json.load(f)
            except json.JSONDecodeError:
                pass

    log.info(f"Loaded {len(existing_wiki)} existing wiki pages. These will be skipped to allow resuming.")
    
    # Create directory if it doesn't exist
    theories_path.parent.mkdir(parents=True, exist_ok=True)
    wiki_path.parent.mkdir(parents=True, exist_ok=True)
    
    for idx, title in enumerate(titles, 1):
        if title in existing_wiki:
            log.info(f"--- Skipping {idx}/{len(titles)}: {title} (Already scraped) ---")
            continue
            
        log.info(f"--- Processing {idx}/{len(titles)}: {title} ---")
        html = get_page_content(title)
        intro_text, extracted_theories = extract_info_from_html(title, html)
        
        if intro_text:
            existing_wiki[title] = intro_text
            
        if extracted_theories:
            existing_theories.update(extracted_theories)
            log.info(f" -> Found {len(extracted_theories)} relevant sections.")
        else:
            log.info(" -> No theory/trivia sections found.")
            
        # Auto-save every 50 pages to prevent Colab timeout data loss
        if idx % 50 == 0:
            log.info(f"💾 Auto-saving progress at page {idx}...")
            with open(theories_path, 'w', encoding='utf-8') as f:
                json.dump(existing_theories, f, indent=4)
            with open(wiki_path, 'w', encoding='utf-8') as f:
                json.dump(existing_wiki, f, indent=4)
                
        # Be a good citizen, sleep 1.5 seconds between API calls
        time.sleep(1.5)
        
    # Final Save
    with open(theories_path, 'w', encoding='utf-8') as f:
        json.dump(existing_theories, f, indent=4)
    with open(wiki_path, 'w', encoding='utf-8') as f:
        json.dump(existing_wiki, f, indent=4)
        
    log.info(f"Finished! Total theories in DB: {len(existing_theories)} | Total wiki entries: {len(existing_wiki)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Fandom Wiki using official API")
    parser.add_argument("--url", default=None, help="The base API url, e.g. https://ben10.fandom.com/api.php")
    parser.add_argument("--core-pages", nargs='+', default=None, help="Specific pages to forcefully scrape")
    parser.add_argument("--scrape-all", action="store_true", help="Scrape every single page on the wiki (Warning: can take hours for large wikis)")
    args = parser.parse_args()
    
    config = load_pipeline_config()
    scrape_all_fandom_data(config, custom_url=args.url, core_pages=args.core_pages, scrape_all=args.scrape_all)
