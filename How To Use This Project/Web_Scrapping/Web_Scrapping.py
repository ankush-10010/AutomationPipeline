import json
from scrapling.spiders import Spider, Response

class Ben10Spider(Spider):
    name = "ben10_vector_db_crawler"
    # Fandom's 'Special:AllPages' is a great trick to see literally every page on the wiki.
    start_urls = ["https://ben10.fandom.com/wiki/Special:AllPages"]

    async def parse(self, response: Response):
        """ This function handles finding links to pages to scrape. """
        
        # 1. Grab all the page links from the Special:AllPages list
        page_links = response.css('.mw-allpages-chunk a').xpath('@href').getall()
        for link in page_links:
            # We don't want to scrape user profiles or talk pages for a vector DB.
            if "/wiki/User:" not in link and "/wiki/Message_Wall:" not in link:
                # Tell the spider to visit this link and run 'parse_page' on it
                yield response.follow(link, callback=self.parse_page)

        # 2. Find the "Next" button to go to the next page of the Special:AllPages list
        next_page = response.css('.mw-allpages-nav a:contains("Next")').xpath('@href').get()
        if next_page:
             yield response.follow(next_page, callback=self.parse)

    async def parse_page(self, response: Response):
        """ This function extracts the actual text for your Vector DB. """
        
        url = response.url
        title = response.css('.mw-page-title-main::text').get(default="Unknown Title")
        
        # Fandom's main text content is inside the .mw-parser-output class.
        # We grab all paragraph text to feed the vector DB.
        paragraphs = response.css('.mw-parser-output p::text').getall()
        # Clean it up by joining into one big string and stripping weird whitespace
        content = " ".join([p.strip() for p in paragraphs if p.strip()])

        # If the page is empty, don't save it
        if not content:
            return

        # Figure out if this is a theory or general wiki knowledge.
        # Often Fandom wikis put theories in a "Theory:" namespace or categorize them.
        is_theory = False
        categories = response.css('.page-header__categories a::text').getall()
        
        if "Theories" in categories or "/wiki/Theory:" in url:
            is_theory = True

        # Yield the data back to Scrapling
        yield {
            "type": "theory" if is_theory else "wiki",
            "title": title,
            "url": url,
            "content": content,
            "categories": categories
        }

# --- This code runs when you execute the script ---
if __name__ == "__main__":
    print("Starting the Ben 10 crawler... this may take a while!")
    
    # 1. Start the spider and wait for it to finish crawling
    spider = Ben10Spider()
    result = spider.start() 
    
    # 'result.items' contains everything the spider yielded
    all_data = list(result.items)
    
    # 2. Filter the data into two separate lists based on the "type" tag
    wiki_data = [item for item in all_data if item["type"] == "wiki"]
    theory_data = [item for item in all_data if item["type"] == "theory"]
    
    # We don't need the 'type' key in the final JSON, so let's remove it for cleanliness
    for item in wiki_data + theory_data:
        del item["type"]
        
    # 3. Save them out to your two JSON files
    with open("wiki.json", "w", encoding="utf-8") as f:
        json.dump(wiki_data, f, indent=2, ensure_ascii=False)
        
    with open("theory.json", "w", encoding="utf-8") as f:
        json.dump(theory_data, f, indent=2, ensure_ascii=False)
        
    print(f"Scraping Complete! Saved {len(wiki_data)} wiki entries and {len(theory_data)} theory entries.")
