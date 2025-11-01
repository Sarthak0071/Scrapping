import os
import re
import json
import hashlib
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from sentence_transformers import SentenceTransformer
import numpy as np

class IntelligentWebScraper:
    def __init__(self, cache_dir="scraper_cache", model_name="all-MiniLM-L6-v2"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[LOADING] {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"[READY]\n")
    
    def parse_prompt(self, prompt):
        """Extract intent and website"""
        patterns = [r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)', 
                   r'from\s+([a-zA-Z0-9-]+\.[a-zA-Z]{2,})', r'at\s+([a-zA-Z0-9-]+\.[a-zA-Z]{2,})']
        
        domain = next((m.group(1) for p in patterns if (m := re.search(p, prompt)) and '.' in m.group(1)), None)
        if not domain:
            return None, None
        
        intent = prompt.lower()
        for word in ['find', 'get', 'fetch', 'scrape', 'extract', 'search', 'look for', 'from', 'at', domain]:
            intent = intent.replace(word, '')
        
        return re.sub(r'\s+', ' ', intent).strip(), f"https://{domain}"
    
    def get_cache_path(self, url):
        domain = urlparse(url).netloc.replace('www.', '')
        domain_dir = os.path.join(self.cache_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        return os.path.join(domain_dir, f"page_{hashlib.md5(url.encode()).hexdigest()}.json")
    
    def cache_page(self, url, html, links):
        with open(self.get_cache_path(url), 'w', encoding='utf-8') as f:
            json.dump({'url': url, 'scraped_at': datetime.now().isoformat(), 'html': html, 'links': links}, f, ensure_ascii=False, indent=2)
        print(f"[CACHED] {url}")
    
    def get_cached_page(self, url):
        cache_path = self.get_cache_path(url)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def normalize_url(self, url):
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip('/') or '/', p.params, p.query, ''))
    
    async def fetch_with_playwright(self, url, use_cache=True):
        url = self.normalize_url(url)
        
        if use_cache and (cached := self.get_cached_page(url)):
            print(f"[CACHE HIT] {url}")
            return cached['html'], cached['links']
        
        print(f"[FETCHING] {url}")
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await (await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')).new_page()
                await page.goto(url, wait_until='networkidle', timeout=30000)
                await page.wait_for_timeout(2000)
                
                html = await page.content()
                links = await page.evaluate('''() => {
                    return Array.from(document.querySelectorAll('a')).map(a => {
                        if (!a.href || ['javascript:', 'mailto:', 'tel:'].some(x => a.href.startsWith(x))) return null;
                        let context = '', parent = a.parentElement;
                        for (let i = 0; i < 3 && parent; i++) {
                            if (parent.innerText?.length < 300) { context = parent.innerText; break; }
                            parent = parent.parentElement;
                        }
                        return {
                            url: a.href, text: a.innerText.trim(), title: a.getAttribute('title') || '',
                            ariaLabel: a.getAttribute('aria-label') || '',
                            heading: a.closest('section, article, div')?.querySelector('h1,h2,h3,h4,h5,h6')?.innerText.trim() || '',
                            context: context.substring(0, 300)
                        };
                    }).filter(x => x);
                }''')
                
                await browser.close()
                
                normalized = [
                    {**link, 'url': self.normalize_url(urljoin(url, link['url']))} 
                    for link in links 
                    if urlparse(urljoin(url, link['url'])).scheme in ['http', 'https']
                ]
                
                print(f"[EXTRACTED] {len(normalized)} links")
                self.cache_page(url, html, normalized)
                return html, normalized
                
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return None, []
    
    def compute_similarity(self, intent, links):
        if not links:
            return []
        
        print(f"[EMBEDDING] {len(links)} links...")
        
        intent_emb = self.model.encode(intent, convert_to_tensor=False)
        link_texts = [re.sub(r'\s+', ' ', f"{l['text']} {l['title']} {l['ariaLabel']} {l['heading']} {l['url']}").strip() or "empty" for l in links]
        link_embs = self.model.encode(link_texts, convert_to_tensor=False, show_progress_bar=False)
        
        results = sorted([
            {**link, 'similarity_score': float(np.dot(intent_emb, link_embs[i]) / (np.linalg.norm(intent_emb) * np.linalg.norm(link_embs[i]))), 
             'combined_text': link_texts[i]}
            for i, link in enumerate(links)
        ], key=lambda x: x['similarity_score'], reverse=True)
        
        return results
    
    async def scrape(self, prompt, use_cache=True, min_similarity=0.3):
        print(f"\nSCRAPING REQUEST: {prompt}\n")
        
        intent, website = self.parse_prompt(prompt)
        if not intent or not website:
            return {'error': 'Could not parse intent or website', 'prompt': prompt}
        
        print(f"Intent: '{intent}'\nWebsite: {website}\n")
        
        html, links = await self.fetch_with_playwright(website, use_cache)
        if not html:
            return {'error': 'Failed to fetch website', 'website': website}
        if not links:
            return {'error': 'No links found', 'website': website}
        
        matched = self.compute_similarity(intent, links)
        filtered = [l for l in matched if l['similarity_score'] >= min_similarity]
        
        print(f"\n[MATCHED] {len(filtered)} links above {min_similarity}\n")
        
        return {
            'prompt': prompt, 'intent': intent, 'website': website, 'scraped_at': datetime.now().isoformat(),
            'total_links': len(links), 'matched_links': filtered, 'cache_location': self.cache_dir,
            'similarity_threshold': min_similarity
        }
    
    def print_results(self, results):
        if 'error' in results:
            print(f"\n[ERROR] {results['error']}")
            return
        
        print(f"\nRESULTS\n")
        print(f"Intent: {results['intent']}\nWebsite: {results['website']}")
        print(f"Total: {results['total_links']} | Matched: {len(results['matched_links'])} (threshold: {results['similarity_threshold']})")
        print(f"Cache: {results['cache_location']}\n")
        
        if results['matched_links']:
            print(f"TOP MATCHES\n")
            for i, link in enumerate(results['matched_links'][:20], 1):
                print(f"{i}. [{link['similarity_score']:.3f}] {link['text'][:80]}")
                print(f"   URL: {link['url']}")
                if link['heading']:
                    print(f"   Section: {link['heading'][:80]}")
                if link['title']:
                    print(f"   Title: {link['title'][:80]}")
                print()
    
    def save_results(self, results, output_file="scraping_results.json"):
        """Append results to JSON file instead of overwriting"""
        history = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = [history]
                except:
                    history = []
        
        # Remove combined_text from matched links to reduce file size
        cleaned_results = {**results}
        if 'matched_links' in cleaned_results:
            cleaned_results['matched_links'] = [
                {k: v for k, v in link.items() if k != 'combined_text'} 
                for link in cleaned_results['matched_links']
            ]
        
        history.append(cleaned_results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults appended to: {output_file} (total entries: {len(history)})")

async def main():
    scraper = IntelligentWebScraper()
    
    print("INTELLIGENT WEB SCRAPER")
    print("\nExample: 'find annual reports from nabilbank.com'\n")
    
    user_prompt = input("Your prompt: ").strip()
    
    if user_prompt:
        results = await scraper.scrape(user_prompt, use_cache=True, min_similarity=0.25)
        scraper.print_results(results)
        scraper.save_results(results)
        
        # print("\nThis scraper extracts your intent from natural language, fetches the website using Playwright for dynamic content, caches pages locally, and uses AI embeddings to find links semantically similar to what you're looking for. Results are appended to a JSON file with timestamps for history tracking.")
    else:
        print("No prompt provided.")

if __name__ == "__main__":
    asyncio.run(main())