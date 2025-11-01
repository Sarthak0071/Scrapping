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
        self.model = SentenceTransformer(model_name)
    
    def parse_prompt(self, prompt):
        """Extract intent and website from natural language"""
        # Find domain/website
        patterns = [
            r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)',
            r'(?:from|at|on|in)\s+([a-zA-Z0-9-]+\.[a-zA-Z]{2,})',
            r'([a-zA-Z0-9-]+\.com|\.org|\.net|\.edu|\.gov)'
        ]
        
        domain = None
        for pattern in patterns:
            if match := re.search(pattern, prompt, re.IGNORECASE):
                domain = match.group(1) if '.' in match.group(1) else None
                if domain:
                    break
        
        if not domain:
            return None, None
        
        # Extract intent - keep the natural language, just clean noise words
        intent = prompt.lower()
        
        # Remove the domain reference
        intent = re.sub(rf'\b{re.escape(domain)}\b', '', intent, flags=re.IGNORECASE)
        
        # Remove only obvious scraping commands, keep natural phrases
        noise = [
            r'\b(find|get|fetch|scrape|extract|search|show|give|tell)\s+(me\s+)?(the\s+)?(some\s+)?',
            r'\b(from|at|on|in|of)\s+',
            r'\bhttps?://\S+',
            r'\bwww\.\S+'
        ]
        
        for pattern in noise:
            intent = re.sub(pattern, ' ', intent, flags=re.IGNORECASE)
        
        # Clean up spacing
        intent = re.sub(r'\s+', ' ', intent).strip()
        
        return intent, f"https://{domain}"
    
    def get_cache_path(self, url):
        domain = urlparse(url).netloc.replace('www.', '')
        domain_dir = os.path.join(self.cache_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        return os.path.join(domain_dir, f"page_{hashlib.md5(url.encode()).hexdigest()}.json")
    
    def cache_page(self, url, html, links):
        with open(self.get_cache_path(url), 'w', encoding='utf-8') as f:
            json.dump({'url': url, 'scraped_at': datetime.now().isoformat(), 'html': html, 'links': links}, f, ensure_ascii=False, indent=2)
    
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
            return cached['html'], cached['links']
        
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
                
                # Get base domain for filtering
                base_domain = urlparse(url).netloc.lower().replace('www.', '')
                
                normalized = []
                for link in links:
                    try:
                        full_url = urljoin(url, link['url'])
                        parsed = urlparse(full_url)
                        
                        # Only keep links from same domain
                        link_domain = parsed.netloc.lower().replace('www.', '')
                        if parsed.scheme in ['http', 'https'] and link_domain == base_domain:
                            link['url'] = self.normalize_url(full_url)
                            normalized.append(link)
                    except:
                        continue
                
                self.cache_page(url, html, normalized)
                return html, normalized
                
        except Exception as e:
            return None, []
    
    def compute_similarity(self, intent, links):
        if not links:
            return []
        
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
        intent, website = self.parse_prompt(prompt)
        if not intent or not website:
            return {'error': 'Could not parse intent or website', 'prompt': prompt}
        
        html, links = await self.fetch_with_playwright(website, use_cache)
        if not html:
            return {'error': 'Failed to fetch website', 'website': website}
        if not links:
            return {'error': 'No links found', 'website': website}
        
        matched = self.compute_similarity(intent, links)
        filtered = [l for l in matched if l['similarity_score'] >= min_similarity]
        
        return {
            'prompt': prompt, 'intent': intent, 'website': website, 'scraped_at': datetime.now().isoformat(),
            'total_links': len(links), 'matched_links': filtered, 'cache_location': self.cache_dir,
            'similarity_threshold': min_similarity
        }
    
    def print_results(self, results):
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"Intent: {results['intent']}")
        print(f"Website: {results['website']}")
        
        # Filter and deduplicate
        seen_urls = set()
        filtered = []
        for link in results['matched_links']:
            # Skip empty text, social media, and duplicates
            if (not link['text'].strip() or 
                link['url'] in seen_urls or
                any(sm in link['url'] for sm in ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'youtube.com', 'tiktok.com'])):
                continue
            seen_urls.add(link['url'])
            filtered.append(link)
        
        print(f"Matched: {len(filtered)}\n")
        
        for i, link in enumerate(filtered[:10], 1):
            print(f"{i}. [{link['similarity_score']:.3f}] {link['text'][:80]}")
            print(f"   {link['url']}\n")
    
    def save_results(self, results, output_file="scraping_results.json"):
        history = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = [history]
                except:
                    history = []
        
        cleaned_results = {**results}
        if 'matched_links' in cleaned_results:
            cleaned_results['matched_links'] = [
                {k: v for k, v in link.items() if k != 'combined_text'} 
                for link in cleaned_results['matched_links']
            ]
        
        history.append(cleaned_results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

async def main():
    scraper = IntelligentWebScraper()
    user_prompt = input("Prompt: ").strip()
    
    if user_prompt:
        results = await scraper.scrape(user_prompt, use_cache=True, min_similarity=0.15)
        scraper.print_results(results)
        scraper.save_results(results)
    else:
        print("No prompt provided.")

if __name__ == "__main__":
    asyncio.run(main())