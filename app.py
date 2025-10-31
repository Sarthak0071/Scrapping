import asyncio
import re
import json
import os
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from typing import List, Dict
from playwright.async_api import async_playwright

@dataclass
class Config:
    max_depth: int = 3
    max_pages: int = 50
    timeout: int = 15000
    concurrent: int = 3

class NLP:
    STOP = {'the','a','an','and','or','but','in','on','at','to','for','of','with',
            'by','from','this','that','is','are','was','were','be','get','find',
            'show','me','about','site','page','pages'}
    
    @staticmethod
    def extract(prompt: str) -> Dict:
        words = re.findall(r'\b[a-z]{2,}\b', prompt.lower())
        keywords = [w for w in words if w not in NLP.STOP]
        patterns = [*keywords, *[f"{k}s" for k in keywords], 
                   '-'.join(keywords) if len(keywords)>1 else '']
        return {'keywords': keywords, 'patterns': [p for p in patterns if p]}

class Scorer:
    @staticmethod
    def score_page(url: str, title: str, text: str, keywords: List[str], patterns: List[str]) -> float:
        if not keywords: return 0.0
        
        u = url.lower()
        t = title.lower()
        txt = text.lower()
        
        score = 0.0
        
        # URL match - highest priority
        for p in patterns:
            if p in u:
                score += 5.0
        
        # Title match - high priority
        for kw in keywords:
            if kw in t:
                score += 3.0
        
        # Content match
        for kw in keywords:
            if kw in txt:
                count = txt.count(kw)
                score += min(count * 0.5, 2.0)
        
        return min(score, 10.0)
    
    @staticmethod
    def score_link(url: str, text: str, keywords: List[str], patterns: List[str]) -> float:
        u = url.lower()
        t = text.lower()
        
        score = 0.0
        for p in patterns:
            if p in u:
                score += 3.0
        for kw in keywords:
            if kw in t or kw in u:
                score += 1.0
        
        return score

class Scraper:
    def __init__(self, config: Config = Config()):
        self.cfg = config
        self.visited = set()
        self.results = []
    
    async def scrape(self, url: str, prompt: str) -> List[Dict]:
        intent = NLP.extract(prompt)
        if not intent['keywords']:
            print("No keywords found")
            return []
        
        print(f"Keywords: {', '.join(intent['keywords'])}")
        print(f"Starting: {url}\n")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            try:
                queue = [(url, 0)]
                while queue and len(self.visited) < self.cfg.max_pages:
                    batch = [queue.pop(0) for _ in range(min(self.cfg.concurrent, len(queue))) if queue]
                    tasks = [self._process(ctx, u, d, intent) for u, d in batch]
                    for res in await asyncio.gather(*tasks, return_exceptions=True):
                        if isinstance(res, dict) and res.get('links') and res['depth'] < self.cfg.max_depth:
                            for l in res['links']:
                                if l['url'] not in self.visited and (l['url'], res['depth']+1) not in queue:
                                    queue.append((l['url'], res['depth']+1))
            finally:
                await ctx.close()
                await browser.close()
        
        self.results.sort(key=lambda x: x['data'].get('score', 0), reverse=True)
        
        # Filter: only return pages with score >= 2.0
        relevant = [r for r in self.results if r['data'].get('score', 0) >= 2.0]
        return relevant
    
    async def _process(self, ctx, url: str, depth: int, intent: Dict) -> Dict:
        if url in self.visited: return {}
        self.visited.add(url)
        page = await ctx.new_page()
        
        try:
            print(f"[{depth}] {url[:70]}...")
            await page.goto(url, wait_until='domcontentloaded', timeout=self.cfg.timeout)
            await page.wait_for_timeout(500)
            
            title = await page.title()
            content = await page.evaluate('document.body.innerText')
            headings = await page.evaluate('''() => Array.from(document.querySelectorAll('h1,h2,h3,h4'))
                .map(h=>h.innerText.trim()).filter(t=>t).slice(0,15)''')
            
            full_text = f"{title} {' '.join(headings)} {content[:3000]}"
            score = Scorer.score_page(url, title, full_text, intent['keywords'], intent['patterns'])
            
            links = []
            base_domain = urlparse(url).netloc
            for elem in await page.query_selector_all('a[href]'):
                try:
                    href = await elem.get_attribute('href')
                    text = (await elem.inner_text()).strip()
                    if not href: continue
                    full = urljoin(url, href)
                    parsed = urlparse(full)
                    if not parsed.scheme.startswith('http'): continue
                    if parsed.netloc != base_domain: continue
                    
                    link_score = Scorer.score_link(full, text, intent['keywords'], intent['patterns'])
                    if link_score > 1.0:
                        links.append({'url': full, 'text': text, 'score': link_score})
                except: pass
            
            links = sorted(links, key=lambda x: x['score'], reverse=True)
            seen = set()
            unique = [l for l in links if l['url'] not in seen and not seen.add(l['url'])][:30]
            
            result = {
                'url': url,
                'depth': depth,
                'data': {
                    'title': title,
                    'headings': headings,
                    'preview': content[:600].strip(),
                    'word_count': len(content.split()),
                    'score': round(score, 2)
                },
                'links': unique
            }
            
            self.results.append(result)
            print(f"   Score: {score:.2f}")
            return result
        except Exception as e:
            print(f"   Error: {str(e)[:60]}")
            return {'url': url, 'depth': depth}
        finally:
            await page.close()

def parse_prompt(user_input: str) -> Dict:
    prompt_match = re.search(r"prompt:\s*['\"]([^'\"]+)['\"]", user_input, re.IGNORECASE)
    url_match = re.search(r"url:\s*['\"]([^'\"]+)['\"]", user_input, re.IGNORECASE)
    
    if prompt_match and url_match:
        url = url_match.group(1)
        if not url.startswith('http'):
            url = 'https://' + url
        return {'prompt': prompt_match.group(1), 'url': url}
    
    pattern = re.search(r'find\s+(.+?)\s+from\s+([a-z0-9\-\.]+\.[a-z]{2,})', user_input, re.IGNORECASE)
    if pattern:
        keyword = pattern.group(1).strip()
        domain = pattern.group(2).strip()
        url = domain if domain.startswith('http') else f'https://{domain}'
        return {'prompt': f'find {keyword}', 'url': url}
    
    return None

async def main():
    print("Web Scraper\n")
    user_input = input("Prompt: ").strip()
    
    parsed = parse_prompt(user_input)
    if not parsed:
        print("Error: Invalid format")
        print("Examples:")
        print("  {prompt: 'find annual report', url: 'site.com'}")
        print("  find career from nabilbank.com")
        return
    
    print(f"\nSearching: {parsed['prompt']}")
    print(f"URL: {parsed['url']}\n")
    
    scraper = Scraper()
    
    try:
        results = await scraper.scrape(parsed['url'], parsed['prompt'])
        print(f"\nFound {len(results)} pages\n")
        
        for i, r in enumerate(results, 1):
            d = r['data']
            print(f"{i}. Score: {d.get('score',0):.2f}")
            print(f"   {r['url']}")
            print(f"   {d.get('title','No title')}")
            if d.get('headings'): 
                print(f"   Sections: {', '.join(d['headings'][:4])}")
            print()
        
        # Load existing results if file exists
        existing_results = []
        if os.path.exists('results.json'):
            with open('results.json', 'r', encoding='utf-8') as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    existing_results = []

        # Append new results
        combined_results = existing_results + results

        # Save combined data back
        with open('results.json', 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)

        print("Results appended and saved to results.json")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
