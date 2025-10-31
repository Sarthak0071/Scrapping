import asyncio
import re
import json
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from typing import List, Dict
from playwright.async_api import async_playwright

@dataclass
class Config:
    max_depth: int = 2
    max_pages: int = 15
    timeout: int = 15000
    min_score: float = 0.90
    concurrent: int = 3

class NLP:
    STOP = {'the','a','an','and','or','but','in','on','at','to','for','of','with',
            'by','from','this','that','is','are','was','were','be','get','find',
            'show','me','about','site','page'}
    
    @staticmethod
    def extract(prompt: str) -> Dict:
        words = re.findall(r'\b[a-z]{2,}\b', prompt.lower())
        keywords = [w for w in words if w not in NLP.STOP][:5]
        patterns = [*keywords, *[f"{k}s" for k in keywords], 
                   '-'.join(keywords) if len(keywords)>1 else '']
        return {'keywords': keywords, 'patterns': [p for p in patterns if p]}

class Scorer:
    @staticmethod
    def text(text: str, keywords: List[str]) -> float:
        if not text or not keywords: return 0.0
        txt = text.lower()
        total = 0.0
        for kw in keywords:
            if kw in txt:
                count = txt.count(kw)
                total += min(count * 0.3, 1.0)
        return min(total / len(keywords), 1.0)
    
    @staticmethod
    def url(url: str, patterns: List[str]) -> float:
        if not patterns: return 0.0
        url_lower = url.lower()
        matches = sum(2.0 if p in url_lower else 0.0 for p in patterns)
        return min(matches / len(patterns), 1.0)

class Scraper:
    def __init__(self, config: Config = Config()):
        self.cfg = config
        self.visited = set()
        self.results = []
    
    async def scrape(self, url: str, prompt: str) -> List[Dict]:
        intent = NLP.extract(prompt)
        print(f"Keywords: {', '.join(intent['keywords'])}")
        print(f"Starting: {url}\n")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            try:
                queue = [(url, 0)]
                while queue and len(self.results) < self.cfg.max_pages:
                    batch = [queue.pop(0) for _ in range(min(self.cfg.concurrent, len(queue))) if queue]
                    tasks = [self._process(ctx, u, d, intent) for u, d in batch]
                    for res in await asyncio.gather(*tasks, return_exceptions=True):
                        if isinstance(res, dict) and res.get('links'):
                            queue.extend((l['url'], res['depth']+1) for l in res['links']
                                       if l['url'] not in self.visited and res['depth'] < self.cfg.max_depth)
            finally:
                await ctx.close()
                await browser.close()
        
        # Sort all results by score descending
        sorted_results = sorted(self.results, key=lambda x: x['data'].get('score', 0), reverse=True)

        # Take top 2 results (or all if less than 2)
        top_results = sorted_results[:2]

        return top_results
    
    async def _process(self, ctx, url: str, depth: int, intent: Dict) -> Dict:
        if url in self.visited: return {}
        self.visited.add(url)
        page = await ctx.new_page()
        
        try:
            print(f"Scraping: {url[:70]}...")
            await page.goto(url, wait_until='networkidle', timeout=self.cfg.timeout)
            await page.wait_for_timeout(1000)
            
            title = await page.title()
            content = await page.evaluate('document.body.innerText')
            
            headings = await page.evaluate('''() => Array.from(document.querySelectorAll('h1,h2,h3,h4'))
                .map(h=>h.innerText.trim()).filter(t=>t).slice(0,15)''')
            
            full_text = f"{title} {' '.join(headings)} {content}"
            text_score = Scorer.text(full_text, intent['keywords'])
            url_score = Scorer.url(url, intent['patterns'])
            score = (text_score * 0.7) + (url_score * 0.3)
            
            links = []
            for elem in await page.query_selector_all('a[href]'):
                try:
                    href = await elem.get_attribute('href')
                    text = (await elem.inner_text()).strip()
                    if not href: continue
                    full = urljoin(url, href)
                    parsed = urlparse(full)
                    if not parsed.scheme.startswith('http'): continue
                    if parsed.netloc != urlparse(url).netloc: continue
                    
                    link_url_score = Scorer.url(full, intent['patterns'])
                    link_text_score = Scorer.text(text, intent['keywords'])
                    s = (link_url_score * 0.6) + (link_text_score * 0.4)
                    if s > 0.05:
                        links.append({'url': full, 'text': text, 'score': s})
                except: pass
            
            links = sorted(links, key=lambda x: x['score'], reverse=True)
            seen = set()
            unique = [l for l in links if l['url'] not in seen and not seen.add(l['url'])][:25]
            
            result = {
                'url': url,
                'depth': depth,
                'data': {
                    'title': title,
                    'headings': headings,
                    'preview': content[:600].strip(),
                    'word_count': len(content.split()),
                    'score': round(score, 3)
                },
                'links': unique
            }
            
            self.results.append(result)  # Always append
            print(f"Scraped: {score:.2f}")
            
            return result
        except Exception as e:
            print(f"Error: {str(e)[:60]}")
            return {'url': url, 'depth': depth, 'error': str(e)}
        finally:
            await page.close()

async def main():
    print("Universal Web Scraper\n")
    url = input("Website URL: ").strip()
    if not url.startswith('http'): url = 'https://' + url
    prompt = input("What to find: ").strip()
    if not prompt:
        print("Error: Empty prompt")
        return
    
    print()
    scraper = Scraper()
    
    try:
        results = await scraper.scrape(url, prompt)
        print(f"\nFound {len(results)} top pages\n")
        
        for i, r in enumerate(results, 1):
            d = r['data']
            print(f"{i}. Score: {d.get('score',0):.2f}")
            print(f"   {r['url']}")
            print(f"   {d.get('title','No title')}")
            if d.get('headings'): 
                print(f"   Sections: {', '.join(d['headings'][:4])}")
            print()
        
        with open('results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Saved to results.json")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
