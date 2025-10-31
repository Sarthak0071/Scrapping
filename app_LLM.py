import asyncio
import re
import json
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from typing import List, Dict, Optional
from playwright.async_api import async_playwright
from groq import Groq

@dataclass
class Config:
    max_depth: int = 2
    max_pages: int = 20
    timeout: int = 15000
    concurrent: int = 3
    # groq_key: str = ""

class AI:
    def __init__(self, key: str):
        self.client = Groq(api_key=key)
    
    def parse(self, prompt: str) -> Dict:
        sys = """Extract URL and search intent from user input. Return only JSON:
{"url": "https://example.com", "intent": "what to find"}

Examples:
"pricing from stripe" -> {"url": "https://stripe.com", "intent": "pricing"}
"career from golyan" -> {"url": "https://golyan.com", "intent": "career"}"""

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            content = resp.choices[0].message.content.strip()
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            
            data = json.loads(content)
            return {'url': data.get('url', ''), 'intent': data.get('intent', '')}
        except:
            urls = re.findall(r'https?://[^\s]+', prompt)
            if urls:
                return {'url': urls[0], 'intent': prompt}
            
            for word in prompt.split():
                if '.' in word:
                    url = word if word.startswith('http') else f'https://{word}'
                    return {'url': url, 'intent': prompt}
            
            return {'url': '', 'intent': prompt}

class Matcher:
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        stop = {'the','a','an','and','or','but','in','on','at','to','for','of','with',
                'by','from','this','that','is','are','was','were','be','get','find','page'}
        words = re.findall(r'\b[a-z]{2,}\b', text.lower())
        return [w for w in words if w not in stop][:5]
    
    @staticmethod
    def is_relevant(url: str, text: str, keywords: List[str]) -> bool:
        if not keywords:
            return True
        
        combined = f"{url} {text}".lower()
        return any(kw in combined for kw in keywords)

class Scraper:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.visited = set()
        self.results = []
        self.ai = AI(cfg.groq_key)
    
    async def verify(self, url: str) -> Optional[str]:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                resp = await page.goto(url, timeout=self.cfg.timeout, wait_until='domcontentloaded')
                final = page.url if resp and resp.status < 400 else None
                await browser.close()
                return final
        except:
            return None
    
    async def run(self, prompt: str) -> List[Dict]:
        print("Analyzing request...")
        parsed = self.ai.parse(prompt)
        
        if not parsed['url']:
            print("Error: Could not extract URL")
            return []
        
        print(f"URL: {parsed['url']}")
        print(f"Intent: {parsed['intent']}\n")
        
        url = await self.verify(parsed['url'])
        if not url:
            print("Error: URL not accessible")
            return []
        
        print(f"Verified: {url}\n")
        return await self.scrape(url, parsed['intent'])
    
    async def scrape(self, url: str, intent: str) -> List[Dict]:
        keywords = Matcher.extract_keywords(intent)
        print(f"Keywords: {', '.join(keywords)}")
        print(f"Scraping...\n")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context()
            
            queue = [(url, 0)]
            while queue and len(self.results) < self.cfg.max_pages:
                batch = [queue.pop(0) for _ in range(min(self.cfg.concurrent, len(queue))) if queue]
                tasks = [self._process(ctx, u, d, keywords) for u, d in batch]
                
                for res in await asyncio.gather(*tasks, return_exceptions=True):
                    if isinstance(res, dict) and res.get('links'):
                        new = [(l['url'], res['depth']+1) for l in res['links']
                               if l['url'] not in self.visited and res['depth'] < self.cfg.max_depth]
                        queue.extend(new)
            
            await browser.close()
        
        # Filter results that match keywords
        relevant = [r for r in self.results if r.get('is_relevant')]
        return relevant if relevant else self.results[:3]
    
    async def _process(self, ctx, url: str, depth: int, keywords: List[str]) -> Dict:
        if url in self.visited:
            return {}
        
        self.visited.add(url)
        page = await ctx.new_page()
        
        try:
            print(f"Processing: {url[:60]}")
            await page.goto(url, wait_until='networkidle', timeout=self.cfg.timeout)
            
            title = await page.title()
            content = await page.evaluate('document.body.innerText')
            headings = await page.evaluate(
                '() => Array.from(document.querySelectorAll("h1,h2,h3")).map(h=>h.innerText.trim()).filter(t=>t).slice(0,10)'
            )
            
            # Check relevance
            full_text = f"{title} {' '.join(headings)} {content}"
            is_relevant = Matcher.is_relevant(url, full_text, keywords)
            
            # Extract links
            links = []
            base_domain = urlparse(url).netloc
            for el in await page.query_selector_all('a[href]'):
                try:
                    href = await el.get_attribute('href')
                    if not href:
                        continue
                    
                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)
                    
                    if not parsed.scheme.startswith('http') or parsed.netloc != base_domain:
                        continue
                    
                    link_text = (await el.inner_text()).strip()
                    
                    # Only follow relevant links
                    if Matcher.is_relevant(full_url, link_text, keywords):
                        links.append({'url': full_url, 'text': link_text})
                except:
                    pass
            
            # Dedupe
            seen = set()
            unique = [l for l in links if l['url'] not in seen and not seen.add(l['url'])][:25]
            
            result = {
                'url': url,
                'depth': depth,
                'title': title,
                'headings': headings,
                'preview': content[:500].strip(),
                'is_relevant': is_relevant,
                'links': unique
            }
            
            self.results.append(result)
            status = "MATCH" if is_relevant else "visited"
            print(f"{status}\n")
            return result
            
        except Exception as e:
            print(f"Error: {str(e)[:50]}\n")
            return {'url': url, 'depth': depth, 'error': str(e)}
        finally:
            await page.close()

async def main():
    print("AI Web Scraper\n")
    prompt = input("What do you want to find? ").strip()
    
    if not prompt:
        print("Error: Empty input")
        return
    
    print()
    scraper = Scraper()
    
    try:
        results = await scraper.run(prompt)
        
        if not results:
            print("No results found")
            return
        
        print(f"\nFound {len(results)} relevant pages:\n")
        
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['url']}")
            if r.get('headings'):
                print(f"   Sections: {', '.join(r['headings'][:3])}")
            print()
        
        with open('results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("Saved to results.json")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())