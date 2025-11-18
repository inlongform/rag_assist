import asyncio, os, re
from pathlib import Path
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter

BASE_URL = "http://127.0.0.1:5500/docs/_build/current"
START_URL = f"{BASE_URL}/index.html"
OUTPUT_DIR = Path("docs_v2")

# Regex to keep only your docs tree
allowed_pattern = re.compile(r"^http://127\.0\.0\.1:5500/docs/_build/current.*")

def safe_filename(url: str) -> str:
    """Turn a URL into a safe filename for saving."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        path = "index"
    return path.replace("/", "_") + ".md"

# def url_matcher(url: str) -> bool:
#     # ✅ Only allow URLs that begin with your docs prefix
#     return url.startswith(BASE_URL)
allow_filter = URLPatternFilter(patterns=[f"{BASE_URL}*.html"])     # allow only under your docs prefix
deny_filter  = URLPatternFilter(patterns=["*.rst", "*.jpg"]) # block any .rst files



async def main():
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=3,
            include_external=True,
            filter_chain=FilterChain([allow_filter]),
            # max_pages=500
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        markdown_generator=DefaultMarkdownGenerator(),
        verbose=True,
        exclude_all_images=True,
        exclude_social_media_links=True,
        excluded_selector="#pst-primary-sidebar",
        mean_delay=0.5,
        # url_matcher=url_matcher,

    )

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(START_URL, config=config)
        print(f"\nCrawled {len(results)} pages in total\n")

        for result in results:
            if result.success and allowed_pattern.match(result.url):
                filename = safe_filename(result.url)
                filepath = OUTPUT_DIR / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# Source: {result.url}\n\n")
                    f.write(result.markdown.raw_markdown)
                print(f"📝 Saved {result.url} -> {filepath}")
            else:
                if not result.success:
                    print(f"❌ Failed: {result.url} - {result.error_message}")
                else:
                    print(f"⏭️ Skipped (outside base): {result.url}")

if __name__ == "__main__":
    asyncio.run(main())