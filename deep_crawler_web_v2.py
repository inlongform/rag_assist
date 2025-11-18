from urllib.parse import urljoin, urlparse
import requests, os, time
from bs4 import BeautifulSoup
import html2text

# -----------------------
# CONFIG
# -----------------------
START_URL = "https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/108.1.0/pxr.html"
BASE_PREFIX = "https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/108.1.0"

OUTPUT_DIR = "docs/pxr_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

visited = set()
queue = [START_URL]
CRAWL_DELAY = 1.0

# Markdown converter
html2md = html2text.HTML2Text()
html2md.ignore_links = False
html2md.body_width = 0  # Keep formatting

def remove_media_and_social(soup):
    # Remove media elements entirely
    for tag in soup.find_all(["img", "picture", "video", "iframe", "figure"]):
        tag.decompose()

    # Remove social/external marketing links
    SOCIAL_DOMAINS = [
        "youtube.com", "youtu.be", "twitter.com", "x.com",
        "linkedin.com", "facebook.com", "instagram.com",
        "discord.com", "tiktok.com"
    ]

    for a in soup.find_all("a", href=True):
        if any(domain in a["href"] for domain in SOCIAL_DOMAINS):
            a.decompose()


def url_to_filename(url):
    path = urlparse(url).path
    if path.endswith("/"):
        path += "index.html"
    filename = path.replace("/", "_").replace(".html", "")
    return os.path.join(OUTPUT_DIR, filename + ".md")


def preserve_code_blocks(soup):
    """Convert <pre><code> blocks to fenced code blocks before markdown conversion."""
    for pre in soup.find_all("pre"):
        code = pre.get_text("\n", strip=False)
        fenced = f"\n```\n{code}\n```\n"
        pre.replace_with(fenced)


# -----------------------
# CRAWL LOOP
# -----------------------
MEDIA_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".mp4", ".mov", ".pdf", "changelog.html", "contributing.html")

while queue:
    url = queue.pop(0)

    if url in visited:
        continue
    if not url.startswith(BASE_PREFIX):
        continue
    if "/api/" in url or url.lower().endswith(MEDIA_EXTENSIONS):
        continue

    visited.add(url)
    filename = url_to_filename(url)

    if os.path.exists(filename):
        print(f"⏭ Skipping: {url}")
        continue

    try:
      resp = requests.get(url, timeout=10)
      soup = BeautifulSoup(resp.text, "html.parser")

      # Remove sidebar
      sidebar = soup.select_one("#pst-primary-sidebar")
      if sidebar:
          sidebar.decompose()

      # Remove images, video, and social links
      remove_media_and_social(soup)

      # Preserve code blocks
      preserve_code_blocks(soup)

      # Convert to markdown
      markdown = html2md.handle(str(soup))

      # Save .md
      with open(filename, "w", encoding="utf-8") as f:
          f.write(markdown)

      print(f"✅ Saved: {url}")

      # Collect more links
      for a in soup.find_all("a", href=True):
          link = urljoin(url, a["href"])
          if link.startswith(BASE_PREFIX) and "/api/" not in link:
              queue.append(link)

      time.sleep(CRAWL_DELAY)

    except Exception as e:
        print(f"❌ Failed {url}: {e}")

print(f"\n✅ Done. Pages saved: {len(visited)}")
print(f"📂 Output folder: {OUTPUT_DIR}")
