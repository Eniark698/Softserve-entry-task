# scraper.py  –  DeepLearning.AI “The Batch” full scraper
# -------------------------------------------------------
# pip install playwright beautifulsoup4 requests
# playwright install
# -------------------------------------------------------
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin, urlparse, parse_qs, unquote
import requests, os, json, time, re, random
from concurrent.futures import ThreadPoolExecutor, as_completed

SEARCH_URL = "https://www.deeplearning.ai/search/"
HEADLESS   = False
PAGE_DELAY = 3.0
IMG_TIMEOUT= 10

ARTICLES_DIR = "data/raw/articles"
IMAGES_DIR   = "data/raw/images"
JSONL_PATH   = "data/raw/batch_articles.jsonl"

os.makedirs(ARTICLES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ───────────────────────── helpers ───────────────────────────────────────
HEADER_PAT = re.compile(
    r".*?\bArticle\b(?:\n+)", re.DOTALL | re.IGNORECASE
)  # non-greedy to 'Article' + blank lines


def clean_text(txt: str) -> str:
    return HEADER_PAT.sub("", txt).lstrip()


def headline_only(h1_tag) -> str:
    if not h1_tag:
        return ""
    span = h1_tag.find("span")
    if span:                                    # headline + <span>subtitle</span>
        parts = [
            n.strip() for n in h1_tag.contents
            if isinstance(n, NavigableString) and n.strip()
        ]
        return " ".join(parts).strip(' "')
    return h1_tag.get_text(" ", strip=True)


def resolve_img(src: str) -> str | None:
    if not src or src.startswith("data:"):
        return None
    if "/_next/image" in src and "url=" in src:
        try:
            src = unquote(parse_qs(urlparse(src).query)["url"][0])
        except Exception:
            pass
    return src


def download_img(url: str, slug: str) -> str | None:
    try:
        r = requests.get(url, timeout=IMG_TIMEOUT,
                         headers={"User-Agent": "Mozilla/5.0"})
        if not r.ok:
            return None
        ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
        fname = f"{slug}{ext}"
        with open(os.path.join(IMAGES_DIR, fname), "wb") as fp:
            fp.write(r.content)
        return fname
    except Exception:
        return None


def links_from_grid(html: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    return {
        urljoin(SEARCH_URL, a["href"])
        for art in soup.find_all("article")
        for a in art.find_all("a", href=True)
    }


# ───────────────────────── main ──────────────────────────────────────────
def scrape():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page    = browser.new_page()
        page.goto(SEARCH_URL, wait_until="load")
        time.sleep(PAGE_DELAY)

        # --- collect all article links via pagination ---
        all_links, pg = set(), 1
        while True:
            page.wait_for_selector("article", timeout=10_000)
            all_links |= links_from_grid(page.content())
            print(f"Page {pg}: total {len(all_links)}")
            next_btn = page.locator("a[aria-label='Next page']").last
            if not next_btn or not next_btn.is_visible():
                break
            next_btn.scroll_into_view_if_needed()
            next_btn.click(force=True)
            page.wait_for_load_state("load")
            pg += 1
            time.sleep(PAGE_DELAY)

        browser.close()
        print(f"\nCollected {len(all_links)} unique URLs\n")

    # --- concurrent fetch & save ---
    ses = requests.Session()
    ses.headers.update({"User-Agent": "Mozilla/5.0"})

    def fetch(url: str):
        slug = url.rstrip("/").split("/")[-1]
        try:
            html = ses.get(url, timeout=15).text
        except Exception as e:
            print(f"!! skip {slug}: {e}")
            return False

        soup = BeautifulSoup(html, "html.parser")
        title = headline_only(soup.find("h1")) or slug.replace("-", " ").title()

        holder = soup.find("div", class_="post-content") or soup.find("article") or soup
        body   = "\n".join(t.get_text(" ", strip=True)
                            for t in holder.find_all(["p", "li"]))
        text   = clean_text(body)

        img_fn = None
        for img in holder.find_all("img", src=True):
            src = resolve_img(img["src"])
            if src and not src.startswith("http"):
                src = urljoin(url, src)
            if src:
                img_fn = download_img(src, slug)
                break

        rec = {"title": title, "url": url, "text": text, "image_filename": img_fn}

        with open(os.path.join(ARTICLES_DIR, f"{slug}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        with open(JSONL_PATH, "a", encoding="utf-8") as jl:
            jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True

    ok = 0
    print("Fetching …")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for res in as_completed(pool.submit(fetch, u) for u in all_links):
            if res.result(): ok += 1
            time.sleep(random.uniform(0.2, 0.5))
    print(f"✅ done — saved {ok} articles")


if __name__ == "__main__":
    scrape()
