"""
Fashion Product Scraper — Hybrid Data Collection
=================================================
Scrapes real clothing product data from publicly accessible sources and outputs
a CSV that matches the existing fashion_data_2018_2022.csv schema (plus a
product_url column so recommendations can link back to the actual item).

Sources:
  1. H&M (hm.com)
     Static HTML product listings. Scraped with requests + BeautifulSoup.
     Categories: Women's, Men's — Tops, Dresses, Jackets, Trousers, Skirts.

  2. Shopify stores (public /products.json endpoint)
     Any Shopify store exposes structured product JSON at /products.json
     with no authentication required. We target a few fashion-focused stores.
     Returns: title, vendor, product_type, variants (price, color, size), images.

Output:
  ./data/scraped_products.csv  — same schema as fashion_data_2018_2022.csv
                                 plus product_url column

Usage:
  pip install requests beautifulsoup4 lxml
  python scraper.py

  # Scrape only specific sources:
  python scraper.py --source hm
  python scraper.py --source shopify
  python scraper.py --source all   (default)

  # Limit items per source (useful for quick tests):
  python scraper.py --limit 50

Notes:
  - Adds a 1-2 second delay between requests to be polite to servers.
  - Respects robots.txt guidance (no login-gated pages, no checkout flows).
  - For a class proof-of-concept, 200-500 items across sources is plenty.
"""

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_FILE = "./data/scraped_products.csv"

# Polite delay range between requests (seconds)
DELAY_MIN = 1.0
DELAY_MAX = 2.5

# Browser-like headers to avoid being blocked by basic bot detection
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# CSV columns — matches fashion_data_2018_2022.csv schema + product_url
CSV_COLUMNS = [
    "product_id", "product_name", "gender", "category", "pattern",
    "color", "age_group", "season", "price", "material",
    "sales_count", "reviews_count", "average_rating",
    "out_of_stock_times", "brand", "discount", "last_stock_date",
    "wish_list_count", "month_of_sale", "year_of_sale",
    "product_url",   # added for hybrid mode — links back to the real item
]

# Current year for metadata
CURRENT_YEAR = datetime.now().year


# =============================================================================
# Utility Helpers
# =============================================================================

_product_id_counter = 9000  # start above the existing dataset's IDs


def next_id() -> int:
    global _product_id_counter
    _product_id_counter += 1
    return _product_id_counter


def polite_sleep():
    """Sleep a random amount between requests to avoid hammering servers."""
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def safe_get(url: str, params: dict = None, timeout: int = 15) -> Optional[requests.Response]:
    """
    GET a URL with browser-like headers. Returns None on any error so the
    scraper can skip a failed page and continue rather than crashing.
    """
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        print(f"    [WARN] Request failed for {url}: {e}")
        return None


def infer_season(product_name: str, description: str = "") -> str:
    """
    Infer a season from product name / description keywords.
    Falls back to 'All' if no signal is found.
    """
    text = (product_name + " " + description).lower()
    if any(w in text for w in ["summer", "beach", "swim", "linen", "shorts", "tank"]):
        return "Summer"
    if any(w in text for w in ["winter", "wool", "knit", "coat", "puffer", "fleece", "thermal"]):
        return "Winter"
    if any(w in text for w in ["spring", "floral", "light jacket", "trench"]):
        return "Spring"
    if any(w in text for w in ["autumn", "fall", "corduroy", "suede", "leather jacket"]):
        return "Autumn"
    return "All"


def infer_pattern(product_name: str, description: str = "") -> str:
    """Infer a pattern from product name / description keywords."""
    text = (product_name + " " + description).lower()
    if any(w in text for w in ["stripe", "striped"]):
        return "Striped"
    if any(w in text for w in ["floral", "flower", "botanical"]):
        return "Floral"
    if any(w in text for w in ["check", "plaid", "tartan", "gingham"]):
        return "Checked"
    if any(w in text for w in ["polka", "dot"]):
        return "Polka Dots"
    if any(w in text for w in ["animal", "leopard", "zebra", "snake"]):
        return "Animal Print"
    if any(w in text for w in ["geometric", "abstract"]):
        return "Geometric"
    if any(w in text for w in ["camo", "camouflage"]):
        return "Camouflage"
    return "Solid"


def infer_material(product_name: str, description: str = "") -> str:
    """Infer a material from product name / description keywords."""
    text = (product_name + " " + description).lower()
    for mat in ["cotton", "linen", "wool", "silk", "polyester", "denim",
                "leather", "suede", "velvet", "satin", "nylon", "cashmere",
                "viscose", "rayon", "synthetic", "jersey", "fleece"]:
        if mat in text:
            return mat.capitalize()
    return "Mixed"


def infer_age_group(product_name: str, description: str = "") -> str:
    """Infer target age group from product name / description keywords."""
    text = (product_name + " " + description).lower()
    if any(w in text for w in ["teen", "junior", "youth", "girl", "boy"]):
        return "13-17"
    if any(w in text for w in ["senior", "mature", "classic fit"]):
        return "45+"
    return "18-35"  # default for most fashion-forward items


def normalise_category(raw: str) -> str:
    """
    Map a raw product type string to one of the categories used in the
    existing dataset: Dress, Shirt, Jacket, Jeans, Shorts, Skirt, Shoes,
    Blouse, Activewear.
    Returns None for swimwear so it can be filtered out at save time.
    Falls back to the raw value (title-cased) if no match.
    """
    raw_lower = raw.lower()

    # Blocked content — return None so the caller skips this product entirely
    if any(w in raw_lower for w in [
        # Swimwear
        "swim", "swimwear", "swimsuit", "bikini", "bathing suit",
        "board short", "rashguard", "wetsuit", "one-piece",
        # Bodysuits / intimates / lingerie
        "bodysuit", "body suit", "intimates", "lingerie", "corset",
        "bralette", "thong", "garter", "lace pack", "bustier",
        "plunge shell", "sexy",
    ]):
        return None

    mapping = {
        "dress":      "Dress",    "dresses":    "Dress",
        "top":        "Shirt",    "tops":       "Shirt",    "t-shirt":   "Shirt",
        "shirt":      "Shirt",    "shirts":     "Shirt",    "blouse":    "Blouse",
        "jacket":     "Jacket",   "jackets":    "Jacket",   "coat":      "Jacket",
        "coats":      "Jacket",   "blazer":     "Jacket",
        "jeans":      "Jeans",    "denim":      "Jeans",
        "trousers":   "Jeans",    "pants":      "Jeans",    "leggings":  "Activewear",
        "shorts":     "Shorts",
        "skirt":      "Skirt",    "skirts":     "Skirt",
        "shoes":      "Shoes",    "sneakers":   "Shoes",    "boots":     "Shoes",
        "heels":      "Shoes",    "sandals":    "Shoes",    "footwear":  "Shoes",
        "sweater":    "Shirt",    "hoodie":     "Shirt",    "sweatshirt":"Shirt",
        "jumpsuit":   "Dress",    "romper":     "Dress",
        # Activewear categories
        "activewear": "Activewear", "sportswear": "Activewear",
        "sports bra": "Activewear", "sports top": "Activewear",
        "training":   "Activewear", "workout":    "Activewear",
        "yoga":       "Activewear", "gym":        "Activewear",
        "compression":"Activewear", "performance":"Activewear",
        "athletic":   "Activewear", "running":    "Activewear",
        "cycling":    "Activewear", "tennis":     "Activewear",
        "tights":     "Activewear", "biker short":"Activewear",
    }
    for key, val in mapping.items():
        if key in raw_lower:
            return val
    return raw.title() if raw else "Other"


def normalise_gender(raw: str) -> str:
    """Map raw gender string to Male / Female / Unisex."""
    raw_lower = raw.lower()
    if any(w in raw_lower for w in ["women", "woman", "female", "girl", "ladies"]):
        return "Female"
    if any(w in raw_lower for w in ["men", "man", "male", "boy", "guys"]):
        return "Male"
    return "Unisex"


def extract_color(title: str, option_name: str = "", option_value: str = "") -> str:
    """
    Extract a color from a product title or variant option.
    Returns 'Multicolor' if nothing useful is found.
    """
    # Prefer explicit variant color option
    if option_name.lower() in ("color", "colour") and option_value:
        return option_value.strip().title()

    # Common color words to scan for in the title
    colors = [
        "black", "white", "grey", "gray", "navy", "blue", "red", "green",
        "yellow", "orange", "pink", "purple", "brown", "beige", "cream",
        "ivory", "camel", "khaki", "olive", "teal", "coral", "burgundy",
        "mustard", "rust", "lavender", "mint", "gold", "silver",
    ]
    title_lower = title.lower()
    for color in colors:
        if color in title_lower:
            return color.capitalize()
    return "Multicolor"


# =============================================================================
# Source 1 — Walmart  (walmart.com)
# =============================================================================
# NOTE: Walmart's product grid is rendered by React on the client side.
# A plain requests fetch returns the page shell but product cards are injected
# by JavaScript after load — so BeautifulSoup finds 0 cards.
# Walmart is therefore skipped in favour of additional Shopify stores below.

def scrape_walmart(limit: int = 200) -> list[dict]:
    """Walmart scraping is not supported (JS-rendered, requires a browser engine)."""
    print("\n[INFO] Walmart scraping skipped — product grid requires JavaScript rendering.")
    print("       Use --source shopify to collect real products instead.")
    return []


# =============================================================================
# Source 2 — Shopify /products.json  (public endpoint, no auth needed)
# =============================================================================
# Every Shopify store exposes a public JSON endpoint at /products.json that
# returns structured product data: title, vendor, product_type, variants
# (with price, color, size options), and image URLs.
#
# Pagination: add ?page=N&limit=250 (max 250 per page).
#
# We target a few fashion-focused Shopify stores that carry clothing relevant
# to college students / young adults. All are publicly accessible.

SHOPIFY_STORES = [
    # All stores below confirmed returning {"products":[...]} JSON — tested live.

    # ── Fashion ───────────────────────────────────────────────────────────
    # Princess Polly — women's fashion, huge college-age audience
    {"base_url": "https://us.princesspolly.com",  "brand": "Princess Polly",  "gender_hint": "Female"},
    # Cuts Clothing — clean elevated basics, men's and women's
    {"base_url": "https://cutsclothing.com",       "brand": "Cuts Clothing",   "gender_hint": "Unisex"},
    # Represent — premium streetwear / casual, men's focus
    {"base_url": "https://representclo.com",       "brand": "Represent",       "gender_hint": "Male"},
    # Allbirds — sustainable casual footwear + apparel, unisex
    {"base_url": "https://www.allbirds.com",       "brand": "Allbirds",        "gender_hint": "Unisex"},
    # Frank And Oak — minimalist everyday wear, unisex
    {"base_url": "https://www.frankandoak.com",    "brand": "Frank And Oak",   "gender_hint": "Unisex"},
    # MNML — affordable streetwear, men's
    {"base_url": "https://www.mnml.la",            "brand": "MNML",            "gender_hint": "Male"},
    # I AM GIA — edgy women's fashion, popular with young adults
    {"base_url": "https://www.iamgia.com",         "brand": "I AM GIA",        "gender_hint": "Female"},

    # ── Women's Activewear ────────────────────────────────────────────────
    # Alo Yoga — premium yoga + gym wear, women's focus
    {"base_url": "https://aloyoga.com",            "brand": "Alo Yoga",        "gender_hint": "Female"},
    # Buff Bunny — women's gym / bodybuilding apparel
    {"base_url": "https://www.buffbunny.com",      "brand": "Buff Bunny",      "gender_hint": "Female"},
    # Girlfriend Collective — sustainable women's activewear
    {"base_url": "https://www.girlfriend.com",     "brand": "Girlfriend Collective", "gender_hint": "Female"},

    # ── Men's Activewear ──────────────────────────────────────────────────
    # NoBull — training shoes + apparel, men's and women's
    {"base_url": "https://www.nobullproject.com",  "brand": "NoBull",          "gender_hint": "Unisex"},
    # Hylete — men's performance training apparel
    {"base_url": "https://www.hylete.com",         "brand": "Hylete",          "gender_hint": "Male"},
    # Born Primitive — men's CrossFit / strength training apparel
    {"base_url": "https://www.bornprimitive.com",  "brand": "Born Primitive",  "gender_hint": "Male"},

    # ── Unisex Activewear ─────────────────────────────────────────────────
    # Ryderwear — gym wear for all, bodybuilding focus
    {"base_url": "https://www.ryderwear.com",      "brand": "Ryderwear",       "gender_hint": "Unisex"},
    # 2XU — compression + performance sportswear, unisex
    {"base_url": "https://www.2xu.com",            "brand": "2XU",             "gender_hint": "Unisex"},
]


def scrape_shopify_store(base_url: str, brand: str, gender_hint: str,
                          max_pages: int = 4) -> list[dict]:
    """
    Scrape a Shopify store's public /products.json endpoint.

    Each product in the JSON has:
      - title:        product name
      - product_type: category (e.g., "Tops", "Dresses")
      - vendor:       brand name
      - variants:     list of {price, option1, option2, option3}
      - options:      list of {name, values} — tells us which option is color/size
      - handle:       URL slug for building the product URL

    We take the first variant's price and color option as representative.
    """
    products = []
    endpoint = f"{base_url}/products.json"

    for page in range(1, max_pages + 1):
        print(f"  Shopify | {brand} | page {page} → {endpoint}")
        resp = safe_get(endpoint, params={"limit": 250, "page": page})
        if resp is None:
            break

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print(f"    [WARN] Could not parse JSON from {base_url}")
            break

        items = data.get("products", [])
        if not items:
            print(f"    No more products on page {page}.")
            break

        for item in items:
            try:
                title        = item.get("title", "").strip()
                product_type = item.get("product_type", "").strip()
                vendor       = item.get("vendor", brand).strip()
                handle       = item.get("handle", "")
                product_url  = f"{base_url}/products/{handle}"

                # Find which option index corresponds to color
                color_option_index = 0  # default: option1
                for i, opt in enumerate(item.get("options", [])):
                    if opt.get("name", "").lower() in ("color", "colour"):
                        color_option_index = i
                        break

                # Use first variant for price + color
                variants = item.get("variants", [])
                if not variants:
                    continue

                first_variant = variants[0]
                price_str = first_variant.get("price", "0")
                try:
                    price = float(price_str)
                except (ValueError, TypeError):
                    price = 0.0

                # option1/option2/option3 correspond to the options list order
                option_keys = ["option1", "option2", "option3"]
                color_raw = first_variant.get(option_keys[color_option_index], "") or ""
                color = extract_color(title, option_value=color_raw)

                # Infer gender from product type or title if hint is Unisex
                gender = gender_hint
                if gender_hint == "Unisex":
                    combined = (title + " " + product_type).lower()
                    # Explicit female signals
                    if any(w in combined for w in [
                        "women", "woman", "female", "girl", "ladies",
                        "dress", "skirt", "blouse", "cami", "bra",
                        "bikini", "swimsuit", "lace", "ruffle", "strapless",
                        "wrap dress", "midi", "maxi", "corset", "bralette",
                        "romper", "jumpsuit", "crop top", "bodysuit",
                    ]):
                        gender = "Female"
                    # Explicit male signals
                    elif any(w in combined for w in [
                        " men", "man ", "male", " boy", "guys",
                        "polo", "button up", "button-up", "chino",
                        "dad hat", "a-frame hat", "rope hat",
                        "5-pocket pant", "lined short", "crossover short",
                        "riviera knit", "alpha vest", "ao ", "tfp ",
                        "pyca pro", "versaknit", "slim-fit pant",
                        "classic-fit short",
                    ]):
                        gender = "Male"

                # Build a description string from title + type for inference helpers
                desc = f"{title} {product_type}"

                category = normalise_category(product_type or title)
                # Skip swimwear entirely
                if category is None:
                    continue

                products.append({
                    "product_id":       next_id(),
                    "product_name":     title,
                    "gender":           gender,
                    "category":         category,
                    "pattern":          infer_pattern(desc),
                    "color":            color,
                    "age_group":        infer_age_group(desc),
                    "season":           infer_season(desc),
                    "price":            round(price, 2),
                    "material":         infer_material(desc),
                    "sales_count":      "",
                    "reviews_count":    "",
                    "average_rating":   "",
                    "out_of_stock_times": "",
                    "brand":            vendor,
                    "discount":         "",
                    "last_stock_date":  "",
                    "wish_list_count":  "",
                    "month_of_sale":    "",
                    "year_of_sale":     CURRENT_YEAR,
                    "product_url":      product_url,
                })
            except Exception as e:
                print(f"    [WARN] Skipped item '{item.get('title', '?')}': {e}")
                continue

        print(f"    → {len(products)} products collected so far from {brand}")
        polite_sleep()

    return products


def scrape_shopify(limit: int = 300) -> list[dict]:
    """Scrape all configured Shopify stores."""
    print("\n" + "=" * 60)
    print("Scraping Shopify stores...")
    print("=" * 60)
    all_products = []
    per_store = max(1, limit // len(SHOPIFY_STORES))
    pages_per_store = max(1, (per_store // 250) + 1)

    for store in SHOPIFY_STORES:
        if len(all_products) >= limit:
            break
        products = scrape_shopify_store(
            store["base_url"], store["brand"], store["gender_hint"],
            max_pages=pages_per_store,
        )
        all_products.extend(products[:per_store])
        polite_sleep()

    print(f"\nShopify total: {len(all_products)} products scraped")
    return all_products[:limit]


# =============================================================================
# CSV Output
# =============================================================================

def save_csv(products: list[dict], output_path: str) -> None:
    """
    Write the scraped products to a CSV file matching the existing dataset schema.

    If the file already exists, new rows are appended (not overwritten) so you
    can run the scraper multiple times without losing previous data.
    """
    import os
    file_exists = os.path.exists(output_path)
    mode = "a" if file_exists else "w"

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(products)

    action = "Appended" if file_exists else "Created"
    print(f"\n{action} {len(products)} rows → {output_path}")


def deduplicate_csv(path: str) -> int:
    """
    Remove duplicate rows from the output CSV (same product_name + brand).
    Returns the number of rows after deduplication.
    """
    import os
    if not os.path.exists(path):
        return 0

    seen = set()
    unique_rows = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("product_name", "").lower(), row.get("brand", "").lower())
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"Deduplicated → {len(unique_rows)} unique products in {path}")
    return len(unique_rows)


# =============================================================================
# Summary Report
# =============================================================================

def print_summary(products: list[dict]) -> None:
    """Print a breakdown of scraped products by source, gender, and category."""
    from collections import Counter
    print("\n" + "=" * 60)
    print("Scrape Summary")
    print("=" * 60)
    print(f"Total products: {len(products)}")

    brands = Counter(p["brand"] for p in products)
    print("\nBy brand/source:")
    for brand, count in brands.most_common():
        print(f"  {brand:<25} {count}")

    genders = Counter(p["gender"] for p in products)
    print("\nBy gender:")
    for g, count in genders.most_common():
        print(f"  {g:<15} {count}")

    categories = Counter(p["category"] for p in products)
    print("\nBy category:")
    for cat, count in categories.most_common():
        print(f"  {cat:<15} {count}")

    with_url = sum(1 for p in products if p.get("product_url"))
    print(f"\nProducts with URL: {with_url} / {len(products)}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scrape fashion products from H&M and Shopify stores."
    )
    parser.add_argument(
        "--source", choices=["shopify", "all"], default="all",
        help="Which source(s) to scrape — currently only 'shopify' produces data (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Max total products to collect (default: 500)",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE,
        help=f"Output CSV path (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Fashion Product Scraper — Hybrid Data Collection")
    print("=" * 60)
    print(f"  Source : {args.source}")
    print(f"  Limit  : {args.limit} products")
    print(f"  Output : {args.output}")
    print("=" * 60)

    all_products = []

    if args.source in ("walmart", "all"):
        # Walmart requires JS rendering — skipped automatically
        scrape_walmart(limit=0)

    if args.source in ("shopify", "all"):
        shopify_limit = args.limit // 2 if args.source == "all" else args.limit
        shopify_products = scrape_shopify(limit=shopify_limit)
        all_products.extend(shopify_products)

    if not all_products:
        print("\nNo products scraped. Check your internet connection and try again.")
        return

    print_summary(all_products)
    save_csv(all_products, args.output)
    final_count = deduplicate_csv(args.output)

    print(f"\nDone. {final_count} unique products saved to: {args.output}")
    print("You can now use this file alongside fashion_data_2018_2022.csv in fashion_rag.py")


if __name__ == "__main__":
    main()
