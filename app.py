"""
MBTI Fashion Recommender — Streamlit UI
========================================
A personalized closet experience. Users complete two quick profiling steps
(appearance + MBTI personality), then see a scrollable closet grid of
recommended clothing items with product images pulled live from Shopify,
clickable links back to the store, and a styled recommendation from the LLM.

Run:
    streamlit run app.py
"""

import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# How many top candidates to retrieve before randomly sampling for display.
# This gives genuine variety on each refresh while still staying relevant.
CANDIDATE_POOL = 40

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Your Style Closet",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
SCRAPED_CSV       = BASE_DIR / "data" / "scraped_products.csv"
BASE_DATASET_CSV  = BASE_DIR / "data" / "fashion_data_2018_2022.csv"
CHROMA_DB_PATH    = str(BASE_DIR / "data" / "fashion_chroma_db")
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
TOP_K             = 12   # products to show in the closet grid

# ── External links ────────────────────────────────────────────────────────────
MBTI_TEST_URL     = "https://www.16personalities.com"
APPEARANCE_FORM   = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLScGeEKu5EJALrkkHmJpnzfyBxpd9ezzBgxVnzu9FPl9155wHw/viewform"
)

# Shopify base URLs for image fetching
SHOPIFY_BASES = {
    "princess polly":  "https://us.princesspolly.com",
    "cuts clothing":   "https://cutsclothing.com",
    "represent":       "https://representclo.com",
    "allbirds":        "https://www.allbirds.com",
    "frank and oak":   "https://www.frankandoak.com",
    "mnml":            "https://www.mnml.la",
    "i am gia":        "https://www.iamgia.com",
}

FALLBACK_IMAGE = "https://via.placeholder.com/400x500/f0e6ff/9b59b6?text=No+Image"


# =============================================================================
# Appearance + MBTI data (imported from fashion_rag.py)
# =============================================================================
# Import the mapping tables directly so we don't duplicate them.
import sys
sys.path.insert(0, str(BASE_DIR))

from fashion_rag import (
    MBTI_FASHION_MAP,
    VALID_MBTI_TYPES,
    SEASON_COLORS,
    SEASON_STYLE_NOTES,
    VALID_EYE_COLORS,
    VALID_HAIR_COLORS,
    VALID_SKIN_TONES,
    VALID_GENDER_OPTIONS,
    GENDER_TO_DATASET,
    derive_season,
    build_combined_query,
    build_mbti_query,
)

# =============================================================================
# Custom CSS — closet aesthetic
# =============================================================================
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header ── */
.closet-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.closet-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #2c1a4e;
    margin-bottom: 0.2rem;
}
.closet-header p {
    color: #7c6b8a;
    font-size: 1.1rem;
    font-weight: 300;
}

/* ── Season badge ── */
.season-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0.2rem;
}
.Spring  { background: #fff3e0; color: #e65100; }
.Summer  { background: #e8f5e9; color: #2e7d32; }
.Autumn  { background: #fbe9e7; color: #bf360c; }
.Winter  { background: #e3f2fd; color: #1565c0; }

/* ── Product card ── */
.product-card {
    background: white;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1.2rem;
    height: 100%;
}
.product-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.14);
}
.product-img {
    width: 100%;
    aspect-ratio: 3/4;
    object-fit: cover;
    display: block;
}
.product-info {
    padding: 0.8rem 1rem 1rem;
}
.product-name {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    color: #2c1a4e;
    margin-bottom: 0.3rem;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.product-brand {
    font-size: 0.75rem;
    color: #9e8aad;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.product-price {
    font-size: 1rem;
    font-weight: 500;
    color: #2c1a4e;
    margin-bottom: 0.6rem;
}
.product-meta {
    font-size: 0.75rem;
    color: #b0a0bc;
    margin-bottom: 0.6rem;
}
.shop-btn {
    display: block;
    text-align: center;
    background: #2c1a4e;
    color: white !important;
    padding: 0.5rem;
    border-radius: 8px;
    text-decoration: none !important;
    font-size: 0.85rem;
    font-weight: 500;
    transition: background 0.2s;
}
.shop-btn:hover { background: #4a2d7a; }

/* ── Profile summary card ── */
.profile-card {
    background: linear-gradient(135deg, #f3e8ff 0%, #e8f4ff 100%);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.profile-card h3 {
    font-family: 'Playfair Display', serif;
    color: #2c1a4e;
    margin-bottom: 0.8rem;
}

/* ── Section divider ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: #2c1a4e;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e8d5f5;
}

/* ── Recommendation box ── */
.rec-box {
    background: linear-gradient(135deg, #2c1a4e 0%, #4a2d7a 100%);
    color: white;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    font-size: 1rem;
    line-height: 1.7;
}
.rec-box h4 {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    margin-bottom: 0.8rem;
    color: #e8d5f5;
}

/* ── Step indicator ── */
.step-indicator {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin: 1rem 0 2rem;
}
.step {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.9rem;
    color: #9e8aad;
}
.step.active { color: #2c1a4e; font-weight: 600; }
.step-num {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #e8d5f5;
    color: #2c1a4e;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 600;
}
.step.active .step-num { background: #2c1a4e; color: white; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data loading
# =============================================================================

@st.cache_data(show_spinner=False)
def load_products() -> list[dict]:
    """Load scraped products CSV into a list of dicts."""
    products = []
    if not SCRAPED_CSV.exists():
        return products
    with open(SCRAPED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items()}
            if clean.get("product_name") and clean.get("product_url"):
                products.append(clean)
    return products


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


# =============================================================================
# Image fetching — pull from Shopify product JSON
# =============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_shopify_image(product_url: str, brand: str) -> str:
    """
    Fetch the first product image from a Shopify store's product JSON endpoint.

    Shopify product URLs follow the pattern:
        https://store.com/products/product-handle
    The JSON endpoint is:
        https://store.com/products/product-handle.json

    Returns the image src URL, or a fallback placeholder.
    """
    if not product_url:
        return FALLBACK_IMAGE

    # Build the .json URL from the product URL
    json_url = product_url.rstrip("/") + ".json"

    try:
        resp = requests.get(
            json_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
        )
        if resp.status_code == 200:
            data = resp.json()
            images = data.get("product", {}).get("images", [])
            if images:
                src = images[0].get("src", "")
                # Use a smaller size variant for faster loading
                src = re.sub(r'\.(jpg|jpeg|png|webp)(\?.*)?$',
                             r'_400x.\1', src, flags=re.IGNORECASE)
                return src
    except Exception:
        pass

    return FALLBACK_IMAGE


# =============================================================================
# Cosine similarity retrieval (no ChromaDB needed — runs directly on CSV)
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@st.cache_data(show_spinner=False)
def embed_products(_model: SentenceTransformer, product_tuples: tuple) -> np.ndarray:
    """
    Embed products with enriched text so semantic search works for
    activity-based queries like 'gym', 'yoga', 'workout', 'running'.

    Each product gets a descriptive sentence that includes:
    - Product name (the raw title)
    - Category (Activewear, Shirt, Dress, etc.)
    - Brand (Alo Yoga, Born Primitive, etc.)
    - Color and season
    - Activity keywords inferred from the name and category
    """
    # Activity keyword expansion — maps category/brand signals to activity terms
    # so "Airbrush Legging - Alo Yoga" becomes searchable as "yoga gym workout"
    ACTIVITY_EXPANSIONS = {
        "activewear": "gym workout fitness training athletic sports exercise",
        "legging":    "gym yoga workout running fitness leggings",
        "sports bra": "gym workout fitness training sports bra activewear",
        "compression":"running training performance compression workout",
        "alo yoga":   "yoga gym pilates workout mindful movement",
        "buff bunny": "gym bodybuilding fitness workout training",
        "girlfriend": "yoga gym workout sustainable activewear",
        "nobull":     "crossfit training gym workout performance",
        "hylete":     "gym training performance workout athletic",
        "born primitive": "crossfit strength training gym workout",
        "ryderwear":  "gym bodybuilding training workout fitness",
        "2xu":        "running compression performance training triathlon",
        "skort":      "tennis golf activewear athletic sport",
        "pump cover": "gym workout warmup activewear training",
        "biker short":"gym cycling workout activewear",
        "track":      "running athletic training workout",
        "jogger":     "gym casual athletic workout",
        "hoodie":     "casual gym warmup athletic streetwear",
        "sweatshirt": "casual gym warmup athletic streetwear",
    }

    texts = []
    for t in product_tuples:
        name, category, color, brand, season = t[0], t[1], t[2], t[3], t[4]
        name_lower     = name.lower()
        brand_lower    = brand.lower()
        category_lower = category.lower()

        # Collect activity expansions that apply to this product
        activity_terms = []
        for signal, expansion in ACTIVITY_EXPANSIONS.items():
            if signal in name_lower or signal in brand_lower or signal in category_lower:
                activity_terms.append(expansion)

        activity_str = " ".join(activity_terms)

        text = (
            f"{name}. "
            f"Category: {category}. "
            f"Brand: {brand}. "
            f"Color: {color}. "
            f"Season: {season}. "
            f"{activity_str}"
        ).strip()
        texts.append(text)

    return _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


# =============================================================================
# Content blocklist — products excluded from ALL recommendations
# =============================================================================
# Swimwear, bodysuits, intimates, lingerie, and corsets are blocked.
# This keeps recommendations appropriate for a class presentation.

_BLOCKED_SIGNALS = {
    # Swimwear
    "swim", "swimwear", "swimsuit", "bikini", "bathing suit", "one-piece",
    "board short", "rashguard", "wetsuit", "swim top", "swim bottom",
    "swim trunk", "tankini",
    # Bodysuits / intimates / lingerie
    "bodysuit", "body suit", "intimates", "lingerie", "corset",
    "bralette", "thong", "g-string", "teddy", "chemise", "bustier",
    "garter", "lace pack", "sexy", "plunge bodysuit", "plunge shell",
}


def _is_blocked(product: dict) -> bool:
    """Return True if this product should be excluded from recommendations."""
    name     = product.get("product_name", "").lower()
    category = product.get("category", "").lower()
    combined = name + " " + category
    return any(sig in combined for sig in _BLOCKED_SIGNALS)


# Keywords that strongly signal a product is masculine-coded
_MASCULINE_SIGNALS = {
    "polo", "button up", "button-up", "men's", "mens", "boyfriend",
    "dad hat", "a-frame hat", "rope hat", "vest", "5-pocket pant",
    "slim-fit", "classic-fit", "signature-fit", "lined short",
    "crossover short", "riviera knit", "ao ", "tfp ", "pyca pro",
    "versaknit", "alpha vest", "script hat", "x tech", "c tech",
}

# Keywords that strongly signal a product is feminine-coded
_FEMININE_SIGNALS = {
    "women's", "womens", "dress", "skirt", "blouse", "cami", "crop top",
    "bodysuit", "bikini", "swimsuit", "floral", "lace", "ruffle",
    "strapless", "wrap", "midi", "maxi", "mini skirt", "corset",
    "bralette", "lingerie", "romper", "jumpsuit",
}


def _is_masculine_coded(product_name: str) -> bool:
    name = product_name.lower()
    return any(sig in name for sig in _MASCULINE_SIGNALS)


def _is_feminine_coded(product_name: str) -> bool:
    name = product_name.lower()
    return any(sig in name for sig in _FEMININE_SIGNALS)


def _passes_gender_filter(product: dict, target_gender: str) -> bool:
    """Return True if this product should be shown to a user with target_gender."""
    prod_gender  = product.get("gender", "").strip()
    product_name = product.get("product_name", "")

    if target_gender == "Female":
        if prod_gender == "Female":   return True
        if prod_gender == "Male":     return False
        return not _is_masculine_coded(product_name)

    if target_gender == "Male":
        if prod_gender == "Male":     return True
        if prod_gender == "Female":   return False
        return not _is_feminine_coded(product_name)

    return True  # Non-binary / unspecified — include everything


def _is_swimwear(product_name: str) -> bool:
    name = product_name.lower()
    return any(sig in name for sig in _SWIMWEAR_SIGNALS)


def retrieve_top_products(
    query: str,
    products: list[dict],
    product_embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = TOP_K,
    gender: str = "",
    refresh_seed: int = 0,
) -> list[dict]:
    """
    Retrieve clothing recommendations via cosine similarity with:
      - Swimwear always excluded
      - Hard gender filtering (Woman → Female/non-masculine-Unisex,
        Man → Male/non-feminine-Unisex, others → full catalog)
      - Variety on refresh: retrieves CANDIDATE_POOL top matches, then
        randomly samples top_k from them using refresh_seed as the RNG seed.
        Each new seed gives a genuinely different selection of items.
    """
    dataset_genders = GENDER_TO_DATASET.get(gender, [])
    apply_filter = len(dataset_genders) == 1
    target = dataset_genders[0] if apply_filter else ""

    q_emb  = model.encode(query, convert_to_numpy=True)
    scores = [cosine_sim(q_emb, product_embeddings[i]) for i in range(len(products))]

    scored = []
    for score, product in zip(scores, products):
        # Skip blocked content (swimwear, bodysuits, intimates, lingerie)
        if _is_blocked(product):
            continue
        p = dict(product)
        p["similarity"] = round(score, 4)
        scored.append(p)

    # Gender filter
    if apply_filter and target:
        filtered = [p for p in scored if _passes_gender_filter(p, target)]
        pool = filtered if len(filtered) >= 4 else scored
    else:
        pool = scored

    # Sort by similarity to get the best candidates
    pool.sort(key=lambda x: x["similarity"], reverse=True)

    # Take a larger candidate pool, then randomly sample for variety on refresh.
    # We weight the random sample by similarity so highly relevant items still
    # appear more often, but lower-ranked items get a chance to surface.
    candidates = pool[:CANDIDATE_POOL]
    if len(candidates) <= top_k:
        return candidates

    # Use refresh_seed so each refresh click gives a different selection
    rng = random.Random(refresh_seed)
    # Weighted sampling: similarity score as weight
    weights = [p["similarity"] for p in candidates]
    selected = rng.choices(candidates, weights=weights, k=min(top_k * 2, len(candidates)))
    # Deduplicate while preserving order
    seen = set()
    result = []
    for p in selected:
        pid = p.get("product_id", p.get("product_name", ""))
        if pid not in seen:
            seen.add(pid)
            result.append(p)
        if len(result) == top_k:
            break
    # If we didn't get enough after dedup, pad from the top of the pool
    if len(result) < top_k:
        for p in candidates:
            pid = p.get("product_id", p.get("product_name", ""))
            if pid not in seen:
                seen.add(pid)
                result.append(p)
            if len(result) == top_k:
                break

    return result


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    st.markdown("""
    <div class="closet-header">
        <h1>👗 Your Style Closet</h1>
        <p>Personalised clothing recommendations based on your appearance & personality</p>
    </div>
    """, unsafe_allow_html=True)


def render_steps(current: int):
    steps = ["Appearance", "Personality", "Your Closet"]
    html = '<div class="step-indicator">'
    for i, label in enumerate(steps, 1):
        active = "active" if i == current else ""
        html += f"""
        <div class="step {active}">
            <div class="step-num">{i}</div>
            <span>{label}</span>
        </div>"""
        if i < len(steps):
            html += '<span style="color:#e0d0f0;font-size:1.2rem">→</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_product_card(product: dict, image_url: str) -> str:
    """Return HTML for a single product card."""
    name    = product.get("product_name", "Unknown")
    brand   = product.get("brand", "")
    price   = product.get("price", "")
    color   = product.get("color", "")
    cat     = product.get("category", "")
    url     = product.get("product_url", "#")
    sim     = product.get("similarity", 0)

    price_str = f"${float(price):.2f}" if price else ""
    meta_str  = " · ".join(filter(None, [color, cat]))

    return f"""
    <div class="product-card">
        <a href="{url}" target="_blank">
            <img class="product-img" src="{image_url}"
                 alt="{name}"
                 onerror="this.src='{FALLBACK_IMAGE}'"/>
        </a>
        <div class="product-info">
            <div class="product-brand">{brand}</div>
            <div class="product-name">{name}</div>
            {"<div class='product-price'>" + price_str + "</div>" if price_str else ""}
            <div class="product-meta">{meta_str}</div>
            <a class="shop-btn" href="{url}" target="_blank">Shop Now →</a>
        </div>
    </div>
    """


def render_closet_grid(products: list[dict]):
    """Render the full closet grid — 4 columns, images + shop links."""
    if not products:
        st.info("No products found. Run scraper.py first to populate your data.")
        return

    st.markdown('<div class="section-title">✨ Your Personalised Closet</div>',
                unsafe_allow_html=True)

    # Fetch all images in parallel using st.spinner
    with st.spinner("Loading your closet..."):
        image_urls = []
        for p in products:
            img = fetch_shopify_image(p.get("product_url", ""), p.get("brand", ""))
            image_urls.append(img)

    # Render 4-column grid
    cols_per_row = 4
    for row_start in range(0, len(products), cols_per_row):
        row_products = products[row_start:row_start + cols_per_row]
        row_images   = image_urls[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, product, img_url in zip(cols, row_products, row_images):
            with col:
                st.markdown(render_product_card(product, img_url),
                            unsafe_allow_html=True)


def render_profile_summary(appearance: dict, mbti_type: str):
    """Show a styled summary of the user's profile above the closet."""
    season  = appearance.get("season", "")
    colors  = appearance.get("colors", [])
    profile = MBTI_FASHION_MAP.get(mbti_type.upper(), {})
    archetype = profile.get("archetype", mbti_type)

    season_class = season if season in ("Spring","Summer","Autumn","Winter") else "Summer"

    color_swatches = ""
    color_map = {
        "black":"#222","white":"#f5f5f5","grey":"#9e9e9e","gray":"#9e9e9e",
        "navy":"#1a237e","blue":"#1976d2","red":"#c62828","green":"#2e7d32",
        "pink":"#e91e8c","purple":"#7b1fa2","brown":"#5d4037","beige":"#d7ccc8",
        "cream":"#fff8e1","ivory":"#fffff0","camel":"#c19a6b","olive":"#827717",
        "teal":"#00695c","coral":"#ff7043","burgundy":"#880e4f","mustard":"#f9a825",
        "rust":"#bf360c","lavender":"#9575cd","peach":"#ffab91","sage":"#a5d6a7",
        "gold":"#f9a825","silver":"#bdbdbd","mint":"#80cbc4","mauve":"#ce93d8",
        "charcoal":"#455a64","rose":"#e91e63","warm beige":"#d7ccc8",
        "soft pink":"#f48fb1","powder blue":"#b3e5fc","forest green":"#1b5e20",
        "royal blue":"#1565c0","emerald green":"#1b5e20","ruby red":"#b71c1c",
        "icy pink":"#fce4ec","bright white":"#ffffff","warm red":"#d32f2f",
        "terracotta":"#bf360c","chocolate brown":"#4e342e","burnt orange":"#e65100",
        "golden yellow":"#f9a825","warm green":"#558b2f","dusty purple":"#7b1fa2",
        "soft white":"#fafafa","light grey":"#eeeeee",
    }
    for c in colors[:6]:
        hex_c = color_map.get(c.lower(), "#c9b8d8")
        color_swatches += (
            f'<span title="{c}" style="display:inline-block;width:20px;height:20px;'
            f'border-radius:50%;background:{hex_c};margin:2px;'
            f'border:1px solid rgba(0,0,0,0.1)"></span>'
        )

    st.markdown(f"""
    <div class="profile-card">
        <h3>Your Style Profile</h3>
        <div style="display:flex;gap:2rem;flex-wrap:wrap">
            <div>
                <div style="font-size:0.75rem;color:#9e8aad;text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:0.3rem">Gender</div>
                <span style="font-weight:600;color:#2c1a4e">{appearance.get("gender","")}</span>
            </div>
            <div>
                <div style="font-size:0.75rem;color:#9e8aad;text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:0.3rem">Seasonal Palette</div>
                <span class="season-badge {season_class}">{season}</span>
                <div style="margin-top:0.5rem">{color_swatches}</div>
            </div>
            <div>
                <div style="font-size:0.75rem;color:#9e8aad;text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:0.3rem">Personality</div>
                <span style="font-weight:600;color:#2c1a4e">{mbti_type.upper()}</span>
                <span style="color:#7c6b8a"> — {archetype}</span>
            </div>
            <div style="flex:1;min-width:200px">
                <div style="font-size:0.75rem;color:#9e8aad;text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:0.3rem">Style Note</div>
                <span style="color:#4a3560;font-size:0.9rem">
                    {appearance.get("style_note","")}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Step 1 — Appearance Profile
# =============================================================================

def step_appearance():
    render_steps(1)
    st.markdown("### Step 1 — Tell us about yourself")
    st.markdown(
        f"These questions match the [Google Form]({APPEARANCE_FORM}) "
        "and help us find colors and styles that complement you."
    )

    # ── Gender ────────────────────────────────────────────────────────────
    st.markdown("#### 🧑 Gender Identity")
    st.caption("This helps us personalise which clothing categories we recommend. All options are welcome.")

    gender_cols = st.columns(4)
    # First 4 options in a row
    selected_gender = st.session_state.get("_gender_pick", VALID_GENDER_OPTIONS[0])

    for i, opt in enumerate(VALID_GENDER_OPTIONS[:-1]):  # all except "Other"
        col = gender_cols[i % 4]
        with col:
            is_selected = selected_gender == opt
            btn_style = (
                "background:#2c1a4e;color:white;border-radius:8px;"
                "padding:0.4rem 0.8rem;border:none;width:100%;cursor:pointer;"
                "font-size:0.9rem;margin-bottom:0.4rem"
                if is_selected else
                "background:#f3e8ff;color:#2c1a4e;border-radius:8px;"
                "padding:0.4rem 0.8rem;border:1px solid #d8b4fe;width:100%;cursor:pointer;"
                "font-size:0.9rem;margin-bottom:0.4rem"
            )
            if st.button(opt, key=f"gender_{i}", use_container_width=True):
                st.session_state["_gender_pick"] = opt
                st.rerun()

    # Self-describe option
    with st.expander("✏️ Self-describe or prefer not to say"):
        custom = st.text_input(
            "Enter your gender identity",
            value="" if selected_gender in VALID_GENDER_OPTIONS else selected_gender,
            placeholder="e.g. Two-spirit, Demi-girl, Prefer not to say...",
            key="gender_custom",
        )
        if custom.strip():
            if st.button("Use this", key="gender_custom_btn"):
                st.session_state["_gender_pick"] = custom.strip()
                st.rerun()

    gender = st.session_state.get("_gender_pick", VALID_GENDER_OPTIONS[0])

    # Show what categories this unlocks
    dataset_genders = GENDER_TO_DATASET.get(gender, ["Female", "Male", "Unisex"])
    if len(dataset_genders) == 1:
        scope_note = f"We'll show you **{dataset_genders[0].lower()}** clothing."
    else:
        scope_note = "We'll show you clothing from **all categories** — women's, men's, and unisex."
    st.caption(f"✓ {scope_note}")

    st.markdown("---")

    # ── Appearance questions ───────────────────────────────────────────────
    st.markdown("#### 🎨 Your Features")
    st.caption("Used for seasonal color analysis — finding shades that complement your natural coloring.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👁️ Eye Color**")
        eye = st.selectbox(
            "Eye color",
            options=[e.capitalize() for e in VALID_EYE_COLORS],
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("**💇 Hair Color**")
        hair = st.selectbox(
            "Hair color",
            options=[h.capitalize() for h in VALID_HAIR_COLORS],
            label_visibility="collapsed",
        )

    with col3:
        st.markdown("**🎨 Skin Tone**")
        skin_display = [s.title() for s in VALID_SKIN_TONES]
        skin_idx = st.selectbox(
            "Skin tone",
            options=range(len(skin_display)),
            format_func=lambda i: skin_display[i],
            label_visibility="collapsed",
        )
        skin = VALID_SKIN_TONES[skin_idx]

    # Live palette preview
    season = derive_season(eye.lower(), hair.lower(), skin)
    colors = SEASON_COLORS[season]
    style_note = SEASON_STYLE_NOTES[season]

    season_colors_map = {
        "Spring": "#fff3e0", "Summer": "#e8f5e9",
        "Autumn": "#fbe9e7", "Winter": "#e3f2fd",
    }
    bg = season_colors_map.get(season, "#f3e8ff")

    st.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:1rem 1.5rem;margin:1rem 0">
        <strong>Your Seasonal Palette: {season}</strong><br>
        <span style="font-size:0.9rem;color:#555">
            Best colors: {', '.join(colors[:6])}<br>
            {style_note}
        </span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Continue to Personality →", type="primary", use_container_width=True):
        st.session_state.appearance = {
            "eye_color":  eye.lower(),
            "hair_color": hair.lower(),
            "skin_tone":  skin,
            "gender":     gender,
            "season":     season,
            "colors":     colors,
            "style_note": style_note,
        }
        st.session_state.step = 2
        st.rerun()


# =============================================================================
# Step 2 — MBTI Personality
# =============================================================================

def step_personality():
    render_steps(2)
    st.markdown("### Step 2 — What's your personality type?")

    # ── "Don't know" banner — prominent, not buried ───────────────────────
    with st.container():
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.info(
                "🔍 **Don't know your type?** Take the free 5-minute assessment, "
                "then come back here and select your result below. "
                "The app will stay on this page waiting for you."
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            st.link_button(
                "Take the MBTI Test →",
                url=MBTI_TEST_URL,
                use_container_width=True,
                type="primary",
            )

    st.markdown("---")
    st.markdown("#### Select your type below:")
    st.caption("Hover over any type to see a description of that personality's style.")

    # Group MBTI types by temperament for easier selection
    groups = {
        "🔬 Analysts":   ["INTJ","INTP","ENTJ","ENTP"],
        "💚 Diplomats":  ["INFJ","INFP","ENFJ","ENFP"],
        "🛡️ Sentinels":  ["ISTJ","ISFJ","ESTJ","ESFJ"],
        "🎯 Explorers":  ["ISTP","ISFP","ESTP","ESFP"],
    }

    selected_mbti = None
    for group_name, types in groups.items():
        st.markdown(f"**{group_name}**")
        cols = st.columns(4)
        for col, mbti in zip(cols, types):
            profile = MBTI_FASHION_MAP[mbti]
            with col:
                if st.button(
                    f"**{mbti}**  \n{profile['archetype']}",
                    key=f"mbti_{mbti}",
                    use_container_width=True,
                    help=f"{profile['description']}\n\nColors: {', '.join(profile['colors'][:3])}",
                ):
                    selected_mbti = mbti

    if selected_mbti:
        st.session_state.mbti = selected_mbti
        st.session_state.step = 3
        st.rerun()

    # Reminder at the bottom for people returning from the test
    st.markdown("---")
    st.markdown(
        "📋 **Just finished the test?** Your 4-letter result is shown at the top of "
        "the 16Personalities results page — find it above and click it here."
    )

    col_back, _ = st.columns([1, 3])
    with col_back:
        if st.button("← Back to Appearance"):
            st.session_state.step = 1
            st.rerun()


# =============================================================================
# Step 3 — The Closet
# =============================================================================

def step_closet(products: list[dict], product_embeddings: np.ndarray,
                model: SentenceTransformer):
    render_steps(3)

    appearance = st.session_state.appearance
    mbti_type  = st.session_state.mbti

    render_profile_summary(appearance, mbti_type)

    # ── Refinement controls — always visible, not buried in expander ──────
    with st.container():
        col_extra, col_refresh, col_back = st.columns([3, 1, 1])
        with col_extra:
            extra = st.text_input(
                "✏️ Refine your recommendations",
                placeholder="e.g. casual summer, going out, work outfit...",
                key="extra_pref",
                label_visibility="visible",
            )
        with col_refresh:
            st.markdown("<br>", unsafe_allow_html=True)
            refresh = st.button(
                "✨ Refresh",
                type="primary",
                use_container_width=True,
                key="refresh_btn",
            )
        with col_back:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Change Profile", use_container_width=True, key="change_profile_btn"):
                st.session_state.step = 1
                st.rerun()

    # ── Build query and retrieve products ─────────────────────────────────
    # Re-run retrieval when: first load, extra text changed, or refresh clicked
    extra_val = st.session_state.get("extra_pref", "")
    query_key = f"{mbti_type}_{appearance['season']}_{extra_val}"

    # Increment seed on each refresh so the random sample changes
    if refresh:
        st.session_state["refresh_seed"] = st.session_state.get("refresh_seed", 0) + 1

    refresh_seed = st.session_state.get("refresh_seed", 0)

    if (
        "last_query" not in st.session_state
        or st.session_state.last_query != query_key
        or refresh
    ):
        query = build_combined_query(mbti_type, appearance, extra_input=extra_val)
        recommended = retrieve_top_products(
            query, products, product_embeddings, model,
            top_k=TOP_K,
            gender=appearance.get("gender", ""),
            refresh_seed=refresh_seed,
        )
        st.session_state.recommended = recommended
        st.session_state.last_query  = query_key

    recommended = st.session_state.get("recommended", [])

    # Render the closet grid
    render_closet_grid(recommended)

    # ── Sidebar filters ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🗂️ Filter Your Closet")
        st.caption("Filters apply to your current recommendations.")

        # ── Category ──────────────────────────────────────────────────────
        st.markdown("**👗 Clothing Type**")
        # Use the full product list for filter options, not just current recs
        all_cats = sorted(set(
            p.get("category", "") for p in products
            if p.get("category") and p.get("category") not in ("Other", "")
        ))
        sel_cats = st.multiselect(
            "Category", all_cats,
            placeholder="All types",
            label_visibility="collapsed",
        )

        # ── Brand ──────────────────────────────────────────────────────────
        st.markdown("**🏷️ Brand**")
        all_brands = sorted(set(p.get("brand", "") for p in products if p.get("brand")))
        sel_brands = st.multiselect(
            "Brand", all_brands,
            placeholder="All brands",
            label_visibility="collapsed",
        )

        # ── Season ─────────────────────────────────────────────────────────
        st.markdown("**🌸 Season**")
        all_seasons = sorted(set(p.get("season", "") for p in products if p.get("season")))
        sel_seasons = st.multiselect(
            "Season", all_seasons,
            placeholder="All seasons",
            label_visibility="collapsed",
        )

        # ── Color ──────────────────────────────────────────────────────────
        st.markdown("**🎨 Color**")
        all_colors = sorted(set(p.get("color", "") for p in products
                                if p.get("color") and p.get("color") != "Multicolor"))
        sel_colors = st.multiselect(
            "Color", all_colors,
            placeholder="Any color",
            label_visibility="collapsed",
        )

        # ── Price ──────────────────────────────────────────────────────────
        st.markdown("**💰 Max Price**")
        prices = [float(p.get("price") or 0) for p in products if p.get("price")]
        max_in_data = int(max(prices)) if prices else 500
        max_price = st.slider(
            "Max price", 0, max_in_data, max_in_data,
            format="$%d",
            label_visibility="collapsed",
        )

        st.markdown("---")

        col_apply, col_reset = st.columns(2)
        with col_apply:
            apply = st.button("✓ Apply", type="primary", use_container_width=True)
        with col_reset:
            reset = st.button("↺ Reset", use_container_width=True)

        if apply:
            filtered = list(recommended)  # start from current recs
            if sel_cats:
                filtered = [p for p in filtered if p.get("category") in sel_cats]
            if sel_brands:
                filtered = [p for p in filtered if p.get("brand") in sel_brands]
            if sel_seasons:
                filtered = [p for p in filtered if p.get("season") in sel_seasons]
            if sel_colors:
                filtered = [p for p in filtered if p.get("color") in sel_colors]
            filtered = [
                p for p in filtered
                if not p.get("price") or float(p.get("price") or 0) <= max_price
            ]
            if not filtered:
                st.warning("No items match those filters. Try broadening your selection.")
            else:
                st.session_state.recommended = filtered
                st.rerun()

        if reset:
            st.session_state.pop("last_query", None)
            st.session_state.pop("recommended", None)
            st.rerun()

        st.markdown("---")
        st.markdown(f"**{len(recommended)}** items in your closet")
        st.markdown(
            f"[🧠 Take MBTI Test]({MBTI_TEST_URL})  \n"
            f"[📋 Appearance Form]({APPEARANCE_FORM})"
        )


# =============================================================================
# Main app
# =============================================================================

def main():
    render_header()

    # ── Session state defaults ─────────────────────────────────────────────
    if "step"       not in st.session_state: st.session_state.step       = 1
    if "appearance" not in st.session_state: st.session_state.appearance = {}
    if "mbti"       not in st.session_state: st.session_state.mbti       = ""

    # ── Load data (cached) ─────────────────────────────────────────────────
    products = load_products()
    if not products:
        st.warning(
            "No scraped products found. "
            "Run `python scraper.py --limit 500` from the project folder first, "
            "then refresh this page."
        )
        st.stop()

    model              = load_embedding_model()
    product_embeddings = embed_products(model, tuple(
        (p.get("product_name",""), p.get("category",""),
         p.get("color",""), p.get("brand",""), p.get("season",""))
        for p in products
    ))

    # ── Route to current step ──────────────────────────────────────────────
    step = st.session_state.step

    if step == 1:
        step_appearance()
    elif step == 2:
        step_personality()
    elif step == 3:
        if not st.session_state.appearance or not st.session_state.mbti:
            st.session_state.step = 1
            st.rerun()
        step_closet(products, product_embeddings, model)


if __name__ == "__main__":
    main()
