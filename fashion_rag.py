"""
Fashion Data RAG Pipeline — Appearance + MBTI Personality Clothing Recommender
===============================================================================
Retrieval-Augmented Generation pipeline using the ZARA fashion dataset (2018-2022).

User Flow (two-stage profiling):
  Stage 1 — Appearance Profile (Google Form / in-app questions):
    Users answer three questions about their physical appearance:
      • Eye color
      • Hair color
      • Skin tone
    These are mapped to a seasonal color palette (Spring / Summer / Autumn / Winter)
    using color theory, which determines which clothing colors will complement them.
    Form: https://docs.google.com/forms/d/e/1FAIpQLScGeEKu5EJALrkkHmJpnzfyBxpd9ezzBgxVnzu9FPl9155wHw/viewform

  Stage 2 — Personality Profile (MBTI):
    Users take the MBTI assessment at https://mindprofile.co/personality and enter
    their 4-letter type. Each type maps to a fashion archetype (style, silhouette,
    preferred categories).

  Both profiles are merged into a single rich query string that is embedded and
  matched against product chunks via cosine similarity.

Retrieval Method — Cosine Similarity:
  cos_sim(q, d) = (q · d) / (||q|| * ||d||)

  ChromaDB stores embeddings with hnsw:space="cosine". Distances returned are
  cosine distances (0 = identical). Similarity = 1 - distance.

Setup:
  pip install -r requirements_fashion.txt
  Set AWS credentials (AWS_PROFILE or env vars) for Bedrock access.

Usage:
  python fashion_rag.py
"""

import csv
import json
import os
import webbrowser
from collections import defaultdict
from typing import Optional

import boto3
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# Configuration
# =============================================================================

DATA_FILE         = "./data/fashion_data_2018_2022.csv"
SCRAPED_DATA_FILE = "./data/scraped_products.csv"   # produced by scraper.py
CHROMA_DB_PATH    = "./data/fashion_chroma_db"
COLLECTION_NAME = "fashion_products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BEDROCK_REGION = "us-east-1"
LLM_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

TOP_K = 5

# URL for the MBTI personality assessment
MBTI_TEST_URL = "https://mindprofile.co/personality"

# Google Form — appearance profile (eye color, hair color, skin tone)
APPEARANCE_FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLScGeEKu5EJALrkkHmJpnzfyBxpd9ezzBgxVnzu9FPl9155wHw/viewform"
)


# =============================================================================
# Appearance → Color Palette Mapping  (Seasonal Color Analysis)
# =============================================================================
# The Google Form (APPEARANCE_FORM_URL) collects eye color, hair color, and
# skin tone. These three signals are combined to assign a seasonal color palette
# (Spring / Summer / Autumn / Winter), which determines which clothing colors
# will be most flattering for the user.
#
# Seasonal Color Analysis basics:
#   Spring  — warm + light:  peach, coral, warm beige, camel, ivory, warm green
#   Summer  — cool + light:  soft pink, lavender, powder blue, rose, mauve
#   Autumn  — warm + deep:   rust, burnt orange, olive, mustard, chocolate, teal
#   Winter  — cool + deep:   black, white, navy, jewel tones, icy pastels, red
#
# The mapping is heuristic — real color analysis is nuanced — but it gives a
# meaningful, personalized starting point for the recommendation query.

# Eye color → warmth signal
EYE_WARMTH = {
    "brown":  "warm",
    "black":  "neutral",
    "hazel":  "warm",
    "green":  "cool",
    "blue":   "cool",
    "gray":   "cool",
    "grey":   "cool",
    "red":    "warm",
    "other":  "neutral",
}

# Hair color → warmth + depth signals
HAIR_PROFILE = {
    "black":   {"warmth": "cool",    "depth": "deep"},
    "brown":   {"warmth": "warm",    "depth": "deep"},
    "blonde":  {"warmth": "warm",    "depth": "light"},
    "blue":    {"warmth": "cool",    "depth": "deep"},
    "purple":  {"warmth": "cool",    "depth": "deep"},
    "red":     {"warmth": "warm",    "depth": "deep"},
    "green":   {"warmth": "cool",    "depth": "deep"},
    "yellow":  {"warmth": "warm",    "depth": "light"},
    "gray":    {"warmth": "cool",    "depth": "light"},
    "grey":    {"warmth": "cool",    "depth": "light"},
    "white":   {"warmth": "cool",    "depth": "light"},
    "other":   {"warmth": "neutral", "depth": "medium"},
}

# Skin tone → warmth + depth signals
SKIN_PROFILE = {
    "porcelain":  {"warmth": "cool",    "depth": "light"},
    "almond":     {"warmth": "neutral", "depth": "light"},
    "ivory":      {"warmth": "neutral", "depth": "light"},
    "fair":       {"warmth": "cool",    "depth": "light"},
    "light":      {"warmth": "warm",    "depth": "light"},
    "honey":      {"warmth": "warm",    "depth": "light"},
    "medium":     {"warmth": "neutral", "depth": "medium"},
    "beige":      {"warmth": "neutral", "depth": "medium"},
    "olive":      {"warmth": "warm",    "depth": "medium"},
    "dusky":      {"warmth": "warm",    "depth": "medium"},
    "wheatish":   {"warmth": "warm",    "depth": "deep"},
    "tan":        {"warmth": "warm",    "depth": "deep"},
    "dark":       {"warmth": "neutral", "depth": "deep"},
    "deep":       {"warmth": "neutral", "depth": "deep"},
    "ebony":      {"warmth": "cool",    "depth": "deep"},
    "rich black": {"warmth": "cool",    "depth": "deep"},
}

# (warmth, depth) → seasonal palette
SEASON_FROM_SIGNALS = {
    ("warm",    "light"):  "Spring",
    ("cool",    "light"):  "Summer",
    ("warm",    "deep"):   "Autumn",
    ("cool",    "deep"):   "Winter",
    ("neutral", "light"):  "Summer",   # lean cool-light
    ("neutral", "medium"): "Autumn",   # lean warm-medium
    ("neutral", "deep"):   "Winter",   # lean cool-deep
    ("warm",    "medium"): "Autumn",
    ("cool",    "medium"): "Summer",
}

# Seasonal palette → recommended clothing colors (used in the query)
SEASON_COLORS = {
    "Spring":  ["peach", "coral", "warm beige", "camel", "ivory", "warm green",
                "golden yellow", "light orange", "salmon"],
    "Summer":  ["soft pink", "lavender", "powder blue", "rose", "mauve",
                "dusty purple", "soft white", "light grey", "sage"],
    "Autumn":  ["rust", "burnt orange", "olive", "mustard", "chocolate brown",
                "teal", "terracotta", "warm red", "forest green"],
    "Winter":  ["black", "white", "navy", "royal blue", "emerald green",
                "ruby red", "icy pink", "charcoal", "bright white"],
}

SEASON_STYLE_NOTES = {
    "Spring":  "light, fresh, and warm-toned fabrics with soft floral or natural patterns",
    "Summer":  "cool, muted, and soft-toned pieces with delicate or watercolor-inspired patterns",
    "Autumn":  "rich, earthy, and warm-toned textures like wool, suede, and layered knits",
    "Winter":  "bold, high-contrast, and cool-toned pieces with clean lines or jewel-toned accents",
}

# Full list of valid options (mirrors the Google Form exactly)
VALID_EYE_COLORS = ["brown", "blue", "black", "green", "hazel", "gray", "red", "other"]
VALID_HAIR_COLORS = ["black", "brown", "blonde", "blue", "purple", "red",
                     "green", "yellow", "gray", "white", "other"]
VALID_SKIN_TONES = [
    "porcelain", "almond / ivory", "fair", "light / honey",
    "medium / beige", "olive / dusky", "wheatish", "tan",
    "dark / deep", "ebony / rich black",
]

# Gender options — inclusive, mirrors what will be added to the Google Form.
# Used to filter and personalise clothing recommendations.
VALID_GENDER_OPTIONS = [
    "Woman",
    "Man",
    "Non-binary",
    "Genderfluid",
    "Agender",
    "Prefer not to say",
    "Other / Self-describe",
]

# Maps each gender option to the product dataset's gender field values.
# Non-binary / fluid / agender users get recommendations from all categories
# so they aren't artificially limited to one side of the catalog.
GENDER_TO_DATASET = {
    "Woman":               ["Female"],
    "Man":                 ["Male"],
    "Non-binary":          ["Female", "Male", "Unisex"],
    "Genderfluid":         ["Female", "Male", "Unisex"],
    "Agender":             ["Female", "Male", "Unisex"],
    "Prefer not to say":   ["Female", "Male", "Unisex"],
    "Other / Self-describe": ["Female", "Male", "Unisex"],
}

# Normalise skin tone input to the keys used in SKIN_PROFILE
def _normalise_skin(raw: str) -> str:
    raw = raw.lower().strip()
    mapping = {
        "porcelain": "porcelain",
        "almond / ivory": "almond", "almond": "almond", "ivory": "ivory",
        "fair": "fair",
        "light / honey": "light", "light": "light", "honey": "honey",
        "medium / beige": "medium", "medium": "medium", "beige": "beige",
        "olive / dusky": "olive", "olive": "olive", "dusky": "dusky",
        "wheatish": "wheatish",
        "tan": "tan",
        "dark / deep": "dark", "dark": "dark", "deep": "deep",
        "ebony / rich black": "ebony", "ebony": "ebony", "rich black": "rich black",
    }
    return mapping.get(raw, "medium")  # default to medium if unrecognised


def derive_season(eye_color: str, hair_color: str, skin_tone: str) -> str:
    """
    Derive a seasonal color palette from the three appearance inputs.

    Uses a majority-vote approach across the warmth signals from eye, hair,
    and skin, then combines with the depth signal from hair and skin to
    look up the season in SEASON_FROM_SIGNALS.

    Returns one of: "Spring", "Summer", "Autumn", "Winter"
    """
    eye_w  = EYE_WARMTH.get(eye_color.lower().strip(), "neutral")
    hair_p = HAIR_PROFILE.get(hair_color.lower().strip(), {"warmth": "neutral", "depth": "medium"})
    skin_k = _normalise_skin(skin_tone)
    skin_p = SKIN_PROFILE.get(skin_k, {"warmth": "neutral", "depth": "medium"})

    # Majority vote on warmth across all three signals
    warmth_votes = [eye_w, hair_p["warmth"], skin_p["warmth"]]
    warm_count = warmth_votes.count("warm")
    cool_count = warmth_votes.count("cool")
    if warm_count > cool_count:
        warmth = "warm"
    elif cool_count > warm_count:
        warmth = "cool"
    else:
        warmth = "neutral"

    # Depth from hair + skin (skin carries more weight for clothing)
    depth_votes = [hair_p["depth"], skin_p["depth"], skin_p["depth"]]
    light_count = depth_votes.count("light")
    deep_count  = depth_votes.count("deep")
    if light_count > deep_count:
        depth = "light"
    elif deep_count > light_count:
        depth = "deep"
    else:
        depth = "medium"

    return SEASON_FROM_SIGNALS.get((warmth, depth), "Autumn")


def collect_appearance_profile() -> dict:
    """
    Collect the user's appearance profile in-terminal, mirroring the Google Form.

    The Google Form (APPEARANCE_FORM_URL) asks three questions:
      1. Eye color
      2. Hair color
      3. Skin tone

    This function presents the same questions interactively, validates input,
    derives the seasonal color palette, and returns a profile dict.

    Returns:
        {
          "eye_color":  str,
          "hair_color": str,
          "skin_tone":  str,
          "season":     str,   # "Spring" | "Summer" | "Autumn" | "Winter"
          "colors":     list,  # recommended clothing colors for this season
          "style_note": str,   # fabric/texture guidance
        }
    """
    print("\n" + "=" * 60)
    print("  Step 1 of 2 — Appearance Profile")
    print("  (mirrors the Google Form at the link below)")
    print(f"  {APPEARANCE_FORM_URL}")
    print("=" * 60)
    print("Your physical features help us identify which clothing colors")
    print("will complement you most — based on seasonal color analysis.\n")

    # ── Gender ─────────────────────────────────────────────────────────────
    print("Gender (helps personalise clothing categories):")
    for i, opt in enumerate(VALID_GENDER_OPTIONS, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input("\nEnter your gender (name or number): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(VALID_GENDER_OPTIONS):
            gender = VALID_GENDER_OPTIONS[int(raw) - 1]
            break
        matches = [g for g in VALID_GENDER_OPTIONS if g.lower() == raw.lower()]
        if matches:
            gender = matches[0]
            break
        # Allow free-text self-description
        if raw:
            gender = raw
            break
        print(f"  Please enter a number 1–{len(VALID_GENDER_OPTIONS)} or type your gender.")

    # ── Eye Color ──────────────────────────────────────────────────────────
    print("Eye Color options:")
    for i, opt in enumerate(VALID_EYE_COLORS, 1):
        print(f"  {i}. {opt.capitalize()}")
    while True:
        raw = input("\nEnter your eye color (name or number): ").strip().lower()
        if raw.isdigit() and 1 <= int(raw) <= len(VALID_EYE_COLORS):
            eye_color = VALID_EYE_COLORS[int(raw) - 1]
            break
        if raw in VALID_EYE_COLORS:
            eye_color = raw
            break
        print(f"  Please choose from: {', '.join(VALID_EYE_COLORS)}")

    # ── Hair Color ─────────────────────────────────────────────────────────
    print("\nHair Color options:")
    for i, opt in enumerate(VALID_HAIR_COLORS, 1):
        print(f"  {i}. {opt.capitalize()}")
    while True:
        raw = input("\nEnter your hair color (name or number): ").strip().lower()
        if raw.isdigit() and 1 <= int(raw) <= len(VALID_HAIR_COLORS):
            hair_color = VALID_HAIR_COLORS[int(raw) - 1]
            break
        if raw in VALID_HAIR_COLORS:
            hair_color = raw
            break
        print(f"  Please choose from: {', '.join(VALID_HAIR_COLORS)}")

    # ── Skin Tone ──────────────────────────────────────────────────────────
    print("\nSkin Tone options:")
    for i, opt in enumerate(VALID_SKIN_TONES, 1):
        print(f"  {i}. {opt.title()}")
    while True:
        raw = input("\nEnter your skin tone (name or number): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(VALID_SKIN_TONES):
            skin_tone = VALID_SKIN_TONES[int(raw) - 1]
            break
        if raw.lower() in [s.lower() for s in VALID_SKIN_TONES]:
            skin_tone = raw
            break
        print(f"  Please choose a number 1–{len(VALID_SKIN_TONES)} or type the name.")

    # ── Derive Season ──────────────────────────────────────────────────────
    season = derive_season(eye_color, hair_color, skin_tone)
    colors = SEASON_COLORS[season]
    style_note = SEASON_STYLE_NOTES[season]

    print(f"\n{'─'*60}")
    print(f"  Gender               : {gender}")
    print(f"  Your Seasonal Palette: {season.upper()}")
    print(f"  Best colors for you  : {', '.join(colors[:6])}")
    print(f"  Style note           : {style_note}")
    print(f"{'─'*60}\n")

    return {
        "eye_color":  eye_color,
        "hair_color": hair_color,
        "skin_tone":  skin_tone,
        "gender":     gender,
        "season":     season,
        "colors":     colors,
        "style_note": style_note,
    }


# =============================================================================
# MBTI → Fashion Preference Mapping
# =============================================================================
# Each MBTI type is mapped to a fashion archetype describing preferred colors,
# styles, categories, and keywords. These are used to build a rich query string
# that is then embedded and matched via cosine similarity.
#
# Sources: fashion psychology research + MBTI trait descriptions.
# Archetypes are intentionally broad so they work with the ZARA dataset's
# available categories (Dress, Shirt, Jacket, Jeans, Shorts, Skirt, Shoes, Blouse).

MBTI_FASHION_MAP = {
    # ── Analysts ──────────────────────────────────────────────────────────────
    "INTJ": {
        "archetype": "The Architect",
        "description": "Minimalist, structured, and intentional. Prefers clean lines and neutral palettes.",
        "colors": ["black", "navy", "grey", "white"],
        "styles": ["minimalist", "structured", "tailored", "monochrome"],
        "categories": ["Jacket", "Shirt", "Jeans"],
        "keywords": "minimalist structured tailored clean lines neutral colors professional",
    },
    "INTP": {
        "archetype": "The Logician",
        "description": "Casual and functional. Comfort over convention, with occasional quirky details.",
        "colors": ["grey", "blue", "white", "olive"],
        "styles": ["casual", "functional", "relaxed", "understated"],
        "categories": ["Jeans", "Shirt", "Shorts"],
        "keywords": "casual comfortable relaxed functional understated everyday wear",
    },
    "ENTJ": {
        "archetype": "The Commander",
        "description": "Bold, polished, and power-dressing. Commands attention with sharp silhouettes.",
        "colors": ["black", "white", "red", "navy"],
        "styles": ["bold", "polished", "power dressing", "sharp"],
        "categories": ["Jacket", "Dress", "Shirt"],
        "keywords": "bold polished power dressing sharp silhouette professional confident",
    },
    "ENTP": {
        "archetype": "The Debater",
        "description": "Eclectic and trend-forward. Mixes unexpected pieces with confidence.",
        "colors": ["mixed", "bright", "contrasting"],
        "styles": ["eclectic", "trend-forward", "experimental", "mixed"],
        "categories": ["Shirt", "Jacket", "Shorts"],
        "keywords": "eclectic experimental trend-forward mixed patterns bright colors unique",
    },
    # ── Diplomats ─────────────────────────────────────────────────────────────
    "INFJ": {
        "archetype": "The Advocate",
        "description": "Thoughtful and artistic. Drawn to flowing fabrics, earthy tones, and meaningful details.",
        "colors": ["earth tones", "burgundy", "forest green", "cream"],
        "styles": ["artistic", "flowing", "earthy", "meaningful"],
        "categories": ["Dress", "Blouse", "Skirt"],
        "keywords": "flowing artistic earthy tones meaningful details soft fabrics elegant",
    },
    "INFP": {
        "archetype": "The Mediator",
        "description": "Romantic and whimsical. Loves floral patterns, soft colors, and vintage-inspired pieces.",
        "colors": ["pastel", "floral", "soft pink", "lavender"],
        "styles": ["romantic", "whimsical", "vintage", "floral"],
        "categories": ["Dress", "Blouse", "Skirt"],
        "keywords": "romantic whimsical floral patterns pastel colors vintage inspired soft",
    },
    "ENFJ": {
        "archetype": "The Protagonist",
        "description": "Warm and expressive. Chooses vibrant colors and welcoming silhouettes.",
        "colors": ["warm tones", "coral", "yellow", "teal"],
        "styles": ["expressive", "vibrant", "warm", "welcoming"],
        "categories": ["Dress", "Blouse", "Shirt"],
        "keywords": "vibrant expressive warm colors welcoming silhouette cheerful approachable",
    },
    "ENFP": {
        "archetype": "The Campaigner",
        "description": "Playful and colorful. Embraces bold prints, layering, and spontaneous style.",
        "colors": ["bright", "multicolor", "bold prints"],
        "styles": ["playful", "colorful", "layered", "spontaneous"],
        "categories": ["Dress", "Shirt", "Skirt"],
        "keywords": "playful colorful bold prints layering spontaneous fun expressive",
    },
    # ── Sentinels ─────────────────────────────────────────────────────────────
    "ISTJ": {
        "archetype": "The Logistician",
        "description": "Classic and reliable. Invests in timeless basics and well-made staples.",
        "colors": ["navy", "white", "grey", "khaki"],
        "styles": ["classic", "timeless", "reliable", "traditional"],
        "categories": ["Shirt", "Jeans", "Jacket"],
        "keywords": "classic timeless reliable traditional well-made staples neutral wardrobe",
    },
    "ISFJ": {
        "archetype": "The Defender",
        "description": "Soft and nurturing. Prefers comfortable, modest, and subtly feminine pieces.",
        "colors": ["soft blue", "blush", "cream", "sage"],
        "styles": ["soft", "modest", "feminine", "comfortable"],
        "categories": ["Blouse", "Dress", "Skirt"],
        "keywords": "soft modest feminine comfortable nurturing subtle delicate blouse dress",
    },
    "ESTJ": {
        "archetype": "The Executive",
        "description": "Polished and authoritative. Prefers structured, professional attire.",
        "colors": ["navy", "charcoal", "white", "black"],
        "styles": ["structured", "professional", "authoritative", "neat"],
        "categories": ["Jacket", "Shirt", "Dress"],
        "keywords": "structured professional authoritative neat polished business attire",
    },
    "ESFJ": {
        "archetype": "The Consul",
        "description": "Friendly and put-together. Loves coordinated outfits and crowd-pleasing styles.",
        "colors": ["warm neutrals", "pink", "light blue", "white"],
        "styles": ["coordinated", "friendly", "put-together", "crowd-pleasing"],
        "categories": ["Dress", "Blouse", "Skirt"],
        "keywords": "coordinated friendly put-together crowd-pleasing warm colors feminine",
    },
    # ── Explorers ─────────────────────────────────────────────────────────────
    "ISTP": {
        "archetype": "The Virtuoso",
        "description": "Utilitarian and cool. Favors functional pieces with an effortless edge.",
        "colors": ["black", "grey", "olive", "tan"],
        "styles": ["utilitarian", "cool", "functional", "effortless"],
        "categories": ["Jeans", "Jacket", "Shorts"],
        "keywords": "utilitarian functional cool effortless casual streetwear practical",
    },
    "ISFP": {
        "archetype": "The Adventurer",
        "description": "Artistic and sensory. Drawn to textures, unique prints, and self-expression.",
        "colors": ["earthy", "terracotta", "mustard", "rust"],
        "styles": ["artistic", "sensory", "unique", "expressive"],
        "categories": ["Dress", "Blouse", "Skirt"],
        "keywords": "artistic unique prints textures self-expression earthy colors bohemian",
    },
    "ESTP": {
        "archetype": "The Entrepreneur",
        "description": "Bold and trend-chasing. Loves statement pieces and high-energy looks.",
        "colors": ["bold", "black", "red", "white"],
        "styles": ["bold", "trendy", "statement", "high-energy"],
        "categories": ["Jacket", "Shirt", "Jeans"],
        "keywords": "bold trendy statement pieces high-energy streetwear confident dynamic",
    },
    "ESFP": {
        "archetype": "The Entertainer",
        "description": "Fun and glamorous. Embraces sequins, bright colors, and show-stopping outfits.",
        "colors": ["bright", "gold", "hot pink", "electric blue"],
        "styles": ["glamorous", "fun", "show-stopping", "vibrant"],
        "categories": ["Dress", "Skirt", "Blouse"],
        "keywords": "glamorous fun vibrant show-stopping bright colors party wear festive",
    },
}

VALID_MBTI_TYPES = list(MBTI_FASHION_MAP.keys())


# =============================================================================
# Data Loading & Chunking
# =============================================================================

def load_csv(file_path: str) -> list[dict]:
    """Load a fashion CSV into a list of row dicts."""
    rows = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(clean)
    print(f"Loaded {len(rows)} rows from {file_path}")
    return rows


def load_all_data() -> list[dict]:
    """
    Load the base dataset and, if it exists, the scraped products CSV.
    Merges both into a single list so the vector store covers real products
    with clickable URLs alongside the original ZARA dataset.
    """
    rows = []

    if os.path.exists(DATA_FILE):
        rows.extend(load_csv(DATA_FILE))
    else:
        print(f"[WARN] Base data file not found: {DATA_FILE}")

    if os.path.exists(SCRAPED_DATA_FILE):
        scraped = load_csv(SCRAPED_DATA_FILE)
        rows.extend(scraped)
        print(f"  → Merged scraped data: {len(scraped)} real products with URLs")
    else:
        print(f"  [INFO] No scraped data found at {SCRAPED_DATA_FILE}")
        print(f"         Run scraper.py to add real products: python scraper.py")

    print(f"Total rows loaded: {len(rows)}")
    return rows


def chunk_by_product_group(rows: list[dict]) -> list[dict]:
    """
    Chunk the dataset by (category, gender, season) triplets.

    Each chunk is a natural-language product profile summarizing all products
    that share those three attributes. This grouping captures the market segment
    structure of the data and gives the embedding model rich semantic signal.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("category", "Unknown").strip(),
            row.get("gender", "Unknown").strip(),
            row.get("season", "Unknown").strip(),
        )
        groups[key].append(row)

    chunks = []
    for (category, gender, season), products in groups.items():
        prices, materials, colors, patterns = [], set(), set(), set()
        age_groups, product_names, product_ids, ratings, brands = set(), set(), [], [], set()
        product_urls = []   # collect URLs from scraped products

        for p in products:
            pid = p.get("product_id", "")
            if pid:
                product_ids.append(pid.rstrip(".0"))
            name = p.get("product_name", "")
            if name:
                product_names.add(name)
            url = p.get("product_url", "")
            if url:
                product_urls.append((name, url))
            try:
                prices.append(float(p.get("price", 0) or 0))
            except ValueError:
                pass
            for field, target in [("material", materials), ("color", colors),
                                   ("pattern", patterns), ("age_group", age_groups),
                                   ("brand", brands)]:
                val = p.get(field, "")
                if val:
                    target.add(val)
            try:
                r = float(p.get("average_rating", "") or 0)
                if r > 0:
                    ratings.append(r)
            except ValueError:
                pass

        avg_price = round(np.mean(prices), 2) if prices else 0.0
        min_price = round(min(prices), 2) if prices else 0.0
        max_price = round(max(prices), 2) if prices else 0.0
        avg_rating = round(np.mean(ratings), 2) if ratings else None
        count = len(products)

        narrative_parts = [
            f"Product Group: {gender} {category} for {season} season.",
            f"This group contains {count} product(s) from brand(s): {', '.join(sorted(brands))}.",
            f"Product types include: {', '.join(sorted(product_names))}.",
            f"Available colors: {', '.join(sorted(colors))}.",
            f"Patterns: {', '.join(sorted(patterns))}.",
            f"Materials: {', '.join(sorted(materials))}.",
            f"Target age groups: {', '.join(sorted(age_groups))}.",
            f"Price range: ${min_price} to ${max_price} (average ${avg_price}).",
        ]
        if avg_rating:
            narrative_parts.append(f"Average customer rating: {avg_rating} out of 5.")
        narrative_parts.append(
            f"Source product IDs: {', '.join(product_ids[:20])}"
            + (" (and more)" if len(product_ids) > 20 else ".")
        )
        # Include up to 5 real product URLs in the narrative so the LLM can cite them
        if product_urls:
            url_snippets = [f"{name} ({url})" for name, url in product_urls[:5]]
            narrative_parts.append(f"Shop these items: {'; '.join(url_snippets)}.")

        text = " ".join(narrative_parts)
        chunk_id = f"{gender}_{category}_{season}".replace(" ", "_").replace("/", "-")

        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": {
                "category": category,
                "gender": gender,
                "season": season,
                "product_count": count,
                "avg_price": avg_price,
                "min_price": min_price,
                "max_price": max_price,
                "avg_rating": str(avg_rating) if avg_rating else "",
                "colors": ", ".join(sorted(colors)),
                "materials": ", ".join(sorted(materials)),
                "product_names": ", ".join(sorted(product_names)),
                # Store first 3 URLs as a semicolon-separated string for display
                "product_urls": "; ".join(f"{n}|{u}" for n, u in product_urls[:3]),
            },
            "product_ids": product_ids,
        })

    print(f"Created {len(chunks)} chunks from {len(rows)} rows")
    return chunks


# =============================================================================
# Embedding
# =============================================================================

_sentence_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the SentenceTransformer model (downloads on first run)."""
    global _sentence_model
    if _sentence_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _sentence_model = SentenceTransformer(EMBEDDING_MODEL)
    return _sentence_model


def embed_text(text: str) -> list[float]:
    """
    Generate a 384-dimensional embedding for a text string using all-MiniLM-L6-v2.

    Returns:
        A list of 384 floats representing the text in semantic vector space.
    """
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    cos_sim(a, b) = (a · b) / (||a|| * ||b||)

    Returns a value in [-1, 1] where:
      1.0  = identical direction (most similar)
      0.0  = orthogonal (unrelated)
     -1.0  = opposite direction (most dissimilar)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# =============================================================================
# Vector Store (ChromaDB)
# =============================================================================

def build_vector_store(chunks: list[dict]) -> chromadb.Collection:
    """
    Embed all chunks and store them in a persistent ChromaDB collection.

    ChromaDB is configured with hnsw:space="cosine" so all distance metrics
    are cosine distances. Similarity = 1 - distance.

    Skips re-embedding if the collection already exists with the correct count.
    Delete CHROMA_DB_PATH to force a full rebuild.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == len(chunks):
            print(f"Vector store already built ({collection.count()} chunks). Skipping re-embedding.")
            return collection
        else:
            print(f"Collection exists but has {collection.count()} docs (expected {len(chunks)}). Rebuilding.")
            client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance for all similarity queries
    )

    print(f"Embedding {len(chunks)} chunks and storing in ChromaDB...")
    ids, embeddings, documents, metadatas = [], [], [], []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(embed_text(chunk["text"]))
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Stored {collection.count()} chunks in ChromaDB at: {CHROMA_DB_PATH}")
    return collection


# =============================================================================
# MBTI Profile Builder
# =============================================================================

def build_mbti_query(mbti_type: str, extra_input: str = "") -> str:
    """
    Build a rich natural-language query from an MBTI type and optional user input.
    For a query that also incorporates appearance, use build_combined_query() instead.
    """
    mbti_type = mbti_type.upper().strip()
    if mbti_type not in MBTI_FASHION_MAP:
        raise ValueError(f"Unknown MBTI type: {mbti_type}. Valid types: {VALID_MBTI_TYPES}")

    profile = MBTI_FASHION_MAP[mbti_type]

    query_parts = [
        f"I am a {profile['archetype']} personality ({mbti_type}).",
        profile["description"],
        f"I prefer clothing in colors like {', '.join(profile['colors'])}.",
        f"My style is {', '.join(profile['styles'])}.",
        f"I am looking for {', '.join(profile['categories'])}.",
        profile["keywords"],
    ]

    if extra_input.strip():
        query_parts.append(f"Additional preference: {extra_input.strip()}")

    return " ".join(query_parts)


def build_combined_query(
    mbti_type: str,
    appearance: dict,
    extra_input: str = "",
) -> str:
    """
    Build a combined query from both the appearance profile and MBTI type.

    The appearance profile contributes:
      - Seasonal color palette (the colors that complement the user's features)
      - Fabric/texture style note for the season

    The MBTI profile contributes:
      - Personality archetype description
      - Preferred styles and silhouettes
      - Preferred clothing categories

    Both are merged into a single natural-language string that is embedded and
    matched against product chunks via cosine similarity. The richer the query,
    the more accurate the match.

    Args:
        mbti_type:  One of the 16 MBTI type strings (e.g., "INFP")
        appearance: Dict returned by collect_appearance_profile()
        extra_input: Optional free-text from the user

    Returns:
        A natural-language query string ready for embedding
    """
    mbti_type = mbti_type.upper().strip()
    if mbti_type not in MBTI_FASHION_MAP:
        raise ValueError(f"Unknown MBTI type: {mbti_type}. Valid types: {VALID_MBTI_TYPES}")

    mbti = MBTI_FASHION_MAP[mbti_type]
    season     = appearance["season"]
    colors     = appearance["colors"]
    style_note = appearance["style_note"]
    gender     = appearance.get("gender", "")

    # Build a gender-aware style phrase for the query
    gender_phrase = ""
    if gender:
        dataset_genders = GENDER_TO_DATASET.get(gender, ["Female", "Male", "Unisex"])
        if len(dataset_genders) == 1:
            gender_phrase = f"I identify as {gender} and prefer {dataset_genders[0].lower()} clothing."
        else:
            gender_phrase = (
                f"I identify as {gender} and am open to clothing from any gender category "
                "including women's, men's, and unisex styles."
            )

    # Merge color lists: appearance palette takes priority, MBTI colors add variety
    all_colors = colors[:4] + [c for c in mbti["colors"] if c not in colors][:2]

    query_parts = [
        # Gender context
        gender_phrase,
        # Appearance-driven color context
        f"My seasonal color palette is {season}, so I look best in "
        f"{', '.join(colors[:5])}.",
        f"I suit {style_note}.",
        # MBTI-driven style context
        f"I am a {mbti['archetype']} personality ({mbti_type}): {mbti['description']}",
        f"My preferred clothing colors are {', '.join(all_colors)}.",
        f"My style is {', '.join(mbti['styles'])}.",
        f"I am looking for {', '.join(mbti['categories'])}.",
        mbti["keywords"],
    ]

    # Filter out empty strings before joining
    query_parts = [p for p in query_parts if p.strip()]

    if extra_input.strip():
        query_parts.append(f"Additional preference: {extra_input.strip()}")

    return " ".join(query_parts)


def display_mbti_profile(mbti_type: str) -> None:
    """Print a formatted summary of the MBTI fashion profile."""
    mbti_type = mbti_type.upper().strip()
    if mbti_type not in MBTI_FASHION_MAP:
        print(f"Unknown MBTI type: {mbti_type}")
        return

    profile = MBTI_FASHION_MAP[mbti_type]
    print(f"\n{'─'*60}")
    print(f"  MBTI Type : {mbti_type} — {profile['archetype']}")
    print(f"  Style     : {profile['description']}")
    print(f"  Colors    : {', '.join(profile['colors'])}")
    print(f"  Styles    : {', '.join(profile['styles'])}")
    print(f"  Categories: {', '.join(profile['categories'])}")
    print(f"{'─'*60}\n")


# =============================================================================
# Retrieval — Pure Cosine Similarity
# =============================================================================

def retrieve_by_cosine_similarity(
    query: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve the top-K most relevant product chunks using cosine similarity.

    Method:
      1. Embed the query text with SentenceTransformer (all-MiniLM-L6-v2).
      2. Query ChromaDB — which stores embeddings with hnsw:space="cosine" —
         to find the top_k chunks with the smallest cosine distance to the query.
      3. Convert cosine distance → cosine similarity: similarity = 1 - distance.
      4. Return results sorted by similarity (highest first).

    No metadata pre-filtering is applied. Every chunk in the collection is
    ranked purely by its semantic closeness to the query embedding.

    Args:
        query:      Natural-language query (typically built from MBTI profile)
        collection: Populated ChromaDB collection
        top_k:      Number of top results to return

    Returns:
        List of result dicts with keys: id, text, metadata, distance, similarity
    """
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    ids   = results["ids"][0]
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, chunk_id in enumerate(ids):
        cosine_dist = dists[i]
        similarity  = round(1.0 - cosine_dist, 4)   # convert distance → similarity
        formatted.append({
            "id":         chunk_id,
            "text":       docs[i],
            "metadata":   metas[i],
            "distance":   round(cosine_dist, 4),
            "similarity": similarity,
        })

    # Results are already sorted by distance (ascending) = similarity (descending)
    return formatted


# =============================================================================
# LLM (AWS Bedrock — Claude Haiku)
# =============================================================================

def get_bedrock_client(region: str = BEDROCK_REGION):
    """Create a boto3 Bedrock runtime client."""
    session_kwargs = {"region_name": region}
    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        session_kwargs["profile_name"] = profile_name
    session = boto3.Session(**session_kwargs)
    return session.client(service_name="bedrock-runtime", region_name=region)


def ask_llm(query: str, context_chunks: list[dict], mbti_type: str = "") -> str:
    """
    Send the query + retrieved context to Claude on AWS Bedrock and return the answer.

    The system prompt is personalised with the MBTI archetype when available,
    so the LLM frames its answer in terms of the user's personality and style.
    """
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk["metadata"]
        # Format any real product URLs stored in this chunk
        url_lines = ""
        raw_urls = meta.get("product_urls", "")
        if raw_urls:
            url_lines = "\n  Shop links:"
            for entry in raw_urls.split("; "):
                if "|" in entry:
                    name, url = entry.split("|", 1)
                    url_lines += f"\n    • {name.strip()} → {url.strip()}"

        context_text += (
            f"\n[Context {i}] {meta.get('gender', '')} {meta.get('category', '')} "
            f"({meta.get('season', '')} season) — cosine similarity: {chunk['similarity']:.4f}\n"
            f"{chunk['text']}"
            f"{url_lines}\n"
        )

    personality_note = ""
    if mbti_type and mbti_type.upper() in MBTI_FASHION_MAP:
        profile = MBTI_FASHION_MAP[mbti_type.upper()]
        personality_note = (
            f"The user is an {mbti_type.upper()} ({profile['archetype']}): "
            f"{profile['description']} "
            f"Frame your recommendations in terms of their personality and style identity. "
        )

    system_prompt = (
        "You are a personal fashion stylist and retail analyst. "
        f"{personality_note}"
        "Answer questions using ONLY the provided product context. "
        "When citing specific products or statistics, reference the product IDs "
        "from the context (e.g., 'Product #1001'). "
        "If shop links are provided in the context, include them in your answer "
        "so the user can click through to buy the item. "
        "If the context does not contain enough information, say so clearly. "
        "Be concise, warm, and style-forward in your recommendations."
    )

    user_message = (
        f"Product context from the ZARA fashion dataset (2018-2022):\n"
        f"{context_text}\n"
        f"---\n"
        f"Question / Style Request: {query}\n\n"
        f"Provide personalised clothing recommendations based on the context above. "
        f"Cite product IDs where relevant."
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 600,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    })

    try:
        client = get_bedrock_client()
        response = client.invoke_model(
            body=body,
            modelId=LLM_MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
    except Exception as e:
        return (
            f"[LLM Error] Could not reach AWS Bedrock: {e}\n\n"
            f"Top matches by cosine similarity (no LLM answer):\n{context_text}"
        )


# =============================================================================
# Full RAG Pipeline
# =============================================================================

def recommend_for_mbti(
    mbti_type: str,
    collection: chromadb.Collection,
    appearance: Optional[dict] = None,
    extra_input: str = "",
) -> str:
    """
    End-to-end clothing recommendation pipeline combining appearance + MBTI.

    Pipeline:
      1. Look up the MBTI fashion profile and display it.
      2. Build a combined query from appearance palette + MBTI profile.
         (Falls back to MBTI-only query if no appearance profile provided.)
      3. Embed the query and retrieve top-K chunks via cosine similarity.
      4. Send retrieved context + query to the LLM for a personalised answer.
      5. Return the styled recommendation.

    Args:
        mbti_type:   One of the 16 MBTI types (e.g., "INFP")
        collection:  Populated ChromaDB collection
        appearance:  Dict from collect_appearance_profile() — optional
        extra_input: Optional free-text to refine the recommendation
    """
    mbti_type = mbti_type.upper().strip()
    display_mbti_profile(mbti_type)

    if appearance:
        query = build_combined_query(mbti_type, appearance, extra_input)
        print(f"  Seasonal palette : {appearance['season']} "
              f"({', '.join(appearance['colors'][:4])}...)")
    else:
        query = build_mbti_query(mbti_type, extra_input)

    print(f"  Query preview    : {query[:120]}...\n")

    print(f"Retrieving top {TOP_K} matches via cosine similarity...")
    chunks = retrieve_by_cosine_similarity(query, collection, top_k=TOP_K)

    print(f"Top {len(chunks)} results:")
    for c in chunks:
        meta = c["metadata"]
        print(f"  [{c['id']}]  similarity={c['similarity']:.4f}  |  "
              f"{meta.get('gender')} {meta.get('category')} ({meta.get('season')})")

    print("\nGenerating personalised recommendation...")
    return ask_llm(query, chunks, mbti_type=mbti_type)


def answer_query(query: str, collection: chromadb.Collection) -> str:
    """
    General-purpose RAG query (no MBTI context). Uses pure cosine similarity retrieval.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    chunks = retrieve_by_cosine_similarity(query, collection, top_k=TOP_K)
    print(f"Retrieved {len(chunks)} chunks:")
    for c in chunks:
        meta = c["metadata"]
        print(f"  [{c['id']}] similarity={c['similarity']:.4f} | "
              f"{meta.get('gender')} {meta.get('category')} ({meta.get('season')})")

    print("Sending to LLM...")
    return ask_llm(query, chunks)


# =============================================================================
# Demo Queries
# =============================================================================

DEMO_MBTI_TYPES = ["INFP", "ENTJ", "ISTP", "ESFP"]

DEMO_QUERIES = [
    "What women's dresses are available for summer, and what is the price range?",
    "Which products have the highest average customer ratings?",
    "What materials are used in men's jackets for winter?",
]


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """
    Main entry point — two-stage profiling flow.

    Stage 1: Appearance Profile
      Opens the Google Form in the browser (optional) then collects eye color,
      hair color, and skin tone in-terminal. Derives a seasonal color palette.

    Stage 2: MBTI Personality
      Prompts the user to take the MBTI test (opens mindprofile.co) then enter
      their 4-letter type. Merges both profiles into a combined query.

    Retrieval: Pure cosine similarity against ChromaDB product embeddings.
    """
    print("=" * 60)
    print("  Fashion Recommender — Appearance + Personality")
    print("=" * 60)

    # ── Load and index data ────────────────────────────────────────────────
    if not os.path.exists(DATA_FILE) and not os.path.exists(SCRAPED_DATA_FILE):
        print(f"ERROR: No data files found.")
        print(f"  Base dataset : {DATA_FILE}")
        print(f"  Scraped data : {SCRAPED_DATA_FILE}")
        print("Run scraper.py first, or ensure fashion_data_2018_2022.csv is present.")
        return

    rows = load_all_data()
    chunks = chunk_by_product_group(rows)
    collection = build_vector_store(chunks)

    # ── Stage 1: Appearance Profile ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 — Appearance Profile")
    print("=" * 60)
    print("We'll start with a few questions about your appearance.")
    print("This helps us find colors that actually complement your features.\n")
    print(f"You can also fill in the Google Form version here:")
    print(f"  {APPEARANCE_FORM_URL}\n")

    open_form = input("Open the Google Form in your browser first? (y/n): ").strip().lower()
    if open_form == "y":
        webbrowser.open(APPEARANCE_FORM_URL)
        input("Press Enter when you're ready to enter your answers here...")

    appearance = collect_appearance_profile()

    # ── Stage 2: MBTI Personality ──────────────────────────────────────────
    print("=" * 60)
    print("  STEP 2 — Personality Profile (MBTI)")
    print("=" * 60)
    print("Now we'll layer in your personality type for style + silhouette matching.")
    print(f"\nDon't know your MBTI type? Take the free test at:")
    print(f"  {MBTI_TEST_URL}\n")

    open_mbti = input("Open the MBTI test in your browser? (y/n): ").strip().lower()
    if open_mbti == "y":
        webbrowser.open(MBTI_TEST_URL)
        input("Press Enter when you have your result...")

    print(f"\nValid MBTI types: {', '.join(VALID_MBTI_TYPES)}")

    while True:
        mbti = input("\nEnter your MBTI type (or 'skip' to use appearance only): ").strip().upper()

        if mbti in ("SKIP", "S", ""):
            # Appearance-only mode: build query from season/colors alone
            print("\nUsing appearance profile only (no MBTI).")
            season = appearance["season"]
            colors = appearance["colors"]
            style_note = appearance["style_note"]
            query = (
                f"I suit a {season} seasonal color palette. "
                f"I look best in {', '.join(colors[:5])}. "
                f"I prefer {style_note}. "
                f"Please recommend clothing that complements these colors and style."
            )
            print(f"\nRetrieving recommendations via cosine similarity...")
            chunks = retrieve_by_cosine_similarity(query, collection, top_k=TOP_K)
            print(f"Top {len(chunks)} results:")
            for c in chunks:
                meta = c["metadata"]
                print(f"  [{c['id']}]  similarity={c['similarity']:.4f}  |  "
                      f"{meta.get('gender')} {meta.get('category')} ({meta.get('season')})")
            answer = ask_llm(query, chunks)
            print(f"\nRecommendation:\n{answer}\n")
            break

        if mbti not in MBTI_FASHION_MAP:
            print(f"  '{mbti}' is not a valid MBTI type. Try again or type 'skip'.")
            continue

        extra = input("Any extra preferences? (e.g., 'casual summer', press Enter to skip): ").strip()

        print(f"\n{'='*60}")
        answer = recommend_for_mbti(mbti, collection, appearance=appearance, extra_input=extra)
        print(f"\nRecommendation:\n{answer}\n")

        again = input("Try a different MBTI type? (y/n): ").strip().lower()
        if again != "y":
            break

    # ── Optional: free-text follow-up ─────────────────────────────────────
    followup = input("\nWant to ask a follow-up fashion question? (y/n): ").strip().lower()
    if followup == "y":
        print("Type 'quit' to exit.")
        while True:
            user_query = input("\nYour question: ").strip()
            if user_query.lower() in ("quit", "exit", "q"):
                break
            if not user_query:
                continue
            answer = answer_query(user_query, collection)
            print(f"\nAnswer:\n{answer}")

    print("\nThanks for using the Fashion Recommender. Happy styling!")


if __name__ == "__main__":
    main()
