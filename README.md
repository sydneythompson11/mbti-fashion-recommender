# MBTI Fashion Recommender

A clothing recommendation app that combines **appearance-based color analysis** and **MBTI personality profiling** to suggest outfits that match both how you look and who you are.

## How it works

1. **Appearance Profile** — Answer 3 questions (eye color, hair color, skin tone) based on the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLScGeEKu5EJALrkkHmJpnzfyBxpd9ezzBgxVnzu9FPl9155wHw/viewform). Your features are mapped to a seasonal color palette (Spring / Summer / Autumn / Winter) using color theory.

2. **Personality Profile** — Take the MBTI test at [mindprofile.co/personality](https://mindprofile.co/personality) and enter your 4-letter type. Each type maps to a fashion archetype (style, silhouette, preferred categories).

3. **Recommendations** — Both profiles are merged into a query, embedded with SentenceTransformer, and matched against product chunks via **cosine similarity** using ChromaDB. AWS Bedrock (Claude) generates a personalised recommendation with shop links.

## Setup

```bash
# Install dependencies
pip install -r requirements_fashion.txt

# Scrape real product data from Shopify stores
python scraper.py --limit 500

# Run the recommender
python fashion_rag.py
```

Requires AWS credentials configured for Bedrock access (Claude Haiku).

## Files

| File | Purpose |
|---|---|
| `fashion_rag.py` | Main RAG pipeline — appearance + MBTI profiling, cosine similarity retrieval, LLM recommendations |
| `scraper.py` | Scrapes clothing products from public Shopify store endpoints |
| `requirements_fashion.txt` | Python dependencies |
| `data/scraped_products.csv` | Seed file — populated by running `scraper.py` |

## Data Sources

- **Base dataset**: ZARA fashion data 2018–2022 (place `fashion_data_2018_2022.csv` in `data/`)
- **Scraped products**: Princess Polly, Cuts Clothing, Represent, Allbirds, Frank And Oak, MNML, I AM GIA — all via public Shopify `/products.json` endpoints
