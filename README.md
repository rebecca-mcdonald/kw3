# SERP‑powered Question Map — v5

### What’s new
- **SERP expansion:** People Also Ask, Autocomplete, Related Searches (via SerpAPI).
- **Optional LLM expansion:** supply `OPENAI_API_KEY` to add 25+ natural questions per seed.
- **Site context:** mini-crawler extracts headings to inform prompts/fallbacks.
- **Pure‑Python:** safe to deploy on Streamlit Cloud. `runtime.txt` pins Python 3.12.

### Setup
1. Create environment/Streamlit secret `SERP_API_KEY` for SerpAPI (or paste in the sidebar).
2. (Optional) Set `OPENAI_API_KEY` for LLM expansion.

### Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Notes
- Volume/difficulty columns are labeled **proxy**. For real metrics, wire to your preferred keyword DB (e.g., DataForSEO Keywords Data) and map the returned `search_volume` into bands.
- Provider abstraction is ready; add another `expand_with_serp` branch for a different API if you prefer.
