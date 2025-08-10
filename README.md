# v7.8 — Canonical Crawl + Full Features

**What changed**
- Always strips `#fragment` (e.g., `https://lbclightingpro.com#site-main` → `https://lbclightingpro.com/`)
- Toggle to strip `?query` params (default **ON**)
- Treat **www** and **non‑www** as the same (default **ON**)
- Normalize trailing slash so `/page` and `/page/` are treated identically (default **ON**)
- Skip non‑HTML assets (`.pdf, .jpg, .png, .css, .js, .svg, .zip, …`)
- Retains all v7.6 functionality (SERP/LLM, targeted crawl, strong matching, coverage, content guides, diagnostics)

**How to run**
```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep **“Strip ?query parameters”** ON to dedupe URLs; turn it OFF only if your site relies on query strings.
