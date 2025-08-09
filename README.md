# Universal, Site‑Aware Question Map — v4

**Works with any website and any seed keywords** (no external APIs).  
Pure‑Python dependencies only. Mini‑crawler extracts headings and top terms to contextualize question generation.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Cloud)
- Push to GitHub.
- `runtime.txt` pins to Python 3.12 for reliable wheels.
- Create a new app pointing at `app.py`.

## Why this fixes previous issues
- No compiled dependencies (no scikit‑learn / scipy / numpy builds).
- Robust guards to avoid KeyErrors and empty tables.
- Universal question expansion that doesn’t rely on templates tied to a niche.
- Site‑aware topics extracted from page titles/H1/H2/H3/meta to make questions relevant to *any* industry.
