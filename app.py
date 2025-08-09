# Question-First Keyword App (Streamlit MVP, pure-Python deps)
import base64, json
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from pydantic import BaseModel

APP_TITLE = "Question‑First Keyword Map (MVP, pure‑Python)"
DEFAULT_SEEDS = "title 24 ja8, led wall pack, troffer 2x4, recessed downlight commercial, 0-10v dimming"

class AnalyzeRequest(BaseModel):
    project_url: str
    seeds: List[str]
    geography: str = "US"
    business_value_overrides: Dict[str, float] | None = None

def band_value(band: str, kind: str = "volume") -> float:
    maps = {
        "volume": {"High": 0.9, "Med–High": 0.75, "Med": 0.6, "Low–Med": 0.4, "Low": 0.25},
        "difficulty": {"Low": 0.85, "Low–Med": 0.75, "Med": 0.6, "Med–High": 0.45, "High": 0.3},
    }
    return maps[kind].get(band, 0.6)

def compute_os(volume_band: str, difficulty_band: str, geography: str = "US", sitegap: float = 0.5,
               business_value: float = 0.5, serp_potential: float = 0.5) -> float:
    vol_w = band_value(volume_band, "volume")
    diff_w = band_value(difficulty_band, "difficulty")
    geo_w = 0.8 if geography == "US" else 0.6
    oscore = 0.25*vol_w + 0.15*diff_w + 0.25*sitegap + 0.10*geo_w + 0.20*business_value + 0.05*serp_potential
    return round(float(oscore), 3)

def df_to_csv_download(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    return f'<a download="{filename}" href="data:text/csv;base64,{b64}">Download CSV</a>'

SEED_TEMPLATES: Dict[str, List[str]] = {
    "title 24 ja8": [
        "What is Title 24 lighting (California) and who must comply?",
        "Which LED downlights are JA8 compliant (2024/2025 lists)?",
        "How to pass a Title 24 lighting inspection (checklist)?",
        "Is LED tape lighting eligible for JA8 residential projects?",
    ],
    "led wall pack": [
        "What wattage LED wall pack replaces 250W metal halide?",
        "Cutoff vs full‑cutoff wall packs—light pollution rules",
        "Photocell and 0–10V dimming on wall packs—how to wire?",
        "Dark‑Sky friendly wall packs—what to look for?",
    ],
    "troffer 2x4": [
        "2x2 vs 2x4 LED troffer—when to use each?",
        "Back‑lit vs edge‑lit troffers—pros and cons",
        "What is a lay‑in vs surface‑mount troffer?",
    ],
    "recessed downlight commercial": [
        "Best commercial recessed LED housings for offices (specs)",
        "4\" vs 6\" recessed downlights—when to use each?",
        "Canless vs housing recessed lights for new construction",
        "Flanged vs flangeless trims—what’s the difference?",
    ],
    "0-10v dimming": [
        "0–10V vs TRIAC dimming—what’s the difference?",
        "Do I need an ELV dimmer for LED fixtures?",
        "How to troubleshoot flicker with 0–10V drivers",
    ],
}

INTENT_RULES = {
    "how": "Informational",
    "what": "Informational",
    "which": "Informational",
    "best": "Commercial",
    "vs": "Commercial",
    "price": "Commercial",
    "buy": "Transactional",
    "shipping": "Transactional",
}

PAGE_TYPE_RULES = {
    "title 24": "Guide + Product hub + FAQ",
    "wall pack": "Buying guide + How‑to",
    "troffer": "Guide + Calculator + Product hub",
    "recessed": "Category pillar + Comparison",
    "dimming": "Troubleshooting guide + FAQ",
}

def serp_expand(seed: str) -> List[str]:
    base = SEED_TEMPLATES.get(seed.lower().strip(), [])
    variants = [q.replace("?", "").lower() for q in base]
    extra = [f"why {variants[0]}"] if variants else []
    if len(variants) > 1:
        extra += [f"does {variants[1]}"]
    return base + [e.capitalize() + "?" for e in extra]

def estimate_bands(question: str) -> tuple[str, str]:
    q = question.lower()
    if any(k in q for k in ["best", " vs", "replace", "2x4", "recessed"]):
        return "High", "Med–High"
    if any(k in q for k in ["title 24", "ja8", "0–10v", "0-10v", "photocell", "dark‑sky", "dark-sky"]):
        return "Med", "Med"
    return "Low–Med", "Low–Med"

def rules_page_type(question: str) -> str:
    for k, v in PAGE_TYPE_RULES.items():
        if k in question.lower():
            return v
    return "Guide"

def rules_intent(question: str) -> str:
    q = question.lower()
    if " buy" in q or "shipping" in q or "lead time" in q:
        return "Transactional"
    if " vs" in q or "best" in q:
        return "Commercial"
    for stem, label in INTENT_RULES.items():
        if q.startswith(stem + " "):
            return label
    return "Informational"

def simple_rule_clusters(df: pd.DataFrame) -> pd.DataFrame:
    rules = [
        ("title-24-ja8", ["title 24", "ja8"]),
        ("wall-packs", ["wall pack"]),
        ("troffers", ["troffer"]),
        ("recessed", ["recessed", "trim"]),
        ("dimming-controls", ["0–10v", "0-10v", "dimming", "triac", "elv"]),
        ("led-tape", ["tape"]),
        ("track", ["track"]),
        ("emergency-exit", ["emergency", "exit"]),
    ]
    labels = []
    for q in df["Question"].str.lower().tolist():
        label = "misc"
        for name, keys in rules:
            if any(k in q for k in keys):
                label = name
                break
        labels.append(label)
    df["Cluster Label"] = labels
    return df

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown("Design an SEO/LLM‑SEO question map from a site + seed keywords, score them by opportunity, and export briefs/FAQs.")

with st.sidebar:
    st.header("Inputs")
    url = st.text_input("Client URL", placeholder="https://example.com")
    seeds_text = st.text_area("Seed keywords (comma or newline)", value=DEFAULT_SEEDS, height=110)
    geography = st.selectbox("Target geography", ["US", "CA", "Global"], index=0)
    use_serp = st.toggle("Use SERP expansion (demo stub)", value=True)
    use_volumes = st.toggle("Attach volume/difficulty (demo bands)", value=True)
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    seeds = [s.strip() for s in seeds_text.replace("\n", ",").split(",") if s.strip()]
    req = AnalyzeRequest(project_url=url or "https://example.com", seeds=seeds, geography=geography)

    rows = []
    for seed in req.seeds:
        qs = serp_expand(seed) if use_serp else SEED_TEMPLATES.get(seed.lower(), [])
        for q in qs:
            vol_band, diff_band = estimate_bands(q) if use_volumes else ("Med", "Med")
            page_type = rules_page_type(q)
            intent = rules_intent(q)
            oscore = compute_os(vol_band, diff_band, geography=req.geography)
            rows.append({
                "Seed": seed,
                "Question": q,
                "Intent": intent,
                "Volume Band": vol_band,
                "Difficulty Band": diff_band,
                "Recommended Page Type": page_type,
                "On‑Site Coverage": "TBD",
                "Opportunity Score": oscore,
            })

    df = pd.DataFrame(rows)
    df = simple_rule_clusters(df)

    st.success("Analysis complete")

    st.subheader("Questions Table")
    st.dataframe(df.sort_values(["Opportunity Score"], ascending=False), use_container_width=True)

    st.subheader("Content Briefs (per cluster)")
    def build_brief(cluster_label: str, questions: List[str], page_type: str, url_slug: str) -> str:
        h2 = [
            "What it is / why it matters",
            "Specs to compare (lumens, CCT, CRI, efficacy)",
            "How to choose (use‑case driven)",
            "Compliance & standards",
            "Wiring/controls or installation notes",
            "Recommended products",
            "FAQ",
        ]
        md = [
            f"# Content Brief: {cluster_label}",
            f"**Page Type:** {page_type}",
            f"**Proposed URL:** {url_slug}",
            "\n**Primary Questions**",
        ]
        md += [f"- {q}" for q in questions[:5]]
        md += ["\n## Outline (H2/H3)"] + [f"- {x}" for x in h2]
        md += ["\n## Internal Links\n- Relevant category pages\n- Controls/dimmers\n- Volume quote / Contact\n- Shipping & pickup policy"]
        md += ["\n## CTA\n- Get a volume quote\n- Request submittals/spec sheets"]
        return "\n".join(md)

    def build_faq_jsonld(questions: List[str]) -> Dict[str, Any]:
        entities = [{
            "@type": "Question",
            "name": q,
            "acceptedAnswer": {"@type": "Answer", "text": "Short, helpful answer (90–160 words)."}
        } for q in questions[:5]]
        return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": entities}

    briefs = []
    for label in sorted(df["Cluster Label"].unique()):
        subset = df[df["Cluster Label"] == label].sort_values("Opportunity Score", ascending=False)
        top_qs = subset["Question"].tolist()
        page_type = subset["Recommended Page Type"].mode().iloc[0]
        slug_token = label.replace(" ", "-")
        brief_md = build_brief(label, top_qs, page_type, f"/resources/{slug_token}-guide")
        briefs.append((label, brief_md, top_qs[:5]))

    for label, md, top_qs in briefs:
        with st.expander(f"Brief: {label}"):
            st.markdown(md)
            faq_json = build_faq_jsonld(top_qs)
            st.code(json.dumps(faq_json, indent=2), language="json")

    st.subheader("Export")
    st.markdown(df_to_csv_download(df, "question_map.csv"), unsafe_allow_html=True)
    all_md = "\n\n".join([b[1] for b in briefs])
    st.download_button("Download Briefs (Markdown)", data=all_md.encode("utf-8"), file_name="briefs.md", mime="text/markdown")

    st.subheader("Summary")
    st.write({
        "Total questions": int(len(df)),
        "Clusters": int(df["Cluster Label"].nunique()),
        "Avg opportunity": float(df["Opportunity Score"].mean()) if not df.empty else 0.0,
    })

else:
    st.info("Enter a URL and seeds, then click Run Analysis. Use the sidebar toggles to simulate SERP expansion and volume bands.")
