# Question-First Keyword App (Streamlit MVP, v3 robust + universal)
import base64, json, re
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from pydantic import BaseModel

APP_TITLE = "Question‑First Keyword Map (Universal MVP)"
DEFAULT_SEEDS = "your main service, pricing, installation, warranty, near me"

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

# --- Universal expansion (no external API) ---
QUESTION_TEMPLATES = [
    "What is {kw} and how does it work?",
    "How to choose the right {kw}?",
    "What does {kw} cost?",
    "Is {kw} worth it for {industry}?",
    "Best {kw} options for {use_case}",
    "{kw} vs alternatives—what’s the difference?",
    "Common problems with {kw} and how to fix them",
    "What size/spec do I need for {kw}?",
    "Is {kw} available near me?",
    "How long does {kw} take to install?",
]

DEFAULT_USE_CASES = ["small business", "enterprise", "home", "commercial", "ecommerce"]
DEFAULT_INDUSTRIES = ["retail", "healthcare", "finance", "construction", "education"]

def normalize_kw(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s

def universal_expand(seed: str, geography: str = "US", max_q: int = 10) -> List[str]:
    seed = normalize_kw(seed)
    if not seed:
        return []
    use_cases = DEFAULT_USE_CASES[:2]
    industries = DEFAULT_INDUSTRIES[:2]
    qs = []
    for t in QUESTION_TEMPLATES:
        q = t.format(kw=seed, use_case=use_cases[0], industry=industries[0])
        qs.append(q)
    # Add a couple variations with the second use case/industry
    qs.append(QUESTION_TEMPLATES[1].format(kw=seed, use_case=use_cases[1], industry=industries[1]))
    qs.append(QUESTION_TEMPLATES[4].format(kw=seed, use_case=use_cases[1], industry=industries[1]))
    # De-dup and cap
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            out.append(q)
            seen.add(q)
        if len(out) >= max_q:
            break
    return out

def estimate_bands(question: str) -> tuple[str, str]:
    q = question.lower()
    if any(k in q for k in ["best", " vs", "price", "cost", "near me"]):
        return "High", "Med–High"
    if any(k in q for k in ["install", "how to", "size", "spec", "problems", "fix"]):
        return "Med", "Med"
    return "Low–Med", "Low–Med"

def rules_page_type(question: str) -> str:
    q = question.lower()
    if " vs" in q or "best" in q or "cost" in q or "price" in q:
        return "Comparison / Buying guide"
    if "how to" in q or "install" in q or "fix" in q:
        return "How‑to / Troubleshooting"
    if "near me" in q:
        return "Location / Service page"
    return "Guide"

def rules_intent(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ["buy", "price", "cost", "near me"]):
        return "Commercial/Transactional"
    if " vs" in q or "best" in q:
        return "Commercial"
    if "how to" in q or "install" in q or "fix" in q:
        return "Informational"
    return "Informational"

def simple_rule_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Question" not in df.columns or df.empty:
        if "Question" not in df.columns:
            df["Question"] = pd.Series(dtype=str)
        df["Cluster Label"] = pd.Series(dtype=str)
        return df
    labels = []
    for q in df["Question"].astype(str).str.lower().tolist():
        if " vs" in q:
            labels.append("comparisons")
        elif "cost" in q or "price" in q or "best" in q:
            labels.append("buying-guide")
        elif "how to" in q or "install" in q or "fix" in q:
            labels.append("how-to")
        elif "near me" in q:
            labels.append("local")
        else:
            labels.append("general")
    df["Cluster Label"] = labels
    return df

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown("Turn any site + any seed keywords into a question-first keyword map. Universal expansion (no external APIs) + safe guards.")

with st.sidebar:
    st.header("Inputs")
    url = st.text_input("Client URL", placeholder="https://example.com")
    seeds_text = st.text_area("Seed keywords (comma or newline)", value=DEFAULT_SEEDS, height=110)
    geography = st.selectbox("Target geography", ["US", "CA", "Global"], index=0)
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    seeds = [s.strip() for s in seeds_text.replace("\n", ",").split(",") if s.strip()]
    if not seeds:
        seeds = [s.strip() for s in DEFAULT_SEEDS.split(",")]
    req = AnalyzeRequest(project_url=url or "https://example.com", seeds=seeds, geography=geography)

    rows = []
    for seed in req.seeds:
        qs = universal_expand(seed, geography=req.geography, max_q=12)
        for q in qs:
            vol_band, diff_band = estimate_bands(q)
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

    cols = ["Seed","Question","Intent","Volume Band","Difficulty Band","Recommended Page Type","On‑Site Coverage","Opportunity Score"]
    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        st.warning("No questions were generated. Try different seeds.")
        st.stop()

    df = simple_rule_clusters(df)

    st.success("Analysis complete")
    st.subheader("Questions Table")
    st.dataframe(df.sort_values(["Opportunity Score"], ascending=False), use_container_width=True)

    st.subheader("Content Briefs (per cluster)")
    def build_brief(cluster_label: str, questions: List[str], page_type: str, url_slug: str) -> str:
        h2 = [
            "What it is / why it matters",
            "Specs to compare / decision criteria",
            "How to choose (use‑case)",
            "Compliance / policies / warranties",
            "Troubleshooting or implementation notes",
            "Recommended products/services",
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
        md += ["\n## Internal Links\n- Category pages or core services\n- Pricing / Contact / Quote\n- Shipping / Warranty / Policies"]
        md += ["\n## CTA\n- Get a quote\n- Request a demo / consult"]
        return "\n".join(md)

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
            faq_entities = [{
                "@type": "Question",
                "name": q,
                "acceptedAnswer": {"@type": "Answer", "text": "Short, helpful answer (90–160 words)."}
            } for q in top_qs]
            faq_json = {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": faq_entities}
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
    st.info("Enter any URL and seed keywords (any niche). Click Run Analysis to generate universal question sets, clusters, briefs, and FAQ JSON-LD.")
