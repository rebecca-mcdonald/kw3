# Question Map (Universal, Site-Aware) — v4
# Pure-Python Streamlit app that works with ANY website + ANY seeds.
# - Mini crawler (same-domain, configurable page cap)
# - Extracts headings and keywords from the site
# - Universal question generation that blends seed + site topics
# - No compiled deps; safe on Streamlit Cloud and Python 3.12

import base64, json, re, time
from typing import List, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
from pydantic import BaseModel

APP_TITLE = "Question‑First Keyword Map (Universal, Site‑Aware v4)"
DEFAULT_SEEDS = "pricing, installation, warranty, near me, comparison, troubleshooting"
DEFAULT_MAX_PAGES = 12
CRAWL_TIMEOUT = 10
UA = "Mozilla/5.0 (compatible; QuestionMapBot/0.1; +https://example.com/bot)"

STOPWORDS = set((
"the a an and or but if then with without into onto from for in on at by as to of over under around near within between above below "
"what which who whom this that these those when where why how is are was were be being been do does did has have had having can could "
"should would will shall may might about across after against all almost also although always among amongs amount anyone anything "
"because before behind being both came come comes during each either enough especially ever every few first following forgo further "
"however into it its itself last later least less many more most much must near nearly never next not nothing now often only other "
"our ours ourselves per rather same several since so some such than that their theirs them themselves there therefore these they "
"through throughout thus toward towards until up upon us use used using very via we were what when where whether which while who "
"whom whose why will with within without you your yours yourself yourselves").split())

class AnalyzeRequest(BaseModel):
    project_url: str
    seeds: List[str]
    geography: str = "US"
    max_pages: int = DEFAULT_MAX_PAGES

def df_to_csv_download(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    return f'<a download="{filename}" href="data:text/csv;base64,{b64}">Download CSV</a>'

# --------------- Site crawling / extraction ---------------

def same_domain(url: str, base: str) -> bool:
    try:
        a = urlparse(url)
        b = urlparse(base)
        return a.netloc.split(":")[0].lower() == b.netloc.split(":")[0].lower() or a.netloc == ""  # allow relative
    except Exception:
        return False

def fetch(url: str) -> str:
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=CRAWL_TIMEOUT)
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
            return resp.text
    except Exception:
        return ""
    return ""

def extract_text_bits(html: str) -> Dict[str, List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    bits = {"title": [], "h1": [], "h2": [], "h3": [], "meta_desc": []}
    t = soup.find("title")
    if t and t.text: bits["title"].append(t.text.strip())
    for tag in soup.find_all(["h1","h2","h3"]):
        if tag.text and tag.text.strip():
            bits[tag.name].append(re.sub(r"\s+", " ", tag.text.strip()))
    m = soup.find("meta", attrs={"name": "description"})
    if m and m.get("content"): bits["meta_desc"].append(m["content"].strip())
    return bits

def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^A-Za-z0-9\s\-]", " ", text)
    toks = [w.lower() for w in re.split(r"\s+", text) if len(w) > 2]
    return [w for w in toks if w not in STOPWORDS and not w.isdigit()]

def top_keywords(pages_bits: List[Dict[str, List[str]]], k: int = 25) -> List[str]:
    freq = {}
    for bits in pages_bits:
        for key in ["title","h1","h2","h3","meta_desc"]:
            for seg in bits.get(key, []):
                for w in tokenize(seg):
                    freq[w] = freq.get(w, 0) + 1
    # keep words that look topical (avoid very generic)
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keep = []
    for w,c in items:
        if len(keep) >= k: break
        if w in ("home","products","services","contact","about","menu","shop"): continue
        keep.append(w)
    return keep

def crawl_site(start_url: str, max_pages: int = DEFAULT_MAX_PAGES) -> Tuple[List[str], List[Dict[str, List[str]]]]:
    seen = set()
    queue = [start_url]
    pages = []
    bits_all = []
    base = start_url
    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen: continue
        seen.add(url)
        html = fetch(url)
        if not html: continue
        pages.append(url)
        bits = extract_text_bits(html)
        bits_all.append(bits)
        # discover a few more links
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            next_url = urljoin(url, href)
            if next_url.startswith("mailto:") or next_url.startswith("tel:"): continue
            if same_domain(next_url, base) and next_url not in seen and len(queue) < max_pages*3:
                queue.append(next_url)
    return pages, bits_all

# --------------- Question generation ---------------

QUESTION_TEMPLATES = [
    "What is {topic} and how does it work?",
    "How to choose the right {topic}?",
    "What does {topic} cost?",
    "{topic} vs alternatives—what’s the difference?",
    "Common problems with {topic} and how to fix them",
    "What size/spec do I need for {topic}?",
    "Is {topic} available near me?",
    "How long does {topic} take to install?",
    "Best {topic} options for {use_case}",
    "Is {topic} worth it for {industry}?",
]

DEFAULT_USE_CASES = ["small business", "enterprise", "home", "commercial"]
DEFAULT_INDUSTRIES = ["retail", "healthcare", "finance", "construction"]

def generate_questions_from_topics(topics: List[str], seed: str, max_q: int = 20) -> List[str]:
    out = []
    use_case = DEFAULT_USE_CASES[0]
    industry = DEFAULT_INDUSTRIES[0]
    for t in topics[:10] or [seed]:
        for tpl in QUESTION_TEMPLATES:
            q = tpl.format(topic=f"{t} {seed}".strip(), use_case=use_case, industry=industry)
            out.append(q)
    # dedupe and cap
    seen, final = set(), []
    for q in out:
        q = re.sub(r"\s+", " ", q).strip()
        if q not in seen:
            final.append(q)
            seen.add(q)
        if len(final) >= max_q: break
    return final

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

def label_cluster(question: str) -> str:
    q = question.lower()
    if " vs" in q: return "comparisons"
    if "cost" in q or "price" in q or "best" in q: return "buying-guide"
    if "how to" in q or "install" in q or "fix" in q: return "how-to"
    if "near me" in q: return "local"
    return "general"

def compute_os(volume_band: str, difficulty_band: str, geography: str = "US", sitegap: float = 0.5,
               business_value: float = 0.5, serp_potential: float = 0.5) -> float:
    vol_map = {"High": 0.9, "Med–High": 0.75, "Med": 0.6, "Low–Med": 0.4, "Low": 0.25}
    diff_map = {"Low": 0.85, "Low–Med": 0.75, "Med": 0.6, "Med–High": 0.45, "High": 0.3}
    vol_w = vol_map.get(volume_band, 0.6)
    diff_w = diff_map.get(difficulty_band, 0.6)
    geo_w = 0.8 if geography == "US" else 0.6
    oscore = 0.25*vol_w + 0.15*diff_w + 0.25*sitegap + 0.10*geo_w + 0.20*0.5 + 0.05*0.5
    return round(float(oscore), 3)

# --------------- Streamlit UI ---------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Universal, site‑aware question generator: crawl headings, infer topics, expand seeds, and export briefs/FAQ JSON‑LD.")

with st.sidebar:
    st.header("Inputs")
    url = st.text_input("Client URL (any site)", placeholder="https://example.com")
    seeds_text = st.text_area("Seed keywords (comma or newline)", value=DEFAULT_SEEDS, height=110)
    geography = st.selectbox("Target geography", ["US", "CA", "Global"], index=0)
    max_pages = st.number_input("Max pages to crawl (same domain)", min_value=3, max_value=50, value=DEFAULT_MAX_PAGES, step=1)
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    seeds = [s.strip() for s in seeds_text.replace("\n", ",").split(",") if s.strip()]
    if not seeds:
        seeds = [s.strip() for s in DEFAULT_SEEDS.split(",")]
    req = AnalyzeRequest(project_url=url or "https://example.com", seeds=seeds, geography=geography, max_pages=int(max_pages))

    st.write("Crawling site… (caps at", req.max_pages, "pages)")
    pages, bits_all = crawl_site(req.project_url, max_pages=req.max_pages)
    st.write(f"Crawled {len(pages)} page(s).")
    topics = top_keywords(bits_all, k=25) if bits_all else []

    rows = []
    for seed in req.seeds:
        qs = generate_questions_from_topics(topics, seed, max_q=25) if topics else generate_questions_from_topics([seed], seed, max_q=25)
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
                "Cluster Label": label_cluster(q),
            })

    cols = ["Seed","Question","Intent","Volume Band","Difficulty Band","Recommended Page Type","On‑Site Coverage","Opportunity Score","Cluster Label"]
    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        st.warning("No questions were generated. Try different seeds or increase crawl pages.")
        st.stop()

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
        md += [f"- {q}" for q in questions[:6]]
        md += ["\n## Outline (H2/H3)"] + [f"- {x}" for x in h2]
        md += ["\n## Internal Links\n- Category / services\n- Pricing / Contact / Quote\n- Shipping / Warranty / Policies"]
        md += ["\n## CTA\n- Get a quote\n- Request a demo / consult"]
        return "\n".join(md)

    briefs = []
    for label in sorted(df["Cluster Label"].unique()):
        subset = df[df["Cluster Label"] == label].sort_values("Opportunity Score", ascending=False)
        top_qs = subset["Question"].tolist()
        page_type = subset["Recommended Page Type"].mode().iloc[0]
        slug_token = label.replace(" ", "-")
        brief_md = build_brief(label, top_qs, page_type, f"/resources/{slug_token}-guide")
        briefs.append((label, brief_md, top_qs[:6]))

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
        "Crawled pages": int(len(pages)),
        "Total questions": int(len(df)),
        "Clusters": int(df["Cluster Label"].nunique()),
        "Avg opportunity": float(df["Opportunity Score"].mean()) if not df.empty else 0.0,
        "Top inferred topics": topics[:10] if topics else []
    })
else:
    st.info("Enter any site URL + any seeds. The app crawls headings on-site, infers topics, generates universal questions, clusters by intent, and exports briefs + FAQ JSON-LD.")
