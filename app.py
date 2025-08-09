# Question Map (Universal, SERP-powered) — v5
# - Crawls the target site for topical context
# - Expands seeds using SERP sources: People Also Ask, Autocomplete, Related Searches
# - (Optional) LLM expansion when OPENAI_API_KEY is set
# - Pure-Python dependencies; designed for Streamlit Cloud

import base64, json, os, re, time
from typing import List, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
from pydantic import BaseModel

APP_TITLE = "Question‑First Keyword Map (SERP‑powered v5)"
DEFAULT_SEEDS = "pricing, installation, warranty, near me, comparison, troubleshooting"
DEFAULT_MAX_PAGES = 8
CRAWL_TIMEOUT = 10
UA = "Mozilla/5.0 (compatible; QuestionMapBot/0.2; +https://example.com/bot)"

# ---------------- Utilities ----------------

def df_to_csv_download(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    return f'<a download="{filename}" href="data:text/csv;base64,{b64}">Download CSV</a>'

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

# ---------------- Models ----------------

class AnalyzeRequest(BaseModel):
    project_url: str
    seeds: List[str]
    geography: str = "US"
    max_pages: int = DEFAULT_MAX_PAGES
    serp_provider: str = "serpapi"  # or "none"
    serp_api_key: str | None = None
    use_llm: bool = False
    openai_model: str = "gpt-4o-mini"

# ---------------- Crawl ----------------

def same_domain(url: str, base: str) -> bool:
    try:
        a = urlparse(url)
        b = urlparse(base)
        return a.netloc.split(":")[0].lower() == b.netloc.split(":")[0].lower() or a.netloc == ""
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
    if t and t.text: bits["title"].append(clean_text(t.text))
    for tag in soup.find_all(["h1","h2","h3"]):
        if tag.text and clean_text(tag.text):
            bits[tag.name].append(clean_text(tag.text))
    m = soup.find("meta", attrs={"name": "description"})
    if m and m.get("content"): bits["meta_desc"].append(clean_text(m["content"]))
    return bits

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
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            next_url = urljoin(url, href)
            if next_url.startswith(("mailto:", "tel:", "javascript:")): continue
            if same_domain(next_url, base) and next_url not in seen and len(queue) < max_pages*3:
                queue.append(next_url)
    return pages, bits_all

def top_keywords(pages_bits: List[Dict[str, List[str]]], k: int = 25) -> List[str]:
    STOP = set("the a an and or but if then with without into onto from for in on at by as to of over under around near within between above below what which who whom this that these those when where why how is are was were be being been do does did has have had having can could should would will shall may might about across after against all almost also although always among amount anyone anything because before behind being both came come comes during each either enough especially ever every few first following forgo further however into it its itself last later least less many more most much must near nearly never next not nothing now often only other our ours ourselves per rather same several since so some such than that their theirs them themselves there therefore these they through throughout thus toward towards until up upon us use used using very via we were what when where whether which while who whom whose why will with within without you your yours yourself yourselves".split())
    freq = {}
    for bits in pages_bits:
        for key in ["title","h1","h2","h3","meta_desc"]:
            for seg in bits.get(key, []):
                for w in re.findall(r"[A-Za-z0-9\-]{3,}", seg.lower()):
                    if w in STOP: continue
                    if w.isdigit(): continue
                    freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keep = []
    for w,c in items:
        if len(keep) >= k: break
        if w in ("home","products","services","contact","about","menu","shop"): continue
        keep.append(w)
    return keep

# ---------------- SERP Expansion ----------------

def serpapi_related(seed: str, api_key: str, location: str = "United States") -> Dict[str, List[str]]:
    """Use SerpAPI to fetch PAA, related searches, and autocomplete suggestions"""
    out = {"paa": [], "related": [], "suggestions": []}
    try:
        # Google search (related + PAA)
        params = {"engine":"google", "q": seed, "location": location, "api_key": api_key}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=12)
        if r.status_code == 200:
            data = r.json()
            # People Also Ask
            for item in data.get("related_questions", []) or []:
                q = clean_text(item.get("question"))
                if q: out["paa"].append(q)
            # Related searches
            for item in data.get("related_searches", []) or []:
                q = clean_text(item.get("query"))
                if q: out["related"].append(q)
        # Autocomplete suggestions
        ac_params = {"engine":"google_autocomplete", "q": seed, "api_key": api_key}
        r2 = requests.get("https://serpapi.com/search.json", params=ac_params, timeout=12)
        if r2.status_code == 200:
            data2 = r2.json()
            for s in data2.get("suggestions", []) or []:
                term = clean_text(s.get("value"))
                if term: out["suggestions"].append(term)
    except Exception:
        pass
    return out

def expand_with_serp(seed: str, provider: str, api_key: str | None, geography: str) -> List[str]:
    if provider == "serpapi" and api_key:
        data = serpapi_related(seed, api_key, "United States" if geography == "US" else geography)
        bag = []
        bag += [q if q.endswith("?") else f"{q}?" for q in data.get("paa", [])]
        bag += data.get("suggestions", [])
        bag += data.get("related", [])
        # Normalize into questions where possible
        norm = []
        for q in bag:
            q2 = q.strip()
            if not q2: continue
            if not q2.endswith("?") and not any(q2.lower().startswith(x) for x in ["what","how","why","when","where","who","does","can","is","are","should"]):
                q2 = f"What about {q2}?"
            norm.append(q2)
        # Dedup
        seen, out = set(), []
        for q in norm:
            if q.lower() not in seen:
                out.append(q)
                seen.add(q.lower())
        return out[:40]
    return []

# ---------------- Optional LLM Expansion ----------------

def llm_expand(seed: str, topics: List[str], model: str = "gpt-4o-mini") -> List[str]:
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Generate 25 searcher-style questions (short, natural) a US user might ask about: '{seed}'. "
            "Bias toward high-intent commercial & informational queries. "
            "Mix cost/pricing, vs/comparisons, troubleshooting, and how-to. "
            "Incorporate these site topics when relevant: " + ", ".join(topics[:10])
        ).format(seed=seed)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        text = resp.choices[0].message.content
        lines = [clean_text(x) for x in text.split("\n") if clean_text(x)]
        # Clean bullets/numbers
        out = []
        for ln in lines:
            ln = re.sub(r"^\d+[\).\s-]+", "", ln)
            if not ln.endswith("?"):
                ln = ln.rstrip(".") + "?"
            out.append(ln)
        # Dedup
        seen, uniq = set(), []
        for q in out:
            if q.lower() not in seen:
                uniq.append(q)
                seen.add(q.lower())
        return uniq[:40]
    except Exception:
        return []

# ---------------- Scoring / Labeling ----------------

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
               business_value: float = 0.5, serp_potential: float = 0.8) -> float:
    vol_map = {"High": 0.9, "Med–High": 0.75, "Med": 0.6, "Low–Med": 0.4, "Low": 0.25}
    diff_map = {"Low": 0.85, "Low–Med": 0.75, "Med": 0.6, "Med–High": 0.45, "High": 0.3}
    vol_w = vol_map.get(volume_band, 0.6)
    diff_w = diff_map.get(difficulty_band, 0.6)
    geo_w = 0.8 if geography == "US" else 0.6
    oscore = 0.25*vol_w + 0.15*diff_w + 0.20*sitegap + 0.10*geo_w + 0.20*business_value + 0.10*serp_potential
    return round(float(oscore), 3)

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Now with SERP sources: People Also Ask, Autocomplete, Related Searches. Optional LLM expansion.")

with st.sidebar:
    st.header("Inputs")
    url = st.text_input("Client URL (any site)", placeholder="https://example.com")
    seeds_text = st.text_area("Seed keywords (comma or newline)", value=DEFAULT_SEEDS, height=110)
    geography = st.selectbox("Target geography", ["US", "CA", "Global"], index=0)
    max_pages = st.number_input("Max pages to crawl (same domain)", min_value=3, max_value=40, value=DEFAULT_MAX_PAGES, step=1)
    st.divider()
    st.subheader("SERP Expansion")
    provider = st.selectbox("Provider", ["serpapi", "none"], index=0)
    api_key = st.text_input("SERP API key (or set SERP_API_KEY env/secret)", type="password")
    use_llm = st.toggle("Use LLM expansion (OpenAI)", value=False)
    llm_model = st.text_input("OpenAI model", value="gpt-4o-mini")
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    seeds = [s.strip() for s in seeds_text.replace("\n", ",").split(",") if s.strip()]
    if not seeds:
        st.warning("Please enter at least one seed.")
        st.stop()
    serp_key = api_key or os.environ.get("SERP_API_KEY") or st.secrets.get("SERP_API_KEY", None)
    req = AnalyzeRequest(project_url=url or "https://example.com", seeds=seeds, geography=geography,
                         max_pages=int(max_pages), serp_provider=provider, serp_api_key=serp_key,
                         use_llm=use_llm, openai_model=llm_model)

    # Crawl site for context
    with st.status("Crawling site for topical context…", expanded=False):
        pages, bits_all = crawl_site(req.project_url, max_pages=req.max_pages)
        topics = top_keywords(bits_all, k=25) if bits_all else []
        st.write(f"Crawled {len(pages)} page(s).")
        if topics:
            st.write("Top topics:", ", ".join(topics[:12]))

    # Build questions
    rows = []
    for seed in req.seeds:
        qset = []
        # SERP expansion first
        serp_qs = expand_with_serp(seed, req.serp_provider, req.serp_api_key, req.geography)
        qset += serp_qs
        # Optional LLM expansion
        if req.use_llm:
            llm_qs = llm_expand(seed, topics, model=req.openai_model)
            qset += llm_qs
        # If SERP/LLM empty, fallback to generic templates blended with topics
        if not qset:
            base_templates = [
                f"What is {seed} and how does it work?",
                f"How to choose {seed}?",
                f"What does {seed} cost?",
                f"{seed} vs alternatives—what’s the difference?",
                f"Common problems with {seed} and how to fix them",
            ]
            if topics:
                base_templates += [f"How does {seed} compare to {topics[0]}?", f"Is {seed} available near me?"]
            qset = base_templates

        # Build rows
        for q in qset:
            qn = clean_text(q)
            vol_band, diff_band = estimate_bands(qn)
            page_type = rules_page_type(qn)
            intent = rules_intent(qn)
            oscore = compute_os(vol_band, diff_band, geography=req.geography, serp_potential=0.9 if serp_qs else 0.5)
            rows.append({
                "Seed": seed,
                "Question": qn if qn.endswith("?") else (qn + "?"),
                "Intent": intent,
                "Volume Band (proxy)": vol_band,
                "Difficulty Band (proxy)": diff_band,
                "Recommended Page Type": page_type,
                "On‑Site Coverage": "TBD",
                "Opportunity Score": oscore,
                "Cluster Label": (
                    "comparisons" if " vs" in qn.lower() else
                    "buying-guide" if any(x in qn.lower() for x in ["best","cost","price"]) else
                    "how-to" if any(x in qn.lower() for x in ["how to","install","fix"]) else
                    "local" if "near me" in qn.lower() else "general"
                ),
                "Source": ("SERP" if q in serp_qs else ("LLM" if req.use_llm else "Fallback"))
            })

    df = pd.DataFrame(rows, columns=[
        "Seed","Question","Intent","Volume Band (proxy)","Difficulty Band (proxy)",
        "Recommended Page Type","On‑Site Coverage","Opportunity Score","Cluster Label","Source"
    ])

    if df.empty:
        st.error("No questions generated. Check your SERP API key or toggle LLM fallback.")
        st.stop()

    st.success("Analysis complete")

    st.subheader("Questions (SERP + optional LLM)")
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
        md += [f"- {q}" for q in questions[:8]]
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
    st.markdown(df_to_csv_download(df, "question_map_serp.csv"), unsafe_allow_html=True)
    all_md = "\n\n".join([b[1] for b in briefs])
    st.download_button("Download Briefs (Markdown)", data=all_md.encode("utf-8"), file_name="briefs.md", mime="text/markdown")

    st.subheader("Summary")
    st.write({
        "Crawled pages": int(len(df)),
        "Clusters": int(df["Cluster Label"].nunique()),
        "SERP used": req.serp_provider if req.serp_provider != "none" and req.serp_api_key else "No",
        "LLM used": req.use_llm,
    })

else:
    st.info("Enter a site URL + seed keywords. Provide a SERP API key (SerpAPI) to pull PAA/autocomplete/related searches. Toggle LLM for extra expansions.")
