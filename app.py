# Question Map — v7.6 (Full features + diagnostics)
import base64, json, os, re
from typing import List, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
from pydantic import BaseModel

APP_TITLE = "Question‑First Keyword Map (v7.6: SERP + LLM + Coverage + Guides + Diagnostics)"
DEFAULT_SEEDS = "pricing, installation, warranty, near me, comparison, troubleshooting"
DEFAULT_MAX_PAGES = 60
CRAWL_TIMEOUT = 15
UA = "Mozilla/5.0 (compatible; QuestionMapBot/0.8; +https://example.com/bot)"

def df_to_csv_download(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    return f'<a download="{filename}" href="data:text/csv;base64,{b64}">Download CSV</a>'

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

STOP = set("the a an and or but if then with without into onto from for in on at by as to of over under around near within between above below what which who whom this that these those when where why how is are was were be being been do does did has have had having can could should would will shall may might about across after against all almost also although always among amount anyone anything because before behind being both came come comes during each either enough especially ever every few first following forgo further however into it its itself last later least less many more most much must near nearly never next not nothing now often only other our ours ourselves per rather same several since so some such than that their theirs them themselves there therefore these they through throughout thus toward towards until up upon us use used using very via we were what when where whether which while who whom whose why will with within without you your yours yourself yourselves".split())

def norm_tokens(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    toks = [t for t in re.split(r"\s+", s) if t]
    toks = [t for t in toks if t not in STOP and not t.isdigit() and len(t) > 2]
    return toks

def bigrams(toks: List[str]) -> List[str]:
    return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)] if len(toks) > 1 else []

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

class AnalyzeRequest(BaseModel):
    project_url: str
    seeds: List[str]
    geography: str = "US"
    max_pages: int = DEFAULT_MAX_PAGES
    serp_provider: str = "serpapi"
    serp_api_key: str | None = None
    use_llm: bool = False
    openai_model: str = "gpt-4o-mini"
    crawl_scope: str = "Homepage + Blog"  # or "Whole site (same domain)"
    extra_roots: List[str] = []  # manual blog roots

BLOG_HINTS = ("blog", "news", "insights", "articles", "learn", "resources", "guides", "knowledge", "academy")

def path_matches_scope(path: str, scope: str, extra_roots: List[str]) -> bool:
    if scope == "Whole site (same domain)":
        return True
    if path in ("", "/"):
        return True
    for r in extra_roots:
        r = "/" + r.strip("/").lower() + "/"
        if r in (path + "/").lower():
            return True
    return any(f"/{hint}/" in (path + "/").lower() for hint in BLOG_HINTS)

def fetch(url: str) -> tuple[str, int, str]:
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=CRAWL_TIMEOUT, allow_redirects=True)
        ctype = resp.headers.get("Content-Type", "").lower()
        if resp.status_code == 200 and ("text/html" in ctype or ctype == "" or "charset" in ctype):
            return resp.text, resp.status_code, ctype
        return "", resp.status_code, ctype
    except Exception:
        return "", 0, ""

def extract_text_bits(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    bits = {"title": "", "h": [], "meta_desc": "", "og_desc": "", "text": ""}
    t = soup.find("title")
    if t and t.text: bits["title"] = clean_text(t.text)
    for tag in soup.find_all(["h1","h2","h3"]):
        if tag.text and clean_text(tag.text):
            bits["h"].append(clean_text(tag.text))
    m = soup.find("meta", attrs={"name": "description"})
    if m and m.get("content"): bits["meta_desc"] = clean_text(m["content"])
    og = soup.find("meta", property="og:description")
    if og and og.get("content"): bits["og_desc"] = clean_text(og["content"])
    ps = [clean_text(p.get_text(" ")) for p in soup.find_all(["p","li"])]
    bits["text"] = " ".join(ps)[:60000]
    return bits

def crawl_site(start_url: str, max_pages: int, scope: str, extra_roots: List[str]):
    seen = set()
    queue = [start_url]
    pages, bits_all, crawl_log = [], [], []
    base_netloc = urlparse(start_url).netloc
    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen: continue
        seen.add(url)
        html, code, ctype = fetch(url)
        crawl_log.append({"url": url, "status": code, "content_type": ctype, "ok": bool(html)})
        if not html: continue
        pages.append(url)
        bits_all.append(extract_text_bits(html))
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            next_url = urljoin(url, href)
            u = urlparse(next_url)
            if next_url.startswith(("mailto:", "tel:", "javascript:")): continue
            if u.netloc and u.netloc != base_netloc: continue
            if not path_matches_scope(u.path, scope, extra_roots): continue
            if next_url not in seen and len(queue) < max_pages*4:
                queue.append(next_url)
    return pages, bits_all, crawl_log

def build_index(pages: List[str], bits_all: List[Dict[str, Any]]):
    index = []
    for url, bits in zip(pages, bits_all):
        title = bits.get("title", "")
        headings = " ".join(bits.get("h", []))
        meta_desc = bits.get("meta_desc", "")
        og_desc = bits.get("og_desc", "")
        body = bits.get("text", "")
        index.append({
            "url": url,
            "title": title,
            "headings": headings,
            "meta_desc": meta_desc,
            "og_desc": og_desc,
            "body": body,
            "title_tokens": set(norm_tokens(title)),
            "head_tokens": set(norm_tokens(headings)),
            "meta_tokens": set(norm_tokens(meta_desc + " " + og_desc)),
            "body_tokens": set(norm_tokens(body)),
        })
    return index

def text_contains_bigram(text: str, bigrams_list: List[str]) -> int:
    t = text.lower()
    return sum(1 for bg in bigrams_list if bg in t)

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def norm_tokens(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    toks = [t for t in re.split(r"\s+", s) if t]
    toks = [t for t in toks if t not in STOP and not t.isdigit() and len(t) > 2]
    return toks

def best_page_match(question: str, index: List[Dict[str, Any]]):
    def bigrams(toks: List[str]) -> List[str]:
        return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)] if len(toks) > 1 else []

    q_norm = clean_text(question).lower().rstrip("?")
    q_tokens = set(norm_tokens(q_norm)) or set(re.findall(r"[a-z0-9]+", q_norm))
    q_bigrams = bigrams(list(q_tokens))

    candidates = []
    for doc in index:
        title_l = doc["title"].lower()
        heads_l = doc["headings"].lower()
        meta_l  = (doc.get("meta_desc","") + " " + doc.get("og_desc","")).lower()
        body_l  = doc["body"].lower()

        exact_title = 1.0 if q_norm and q_norm in title_l else 0.0
        exact_head  = 1.0 if q_norm and q_norm in heads_l else 0.0
        exact_meta  = 1.0 if q_norm and q_norm in meta_l else 0.0
        exact_body  = 1.0 if q_norm and q_norm in body_l else 0.0

        jac_title = jaccard(q_tokens, doc["title_tokens"])
        jac_head  = jaccard(q_tokens, doc["head_tokens"])
        jac_meta  = jaccard(q_tokens, doc["meta_tokens"])
        jac_body  = jaccard(q_tokens, doc["body_tokens"])

        bg_title = text_contains_bigram(title_l, q_bigrams)
        bg_head  = text_contains_bigram(heads_l, q_bigrams)
        bg_body  = text_contains_bigram(body_l, q_bigrams)
        bg_norm  = min(1.0, (bg_title*1.0 + bg_head*0.7 + bg_body*0.4) / 3.0)

        score = (
            0.28*exact_title + 0.20*exact_head + 0.08*exact_meta + 0.08*exact_body +
            0.12*jac_title + 0.12*jac_head + 0.06*jac_meta + 0.10*jac_body +
            0.06*bg_norm
        )

        # snippet
        snippet = ""
        if exact_title: snippet = doc["title"]
        elif exact_head: snippet = doc["headings"][:220]
        elif exact_meta: snippet = (doc.get("meta_desc") or doc.get("og_desc",""))[:220]
        else:
            body = doc["body"]
            pos = -1
            for t in q_tokens:
                i = body.lower().find(t)
                if i != -1 and (pos == -1 or i < pos):
                    pos = i
            if pos != -1:
                start = max(0, pos - 80); end = min(len(body), pos + 220)
                snippet = clean_text(body[start:end])
            else:
                snippet = clean_text(body[:220])

        candidates.append((doc["url"], float(score), snippet))

    candidates.sort(key=lambda x: x[1], reverse=True)
    if not candidates:
        return ("", 0.0, "", [])
    return candidates[0][0], candidates[0][1], candidates[0][2], candidates[:5]

def coverage_label(score: float) -> str:
    if score >= 0.58: return "Yes"
    if score >= 0.33: return "Partial"
    return "No"

def serpapi_related(seed: str, api_key: str, location: str = "United States") -> Dict[str, List[str]]:
    out = {"paa": [], "related": [], "suggestions": []}
    try:
        params = {"engine":"google", "q": seed, "location": location, "api_key": api_key}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=12)
        if r.status_code == 200:
            data = r.json()
            for item in data.get("related_questions", []) or []:
                q = clean_text(item.get("question"))
                if q: out["paa"].append(q)
            for item in data.get("related_searches", []) or []:
                q = clean_text(item.get("query"))
                if q: out["related"].append(q)
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
        bag += [q if q.endswith("?") else q for q in data.get("paa", [])]
        bag += data.get("suggestions", [])
        bag += data.get("related", [])
        seen, out = set(), []
        for q in [clean_text(x) for x in bag if clean_text(x)]:
            k = q.lower()
            if k not in seen:
                out.append(q)
                seen.add(k)
        return out[:80]
    return []

def llm_expand(seed: str, topics: List[str], model: str = "gpt-4o-mini") -> List[str]:
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Generate 25 US-searcher-style queries (short, natural) about '{seed}'. "
            "Blend commercial & informational intent. Prefer phrases users actually type; avoid awkward prefixes. "
            "Cover cost/pricing, comparisons, near-me, how-to, troubleshooting."
        ).format(seed=seed)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        text = resp.choices[0].message.content
        lines = [clean_text(x) for x in text.split("\n") if clean_text(x)]
        out = []
        for ln in lines:
            ln = re.sub(r"^\d+[\).\s-]+", "", ln)
            out.append(ln)
        seen, uniq = set(), []
        for q in out:
            k = q.lower()
            if k not in seen:
                uniq.append(q)
                seen.add(k)
        return uniq[:40]
    except Exception:
        return []

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
               business_value: float = 0.5, serp_potential: float = 0.9) -> float:
    vol_map = {"High": 0.9, "Med–High": 0.75, "Med": 0.6, "Low–Med": 0.4, "Low": 0.25}
    diff_map = {"Low": 0.85, "Low–Med": 0.75, "Med": 0.6, "Med–High": 0.45, "High": 0.3}
    vol_w = vol_map.get(volume_band, 0.6)
    diff_w = diff_map.get(difficulty_band, 0.6)
    geo_w = 0.8 if geography == "US" else 0.6
    oscore = 0.25*vol_w + 0.15*diff_w + 0.20*sitegap + 0.10*geo_w + 0.20*business_value + 0.10*serp_potential
    return round(float(oscore), 3)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Full feature pack + diagnostics: coverage that actually fills, guides, and visibility into crawl/indexing.")

with st.sidebar:
    st.header("Inputs")
    url = st.text_input("Client URL (any site)", placeholder="https://example.com")
    seeds_text = st.text_area("Seed keywords (comma or newline)", value=DEFAULT_SEEDS, height=110)
    geography = st.selectbox("Target geography", ["US", "CA", "Global"], index=0)
    crawl_scope = st.selectbox("Crawl scope", ["Homepage + Blog", "Whole site (same domain)"], index=0)
    extra_roots_text = st.text_input("Extra blog roots (comma-separated, e.g. knowledge, academy)")
    max_pages = st.number_input("Max pages to crawl (cap)", min_value=5, max_value=200, value=DEFAULT_MAX_PAGES, step=1)
    st.divider()
    st.subheader("SERP Expansion")
    provider = st.selectbox("Provider", ["serpapi", "none"], index=0)
    api_key = st.text_input("SERP API key (or set SERP_API_KEY env/secret)", type="password")
    st.subheader("LLM Expansion")
    use_llm = st.toggle("Use LLM (OpenAI)", value=False)
    llm_model = st.text_input("OpenAI model", value="gpt-4o-mini")
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    seeds = [s.strip() for s in seeds_text.replace("\n", ",").split(",") if s.strip()]
    if not seeds:
        st.warning("Please enter at least one seed.")
        st.stop()
    extra_roots = [x.strip() for x in (extra_roots_text or "").split(",") if x.strip()]
    serp_key = api_key or os.environ.get("SERP_API_KEY") or st.secrets.get("SERP_API_KEY", None)
    req = AnalyzeRequest(project_url=url or "https://example.com", seeds=seeds, geography=geography,
                         max_pages=int(max_pages), serp_provider=provider, serp_api_key=serp_key,
                         use_llm=use_llm, openai_model=llm_model, crawl_scope=crawl_scope, extra_roots=extra_roots)

    # Crawl + index + diagnostics
    with st.status("Crawling and indexing…", expanded=True):
        pages, bits_all, crawl_log = crawl_site(req.project_url, max_pages=req.max_pages, scope=req.crawl_scope, extra_roots=req.extra_roots)
        index = build_index(pages, bits_all) if pages else []
        st.write(f"Crawled pages: {len(pages)} / Cap: {req.max_pages}")
        bad = [row for row in crawl_log if not row["ok"]]
        if bad:
            st.warning(f"{len(bad)} pages returned non-HTML or error; they were skipped.")
        if pages:
            st.write("Sample URLs:", pages[:8])

    if not index:
        st.error("No crawlable HTML pages were indexed.")
        st.info("Try: (1) ensure URL includes https://, (2) switch scope to 'Whole site', (3) add custom blog roots, (4) increase Max pages.")
        st.stop()

    # Build questions
    q_all = []
    for seed in req.seeds:
        qset = []
        serp_qs = expand_with_serp(seed, req.serp_provider, req.serp_api_key, req.geography)
        qset += serp_qs
        if req.use_llm:
            qset += llm_expand(seed, [], model=req.openai_model)
        if not qset:
            qset = [
                f"What is {seed} and how does it work?",
                f"How to choose {seed}?",
                f"What does {seed} cost?",
                f"{seed} vs alternatives—what’s the difference?",
                f"Common problems with {seed} and how to fix them"
            ]
        q_all.extend(qset)

    # Dedup
    seen, uniq_qs = set(), []
    for q in [clean_text(x) for x in q_all if clean_text(x)]:
        k = q.lower()
        if k not in seen:
            uniq_qs.append(q)
            seen.add(k)

    # Score + coverage
    rows = []
    for qn in uniq_qs:
        vol_band, diff_band = estimate_bands(qn)
        page_type = rules_page_type(qn)
        intent = rules_intent(qn)
        oscore = compute_os(vol_band, diff_band, geography=req.geography, serp_potential=0.9 if req.serp_api_key else 0.5)

        best_url, match_score, snippet, _ = best_page_match(qn, index)
        cov = coverage_label(match_score)

        rows.append({
            "Question": qn if qn.endswith("?") else (qn + "?"),
            "Intent": intent,
            "Volume Band (proxy)": vol_band,
            "Difficulty Band (proxy)": diff_band,
            "Recommended Page Type": page_type,
            "On‑Site Coverage": cov,
            "Match Score": round(match_score, 3),
            "Best Match URL": best_url,
            "Snippet": snippet,
            "Opportunity Score": oscore,
            "Cluster Label": label_cluster(qn),
            "Source": "SERP/LLM"
        })

    df = pd.DataFrame(rows)

    st.success("Analysis complete")

    st.subheader("Questions + Coverage")
    st.dataframe(df.sort_values(["On‑Site Coverage","Opportunity Score"], ascending=[True, False]), use_container_width=True)

    # Coverage diagnostics (token counts)
    st.subheader("Coverage diagnostics")
    diag = []
    for row in index:
        diag.append({
            "URL": row["url"],
            "Title tokens": len(row["title_tokens"]),
            "Head tokens": len(row["head_tokens"]),
            "Meta tokens": len(row["meta_tokens"]),
            "Body tokens": len(row["body_tokens"]),
            "Title": row["title"][:120],
        })
    st.dataframe(pd.DataFrame(diag), use_container_width=True)

    # Content Guides (per cluster, focus on gaps)
    st.subheader("Content Guides (auto-briefs per cluster, focus on gaps)")
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
            f"# Content Guide: {cluster_label}",
            f"**Page Type:** {page_type}",
            f"**Proposed URL:** {url_slug}",
            "\n**Primary Questions to Answer**",
        ]
        md += [f"- {q}" for q in questions[:8]]
        md += ["\n## Outline (H2/H3)"] + [f"- {x}" for x in h2]
        md += ["\n## Internal Links\n- Category / services\n- Pricing / Contact / Quote\n- Shipping / Warranty / Policies"]
        md += ["\n## CTA\n- Get a quote\n- Request a demo / consult"]
        return "\n".join(md)

    briefs = []
    for label in sorted(df["Cluster Label"].unique()):
        subset = df[(df["Cluster Label"] == label) & (df["On‑Site Coverage"] != "Yes")].sort_values("Opportunity Score", ascending=False)
        top_qs = subset["Question"].tolist() or df[df["Cluster Label"] == label]["Question"].tolist()
        page_type = subset["Recommended Page Type"].mode().iloc[0] if not subset.empty else df[df["Cluster Label"] == label]["Recommended Page Type"].mode().iloc[0]
        slug_token = label.replace(" ", "-")
        brief_md = build_brief(label, top_qs, page_type, f"/resources/{slug_token}-guide")
        briefs.append((label, brief_md, top_qs[:6]))

    for label, md, top_qs in briefs:
        with st.expander(f"Guide: {label} — focus on gaps"):
            st.markdown(md)
            faq_entities = [{
                "@type": "Question",
                "name": q,
                "acceptedAnswer": {"@type": "Answer", "text": "Short, helpful answer (90–160 words)."}
            } for q in top_qs]
            faq_json = {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": faq_entities}
            st.code(json.dumps(faq_json, indent=2), language="json")

    # Export
    st.subheader("Export")
    st.markdown(df_to_csv_download(df, "question_map_v7_6.csv"), unsafe_allow_html=True)
    all_md = "\n\n".join([b[1] for b in briefs]) if briefs else ""
    st.download_button("Download Content Guides (Markdown)", data=all_md.encode("utf-8"), file_name="content_guides_v7_6.md", mime="text/markdown")

else:
    st.info("Paste a site URL + seeds. v7.6 merges targeted crawl, strong matching, diagnostics, and Content Guides.")
