# streamlit_grc_appwithapi.py


import streamlit as st
import os
import json
import time
import textwrap
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import contextlib
import io

# Optional import of local LLM (deferred)
try:
    from gpt4all import GPT4All
    HAS_GPT4ALL = True
except Exception:
    HAS_GPT4ALL = False

from pathlib import Path

# -------- Base paths (Cloud-safe) --------
BASE_DIR = Path(__file__).resolve().parent

# -------- Mistral API config --------
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MISTRAL_MODEL = "mistral-small-latest"

# -------- Local LLM (GPT4All) defaults --------
DEFAULT_MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_S.gguf"

THREADS = 8
N_CTX = 4096
ALLOW_DOWNLOAD = False
USE_CUDA = False   # keep False for stability unless explicitly tested

# -------- Vector index --------
INDEX_DIR = BASE_DIR / "rag_index"
POLICY_INDEX_PATH = BASE_DIR / "rag_index" / "policies.faiss"
POLICY_META_PATH  = BASE_DIR / "rag_index" / "policies_meta.json"

CONTROL_INDEX_PATH = BASE_DIR / "rag_index" / "controls.faiss"
CONTROL_META_PATH  = BASE_DIR / "rag_index" / "controls_meta.json"


# -------- Embeddings --------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------- RAG behavior --------
MAX_CONTEXT_CHARS = 4000
# DEFAULT_TOP_K = 4
MAX_LOCAL_CHARS = 800

# -------------------------------------------------------

SUPPORTED_FRAMEWORKS = {
    "NIST": ["NIST", "800-53", "800-171"],
    "PCI-DSS": ["PCI", "PCI-DSS"],
    "HIPAA": ["HIPAA"],
    "ISO": ["ISO", "27001", "27002"],
    "SOC2": ["SOC", "SOC2"],
    "FEDRAMP": ["FEDRAMP"],
    "GDPR": ["GDPR"],
    "CIS": ["CIS"]
}


st.set_page_config(page_title="GRC RAG — Embedding + FAISS & Metadata + LLM Mistral", layout="wide")
st.title("Cybersecurity GRC RAG with LLM")
st.markdown("### **Embedding Model • FAISS Index • Metadata • LLM Mistral • GPT4ALL**")
 
# --- Sidebar: model, index, runtime settings ---
with st.sidebar:
    st.header("LLM Selection")

    llm_backend = st.radio(
        "Choose answer generation backend",
        options=[
            "Mistral API (Cloud)",
            "Mistral LLM (Local)",
        ],
        index=0,
    )

    st.session_state["llm_backend"] = llm_backend

    st.markdown("---")

    if llm_backend == "Mistral API (Cloud)":
        st.info(
            "Uses Mistral hosted API.\n\n"
            "✔ Public Hosted LLM \n\n"
            "✔ No local model required\n\n"
            "Can use Sec-LLMs for security contextual feeds"
        )

    else:
        st.info(
            "Uses local Mistral LLM model.\n\n"
            "✔ Private DC/ Air-gapped Env \n\n"
            "✔ Requires local model file\n\n"
            "No Local LLM hosted hence will not generate LLM Answer"
        )

        st.text(f"Model dir: {DEFAULT_MODEL_DIR}")
        st.text(f"Model file: {DEFAULT_MODEL_FILENAME}")

# --- Session state caching ---
# if "faiss_index" not in st.session_state:
#    st.session_state.faiss_index = None
# if "meta" not in s t.session_state:
#    st.session_state.meta = None
if "policy_index" not in st.session_state:
    st.session_state.policy_index = None
if "policy_meta" not in st.session_state:
    st.session_state.policy_meta = None

if "control_index" not in st.session_state:
    st.session_state.control_index = None
if "control_meta" not in st.session_state:
    st.session_state.control_meta = None

if "embed_model" not in st.session_state:
    st.session_state.embed_model = None

if "gpt_model" not in st.session_state:
    st.session_state.gpt_model = None

if "model_loaded_info" not in st.session_state:
    st.session_state.model_loaded_info = ""

def wrap_for_local_mistral(prompt: str) -> str:
    return f"""[INST]
You are a senior GRC compliance auditor.

TASK:
- Identify relevant NIST AC-17 controls
- Compare policy statements to each control
- Mark each as Compliant / Partially Compliant / Non-Compliant
- Provide bullet-point justification
- End with a short executive summary

STRICT RULES:
- STOP after the executive summary
- DO NOT repeat the analysis
- DO NOT restate controls
- DO NOT add extra sections

{prompt}
[/INST]
"""

# --- helpers ---

def load_policy_index():
    if st.session_state.policy_index is not None:
        return True
    if not POLICY_INDEX_PATH.exists() or not POLICY_META_PATH.exists():
        st.error("Policy index or metadata missing.")
        return False

    st.session_state.policy_index = faiss.read_index(str(POLICY_INDEX_PATH))
    with open(POLICY_META_PATH, "r", encoding="utf-8") as f:
        st.session_state.policy_meta = json.load(f)
    return True


def load_control_index():
    if st.session_state.control_index is not None:
        return True
    if not CONTROL_INDEX_PATH.exists() or not CONTROL_META_PATH.exists():
        st.error("Control index or metadata missing.")
        return False

    st.session_state.control_index = faiss.read_index(str(CONTROL_INDEX_PATH))
    with open(CONTROL_META_PATH, "r", encoding="utf-8") as f:
        st.session_state.control_meta = json.load(f)
    return True


def load_embedder():
    if st.session_state.embed_model is not None:
        return st.session_state.embed_model
    st.info("Loading embedding model (this may take a moment)...")
    embed = SentenceTransformer(EMBED_MODEL_NAME)
    st.session_state.embed_model = embed
    st.success("Embedder loaded.")
    return embed

def embed_query(q: str):
    embed_model = load_embedder()
    v = embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    v = v.astype(np.float32)
    return v

# --------------------------------------------------------------
# Helper 1: detect referenced filename inside the user query
# --------------------------------------------------------------
import re

def _normalize_text_for_match(s: str) -> str:
    # lowercase, replace non-alphanumeric with space, collapse spaces
    s = s.lower()
    s = re.sub(r'[_\-\.\,]', ' ', s)            # make common filename separators spaces
    s = re.sub(r'[^a-z0-9\s]', ' ', s)          # remove other punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def find_referenced_filename(query: str, meta, min_token_overlap: int = 2):
    """
    Return the best-matching filename from metadata or None.
    Uses normalized token overlap and some heuristics.
    """
    if not query or not meta:
        return None

    q_norm = _normalize_text_for_match(query)

    # fast exact-match: filename (with or without extension) present in query
    for md in meta["metadata"]:
        fn = md.get("filename", "")
        if not fn:
            continue
        fn_norm = _normalize_text_for_match(fn)
        if fn_norm and fn_norm in q_norm:
            return md.get("filename")

    # compute token-overlap scores for each filename base
    q_tokens = set(q_norm.split())
    best = (None, 0)   # (filename, overlap_count)
    for md in meta["metadata"]:
        fn = md.get("filename", "")
        if not fn:
            continue
        base = fn.rsplit(".", 1)[0]
        base_norm = _normalize_text_for_match(base)
        base_tokens = set(base_norm.split())
        overlap = len(q_tokens & base_tokens)
        if overlap > best[1]:
            best = (md.get("filename"), overlap)

    # if best overlap meets threshold, return it
    if best[0] and best[1] >= min_token_overlap:
        return best[0]

    # Last-resort: if query contains the single word "remote" AND "access" AND "policy", pick the file
    if all(tok in q_tokens for tok in ("remote", "access", "policy")):
        for md in meta["metadata"]:
            fn = md.get("filename","").lower()
            if "remote" in fn and "access" in fn and "policy" in fn:
                return md.get("filename")

    return None

def find_controls_by_id(control_meta, control_ids):
    """
    Deterministically fetch controls by control_id
    """
    hits = []

    for idx, md in enumerate(control_meta["metadata"]):
        cid = get_control_id(md)
        if cid and cid in control_ids:
            hits.append({
                "score": 999.0,  # force top
                "index": idx,
                "passage": control_meta["passages"][idx],
                "meta": md,
            })

    return hits

# --------------------------------------------------
# Generic FAISS search helper (shared by policy/control)
# --------------------------------------------------
def search_index(index, meta, q_vec, k):
    D, I = index.search(q_vec, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        hits.append({
            "score": float(score),
            "index": int(idx),
            "passage": meta["passages"][idx],
            "meta": meta["metadata"][idx],
        })
    return hits

# diversify_by_filename during retrieval

def diversify_by_filename(hits, max_per_file=1):
    """
    Keep at most `max_per_file` chunks per document filename.
    Preserves ranking order.
    """
    out = []
    seen = {}

    for h in hits:
        fn = h["meta"].get("filename")
        seen.setdefault(fn, 0)

        if seen[fn] < max_per_file:
            out.append(h)
            seen[fn] += 1

    return out

# For policy_specific_question during retrieval

def is_policy_specific_question(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in ["policy", "procedure", "standard"])

# Standard/Framework detection

def detect_frameworks_from_query(query: str):
    q = query.upper()
    matched = set()

    for fw, keywords in SUPPORTED_FRAMEWORKS.items():
        for kw in keywords:
            if kw in q:
                matched.add(fw)
                break

    return matched

def format_control_label(meta: Dict) -> str:
    """
    Returns a human-readable control label for UI + prompt.
    Works with new deterministic control metadata.
    """
    framework = meta.get("framework") or (
        meta.get("frameworks")[0] if meta.get("frameworks") else "UNKNOWN"
    )
    control_id = meta.get("control_id", "UNKNOWN-CONTROL")
    title = meta.get("title", "")

    if title:
        return f"{framework} {control_id} — {title}"
    return f"{framework} {control_id}"

def normalize_framework(fw: str) -> str:
    return fw.upper().replace(" ", "").replace("-", "")

import re

CONTROL_ID_PATTERN = re.compile(
    r'\b([A-Z]{1,3}-\d+(?:\(\d+\))?|A\.\d+(?:\.\d+)*|\d+(?:\.\d+)+)\b'
)

def extract_control_ids(query: str) -> set[str]:
    return {m.group(1) for m in CONTROL_ID_PATTERN.finditer(query.upper())}

def normalize_control_id(cid: str):
    # IA-2(1) → IA-2
    if "(" in cid:
        return cid.split("(")[0]

    # 7.1.1 → 7.1
    parts = cid.split(".")
    if len(parts) > 2:
        return ".".join(parts[:2])

    return cid

def get_control_id(meta: dict) -> str:
    """
    Robustly extract canonical control ID (e.g., IA-2, AC-17)
    from control metadata.
    """
    # Preferred explicit field
    if meta.get("control_id"):
        return meta["control_id"].upper()

    # Fallbacks
    for k in ("ref", "control", "id"):
        if meta.get(k):
            return str(meta[k]).upper()

    # Last resort: parse from title or text
    text = " ".join(
        str(meta.get(k, "")) for k in ("title", "summary")
    ).upper()

    import re
    m = re.search(r"\b[A-Z]{2,3}-\d+\b", text)
    return m.group(0) if m else ""


# --------------------------------------------------------------
# Helper 2: collect all chunks of a referenced document (for GUARANTEED inclusion)
# --------------------------------------------------------------
def collect_chunks_for_file(filename: str, meta):
    
    guaranteed = []
    if not filename:
        return guaranteed
    for idx, md in enumerate(meta["metadata"]):
        if md.get("filename", "").lower() == filename.lower():
            guaranteed.append({
                "score": 999.0,  # force it to top (top priority)
                "index": idx,
                "passage": meta["passages"][idx],
                "meta": md
            })
    return guaranteed

# --------------------------------------------------------------
# NEW retrieve() with Guaranteed Document Inclusion
# --------------------------------------------------------------
def retrieve(query: str, k: int, framework: str):
    """
    Retrieve relevant policy + control chunks using separate FAISS indexes.
    Policies always included.
    Controls filtered by framework unless ALL.
    """

    # 1️⃣ Load both indexes
    if not load_policy_index() or not load_control_index():
        return []

    policy_meta = st.session_state.policy_meta
    control_meta = st.session_state.control_meta

    # 2️⃣ Detect referenced policy filename (POLICIES ONLY)
    referenced_file = find_referenced_filename(query, policy_meta)

    # 3️⃣ Guaranteed policy chunks (only from policy metadata)
    guaranteed = collect_chunks_for_file(referenced_file, policy_meta)

    # 4️⃣ Embed query once
    qv = embed_query(query)
    overfetch = max(64, k * 8)

    # ==========================
    # 🔹 POLICY SEARCH
    # ==========================
    Dp, Ip = st.session_state.policy_index.search(qv, overfetch)

    policy_candidates = []
    for score, iid in zip(Dp[0], Ip[0]):
        if iid < 0:
            continue

        md = policy_meta["metadata"][int(iid)]
        passage = policy_meta["passages"][int(iid)]

        policy_candidates.append({
            "score": float(score),
            "index": int(iid),
            "passage": passage,
            "meta": md,
        })

    # ==========================
    # 🔹 CONTROL SEARCH
    # ==========================
    Dc, Ic = st.session_state.control_index.search(qv, overfetch)

    control_candidates = []
    for score, iid in zip(Dc[0], Ic[0]):
        if iid < 0:
            continue

        md = control_meta["metadata"][int(iid)]
        passage = control_meta["passages"][int(iid)]

        # Framework filtering
        if framework.upper() != "ALL":
            fws = md.get("frameworks") or []
            if framework.upper() not in [f.upper() for f in fws]:
                continue

        control_candidates.append({
            "score": float(score),
            "index": int(iid),
            "passage": passage,
            "meta": md,
        })

    # ==========================
    # 🔹 MERGE RESULTS
    # ==========================
    seen = set()
    results = []

    # Guaranteed policies first
    for g in guaranteed:
        if g["index"] not in seen:
            results.append(g)
            seen.add(g["index"])
            if len(results) >= k:
                return results

    # Policy candidates (sorted)
    for p in sorted(policy_candidates, key=lambda x: x["score"], reverse=True):
        if p["index"] not in seen:
            results.append(p)
            seen.add(p["index"])
            if len(results) >= k:
                return results

    # Control candidates (sorted)
    for c in sorted(control_candidates, key=lambda x: x["score"], reverse=True):
        key = f"control_{c['index']}"
        if key not in seen:
            results.append(c)
            seen.add(key)
            if len(results) >= k:
                break

    return results

# --------------------------------------------------------------
#   Build a prompt that contains a Standards block and a Policy block (policy passages first as you requested)
# --------------------------------------------------------------
def build_policy_first_prompt(query, hits, instruction=None):
    policy_block = []
    standards_block = []
    for h in hits:
        meta = h["meta"]

        if meta.get("type") == "control":
            label = format_control_label(meta)
            entry = f"Control: {label}\n{h['passage']}"
            standards_block.append(entry)
        else:
            entry = f"Source: {meta.get('filename')} (chunk={meta.get('chunk_id')})\n{h['passage']}"
            policy_block.append(entry)

    policy_text = "\n\n---\n\n".join(policy_block) if policy_block else "No policy passages retrieved."
    standards_text = "\n\n---\n\n".join(standards_block) if standards_block else "No standards passages retrieved."

    instr = (instruction.strip() + "\n\n") if instruction else ""
    prompt = (
        "You are an expert GRC analyst. Use ONLY the retrieved excerpts below to answer the user's question.\n\n"
        "Policy (top passages):\n"
        f"{policy_text}\n\n"
        "Standards (NIST) (top passages):\n"
        f"{standards_text}\n\n"
        f"User question: {query}\n\n"
        f"{instr}"
        "Compare the policy against the standards. For each relevant control/state requirement, state: (1) compliant / partially compliant / non-compliant; "
        "(2) cite evidence using (filename, chunk); (3) list remediation if partial/non-compliant. End with a short compliance summary and confidence (High/Medium/Low)."
    )
    return prompt

def ensure_gpt_model_loaded(model_dir: str, model_filename: str, device: str, n_threads: int, n_ctx: int, allow_download: bool):
    if not HAS_GPT4ALL:
        st.error("gpt4all is not installed in this environment. Install gpt4all to use local LLM generation.")
        return None
    if st.session_state.gpt_model is not None:
        return st.session_state.gpt_model
    model_path = Path(model_dir) / model_filename
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None
    st.info("Instantiating local gpt4all model (this may take a while)...")
    try:
        gm = GPT4All(model_filename, model_path=str(model_dir), device=device, n_threads=n_threads, n_ctx=n_ctx, allow_download=allow_download, verbose=False)
        st.session_state.gpt_model = gm
        st.session_state.model_loaded_info = f"{model_filename} (device={device}, threads={n_threads})"
        st.success(f"LLM loaded: {st.session_state.model_loaded_info}")
        return gm
    except Exception as e:
        st.error(f"Failed to instantiate gpt4all model: {repr(e)}")
        return None

def generate_with_llm(
    model_obj,
    prompt: str,
    n_predict: int = 512,
    temp: float = 0.1,
    top_p: float = 0.9,
):
    try:
        # Most GPT4All builds support this signature
        out = model_obj.generate(
            prompt,
            n_predict=n_predict,
            temp=temp,
            top_p=top_p,
        )
    except TypeError:
        # Fallback for older builds
        out = model_obj.generate(
            prompt,
            max_tokens=n_predict,
        )

    return out.strip()


def generate_with_mistral_api(prompt: str, temperature: float = 0.2, max_tokens: int = 500):
    """
    Calls Mistral API for text generation.
    Returns string output or raises Exception.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")

    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "mistral-small-latest",   # or mistral-medium-latest
        "messages": [
            {"role": "system", "content": "You are an expert GRC and cybersecurity compliance analyst."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Mistral API error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]

def generate_answer(prompt: str):
    backend = st.session_state.get("llm_backend")

    if backend == "Mistral API (Cloud)":
        st.info("Using Mistral API (Cloud)")
        st.session_state.model_loaded_info = "Mistral API (Cloud)"
        return generate_with_mistral_api(prompt)

    elif backend == "Mistral LLM (Local)":
        st.info("Using Local Mistral LLM (GPT4All)")

        model_obj = ensure_gpt_model_loaded(
            model_dir=str(DEFAULT_MODEL_DIR),
            model_filename=DEFAULT_MODEL_FILENAME,
            device="cuda" if USE_CUDA else "cpu",
            n_threads=THREADS,
            n_ctx=N_CTX,
            allow_download=ALLOW_DOWNLOAD,
        )

        if model_obj is None:
            raise RuntimeError("Local LLM could not be loaded")

        # ✅ HARD REQUIREMENT for Mistral GGUF
        local_prompt = wrap_for_local_mistral(prompt)

        output = generate_with_llm(
            model_obj,
            local_prompt,
            n_predict=900,      # 🔑 important
            temp=0.1,
            top_p=0.9,
        )

        # ✅ Fail loudly if model is silent
        if not output or not output.strip():
            raise RuntimeError(
                "Local LLM returned empty output. "
                "Increase n_predict or verify GGUF model integrity."
            )
        st.code(output[:1000], language="text")
        st.session_state.model_loaded_info = f"Local LLM ({DEFAULT_MODEL_FILENAME})"
        return output.strip()

    else:
        raise RuntimeError(f"Invalid LLM backend selection: {backend}")


# ---------------- UI layout ----------------
left_col, right_col = st.columns((3, 1))

with left_col:
    sample_questions = [
    "Does our policy meets NIST, ISO, PCI-DSS and HIPAA standards/regulations supported encryption?",
    "Does our identity and authentication policy meet ISO/IEC 27001 Annex A (A.5.15, A.5.16, A.8.5)?",
    "Does our policies support NIST AC-17 remote access controls?",
    "Does MFA enforced for all remote and privileged access?",
    "Does our policies meets FedRAMP IA-2 control?", 
    "Are adaptive access controls (device posture, location, risk) defined and enforced?",
    "Does our authentication policy satisfy PCI-DSS control requirements for 7.1 and 8.1?",
    "Which policies covers users identity and machines secure access?",
    "Does our encryption policy align with NIST SP 800-53 SC-12, SC-28, and KM controls?",
    "Does the policy meet HIPAA requirements for ePHI protection at data At-Rest?",
    "Are GDPR data minimization and access limitation principles covered in policies?"
    ]

    selected_question = st.selectbox("Choose a sample GRC question", sample_questions)
    user_query = st.text_area("Ask your own GRC question (policies/controls)", height=80, value=selected_question)
    extra_instruction = st.text_area("Optional additional instruction to LLM (e.g., tone, verbosity)", height=50, value="Answer in clear summary for an auditor.")
    submit = st.button("Retrieve & Explain (RAG)")

    retrieved_container = st.container()
    answer_container = st.container()
    provenance_container = st.container()
# debug_container = st.expander("Debug / Prompt", expanded=False)

#   retrieved_area = st.empty()
#   answer_area = st.empty()

with right_col:
    st.markdown("## 🧭 System & Trust Panel")

    backend = st.session_state.get("llm_backend", "Mistral API (Cloud)")

    # =========================
    # 1️⃣ LLM Runtime Summary
    # =========================
    st.markdown("### 🤖 LLM Runtime")

    if backend == "Mistral API (Cloud)":
        st.success("Backend: Mistral API (Cloud)")
        st.markdown(
            f"""
- **Model:** {DEFAULT_MISTRAL_MODEL}  
- **Context window:** Managed by provider
"""
        )

    elif backend == "Mistral LLM (Local)":
        if st.session_state.gpt_model:
            st.success("Backend: Local Mistral (GPT4All)")
            st.markdown(
                f"""
- **Model:** {DEFAULT_MODEL_FILENAME}  
- **Threads:** {THREADS}  
- **Context window:** {N_CTX} tokens  
- **Device:** {"CUDA" if USE_CUDA else "CPU"}
"""
            )
        else:
            st.warning("Local LLM not loaded")

    st.markdown("<hr style='margin:0.4rem 0'>", unsafe_allow_html=True)

    # =========================
    # 2️⃣ RAG Coverage Snapshot
    # =========================
    st.markdown("### 📄 Retrieval Coverage")

    if "last_hits" in st.session_state:
        hits = st.session_state.last_hits

        policy_cnt = sum(1 for h in hits if h["meta"].get("type") == "policy")
        control_cnt = sum(1 for h in hits if h["meta"].get("type") == "control")

        frameworks = sorted({
            h["meta"].get("framework")
            for h in hits
            if h["meta"].get("type") == "control" and h["meta"].get("framework")
        })

        st.metric("Total Chunks Used", len(hits))
        st.markdown(
            f"""
- **Policies:** {policy_cnt}  
- **Controls:** {control_cnt}  
- **Frameworks:** {", ".join(frameworks) if frameworks else "None"}
"""
        )

        if len(hits) >= 4 and backend == "Mistral LLM (Local)":
            st.warning("Context truncated for Local LLM")
    else:
        st.info("No retrieval run yet")

    st.markdown("<hr style='margin:0.4rem 0'>", unsafe_allow_html=True)

    # =========================
    # 3️⃣ Index Health & Scope
    # =========================
    st.markdown("### 🧠 Knowledge Base")

    policy_ready = POLICY_INDEX_PATH.exists() and POLICY_META_PATH.exists()
    control_ready = CONTROL_INDEX_PATH.exists() and CONTROL_META_PATH.exists()

    if policy_ready or control_ready:
        st.success("Knowledge base ready")

        # -------- Policies --------
        if policy_ready:
            with open(POLICY_META_PATH, "r", encoding="utf-8") as f:
                policy_meta = json.load(f)

            policy_docs = len({m["filename"] for m in policy_meta["metadata"]})
            policy_chunks = len(policy_meta["metadata"])

            st.markdown(
                f"""
**📄 Policies**
- Documents: {policy_docs}  
- Chunks: {policy_chunks}
"""
            )
        else:
            st.warning("Policy index missing")

        # -------- Controls --------
        if control_ready:
            with open(CONTROL_META_PATH, "r", encoding="utf-8") as f:
                control_meta = json.load(f)

            control_chunks = len(control_meta["metadata"])
            frameworks = sorted(
                {fw for m in control_meta["metadata"] for fw in (m.get("frameworks") or [])}
            )

            st.markdown(
                f"""
**📘 Controls**
- Chunks: {control_chunks}  
- Frameworks: {", ".join(frameworks) if frameworks else "Unknown"}
"""
            )
        else:
            st.warning("Control index missing")
    else:
        st.error("No FAISS indexes found")

    st.markdown("<hr style='margin:0.4rem 0'>", unsafe_allow_html=True)

    # =========================
    # 4️⃣ Actions (Meaningful)
    # =========================
    st.markdown("### ⚙️ Actions")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("🔄 Reload Index"):
            load_faiss_and_meta()
            load_embedder()
            st.success("Index reloaded")

    with col_b:
        if st.button("🧹 Clear Session"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.experimental_rerun()

    if backend == "Mistral LLM (Local)" and not st.session_state.get("gpt_model"):
        if st.button("⚡ Load Local LLM"):
            ensure_gpt_model_loaded(
                model_dir=str(DEFAULT_MODEL_DIR),
                model_filename=DEFAULT_MODEL_FILENAME,
                device="cuda" if USE_CUDA else "cpu",
                n_threads=THREADS,
                n_ctx=N_CTX,
                allow_download=ALLOW_DOWNLOAD,
            )


# ---------------- Run retrieval & (optional) LLM generate ----------------
if submit:
    # --------------------------------------------------
    # 1. Load both indexes
    # --------------------------------------------------
    if not load_policy_index() or not load_control_index():
        st.stop()

    policy_index  = st.session_state.policy_index
    policy_meta   = st.session_state.policy_meta
    control_index = st.session_state.control_index
    control_meta  = st.session_state.control_meta

    # --------------------------------------------------
    # 2. Embed query ONCE
    # --------------------------------------------------
    q_vec = embed_query(user_query)

    # --------------------------------------------------
    # 3. Detect referenced policy (if any)
    # --------------------------------------------------
    referenced_policy = find_referenced_filename(user_query, policy_meta)
#    st.info(f"DEBUG: referenced policy = {referenced_policy}")

    policy_specific = referenced_policy and is_policy_specific_question(user_query)

    # --------------------------------------------------
    # 4. Search POLICY index (overfetch for diversity)
    # --------------------------------------------------
    raw_policy_hits = search_index(
        policy_index,
        policy_meta,
        q_vec,
        k=64
    )

    # If policy-specific → enforce same file
    if policy_specific:
        raw_policy_hits = [
            h for h in raw_policy_hits
            if h["meta"]["filename"].lower() == referenced_policy.lower()
        ]

    # Guarantee referenced policy chunks (top-priority)
    guaranteed = collect_chunks_for_file(referenced_policy, policy_meta) if policy_specific else []

    # Merge guaranteed + semantic
    policy_hits = []
    seen_idx = set()

    for h in guaranteed + raw_policy_hits:
        if h["index"] not in seen_idx:
            policy_hits.append(h)
            seen_idx.add(h["index"])

    # 🔑 Diversify across policy documents
    policy_hits = diversify_by_filename(policy_hits, max_per_file=1)

    # Final cap
    policy_hits = policy_hits[:3]

    # --------------------------------------------------
    # 5. Search CONTROL index (robust, multi-framework)
    # --------------------------------------------------
    raw_control_hits = search_index(
        control_index,
        control_meta,
        q_vec,
        k=64
    )

    mentioned_controls = extract_control_ids(user_query)
    requested_frameworks = detect_frameworks_from_query(user_query)
    norm_requested = {normalize_framework(fw) for fw in requested_frameworks}

    control_hits = []
    seen = set()

    # ======================================================
    # 1️⃣ ABSOLUTE PRIORITY: DIRECT ID LOOKUP (NO FAISS)
    # ======================================================
    if mentioned_controls:
        for idx, meta in enumerate(control_meta["metadata"]):
            cid = get_control_id(meta)
            fw = meta.get("framework", "")

            if not cid or not fw:
                continue

            if cid in mentioned_controls:
                h = {
                    "index": idx,
                    "score": 1.0,  # highest possible
                    "passage": control_meta["passages"][idx],
                    "meta": meta,
                }

                key = (fw, cid)
                if key not in seen:
                    control_hits.append(h)
                    seen.add(key)

    # ======================================================
    # 2️⃣ SEMANTIC + FRAMEWORK FILTERED CONTROLS
    # ======================================================
    for h in raw_control_hits:
        if len(control_hits) >= 3:
            break

        meta = h["meta"]
        cid = get_control_id(meta)
        fw = meta.get("framework", "")

        if not cid or not fw:
            continue

        key = (fw, cid)
        if key in seen:
            continue

        if norm_requested:
            if normalize_framework(fw) not in norm_requested:
                continue

        control_hits.append(h)
        seen.add(key)


#    st.write("DEBUG mentioned controls:", mentioned_controls)


    # --------------------------------------------------
    # 6. Merge policy first, then controls
    # --------------------------------------------------
    merged = []
    seen = set()

    for h in policy_hits + control_hits:
        m = h["meta"]

        # 🔑 Different identity keys for policy vs control
        if m.get("type") == "policy":
            key = ("policy", m.get("filename"), m.get("chunk_id"))
        else:
            key = ("control", m.get("framework"), m.get("control_id"))

        if key not in seen:
            merged.append(h)
            seen.add(key)

    # --------------------------------------------------
    # 7. Context safety for Local LLM
    # --------------------------------------------------
    backend = st.session_state.get("llm_backend")
    if backend == "Mistral LLM (Local)":
        merged = merged[:4]

    st.session_state.last_hits = merged

    # --------------------------------------------------
    # 8. Render Retrieved Context
    # --------------------------------------------------
    with retrieved_container:
        st.markdown("## 🔍 Retrieved Context")

        # 🔐 FIX 3 — Explicit guardrail
        has_control = any(h["meta"].get("type") == "control" for h in merged)
        if not has_control:
            st.warning(
                "⚠️ No control text was retrieved for this query. "
                "The answer may rely on model inference rather than cited standards."
            )

        for i, h in enumerate(merged, 1):
            m = h["meta"]

            if m.get("type") == "control":
                label = format_control_label(m)
                st.markdown(
                    f"**[{i}] CONTROL → {label} | score={h['score']:.4f}**"
                )
            else:
                st.markdown(
                    f"**[{i}] {m.get('filename')} | chunk {m.get('chunk_id')} | score={h['score']:.4f}**"
                )

            st.text(h["passage"][:800])

    
    # --------------------------------------------------
    # 9. Build prompt (UNCHANGED)
    # --------------------------------------------------
    rag_prompt = build_policy_first_prompt(
        user_query,
        merged,
        instruction=extra_instruction
    )

    # --------------------------------------------------
    # 10. Generate answer
    # --------------------------------------------------
    with answer_container:
        st.markdown("## 🧠 LLM Answer")
        answer_placeholder = st.empty()

    with st.spinner("Generating answer…"):
        llm_out = generate_answer(rag_prompt)

    answer_placeholder.text_area(
        "Answer",
        value=llm_out,
        height=420,
    )

    # --------------------------------------------------
    # 11. Provenance
    # --------------------------------------------------
    with provenance_container:
        st.markdown("## 📚 Provenance")
        for h in merged:
            m = h["meta"]
            if m.get("type") == "control":
                st.write(
                    f"- CONTROL: {format_control_label(m)} | score={h['score']:.4f}"
                )
            else:
                st.write(
                    f"- POLICY: {m.get('filename')} | chunk {m.get('chunk_id')} | score={h['score']:.4f}"
                )
    
st.markdown("---")
st.caption("Notes: This demo is created with few sample policies (from templates) of ficticious organization ABC. Keep your policies & controls in protected storage. Rebuild the FAISS index after updating policies/controls.")




