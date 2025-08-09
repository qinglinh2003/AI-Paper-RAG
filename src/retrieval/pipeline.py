# src/retrieval/pipeline.py
import yaml, re
from functools import lru_cache
from .retriever import VectorRetriever
from .reranker import BGEReranker

@lru_cache(maxsize=4)
def _get_vr(cfg_path: str):
    return VectorRetriever(cfg_path)

@lru_cache(maxsize=4)
def _get_rr(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    if cfg.get("reranker", {}).get("enabled", False):
        return BGEReranker(cfg_path)
    return None

def _cap_per_source(docs, cap=1):
    seen, out = {}, []
    for d in docs:
        src = (d.metadata.get("source") or d.metadata.get("path") or "").lower()
        seen[src] = seen.get(src, 0) + 1
        if seen[src] <= cap:
            out.append(d)
    return out

_sent_splitter = re.compile(r'(?<=[\.!?。！？])\s+')
_tok = re.compile(r"[A-Za-z0-9\-\_]+")

def _split_sentences(text: str):
    text = (text or "").strip()
    if not text: return []
    return [p.strip() for p in _sent_splitter.split(text) if p.strip()]

def _best_sent_score_and_cov(query: str, text: str):
    qset = set(_tok.findall((query or "").lower()))
    sents = _split_sentences(text)
    best = 0.0
    alltok = set()
    for s in sents:
        toks = set(_tok.findall(s.lower()))
        alltok |= toks
        overlap = len(qset & toks)
        bonus = sum(1 for t in qset if t in toks)
        sc = overlap + 0.1 * bonus
        if sc > best: best = sc
    cov = (len(qset & alltok) / max(1, len(qset))) if qset else 0.0
    return best, cov

def retrieve_then_rerank(query: str, cfg_path: str, topn: int = None, mode: str = "accuracy"):
    cfg = yaml.safe_load(open(cfg_path))
    vr = _get_vr(cfg_path)
    rr = _get_rr(cfg_path)

    k0 = int(cfg["retrieval"].get("top_k", 60))
    need = max(topn or k0, k0)
    docs = vr.get(query, k=need)

    if rr is not None:
        docs = rr.apply(query, docs)

    rk = int(cfg["retrieval"].get("rerank_k", 8))

    if mode == "recall":
        limit = topn if topn is not None else max(10, rk)
        return docs[:limit]

    cap = int(cfg["retrieval"].get("diversity_cap", 1))
    docs = _cap_per_source(docs, cap=cap)
    cands = docs[:rk]

    adp = cfg["retrieval"].get("adaptive", {})
    use_adp = bool(adp.get("enabled", True))
    if not use_adp or not cands:
        cutoff = topn if topn is not None else 1
        return cands[:cutoff]

    scored = []
    for d in cands:
        s, cov = _best_sent_score_and_cov(query, d.page_content or "")
        scored.append((s, cov, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    s1, cov1, d1 = scored[0]
    s2 = scored[1][0] if len(scored) > 1 else 0.0

    tau_abs = float(adp.get("abs", 1.5))
    tau_gap = float(adp.get("gap", 0.5))
    tau_cov = float(adp.get("cov", 0.5))
    maxk = int(cfg["app"].get("max_context_docs", 3))

    K = 1
    if (s1 < tau_abs) or (s1 - s2 < tau_gap) or (cov1 < tau_cov):
        K = min(maxk, 2)
        if s1 < 0.5 and cov1 < 0.3:
            K = min(maxk, 3)

    selected = [t[2] for t in scored[:K]]
    cutoff = topn if topn is not None else K
    return selected[:cutoff]
