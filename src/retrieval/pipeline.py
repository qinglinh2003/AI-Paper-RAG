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

def _split_sentences(text: str):
    text = (text or "").strip()
    if not text: return []
    parts = _sent_splitter.split(text)
    return [p.strip() for p in parts if p.strip()]

def _score_sentences_simple(query: str, sentences):
    q = set(re.findall(r"[A-Za-z0-9\-\_]+", (query or "").lower()))
    scores = []
    for s in sentences:
        toks = set(re.findall(r"[A-Za-z0-9\-\_]+", s.lower()))
        overlap = len(q & toks)
        bonus = sum(1 for t in q if t and t in s) 
        scores.append(overlap + 0.1 * bonus)
    return scores

def _pick_chunk_by_best_sentence(query: str, cand_docs):
    best_doc, best_score = None, float("-inf")
    for d in cand_docs:
        sents = _split_sentences(d.page_content)
        if not sents:
            continue
        scores = _score_sentences_simple(query, sents)
        s = max(scores) if scores else 0.0
        if s > best_score:
            best_score, best_doc = s, d
    return best_doc if best_doc is not None else (cand_docs[0] if cand_docs else None)

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

    if mode == "accuracy":
        docs = _cap_per_source(docs, cap=1)
        cands = docs[:rk]
        if cands:
            best = _pick_chunk_by_best_sentence(query, cands)
            docs = [best] if best is not None else cands[:1]
        cutoff = topn if topn is not None else 1
        return docs[:cutoff]

    return docs[: (topn if topn is not None else rk)]


