import yaml
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

def retrieve_then_rerank(query: str, cfg_path: str, topn: int = None):
    cfg = yaml.safe_load(open(cfg_path))
    vr = _get_vr(cfg_path)
    k0 = cfg["retrieval"].get("top_k", 8)
    need = max(topn or k0, k0)
    docs = vr.get(query, k=need)
    rr = _get_rr(cfg_path)
    if rr is not None:
        docs = rr.apply(query, docs)
    if topn is not None:
        docs = docs[:topn]
    return docs
