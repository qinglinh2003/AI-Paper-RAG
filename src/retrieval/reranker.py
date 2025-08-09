import yaml, torch
from FlagEmbedding import FlagReranker

class Reranker:
    def score(self, query, docs): ...
    def apply(self, query, docs):
        if not docs: return docs
        s = self.score(query, docs)
        return [d for d,_ in sorted(zip(docs, s), key=lambda x: x[1], reverse=True)]

class NoOpReranker(Reranker):
    def score(self, query, docs): return [0]*len(docs)

class BGEReranker(Reranker):
    def __init__(self, cfg_path: str):
        cfg = yaml.safe_load(open(cfg_path))
        m = cfg.get("reranker", {}).get("model")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rr = FlagReranker(m, use_fp16=(device=="cuda"), device=device)
    def score(self, query, docs):
        pairs = [[query, d.page_content] for d in docs]
        return self.rr.compute_score(pairs, normalize=True)
