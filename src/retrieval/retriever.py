import yaml
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorRetriever:
    def __init__(self, cfg_path: str):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.embed = HuggingFaceEmbeddings(model_name=self.cfg["embedding"]["model"])
        self.vs = FAISS.load_local(self.cfg["vectorstore"]["path"], self.embed, allow_dangerous_deserialization=True)
        self.top_k = int(self.cfg.get("retrieval", {}).get("top_k", 8))

    def get(self, query: str, k: int = None):
        k = k or self.top_k
        return self.vs.similarity_search(query, k=k)