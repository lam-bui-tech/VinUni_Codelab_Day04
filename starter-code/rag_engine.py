"""
Zero-dependency Hybrid RAG & Semantic Retrieval Engine (Pure Python).
Hỗ trợ BM25, TF-IDF và Cosine Similarity, không cần thư viện bên ngoài.
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional


def tokenize(text: str) -> List[str]:
    """Tách từ, chuẩn hoá chữ thường và loại bỏ ký tự đặc biệt."""
    if not text:
        return []
    # Giữ chữ cái tiếng Việt, số, và ký tự chữ
    tokens = re.findall(r'[\w]+', text.lower())
    # Loại bỏ stop words cơ bản nếu có
    return tokens


class SimpleHybridRetriever:
    """Bộ máy tìm kiếm Hybrid kết hợp BM25 + Vector TF-IDF Cosine Similarity."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.tfidf_vectors: List[Dict[str, float]] = []

    def index_documents(self, documents: List[Dict[str, Any]], text_fields: List[str]):
        """Đánh chỉ mục danh sách documents dựa trên các trường văn bản chỉ định."""
        self.documents = documents
        self.doc_tokens = []
        self.doc_lens = []
        self.df = defaultdict(int)

        N = len(documents)
        if N == 0:
            return

        for doc in documents:
            combined_text = []
            for field in text_fields:
                val = doc.get(field, "")
                if isinstance(val, list):
                    combined_text.append(" ".join(str(v) for v in val))
                elif isinstance(val, str):
                    combined_text.append(val)
                elif val is not None:
                    combined_text.append(str(val))
            
            full_text = " ".join(combined_text)
            tokens = tokenize(full_text)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))

            # Tính Document Frequency (DF)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.df[t] += 1

        self.avg_dl = sum(self.doc_lens) / max(N, 1)

        # Tính IDF chuẩn BM25
        self.idf = {}
        for term, freq in self.df.items():
            # BM25 IDF formulation
            self.idf[term] = math.log(1.0 + (N - freq + 0.5) / (freq + 0.5))

        # Tính TF-IDF Vector và chuẩn hoá vector
        self.tfidf_vectors = []
        for tokens, doc_len in zip(self.doc_tokens, self.doc_lens):
            tf = Counter(tokens)
            vec = {}
            for t, count in tf.items():
                w = (count / max(doc_len, 1)) * self.idf.get(t, 1.0)
                vec[t] = w
            
            # Normalize vector (L2 norm)
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {k: v / norm for k, v in vec.items()}
            self.tfidf_vectors.append(vec)

    def _score_bm25(self, query_tokens: List[str], doc_idx: int) -> float:
        """Tính điểm BM25 cho một document."""
        score = 0.0
        doc_tokens = self.doc_tokens[doc_idx]
        tf_map = Counter(doc_tokens)
        dl = self.doc_lens[doc_idx]

        for token in query_tokens:
            if token not in tf_map:
                continue
            tf = tf_map[token]
            idf = self.idf.get(token, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avg_dl, 1e-6)))
            score += idf * (numerator / max(denominator, 1e-6))
        return score

    def _score_cosine(self, query_tokens: List[str], doc_idx: int) -> float:
        """Tính Cosine Similarity giữa Query TF-IDF vector và Document TF-IDF vector."""
        if not query_tokens:
            return 0.0
        
        q_tf = Counter(query_tokens)
        q_vec = {}
        for t, count in q_tf.items():
            idf = self.idf.get(t, 1.0)
            q_vec[t] = count * idf
        
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return 0.0
        q_vec = {k: v / q_norm for k, v in q_vec.items()}

        doc_vec = self.tfidf_vectors[doc_idx]
        dot_product = sum(q_vec[t] * doc_vec[t] for t in q_vec if t in doc_vec)
        return dot_product

    def search(
        self,
        query: str,
        filter_fn: Optional[Any] = None,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm Hybrid kết hợp:
        Score = alpha * BM25_normalized + (1 - alpha) * Cosine_Similarity
        """
        if not self.documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            # Nếu không có query text, trả về các doc thỏa mãn filter_fn
            filtered = []
            for doc in self.documents:
                if filter_fn is None or filter_fn(doc):
                    filtered.append(doc)
            return filtered[:top_k]

        candidates = []
        raw_bm25_scores = []
        cosine_scores = []

        for idx, doc in enumerate(self.documents):
            if filter_fn and not filter_fn(doc):
                continue
            
            bm25 = self._score_bm25(query_tokens, idx)
            cosine = self._score_cosine(query_tokens, idx)
            candidates.append((doc, bm25, cosine))
            raw_bm25_scores.append(bm25)
            cosine_scores.append(cosine)

        if not candidates:
            return []

        # Chuẩn hoá điểm BM25 về [0, 1]
        max_bm25 = max(raw_bm25_scores) if raw_bm25_scores else 0.0
        
        scored_results = []
        for doc, bm25, cosine in candidates:
            norm_bm25 = (bm25 / max_bm25) if max_bm25 > 0 else 0.0
            hybrid_score = alpha * norm_bm25 + (1.0 - alpha) * cosine
            scored_results.append((doc, hybrid_score))

        # Sắp xếp giảm dần theo hybrid_score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_results[:top_k]]
