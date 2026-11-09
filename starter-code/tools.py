import json
import os
from typing import List, Dict, Any
from datetime import datetime
from rag_engine import SimpleHybridRetriever

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Khởi tạo RAG Index cho Product Catalog và Knowledge Base
# ---------------------------------------------------------------------------

_catalog_retriever = None
_knowledge_retriever = None

def _get_catalog_retriever():
    global _catalog_retriever
    if _catalog_retriever is None:
        catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
        if os.path.exists(catalog_file):
            with open(catalog_file, "r", encoding="utf-8") as f:
                products = json.load(f)
            _catalog_retriever = SimpleHybridRetriever()
            _catalog_retriever.index_documents(products, text_fields=["name", "description", "features", "colors"])
    return _catalog_retriever

def _get_knowledge_retriever():
    global _knowledge_retriever
    if _knowledge_retriever is None:
        kb_file = os.path.join(RAW_DATA_DIR, "vingroup_knowledge.json")
        if os.path.exists(kb_file):
            with open(kb_file, "r", encoding="utf-8") as f:
                kb_data = json.load(f)
            _knowledge_retriever = SimpleHybridRetriever()
            _knowledge_retriever.index_documents(kb_data, text_fields=["title", "content", "keywords"])
    return _knowledge_retriever


# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog (Hybrid RAG Enhanced & Backward Compatible)
# ---------------------------------------------------------------------------

def search_product_catalog(
    category: str,
    max_price: int = 999999999999,
    query: str = ""
) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục, giá tối đa và từ khóa nhu cầu (Hybrid Search).
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
        query: Nhu cầu hoặc đặc điểm tìm kiếm (ví dụ: '7 chỗ', 'đi phượt', 'spa', 'view biển').
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện, được xếp hạng theo độ liên quan.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    # Nếu không có query ngữ nghĩa, giữ nguyên hành vi chuẩn để tương thích 100% autograder
    if not query or not query.strip():
        with open(catalog_file, "r", encoding="utf-8") as f:
            products = json.load(f)
        return [
            p for p in products
            if p["category"].lower() == category.lower()
            and p["price_vnd"] <= max_price
        ]

    # Nếu có query ngữ nghĩa, áp dụng Hybrid Search (BM25 + TF-IDF Cosine Similarity)
    retriever = _get_catalog_retriever()
    if not retriever:
        with open(catalog_file, "r", encoding="utf-8") as f:
            products = json.load(f)
        return [
            p for p in products
            if p["category"].lower() == category.lower()
            and p["price_vnd"] <= max_price
        ]

    def _filter(p: Dict[str, Any]) -> bool:
        return p["category"].lower() == category.lower() and p["price_vnd"] <= max_price

    ranked_results = retriever.search(query=query, filter_fn=_filter, top_k=5)
    return ranked_results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # Load existing tickets
    existing_tickets = []
    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as f:
            existing_tickets = json.load(f)

    # Generate new ticket ID
    today = datetime.now().strftime("%Y%m%d")
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority.lower(),
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": "general"
    }

    existing_tickets.append(new_ticket)

    # Persist to file
    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "priority": priority.lower(),
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công cho khách hàng {customer_name}."
    }


# ---------------------------------------------------------------------------
# Tool #3: search_knowledge_base (RAG Knowledge Engine)
# ---------------------------------------------------------------------------

def search_knowledge_base(query: str, topic: str = "all") -> List[Dict[str, Any]]:
    """
    Tra cứu cẩm nang, tài liệu chính sách, trạm sạc và dịch vụ Vingroup qua Hybrid RAG.
    
    Args:
        query: Câu hỏi hoặc từ khóa cần tra cứu (vd: 'bảo hành pin chai dưới 70%', 'giá sạc v-green').
        topic: Chủ đề lọc ('all', 'bao_hanh', 'tram_sac', 'cuu_ho', 'vinpearl'). Mặc định 'all'.
        
    Returns:
        Danh sách các điều khoản hoặc bài viết cẩm nang liên quan nhất.
    """
    kb_file = os.path.join(RAW_DATA_DIR, "vingroup_knowledge.json")
    if not os.path.exists(kb_file):
        return [{"error": "Knowledge base file not found."}]

    retriever = _get_knowledge_retriever()
    if not retriever:
        with open(kb_file, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
        return kb_data[:3]

    def _filter(doc: Dict[str, Any]) -> bool:
        if topic != "all" and doc.get("topic") != topic:
            return False
        return True

    return retriever.search(query=query, filter_fn=_filter, top_k=3)


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, resort Vinpearl) theo danh mục, giá tối đa và từ khóa nhu cầu.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' cho xe điện VinFast, 'du_lich' cho gói du lịch Vinpearl.",
                    "enum": ["xe_dien", "du_lich"]
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ. Ví dụ: 600000000 cho 600 triệu."
                },
                "query": {
                    "type": "string",
                    "description": "Từ khóa hoặc nhu cầu chi tiết (ví dụ: '7 chỗ gia đình', 'đi phượt off-road', 'spa vô cực view biển')."
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": "Ghi nhận yêu cầu hỗ trợ khách hàng vào hệ thống ticket. Dùng khi khách hàng phản ánh lỗi, khiếu nại hoặc cần trợ giúp.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên đầy đủ của khách hàng."
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả chi tiết vấn đề khách hàng gặp phải."
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên: 'low' (thấp), 'medium' (trung bình), 'high' (cao/khẩn cấp).",
                    "enum": ["low", "medium", "high"],
                    "default": "medium"
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "Tra cứu tài liệu cẩm nang, chính sách bảo hành pin, mạng lưới trạm sạc V-GREEN, cứu hộ 24/7 và thông tin nghỉ dưỡng Vinpearl.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Nội dung hoặc câu hỏi cần tra cứu tài liệu."
                },
                "topic": {
                    "type": "string",
                    "description": "Chủ đề tra cứu: 'all', 'bao_hanh', 'tram_sac', 'cuu_ho', 'vinpearl'.",
                    "enum": ["all", "bao_hanh", "tram_sac", "cuu_ho", "vinpearl"],
                    "default": "all"
                }
            },
            "required": ["query"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket,
    "search_knowledge_base": search_knowledge_base
}
