"""
Unit Tests for Advanced Hybrid RAG & Semantic Retrieval Engine.
Kiểm tra các tính năng RAG mở rộng:
  - Semantic Search cho Catalog sản phẩm
  - Tra cứu Knowledge Base cẩm nang & chính sách Vingroup
"""

import sys
import os
import pytest

STARTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "starter-code"))
if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)

from tools import search_product_catalog, search_knowledge_base
from template import ToolCallingAgent


def test_rag_semantic_search_7_cho():
    """Tìm kiếm xe 7 chỗ cho gia đình bằng Semantic Search."""
    results = search_product_catalog(category="xe_dien", query="xe 7 chỗ gia đình")
    assert len(results) > 0
    # VF 9 Plus là xe 7 chỗ, phải được xếp đầu tiên
    assert results[0]["name"] == "VinFast VF 9 Plus"
    assert "7 chỗ" in results[0]["features"] or "7 chỗ" in results[0]["description"]


def test_rag_semantic_search_off_road():
    """Tìm kiếm xe bán tải địa hình off-road."""
    results = search_product_catalog(category="xe_dien", query="bán tải off-road vượt địa hình")
    assert len(results) > 0
    # VF Wild là xe bán tải off-road
    assert results[0]["name"] == "VinFast VF Wild"


def test_rag_semantic_search_resort_spa():
    """Tìm kiếm resort có spa và hồ bơi vô cực."""
    results = search_product_catalog(category="du_lich", query="spa hồ bơi vô cực")
    assert len(results) > 0
    assert results[0]["name"] == "Vinpearl Resort & Spa Nha Trang"


def test_rag_knowledge_base_tram_sac():
    """Tra cứu trạm sạc V-GREEN từ Knowledge Base."""
    results = search_knowledge_base(query="chi phí và cổng sạc nhanh trạm v-green", topic="tram_sac")
    assert len(results) > 0
    top_doc = results[0]
    assert "V-GREEN" in top_doc["title"] or "trạm sạc" in top_doc["title"].lower()
    assert "3.858" in top_doc["content"] or "V-GREEN" in top_doc["content"]


def test_rag_knowledge_base_cuu_ho_247():
    """Tra cứu dịch vụ cứu hộ pin khẩn cấp 24/7."""
    results = search_knowledge_base(query="xe bị cạn pin trên đường cần cứu hộ gấp", topic="cuu_ho")
    assert len(results) > 0
    top_doc = results[0]
    assert "cứu hộ" in top_doc["title"].lower()
    assert "1900-232389" in top_doc["content"]


def test_agent_end_to_end_semantic_rag():
    """Agent trả lời câu hỏi tìm xe 7 chỗ dựa trên Semantic RAG."""
    agent = ToolCallingAgent(max_iterations=5)
    res = agent.run("Tôi cần tìm xe điện 7 chỗ rộng rãi cho gia đình đi du lịch.")
    assert res["status"] == "completed"
    assert "VF 9" in res["answer"]
    assert len(res["trace"]) > 0
