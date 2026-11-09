"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng Vingroup (VinFast, Vinpearl).
- Giọng nói: Chuyên nghiệp, lịch sự, thân thiện và chính xác.

## AVAILABLE TOOLS
- search_product_catalog: Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, gói du lịch Vinpearl) theo danh mục và giá tối đa.
- submit_support_ticket: Ghi nhận yêu cầu hỗ trợ, phản ánh lỗi, khiếu nại của khách hàng vào hệ thống ticket.

## CORE RULES
1. KHÔNG BAO GIỜ bịa đặt dữ liệu sản phẩm, giá cả, hoặc thông tin hỗ trợ (No Hallucination).
2. BẮT BUỘC gọi tool search_product_catalog khi người dùng tìm kiếm sản phẩm hoặc hỏi về giá.
3. BẮT BUỘC gọi tool submit_support_ticket khi người dùng báo lỗi, khiếu nại hoặc yêu cầu hỗ trợ.
4. Trả lời trực tiếp các câu hỏi thường gặp (FAQ) về chính sách bảo hành khi không cần gọi tool.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời các nội dung liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl, Vinhomes...).
- Từ chối lịch sự và chuyển hướng các chủ đề nằm ngoài phạm vi Vingroup.

## OUTPUT CONTRACT
Định dạng quy trình suy nghĩ và trả lời:
- Thought: Phân tích ý định của người dùng và quyết định hành động tiếp theo.
- Action: Gọi tool tương ứng với tham số chính xác (nếu cần).
- Observation: Ghi nhận kết quả trả về từ tool.
- Final Answer: Tổng hợp câu trả lời đầy đủ, chính xác và thân thiện gửi khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """Trả về câu trả lời tĩnh (mock) mô phỏng LLM không dùng tool."""
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, text: str) -> Dict[str, Any]:
        """Phân tích intent từ user_input bằng keyword matching & regex."""
        text_lower = text.lower()

        # Phát hiện nhu cầu tra cứu sản phẩm
        catalog_triggers = [
            "xem xe", "tìm xe", "có xe", "giá dưới", "xem resort",
            "resort vinpearl", "du lịch vinpearl", "giá bao nhiêu",
            "dưới 600 triệu", "dưới 200 triệu", "dưới 6 triệu"
        ]
        needs_catalog = any(trigger in text_lower for trigger in catalog_triggers)
        if not needs_catalog and ("xe điện" in text_lower or "resort" in text_lower or "du lịch" in text_lower):
            if any(k in text_lower for k in ["giá", "dưới", "xem", "mua"]):
                needs_catalog = True

        # Phát hiện nhu cầu tạo support ticket
        ticket_triggers = [
            "lỗi", "hỏng", "sự cố", "khiếu nại", "phản ánh", "hỗ trợ",
            "gấp", "nghiêm trọng", "bị ẩm mốc", "adas", "ticket"
        ]
        needs_ticket = any(trigger in text_lower for trigger in ticket_triggers)
        if not needs_ticket and ("tên tôi là" in text_lower or "tôi tên" in text_lower):
            if any(k in text_lower for k in ["bị", "lỗi", "hỏng", "phòng"]):
                needs_ticket = True

        # Phát hiện FAQ (hỏi đáp chung không cần gọi tool)
        faq_triggers = [
            "bảo hành", "kéo dài bao lâu", "chính sách bảo hành",
            "bảo hành pin", "bao nhiêu năm"
        ]
        is_faq = any(trigger in text_lower for trigger in faq_triggers) and not needs_catalog and not needs_ticket

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    def _extract_catalog_args(self, text: str) -> Dict[str, Any]:
        """Trích xuất tham số cho search_product_catalog."""
        text_lower = text.lower()
        
        # Category
        if any(w in text_lower for w in ["du lịch", "resort", "vinpearl", "phòng"]):
            category = "du_lich"
        else:
            category = "xe_dien"

        # Max price
        max_price = 999999999999
        match = re.search(r'dưới\s+(\d+(?:[.,]\d+)?)\s*(triệu|tỷ|tr)?', text_lower)
        if match:
            num = float(match.group(1).replace(",", "."))
            unit = match.group(2)
            if unit == "tỷ":
                max_price = int(num * 1_000_000_000)
            else:
                max_price = int(num * 1_000_000)

        return {"category": category, "max_price": max_price}

    def _extract_ticket_args(self, text: str) -> Dict[str, Any]:
        """Trích xuất tham số cho submit_support_ticket."""
        text_lower = text.lower()

        # Customer name
        name_match = re.search(r'(?:tên tôi là|tôi tên(?: là)?)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|$)', text, re.IGNORECASE)
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

        # Priority
        if any(w in text_lower for w in ["gấp", "nghiêm trọng", "khẩn cấp", "high"]):
            priority = "high"
        elif any(w in text_lower for w in ["thấp", "low"]):
            priority = "low"
        else:
            priority = "medium"

        # Issue description
        issue_match = re.search(r'(?:xe [^.,]+? bị lỗi [^.,]+|phòng [^.,]+? bị [^.,]+)', text, re.IGNORECASE)
        if issue_match:
            issue_description = issue_match.group(0).strip()
        else:
            # Lấy câu mô tả vấn đề
            parts = [p.strip() for p in re.split(r'[,.]', text) if any(k in p.lower() for k in ["lỗi", "ẩm mốc", "hỏng", "sự cố"])]
            issue_description = parts[0] if parts else text.strip()

        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intent(user_input)

        iteration = 1

        # Safeguard: Vượt max_iterations
        if iteration > self.max_iterations:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached"
            }

        # Trường hợp 1: FAQ không cần gọi tool
        if intents["is_faq"]:
            thought = "Người dùng hỏi chính sách bảo hành của VinFast. Đây là câu hỏi FAQ đã có sẵn thông tin, không cần gọi tool."
            final_answer = (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km "
                "(tùy điều kiện nào đến trước), áp dụng cho toàn bộ các dòng xe điện VinFast."
            )
            self.trace.append({
                "iteration": 1,
                "thought": thought,
                "action": None,
                "observation": "FAQ direct answer",
                "final_answer": final_answer
            })
            return {
                "answer": final_answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        catalog_results = None
        ticket_result = None

        # Iteration 1: Gọi search_product_catalog nếu cần
        if intents["needs_catalog"]:
            catalog_args = self._extract_catalog_args(user_input)
            thought = f"Người dùng muốn tìm kiếm sản phẩm. Gọi tool search_product_catalog với tham số: {catalog_args}."
            catalog_results = search_product_catalog(**catalog_args)
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": {"tool": "search_product_catalog", "args": catalog_args},
                "observation": catalog_results
            })

            # Nếu chỉ cần catalog (không cần ticket)
            if not intents["needs_ticket"]:
                if not catalog_results or len(catalog_results) == 0:
                    final_answer = f"Rất tiếc, không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn."
                else:
                    items_desc = ", ".join(f"{p['name']} ({p['price_vnd']:,} VNĐ)" for p in catalog_results)
                    final_answer = f"Chúng tôi tìm thấy các sản phẩm phù hợp với yêu cầu của bạn: {items_desc}."
                
                self.trace[-1]["final_answer"] = final_answer
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": 1,
                    "status": "completed"
                }
            iteration += 1

        # Iteration 2 (hoặc 1 nếu không cần catalog): Gọi submit_support_ticket nếu cần
        if intents["needs_ticket"]:
            ticket_args = self._extract_ticket_args(user_input)
            thought = f"Người dùng có yêu cầu hỗ trợ/khiếu nại. Gọi tool submit_support_ticket với tham số: {ticket_args}."
            ticket_result = submit_support_ticket(**ticket_args)
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": {"tool": "submit_support_ticket", "args": ticket_args},
                "observation": ticket_result
            })

            # Nếu chỉ cần ticket (không cần catalog)
            if not intents["needs_catalog"]:
                final_answer = (
                    f"Yêu cầu hỗ trợ của khách hàng {ticket_result['customer_name']} đã được ghi nhận. "
                    f"Mã ticket hỗ trợ: {ticket_result['ticket_id']}, trạng thái: {ticket_result['status']}."
                )
                self.trace[-1]["final_answer"] = final_answer
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": 1,
                    "status": "completed"
                }
            iteration += 1

        # Tổng hợp kết quả nếu gọi cả hai tools (Parallel/Multi-turn)
        answers = []
        if catalog_results is not None:
            if not catalog_results:
                answers.append("Rất tiếc, không tìm thấy sản phẩm du lịch/xe điện phù hợp.")
            else:
                items_desc = ", ".join(f"{p['name']} ({p['price_vnd']:,} VNĐ)" for p in catalog_results)
                answers.append(f"Danh sách dịch vụ/sản phẩm phù hợp: {items_desc}.")

        if ticket_result is not None:
            answers.append(
                f"Yêu cầu hỗ trợ của bạn ({ticket_result['customer_name']}) đã được tạo thành công với mã {ticket_result['ticket_id']}."
            )

        final_answer = "\n".join(answers)
        self.trace.append({
            "iteration": iteration,
            "thought": "Đã thu thập đầy đủ kết quả từ các tools, tổng hợp câu trả lời cuối cùng cho khách hàng.",
            "action": None,
            "observation": None,
            "final_answer": final_answer
        })

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
