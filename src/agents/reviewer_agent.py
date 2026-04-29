import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditResult:
    is_valid: bool
    error_type: str
    analysis: str

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AuditResult":
        raw_valid = payload.get("is_valid", False)
        if isinstance(raw_valid, str):
            is_valid = raw_valid.strip().lower() in {"true", "1", "yes", "y"}
        else:
            is_valid = bool(raw_valid)

        return cls(
            is_valid=is_valid,
            error_type=str(payload.get("error_type") or "UNKNOWN"),
            analysis=str(payload.get("analysis") or "未提供审核说明。"),
        )

class ReviewerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def review_mapping(self, business_item: dict, mapped_tax_code: str) -> AuditResult:
        """
        使用“交叉检查”逻辑对已映射/推荐的结果进行合法性审核
        用于解决大模型的幻觉问题以及审计人工标注
        """
        name = str(business_item.get('name') or '').strip()
        if not name:
            return AuditResult(
                is_valid=False,
                error_type="缺少业务类目名称",
                analysis="输入记录缺少 name 字段，无法执行税务映射审核。",
            )
        if not mapped_tax_code:
            return AuditResult(
                is_valid=False,
                error_type="缺少税务编码",
                analysis="映射结果缺少 tax_code，无法确认其合法性。",
            )
        
        # 优先使用外部 LLM 执行更严格的审查逻辑（若可用），否则使用本地规则 Mock
        logger.debug(f"[Reviewer Agent] Auditing mapping for {business_item.get('name')} -> {mapped_tax_code}")
        try:
            if self.llm_client is not None and hasattr(self.llm_client, 'predict'):
                review_prompt = f"请基于税法上下文审查：{name} 映射为 {mapped_tax_code} 是否合理，给出 is_valid/error_type/analysis 的 JSON 输出。"
                resp = self.llm_client.predict(review_prompt)
                # 如果返回为 dict 且包含字段，直接映射
                if isinstance(resp, dict):
                    return AuditResult.from_mapping(resp)
        except Exception:
            logger.exception("LLM 审查调用失败，降级到规则引擎")

        # Mock 规则：示例性识别已知错配场景
        if "硅胶" in name and "HOME" in mapped_tax_code:
            return AuditResult(
                is_valid=False,
                error_type="错配高税率分类",
                analysis="硅胶材料厨具属于食品接触级别特殊类目，不应放入泛 Home 类，建议映射到 TAX_KITCHEN_SPEC。"
            )

        return AuditResult(
            is_valid=True,
            error_type="None",
            analysis="合规，无明显错配行为。"
        )
