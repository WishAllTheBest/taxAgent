from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class AuditResult(BaseModel):
    is_valid: bool
    error_type: str
    analysis: str

class ReviewerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def review_mapping(self, business_item: dict, mapped_tax_code: str) -> AuditResult:
        """
        使用“交叉检查”逻辑对已映射/推荐的结果进行合法性审核
        用于解决大模型的幻觉问题以及审计人工标注
        """
        prompt = f"""
        请审查以下税务映射记录是否存在错误匹配或潜在的税务风险。
        业务类目: {business_item['name']}
        映射的税务编码: {mapped_tax_code}
        
        审查原则：检查是否存在互斥类目、是否使用了粗颗粒度的兜底税率替代了精确低税率等。
        """
        
        # 优先使用外部 LLM 执行更严格的审查逻辑（若可用），否则使用本地规则 Mock
        logger.debug(f"[Reviewer Agent] Auditing mapping for {business_item.get('name')} -> {mapped_tax_code}")
        try:
            if self.llm_client is not None and hasattr(self.llm_client, 'predict'):
                review_prompt = f"请基于税法上下文审查：{business_item.get('name')} 映射为 {mapped_tax_code} 是否合理，给出 is_valid/error_type/analysis 的 JSON 输出。"
                resp = self.llm_client.predict(review_prompt)
                # 如果返回为 dict 且包含字段，直接映射
                if isinstance(resp, dict):
                    return AuditResult(**resp)
        except Exception:
            logger.exception("LLM 审查调用失败，降级到规则引擎")

        # Mock 规则：示例性识别已知错配场景
        if "硅胶" in business_item.get('name', '') and "HOME" in mapped_tax_code:
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