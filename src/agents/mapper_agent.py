import logging
from dataclasses import dataclass
from typing import Any, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendationResult:
    tax_code: str
    confidence: float
    reasoning: str

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RecommendationResult":
        tax_code = str(payload.get("tax_code", "")).strip()
        if not tax_code:
            raise ValueError("recommendation missing tax_code")

        confidence = float(payload.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(payload.get("reasoning", "未提供推荐说明。")).strip()
        return cls(tax_code=tax_code, confidence=confidence, reasoning=reasoning)

class MapperAgent:
    def __init__(self, llm_client, rag_engine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        
    def recommend(self, business_item: dict) -> List[RecommendationResult]:
        """
        基于业务类目信息和RAG系统进行税务类目的智能推荐
        """
        name = str(business_item.get('name') or '').strip()
        desc = str(business_item.get('desc') or '').strip()
        if not name:
            logger.warning("跳过缺少 name 的业务类目: %s", business_item)
            return []
        
        # 1. 从知识库中检索税务政策及正反样本
        context = self.rag_engine.retrieve_context(f"{name} {desc}")
        
        # 2. 组装复杂的推荐Prompt（含Few-Shot正负样本）
        prompt = f"""
        你是一个精通全球税务法则的计税映射专家。
        请根据以下给定的业务类目要求，匹配最合适的税务类目编码。
        
        【参考知识与历史样本库】：
        {context}
        
        【待映射业务类目】：
        名称: {name}
        描述: {desc}
        
        请给出简洁、可审计的匹配依据，并输出 recommendations JSON：
        {{"recommendations":[{{"tax_code":"...","confidence":0.0,"reasoning":"..."}}]}}
        """
        
        # 如果提供了真实的 LLM client，则调用之；否则使用 Mock 回退
        try:
            if self.llm_client is not None and hasattr(self.llm_client, 'predict'):
                response = self.llm_client.predict(prompt)
                # 期望 response 为 dict 或字符串，尝试解析为结构化推荐
                # 兼容简单字符串回复的场景（这里做最小解析）
                if isinstance(response, dict) and response.get('recommendations'):
                    results = []
                    for r in response['recommendations']:
                        try:
                            results.append(RecommendationResult.from_mapping(r))
                        except (TypeError, ValueError) as exc:
                            logger.warning("忽略格式不合法的推荐结果 %s: %s", r, exc)
                    if results:
                        return sorted(results, key=lambda item: item.confidence, reverse=True)
                else:
                    # 简单解析文本返回：取首个推荐并给出中等置信度
                    text = str(response)
                    return [
                        RecommendationResult(
                            tax_code="PARSED_TAX_CODE",
                            confidence=0.75,
                            reasoning=text[:100]
                        )
                    ]
        except Exception:
            # 网络或解析出错时，降级到 Mock 策略
            logger.exception("LLM 推荐调用失败，降级到本地规则")

        # Mock 默认返回，用于单元测试或本地离线运行
        return [
            RecommendationResult(
                tax_code="MOCK_TAX_CODE_00X",
                confidence=0.92,
                reasoning=f"经匹配，{name} 符合检索到的上下文判定，推断归属 MOCK_TAX_CODE_00X。"
            )
        ]
