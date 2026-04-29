import logging
import os
from src.agents.mapper_agent import MapperAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.core.rag_engine import RAGEngine

try:
    from src.clients.gpt4_client import GPT4Client
except Exception:
    GPT4Client = None

logger = logging.getLogger(__name__)

class MockLLMClient:
    pass

class TaxCategoryPipeline:
    def __init__(self):
        # 优先使用 GPT4Client（需设置 OPENAI_API_KEY），否则使用 MockLLMClient
        if GPT4Client is not None and os.environ.get("OPENAI_API_KEY"):
            try:
                self.llm = GPT4Client()
            except Exception as exc:
                logger.warning("GPT4Client 初始化失败，使用本地 Mock: %s", exc)
                self.llm = MockLLMClient()
        else:
            self.llm = MockLLMClient()
        self.rag = RAGEngine()
        
        self.mapper = MapperAgent(self.llm, self.rag)
        self.reviewer = ReviewerAgent(self.llm)
        
    def run_audit_task(self, historical_data: list) -> dict:
        """
        离线校验历史人工匹配数据
        """
        error_count = 0
        skipped_count = 0
        total = len(historical_data)
        if total == 0:
            logger.info("历史数据校验完成。总数:0, 无数据可校验。")
            return {"total": 0, "errors": 0, "skipped": 0, "error_rate": 0.0}

        for item in historical_data:
            tax_code = item.get('manual_tax_code')
            name = item.get('name', '<未命名类目>')
            if not tax_code:
                skipped_count += 1
                logger.warning("跳过缺少 manual_tax_code 的历史记录: %s", item)
                continue

            audit_res = self.reviewer.review_mapping(item, tax_code)
            if not audit_res.is_valid:
                error_count += 1
                logger.warning(f"识别出错配! 类目: {name}, 错配原因: {audit_res.analysis}")

        checked_count = total - skipped_count
        error_rate = (error_count / checked_count * 100) if checked_count else 0.0
        logger.info(f"历史数据校验完成。总数:{total}, 跳过:{skipped_count}, 识别错配:{error_count}, 错配率:{error_rate:.2f}%")
        return {"total": total, "errors": error_count, "skipped": skipped_count, "error_rate": error_rate}

    def run_recommendation_task(self, new_business_data: list) -> dict:
        """
        分布式调度执行新增类目的智能推荐流水线
        """
        success_count = 0
        skipped_count = 0
        fallback_count = 0
        for item in new_business_data:
            name = item.get('name', '<未命名类目>')
            # 1. 召回与推荐
            candidates = self.mapper.recommend(item)
            if not candidates:
                skipped_count += 1
                logger.warning(f"类目 {name} 未生成候选映射，跳过处理。")
                continue
            best_candidate = candidates[0]
            
            # 2. 结果二次校验防幻觉
            review_res = self.reviewer.review_mapping(item, best_candidate.tax_code)
            
            if review_res.is_valid and best_candidate.confidence > 0.85:
                # 3. 模拟写库同步
                logger.info(f"[写库流水] 类目 {name} 映射至 {best_candidate.tax_code} 成功！")
                success_count += 1
            else:
                fallback_count += 1
                logger.warning(f"类目 {name} 映射校验不通过或置信度不足，降级处理。")

        total = len(new_business_data)
        logger.info(f"新增类目推荐完成。总数:{total}, 成功:{success_count}, 跳过:{skipped_count}, 降级:{fallback_count}。")
        return {"total": total, "success": success_count, "skipped": skipped_count, "fallback": fallback_count}
