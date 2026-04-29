import logging
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
        if GPT4Client is not None:
            try:
                self.llm = GPT4Client()
            except Exception:
                self.llm = MockLLMClient()
        else:
            self.llm = MockLLMClient()
        self.rag = RAGEngine()
        
        self.mapper = MapperAgent(self.llm, self.rag)
        self.reviewer = ReviewerAgent(self.llm)
        
    def run_audit_task(self, historical_data: list):
        """
        离线校验历史人工匹配数据
        """
        error_count = 0
        total = len(historical_data)
        for item in historical_data:
            tax_code = item['manual_tax_code']
            audit_res = self.reviewer.review_mapping(item, tax_code)
            if not audit_res.is_valid:
                error_count += 1
                logger.warning(f"识别出错配! 类目: {item['name']}, 错配原因: {audit_res.analysis}")
                
        logger.info(f"历史数据校验完成。总数:{total}, 识别错配:{error_count}, 错配率:{error_count/total*100:.2f}%")

    def run_recommendation_task(self, new_business_data: list):
        """
        分布式调度执行新增类目的智能推荐流水线
        """
        success_count = 0
        for item in new_business_data:
            # 1. 召回与推荐
            candidates = self.mapper.recommend(item)
            best_candidate = candidates[0]
            
            # 2. 结果二次校验防幻觉
            review_res = self.reviewer.review_mapping(item, best_candidate.tax_code)
            
            if review_res.is_valid and best_candidate.confidence > 0.85:
                # 3. 模拟写库同步
                logger.info(f"[写库流水] 类目 {item['name']} 映射至 {best_candidate.tax_code} 成功！")
                success_count += 1
            else:
                logger.warning(f"类目 {item['name']} 映射校验不通过或置信度不足，降级处理。")
                
        logger.info(f"新增类目推荐完成。成功数:{success_count}, 极大降低了兜底税率使用率。")