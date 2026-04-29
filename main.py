import time
import logging
from src.pipeline import TaxCategoryPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logger = logging.getLogger(__name__)
    logger.info("Initializing Smart Tax Category Recommendation System...")
    
    # 模拟环境配置加载
    pipeline = TaxCategoryPipeline()
    
    # 场景1：离线校验历史人工匹配数据
    logger.info("--- 任务一: 校验历史 3.77 万条人工匹配 ---")
    mock_historical_data = [
        {"business_id": "B001", "name": "硅胶厨具套装", "manual_tax_code": "TAX_HOME_001"},
        {"business_id": "B002", "name": "锂电池电动玩具", "manual_tax_code": "TAX_TOY_005"}
    ]
    pipeline.run_audit_task(mock_historical_data)
    
    # 场景2：处理新增业务类目映射任务
    logger.info("--- 任务二: 处理新增 3.1 万条业务类目推荐 ---")
    mock_new_data = [
        {"business_id": "N001", "name": "蓝牙降噪耳机", "desc": "消费电子类，带内接电池"},
        {"business_id": "N002", "name": "纯棉婴儿连体衣", "desc": "母婴服装，A类棉"}
    ]
    pipeline.run_recommendation_task(mock_new_data)
    
    logger.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()