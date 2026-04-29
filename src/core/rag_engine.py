class RAGEngine:
    def __init__(self):
        # 实际项目中这里会初始化 VectorStore (如 Chroma/Milvus) 以及 Embedding 模型
        self.vector_db = []
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        加载包含：不同地区的计税类目映射规则、历史优质人工匹配结果(正样本)、人工错配纠正历史(负样本)
        """
        # 这里使用简单的内存列表做示例数据，真实环境应接入向量数据库与 embeddings
        self.vector_db = [
            {
                'id': 'kb_001',
                'text': '纯棉/毛绒服装映射指南：要求必须映射至 TEXTILE_COTTON 分支，基准税率5%。',
                'tags': ['textile', 'cotton', 'positive']
            },
            {
                'id': 'kb_002',
                'text': '含内置电池的玩具应判定为电子消费品，避免映射到玩具低税率类目。',
                'tags': ['electronics', 'toy', 'negative']
            },
            {
                'id': 'kb_003',
                'text': '欧洲区对泛用型电子配件税率较高，建议尽量拆分类目以匹配具体设备。',
                'tags': ['eu', 'electronics', 'policy']
            }
        ]

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        根据查询，检索最相关的税务知识上下文及对比样本
        """
        # 简单基于关键词进行打分并返回 top_k 文本片段（示例）
        q = query.lower()
        scored = []
        for item in self.vector_db:
            score = 0
            text_lower = item['text'].lower()
            if any(tok in q for tok in ['棉', '纯棉', '服装']):
                if 'textile' in item['tags']:
                    score += 2
            if any(tok in q for tok in ['电池', '电子', '玩具']):
                if 'electronics' in item['tags']:
                    score += 2
            if 'eu' in item['tags'] and any(tok in q for tok in ['欧洲', '欧盟', 'eu']):
                score += 1
            # 基础包含词匹配
            if any(w in text_lower for w in q.split() if len(w) > 2):
                score += 1
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_texts = [f"- {it['text']}" for s, it in scored[:top_k] if s > 0]
        if not top_texts:
            # 返回默认上下文
            return "无高相关知识，返回通用检索提示。"
        return "\n".join(top_texts)