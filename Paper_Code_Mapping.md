# AI-Product Manager: 论文章节与代码文件对应关系

## 📚 **论文结构与代码映射**

### **第2节: Value-Bench评估基准**
- **2.1 Task Formulation** → `evaluation_framework.py`
- **2.2 Benchmark Construction** → `benchmark/final/` + `evaluation_framework.py`
- **2.3 AI-PM评估** → `value_bench.py` + `evaluation_framework.py`

### **第3节: AI-PM Framework**
- **3.1.1 市场数据生成** → `market_intelligence_agent.py` (主要) + `perception_module.py` (基础)
- **3.1.2 用户洞察生成** → `user_insight_generator.py`
- **3.2.1 多阶段产品开发优化架构** → `decision_module.py`
- **3.2.2 产品定义与设计框架** → `product_definition_agent.py`
- **3.2.3 市场与用户验证框架** → `product_strategy_advisor.py`
- **3.2.4 渐进式产品验证循环** → `execution_module.py`
- **3.3 自动化产品策略文档撰写** → `product_documentation_agent.py` + `interaction_module.py`

### **第4节: Experiments**
- **4.1-4.7 六个研究问题评估** → `evaluation_framework.py`

### **系统集成**
- **完整AI-PM框架** → `aipm_orchestrator.py` (新框架) + `framework.py` (传统框架)
- **Web界面** → `web_ai_product_manager.py`
- **主程序** → `main_ai_researcher.py`

## 🎯 **文件重要性**
- **核心实现**: `market_intelligence_agent.py`, `user_insight_generator.py`, `product_definition_agent.py`, `evaluation_framework.py`, `aipm_orchestrator.py`
- **评估系统**: `value_bench.py`, `benchmark/final/`
- **传统框架**: `framework.py`, `perception_module.py`, `decision_module.py`, `execution_module.py`, `learning_module.py`, `interaction_module.py`
- **应用界面**: `web_ai_product_manager.py`