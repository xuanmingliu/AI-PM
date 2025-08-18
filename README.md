# 🤖 AI-Product Manager

**自主科学创新的智能产品经理系统**

AI-Product Manager是一个完全自主的产品管理系统，通过五个核心模块（感知、决策、执行、学习、交互）实现端到端的智能产品管理，从需求洞察到商业落地形成闭环创新。

## 📖 项目概述

### 🎯 核心功能

本系统实现了论文《AI-Product Manager: Autonomous Scientific Innovation》中描述的完整框架：

- **🔍 感知模块**: 多渠道数据收集、用户需求洞察、市场分析
- **🎯 决策模块**: 任务优先级排序、场景选择、模型选型、商业模式设计  
- **⚡ 执行模块**: 数据集构建、产品迭代、营销活动执行
- **🧠 学习模块**: 强化学习优化、Bad Case分析、持续改进
- **💬 交互模块**: 自然语言交互、可视化报告、系统集成

### 🏗️ 系统架构

```
AI-Product Manager
├── aipm_core/                 # 核心框架
│   ├── framework.py          # 主框架
│   ├── perception_module.py  # 感知模块
│   ├── decision_module.py    # 决策模块
│   ├── execution_module.py   # 执行模块
│   ├── learning_module.py    # 学习模块
│   └── interaction_module.py # 交互模块
├── web_ai_product_manager.py # Web界面
├── evaluation_framework.py   # 评估框架
└── README_AI_Product_Manager.md
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd AI-Researcher

# 创建虚拟环境
python -m venv aipm_env
source aipm_env/bin/activate  # Linux/Mac
# 或
aipm_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 依赖包安装

```bash
pip install asyncio gradio pandas numpy matplotlib seaborn
pip install python-dotenv logging datetime typing dataclasses
pip install requests os base64 threading queue time
```

### 3. 环境配置

创建 `.env` 文件并配置API密钥：

```bash
# 复制环境变量模板
cp .env_template .env

# 编辑环境变量
nano .env
```

在 `.env` 文件中配置：

```bash
# AI模型API密钥
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY="sk-ant-oat01-d64c7a572221d1206bb704d279e6fce595e880b60316fa591d81b0ab47aa1e4e"
DEEPSEEK_API_KEY=your_deepseek_api_key
QWEN_API_KEY=your_qwen_api_key

# Google服务API
GOOGLE_API_KEY=your_google_api_key
SEARCH_ENGINE_ID=your_search_engine_id

# 其他服务API
CHUNKR_API_KEY=your_chunkr_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

### 4. 启动系统

#### 方式一：Web界面模式（推荐）

```bash
python web_ai_product_manager.py
```

访问 `http://localhost:7040` 使用Web界面

#### 方式二：命令行模式

```bash
python -c "
import asyncio
from aipm_core.framework import AIPMFramework

async def main():
    config = {
        'perception': {'data_sources': {'user_feedback': True}},
        'decision': {'priority_weights': {'urgency': 0.3, 'impact': 0.4}},
        'execution': {'max_concurrent_tasks': 5},
        'learning': {'learning_rate': 0.01},
        'interaction': {'visualization_output_dir': './visualizations'}
    }
    
    aipm = AIPMFramework(config)
    await aipm.start()

asyncio.run(main())
"
```

#### 方式三：评估模式

```bash
python evaluation_framework.py
```

## 📊 使用指南

### Web界面操作

1. **🎯 智能控制台**
   - 选择任务类型：需求分析、场景选择、模型选型、商业分析、优化建议、综合分析
   - 输入详细的任务描述
   - 点击"执行任务"获得AI分析结果

2. **📊 实时仪表板**
   - 查看系统运行状态
   - 监控关键性能指标
   - 实时性能图表展示

3. **⚙️ 环境配置**
   - 管理API密钥
   - 配置系统参数
   - 查看配置状态

4. **🧪 实验评估**
   - 运行论文中的评估实验
   - 对比AI与人类专家表现
   - 测试开放式创新能力

### 任务类型说明

#### 🔍 需求分析
```python
# 示例输入
"分析智能客服系统的用户需求，识别核心痛点和改进机会"

# 输出包含
- 用户需求洞察
- 痛点识别与分析
- 优先级排序建议
- 功能设计建议
```

#### 🎯 场景选择
```python
# 示例输入
"为AI推荐系统选择最适合的应用场景"

# 输出包含
- 可选场景评估
- 市场潜力分析
- 技术可行性评估
- 推荐场景及理由
```

#### 🔧 模型选型
```python
# 示例输入
"为自然语言处理任务选择最优的大语言模型"

# 输出包含
- 模型对比分析
- 性能评估
- 成本效益分析
- 推荐方案
```

#### 💼 商业分析
```python
# 示例输入
"分析在线教育AI产品的商业模式和盈利策略"

# 输出包含
- 市场分析
- 商业模式设计
- ROI预测
- 实施路线图
```

#### 🎯 优化建议
```python
# 示例输入
"优化电商推荐系统的性能和用户体验"

# 输出包含
- 当前状态分析
- 优化机会识别
- 改进方案设计
- 实施优先级
```

#### 📈 综合分析
```python
# 示例输入
"对医疗AI产品进行全面的产品策略分析"

# 输出包含
- 多维度分析
- 产品洞察
- 战略建议
- 行动计划
```

## 🧪 实验评估

### 复现论文实验

系统完整实现了论文中的三个核心研究问题：

#### RQ1: 方法实现完整性和正确性
```bash
python evaluation_framework.py --eval-type completeness
```

**评估维度：**
- 流程完整性（需求→决策→执行→迭代）
- 决策正确性（与专家共识匹配度）
- 执行效率（任务完成时间对比）

#### RQ2: AI vs 人类专家对比
```bash
python evaluation_framework.py --eval-type comparison
```

**对比指标：**
- 技术可行性评估准确性
- 商业价值分析质量
- 创新性解决方案提出
- 执行效率对比

#### RQ3: 开放式探索能力
```bash
python evaluation_framework.py --eval-type innovation
```

**评估内容：**
- 创新挑战："如何用AI改善教育公平？"
- 跨领域整合："设计AI+IoT+区块链智慧城市方案"
- 科学假设生成
- 认知能力测试

### 性能基准

根据论文实验结果，系统达到以下性能指标：

- **需求识别准确率**: 提升42%
- **技术方案可行性**: 成功率82%
- **商业文档质量**: 接近人类专家水平
- **创新能力**: 超越指导性任务的自主探索

## 📁 项目结构详解

```
AI-Researcher/
├── aipm_core/                    # 核心框架模块
│   ├── __init__.py
│   ├── framework.py              # 主框架类
│   ├── perception_module.py      # 感知模块实现
│   ├── decision_module.py        # 决策模块实现
│   ├── execution_module.py       # 执行模块实现
│   ├── learning_module.py        # 学习模块实现
│   └── interaction_module.py     # 交互模块实现
├── web_ai_product_manager.py     # Web用户界面
├── evaluation_framework.py       # 评估和实验框架
├── requirements.txt              # Python依赖包
├── .env_template                 # 环境变量模板
├── CLAUDE.md                     # Claude Code配置
└── README_AI_Product_Manager.md  # 项目说明文档
```

### 核心模块功能

#### 🔍 感知模块 (perception_module.py)
- **多渠道数据收集**: 用户反馈、市场数据、社交媒体、支持工单
- **需求洞察**: NLP处理用户反馈，提取痛点和需求
- **市场分析**: 竞品分析、行业趋势、用户画像
- **Bad Case检测**: 自动识别系统问题和异常

#### 🎯 决策模块 (decision_module.py)
- **任务优先级排序**: 基于紧迫性、影响力、可行性的智能排序
- **场景选择**: 多维度评估最优落地场景
- **模型选型**: 综合性能、成本、可解释性的模型推荐
- **商业模式设计**: 订阅、按使用付费、免费增值等模式分析
- **评价体系构建**: 自动化指标体系设计

#### ⚡ 执行模块 (execution_module.py)
- **任务执行引擎**: 异步任务调度和执行
- **数据集构建**: 自动化数据收集、清洗、标注
- **模型训练**: 模拟机器学习模型训练流程
- **部署管理**: 模型部署和服务管理
- **营销活动**: 邮件、社交媒体、广告投放执行

#### 🧠 学习模块 (learning_module.py)
- **强化学习**: Q-learning实现决策优化
- **经验回放**: 历史经验存储和学习
- **Bad Case分析**: 问题模式识别和改进建议
- **持续优化**: 基于反馈的系统自我改进
- **规则更新**: 动态优化规则库维护

#### 💬 交互模块 (interaction_module.py)
- **自然语言处理**: 意图识别、实体提取、情感分析
- **可视化引擎**: 自动图表生成和仪表板
- **报告生成**: HTML/PDF格式的综合报告
- **系统集成**: API接口和外部系统对接
- **用户会话管理**: 多用户对话状态维护

## ⚙️ 配置说明

### 系统配置

在代码中可以通过配置字典自定义系统行为：

```python
config = {
    'perception': {
        'data_sources': {
            'user_feedback': True,
            'market_data': True,
            'social_media': True,
            'support_tickets': True
        },
        'nlp_model': 'gpt-3.5-turbo'
    },
    'decision': {
        'priority_weights': {
            'urgency': 0.3,
            'impact': 0.4,
            'feasibility': 0.2,
            'cost_efficiency': 0.1
        },
        'scenario_weights': {
            'market_potential': 0.25,
            'user_demand': 0.25,
            'technical_feasibility': 0.20,
            'competitive_advantage': 0.15,
            'roi_potential': 0.15
        }
    },
    'execution': {
        'max_concurrent_tasks': 5,
        'data_storage_path': './aipm_data',
        'model_storage_path': './aipm_models'
    },
    'learning': {
        'learning_rate': 0.01,
        'discount_factor': 0.95,
        'exploration_rate': 0.1,
        'max_experiences': 10000
    },
    'interaction': {
        'visualization_output_dir': './aipm_visualizations',
        'report_output_dir': './aipm_reports'
    }
}
```

### 环境变量配置

支持通过环境变量灵活配置：

```bash
# 模型配置
COMPLETION_MODEL=claude-3-5-sonnet-20241022
CHEAP_MODEL=claude-3-5-haiku-20241022

# 系统配置
UPDATE_INTERVAL=3600
AUTO_REPORT_INTERVAL=86400
MAX_CONCURRENT_TASKS=5

# 数据存储
DATA_STORAGE_PATH=./aipm_data
MODEL_STORAGE_PATH=./aipm_models
VISUALIZATION_OUTPUT_DIR=./aipm_visualizations
REPORT_OUTPUT_DIR=./aipm_reports
```

## 🔧 开发指南

### 扩展新功能

#### 添加新的决策算法

```python
# 在 decision_module.py 中扩展
class AdvancedDecisionModule(DecisionModule):
    def custom_algorithm(self, data):
        # 实现自定义决策算法
        pass
```

#### 添加新的数据源

```python
# 在 perception_module.py 中扩展
async def collect_new_data_source(self):
    # 实现新数据源接入
    pass
```

#### 自定义学习策略

```python
# 在 learning_module.py 中扩展
class CustomLearningStrategy:
    def optimize(self, experience_data):
        # 实现自定义学习策略
        pass
```

### API接口

系统提供REST API接口：

```python
# 启动API服务器
from fastapi import FastAPI
from aipm_core.framework import AIPMFramework

app = FastAPI()
aipm = AIPMFramework(config)

@app.post("/api/analyze")
async def analyze_requirement(request_data: dict):
    result = await aipm.process_user_request(
        user_id=request_data["user_id"],
        request=request_data["description"], 
        request_type=request_data["type"]
    )
    return result
```

### 自定义评估

```python
# 添加新的评估任务
custom_task = EvaluationTask(
    task_id="custom_001",
    task_type="自定义分析",
    description="自定义任务描述",
    expected_output={"analysis": "分析结果"},
    evaluation_criteria=["quality", "innovation"],
    difficulty_level="medium",
    time_limit=1800
)

framework = EvaluationFramework(config)
framework.evaluation_tasks.append(custom_task)
```

## 📊 监控和调试

### 日志配置

系统自动生成详细日志：

```
logs/
├── aipm_log_20241201_120000.log  # 系统运行日志
├── evaluation_results/           # 评估结果
└── performance_metrics/          # 性能指标
```

### 性能监控

通过Web界面实时监控：

- 系统运行状态
- 模块性能指标
- 任务执行情况
- 用户反馈统计
- 学习进度跟踪

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 单步调试模式
aipm_framework.debug_mode = True

# 性能分析
import cProfile
cProfile.run('aipm_framework.process_user_request(...)')
```

## 🤝 贡献指南

### 代码贡献

1. Fork 项目仓库
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交Pull Request

### 问题报告

请通过GitHub Issues报告问题，包含：

- 系统环境信息
- 错误日志详情
- 复现步骤描述
- 预期行为说明

### 功能建议

欢迎提出新功能建议：

- 描述使用场景
- 说明预期价值
- 提供实现思路
- 考虑兼容性影响

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢所有贡献者的支持
- 基于开源社区的优秀项目构建
- 参考了相关学术研究成果

## 📞 联系方式

- 项目主页: [GitHub Repository]
- 技术支持: [Issues Page]
- 邮件联系: [project-email]

---

**🚀 开始使用AI-Product Manager，体验自主科学创新的强大能力！**