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
CLAUDE_API_KEY=your_claude_api_key
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


- 技术支持: [Issues Page]
- 邮件联系: [project-email]

---

**🚀 开始使用AI-Product Manager，体验自主科学创新的强大能力！**
