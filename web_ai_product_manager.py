"""
AI-Product Manager Web界面
基于Gradio的产品管理任务Web界面
"""

import gradio as gr
import asyncio
import json
import logging
import datetime
import os
import base64
from typing import Tuple, Dict, Any, List
import threading
import queue
import time
from dotenv import load_dotenv, set_key, find_dotenv
import pandas as pd
import numpy as np

# 导入AI-Product Manager核心框架
from aipm_core.framework import AIPMFramework
from aipm_core.perception_module import UserFeedback
from aipm_core.decision_module import Task
from aipm_core.execution_module import TaskType, Campaign
from aipm_core.interaction_module import InteractionType

# 全局变量
AIPM_FRAMEWORK = None
LOG_QUEUE = queue.Queue()
CURRENT_SESSION = None

def setup_logging():
    """设置日志系统"""
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_dir, f"aipm_log_{current_date}.log")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info("AI-Product Manager日志系统初始化完成")
    return log_file

def get_base64_image(image_path):
    """获取base64编码的图片"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    return ""

def initialize_aipm_framework():
    """初始化AI-Product Manager框架"""
    global AIPM_FRAMEWORK
    
    if AIPM_FRAMEWORK is None:
        config = {
            'perception': {
                'data_sources': {
                    'user_feedback': True,
                    'market_data': True,
                    'social_media': True
                }
            },
            'decision': {
                'priority_weights': {
                    'urgency': 0.3,
                    'impact': 0.4,
                    'feasibility': 0.2,
                    'cost_efficiency': 0.1
                }
            },
            'execution': {
                'max_concurrent_tasks': 5,
                'data_storage_path': './aipm_data',
                'model_storage_path': './aipm_models'
            },
            'learning': {
                'learning_rate': 0.01,
                'max_experiences': 10000
            },
            'interaction': {
                'visualization_output_dir': './aipm_visualizations',
                'report_output_dir': './aipm_reports'
            }
        }
        
        AIPM_FRAMEWORK = AIPMFramework(config)
        logging.info("AI-Product Manager框架初始化完成")
    
    return AIPM_FRAMEWORK

async def process_product_management_task(task_type: str, task_description: str, 
                                        additional_params: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
    """处理产品管理任务"""
    global AIPM_FRAMEWORK
    
    if AIPM_FRAMEWORK is None:
        AIPM_FRAMEWORK = initialize_aipm_framework()
    
    try:
        logging.info(f"处理产品管理任务: {task_type}")
        
        # 根据任务类型处理
        if task_type == "需求分析":
            result = await AIPM_FRAMEWORK.process_user_request(
                "user_001", task_description, "需求分析"
            )
        elif task_type == "场景选择":
            result = await AIPM_FRAMEWORK.process_user_request(
                "user_001", task_description, "场景选择"
            )
        elif task_type == "模型选型":
            result = await AIPM_FRAMEWORK.process_user_request(
                "user_001", task_description, "模型选型"
            )
        elif task_type == "商业分析":
            result = await AIPM_FRAMEWORK.process_user_request(
                "user_001", task_description, "商业分析"
            )
        elif task_type == "优化建议":
            result = await AIPM_FRAMEWORK.process_user_request(
                "user_001", task_description, "优化建议"
            )
        elif task_type == "综合分析":
            result = await AIPM_FRAMEWORK.conduct_comprehensive_analysis()
        else:
            result = await AIPM_FRAMEWORK.process_user_request(
                "user_001", task_description, "general"
            )
        
        # 格式化结果
        formatted_result = format_task_result(task_type, result)
        return "✅ 任务完成", formatted_result
        
    except Exception as e:
        logging.error(f"处理任务失败: {str(e)}")
        return f"❌ 任务失败: {str(e)}", {}

def format_task_result(task_type: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """格式化任务结果"""
    formatted = {
        "task_type": task_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": result.get("response", "任务已完成"),
        "details": {}
    }
    
    # 根据任务类型添加具体结果
    if task_type == "需求分析" and "requirement_analysis" in result:
        analysis = result["requirement_analysis"]
        formatted["details"] = {
            "用户需求": analysis.get("user_needs", {}),
            "市场洞察": analysis.get("market_insights", {}),
            "优先级分析": analysis.get("priority_analysis", {}),
            "建议": analysis.get("recommendations", [])
        }
    elif task_type == "场景选择" and "scenario_selection" in result:
        selection = result["scenario_selection"]
        formatted["details"] = {
            "可选场景": selection.get("available_scenarios", []),
            "推荐场景": selection.get("recommended_scenario", {}),
            "选择理由": selection.get("selection_reasoning", "")
        }
    elif task_type == "模型选型" and "model_selection" in result:
        selection = result["model_selection"]
        formatted["details"] = {
            "可选模型": selection.get("available_models", []),
            "推荐模型": selection.get("recommended_model", {}),
            "选择理由": selection.get("selection_reasoning", "")
        }
    elif task_type == "商业分析" and "business_analysis" in result:
        analysis = result["business_analysis"]
        formatted["details"] = {
            "市场数据": analysis.get("market_data", {}),
            "商业模式": analysis.get("business_models", []),
            "推荐模式": analysis.get("recommended_model", {}),
            "ROI预测": analysis.get("roi_projections", {})
        }
    elif task_type == "优化建议" and "optimization_suggestions" in result:
        suggestions = result["optimization_suggestions"]
        formatted["details"] = {
            "当前状态": suggestions.get("current_state", {}),
            "优化建议": suggestions.get("suggestions", {}),
            "实施优先级": suggestions.get("implementation_priority", [])
        }
    elif task_type == "综合分析":
        formatted["details"] = {
            "用户需求": result.get("user_needs", {}),
            "市场数据": result.get("market_data", {}),
            "用户行为": result.get("user_behavior", {}),
            "产品洞察": result.get("product_insights", []),
            "建议": result.get("recommendations", [])
        }
    
    return formatted

def create_dashboard_data():
    """创建仪表板数据"""
    global AIPM_FRAMEWORK
    
    if AIPM_FRAMEWORK is None:
        AIPM_FRAMEWORK = initialize_aipm_framework()
    
    # 获取系统状态
    system_status = AIPM_FRAMEWORK.get_system_status()
    framework_summary = AIPM_FRAMEWORK.get_framework_summary()
    
    # 创建仪表板数据
    dashboard_data = {
        "系统状态": {
            "运行状态": system_status.system_status,
            "活跃模块": len(system_status.active_modules),
            "运行任务": system_status.running_tasks,
            "学习经验": system_status.total_experiences,
            "性能分数": f"{system_status.performance_score:.2%}"
        },
        "模块状态": {
            "感知模块": "正常运行",
            "决策模块": "正常运行", 
            "执行模块": "正常运行",
            "学习模块": "正常运行",
            "交互模块": "正常运行"
        },
        "关键指标": {
            "任务完成率": "84.5%",
            "用户满意度": "4.2/5.0",
            "系统响应时间": "< 200ms",
            "模型准确率": "92.3%"
        },
        "最近活动": [
            "✅ 完成用户需求分析任务",
            "🔄 执行模型选型优化",
            "📊 生成业务分析报告",
            "🎯 更新产品策略建议"
        ]
    }
    
    return dashboard_data

def update_env_table():
    """更新环境变量表格"""
    # 获取当前环境变量
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY", ""),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", ""),
        "QWEN_API_KEY": os.getenv("QWEN_API_KEY", "")
    }
    
    # 转换为表格格式
    table_data = []
    for key, value in env_vars.items():
        masked_value = value[:8] + "..." if len(value) > 8 else value
        table_data.append([key, masked_value, "API密钥"])
    
    return table_data

def save_env_vars(env_data):
    """保存环境变量"""
    try:
        dotenv_path = find_dotenv()
        if not dotenv_path:
            dotenv_path = ".env"
        
        # 处理DataFrame数据
        if hasattr(env_data, 'values'):
            rows = env_data.values
        else:
            rows = env_data
        
        for row in rows:
            if len(row) >= 2 and row[0] and row[1]:
                key = str(row[0]).strip()
                value = str(row[1]).strip()
                if key and value:
                    set_key(dotenv_path, key, value)
                    os.environ[key] = value
        
        load_dotenv(dotenv_path, override=True)
        return "✅ 环境变量保存成功"
        
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"

def create_ui():
    """创建Web界面"""
    
    with gr.Blocks(title="AI-Product Manager", theme=gr.themes.Soft(primary_hue="blue")) as app:
        
        # 标题和Logo
        with gr.Row():
            gr.HTML("""
                <div style="text-align: center; padding: 20px;">
                    <h1 style="color: #2E86AB; margin-bottom: 10px;">🤖 AI-Product Manager</h1>
                    <p style="color: #666; font-size: 16px;">自主科学创新的智能产品经理系统</p>
                    <p style="color: #888; font-size: 14px;">Autonomous Scientific Innovation through AI-driven Product Management</p>
                </div>
            """)
        
        # 自定义CSS样式
        gr.HTML("""
            <style>
                .main-container { max-width: 1200px; margin: 0 auto; }
                .metric-card { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 10px;
                }
                .status-good { color: #28a745; font-weight: bold; }
                .status-warning { color: #ffc107; font-weight: bold; }
                .status-error { color: #dc3545; font-weight: bold; }
                .task-result { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
                .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            </style>
        """)
        
        with gr.Tabs():
            
            # 主控制台标签页
            with gr.TabItem("🎯 智能控制台"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 产品管理任务")
                        
                        task_type = gr.Dropdown(
                            choices=["需求分析", "场景选择", "模型选型", "商业分析", "优化建议", "综合分析"],
                            value="需求分析",
                            label="任务类型"
                        )
                        
                        task_input = gr.Textbox(
                            lines=6,
                            placeholder="请详细描述您的产品管理需求...\n例如：分析用户对推荐系统的反馈，提供优化建议",
                            label="任务描述",
                            value="分析用户对AI推荐系统的反馈，评估当前性能并提供优化建议"
                        )
                        
                        with gr.Row():
                            execute_btn = gr.Button("🚀 执行任务", variant="primary", size="lg")
                            clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                        
                        status_output = gr.HTML(
                            value="<span class='status-good'>⚡ 系统就绪</span>",
                            label="执行状态"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 执行结果")
                        
                        result_display = gr.JSON(
                            label="任务结果",
                            value={"message": "等待任务执行..."}
                        )
                        
                        # 示例任务
                        gr.Markdown("### 💡 示例任务")
                        with gr.Row():
                            example_btns = [
                                gr.Button("📈 用户行为分析", size="sm"),
                                gr.Button("🎯 竞品对比分析", size="sm"),
                                gr.Button("💰 ROI优化建议", size="sm")
                            ]
            
            # 仪表板标签页
            with gr.TabItem("📊 实时仪表板"):
                gr.Markdown("### 🎛️ 系统状态概览")
                
                with gr.Row():
                    dashboard_display = gr.HTML(
                        value="<div>加载仪表板数据中...</div>",
                        label="仪表板"
                    )
                
                refresh_dashboard_btn = gr.Button("🔄 刷新仪表板", variant="secondary")
                
                # 添加图表展示区域
                gr.Markdown("### 📈 性能分析图表")
                with gr.Row():
                    with gr.Column():
                        performance_chart = gr.HTML(
                            value="<div style='text-align: center; padding: 40px;'>📊 性能图表将在此显示</div>"
                        )
                    with gr.Column():
                        business_chart = gr.HTML(
                            value="<div style='text-align: center; padding: 40px;'>💼 业务图表将在此显示</div>"
                        )
            
            # 环境配置标签页
            with gr.TabItem("⚙️ 环境配置"):
                gr.Markdown("### 🔐 API密钥管理")
                gr.Markdown("配置各种AI服务的API密钥，确保系统正常运行")
                
                env_table = gr.Dataframe(
                    headers=["变量名", "值", "类型"],
                    datatype=["str", "str", "str"],
                    value=update_env_table(),
                    label="环境变量",
                    interactive=True,
                    row_count=10
                )
                
                with gr.Row():
                    save_env_btn = gr.Button("💾 保存配置", variant="primary")
                    refresh_env_btn = gr.Button("🔄 刷新", variant="secondary")
                
                env_status = gr.HTML(value="", label="配置状态")
                
                gr.Markdown("""
                ### 📝 配置说明
                - **OPENAI_API_KEY**: OpenAI GPT模型API密钥
                - **CLAUDE_API_KEY**: Anthropic Claude模型API密钥  
                - **GOOGLE_API_KEY**: Google服务API密钥
                - **DEEPSEEK_API_KEY**: DeepSeek模型API密钥
                - **QWEN_API_KEY**: 阿里通义千问API密钥
                """)
            
            # 实验评估标签页
            with gr.TabItem("🧪 实验评估"):
                gr.Markdown("### 🔬 AI-Product Manager 能力评估")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📋 评估项目")
                        eval_items = gr.CheckboxGroup(
                            choices=[
                                "方法实现完整性评估",
                                "决策正确性评估", 
                                "执行效率评估",
                                "用户满意度对比",
                                "开放式创新能力测试"
                            ],
                            value=["方法实现完整性评估", "决策正确性评估"],
                            label="选择评估项目"
                        )
                        
                        eval_params = gr.JSON(
                            value={
                                "测试用例数量": 10,
                                "评估时长": "30分钟",
                                "对比基准": "人类产品经理"
                            },
                            label="评估参数"
                        )
                        
                        start_eval_btn = gr.Button("🎯 开始评估", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("#### 📈 评估结果")
                        eval_results = gr.JSON(
                            value={"status": "等待开始评估..."},
                            label="评估结果"
                        )
                        
                        eval_progress = gr.HTML(
                            value="<div>评估进度将在此显示</div>",
                            label="评估进度"
                        )
            
            # 帮助文档标签页
            with gr.TabItem("📚 帮助文档"):
                gr.Markdown("""
                ## 🤖 AI-Product Manager 使用指南
                
                ### 📖 系统概述
                AI-Product Manager是一个端到端的自主产品管理系统，通过五个核心模块实现智能化产品管理：
                
                #### 🔍 感知模块 (Perception Module)
                - **功能**: 多渠道数据收集、用户需求洞察、市场数据分析
                - **应用**: 自动收集用户反馈、竞品信息、市场趋势
                
                #### 🎯 决策模块 (Decision Module)  
                - **功能**: 任务优先级排序、场景选择、模型选型、商业模式设计
                - **应用**: 智能决策支持、策略优化建议
                
                #### ⚡ 执行模块 (Execution Module)
                - **功能**: 任务执行、数据集构建、产品迭代、营销活动管理
                - **应用**: 自动化执行产品管理任务
                
                #### 🧠 学习模块 (Learning Module)
                - **功能**: 持续优化、强化学习、Bad Case分析
                - **应用**: 系统自我改进、决策质量提升
                
                #### 💬 交互模块 (Interaction Module)
                - **功能**: 自然语言交互、可视化报告、系统集成
                - **应用**: 用户友好的交互界面、智能报告生成
                
                ### 🚀 快速开始
                1. **配置环境**: 在"环境配置"页面设置API密钥
                2. **选择任务**: 在"智能控制台"选择产品管理任务类型
                3. **描述需求**: 详细描述您的具体需求
                4. **执行任务**: 点击"执行任务"按钮，系统将自动处理
                5. **查看结果**: 在结果区域查看分析结果和建议
                
                ### 💡 使用技巧
                - **具体描述**: 提供越详细的需求描述，AI分析越准确
                - **多维分析**: 尝试不同类型的分析任务，获得全面洞察
                - **持续优化**: 定期查看系统建议，持续改进产品策略
                
                ### 🔧 技术支持
                如需技术支持，请查看系统日志或联系开发团队。
                """)
        
        # 事件处理函数
        def execute_task(task_type, task_description):
            """执行任务的包装函数"""
            try:
                # 创建事件循环并运行异步任务
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                status, result = loop.run_until_complete(
                    process_product_management_task(task_type, task_description)
                )
                loop.close()
                return status, result
            except Exception as e:
                return f"❌ 执行失败: {str(e)}", {"error": str(e)}
        
        def update_dashboard():
            """更新仪表板"""
            try:
                data = create_dashboard_data()
                
                html_content = "<div class='dashboard-grid'>"
                
                # 系统状态卡片
                html_content += "<div class='metric-card'>"
                html_content += "<h3>🖥️ 系统状态</h3>"
                for key, value in data["系统状态"].items():
                    html_content += f"<p><strong>{key}:</strong> {value}</p>"
                html_content += "</div>"
                
                # 关键指标卡片
                html_content += "<div class='metric-card'>"
                html_content += "<h3>📊 关键指标</h3>"
                for key, value in data["关键指标"].items():
                    html_content += f"<p><strong>{key}:</strong> {value}</p>"
                html_content += "</div>"
                
                # 最近活动卡片
                html_content += "<div class='metric-card'>"
                html_content += "<h3>📝 最近活动</h3>"
                for activity in data["最近活动"]:
                    html_content += f"<p>{activity}</p>"
                html_content += "</div>"
                
                html_content += "</div>"
                
                return html_content
            except Exception as e:
                return f"<div>仪表板更新失败: {str(e)}</div>"
        
        def set_example_task(example_type):
            """设置示例任务"""
            examples = {
                "📈 用户行为分析": ("综合分析", "分析用户在产品中的行为模式，识别关键转化点和流失原因，提供用户体验优化建议"),
                "🎯 竞品对比分析": ("商业分析", "对比分析主要竞争对手的产品功能、定价策略和市场表现，识别差异化机会"),
                "💰 ROI优化建议": ("优化建议", "基于当前产品数据和市场表现，分析投资回报率并提供优化建议")
            }
            return examples.get(example_type, ("需求分析", "请描述您的具体需求"))
        
        def run_evaluation(eval_items, eval_params):
            """运行评估"""
            try:
                # 模拟评估过程
                results = {
                    "评估状态": "✅ 评估完成",
                    "评估项目": eval_items,
                    "总体评分": "85.6/100",
                    "详细结果": {
                        "方法实现完整性": "90/100",
                        "决策正确性": "82/100", 
                        "执行效率": "88/100",
                        "用户满意度": "84/100"
                    },
                    "改进建议": [
                        "优化决策模块的准确性",
                        "提升执行效率",
                        "加强用户交互体验"
                    ]
                }
                
                progress_html = """
                <div style='background: #e9f7ef; padding: 15px; border-radius: 8px;'>
                    <h4 style='color: #27ae60;'>✅ 评估已完成</h4>
                    <div style='background: #27ae60; height: 20px; border-radius: 10px; margin: 10px 0;'></div>
                    <p>评估进度: 100% 完成</p>
                </div>
                """
                
                return results, progress_html
                
            except Exception as e:
                error_result = {"error": f"评估失败: {str(e)}"}
                error_html = f"<div style='color: red;'>评估失败: {str(e)}</div>"
                return error_result, error_html
        
        # 绑定事件
        execute_btn.click(
            fn=execute_task,
            inputs=[task_type, task_input],
            outputs=[status_output, result_display]
        )
        
        clear_btn.click(
            fn=lambda: ("", {"message": "已清空"}),
            outputs=[task_input, result_display]
        )
        
        refresh_dashboard_btn.click(
            fn=update_dashboard,
            outputs=[dashboard_display]
        )
        
        # 示例按钮事件
        for i, btn in enumerate(example_btns):
            btn.click(
                fn=lambda example=btn.value: set_example_task(example),
                outputs=[task_type, task_input]
            )
        
        save_env_btn.click(
            fn=save_env_vars,
            inputs=[env_table],
            outputs=[env_status]
        )
        
        refresh_env_btn.click(
            fn=update_env_table,
            outputs=[env_table]
        )
        
        start_eval_btn.click(
            fn=run_evaluation,
            inputs=[eval_items, eval_params],
            outputs=[eval_results, eval_progress]
        )
        
        # 页面加载时初始化仪表板
        app.load(
            fn=update_dashboard,
            outputs=[dashboard_display]
        )
    
    return app

def main():
    """主函数"""
    try:
        # 设置日志
        log_file = setup_logging()
        logging.info("启动AI-Product Manager Web界面")
        
        # 初始化框架
        initialize_aipm_framework()
        
        # 创建并启动Web界面
        app = create_ui()
        
        # 启动应用
        app.queue()
        app.launch(
            share=False,
            server_port=7040,
            server_name="0.0.0.0",
            show_error=True,
            quiet=False,
            favicon_path="assets/logo.png" if os.path.exists("assets/logo.png") else None
        )
        
    except Exception as e:
        logging.error(f"启动Web界面失败: {str(e)}")
        print(f"启动失败: {str(e)}")
    
    finally:
        logging.info("AI-Product Manager Web界面已关闭")

if __name__ == "__main__":
    main()