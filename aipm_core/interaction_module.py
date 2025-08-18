"""
交互模块 (Interaction Module)
负责用户交互界面、自然语言交互、可视化报告生成、系统集成
"""

import asyncio
import json
import os
import base64
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import requests

class InteractionType(Enum):
    """交互类型"""
    CHAT = "chat"
    VOICE = "voice"
    API = "api"
    DASHBOARD = "dashboard"
    REPORT = "report"
    NOTIFICATION = "notification"

class MessageType(Enum):
    """消息类型"""
    USER_QUERY = "user_query"
    SYSTEM_RESPONSE = "system_response"
    TASK_UPDATE = "task_update"
    ALERT = "alert"
    RECOMMENDATION = "recommendation"

class ReportType(Enum):
    """报告类型"""
    PERFORMANCE = "performance"
    BUSINESS_METRICS = "business_metrics"
    USER_ANALYSIS = "user_analysis"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    TECHNICAL_SUMMARY = "technical_summary"

@dataclass
class Message:
    """消息数据结构"""
    message_id: str
    sender: str
    recipient: str
    content: str
    message_type: MessageType
    interaction_type: InteractionType
    timestamp: datetime
    metadata: Dict[str, Any] = None
    attachments: List[str] = None

@dataclass
class UserSession:
    """用户会话"""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    messages: List[Message]
    context: Dict[str, Any]
    preferences: Dict[str, Any]

@dataclass
class Report:
    """报告结构"""
    report_id: str
    title: str
    report_type: ReportType
    content: Dict[str, Any]
    visualizations: List[str]
    generated_at: datetime
    recipient: str
    format: str  # html, pdf, json

class NaturalLanguageProcessor:
    """自然语言处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 意图识别模式
        self.intent_patterns = {
            "查询状态": ["状态", "进度", "怎么样", "如何"],
            "任务管理": ["创建", "执行", "停止", "任务"],
            "数据分析": ["分析", "统计", "报告", "数据"],
            "优化建议": ["优化", "改进", "建议", "提升"],
            "系统配置": ["配置", "设置", "参数", "调整"]
        }
        
    def process_user_input(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理用户输入"""
        # 意图识别
        intent = self._identify_intent(text)
        
        # 实体提取
        entities = self._extract_entities(text)
        
        # 情感分析
        sentiment = self._analyze_sentiment(text)
        
        return {
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "confidence": self._calculate_confidence(text, intent),
            "original_text": text,
            "processed_at": datetime.now().isoformat()
        }
    
    def _identify_intent(self, text: str) -> str:
        """识别用户意图"""
        text_lower = text.lower()
        
        for intent, keywords in self.intent_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent
        
        return "其他"
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """提取实体"""
        entities = []
        
        # 简化的实体识别
        if "任务" in text:
            entities.append({"type": "object", "value": "任务", "position": text.find("任务")})
        
        if "模型" in text:
            entities.append({"type": "object", "value": "模型", "position": text.find("模型")})
        
        # 提取数字
        import re
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            entities.append({"type": "number", "value": num, "position": text.find(num)})
        
        return entities
    
    def _analyze_sentiment(self, text: str) -> str:
        """情感分析"""
        positive_words = ["好", "棒", "优秀", "满意", "喜欢", "赞"]
        negative_words = ["差", "糟糕", "不满", "问题", "错误", "失望"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(self, text: str, intent: str) -> float:
        """计算置信度"""
        if intent == "其他":
            return 0.3
        
        keywords = self.intent_patterns.get(intent, [])
        matched_keywords = sum(1 for keyword in keywords if keyword in text.lower())
        confidence = min(matched_keywords / len(keywords), 1.0) * 0.8 + 0.2
        
        return confidence
    
    def generate_response(self, processed_input: Dict[str, Any], 
                         context: Dict[str, Any] = None) -> str:
        """生成响应"""
        intent = processed_input.get("intent", "其他")
        entities = processed_input.get("entities", [])
        
        if intent == "查询状态":
            return self._generate_status_response(context)
        elif intent == "任务管理":
            return self._generate_task_response(entities, context)
        elif intent == "数据分析":
            return self._generate_analysis_response(context)
        elif intent == "优化建议":
            return self._generate_optimization_response(context)
        elif intent == "系统配置":
            return self._generate_config_response(entities, context)
        else:
            return "我理解您的询问，但需要更多信息来提供准确的回答。您可以询问系统状态、任务管理、数据分析等相关问题。"
    
    def _generate_status_response(self, context: Dict[str, Any]) -> str:
        """生成状态响应"""
        if not context:
            return "系统运行正常，所有模块都在正常工作中。"
        
        status_info = context.get("system_status", {})
        return f"系统状态：{status_info.get('status', '正常')}。当前运行任务：{status_info.get('running_tasks', 0)}个。"
    
    def _generate_task_response(self, entities: List[Dict[str, str]], 
                              context: Dict[str, Any]) -> str:
        """生成任务响应"""
        return "我已收到您的任务管理请求。您可以通过任务管理界面查看和操作具体任务。"
    
    def _generate_analysis_response(self, context: Dict[str, Any]) -> str:
        """生成分析响应"""
        return "数据分析功能已启动。我将为您生成详细的分析报告，包括业务指标、用户行为和性能统计。"
    
    def _generate_optimization_response(self, context: Dict[str, Any]) -> str:
        """生成优化响应"""
        return "基于当前数据分析，我建议优化以下方面：模型性能、用户体验和资源配置。详细建议将在优化报告中提供。"
    
    def _generate_config_response(self, entities: List[Dict[str, str]], 
                                context: Dict[str, Any]) -> str:
        """生成配置响应"""
        return "系统配置功能已开启。您可以通过配置界面调整相关参数。"

class VisualizationEngine:
    """可视化引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = config.get('visualization_output_dir', './aipm_visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置可视化样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_performance_dashboard(self, metrics: Dict[str, Any]) -> str:
        """创建性能仪表板"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AI产品经理 - 性能仪表板', fontsize=16, fontweight='bold')
        
        # 1. 任务完成率
        task_data = metrics.get('task_completion', {})
        if task_data:
            labels = list(task_data.keys())
            values = list(task_data.values())
            axes[0, 0].pie(values, labels=labels, autopct='%1.1f%%')
            axes[0, 0].set_title('任务完成情况')
        
        # 2. 性能趋势
        performance_data = metrics.get('performance_trends', {})
        if performance_data:
            for metric_name, values in performance_data.items():
                axes[0, 1].plot(range(len(values)), values, label=metric_name, marker='o')
            axes[0, 1].set_title('性能趋势')
            axes[0, 1].legend()
            axes[0, 1].set_xlabel('时间')
            axes[0, 1].set_ylabel('性能分数')
        
        # 3. 资源使用情况
        resource_data = metrics.get('resource_usage', {})
        if resource_data:
            resources = list(resource_data.keys())
            usage = list(resource_data.values())
            bars = axes[1, 0].bar(resources, usage)
            axes[1, 0].set_title('资源使用情况')
            axes[1, 0].set_ylabel('使用率 (%)')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom')
        
        # 4. 用户满意度
        satisfaction_data = metrics.get('user_satisfaction', {})
        if satisfaction_data:
            categories = list(satisfaction_data.keys())
            scores = list(satisfaction_data.values())
            axes[1, 1].barh(categories, scores, color='lightgreen')
            axes[1, 1].set_title('用户满意度')
            axes[1, 1].set_xlabel('满意度分数')
            axes[1, 1].set_xlim(0, 5)
        
        plt.tight_layout()
        
        # 保存图像
        filename = f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_business_metrics_chart(self, business_data: Dict[str, Any]) -> str:
        """创建业务指标图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('业务指标分析', fontsize=16, fontweight='bold')
        
        # 1. 收入趋势
        revenue_data = business_data.get('revenue_trend', [])
        if revenue_data:
            months = list(range(1, len(revenue_data) + 1))
            axes[0, 0].plot(months, revenue_data, marker='o', linewidth=2, color='green')
            axes[0, 0].set_title('收入趋势')
            axes[0, 0].set_xlabel('月份')
            axes[0, 0].set_ylabel('收入 (万元)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 用户增长
        user_growth = business_data.get('user_growth', {})
        if user_growth:
            months = list(user_growth.keys())
            new_users = list(user_growth.values())
            axes[0, 1].bar(months, new_users, color='skyblue')
            axes[0, 1].set_title('用户增长')
            axes[0, 1].set_xlabel('月份')
            axes[0, 1].set_ylabel('新增用户数')
        
        # 3. 转化漏斗
        funnel_data = business_data.get('conversion_funnel', {})
        if funnel_data:
            stages = list(funnel_data.keys())
            values = list(funnel_data.values())
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(stages)))
            axes[1, 0].barh(stages, values, color=colors)
            axes[1, 0].set_title('转化漏斗')
            axes[1, 0].set_xlabel('转化率 (%)')
        
        # 4. ROI分析
        roi_data = business_data.get('roi_analysis', {})
        if roi_data:
            channels = list(roi_data.keys())
            roi_values = list(roi_data.values())
            colors = ['red' if x < 1.0 else 'green' for x in roi_values]
            axes[1, 1].bar(channels, roi_values, color=colors)
            axes[1, 1].set_title('投资回报率 (ROI)')
            axes[1, 1].set_ylabel('ROI')
            axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图像
        filename = f"business_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_user_analysis_chart(self, user_data: Dict[str, Any]) -> str:
        """创建用户分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('用户行为分析', fontsize=16, fontweight='bold')
        
        # 1. 用户画像
        demographics = user_data.get('demographics', {})
        if demographics:
            age_groups = list(demographics.keys())
            percentages = list(demographics.values())
            axes[0, 0].pie(percentages, labels=age_groups, autopct='%1.1f%%')
            axes[0, 0].set_title('用户年龄分布')
        
        # 2. 活跃度热力图
        activity_data = user_data.get('activity_heatmap', np.random.rand(7, 24))
        if isinstance(activity_data, (list, np.ndarray)):
            sns.heatmap(activity_data, 
                       xticklabels=[f'{i}h' for i in range(24)],
                       yticklabels=['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
                       ax=axes[0, 1], cmap='YlOrRd')
            axes[0, 1].set_title('用户活跃度热力图')
        
        # 3. 留存率
        retention_data = user_data.get('retention_rates', {})
        if retention_data:
            days = list(retention_data.keys())
            rates = list(retention_data.values())
            axes[1, 0].plot(days, rates, marker='o', linewidth=2)
            axes[1, 0].set_title('用户留存率')
            axes[1, 0].set_xlabel('天数')
            axes[1, 0].set_ylabel('留存率 (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 功能使用情况
        feature_usage = user_data.get('feature_usage', {})
        if feature_usage:
            features = list(feature_usage.keys())
            usage_counts = list(feature_usage.values())
            axes[1, 1].barh(features, usage_counts, color='lightcoral')
            axes[1, 1].set_title('功能使用情况')
            axes[1, 1].set_xlabel('使用次数')
        
        plt.tight_layout()
        
        # 保存图像
        filename = f"user_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_learning_progress_chart(self, learning_data: Dict[str, Any]) -> str:
        """创建学习进度图表"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('AI学习进度', fontsize=16, fontweight='bold')
        
        # 1. 奖励趋势
        rewards = learning_data.get('reward_history', [])
        if rewards:
            episodes = list(range(1, len(rewards) + 1))
            axes[0].plot(episodes, rewards, linewidth=2, alpha=0.7)
            
            # 添加移动平均线
            if len(rewards) > 10:
                window_size = min(20, len(rewards) // 5)
                moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
                axes[0].plot(episodes, moving_avg, linewidth=3, color='red', label='移动平均')
                axes[0].legend()
            
            axes[0].set_title('学习奖励趋势')
            axes[0].set_xlabel('经验数量')
            axes[0].set_ylabel('奖励值')
            axes[0].grid(True, alpha=0.3)
        
        # 2. 模块性能对比
        module_performance = learning_data.get('module_performance', {})
        if module_performance:
            modules = list(module_performance.keys())
            performance_scores = list(module_performance.values())
            
            bars = axes[1].bar(modules, performance_scores, 
                             color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
            axes[1].set_title('各模块性能')
            axes[1].set_ylabel('性能分数')
            axes[1].set_ylim(0, 1.0)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图像
        filename = f"learning_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.visualization_engine = VisualizationEngine(config)
        self.output_dir = config.get('report_output_dir', './aipm_reports')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_comprehensive_report(self, 
                                    performance_data: Dict[str, Any],
                                    business_data: Dict[str, Any],
                                    user_data: Dict[str, Any],
                                    learning_data: Dict[str, Any],
                                    recipient: str = "product_team") -> Report:
        """生成综合报告"""
        report_id = f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 生成可视化图表
        visualizations = []
        
        try:
            perf_chart = self.visualization_engine.create_performance_dashboard(performance_data)
            visualizations.append(perf_chart)
        except Exception as e:
            self.logger.error(f"生成性能图表失败: {e}")
        
        try:
            business_chart = self.visualization_engine.create_business_metrics_chart(business_data)
            visualizations.append(business_chart)
        except Exception as e:
            self.logger.error(f"生成业务图表失败: {e}")
        
        try:
            user_chart = self.visualization_engine.create_user_analysis_chart(user_data)
            visualizations.append(user_chart)
        except Exception as e:
            self.logger.error(f"生成用户图表失败: {e}")
        
        try:
            learning_chart = self.visualization_engine.create_learning_progress_chart(learning_data)
            visualizations.append(learning_chart)
        except Exception as e:
            self.logger.error(f"生成学习图表失败: {e}")
        
        # 生成报告内容
        report_content = {
            "executive_summary": self._generate_executive_summary(
                performance_data, business_data, user_data, learning_data
            ),
            "performance_analysis": self._analyze_performance(performance_data),
            "business_insights": self._analyze_business_metrics(business_data),
            "user_behavior_analysis": self._analyze_user_behavior(user_data),
            "ai_learning_progress": self._analyze_learning_progress(learning_data),
            "recommendations": self._generate_recommendations(
                performance_data, business_data, user_data, learning_data
            ),
            "next_steps": self._generate_next_steps(),
            "appendix": {
                "data_sources": "系统内置监控、用户反馈、业务指标",
                "methodology": "基于AI产品经理框架的多维度分析",
                "generated_by": "AI-Product Manager System"
            }
        }
        
        # 创建报告对象
        report = Report(
            report_id=report_id,
            title="AI产品经理 - 综合分析报告",
            report_type=ReportType.PERFORMANCE,
            content=report_content,
            visualizations=visualizations,
            generated_at=datetime.now(),
            recipient=recipient,
            format="html"
        )
        
        # 生成HTML报告
        html_content = self._generate_html_report(report)
        html_filepath = os.path.join(self.output_dir, f"{report_id}.html")
        
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"综合报告已生成: {html_filepath}")
        return report
    
    def _generate_executive_summary(self, performance_data: Dict[str, Any],
                                  business_data: Dict[str, Any],
                                  user_data: Dict[str, Any],
                                  learning_data: Dict[str, Any]) -> str:
        """生成执行摘要"""
        summary_points = []
        
        # 性能亮点
        if performance_data.get('overall_score', 0) > 0.8:
            summary_points.append("系统整体性能表现优秀，各项指标均达到预期目标。")
        
        # 业务增长
        revenue_trend = business_data.get('revenue_trend', [])
        if len(revenue_trend) >= 2 and revenue_trend[-1] > revenue_trend[-2]:
            growth_rate = (revenue_trend[-1] - revenue_trend[-2]) / revenue_trend[-2] * 100
            summary_points.append(f"业务收入呈上升趋势，环比增长{growth_rate:.1f}%。")
        
        # 用户活跃度
        user_growth = business_data.get('user_growth', {})
        if user_growth:
            total_users = sum(user_growth.values())
            summary_points.append(f"累计新增用户{total_users}人，用户基础持续扩大。")
        
        # AI学习进展
        learning_metrics = learning_data.get('learning_metrics', {})
        if learning_metrics.get('improvement_rate', 0) > 0:
            summary_points.append("AI系统持续学习优化，决策质量显著提升。")
        
        return " ".join(summary_points) if summary_points else "本期系统运行平稳，各项指标保持稳定。"
    
    def _analyze_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能数据"""
        return {
            "overall_score": performance_data.get('overall_score', 0.0),
            "key_metrics": performance_data.get('key_metrics', {}),
            "trends": performance_data.get('trends', {}),
            "bottlenecks": performance_data.get('bottlenecks', []),
            "improvements": performance_data.get('improvements', [])
        }
    
    def _analyze_business_metrics(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析业务指标"""
        return {
            "revenue_analysis": business_data.get('revenue_analysis', {}),
            "user_acquisition": business_data.get('user_acquisition', {}),
            "conversion_metrics": business_data.get('conversion_metrics', {}),
            "roi_performance": business_data.get('roi_performance', {}),
            "market_insights": business_data.get('market_insights', {})
        }
    
    def _analyze_user_behavior(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析用户行为"""
        return {
            "user_segments": user_data.get('user_segments', {}),
            "engagement_metrics": user_data.get('engagement_metrics', {}),
            "retention_analysis": user_data.get('retention_analysis', {}),
            "feature_adoption": user_data.get('feature_adoption', {}),
            "satisfaction_scores": user_data.get('satisfaction_scores', {})
        }
    
    def _analyze_learning_progress(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析学习进展"""
        return {
            "learning_metrics": learning_data.get('learning_metrics', {}),
            "optimization_results": learning_data.get('optimization_results', {}),
            "model_improvements": learning_data.get('model_improvements', {}),
            "knowledge_acquisition": learning_data.get('knowledge_acquisition', {}),
            "adaptive_capabilities": learning_data.get('adaptive_capabilities', {})
        }
    
    def _generate_recommendations(self, performance_data: Dict[str, Any],
                                business_data: Dict[str, Any],
                                user_data: Dict[str, Any],
                                learning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成建议"""
        recommendations = []
        
        # 基于性能数据的建议
        if performance_data.get('overall_score', 0) < 0.7:
            recommendations.append({
                "category": "性能优化",
                "priority": "高",
                "description": "系统整体性能需要优化，建议重点关注核心算法和资源配置",
                "expected_impact": "提升系统响应速度和用户体验"
            })
        
        # 基于业务数据的建议
        roi_data = business_data.get('roi_analysis', {})
        if any(roi < 1.0 for roi in roi_data.values()):
            recommendations.append({
                "category": "营销优化",
                "priority": "中",
                "description": "部分营销渠道ROI偏低，建议调整投放策略和预算分配",
                "expected_impact": "提高营销效率和投资回报率"
            })
        
        # 基于用户数据的建议
        retention_rates = user_data.get('retention_rates', {})
        if retention_rates and min(retention_rates.values()) < 0.5:
            recommendations.append({
                "category": "用户留存",
                "priority": "高",
                "description": "用户留存率偏低，建议优化用户体验和增加粘性功能",
                "expected_impact": "提升用户生命周期价值"
            })
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """生成下一步行动计划"""
        return [
            "持续监控关键性能指标，及时发现和解决问题",
            "深入分析用户行为数据，优化产品功能和体验",
            "加强AI模型的训练和优化，提升决策准确性",
            "定期评估业务目标达成情况，调整策略方向",
            "建立更完善的反馈循环机制，快速响应市场变化"
        ]
    
    def _generate_html_report(self, report: Report) -> str:
        """生成HTML格式报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{report.title}</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .recommendation {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e9ecef; border-radius: 5px; }}
                h1, h2, h3 {{ color: #333; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p class="timestamp">生成时间: {report.generated_at.strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                <p>报告ID: {report.report_id}</p>
            </div>
            
            <div class="section">
                <h2>执行摘要</h2>
                <p>{report.content['executive_summary']}</p>
            </div>
            
            <div class="section">
                <h2>关键指标概览</h2>
                <div class="metrics-grid">
                    <!-- 这里会插入关键指标 -->
                </div>
            </div>
            
            <div class="section">
                <h2>可视化分析</h2>
        """
        
        # 添加可视化图表
        for i, chart_path in enumerate(report.visualizations):
            if os.path.exists(chart_path):
                # 将图片转换为base64编码嵌入HTML
                with open(chart_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                
                chart_name = os.path.basename(chart_path).replace('.png', '')
                html_content += f"""
                <div class="chart">
                    <h3>{chart_name.replace('_', ' ').title()}</h3>
                    <img src="data:image/png;base64,{img_data}" alt="{chart_name}">
                </div>
                """
        
        # 添加建议部分
        html_content += """
            </div>
            
            <div class="section">
                <h2>优化建议</h2>
        """
        
        for rec in report.content.get('recommendations', []):
            html_content += f"""
            <div class="recommendation">
                <h4>{rec['category']} (优先级: {rec['priority']})</h4>
                <p>{rec['description']}</p>
                <p><strong>预期影响:</strong> {rec['expected_impact']}</p>
            </div>
            """
        
        # 添加下一步行动
        html_content += """
            </div>
            
            <div class="section">
                <h2>下一步行动</h2>
                <ul>
        """
        
        for step in report.content.get('next_steps', []):
            html_content += f"<li>{step}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>附录</h2>
                <p><strong>数据来源:</strong> {}</p>
                <p><strong>分析方法:</strong> {}</p>
                <p><strong>生成系统:</strong> {}</p>
            </div>
        </body>
        </html>
        """.format(
            report.content['appendix']['data_sources'],
            report.content['appendix']['methodology'],
            report.content['appendix']['generated_by']
        )
        
        return html_content

class InteractionModule:
    """交互模块实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.nlp_processor = NaturalLanguageProcessor(config)
        self.visualization_engine = VisualizationEngine(config)
        self.report_generator = ReportGenerator(config)
        
        # 会话管理
        self.active_sessions: Dict[str, UserSession] = {}
        self.message_history: List[Message] = []
        
        # 通知系统
        self.notification_channels = config.get('notification_channels', {})
        
        # API集成配置
        self.api_integrations = config.get('api_integrations', {})
    
    async def process_user_message(self, user_id: str, message_content: str,
                                 interaction_type: InteractionType = InteractionType.CHAT) -> Dict[str, Any]:
        """处理用户消息"""
        session = self._get_or_create_session(user_id)
        
        # 处理用户输入
        processed_input = self.nlp_processor.process_user_input(
            message_content, 
            session.context
        )
        
        # 生成响应
        response_text = self.nlp_processor.generate_response(
            processed_input,
            session.context
        )
        
        # 创建消息记录
        user_message = Message(
            message_id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            sender=user_id,
            recipient="system",
            content=message_content,
            message_type=MessageType.USER_QUERY,
            interaction_type=interaction_type,
            timestamp=datetime.now()
        )
        
        system_message = Message(
            message_id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_resp",
            sender="system",
            recipient=user_id,
            content=response_text,
            message_type=MessageType.SYSTEM_RESPONSE,
            interaction_type=interaction_type,
            timestamp=datetime.now()
        )
        
        # 更新会话
        session.messages.extend([user_message, system_message])
        session.last_activity = datetime.now()
        self.message_history.extend([user_message, system_message])
        
        return {
            "response": response_text,
            "processed_input": processed_input,
            "session_id": session.session_id,
            "suggestions": self._generate_suggestions(processed_input, session.context)
        }
    
    def _get_or_create_session(self, user_id: str) -> UserSession:
        """获取或创建用户会话"""
        # 查找现有活跃会话
        for session in self.active_sessions.values():
            if (session.user_id == user_id and 
                datetime.now() - session.last_activity < timedelta(hours=1)):
                return session
        
        # 创建新会话
        session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            messages=[],
            context={},
            preferences={}
        )
        
        self.active_sessions[session_id] = session
        return session
    
    def _generate_suggestions(self, processed_input: Dict[str, Any], 
                            context: Dict[str, Any]) -> List[str]:
        """生成建议"""
        intent = processed_input.get('intent', '其他')
        
        suggestions = {
            "查询状态": [
                "查看任务执行进度",
                "获取系统性能报告",
                "查看用户满意度数据"
            ],
            "任务管理": [
                "创建新的分析任务",
                "查看历史任务记录",
                "设置任务优先级"
            ],
            "数据分析": [
                "生成业务指标报告",
                "分析用户行为趋势",
                "查看竞品对比分析"
            ],
            "优化建议": [
                "获取性能优化方案",
                "查看AI学习建议",
                "分析改进机会"
            ]
        }
        
        return suggestions.get(intent, ["如何使用AI产品经理?", "查看帮助文档", "联系技术支持"])
    
    def generate_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """生成仪表板数据"""
        # 模拟仪表板数据
        dashboard_data = {
            "overview": {
                "total_tasks": 45,
                "completed_tasks": 38,
                "success_rate": 84.4,
                "avg_completion_time": 2.3  # 小时
            },
            "performance_metrics": {
                "system_uptime": 99.8,
                "response_time": 150,  # ms
                "throughput": 1250,  # requests/hour
                "error_rate": 0.2
            },
            "business_kpis": {
                "monthly_revenue": 156000,
                "new_users": 1240,
                "user_retention": 78.5,
                "conversion_rate": 12.3
            },
            "ai_insights": {
                "learning_score": 0.85,
                "optimization_count": 23,
                "improvement_rate": 15.6,
                "confidence_level": 0.92
            },
            "recent_activities": [
                {"time": "2小时前", "activity": "完成用户行为分析报告"},
                {"time": "4小时前", "activity": "优化推荐算法参数"},
                {"time": "6小时前", "activity": "处理客户反馈数据"},
                {"time": "8小时前", "activity": "生成月度业务报告"}
            ],
            "alerts": [
                {"level": "warning", "message": "API响应时间略有上升"},
                {"level": "info", "message": "新用户增长超预期"}
            ]
        }
        
        return dashboard_data
    
    async def send_notification(self, recipient: str, message: str, 
                              notification_type: str = "info",
                              channel: str = "default") -> bool:
        """发送通知"""
        try:
            notification_config = self.notification_channels.get(channel, {})
            
            if channel == "email":
                return await self._send_email_notification(recipient, message, notification_config)
            elif channel == "webhook":
                return await self._send_webhook_notification(recipient, message, notification_config)
            elif channel == "sms":
                return await self._send_sms_notification(recipient, message, notification_config)
            else:
                # 默认记录到日志
                self.logger.info(f"通知发送给 {recipient}: {message}")
                return True
                
        except Exception as e:
            self.logger.error(f"发送通知失败: {e}")
            return False
    
    async def _send_email_notification(self, recipient: str, message: str, 
                                     config: Dict[str, Any]) -> bool:
        """发送邮件通知（模拟实现）"""
        # 实际实现需要集成邮件服务
        self.logger.info(f"邮件通知发送给 {recipient}: {message}")
        return True
    
    async def _send_webhook_notification(self, recipient: str, message: str,
                                       config: Dict[str, Any]) -> bool:
        """发送Webhook通知"""
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            return False
        
        payload = {
            "recipient": recipient,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "source": "AI-Product Manager"
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Webhook通知发送失败: {e}")
            return False
    
    async def _send_sms_notification(self, recipient: str, message: str,
                                   config: Dict[str, Any]) -> bool:
        """发送短信通知（模拟实现）"""
        # 实际实现需要集成短信服务
        self.logger.info(f"短信通知发送给 {recipient}: {message}")
        return True
    
    def integrate_external_system(self, system_name: str, api_config: Dict[str, Any]) -> bool:
        """集成外部系统"""
        try:
            self.api_integrations[system_name] = api_config
            self.logger.info(f"成功集成外部系统: {system_name}")
            return True
        except Exception as e:
            self.logger.error(f"集成外部系统失败 {system_name}: {e}")
            return False
    
    async def sync_with_external_system(self, system_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """与外部系统同步数据"""
        if system_name not in self.api_integrations:
            return {"error": f"未配置系统: {system_name}"}
        
        config = self.api_integrations[system_name]
        
        try:
            # 模拟API调用
            response = requests.post(
                config['endpoint'],
                json=data,
                headers=config.get('headers', {}),
                timeout=30
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"error": f"API调用失败: {response.status_code}"}
                
        except Exception as e:
            self.logger.error(f"外部系统同步失败 {system_name}: {e}")
            return {"error": str(e)}
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """获取交互模块摘要"""
        return {
            "active_sessions": len(self.active_sessions),
            "total_messages": len(self.message_history),
            "integrated_systems": len(self.api_integrations),
            "notification_channels": len(self.notification_channels),
            "recent_interactions": len([msg for msg in self.message_history 
                                      if datetime.now() - msg.timestamp < timedelta(hours=24)])
        }