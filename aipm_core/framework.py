"""
AI-Product Manager 主框架
整合五个核心模块，实现端到端的自主产品管理系统
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .perception_module import PerceptionModule, UserFeedback, MarketData
from .decision_module import DecisionModule, Task, Scenario, ModelCandidate, BusinessModel
from .execution_module import ExecutionModule, ExecutionTask, TaskType, DatasetSpec, Campaign
from .learning_module import LearningModule, RewardType, ActionType, Experience
from .interaction_module import InteractionModule, InteractionType, MessageType, ReportType

@dataclass
class AIPMStatus:
    """AI-Product Manager状态"""
    system_status: str
    active_modules: List[str]
    running_tasks: int
    total_experiences: int
    last_update: datetime
    performance_score: float

@dataclass
class ProductInsight:
    """产品洞察"""
    insight_id: str
    category: str
    title: str
    description: str
    confidence: float
    impact_score: float
    recommended_actions: List[str]
    timestamp: datetime

class AIPMFramework:
    """AI-Product Manager 主框架"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化五个核心模块
        self.perception = PerceptionModule(config.get('perception', {}))
        self.decision = DecisionModule(config.get('decision', {}))
        self.execution = ExecutionModule(config.get('execution', {}))
        self.learning = LearningModule(config.get('learning', {}))
        self.interaction = InteractionModule(config.get('interaction', {}))
        
        # 系统状态
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # 产品洞察存储
        self.product_insights: List[ProductInsight] = []
        
        # 配置参数
        self.update_interval = config.get('update_interval', 3600)  # 秒
        self.auto_report_interval = config.get('auto_report_interval', 86400)  # 日报
        
        self.logger.info("AI-Product Manager框架初始化完成")
    
    async def start(self):
        """启动AI-Product Manager系统"""
        if self.is_running:
            self.logger.warning("系统已在运行中")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        self.logger.info("启动AI-Product Manager系统")
        
        # 启动各模块的后台任务
        background_tasks = [
            asyncio.create_task(self._perception_loop()),
            asyncio.create_task(self._decision_loop()),
            asyncio.create_task(self._execution_loop()),
            asyncio.create_task(self._learning_loop()),
            asyncio.create_task(self._auto_report_loop())
        ]
        
        try:
            await asyncio.gather(*background_tasks)
        except Exception as e:
            self.logger.error(f"系统运行异常: {e}")
            await self.stop()
    
    async def stop(self):
        """停止AI-Product Manager系统"""
        self.is_running = False
        self.logger.info("停止AI-Product Manager系统")
        
        # 保存学习数据
        self.learning.save_learning_data()
    
    async def process_user_request(self, user_id: str, request: str, 
                                 request_type: str = "general") -> Dict[str, Any]:
        """处理用户请求"""
        self.logger.info(f"处理用户请求: {user_id} - {request_type}")
        
        # 通过交互模块处理
        response = await self.interaction.process_user_message(
            user_id, request, InteractionType.CHAT
        )
        
        # 根据请求类型执行相应操作
        if request_type == "需求分析":
            return await self._handle_requirement_analysis(request, response)
        elif request_type == "场景选择":
            return await self._handle_scenario_selection(request, response)
        elif request_type == "模型选型":
            return await self._handle_model_selection(request, response)
        elif request_type == "商业分析":
            return await self._handle_business_analysis(request, response)
        elif request_type == "优化建议":
            return await self._handle_optimization_request(request, response)
        else:
            return response
    
    async def _handle_requirement_analysis(self, request: str, 
                                         base_response: Dict[str, Any]) -> Dict[str, Any]:
        """处理需求分析请求"""
        # 收集相关数据
        user_needs = await self.perception.collect_user_needs()
        market_data = await self.perception.collect_market_data()
        
        # 分析需求优先级
        analysis_results = {
            "user_needs": user_needs,
            "market_insights": market_data,
            "priority_analysis": self._analyze_requirement_priority(user_needs, market_data),
            "recommendations": self._generate_requirement_recommendations(user_needs, market_data)
        }
        
        base_response["requirement_analysis"] = analysis_results
        return base_response
    
    async def _handle_scenario_selection(self, request: str, 
                                       base_response: Dict[str, Any]) -> Dict[str, Any]:
        """处理场景选择请求"""
        # 模拟场景选择
        scenarios = [
            Scenario(
                scenario_id="scenario_1",
                name="智能推荐系统",
                type=self.decision.ScenarioType.RECOMMENDATION,
                market_potential=0.8,
                technical_complexity=0.6,
                user_demand=0.9,
                competitive_advantage=0.7,
                implementation_cost=0.5,
                roi_potential=0.8
            ),
            Scenario(
                scenario_id="scenario_2", 
                name="自然语言处理",
                type=self.decision.ScenarioType.NLP,
                market_potential=0.7,
                technical_complexity=0.8,
                user_demand=0.6,
                competitive_advantage=0.8,
                implementation_cost=0.7,
                roi_potential=0.6
            )
        ]
        
        # 选择最优场景
        optimal_scenario = self.decision.select_optimal_scenario(scenarios)
        
        base_response["scenario_selection"] = {
            "available_scenarios": [asdict(s) for s in scenarios],
            "recommended_scenario": asdict(optimal_scenario),
            "selection_reasoning": f"基于市场潜力({optimal_scenario.market_potential:.2f})和用户需求({optimal_scenario.user_demand:.2f})选择"
        }
        
        return base_response
    
    async def _handle_model_selection(self, request: str,
                                    base_response: Dict[str, Any]) -> Dict[str, Any]:
        """处理模型选型请求"""
        # 模拟模型候选
        model_candidates = [
            ModelCandidate(
                model_id="model_1",
                name="GPT-4",
                type="LLM",
                performance_score=0.9,
                resource_consumption=0.8,
                accuracy=0.92,
                latency=200,
                cost_per_request=0.02,
                explainability=0.6,
                security_score=0.8,
                maintenance_complexity=0.4
            ),
            ModelCandidate(
                model_id="model_2",
                name="Claude-3.5",
                type="LLM", 
                performance_score=0.88,
                resource_consumption=0.7,
                accuracy=0.89,
                latency=150,
                cost_per_request=0.015,
                explainability=0.7,
                security_score=0.85,
                maintenance_complexity=0.3
            )
        ]
        
        # 选择最优模型
        optimal_model = self.decision.select_optimal_model(model_candidates)
        
        base_response["model_selection"] = {
            "available_models": [asdict(m) for m in model_candidates],
            "recommended_model": asdict(optimal_model),
            "selection_reasoning": f"综合性能分数最高({optimal_model.performance_score:.2f})，成本效益较好"
        }
        
        return base_response
    
    async def _handle_business_analysis(self, request: str,
                                      base_response: Dict[str, Any]) -> Dict[str, Any]:
        """处理商业分析请求"""
        # 收集市场数据
        market_data = await self.perception.collect_market_data()
        
        # 设计商业模式
        business_models = self.decision.design_business_model(
            market_data=market_data,
            user_segments=["企业用户", "个人用户"],
            value_propositions=["AI驱动的产品优化", "自动化决策支持"]
        )
        
        base_response["business_analysis"] = {
            "market_data": market_data,
            "business_models": [asdict(bm) for bm in business_models],
            "recommended_model": asdict(business_models[0]) if business_models else None,
            "roi_projections": self._calculate_roi_projections(business_models[0]) if business_models else {}
        }
        
        return base_response
    
    async def _handle_optimization_request(self, request: str,
                                         base_response: Dict[str, Any]) -> Dict[str, Any]:
        """处理优化建议请求"""
        # 获取当前系统状态
        current_state = {
            "performance_metrics": await self._get_performance_metrics(),
            "user_feedback": await self._get_recent_user_feedback(),
            "business_metrics": await self._get_business_metrics()
        }
        
        # 生成优化建议
        optimization_suggestions = {}
        
        for action_type in ActionType:
            suggestion = self.learning.suggest_optimization(current_state, action_type)
            if suggestion:
                optimization_suggestions[action_type.value] = suggestion
        
        base_response["optimization_suggestions"] = {
            "current_state": current_state,
            "suggestions": optimization_suggestions,
            "implementation_priority": self._prioritize_optimizations(optimization_suggestions)
        }
        
        return base_response
    
    async def conduct_comprehensive_analysis(self) -> Dict[str, Any]:
        """进行综合分析"""
        self.logger.info("开始综合分析")
        
        # 并行收集所有数据
        results = await asyncio.gather(
            self.perception.collect_user_needs(),
            self.perception.collect_market_data(),
            self.perception.collect_user_behavior(),
            self.perception.detect_badcases(),
            return_exceptions=True
        )
        
        user_needs, market_data, user_behavior, badcases = results
        
        # 生成产品洞察
        insights = self._generate_product_insights(
            user_needs, market_data, user_behavior, badcases
        )
        
        # 更新产品洞察库
        self.product_insights.extend(insights)
        
        # 学习模块分析
        if badcases and not isinstance(badcases, Exception):
            badcase_analysis = self.learning.analyze_badcases(badcases)
        else:
            badcase_analysis = {}
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "user_needs": user_needs,
            "market_data": market_data,
            "user_behavior": user_behavior,
            "product_insights": [asdict(insight) for insight in insights],
            "badcase_analysis": badcase_analysis,
            "recommendations": self._generate_comprehensive_recommendations(insights)
        }
    
    def _generate_product_insights(self, user_needs: Dict[str, Any], 
                                 market_data: Dict[str, Any],
                                 user_behavior: Dict[str, Any],
                                 badcases: List[Dict[str, Any]]) -> List[ProductInsight]:
        """生成产品洞察"""
        insights = []
        
        # 基于用户需求的洞察
        if user_needs and user_needs.get('pain_points'):
            top_pain_point = max(user_needs['pain_points'], 
                                key=lambda x: x.get('severity', 0))
            
            insight = ProductInsight(
                insight_id=f"insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pain",
                category="用户痛点",
                title=f"核心痛点: {top_pain_point.get('type', '未知')}",
                description=top_pain_point.get('content', ''),
                confidence=0.8,
                impact_score=top_pain_point.get('severity', 0) / 3.0,
                recommended_actions=[
                    "优化相关功能体验",
                    "增加用户引导",
                    "改进产品设计"
                ],
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        # 基于市场数据的洞察
        if market_data and market_data.get('competitor_analysis'):
            market_insight = ProductInsight(
                insight_id=f"insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}_market",
                category="市场机会",
                title="竞争格局分析",
                description=f"发现{len(market_data['competitor_analysis'])}个主要竞争对手",
                confidence=0.7,
                impact_score=0.6,
                recommended_actions=[
                    "差异化产品定位",
                    "加强核心优势",
                    "寻找市场空白"
                ],
                timestamp=datetime.now()
            )
            insights.append(market_insight)
        
        # 基于用户行为的洞察
        if user_behavior and user_behavior.get('retention_metrics'):
            retention_score = user_behavior['retention_metrics'].get('day_7_retention', 0)
            if retention_score < 0.5:
                retention_insight = ProductInsight(
                    insight_id=f"insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}_retention",
                    category="用户留存",
                    title="留存率偏低警告",
                    description=f"7日留存率仅为{retention_score:.1%}，需要紧急优化",
                    confidence=0.9,
                    impact_score=0.8,
                    recommended_actions=[
                        "优化新用户引导流程",
                        "增加用户粘性功能",
                        "改善核心体验"
                    ],
                    timestamp=datetime.now()
                )
                insights.append(retention_insight)
        
        return insights
    
    def _generate_comprehensive_recommendations(self, insights: List[ProductInsight]) -> List[Dict[str, Any]]:
        """生成综合建议"""
        recommendations = []
        
        # 按影响分数排序洞察
        sorted_insights = sorted(insights, key=lambda x: x.impact_score, reverse=True)
        
        for insight in sorted_insights[:5]:  # 取前5个最重要的洞察
            recommendation = {
                "based_on": insight.title,
                "category": insight.category,
                "priority": "高" if insight.impact_score > 0.7 else "中" if insight.impact_score > 0.4 else "低",
                "actions": insight.recommended_actions,
                "expected_impact": f"{insight.impact_score:.1%}",
                "confidence": f"{insight.confidence:.1%}"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _perception_loop(self):
        """感知模块循环"""
        while self.is_running:
            try:
                # 定期收集数据
                await self.perception.collect_user_needs()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"感知模块异常: {e}")
                await asyncio.sleep(60)
    
    async def _decision_loop(self):
        """决策模块循环"""
        while self.is_running:
            try:
                # 定期优化决策
                await asyncio.sleep(self.update_interval * 2)
            except Exception as e:
                self.logger.error(f"决策模块异常: {e}")
                await asyncio.sleep(60)
    
    async def _execution_loop(self):
        """执行模块循环"""
        while self.is_running:
            try:
                # 处理任务队列
                await self.execution.process_task_queue()
                await asyncio.sleep(30)
            except Exception as e:
                self.logger.error(f"执行模块异常: {e}")
                await asyncio.sleep(60)
    
    async def _learning_loop(self):
        """学习模块循环"""
        while self.is_running:
            try:
                # 定期学习优化
                performance_metrics = await self._get_performance_metrics()
                user_feedback = await self._get_recent_user_feedback()
                
                await self.learning.continuous_learning(performance_metrics, user_feedback)
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"学习模块异常: {e}")
                await asyncio.sleep(60)
    
    async def _auto_report_loop(self):
        """自动报告循环"""
        while self.is_running:
            try:
                # 生成定期报告
                await self._generate_auto_report()
                await asyncio.sleep(self.auto_report_interval)
            except Exception as e:
                self.logger.error(f"自动报告异常: {e}")
                await asyncio.sleep(3600)
    
    async def _generate_auto_report(self):
        """生成自动报告"""
        try:
            # 收集报告数据
            performance_data = await self._get_performance_metrics()
            business_data = await self._get_business_metrics()
            user_data = await self._get_user_metrics()
            learning_data = self.learning.get_learning_summary()
            
            # 生成报告
            report = self.interaction.report_generator.generate_comprehensive_report(
                performance_data=performance_data,
                business_data=business_data,
                user_data=user_data,
                learning_data=learning_data
            )
            
            self.logger.info(f"自动报告已生成: {report.report_id}")
            
        except Exception as e:
            self.logger.error(f"生成自动报告失败: {e}")
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "overall_score": 0.85,
            "response_time": 150,
            "throughput": 1000,
            "error_rate": 0.02,
            "uptime": 99.9,
            "resource_usage": {
                "cpu": 65,
                "memory": 70,
                "disk": 45
            },
            "task_completion": {
                "completed": 85,
                "failed": 10,
                "pending": 5
            }
        }
    
    async def _get_business_metrics(self) -> Dict[str, Any]:
        """获取业务指标"""
        return {
            "revenue_trend": [100, 120, 135, 145, 160],
            "user_growth": {"1月": 100, "2月": 150, "3月": 200, "4月": 280, "5月": 350},
            "conversion_funnel": {
                "访问": 100,
                "注册": 25,
                "试用": 15,
                "付费": 8
            },
            "roi_analysis": {
                "搜索广告": 1.5,
                "社交媒体": 0.8,
                "内容营销": 2.1,
                "推荐": 3.2
            }
        }
    
    async def _get_user_metrics(self) -> Dict[str, Any]:
        """获取用户指标"""
        return {
            "demographics": {"18-25": 25, "26-35": 45, "36-45": 20, "46+": 10},
            "retention_rates": {"day_1": 0.8, "day_7": 0.5, "day_30": 0.3},
            "feature_usage": {"搜索": 850, "推荐": 600, "分析": 300, "设置": 150},
            "satisfaction_scores": {
                "功能": 4.2,
                "性能": 3.8,
                "易用性": 4.0,
                "支持": 4.1
            }
        }
    
    async def _get_recent_user_feedback(self) -> Dict[str, Any]:
        """获取最近用户反馈"""
        return {
            "satisfaction_score": 4.1,
            "feedback_text": "产品功能不错，但界面还需要优化",
            "feedback_trend": "improving"
        }
    
    def _analyze_requirement_priority(self, user_needs: Dict[str, Any], 
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析需求优先级"""
        return {
            "high_priority": ["性能优化", "用户体验改进"],
            "medium_priority": ["新功能开发", "界面美化"],
            "low_priority": ["高级设置", "API扩展"]
        }
    
    def _generate_requirement_recommendations(self, user_needs: Dict[str, Any],
                                            market_data: Dict[str, Any]) -> List[str]:
        """生成需求建议"""
        return [
            "优先解决用户反馈最多的性能问题",
            "基于竞品分析开发差异化功能",
            "提升核心功能的用户体验",
            "增加用户留存和粘性功能"
        ]
    
    def _calculate_roi_projections(self, business_model: BusinessModel) -> Dict[str, float]:
        """计算ROI预测"""
        return {
            "6个月ROI": 1.2,
            "12个月ROI": 2.5,
            "24个月ROI": 4.8,
            "投资回收期": business_model.break_even_time
        }
    
    def _prioritize_optimizations(self, suggestions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优化建议优先级排序"""
        prioritized = []
        
        for action_type, suggestion in suggestions.items():
            priority_score = suggestion.get('estimated_impact', 0) * suggestion.get('confidence', 0.5)
            prioritized.append({
                "action_type": action_type,
                "suggestion": suggestion,
                "priority_score": priority_score
            })
        
        return sorted(prioritized, key=lambda x: x['priority_score'], reverse=True)
    
    def get_system_status(self) -> AIPMStatus:
        """获取系统状态"""
        return AIPMStatus(
            system_status="running" if self.is_running else "stopped",
            active_modules=[
                "perception", "decision", "execution", "learning", "interaction"
            ],
            running_tasks=len(self.execution.running_tasks),
            total_experiences=len(self.learning.experiences),
            last_update=datetime.now(),
            performance_score=0.85  # 模拟性能分数
        )
    
    def get_framework_summary(self) -> Dict[str, Any]:
        """获取框架摘要"""
        return {
            "system_status": asdict(self.get_system_status()),
            "module_summaries": {
                "perception": self.perception.get_perception_summary(),
                "execution": self.execution.get_execution_summary(),
                "learning": self.learning.get_learning_summary(),
                "interaction": self.interaction.get_interaction_summary()
            },
            "product_insights_count": len(self.product_insights),
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }