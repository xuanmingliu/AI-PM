"""
决策模块 (Decision Module)
负责任务优先级排序、场景选择、模型选型、评价体系搭建、商业模式设计
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

class PriorityLevel(Enum):
    """优先级级别"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1

class ScenarioType(Enum):
    """场景类型"""
    RECOMMENDATION = "推荐系统"
    NLP = "自然语言处理"
    COMPUTER_VISION = "计算机视觉"
    PREDICTIVE_ANALYTICS = "预测分析"
    CHATBOT = "智能对话"

@dataclass
class Task:
    """任务数据结构"""
    task_id: str
    name: str
    description: str
    urgency: float  # 紧迫性 0-1
    impact: float   # 影响力 0-1
    feasibility: float  # 可行性 0-1
    resource_cost: float  # 资源成本 0-1
    priority_score: float = 0.0
    assigned_scenario: Optional[str] = None

@dataclass
class Scenario:
    """场景数据结构"""
    scenario_id: str
    name: str
    type: ScenarioType
    market_potential: float  # 市场潜力 0-1
    technical_complexity: float  # 技术复杂度 0-1
    user_demand: float  # 用户需求热度 0-1
    competitive_advantage: float  # 竞争优势 0-1
    implementation_cost: float  # 实施成本 0-1
    roi_potential: float  # ROI潜力 0-1

@dataclass
class ModelCandidate:
    """模型候选结构"""
    model_id: str
    name: str
    type: str  # LLM, CV, NLP等
    performance_score: float  # 性能分数 0-1
    resource_consumption: float  # 资源消耗 0-1
    accuracy: float  # 准确率 0-1
    latency: float  # 延迟(ms)
    cost_per_request: float  # 每次请求成本
    explainability: float  # 可解释性 0-1
    security_score: float  # 安全性分数 0-1
    maintenance_complexity: float  # 维护复杂度 0-1

@dataclass
class BusinessModel:
    """商业模式结构"""
    model_id: str
    name: str
    revenue_streams: List[str]
    target_customers: List[str]
    value_proposition: str
    cost_structure: Dict[str, float]
    revenue_projection: Dict[str, float]
    break_even_time: int  # 盈亏平衡时间(月)
    scalability_score: float  # 可扩展性分数 0-1

class DecisionModule:
    """决策模块实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 权重配置
        self.priority_weights = config.get('priority_weights', {
            'urgency': 0.3,
            'impact': 0.4,
            'feasibility': 0.2,
            'cost_efficiency': 0.1
        })
        
        self.scenario_weights = config.get('scenario_weights', {
            'market_potential': 0.25,
            'user_demand': 0.25,
            'technical_feasibility': 0.20,
            'competitive_advantage': 0.15,
            'roi_potential': 0.15
        })
        
        self.model_weights = config.get('model_weights', {
            'performance': 0.25,
            'efficiency': 0.20,
            'cost': 0.20,
            'reliability': 0.15,
            'explainability': 0.10,
            'security': 0.10
        })
        
    def prioritize_tasks(self, tasks: List[Task], context: Dict[str, Any] = None) -> List[Task]:
        """
        任务优先级排序
        根据紧迫性、影响力、可行性等因素计算优先级分数
        """
        self.logger.info(f"开始对{len(tasks)}个任务进行优先级排序")
        
        for task in tasks:
            # 计算优先级分数
            task.priority_score = self._calculate_task_priority(task, context)
            
        # 根据优先级分数排序
        sorted_tasks = sorted(tasks, key=lambda t: t.priority_score, reverse=True)
        
        self.logger.info("任务优先级排序完成")
        return sorted_tasks
    
    def _calculate_task_priority(self, task: Task, context: Dict[str, Any] = None) -> float:
        """计算单个任务的优先级分数"""
        weights = self.priority_weights
        
        # 成本效率 = 1 - 资源成本（成本越低，效率越高）
        cost_efficiency = 1.0 - task.resource_cost
        
        # 综合分数计算
        priority_score = (
            task.urgency * weights['urgency'] +
            task.impact * weights['impact'] +
            task.feasibility * weights['feasibility'] +
            cost_efficiency * weights['cost_efficiency']
        )
        
        # 考虑上下文因素
        if context:
            # 如果有业务紧急性，提高优先级
            if context.get('business_urgency', False):
                priority_score *= 1.2
            
            # 如果资源有限，降低高成本任务优先级
            if context.get('resource_constrained', False) and task.resource_cost > 0.7:
                priority_score *= 0.8
                
        return min(priority_score, 1.0)  # 确保分数不超过1
    
    def select_optimal_scenario(self, scenarios: List[Scenario], 
                              requirements: Dict[str, Any] = None) -> Scenario:
        """
        场景选择 - 根据多维度评估选择最优落地场景
        """
        self.logger.info(f"开始从{len(scenarios)}个场景中选择最优方案")
        
        best_scenario = None
        best_score = 0.0
        
        for scenario in scenarios:
            score = self._evaluate_scenario(scenario, requirements)
            
            if score > best_score:
                best_score = score
                best_scenario = scenario
                
        self.logger.info(f"选择场景: {best_scenario.name}, 得分: {best_score:.3f}")
        return best_scenario
    
    def _evaluate_scenario(self, scenario: Scenario, requirements: Dict[str, Any] = None) -> float:
        """评估单个场景的综合得分"""
        weights = self.scenario_weights
        
        # 技术可行性 = 1 - 技术复杂度
        technical_feasibility = 1.0 - scenario.technical_complexity
        
        # 综合得分
        score = (
            scenario.market_potential * weights['market_potential'] +
            scenario.user_demand * weights['user_demand'] +
            technical_feasibility * weights['technical_feasibility'] +
            scenario.competitive_advantage * weights['competitive_advantage'] +
            scenario.roi_potential * weights['roi_potential']
        )
        
        # 根据需求调整分数
        if requirements:
            # 如果强调快速上市，降低复杂场景分数
            if requirements.get('time_to_market_critical', False):
                if scenario.technical_complexity > 0.7:
                    score *= 0.7
                    
            # 如果强调创新性，提高高潜力场景分数
            if requirements.get('innovation_focused', False):
                if scenario.competitive_advantage > 0.8:
                    score *= 1.3
                    
        return score
    
    def select_optimal_model(self, models: List[ModelCandidate], 
                           scenario_requirements: Dict[str, Any] = None) -> ModelCandidate:
        """
        模型选型 - 根据性能、成本、可解释性等因素选择最优模型
        """
        self.logger.info(f"开始从{len(models)}个模型中选择最优方案")
        
        best_model = None
        best_score = 0.0
        
        for model in models:
            score = self._evaluate_model(model, scenario_requirements)
            
            if score > best_score:
                best_score = score
                best_model = model
                
        self.logger.info(f"选择模型: {best_model.name}, 得分: {best_score:.3f}")
        return best_model
    
    def _evaluate_model(self, model: ModelCandidate, 
                       requirements: Dict[str, Any] = None) -> float:
        """评估单个模型的综合得分"""
        weights = self.model_weights
        
        # 效率 = 1 - 资源消耗
        efficiency = 1.0 - model.resource_consumption
        
        # 成本效益 = 1 - 标准化成本
        cost_effectiveness = 1.0 - min(model.cost_per_request / 0.1, 1.0)  # 假设0.1为高成本阈值
        
        # 可靠性 = (准确率 + (1-维护复杂度)) / 2
        reliability = (model.accuracy + (1.0 - model.maintenance_complexity)) / 2
        
        # 综合得分
        score = (
            model.performance_score * weights['performance'] +
            efficiency * weights['efficiency'] +
            cost_effectiveness * weights['cost'] +
            reliability * weights['reliability'] +
            model.explainability * weights['explainability'] +
            model.security_score * weights['security']
        )
        
        # 根据场景需求调整分数
        if requirements:
            # 金融场景需要高安全性和可解释性
            if requirements.get('scenario_type') == 'financial':
                if model.security_score > 0.8 and model.explainability > 0.7:
                    score *= 1.2
                    
            # 实时场景需要低延迟
            if requirements.get('real_time_required', False):
                if model.latency < 100:  # 100ms以下
                    score *= 1.1
                elif model.latency > 500:  # 500ms以上
                    score *= 0.8
                    
        return score
    
    def design_evaluation_system(self, product_type: str, 
                                business_goals: List[str]) -> Dict[str, Any]:
        """
        评价体系设计 - 根据产品类型和业务目标设计评价指标体系
        """
        self.logger.info(f"为{product_type}产品设计评价体系")
        
        evaluation_system = {
            "technical_metrics": self._get_technical_metrics(product_type),
            "business_metrics": self._get_business_metrics(business_goals),
            "user_experience_metrics": self._get_ux_metrics(),
            "operational_metrics": self._get_operational_metrics(),
            "weights": self._assign_metric_weights(product_type, business_goals),
            "thresholds": self._set_metric_thresholds(product_type),
            "reporting_frequency": self._determine_reporting_frequency(product_type)
        }
        
        return evaluation_system
    
    def _get_technical_metrics(self, product_type: str) -> List[Dict[str, str]]:
        """获取技术指标"""
        base_metrics = [
            {"name": "accuracy", "description": "模型准确率", "unit": "%"},
            {"name": "latency", "description": "响应延迟", "unit": "ms"},
            {"name": "throughput", "description": "吞吐量", "unit": "requests/s"},
            {"name": "error_rate", "description": "错误率", "unit": "%"}
        ]
        
        # 根据产品类型添加特定指标
        if product_type == "recommendation":
            base_metrics.extend([
                {"name": "precision", "description": "推荐精确率", "unit": "%"},
                {"name": "recall", "description": "推荐召回率", "unit": "%"},
                {"name": "diversity", "description": "推荐多样性", "unit": "score"}
            ])
        elif product_type == "nlp":
            base_metrics.extend([
                {"name": "bleu_score", "description": "BLEU分数", "unit": "score"},
                {"name": "rouge_score", "description": "ROUGE分数", "unit": "score"}
            ])
            
        return base_metrics
    
    def _get_business_metrics(self, business_goals: List[str]) -> List[Dict[str, str]]:
        """获取业务指标"""
        metrics = []
        
        if "revenue_growth" in business_goals:
            metrics.extend([
                {"name": "revenue", "description": "收入", "unit": "currency"},
                {"name": "arpu", "description": "每用户平均收入", "unit": "currency"},
                {"name": "conversion_rate", "description": "转化率", "unit": "%"}
            ])
            
        if "user_acquisition" in business_goals:
            metrics.extend([
                {"name": "new_users", "description": "新用户数", "unit": "count"},
                {"name": "acquisition_cost", "description": "获客成本", "unit": "currency"},
                {"name": "user_growth_rate", "description": "用户增长率", "unit": "%"}
            ])
            
        if "user_retention" in business_goals:
            metrics.extend([
                {"name": "retention_rate", "description": "留存率", "unit": "%"},
                {"name": "churn_rate", "description": "流失率", "unit": "%"},
                {"name": "lifetime_value", "description": "用户生命周期价值", "unit": "currency"}
            ])
            
        return metrics
    
    def _get_ux_metrics(self) -> List[Dict[str, str]]:
        """获取用户体验指标"""
        return [
            {"name": "satisfaction_score", "description": "用户满意度", "unit": "score"},
            {"name": "nps", "description": "净推荐值", "unit": "score"},
            {"name": "task_completion_rate", "description": "任务完成率", "unit": "%"},
            {"name": "time_to_complete", "description": "任务完成时间", "unit": "seconds"}
        ]
    
    def _get_operational_metrics(self) -> List[Dict[str, str]]:
        """获取运营指标"""
        return [
            {"name": "uptime", "description": "系统可用性", "unit": "%"},
            {"name": "support_tickets", "description": "客服工单数", "unit": "count"},
            {"name": "deployment_frequency", "description": "部署频率", "unit": "per_month"},
            {"name": "mttr", "description": "平均修复时间", "unit": "hours"}
        ]
    
    def _assign_metric_weights(self, product_type: str, 
                              business_goals: List[str]) -> Dict[str, float]:
        """分配指标权重"""
        weights = {
            "technical_metrics": 0.3,
            "business_metrics": 0.4,
            "user_experience_metrics": 0.2,
            "operational_metrics": 0.1
        }
        
        # 根据产品类型和业务目标调整权重
        if "user_retention" in business_goals:
            weights["user_experience_metrics"] = 0.3
            weights["business_metrics"] = 0.3
            
        return weights
    
    def _set_metric_thresholds(self, product_type: str) -> Dict[str, Dict[str, float]]:
        """设置指标阈值"""
        return {
            "accuracy": {"good": 0.9, "acceptable": 0.8, "poor": 0.7},
            "latency": {"good": 100, "acceptable": 300, "poor": 500},
            "error_rate": {"good": 0.01, "acceptable": 0.05, "poor": 0.1},
            "satisfaction_score": {"good": 4.5, "acceptable": 4.0, "poor": 3.5}
        }
    
    def _determine_reporting_frequency(self, product_type: str) -> Dict[str, str]:
        """确定报告频率"""
        return {
            "technical_metrics": "daily",
            "business_metrics": "weekly",
            "user_experience_metrics": "weekly", 
            "operational_metrics": "daily"
        }
    
    def design_business_model(self, market_data: Dict[str, Any],
                            user_segments: List[str],
                            value_propositions: List[str]) -> List[BusinessModel]:
        """
        商业模式设计 - 基于市场数据和用户细分设计商业模式
        """
        self.logger.info("开始设计商业模式")
        
        business_models = []
        
        # 订阅模式
        subscription_model = self._design_subscription_model(market_data, user_segments)
        business_models.append(subscription_model)
        
        # 按使用付费模式
        usage_model = self._design_usage_based_model(market_data, user_segments)
        business_models.append(usage_model)
        
        # 免费增值模式
        freemium_model = self._design_freemium_model(market_data, user_segments)
        business_models.append(freemium_model)
        
        # 企业许可模式
        enterprise_model = self._design_enterprise_model(market_data, user_segments)
        business_models.append(enterprise_model)
        
        # 评估并排序商业模式
        evaluated_models = self._evaluate_business_models(business_models, market_data)
        
        return evaluated_models
    
    def _design_subscription_model(self, market_data: Dict[str, Any], 
                                 user_segments: List[str]) -> BusinessModel:
        """设计订阅模式"""
        return BusinessModel(
            model_id="subscription",
            name="订阅模式",
            revenue_streams=["月度订阅费", "年度订阅费", "高级功能订阅"],
            target_customers=["中小企业", "个人用户"],
            value_proposition="稳定的AI服务，可预测的成本",
            cost_structure={
                "development": 0.3,
                "infrastructure": 0.25,
                "marketing": 0.2,
                "support": 0.15,
                "administration": 0.1
            },
            revenue_projection={
                "month_1": 10000,
                "month_6": 50000,
                "month_12": 120000,
                "month_24": 300000
            },
            break_even_time=8,
            scalability_score=0.85
        )
    
    def _design_usage_based_model(self, market_data: Dict[str, Any],
                                user_segments: List[str]) -> BusinessModel:
        """设计按使用付费模式"""
        return BusinessModel(
            model_id="usage_based",
            name="按使用付费",
            revenue_streams=["API调用费", "数据处理费", "存储费"],
            target_customers=["大企业", "开发者"],
            value_proposition="按需付费，成本与使用量成正比",
            cost_structure={
                "infrastructure": 0.4,
                "development": 0.25,
                "support": 0.15,
                "marketing": 0.1,
                "administration": 0.1
            },
            revenue_projection={
                "month_1": 5000,
                "month_6": 40000,
                "month_12": 100000,
                "month_24": 280000
            },
            break_even_time=10,
            scalability_score=0.9
        )
    
    def _design_freemium_model(self, market_data: Dict[str, Any],
                             user_segments: List[str]) -> BusinessModel:
        """设计免费增值模式"""
        return BusinessModel(
            model_id="freemium",
            name="免费增值",
            revenue_streams=["高级功能订阅", "广告收入", "企业版授权"],
            target_customers=["个人用户", "小企业", "学生"],
            value_proposition="免费体验，付费升级高级功能",
            cost_structure={
                "infrastructure": 0.35,
                "development": 0.3,
                "marketing": 0.25,
                "support": 0.1
            },
            revenue_projection={
                "month_1": 2000,
                "month_6": 25000,
                "month_12": 80000,
                "month_24": 250000
            },
            break_even_time=15,
            scalability_score=0.95
        )
    
    def _design_enterprise_model(self, market_data: Dict[str, Any],
                               user_segments: List[str]) -> BusinessModel:
        """设计企业许可模式"""
        return BusinessModel(
            model_id="enterprise",
            name="企业许可",
            revenue_streams=["年度许可费", "定制开发费", "技术支持费"],
            target_customers=["大型企业", "政府机构"],
            value_proposition="定制化解决方案，专业技术支持",
            cost_structure={
                "development": 0.4,
                "sales": 0.25,
                "support": 0.2,
                "infrastructure": 0.1,
                "administration": 0.05
            },
            revenue_projection={
                "month_1": 50000,
                "month_6": 200000,
                "month_12": 500000,
                "month_24": 1200000
            },
            break_even_time=6,
            scalability_score=0.7
        )
    
    def _evaluate_business_models(self, models: List[BusinessModel],
                                market_data: Dict[str, Any]) -> List[BusinessModel]:
        """评估和排序商业模式"""
        model_scores = []
        
        for model in models:
            score = self._calculate_business_model_score(model, market_data)
            model_scores.append((model, score))
            
        # 按分数排序
        sorted_models = sorted(model_scores, key=lambda x: x[1], reverse=True)
        return [model for model, score in sorted_models]
    
    def _calculate_business_model_score(self, model: BusinessModel,
                                      market_data: Dict[str, Any]) -> float:
        """计算商业模式综合得分"""
        # 收入潜力 (基于24个月收入预测)
        revenue_potential = model.revenue_projection.get("month_24", 0) / 1000000  # 标准化到0-1
        
        # 盈亏平衡速度 (越快越好)
        breakeven_score = max(0, (24 - model.break_even_time) / 24)
        
        # 可扩展性
        scalability = model.scalability_score
        
        # 市场适配度 (简化计算)
        market_fit = market_data.get("growth_rate", 0.1) * 5  # 假设增长率影响适配度
        
        # 综合得分
        score = (
            revenue_potential * 0.3 +
            breakeven_score * 0.3 +
            scalability * 0.25 +
            market_fit * 0.15
        )
        
        return min(score, 1.0)