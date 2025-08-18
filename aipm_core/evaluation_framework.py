"""
AI Product Manager Evaluation Framework
Comprehensive evaluation system integrating all AI-PM components with Value-Bench
综合评估框架，整合所有AI产品管理组件并与Value-Bench系统协同
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
import statistics

# Import all AI-PM components
from .market_intelligence_agent import MarketIntelligenceAgent, MarketSegment
from .user_insight_generator import UserInsightGenerator, ProductDirection
from .product_definition_agent import ProductDefinitionAgent, DefinitionStatus
from .product_strategy_advisor import ProductStrategyAdvisor, ValidationDimension
from .product_documentation_agent import ProductDocumentationAgent, DocumentType
from .value_bench import ValueBench, ParticipantType, EvaluationDimension

class EvaluationMode(Enum):
    """评估模式"""
    COMPREHENSIVE = "comprehensive"  # 全面评估
    COMPONENT_SPECIFIC = "component_specific"  # 组件特定评估
    PERFORMANCE_BENCHMARK = "performance_benchmark"  # 性能基准测试
    COMPARATIVE_ANALYSIS = "comparative_analysis"  # 对比分析
    CONTINUOUS_MONITORING = "continuous_monitoring"  # 持续监控

class EvaluationScope(Enum):
    """评估范围"""
    SINGLE_TASK = "single_task"
    WORKFLOW_COMPLETE = "workflow_complete"
    COMPONENT_INTEGRATION = "component_integration"
    SYSTEM_PERFORMANCE = "system_performance"
    BENCHMARKING = "benchmarking"

class MetricCategory(Enum):
    """指标类别"""
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    INNOVATION = "innovation"
    CONSISTENCY = "consistency"
    USABILITY = "usability"
    SCALABILITY = "scalability"
    BUSINESS_VALUE = "business_value"

@dataclass
class EvaluationMetric:
    """评估指标"""
    metric_id: str
    name: str
    category: MetricCategory
    description: str
    measurement_method: str
    target_value: float
    weight: float
    unit: str
    aggregation_method: str

@dataclass
class ComponentEvaluation:
    """组件评估结果"""
    component_name: str
    version: str
    evaluation_date: datetime
    metrics: Dict[str, float]
    performance_score: float
    quality_indicators: Dict[str, Any]
    execution_time: float
    resource_usage: Dict[str, float]
    error_count: int
    success_rate: float
    recommendations: List[str]

@dataclass
class WorkflowEvaluation:
    """工作流评估结果"""
    workflow_id: str
    workflow_name: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    component_evaluations: List[ComponentEvaluation]
    integration_score: float
    overall_effectiveness: float
    user_satisfaction: float
    business_outcome: Dict[str, Any]
    bottlenecks: List[str]
    improvement_opportunities: List[str]

@dataclass
class SystemEvaluation:
    """系统评估结果"""
    evaluation_id: str
    evaluation_mode: EvaluationMode
    evaluation_scope: EvaluationScope
    system_version: str
    evaluation_period: Tuple[datetime, datetime]
    workflow_evaluations: List[WorkflowEvaluation]
    aggregate_metrics: Dict[str, float]
    benchmark_comparison: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    system_health: Dict[str, Any]
    strategic_insights: List[str]
    actionable_recommendations: List[str]

@dataclass
class EvaluationConfiguration:
    """评估配置"""
    config_id: str
    name: str
    description: str
    evaluation_mode: EvaluationMode
    evaluation_scope: EvaluationScope
    metrics: List[EvaluationMetric]
    thresholds: Dict[str, float]
    sampling_strategy: str
    reporting_frequency: str
    alert_conditions: List[Dict[str, Any]]

class AIProductManagerEvaluator:
    """AI产品管理器评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.market_intelligence = MarketIntelligenceAgent(config.get('market_intelligence', {}))
        self.user_insight_generator = UserInsightGenerator(config.get('user_insight', {}))
        self.product_definition = ProductDefinitionAgent(config.get('product_definition', {}))
        self.strategy_advisor = ProductStrategyAdvisor(config.get('strategy_advisor', {}))
        self.documentation_agent = ProductDocumentationAgent(config.get('documentation', {}))
        self.value_bench = ValueBench(config.get('value_bench', {}))
        
        # 评估配置
        self.evaluation_history: List[SystemEvaluation] = []
        self.evaluation_configs: Dict[str, EvaluationConfiguration] = {}
        self.active_monitoring: Dict[str, bool] = {}
        
        # 性能基线
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.quality_thresholds: Dict[str, float] = {
            'accuracy': 0.85,
            'completeness': 0.80,
            'coherence': 0.75,
            'innovation': 0.70,
            'efficiency': 0.80,
            'business_value': 0.75
        }
        
        # 初始化评估指标和配置
        self._initialize_evaluation_metrics()
        self._initialize_evaluation_configurations()
    
    def _initialize_evaluation_metrics(self):
        """初始化评估指标"""
        self.core_metrics = [
            EvaluationMetric(
                metric_id="accuracy_001",
                name="输出准确性",
                category=MetricCategory.ACCURACY,
                description="AI输出结果的准确性和正确性",
                measurement_method="专家评分+自动验证",
                target_value=0.85,
                weight=0.2,
                unit="score",
                aggregation_method="weighted_average"
            ),
            EvaluationMetric(
                metric_id="efficiency_001",
                name="处理效率",
                category=MetricCategory.EFFICIENCY,
                description="完成任务的时间效率",
                measurement_method="执行时间测量",
                target_value=0.8,
                weight=0.15,
                unit="ratio",
                aggregation_method="average"
            ),
            EvaluationMetric(
                metric_id="quality_001",
                name="输出质量",
                category=MetricCategory.QUALITY,
                description="输出内容的整体质量",
                measurement_method="多维度质量评估",
                target_value=0.8,
                weight=0.2,
                unit="score",
                aggregation_method="weighted_average"
            ),
            EvaluationMetric(
                metric_id="innovation_001",
                name="创新水平",
                category=MetricCategory.INNOVATION,
                description="解决方案的创新程度",
                measurement_method="创新指标评估",
                target_value=0.7,
                weight=0.15,
                unit="score",
                aggregation_method="average"
            ),
            EvaluationMetric(
                metric_id="consistency_001",
                name="一致性",
                category=MetricCategory.CONSISTENCY,
                description="多次执行结果的一致性",
                measurement_method="标准差计算",
                target_value=0.9,
                weight=0.1,
                unit="score",
                aggregation_method="average"
            ),
            EvaluationMetric(
                metric_id="business_value_001",
                name="商业价值",
                category=MetricCategory.BUSINESS_VALUE,
                description="生成结果的商业价值",
                measurement_method="商业影响评估",
                target_value=0.75,
                weight=0.2,
                unit="score",
                aggregation_method="weighted_average"
            )
        ]
    
    def _initialize_evaluation_configurations(self):
        """初始化评估配置"""
        # 综合评估配置
        comprehensive_config = EvaluationConfiguration(
            config_id="comprehensive_001",
            name="综合系统评估",
            description="对AI产品管理系统进行全面综合评估",
            evaluation_mode=EvaluationMode.COMPREHENSIVE,
            evaluation_scope=EvaluationScope.SYSTEM_PERFORMANCE,
            metrics=self.core_metrics,
            thresholds=self.quality_thresholds,
            sampling_strategy="stratified",
            reporting_frequency="weekly",
            alert_conditions=[
                {"metric": "accuracy", "threshold": 0.7, "action": "alert"},
                {"metric": "efficiency", "threshold": 0.6, "action": "warning"}
            ]
        )
        self.evaluation_configs["comprehensive"] = comprehensive_config
        
        # 性能基准配置
        benchmark_config = EvaluationConfiguration(
            config_id="benchmark_001",
            name="性能基准测试",
            description="与Value-Bench基准进行性能对比",
            evaluation_mode=EvaluationMode.PERFORMANCE_BENCHMARK,
            evaluation_scope=EvaluationScope.BENCHMARKING,
            metrics=self.core_metrics[:4],  # 主要性能指标
            thresholds={"benchmark_score": 0.8},
            sampling_strategy="random",
            reporting_frequency="monthly",
            alert_conditions=[]
        )
        self.evaluation_configs["benchmark"] = benchmark_config
        
        # 持续监控配置
        monitoring_config = EvaluationConfiguration(
            config_id="monitoring_001",
            name="持续性能监控",
            description="持续监控系统运行性能",
            evaluation_mode=EvaluationMode.CONTINUOUS_MONITORING,
            evaluation_scope=EvaluationScope.SYSTEM_PERFORMANCE,
            metrics=self.core_metrics,
            thresholds=self.quality_thresholds,
            sampling_strategy="continuous",
            reporting_frequency="daily",
            alert_conditions=[
                {"metric": "error_rate", "threshold": 0.05, "action": "immediate_alert"},
                {"metric": "response_time", "threshold": 30, "action": "warning"}
            ]
        )
        self.evaluation_configs["monitoring"] = monitoring_config
    
    async def conduct_comprehensive_evaluation(self,
                                             test_scenarios: List[Dict[str, Any]],
                                             config_name: str = "comprehensive") -> SystemEvaluation:
        """进行综合评估"""
        self.logger.info("开始综合系统评估")
        
        if config_name not in self.evaluation_configs:
            raise ValueError(f"评估配置不存在: {config_name}")
        
        config = self.evaluation_configs[config_name]
        evaluation_start = datetime.now()
        
        workflow_evaluations = []
        
        # 对每个测试场景进行评估
        for i, scenario in enumerate(test_scenarios):
            self.logger.info(f"评估场景 {i+1}/{len(test_scenarios)}: {scenario.get('name', 'Unknown')}")
            
            workflow_eval = await self._evaluate_workflow(scenario, config)
            workflow_evaluations.append(workflow_eval)
        
        evaluation_end = datetime.now()
        
        # 聚合分析
        aggregate_metrics = await self._calculate_aggregate_metrics(workflow_evaluations, config)
        
        # 基准对比
        benchmark_comparison = await self._perform_benchmark_comparison(workflow_evaluations)
        
        # 趋势分析
        trend_analysis = await self._analyze_performance_trends(workflow_evaluations)
        
        # 系统健康评估
        system_health = await self._assess_system_health(workflow_evaluations)
        
        # 生成洞察和建议
        strategic_insights = await self._generate_strategic_insights(workflow_evaluations, aggregate_metrics)
        actionable_recommendations = await self._generate_actionable_recommendations(workflow_evaluations, system_health)
        
        # 创建系统评估结果
        system_evaluation = SystemEvaluation(
            evaluation_id=str(uuid.uuid4()),
            evaluation_mode=config.evaluation_mode,
            evaluation_scope=config.evaluation_scope,
            system_version=self.config.get('version', '1.0.0'),
            evaluation_period=(evaluation_start, evaluation_end),
            workflow_evaluations=workflow_evaluations,
            aggregate_metrics=aggregate_metrics,
            benchmark_comparison=benchmark_comparison,
            trend_analysis=trend_analysis,
            system_health=system_health,
            strategic_insights=strategic_insights,
            actionable_recommendations=actionable_recommendations
        )
        
        # 保存评估结果
        self.evaluation_history.append(system_evaluation)
        
        self.logger.info(f"综合评估完成，总体得分: {aggregate_metrics.get('overall_score', 0):.3f}")
        return system_evaluation
    
    async def _evaluate_workflow(self,
                                scenario: Dict[str, Any],
                                config: EvaluationConfiguration) -> WorkflowEvaluation:
        """评估单个工作流"""
        workflow_start = datetime.now()
        component_evaluations = []
        
        # 1. 市场情报采集评估
        market_eval = await self._evaluate_market_intelligence_component(scenario)
        component_evaluations.append(market_eval)
        
        # 2. 用户洞察生成评估
        user_insight_eval = await self._evaluate_user_insight_component(scenario, market_eval)
        component_evaluations.append(user_insight_eval)
        
        # 3. 产品定义评估
        product_def_eval = await self._evaluate_product_definition_component(scenario, user_insight_eval)
        component_evaluations.append(product_def_eval)
        
        # 4. 策略顾问评估
        strategy_eval = await self._evaluate_strategy_advisor_component(scenario, product_def_eval)
        component_evaluations.append(strategy_eval)
        
        # 5. 文档生成评估
        doc_eval = await self._evaluate_documentation_component(scenario, strategy_eval)
        component_evaluations.append(doc_eval)
        
        workflow_end = datetime.now()
        total_duration = (workflow_end - workflow_start).total_seconds()
        
        # 计算集成分数
        integration_score = await self._calculate_integration_score(component_evaluations)
        
        # 计算整体有效性
        overall_effectiveness = await self._calculate_overall_effectiveness(component_evaluations)
        
        # 模拟用户满意度
        user_satisfaction = await self._simulate_user_satisfaction(component_evaluations)
        
        # 识别瓶颈
        bottlenecks = await self._identify_bottlenecks(component_evaluations)
        
        # 识别改进机会
        improvement_opportunities = await self._identify_improvement_opportunities(component_evaluations)
        
        return WorkflowEvaluation(
            workflow_id=str(uuid.uuid4()),
            workflow_name=scenario.get('name', 'Workflow'),
            start_time=workflow_start,
            end_time=workflow_end,
            total_duration=total_duration,
            component_evaluations=component_evaluations,
            integration_score=integration_score,
            overall_effectiveness=overall_effectiveness,
            user_satisfaction=user_satisfaction,
            business_outcome=scenario.get('expected_outcome', {}),
            bottlenecks=bottlenecks,
            improvement_opportunities=improvement_opportunities
        )
    
    async def _evaluate_market_intelligence_component(self, scenario: Dict[str, Any]) -> ComponentEvaluation:
        """评估市场情报组件"""
        start_time = datetime.now()
        
        try:
            # 执行市场情报采集
            market_context = scenario.get('market_context', {})
            target_segment = scenario.get('target_segment')
            
            competitors, success_cases = await self.market_intelligence.select_competitors_and_success_cases(
                market_context, target_segment
            )
            
            supplementary_data = await self.market_intelligence.gather_supplementary_market_data(
                competitors, success_cases
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 评估指标
            metrics = {
                "competitor_quality": await self._assess_competitor_quality(competitors),
                "data_completeness": await self._assess_data_completeness(supplementary_data),
                "analysis_depth": await self._assess_analysis_depth(supplementary_data),
                "accuracy": await self._assess_market_intelligence_accuracy(competitors, success_cases)
            }
            
            performance_score = sum(metrics.values()) / len(metrics)
            
            return ComponentEvaluation(
                component_name="MarketIntelligenceAgent",
                version="1.0.0",
                evaluation_date=datetime.now(),
                metrics=metrics,
                performance_score=performance_score,
                quality_indicators={
                    "competitors_found": len(competitors),
                    "success_cases_found": len(success_cases),
                    "data_sources": len(supplementary_data.get('competitor_deep_analysis', {}))
                },
                execution_time=execution_time,
                resource_usage={"memory": 0.1, "cpu": 0.2},
                error_count=0,
                success_rate=1.0,
                recommendations=["优化竞品筛选算法", "增加数据源多样性"]
            )
            
        except Exception as e:
            self.logger.error(f"市场情报组件评估失败: {e}")
            return self._create_error_evaluation("MarketIntelligenceAgent", str(e))
    
    async def _evaluate_user_insight_component(self, scenario: Dict[str, Any], 
                                             market_eval: ComponentEvaluation) -> ComponentEvaluation:
        """评估用户洞察组件"""
        start_time = datetime.now()
        
        try:
            # 执行用户洞察生成
            market_context = scenario.get('market_context', {})
            user_data = scenario.get('user_data', {})
            
            insights = await self.user_insight_generator.generate_comprehensive_insights(
                market_context, user_data
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 评估指标
            metrics = {
                "insight_quality": await self._assess_insight_quality(insights),
                "innovation_level": await self._assess_innovation_level(insights),
                "practicality": await self._assess_practicality(insights),
                "coherence": await self._assess_coherence(insights)
            }
            
            performance_score = sum(metrics.values()) / len(metrics)
            
            return ComponentEvaluation(
                component_name="UserInsightGenerator",
                version="1.0.0",
                evaluation_date=datetime.now(),
                metrics=metrics,
                performance_score=performance_score,
                quality_indicators={
                    "pain_points_identified": len(insights.get('pain_points_analysis', [])),
                    "concepts_generated": len(insights.get('divergent_concepts', [])),
                    "insights_generated": len(insights.get('deep_insights', []))
                },
                execution_time=execution_time,
                resource_usage={"memory": 0.15, "cpu": 0.25},
                error_count=0,
                success_rate=1.0,
                recommendations=["增强创新算法", "改进用户画像精度"]
            )
            
        except Exception as e:
            self.logger.error(f"用户洞察组件评估失败: {e}")
            return self._create_error_evaluation("UserInsightGenerator", str(e))
    
    async def _evaluate_product_definition_component(self, scenario: Dict[str, Any],
                                                   insight_eval: ComponentEvaluation) -> ComponentEvaluation:
        """评估产品定义组件"""
        start_time = datetime.now()
        
        try:
            # 执行产品定义
            market_analysis = scenario.get('market_context', {})
            product_concept = scenario.get('product_concept', {})
            strategy_context = scenario.get('strategy_context', {})
            
            product_definition = await self.product_definition.create_product_definition(
                market_analysis, product_concept, strategy_context
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 评估指标
            metrics = {
                "completeness": await self._assess_definition_completeness(product_definition),
                "technical_feasibility": await self._assess_technical_feasibility(product_definition),
                "business_alignment": await self._assess_business_alignment(product_definition),
                "specification_quality": await self._assess_specification_quality(product_definition)
            }
            
            performance_score = sum(metrics.values()) / len(metrics)
            
            return ComponentEvaluation(
                component_name="ProductDefinitionAgent",
                version="1.0.0",
                evaluation_date=datetime.now(),
                metrics=metrics,
                performance_score=performance_score,
                quality_indicators={
                    "core_features": len(product_definition.core_features),
                    "user_personas": len(product_definition.user_personas),
                    "technical_requirements": len(product_definition.technical_requirements),
                    "validation_status": product_definition.status.value
                },
                execution_time=execution_time,
                resource_usage={"memory": 0.2, "cpu": 0.3},
                error_count=0,
                success_rate=1.0,
                recommendations=["优化技术需求规格", "增强用户体验设计"]
            )
            
        except Exception as e:
            self.logger.error(f"产品定义组件评估失败: {e}")
            return self._create_error_evaluation("ProductDefinitionAgent", str(e))
    
    async def _evaluate_strategy_advisor_component(self, scenario: Dict[str, Any],
                                                 product_eval: ComponentEvaluation) -> ComponentEvaluation:
        """评估策略顾问组件"""
        start_time = datetime.now()
        
        try:
            # 执行策略评估
            product_definition = scenario.get('product_definition', {})
            market_context = scenario.get('market_context', {})
            user_research_data = scenario.get('user_research_data', {})
            
            strategy_evaluation = await self.strategy_advisor.evaluate_product_strategy(
                product_definition, market_context, user_research_data
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 评估指标
            metrics = {
                "strategic_insight": await self._assess_strategic_insight(strategy_evaluation),
                "recommendation_quality": await self._assess_recommendation_quality(strategy_evaluation),
                "risk_assessment": await self._assess_risk_assessment_quality(strategy_evaluation),
                "business_value": await self._assess_strategy_business_value(strategy_evaluation)
            }
            
            performance_score = sum(metrics.values()) / len(metrics)
            
            return ComponentEvaluation(
                component_name="ProductStrategyAdvisor",
                version="1.0.0",
                evaluation_date=datetime.now(),
                metrics=metrics,
                performance_score=performance_score,
                quality_indicators={
                    "overall_score": strategy_evaluation.overall_score,
                    "recommendations_count": len(strategy_evaluation.strategic_recommendations),
                    "validation_results": len(strategy_evaluation.validation_results),
                    "confidence_score": strategy_evaluation.confidence_score
                },
                execution_time=execution_time,
                resource_usage={"memory": 0.18, "cpu": 0.28},
                error_count=0,
                success_rate=1.0,
                recommendations=["增强风险量化分析", "优化建议优先级排序"]
            )
            
        except Exception as e:
            self.logger.error(f"策略顾问组件评估失败: {e}")
            return self._create_error_evaluation("ProductStrategyAdvisor", str(e))
    
    async def _evaluate_documentation_component(self, scenario: Dict[str, Any],
                                              strategy_eval: ComponentEvaluation) -> ComponentEvaluation:
        """评估文档生成组件"""
        start_time = datetime.now()
        
        try:
            # 执行文档生成
            market_analysis = scenario.get('market_context', {})
            product_definition = scenario.get('product_definition', {})
            strategy_evaluation = scenario.get('strategy_evaluation', {})
            
            documents = await self.documentation_agent.generate_comprehensive_documentation(
                market_analysis, product_definition, strategy_evaluation
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 评估指标
            metrics = {
                "documentation_quality": await self._assess_documentation_quality(documents),
                "completeness": await self._assess_documentation_completeness(documents),
                "coherence": await self._assess_documentation_coherence(documents),
                "professional_standard": await self._assess_professional_standard(documents)
            }
            
            performance_score = sum(metrics.values()) / len(metrics)
            
            return ComponentEvaluation(
                component_name="ProductDocumentationAgent",
                version="1.0.0",
                evaluation_date=datetime.now(),
                metrics=metrics,
                performance_score=performance_score,
                quality_indicators={
                    "documents_generated": len(documents),
                    "total_pages": sum([doc.metadata.page_count for doc in documents.values()]),
                    "average_quality": sum([doc.quality_metrics.overall_quality_score for doc in documents.values()]) / len(documents),
                    "document_types": len(set([doc.metadata.document_type.value for doc in documents.values()]))
                },
                execution_time=execution_time,
                resource_usage={"memory": 0.25, "cpu": 0.35},
                error_count=0,
                success_rate=1.0,
                recommendations=["优化文档模板", "增强交叉引用能力"]
            )
            
        except Exception as e:
            self.logger.error(f"文档生成组件评估失败: {e}")
            return self._create_error_evaluation("ProductDocumentationAgent", str(e))
    
    def _create_error_evaluation(self, component_name: str, error_message: str) -> ComponentEvaluation:
        """创建错误评估结果"""
        return ComponentEvaluation(
            component_name=component_name,
            version="1.0.0",
            evaluation_date=datetime.now(),
            metrics={"error_score": 0.0},
            performance_score=0.0,
            quality_indicators={"error": error_message},
            execution_time=0.0,
            resource_usage={"memory": 0.0, "cpu": 0.0},
            error_count=1,
            success_rate=0.0,
            recommendations=[f"修复错误: {error_message}"]
        )
    
    # 评估指标计算方法
    async def _assess_competitor_quality(self, competitors) -> float:
        """评估竞品质量"""
        if not competitors:
            return 0.0
        
        quality_score = 0.0
        for competitor in competitors:
            if hasattr(competitor, 'overall_score'):
                quality_score += competitor.overall_score
        
        return quality_score / len(competitors)
    
    async def _assess_data_completeness(self, data) -> float:
        """评估数据完整性"""
        expected_fields = ['competitor_deep_analysis', 'market_trends', 'user_sentiment']
        present_fields = [field for field in expected_fields if field in data]
        return len(present_fields) / len(expected_fields)
    
    async def _assess_analysis_depth(self, data) -> float:
        """评估分析深度"""
        depth_indicators = 0
        if 'competitor_deep_analysis' in data:
            depth_indicators += len(data['competitor_deep_analysis'])
        if 'financial_insights' in data:
            depth_indicators += 1
        if 'technology_analysis' in data:
            depth_indicators += 1
        
        return min(1.0, depth_indicators / 5)
    
    async def _assess_market_intelligence_accuracy(self, competitors, success_cases) -> float:
        """评估市场情报准确性"""
        # 模拟准确性评估
        base_accuracy = 0.8
        if len(competitors) >= 3:
            base_accuracy += 0.1
        if len(success_cases) >= 2:
            base_accuracy += 0.1
        return min(1.0, base_accuracy)
    
    async def _assess_insight_quality(self, insights) -> float:
        """评估洞察质量"""
        quality_factors = 0
        if 'deep_insights' in insights:
            quality_factors += len(insights['deep_insights']) * 0.2
        if 'innovation_assessment' in insights:
            quality_factors += 0.3
        if 'recommended_concept' in insights:
            quality_factors += 0.3
        
        return min(1.0, quality_factors)
    
    async def _assess_innovation_level(self, insights) -> float:
        """评估创新水平"""
        if 'recommended_concept' in insights:
            concept = insights['recommended_concept']
            return concept.get('innovation_level', 0.7)
        return 0.6
    
    async def _assess_practicality(self, insights) -> float:
        """评估实用性"""
        if 'recommended_concept' in insights:
            concept = insights['recommended_concept']
            return concept.get('commercial_feasibility_score', 0.7)
        return 0.6
    
    async def _assess_coherence(self, insights) -> float:
        """评估连贯性"""
        # 检查洞察间的一致性
        coherence_score = 0.75
        if 'deep_insights' in insights and len(insights['deep_insights']) > 1:
            coherence_score += 0.1
        return min(1.0, coherence_score)
    
    async def _assess_definition_completeness(self, product_definition) -> float:
        """评估定义完整性"""
        required_elements = [
            'core_features', 'user_personas', 'technical_requirements',
            'value_propositions', 'architecture_overview'
        ]
        present_elements = 0
        for element in required_elements:
            if hasattr(product_definition, element) and getattr(product_definition, element):
                present_elements += 1
        
        return present_elements / len(required_elements)
    
    async def _assess_technical_feasibility(self, product_definition) -> float:
        """评估技术可行性"""
        if hasattr(product_definition, 'technical_requirements'):
            high_risk_count = sum(1 for req in product_definition.technical_requirements 
                                if req.risk_level == "高风险")
            total_count = len(product_definition.technical_requirements)
            if total_count > 0:
                return 1.0 - (high_risk_count / total_count) * 0.5
        return 0.7
    
    async def _assess_business_alignment(self, product_definition) -> float:
        """评估商业对齐度"""
        alignment_score = 0.7
        if hasattr(product_definition, 'value_propositions') and product_definition.value_propositions:
            alignment_score += 0.2
        if hasattr(product_definition, 'success_metrics') and product_definition.success_metrics:
            alignment_score += 0.1
        return min(1.0, alignment_score)
    
    async def _assess_specification_quality(self, product_definition) -> float:
        """评估规格质量"""
        quality_score = 0.6
        if hasattr(product_definition, 'core_features'):
            avg_criteria_count = sum(len(feature.acceptance_criteria) for feature in product_definition.core_features) / len(product_definition.core_features)
            if avg_criteria_count >= 3:
                quality_score += 0.2
        return min(1.0, quality_score)
    
    async def _assess_strategic_insight(self, strategy_evaluation) -> float:
        """评估战略洞察"""
        return strategy_evaluation.overall_score
    
    async def _assess_recommendation_quality(self, strategy_evaluation) -> float:
        """评估建议质量"""
        if strategy_evaluation.strategic_recommendations:
            critical_count = len([r for r in strategy_evaluation.strategic_recommendations 
                                if r.priority.value == "critical"])
            return min(1.0, 0.6 + critical_count * 0.1)
        return 0.5
    
    async def _assess_risk_assessment_quality(self, strategy_evaluation) -> float:
        """评估风险评估质量"""
        if 'risk_assessment' in strategy_evaluation.risk_assessment:
            return 0.8
        return 0.6
    
    async def _assess_strategy_business_value(self, strategy_evaluation) -> float:
        """评估策略商业价值"""
        return strategy_evaluation.overall_score * 0.9
    
    async def _assess_documentation_quality(self, documents) -> float:
        """评估文档质量"""
        if not documents:
            return 0.0
        
        total_quality = sum(doc.quality_metrics.overall_quality_score for doc in documents.values())
        return total_quality / len(documents)
    
    async def _assess_documentation_completeness(self, documents) -> float:
        """评估文档完整性"""
        expected_types = [DocumentType.EXECUTIVE_SUMMARY, DocumentType.MARKET_ANALYSIS, 
                         DocumentType.PRODUCT_SPECIFICATION, DocumentType.BUSINESS_PLAN]
        present_types = set(doc.metadata.document_type for doc in documents.values())
        return len(present_types.intersection(expected_types)) / len(expected_types)
    
    async def _assess_documentation_coherence(self, documents) -> float:
        """评估文档连贯性"""
        if len(documents) < 2:
            return 0.8
        
        coherence_scores = [doc.quality_metrics.coherence_score for doc in documents.values()]
        return statistics.mean(coherence_scores)
    
    async def _assess_professional_standard(self, documents) -> float:
        """评估专业标准"""
        professional_score = 0.7
        for doc in documents.values():
            if doc.metadata.word_count > 1000:
                professional_score += 0.05
            if doc.quality_metrics.technical_accuracy_score > 0.8:
                professional_score += 0.05
        
        return min(1.0, professional_score)
    
    # 聚合分析方法
    async def _calculate_integration_score(self, component_evals: List[ComponentEvaluation]) -> float:
        """计算集成分数"""
        if not component_evals:
            return 0.0
        
        # 基于组件间的成功率和错误传播
        success_rates = [eval.success_rate for eval in component_evals]
        avg_success_rate = statistics.mean(success_rates)
        
        # 考虑执行时间的平衡性
        exec_times = [eval.execution_time for eval in component_evals]
        time_variance = statistics.stdev(exec_times) if len(exec_times) > 1 else 0
        time_balance_score = max(0.5, 1.0 - time_variance / 10)
        
        return (avg_success_rate * 0.7 + time_balance_score * 0.3)
    
    async def _calculate_overall_effectiveness(self, component_evals: List[ComponentEvaluation]) -> float:
        """计算整体有效性"""
        if not component_evals:
            return 0.0
        
        performance_scores = [eval.performance_score for eval in component_evals]
        return statistics.mean(performance_scores)
    
    async def _simulate_user_satisfaction(self, component_evals: List[ComponentEvaluation]) -> float:
        """模拟用户满意度"""
        # 基于性能分数和执行时间
        avg_performance = statistics.mean([eval.performance_score for eval in component_evals])
        avg_time = statistics.mean([eval.execution_time for eval in component_evals])
        
        # 时间惩罚（超过60秒开始惩罚）
        time_penalty = max(0, (avg_time - 60) / 120)
        satisfaction = avg_performance - time_penalty
        
        return max(0.0, min(1.0, satisfaction))
    
    async def _identify_bottlenecks(self, component_evals: List[ComponentEvaluation]) -> List[str]:
        """识别瓶颈"""
        bottlenecks = []
        
        # 执行时间瓶颈
        exec_times = {eval.component_name: eval.execution_time for eval in component_evals}
        max_time = max(exec_times.values())
        for name, time in exec_times.items():
            if time > max_time * 0.8:
                bottlenecks.append(f"执行时间瓶颈: {name} ({time:.2f}s)")
        
        # 性能分数瓶颈
        for eval in component_evals:
            if eval.performance_score < 0.6:
                bottlenecks.append(f"性能瓶颈: {eval.component_name} (score: {eval.performance_score:.2f})")
        
        return bottlenecks
    
    async def _identify_improvement_opportunities(self, component_evals: List[ComponentEvaluation]) -> List[str]:
        """识别改进机会"""
        opportunities = []
        
        for eval in component_evals:
            if eval.performance_score < 0.8:
                opportunities.append(f"提升 {eval.component_name} 性能至80%以上")
            
            if eval.execution_time > 30:
                opportunities.append(f"优化 {eval.component_name} 执行效率")
            
            if eval.error_count > 0:
                opportunities.append(f"降低 {eval.component_name} 错误率")
        
        return opportunities
    
    async def _calculate_aggregate_metrics(self, workflow_evals: List[WorkflowEvaluation],
                                         config: EvaluationConfiguration) -> Dict[str, float]:
        """计算聚合指标"""
        if not workflow_evals:
            return {}
        
        # 基础聚合指标
        overall_scores = []
        integration_scores = []
        effectiveness_scores = []
        satisfaction_scores = []
        
        for workflow in workflow_evals:
            component_scores = [comp.performance_score for comp in workflow.component_evaluations]
            overall_scores.append(statistics.mean(component_scores))
            integration_scores.append(workflow.integration_score)
            effectiveness_scores.append(workflow.overall_effectiveness)
            satisfaction_scores.append(workflow.user_satisfaction)
        
        aggregate_metrics = {
            "overall_score": statistics.mean(overall_scores),
            "integration_score": statistics.mean(integration_scores),
            "effectiveness_score": statistics.mean(effectiveness_scores),
            "satisfaction_score": statistics.mean(satisfaction_scores),
            "consistency_score": 1.0 - statistics.stdev(overall_scores) if len(overall_scores) > 1 else 1.0,
            "total_workflows": len(workflow_evals),
            "average_duration": statistics.mean([w.total_duration for w in workflow_evals])
        }
        
        # 按组件聚合
        component_metrics = {}
        for workflow in workflow_evals:
            for comp_eval in workflow.component_evaluations:
                comp_name = comp_eval.component_name
                if comp_name not in component_metrics:
                    component_metrics[comp_name] = []
                component_metrics[comp_name].append(comp_eval.performance_score)
        
        for comp_name, scores in component_metrics.items():
            aggregate_metrics[f"{comp_name}_avg_score"] = statistics.mean(scores)
            aggregate_metrics[f"{comp_name}_consistency"] = 1.0 - statistics.stdev(scores) if len(scores) > 1 else 1.0
        
        return aggregate_metrics
    
    async def _perform_benchmark_comparison(self, workflow_evals: List[WorkflowEvaluation]) -> Dict[str, Any]:
        """执行基准对比"""
        # 模拟与Value-Bench的对比
        avg_ai_score = statistics.mean([
            statistics.mean([comp.performance_score for comp in workflow.component_evaluations])
            for workflow in workflow_evals
        ])
        
        # 模拟人类专家基准分数
        human_baseline = 0.75
        
        return {
            "ai_system_score": avg_ai_score,
            "human_baseline": human_baseline,
            "performance_ratio": avg_ai_score / human_baseline,
            "benchmark_result": "优于人类基准" if avg_ai_score > human_baseline else "低于人类基准",
            "improvement_needed": max(0, human_baseline - avg_ai_score),
            "benchmark_date": datetime.now().isoformat()
        }
    
    async def _analyze_performance_trends(self, workflow_evals: List[WorkflowEvaluation]) -> Dict[str, Any]:
        """分析性能趋势"""
        if len(workflow_evals) < 2:
            return {"trend": "insufficient_data"}
        
        # 时间序列分析
        time_series = [(w.start_time, 
                       statistics.mean([comp.performance_score for comp in w.component_evaluations]))
                      for w in workflow_evals]
        time_series.sort(key=lambda x: x[0])
        
        scores = [score for _, score in time_series]
        
        # 简单趋势计算
        if len(scores) >= 2:
            trend_slope = (scores[-1] - scores[0]) / len(scores)
            if trend_slope > 0.01:
                trend = "improving"
            elif trend_slope < -0.01:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "trend_slope": trend_slope if len(scores) >= 2 else 0,
            "score_variance": statistics.stdev(scores) if len(scores) > 1 else 0,
            "latest_score": scores[-1],
            "best_score": max(scores),
            "worst_score": min(scores)
        }
    
    async def _assess_system_health(self, workflow_evals: List[WorkflowEvaluation]) -> Dict[str, Any]:
        """评估系统健康状况"""
        total_errors = sum(sum(comp.error_count for comp in workflow.component_evaluations) 
                          for workflow in workflow_evals)
        total_components = sum(len(workflow.component_evaluations) for workflow in workflow_evals)
        
        error_rate = total_errors / total_components if total_components > 0 else 0
        
        avg_success_rate = statistics.mean([
            statistics.mean([comp.success_rate for comp in workflow.component_evaluations])
            for workflow in workflow_evals
        ]) if workflow_evals else 0
        
        avg_response_time = statistics.mean([
            statistics.mean([comp.execution_time for comp in workflow.component_evaluations])
            for workflow in workflow_evals
        ]) if workflow_evals else 0
        
        # 健康状况评级
        if error_rate < 0.01 and avg_success_rate > 0.95 and avg_response_time < 30:
            health_status = "excellent"
        elif error_rate < 0.05 and avg_success_rate > 0.85 and avg_response_time < 60:
            health_status = "good"
        elif error_rate < 0.1 and avg_success_rate > 0.7 and avg_response_time < 120:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "health_status": health_status,
            "error_rate": error_rate,
            "success_rate": avg_success_rate,
            "average_response_time": avg_response_time,
            "total_evaluations": len(workflow_evals),
            "uptime": 0.99,  # 模拟
            "resource_utilization": {
                "cpu": 0.45,
                "memory": 0.60,
                "storage": 0.30
            }
        }
    
    async def _generate_strategic_insights(self, workflow_evals: List[WorkflowEvaluation],
                                         aggregate_metrics: Dict[str, float]) -> List[str]:
        """生成战略洞察"""
        insights = []
        
        overall_score = aggregate_metrics.get("overall_score", 0)
        
        # 总体性能洞察
        if overall_score > 0.85:
            insights.append("AI产品管理系统表现优异，已达到专家级水平")
        elif overall_score > 0.75:
            insights.append("AI产品管理系统表现良好，接近专业水准")
        elif overall_score > 0.65:
            insights.append("AI产品管理系统表现中等，有明显改进空间")
        else:
            insights.append("AI产品管理系统表现需要大幅提升")
        
        # 组件特定洞察
        component_scores = {k: v for k, v in aggregate_metrics.items() if k.endswith("_avg_score")}
        best_component = max(component_scores.items(), key=lambda x: x[1]) if component_scores else None
        worst_component = min(component_scores.items(), key=lambda x: x[1]) if component_scores else None
        
        if best_component:
            component_name = best_component[0].replace("_avg_score", "")
            insights.append(f"{component_name} 组件表现最佳，可作为其他组件的改进标杆")
        
        if worst_component:
            component_name = worst_component[0].replace("_avg_score", "")
            insights.append(f"{component_name} 组件需要重点优化，是当前主要瓶颈")
        
        # 一致性洞察
        consistency_score = aggregate_metrics.get("consistency_score", 0)
        if consistency_score > 0.9:
            insights.append("系统表现非常稳定，结果一致性优秀")
        elif consistency_score < 0.7:
            insights.append("系统表现存在较大变异性，需要提升一致性")
        
        # 效率洞察
        avg_duration = aggregate_metrics.get("average_duration", 0)
        if avg_duration < 60:
            insights.append("系统响应速度快，效率优势明显")
        elif avg_duration > 180:
            insights.append("系统响应时间较长，需要优化处理效率")
        
        return insights
    
    async def _generate_actionable_recommendations(self, workflow_evals: List[WorkflowEvaluation],
                                                 system_health: Dict[str, Any]) -> List[str]:
        """生成可操作建议"""
        recommendations = []
        
        # 基于系统健康状况的建议
        health_status = system_health.get("health_status", "unknown")
        if health_status == "poor":
            recommendations.append("立即进行系统诊断，识别和修复关键问题")
            recommendations.append("暂停非关键功能，专注于核心组件优化")
        elif health_status == "fair":
            recommendations.append("制定系统改进计划，逐步提升各组件性能")
            recommendations.append("增强错误处理和恢复机制")
        
        # 基于错误率的建议
        error_rate = system_health.get("error_rate", 0)
        if error_rate > 0.05:
            recommendations.append("实施全面的错误分析，建立预防机制")
            recommendations.append("增加单元测试和集成测试覆盖率")
        
        # 基于响应时间的建议
        avg_response_time = system_health.get("average_response_time", 0)
        if avg_response_time > 60:
            recommendations.append("优化算法效率，减少不必要的计算开销")
            recommendations.append("考虑并行处理和缓存机制")
        
        # 基于组件表现的建议
        for workflow in workflow_evals:
            for comp_eval in workflow.component_evaluations:
                if comp_eval.performance_score < 0.7:
                    recommendations.append(f"重点优化 {comp_eval.component_name} 组件性能")
                
                if comp_eval.execution_time > 60:
                    recommendations.append(f"优化 {comp_eval.component_name} 执行效率")
        
        # 通用改进建议
        recommendations.extend([
            "建立持续集成和持续部署流程",
            "实施性能监控和告警机制",
            "定期进行Value-Bench基准测试",
            "收集用户反馈，持续优化用户体验",
            "投资团队培训，提升AI系统运维能力"
        ])
        
        # 去重并限制数量
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:10]
    
    async def run_continuous_monitoring(self, config_name: str = "monitoring") -> None:
        """运行持续监控"""
        if config_name not in self.evaluation_configs:
            raise ValueError(f"监控配置不存在: {config_name}")
        
        config = self.evaluation_configs[config_name]
        self.active_monitoring[config_name] = True
        
        self.logger.info(f"启动持续监控: {config.name}")
        
        while self.active_monitoring.get(config_name, False):
            try:
                # 生成模拟测试场景
                test_scenario = self._generate_monitoring_scenario()
                
                # 执行快速评估
                workflow_eval = await self._evaluate_workflow(test_scenario, config)
                
                # 检查告警条件
                await self._check_alert_conditions(workflow_eval, config)
                
                # 等待下一次监控
                await asyncio.sleep(300)  # 5分钟间隔
                
            except Exception as e:
                self.logger.error(f"持续监控出错: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟再试
    
    def _generate_monitoring_scenario(self) -> Dict[str, Any]:
        """生成监控场景"""
        return {
            "name": "monitoring_scenario",
            "market_context": {
                "market_size_data": {"total_addressable_market": 1000000000},
                "competitive_intensity": 0.5
            },
            "user_data": {
                "satisfaction_scores": {"overall": 3.5},
                "usage_patterns": {"daily_active": 1000}
            },
            "product_concept": {
                "name": "AI助手",
                "key_features": ["智能推荐", "自动化处理"]
            }
        }
    
    async def _check_alert_conditions(self, workflow_eval: WorkflowEvaluation,
                                    config: EvaluationConfiguration) -> None:
        """检查告警条件"""
        for condition in config.alert_conditions:
            metric_name = condition["metric"]
            threshold = condition["threshold"]
            action = condition["action"]
            
            # 简化的指标检查
            if metric_name == "error_rate":
                error_rate = sum(comp.error_count for comp in workflow_eval.component_evaluations) / len(workflow_eval.component_evaluations)
                if error_rate > threshold:
                    await self._trigger_alert(action, f"错误率超标: {error_rate:.3f} > {threshold}")
            
            elif metric_name == "response_time":
                avg_time = workflow_eval.total_duration
                if avg_time > threshold:
                    await self._trigger_alert(action, f"响应时间超标: {avg_time:.1f}s > {threshold}s")
    
    async def _trigger_alert(self, action: str, message: str) -> None:
        """触发告警"""
        self.logger.warning(f"告警触发 [{action}]: {message}")
        # 这里可以集成实际的告警系统
    
    def stop_continuous_monitoring(self, config_name: str = "monitoring") -> None:
        """停止持续监控"""
        self.active_monitoring[config_name] = False
        self.logger.info(f"停止持续监控: {config_name}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        return {
            "total_evaluations": len(self.evaluation_history),
            "evaluation_configs": list(self.evaluation_configs.keys()),
            "active_monitoring": list(k for k, v in self.active_monitoring.items() if v),
            "latest_evaluation": self.evaluation_history[-1].evaluation_id if self.evaluation_history else None,
            "average_overall_score": statistics.mean([e.aggregate_metrics.get("overall_score", 0) for e in self.evaluation_history]) if self.evaluation_history else 0,
            "system_health_trend": "improving" if len(self.evaluation_history) >= 2 and 
                                 self.evaluation_history[-1].aggregate_metrics.get("overall_score", 0) > 
                                 self.evaluation_history[-2].aggregate_metrics.get("overall_score", 0) else "stable",
            "last_updated": datetime.now().isoformat()
        }