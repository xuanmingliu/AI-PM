"""
Product Strategy Advisor Agent
Based on Section 3.2.3 of the AI-Product Manager paper
提供专家反馈，弥合理论产品概念与实际市场可行性之间的鸿沟
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid

class ValidationDimension(Enum):
    """验证维度"""
    MARKET_FEASIBILITY = "market_feasibility"
    USER_VALUE = "user_value"
    TECHNICAL_VIABILITY = "technical_viability"
    BUSINESS_MODEL = "business_model"
    COMPETITIVE_POSITIONING = "competitive_positioning"
    EXECUTION_FEASIBILITY = "execution_feasibility"

class RecommendationPriority(Enum):
    """建议优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ValidationStatus(Enum):
    """验证状态"""
    PASSED = "passed"
    PASSED_WITH_CONCERNS = "passed_with_concerns"
    NEEDS_IMPROVEMENT = "needs_improvement"
    FAILED = "failed"

@dataclass
class ValidationCriteria:
    """验证标准"""
    criteria_id: str
    dimension: ValidationDimension
    name: str
    description: str
    weight: float
    threshold: float
    measurement_method: str

@dataclass
class MarketValidationResult:
    """市场验证结果"""
    validation_id: str
    criteria_id: str
    score: float
    status: ValidationStatus
    evidence: List[str]
    concerns: List[str]
    recommendations: List[str]
    confidence_level: float

@dataclass
class StrategicRecommendation:
    """战略建议"""
    recommendation_id: str
    title: str
    description: str
    priority: RecommendationPriority
    category: str
    rationale: str
    expected_impact: str
    implementation_effort: str
    timeline: str
    dependencies: List[str]
    success_metrics: List[str]
    risks: List[str]

@dataclass
class UserPersonaFeedback:
    """用户画像反馈"""
    persona_id: str
    persona_name: str
    feedback_type: str  # positive, negative, neutral
    feedback_text: str
    concerns: List[str]
    suggestions: List[str]
    adoption_likelihood: float
    value_perception: float

@dataclass
class CompetitiveAnalysis:
    """竞争分析"""
    analysis_id: str
    competitive_landscape: Dict[str, Any]
    positioning_assessment: Dict[str, Any]
    differentiation_opportunities: List[str]
    competitive_threats: List[str]
    market_positioning_recommendation: str

@dataclass
class StrategyEvaluationReport:
    """策略评估报告"""
    report_id: str
    product_definition_id: str
    evaluation_date: datetime
    overall_assessment: str
    overall_score: float
    dimension_scores: Dict[ValidationDimension, float]
    validation_results: List[MarketValidationResult]
    persona_feedback: List[UserPersonaFeedback]
    competitive_analysis: CompetitiveAnalysis
    strategic_recommendations: List[StrategicRecommendation]
    implementation_roadmap: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    next_steps: List[str]
    confidence_score: float

class ProductStrategyAdvisor:
    """产品策略顾问代理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 验证框架配置
        self.validation_criteria = self._initialize_validation_criteria()
        self.market_context_weight = config.get('market_context_weight', 0.3)
        self.user_feedback_weight = config.get('user_feedback_weight', 0.4)
        self.competitive_weight = config.get('competitive_weight', 0.3)
        
        # 评估标准
        self.min_passing_score = config.get('min_passing_score', 0.7)
        self.excellence_threshold = config.get('excellence_threshold', 0.85)
        
        # 数据存储
        self.evaluation_reports: Dict[str, StrategyEvaluationReport] = {}
        self.historical_validations: List[MarketValidationResult] = []
        
    def _initialize_validation_criteria(self) -> List[ValidationCriteria]:
        """初始化验证标准"""
        criteria = [
            # 市场可行性
            ValidationCriteria(
                criteria_id="market_001",
                dimension=ValidationDimension.MARKET_FEASIBILITY,
                name="市场规模与增长潜力",
                description="评估目标市场规模和增长前景",
                weight=0.25,
                threshold=0.7,
                measurement_method="市场数据分析+趋势预测"
            ),
            ValidationCriteria(
                criteria_id="market_002",
                dimension=ValidationDimension.MARKET_FEASIBILITY,
                name="市场进入时机",
                description="评估产品进入市场的时机是否合适",
                weight=0.20,
                threshold=0.65,
                measurement_method="市场成熟度分析+竞争格局评估"
            ),
            
            # 用户价值
            ValidationCriteria(
                criteria_id="user_001",
                dimension=ValidationDimension.USER_VALUE,
                name="痛点解决程度",
                description="评估产品解决用户核心痛点的程度",
                weight=0.30,
                threshold=0.75,
                measurement_method="用户痛点匹配度分析"
            ),
            ValidationCriteria(
                criteria_id="user_002",
                dimension=ValidationDimension.USER_VALUE,
                name="用户采用意愿",
                description="评估目标用户的采用意愿和付费意愿",
                weight=0.25,
                threshold=0.7,
                measurement_method="用户画像反馈+采用模型分析"
            ),
            
            # 技术可行性
            ValidationCriteria(
                criteria_id="tech_001",
                dimension=ValidationDimension.TECHNICAL_VIABILITY,
                name="技术实现可行性",
                description="评估技术方案的实现可行性",
                weight=0.35,
                threshold=0.8,
                measurement_method="技术评估+原型验证"
            ),
            ValidationCriteria(
                criteria_id="tech_002",
                dimension=ValidationDimension.TECHNICAL_VIABILITY,
                name="可扩展性与维护性",
                description="评估技术架构的可扩展性和长期维护性",
                weight=0.25,
                threshold=0.75,
                measurement_method="架构评审+性能分析"
            ),
            
            # 商业模式
            ValidationCriteria(
                criteria_id="biz_001",
                dimension=ValidationDimension.BUSINESS_MODEL,
                name="收入模式可行性",
                description="评估商业模式和收入来源的可行性",
                weight=0.30,
                threshold=0.7,
                measurement_method="商业模式分析+财务建模"
            ),
            ValidationCriteria(
                criteria_id="biz_002",
                dimension=ValidationDimension.BUSINESS_MODEL,
                name="盈利能力与可持续性",
                description="评估长期盈利能力和商业可持续性",
                weight=0.25,
                threshold=0.65,
                measurement_method="财务预测+敏感性分析"
            ),
            
            # 竞争定位
            ValidationCriteria(
                criteria_id="comp_001",
                dimension=ValidationDimension.COMPETITIVE_POSITIONING,
                name="差异化优势",
                description="评估产品的差异化优势和竞争壁垒",
                weight=0.35,
                threshold=0.75,
                measurement_method="竞品对比+SWOT分析"
            ),
            
            # 执行可行性
            ValidationCriteria(
                criteria_id="exec_001",
                dimension=ValidationDimension.EXECUTION_FEASIBILITY,
                name="团队能力匹配度",
                description="评估团队执行能力与产品需求的匹配度",
                weight=0.25,
                threshold=0.7,
                measurement_method="能力评估+资源分析"
            ),
            ValidationCriteria(
                criteria_id="exec_002",
                dimension=ValidationDimension.EXECUTION_FEASIBILITY,
                name="资源需求合理性",
                description="评估所需资源的合理性和可获得性",
                weight=0.20,
                threshold=0.65,
                measurement_method="资源规划+预算分析"
            )
        ]
        
        return criteria
    
    async def evaluate_product_strategy(self,
                                      product_definition: Dict[str, Any],
                                      market_context: Dict[str, Any],
                                      user_research_data: Dict[str, Any] = None) -> StrategyEvaluationReport:
        """
        评估产品策略
        系统地验证产品定义与商业概念和用户痛点的匹配度
        """
        self.logger.info("开始产品策略评估")
        
        report_id = str(uuid.uuid4())
        
        # 1. 执行多维度验证
        validation_results = await self._execute_multidimensional_validation(
            product_definition, market_context, user_research_data
        )
        
        # 2. 生成用户画像反馈
        persona_feedback = await self._generate_persona_feedback(
            product_definition, user_research_data
        )
        
        # 3. 执行竞争分析
        competitive_analysis = await self._perform_competitive_analysis(
            product_definition, market_context
        )
        
        # 4. 生成战略建议
        strategic_recommendations = await self._generate_strategic_recommendations(
            validation_results, persona_feedback, competitive_analysis
        )
        
        # 5. 计算综合评分
        overall_score, dimension_scores = self._calculate_overall_assessment(validation_results)
        
        # 6. 生成实施路线图
        implementation_roadmap = self._generate_implementation_roadmap(
            strategic_recommendations, validation_results
        )
        
        # 7. 风险评估
        risk_assessment = self._assess_strategic_risks(
            validation_results, competitive_analysis, strategic_recommendations
        )
        
        # 8. 制定后续步骤
        next_steps = self._define_next_steps(validation_results, strategic_recommendations)
        
        # 构建评估报告
        evaluation_report = StrategyEvaluationReport(
            report_id=report_id,
            product_definition_id=product_definition.get("definition_id", "unknown"),
            evaluation_date=datetime.now(),
            overall_assessment=self._generate_overall_assessment_text(overall_score),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            validation_results=validation_results,
            persona_feedback=persona_feedback,
            competitive_analysis=competitive_analysis,
            strategic_recommendations=strategic_recommendations,
            implementation_roadmap=implementation_roadmap,
            risk_assessment=risk_assessment,
            next_steps=next_steps,
            confidence_score=self._calculate_confidence_score(validation_results)
        )
        
        # 存储报告
        self.evaluation_reports[report_id] = evaluation_report
        self.historical_validations.extend(validation_results)
        
        self.logger.info(f"产品策略评估完成，总分: {overall_score:.3f}")
        return evaluation_report
    
    async def _execute_multidimensional_validation(self,
                                                 product_definition: Dict[str, Any],
                                                 market_context: Dict[str, Any],
                                                 user_research_data: Dict[str, Any]) -> List[MarketValidationResult]:
        """执行多维度验证"""
        self.logger.info("执行多维度验证")
        
        validation_results = []
        
        for criteria in self.validation_criteria:
            result = await self._validate_single_criteria(
                criteria, product_definition, market_context, user_research_data
            )
            validation_results.append(result)
        
        return validation_results
    
    async def _validate_single_criteria(self,
                                      criteria: ValidationCriteria,
                                      product_definition: Dict[str, Any],
                                      market_context: Dict[str, Any],
                                      user_research_data: Dict[str, Any]) -> MarketValidationResult:
        """验证单个标准"""
        
        if criteria.dimension == ValidationDimension.MARKET_FEASIBILITY:
            score, evidence, concerns = await self._validate_market_feasibility(
                criteria, product_definition, market_context
            )
        elif criteria.dimension == ValidationDimension.USER_VALUE:
            score, evidence, concerns = await self._validate_user_value(
                criteria, product_definition, user_research_data
            )
        elif criteria.dimension == ValidationDimension.TECHNICAL_VIABILITY:
            score, evidence, concerns = await self._validate_technical_viability(
                criteria, product_definition
            )
        elif criteria.dimension == ValidationDimension.BUSINESS_MODEL:
            score, evidence, concerns = await self._validate_business_model(
                criteria, product_definition, market_context
            )
        elif criteria.dimension == ValidationDimension.COMPETITIVE_POSITIONING:
            score, evidence, concerns = await self._validate_competitive_positioning(
                criteria, product_definition, market_context
            )
        elif criteria.dimension == ValidationDimension.EXECUTION_FEASIBILITY:
            score, evidence, concerns = await self._validate_execution_feasibility(
                criteria, product_definition
            )
        else:
            score, evidence, concerns = 0.5, ["未实现的验证维度"], ["需要实现验证逻辑"]
        
        # 确定验证状态
        if score >= self.excellence_threshold:
            status = ValidationStatus.PASSED
        elif score >= criteria.threshold:
            status = ValidationStatus.PASSED_WITH_CONCERNS
        elif score >= self.min_passing_score:
            status = ValidationStatus.NEEDS_IMPROVEMENT
        else:
            status = ValidationStatus.FAILED
        
        # 生成建议
        recommendations = self._generate_criteria_recommendations(
            criteria, score, concerns, product_definition
        )
        
        return MarketValidationResult(
            validation_id=str(uuid.uuid4()),
            criteria_id=criteria.criteria_id,
            score=score,
            status=status,
            evidence=evidence,
            concerns=concerns,
            recommendations=recommendations,
            confidence_level=self._calculate_validation_confidence(score, evidence)
        )
    
    async def _validate_market_feasibility(self,
                                         criteria: ValidationCriteria,
                                         product_definition: Dict[str, Any],
                                         market_context: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """验证市场可行性"""
        evidence = []
        concerns = []
        
        if criteria.name == "市场规模与增长潜力":
            # 评估市场规模
            target_market = product_definition.get("target_market", "")
            market_data = market_context.get("market_size_data", {})
            
            tam = market_data.get("total_addressable_market", 0)
            growth_rate = market_context.get("growth_projections", {}).get("2024", 0)
            
            # 计算分数
            size_score = min(1.0, tam / 10000000000)  # $10B为满分
            growth_score = min(1.0, growth_rate * 2.5)  # 40%增长率为满分
            score = (size_score * 0.6 + growth_score * 0.4)
            
            evidence.append(f"目标市场规模: ${tam/1000000000:.1f}B")
            evidence.append(f"市场增长率: {growth_rate:.1%}")
            
            if tam < 1000000000:  # <$1B
                concerns.append("目标市场规模相对较小")
            if growth_rate < 0.1:  # <10%
                concerns.append("市场增长率偏低")
                
        elif criteria.name == "市场进入时机":
            # 评估市场时机
            market_maturity = market_context.get("market_maturity", "emerging")
            competitive_intensity = market_context.get("competitive_intensity", 0.5)
            
            # 新兴市场时机更好
            maturity_score = {"emerging": 0.9, "growth": 0.8, "mature": 0.5, "declining": 0.2}.get(market_maturity, 0.6)
            competition_score = 1.0 - competitive_intensity  # 竞争越激烈分数越低
            score = (maturity_score * 0.7 + competition_score * 0.3)
            
            evidence.append(f"市场成熟度: {market_maturity}")
            evidence.append(f"竞争强度: {competitive_intensity:.1%}")
            
            if competitive_intensity > 0.8:
                concerns.append("市场竞争过于激烈")
            if market_maturity == "declining":
                concerns.append("市场正在衰退")
        else:
            score = 0.7  # 默认分数
            evidence.append("基础市场可行性评估")
        
        return score, evidence, concerns
    
    async def _validate_user_value(self,
                                 criteria: ValidationCriteria,
                                 product_definition: Dict[str, Any],
                                 user_research_data: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """验证用户价值"""
        evidence = []
        concerns = []
        
        if criteria.name == "痛点解决程度":
            # 评估痛点解决程度
            value_propositions = product_definition.get("value_propositions", [])
            user_personas = product_definition.get("user_personas", [])
            
            if value_propositions and user_personas:
                # 模拟痛点覆盖度评估
                pain_point_coverage = 0.8  # 模拟80%覆盖率
                solution_effectiveness = 0.75  # 模拟75%有效性
                score = (pain_point_coverage * 0.6 + solution_effectiveness * 0.4)
                
                evidence.append(f"痛点覆盖率: {pain_point_coverage:.1%}")
                evidence.append(f"解决方案有效性: {solution_effectiveness:.1%}")
                evidence.append(f"价值主张数量: {len(value_propositions)}")
                
                if pain_point_coverage < 0.7:
                    concerns.append("痛点覆盖度不足")
                if solution_effectiveness < 0.7:
                    concerns.append("解决方案有效性待提升")
            else:
                score = 0.5
                concerns.append("缺少详细的价值主张或用户画像")
                
        elif criteria.name == "用户采用意愿":
            # 评估用户采用意愿
            if user_research_data:
                adoption_indicators = user_research_data.get("adoption_indicators", {})
                willingness_to_pay = adoption_indicators.get("willingness_to_pay", 0.6)
                ease_of_adoption = adoption_indicators.get("ease_of_adoption", 0.7)
                
                score = (willingness_to_pay * 0.6 + ease_of_adoption * 0.4)
                
                evidence.append(f"付费意愿: {willingness_to_pay:.1%}")
                evidence.append(f"采用易用性: {ease_of_adoption:.1%}")
                
                if willingness_to_pay < 0.6:
                    concerns.append("用户付费意愿偏低")
                if ease_of_adoption < 0.6:
                    concerns.append("产品采用门槛较高")
            else:
                score = 0.6
                evidence.append("基于产品设计的采用意愿评估")
        else:
            score = 0.7
            evidence.append("基础用户价值评估")
        
        return score, evidence, concerns
    
    async def _validate_technical_viability(self,
                                          criteria: ValidationCriteria,
                                          product_definition: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """验证技术可行性"""
        evidence = []
        concerns = []
        
        tech_requirements = product_definition.get("technical_requirements", [])
        architecture = product_definition.get("architecture_overview", {})
        
        if criteria.name == "技术实现可行性":
            # 评估技术实现难度
            high_risk_requirements = len([req for req in tech_requirements 
                                        if req.get("risk_level") == "高风险"])
            total_requirements = len(tech_requirements)
            
            if total_requirements > 0:
                risk_ratio = high_risk_requirements / total_requirements
                complexity_score = 1.0 - (risk_ratio * 0.5)  # 高风险需求影响分数
                
                # 架构复杂度
                if architecture:
                    components_count = len(architecture.get("components", {}))
                    architecture_score = min(1.0, (10 - components_count) / 10)  # 组件越多越复杂
                else:
                    architecture_score = 0.7
                
                score = (complexity_score * 0.6 + architecture_score * 0.4)
                
                evidence.append(f"技术需求总数: {total_requirements}")
                evidence.append(f"高风险需求: {high_risk_requirements}")
                evidence.append(f"系统组件数: {components_count if architecture else 0}")
                
                if risk_ratio > 0.3:
                    concerns.append("高风险技术需求过多")
                if components_count > 8:
                    concerns.append("系统架构过于复杂")
            else:
                score = 0.5
                concerns.append("缺少详细的技术需求")
                
        elif criteria.name == "可扩展性与维护性":
            # 评估可扩展性
            scalability_reqs = product_definition.get("scalability_requirements", {})
            performance_reqs = product_definition.get("performance_requirements", {})
            
            if scalability_reqs and performance_reqs:
                # 模拟可扩展性评估
                horizontal_scaling = "horizontal_scaling" in scalability_reqs
                load_balancing = "load_balancing" in str(scalability_reqs)
                
                scalability_score = 0.7
                if horizontal_scaling:
                    scalability_score += 0.1
                if load_balancing:
                    scalability_score += 0.1
                
                maintainability_score = 0.75  # 基于架构模式评估
                
                score = (scalability_score * 0.6 + maintainability_score * 0.4)
                
                evidence.append("支持水平扩展" if horizontal_scaling else "不支持水平扩展")
                evidence.append("支持负载均衡" if load_balancing else "不支持负载均衡")
                evidence.append(f"可维护性评分: {maintainability_score:.2f}")
                
                if not horizontal_scaling:
                    concerns.append("缺少水平扩展能力")
            else:
                score = 0.6
                concerns.append("缺少可扩展性和性能需求定义")
        else:
            score = 0.7
            evidence.append("基础技术可行性评估")
        
        return score, evidence, concerns
    
    async def _validate_business_model(self,
                                     criteria: ValidationCriteria,
                                     product_definition: Dict[str, Any],
                                     market_context: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """验证商业模式"""
        evidence = []
        concerns = []
        
        value_propositions = product_definition.get("value_propositions", [])
        
        if criteria.name == "收入模式可行性":
            # 评估收入模式
            revenue_streams = []
            for vp in value_propositions:
                if "core_value" in vp:
                    if "企业" in str(vp):
                        revenue_streams.append("企业订阅")
                    if "个人" in str(vp):
                        revenue_streams.append("个人订阅")
            
            # 基于市场数据评估收入潜力
            market_size = market_context.get("market_size_data", {}).get("serviceable_obtainable_market", 0)
            
            diversification_score = min(1.0, len(revenue_streams) / 3)  # 收入来源多样性
            market_potential_score = min(1.0, market_size / 1000000000)  # $1B为满分
            
            score = (diversification_score * 0.4 + market_potential_score * 0.6)
            
            evidence.append(f"收入来源: {', '.join(revenue_streams) if revenue_streams else '未明确'}")
            evidence.append(f"可获得市场: ${market_size/1000000:.0f}M")
            
            if len(revenue_streams) < 2:
                concerns.append("收入来源单一，风险较高")
            if market_size < 100000000:  # <$100M
                concerns.append("可获得市场规模较小")
                
        elif criteria.name == "盈利能力与可持续性":
            # 评估盈利能力
            # 模拟成本结构分析
            estimated_gross_margin = 0.7  # 70%毛利率
            customer_acquisition_cost = 500  # $500获客成本
            lifetime_value = 2000  # $2000客户生命周期价值
            
            margin_score = estimated_gross_margin
            ltv_cac_ratio = lifetime_value / customer_acquisition_cost
            efficiency_score = min(1.0, ltv_cac_ratio / 5)  # LTV/CAC=5为满分
            
            score = (margin_score * 0.5 + efficiency_score * 0.5)
            
            evidence.append(f"预估毛利率: {estimated_gross_margin:.1%}")
            evidence.append(f"LTV/CAC比率: {ltv_cac_ratio:.1f}")
            
            if estimated_gross_margin < 0.5:
                concerns.append("毛利率偏低")
            if ltv_cac_ratio < 3:
                concerns.append("客户获取效率需要提升")
        else:
            score = 0.7
            evidence.append("基础商业模式评估")
        
        return score, evidence, concerns
    
    async def _validate_competitive_positioning(self,
                                              criteria: ValidationCriteria,
                                              product_definition: Dict[str, Any],
                                              market_context: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """验证竞争定位"""
        evidence = []
        concerns = []
        
        competitors = market_context.get("competitor_analysis", [])
        value_propositions = product_definition.get("value_propositions", [])
        
        if criteria.name == "差异化优势":
            # 评估差异化程度
            unique_features = 0
            total_features = len(product_definition.get("core_features", []))
            
            # 模拟差异化分析
            if value_propositions:
                for vp in value_propositions:
                    if any(keyword in str(vp).lower() 
                          for keyword in ["智能", "自动", "个性化", "创新"]):
                        unique_features += 1
            
            differentiation_score = min(1.0, unique_features / max(1, total_features))
            
            # 竞争强度影响
            competitive_pressure = len(competitors) / 10  # 假设10个竞争对手为高竞争
            competition_impact = max(0.5, 1.0 - competitive_pressure)
            
            score = (differentiation_score * 0.7 + competition_impact * 0.3)
            
            evidence.append(f"差异化特性: {unique_features}/{total_features}")
            evidence.append(f"主要竞争对手: {len(competitors)}个")
            
            if differentiation_score < 0.5:
                concerns.append("产品差异化程度不足")
            if len(competitors) > 8:
                concerns.append("竞争环境过于激烈")
        else:
            score = 0.7
            evidence.append("基础竞争定位评估")
        
        return score, evidence, concerns
    
    async def _validate_execution_feasibility(self,
                                            criteria: ValidationCriteria,
                                            product_definition: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """验证执行可行性"""
        evidence = []
        concerns = []
        
        features = product_definition.get("core_features", []) + product_definition.get("supporting_features", [])
        tech_requirements = product_definition.get("technical_requirements", [])
        
        if criteria.name == "团队能力匹配度":
            # 评估团队能力需求
            ai_features = len([f for f in features if "智能" in str(f) or "AI" in str(f)])
            total_features = len(features)
            
            # 模拟团队能力评估
            ai_expertise_required = ai_features / max(1, total_features)
            team_capability_score = 0.75  # 假设团队有75%的所需能力
            
            capability_gap = abs(ai_expertise_required - team_capability_score)
            score = max(0.3, 1.0 - capability_gap)
            
            evidence.append(f"AI相关功能占比: {ai_expertise_required:.1%}")
            evidence.append(f"团队能力匹配度: {team_capability_score:.1%}")
            
            if capability_gap > 0.3:
                concerns.append("团队能力与产品需求存在较大差距")
                
        elif criteria.name == "资源需求合理性":
            # 评估资源需求
            high_effort_features = len([f for f in features 
                                      if "人天" in str(f) and ("30" in str(f) or "25" in str(f))])
            total_estimated_effort = len(features) * 20  # 假设平均20人天/功能
            
            # 资源合理性评估
            if total_estimated_effort <= 300:  # <=300人天
                resource_score = 0.9
            elif total_estimated_effort <= 600:  # <=600人天
                resource_score = 0.7
            else:  # >600人天
                resource_score = 0.5
            
            complexity_penalty = high_effort_features * 0.05  # 每个高复杂度功能扣5%
            score = max(0.3, resource_score - complexity_penalty)
            
            evidence.append(f"预估总工作量: {total_estimated_effort}人天")
            evidence.append(f"高复杂度功能: {high_effort_features}个")
            
            if total_estimated_effort > 500:
                concerns.append("项目工作量较大，需要充足的资源投入")
            if high_effort_features > 3:
                concerns.append("高复杂度功能过多，执行风险较高")
        else:
            score = 0.7
            evidence.append("基础执行可行性评估")
        
        return score, evidence, concerns
    
    def _generate_criteria_recommendations(self,
                                         criteria: ValidationCriteria,
                                         score: float,
                                         concerns: List[str],
                                         product_definition: Dict[str, Any]) -> List[str]:
        """生成标准相关建议"""
        recommendations = []
        
        if score < criteria.threshold:
            if criteria.dimension == ValidationDimension.MARKET_FEASIBILITY:
                recommendations.extend([
                    "深入研究目标市场，验证市场规模和增长潜力",
                    "分析竞争格局，寻找市场差异化机会",
                    "考虑调整市场进入策略或时机"
                ])
            elif criteria.dimension == ValidationDimension.USER_VALUE:
                recommendations.extend([
                    "进行更深入的用户研究，验证痛点假设",
                    "优化价值主张，增强用户价值感知",
                    "降低用户采用门槛，提升易用性"
                ])
            elif criteria.dimension == ValidationDimension.TECHNICAL_VIABILITY:
                recommendations.extend([
                    "简化技术架构，降低实现复杂度",
                    "制定详细的技术风险缓解方案",
                    "考虑分阶段实现，先实现核心功能"
                ])
            elif criteria.dimension == ValidationDimension.BUSINESS_MODEL:
                recommendations.extend([
                    "多样化收入来源，降低商业风险",
                    "优化成本结构，提升盈利能力",
                    "验证客户付费意愿和定价策略"
                ])
            elif criteria.dimension == ValidationDimension.COMPETITIVE_POSITIONING:
                recommendations.extend([
                    "强化产品差异化特性",
                    "建立可持续的竞争壁垒",
                    "重新评估竞争策略和市场定位"
                ])
            elif criteria.dimension == ValidationDimension.EXECUTION_FEASIBILITY:
                recommendations.extend([
                    "评估和补充团队能力短板",
                    "合理规划资源投入和项目时间",
                    "建立风险管控机制"
                ])
        
        # 基于具体关切点的建议
        for concern in concerns:
            if "市场规模" in concern:
                recommendations.append("考虑扩大目标市场范围或寻找利基市场")
            elif "竞争" in concern:
                recommendations.append("加强产品差异化或寻找蓝海市场")
            elif "复杂" in concern:
                recommendations.append("简化产品设计，聚焦核心功能")
            elif "成本" in concern:
                recommendations.append("优化成本结构，提升运营效率")
        
        return list(set(recommendations))  # 去重
    
    def _calculate_validation_confidence(self, score: float, evidence: List[str]) -> float:
        """计算验证置信度"""
        # 基于分数和证据质量计算置信度
        score_confidence = score
        evidence_confidence = min(1.0, len(evidence) / 3)  # 3个证据为满信心
        
        return (score_confidence * 0.7 + evidence_confidence * 0.3)
    
    async def _generate_persona_feedback(self,
                                       product_definition: Dict[str, Any],
                                       user_research_data: Dict[str, Any]) -> List[UserPersonaFeedback]:
        """生成用户画像反馈"""
        self.logger.info("生成用户画像反馈")
        
        personas = product_definition.get("user_personas", [])
        feedback_list = []
        
        for persona in personas:
            # 模拟用户画像反馈
            persona_name = persona.get("name", "未知用户")
            
            # 基于用户画像特征生成反馈
            if "企业" in persona_name:
                feedback = UserPersonaFeedback(
                    persona_id=persona.get("persona_id", "unknown"),
                    persona_name=persona_name,
                    feedback_type="positive",
                    feedback_text="产品功能符合企业级需求，能够解决实际业务问题",
                    concerns=["实施成本", "集成复杂度", "ROI实现时间"],
                    suggestions=["提供试用期", "简化集成流程", "增加ROI计算工具"],
                    adoption_likelihood=0.7,
                    value_perception=0.8
                )
            elif "技术" in persona_name:
                feedback = UserPersonaFeedback(
                    persona_id=persona.get("persona_id", "unknown"),
                    persona_name=persona_name,
                    feedback_type="neutral",
                    feedback_text="技术方案可行，但需要更多技术细节和文档",
                    concerns=["技术文档完整性", "API设计合理性", "性能指标"],
                    suggestions=["完善技术文档", "提供API示例", "发布性能基准"],
                    adoption_likelihood=0.75,
                    value_perception=0.7
                )
            else:  # 个人用户
                feedback = UserPersonaFeedback(
                    persona_id=persona.get("persona_id", "unknown"),
                    persona_name=persona_name,
                    feedback_type="positive",
                    feedback_text="产品易于使用，能够提升个人工作效率",
                    concerns=["学习成本", "价格接受度", "数据安全"],
                    suggestions=["提供详细教程", "灵活定价方案", "加强隐私保护"],
                    adoption_likelihood=0.8,
                    value_perception=0.75
                )
            
            feedback_list.append(feedback)
        
        return feedback_list
    
    async def _perform_competitive_analysis(self,
                                          product_definition: Dict[str, Any],
                                          market_context: Dict[str, Any]) -> CompetitiveAnalysis:
        """执行竞争分析"""
        self.logger.info("执行竞争分析")
        
        competitors = market_context.get("competitor_analysis", [])
        
        # 竞争格局分析
        competitive_landscape = {
            "total_competitors": len(competitors),
            "major_players": [comp.get("name", "Unknown") for comp in competitors[:3]],
            "market_concentration": "分散" if len(competitors) > 10 else "集中",
            "competition_intensity": min(1.0, len(competitors) / 10)
        }
        
        # 定位评估
        positioning_assessment = {
            "market_position": "挑战者",  # 基于产品特性评估
            "price_position": "中等价位",
            "feature_position": "差异化",
            "target_position": "专业用户"
        }
        
        # 差异化机会
        differentiation_opportunities = [
            "AI技术先进性",
            "用户体验优化",
            "集成便利性",
            "定制化能力",
            "成本效益优势"
        ]
        
        # 竞争威胁
        competitive_threats = [
            "大公司的资源优势",
            "成熟产品的用户粘性",
            "价格竞争压力",
            "技术同质化风险"
        ]
        
        # 市场定位建议
        market_positioning_recommendation = (
            "建议采用差异化竞争策略，专注于AI技术优势和用户体验创新，"
            "避免与大公司的直接价格竞争，寻找细分市场机会"
        )
        
        return CompetitiveAnalysis(
            analysis_id=str(uuid.uuid4()),
            competitive_landscape=competitive_landscape,
            positioning_assessment=positioning_assessment,
            differentiation_opportunities=differentiation_opportunities,
            competitive_threats=competitive_threats,
            market_positioning_recommendation=market_positioning_recommendation
        )
    
    async def _generate_strategic_recommendations(self,
                                                validation_results: List[MarketValidationResult],
                                                persona_feedback: List[UserPersonaFeedback],
                                                competitive_analysis: CompetitiveAnalysis) -> List[StrategicRecommendation]:
        """生成战略建议"""
        self.logger.info("生成战略建议")
        
        recommendations = []
        
        # 基于验证结果生成建议
        failed_validations = [vr for vr in validation_results if vr.status == ValidationStatus.FAILED]
        low_score_validations = [vr for vr in validation_results if vr.score < 0.7]
        
        # 高优先级建议 - 解决失败的验证
        for validation in failed_validations:
            criteria = next((c for c in self.validation_criteria if c.criteria_id == validation.criteria_id), None)
            if criteria:
                recommendations.append(StrategicRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title=f"紧急改进: {criteria.name}",
                    description=f"该验证维度未通过标准，需要立即改进",
                    priority=RecommendationPriority.CRITICAL,
                    category="风险缓解",
                    rationale=f"验证分数({validation.score:.2f})低于最低标准({criteria.threshold})",
                    expected_impact="避免项目失败风险，提升成功概率",
                    implementation_effort="高",
                    timeline="1-2周",
                    dependencies=[],
                    success_metrics=[f"{criteria.name}验证分数 > {criteria.threshold}"],
                    risks=["如不改进可能导致项目失败"]
                ))
        
        # 中优先级建议 - 优化低分验证
        for validation in low_score_validations:
            if validation not in failed_validations:  # 避免重复
                criteria = next((c for c in self.validation_criteria if c.criteria_id == validation.criteria_id), None)
                if criteria:
                    recommendations.append(StrategicRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        title=f"优化提升: {criteria.name}",
                        description=f"该维度有提升空间，建议优化",
                        priority=RecommendationPriority.HIGH,
                        category="性能提升",
                        rationale=f"验证分数({validation.score:.2f})可以进一步提升",
                        expected_impact="提升产品竞争力和成功概率",
                        implementation_effort="中等",
                        timeline="2-4周",
                        dependencies=[],
                        success_metrics=[f"{criteria.name}验证分数 > 0.8"],
                        risks=["机会成本"]
                    ))
        
        # 基于用户反馈的建议
        negative_feedback = [pf for pf in persona_feedback if pf.feedback_type == "negative"]
        if negative_feedback:
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="用户体验优化",
                description="解决用户反馈中的关键问题",
                priority=RecommendationPriority.HIGH,
                category="用户体验",
                rationale="存在负面用户反馈，需要及时解决",
                expected_impact="提升用户满意度和采用率",
                implementation_effort="中等",
                timeline="2-3周",
                dependencies=["用户研究"],
                success_metrics=["用户满意度 > 4.0", "负面反馈比例 < 10%"],
                risks=["用户流失风险"]
            ))
        
        # 基于竞争分析的建议
        if competitive_analysis.competitive_landscape["competition_intensity"] > 0.7:
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="差异化战略强化",
                description="在激烈竞争中建立独特优势",
                priority=RecommendationPriority.HIGH,
                category="竞争策略",
                rationale="市场竞争激烈，需要建立差异化优势",
                expected_impact="获得竞争优势，提升市场份额",
                implementation_effort="高",
                timeline="4-6周",
                dependencies=["技术研发", "市场调研"],
                success_metrics=["产品差异化程度 > 80%", "竞争优势指数 > 0.7"],
                risks=["开发成本增加"]
            ))
        
        # 通用改进建议
        avg_score = np.mean([vr.score for vr in validation_results])
        if avg_score < 0.8:
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="整体策略优化",
                description="全面提升产品策略各个维度",
                priority=RecommendationPriority.MEDIUM,
                category="整体优化",
                rationale=f"整体评分({avg_score:.2f})有提升空间",
                expected_impact="全面提升产品成功概率",
                implementation_effort="高",
                timeline="6-8周",
                dependencies=["团队协调", "资源投入"],
                success_metrics=["整体评分 > 0.8", "各维度均衡发展"],
                risks=["资源分散"]
            ))
        
        return recommendations
    
    def _calculate_overall_assessment(self, validation_results: List[MarketValidationResult]) -> Tuple[float, Dict[ValidationDimension, float]]:
        """计算总体评估"""
        # 按维度聚合分数
        dimension_scores = {}
        dimension_weights = {}
        
        for result in validation_results:
            criteria = next((c for c in self.validation_criteria if c.criteria_id == result.criteria_id), None)
            if criteria:
                if criteria.dimension not in dimension_scores:
                    dimension_scores[criteria.dimension] = []
                    dimension_weights[criteria.dimension] = []
                
                dimension_scores[criteria.dimension].append(result.score)
                dimension_weights[criteria.dimension].append(criteria.weight)
        
        # 计算加权平均分
        final_dimension_scores = {}
        for dimension, scores in dimension_scores.items():
            weights = dimension_weights[dimension]
            if sum(weights) > 0:
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            else:
                weighted_score = np.mean(scores)
            final_dimension_scores[dimension] = weighted_score
        
        # 计算总体分数
        overall_score = np.mean(list(final_dimension_scores.values()))
        
        return overall_score, final_dimension_scores
    
    def _generate_overall_assessment_text(self, overall_score: float) -> str:
        """生成总体评估文本"""
        if overall_score >= 0.9:
            return "优秀 - 产品策略非常成熟，成功概率很高"
        elif overall_score >= 0.8:
            return "良好 - 产品策略基本可行，有较高成功概率"
        elif overall_score >= 0.7:
            return "合格 - 产品策略可行，但需要改进优化"
        elif overall_score >= 0.6:
            return "需要改进 - 产品策略存在明显不足"
        else:
            return "不合格 - 产品策略存在重大问题，需要重新设计"
    
    def _generate_implementation_roadmap(self,
                                       recommendations: List[StrategicRecommendation],
                                       validation_results: List[MarketValidationResult]) -> Dict[str, Any]:
        """生成实施路线图"""
        # 按优先级分组建议
        critical_items = [r for r in recommendations if r.priority == RecommendationPriority.CRITICAL]
        high_items = [r for r in recommendations if r.priority == RecommendationPriority.HIGH]
        medium_items = [r for r in recommendations if r.priority == RecommendationPriority.MEDIUM]
        
        return {
            "phase_1_immediate": {
                "duration": "1-2周",
                "focus": "关键问题解决",
                "actions": [r.title for r in critical_items],
                "success_criteria": "所有关键验证通过"
            },
            "phase_2_optimization": {
                "duration": "2-4周", 
                "focus": "性能提升",
                "actions": [r.title for r in high_items],
                "success_criteria": "主要指标达到优秀水平"
            },
            "phase_3_enhancement": {
                "duration": "4-8周",
                "focus": "全面优化",
                "actions": [r.title for r in medium_items],
                "success_criteria": "产品策略全面成熟"
            },
            "milestones": [
                "第1周：完成关键问题识别和改进计划",
                "第2周：关键验证指标达标",
                "第4周：主要性能指标优化完成",
                "第8周：产品策略整体优化完成"
            ],
            "resource_requirements": {
                "team_size": "5-8人",
                "key_roles": ["产品经理", "技术架构师", "市场分析师", "用户研究员"],
                "estimated_cost": "中等投入"
            }
        }
    
    def _assess_strategic_risks(self,
                              validation_results: List[MarketValidationResult],
                              competitive_analysis: CompetitiveAnalysis,
                              recommendations: List[StrategicRecommendation]) -> Dict[str, Any]:
        """评估战略风险"""
        risks = {
            "high_risks": [],
            "medium_risks": [],
            "low_risks": [],
            "risk_mitigation": {},
            "overall_risk_level": "中等"
        }
        
        # 基于验证结果识别风险
        failed_validations = len([vr for vr in validation_results if vr.status == ValidationStatus.FAILED])
        if failed_validations > 0:
            risks["high_risks"].append("关键验证维度失败")
            risks["risk_mitigation"]["关键验证维度失败"] = "立即执行改进计划，定期监控进展"
        
        low_scores = len([vr for vr in validation_results if vr.score < 0.6])
        if low_scores > 2:
            risks["medium_risks"].append("多个维度表现不佳")
            risks["risk_mitigation"]["多个维度表现不佳"] = "制定综合优化策略，逐步改善"
        
        # 基于竞争分析识别风险
        if competitive_analysis.competitive_landscape["competition_intensity"] > 0.8:
            risks["high_risks"].append("市场竞争过于激烈")
            risks["risk_mitigation"]["市场竞争过于激烈"] = "强化差异化优势，寻找利基市场"
        
        # 基于建议识别风险
        critical_recommendations = len([r for r in recommendations if r.priority == RecommendationPriority.CRITICAL])
        if critical_recommendations > 3:
            risks["medium_risks"].append("需要大量关键改进")
            risks["risk_mitigation"]["需要大量关键改进"] = "优先处理最关键问题，分阶段实施"
        
        # 确定整体风险级别
        if len(risks["high_risks"]) > 2:
            risks["overall_risk_level"] = "高"
        elif len(risks["high_risks"]) > 0 or len(risks["medium_risks"]) > 3:
            risks["overall_risk_level"] = "中等"
        else:
            risks["overall_risk_level"] = "低"
        
        return risks
    
    def _define_next_steps(self,
                         validation_results: List[MarketValidationResult],
                         recommendations: List[StrategicRecommendation]) -> List[str]:
        """定义后续步骤"""
        next_steps = []
        
        # 基于验证结果
        failed_validations = [vr for vr in validation_results if vr.status == ValidationStatus.FAILED]
        if failed_validations:
            next_steps.append("立即启动关键问题改进计划")
            next_steps.append("为失败验证维度制定详细改进方案")
        
        # 基于建议优先级
        critical_recommendations = [r for r in recommendations if r.priority == RecommendationPriority.CRITICAL]
        if critical_recommendations:
            next_steps.append("优先实施关键优先级建议")
        
        # 通用步骤
        next_steps.extend([
            "建立定期评估机制，跟踪改进进展",
            "收集更多用户反馈，验证改进效果",
            "持续监控市场变化和竞争动态",
            "准备下一轮产品策略评估"
        ])
        
        return next_steps
    
    def _calculate_confidence_score(self, validation_results: List[MarketValidationResult]) -> float:
        """计算置信度分数"""
        confidence_scores = [vr.confidence_level for vr in validation_results]
        return np.mean(confidence_scores) if confidence_scores else 0.5
    
    async def generate_improvement_plan(self,
                                      evaluation_report: StrategyEvaluationReport) -> Dict[str, Any]:
        """生成改进计划"""
        self.logger.info("生成改进计划")
        
        # 基于评估报告生成具体改进计划
        improvement_plan = {
            "executive_summary": self._create_improvement_summary(evaluation_report),
            "priority_actions": self._extract_priority_actions(evaluation_report),
            "detailed_plan": self._create_detailed_improvement_plan(evaluation_report),
            "resource_allocation": self._plan_resource_allocation(evaluation_report),
            "timeline": self._create_improvement_timeline(evaluation_report),
            "success_metrics": self._define_improvement_metrics(evaluation_report),
            "risk_management": self._plan_risk_management(evaluation_report)
        }
        
        return improvement_plan
    
    def _create_improvement_summary(self, report: StrategyEvaluationReport) -> str:
        """创建改进摘要"""
        score = report.overall_score
        critical_count = len([r for r in report.strategic_recommendations 
                            if r.priority == RecommendationPriority.CRITICAL])
        
        return (
            f"基于当前{score:.1%}的策略评分，识别出{critical_count}个关键改进点。"
            f"通过系统性改进，预期可将整体评分提升至85%以上，显著提升项目成功概率。"
        )
    
    def _extract_priority_actions(self, report: StrategyEvaluationReport) -> List[Dict[str, Any]]:
        """提取优先行动"""
        actions = []
        
        critical_recommendations = [r for r in report.strategic_recommendations 
                                  if r.priority == RecommendationPriority.CRITICAL]
        
        for rec in critical_recommendations:
            actions.append({
                "action": rec.title,
                "description": rec.description,
                "timeline": rec.timeline,
                "expected_impact": rec.expected_impact
            })
        
        return actions
    
    def _create_detailed_improvement_plan(self, report: StrategyEvaluationReport) -> Dict[str, Any]:
        """创建详细改进计划"""
        return {
            "market_feasibility_improvements": self._plan_market_improvements(report),
            "user_value_enhancements": self._plan_user_value_improvements(report),
            "technical_optimizations": self._plan_technical_improvements(report),
            "business_model_refinements": self._plan_business_improvements(report),
            "competitive_positioning": self._plan_competitive_improvements(report),
            "execution_preparations": self._plan_execution_improvements(report)
        }
    
    def _plan_market_improvements(self, report: StrategyEvaluationReport) -> List[str]:
        """规划市场改进"""
        market_score = report.dimension_scores.get(ValidationDimension.MARKET_FEASIBILITY, 0.7)
        
        if market_score < 0.7:
            return [
                "进行深度市场研究，验证TAM/SAM/SOM",
                "分析竞争格局，重新评估市场定位",
                "调研市场时机，优化进入策略"
            ]
        else:
            return ["持续监控市场变化，优化市场策略"]
    
    def _plan_user_value_improvements(self, report: StrategyEvaluationReport) -> List[str]:
        """规划用户价值改进"""
        return [
            "深化用户研究，验证核心痛点",
            "优化价值主张表达，增强用户感知",
            "设计用户体验优化方案",
            "建立用户反馈收集机制"
        ]
    
    def _plan_technical_improvements(self, report: StrategyEvaluationReport) -> List[str]:
        """规划技术改进"""
        return [
            "简化技术架构，降低实现复杂度",
            "制定技术风险缓解方案",
            "建立技术原型，验证可行性",
            "优化性能和可扩展性设计"
        ]
    
    def _plan_business_improvements(self, report: StrategyEvaluationReport) -> List[str]:
        """规划商业改进"""
        return [
            "多样化收入来源设计",
            "优化成本结构模型",
            "验证定价策略可行性",
            "建立财务预测模型"
        ]
    
    def _plan_competitive_improvements(self, report: StrategyEvaluationReport) -> List[str]:
        """规划竞争改进"""
        return [
            "强化产品差异化特性",
            "建立可持续竞争壁垒",
            "制定竞争应对策略",
            "寻找细分市场机会"
        ]
    
    def _plan_execution_improvements(self, report: StrategyEvaluationReport) -> List[str]:
        """规划执行改进"""
        return [
            "评估和补强团队能力",
            "优化资源配置方案",
            "建立项目管理机制",
            "制定风险控制措施"
        ]
    
    def _plan_resource_allocation(self, report: StrategyEvaluationReport) -> Dict[str, Any]:
        """规划资源分配"""
        return {
            "人力资源": {
                "产品经理": "1人，负责整体改进协调",
                "市场分析师": "1人，负责市场研究",
                "技术架构师": "1人，负责技术优化",
                "用户研究员": "1人，负责用户调研"
            },
            "时间分配": {
                "市场研究": "30%",
                "用户调研": "25%",
                "技术优化": "25%",
                "商业模式": "20%"
            },
            "预算分配": {
                "市场调研": "40%",
                "技术开发": "35%",
                "用户研究": "15%",
                "其他": "10%"
            }
        }
    
    def _create_improvement_timeline(self, report: StrategyEvaluationReport) -> Dict[str, List[str]]:
        """创建改进时间线"""
        return {
            "第1-2周": [
                "启动关键问题改进",
                "组建改进团队",
                "制定详细工作计划"
            ],
            "第3-4周": [
                "执行市场研究",
                "深化用户调研",
                "技术方案优化"
            ],
            "第5-6周": [
                "商业模式完善",
                "竞争策略制定",
                "中期评估检查"
            ],
            "第7-8周": [
                "整合改进成果",
                "最终评估验证",
                "制定后续计划"
            ]
        }
    
    def _define_improvement_metrics(self, report: StrategyEvaluationReport) -> List[str]:
        """定义改进指标"""
        return [
            f"整体策略评分从{report.overall_score:.1%}提升至85%以上",
            "所有关键验证维度达到合格标准",
            "用户价值感知提升20%",
            "技术可行性风险降低50%",
            "市场定位清晰度提升30%",
            "执行计划完成率90%以上"
        ]
    
    def _plan_risk_management(self, report: StrategyEvaluationReport) -> Dict[str, Any]:
        """规划风险管理"""
        return {
            "主要风险": [
                "改进计划执行不到位",
                "市场环境变化",
                "资源投入不足",
                "团队能力不匹配"
            ],
            "缓解措施": {
                "改进计划执行不到位": "建立定期检查和报告机制",
                "市场环境变化": "持续监控市场动态，灵活调整策略",
                "资源投入不足": "确保必要资源承诺，分阶段投入",
                "团队能力不匹配": "提供必要培训，引入外部专家"
            },
            "应急预案": [
                "建立改进计划B版本",
                "准备外部咨询支持",
                "设立改进预算储备",
                "制定快速响应机制"
            ]
        }
    
    def get_advisor_summary(self) -> Dict[str, Any]:
        """获取顾问摘要"""
        return {
            "total_evaluations": len(self.evaluation_reports),
            "historical_validations": len(self.historical_validations),
            "average_overall_score": np.mean([r.overall_score for r in self.evaluation_reports.values()]) if self.evaluation_reports else 0,
            "validation_criteria_count": len(self.validation_criteria),
            "last_evaluation": max([r.evaluation_date for r in self.evaluation_reports.values()]) if self.evaluation_reports else None,
            "success_rate": len([r for r in self.evaluation_reports.values() if r.overall_score >= 0.7]) / len(self.evaluation_reports) if self.evaluation_reports else 0,
            "configuration": {
                "min_passing_score": self.min_passing_score,
                "excellence_threshold": self.excellence_threshold,
                "validation_enabled": self.validation_enabled
            }
        }