"""
Value-Bench: Benchmark System for AI Product Management Evaluation
Based on Section 2 of the AI-Product Manager paper
提供AI vs 人类产品管理能力的综合评估基准
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid
import statistics

class EvaluationDimension(Enum):
    """评估维度"""
    MARKET_INTELLIGENCE = "market_intelligence"
    USER_INSIGHT_GENERATION = "user_insight_generation"
    PRODUCT_DEFINITION = "product_definition"
    STRATEGY_FORMULATION = "strategy_formulation"
    DOCUMENTATION_QUALITY = "documentation_quality"
    INNOVATION_CREATIVITY = "innovation_creativity"
    BUSINESS_ACUMEN = "business_acumen"
    TECHNICAL_FEASIBILITY = "technical_feasibility"

class EvaluationMetric(Enum):
    """评估指标"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    INNOVATION_LEVEL = "innovation_level"
    PRACTICALITY = "practicality"
    EFFICIENCY = "efficiency"
    CONSISTENCY = "consistency"
    BUSINESS_VALUE = "business_value"

class ParticipantType(Enum):
    """参与者类型"""
    AI_SYSTEM = "ai_system"
    HUMAN_EXPERT = "human_expert"
    HYBRID_TEAM = "hybrid_team"

class TaskComplexity(Enum):
    """任务复杂度"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class BenchmarkTask:
    """基准测试任务"""
    task_id: str
    name: str
    description: str
    dimension: EvaluationDimension
    complexity: TaskComplexity
    input_data: Dict[str, Any]
    expected_outputs: List[str]
    evaluation_criteria: List[EvaluationMetric]
    time_limit: int  # minutes
    scoring_weights: Dict[EvaluationMetric, float]
    reference_solution: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class EvaluationResult:
    """评估结果"""
    result_id: str
    task_id: str
    participant_id: str
    participant_type: ParticipantType
    submission: Dict[str, Any]
    scores: Dict[EvaluationMetric, float]
    overall_score: float
    completion_time: int  # minutes
    feedback: List[str]
    timestamp: datetime
    reviewer_id: Optional[str]

@dataclass
class BenchmarkSuite:
    """基准测试套件"""
    suite_id: str
    name: str
    description: str
    tasks: List[BenchmarkTask]
    total_score: float
    difficulty_level: str
    estimated_duration: int  # minutes
    prerequisites: List[str]

@dataclass
class ParticipantProfile:
    """参与者档案"""
    participant_id: str
    name: str
    type: ParticipantType
    experience_level: str
    specializations: List[str]
    previous_scores: List[float]
    metadata: Dict[str, Any]

@dataclass
class ComparisonReport:
    """对比报告"""
    report_id: str
    comparison_date: datetime
    ai_participants: List[str]
    human_participants: List[str]
    tasks_evaluated: List[str]
    performance_comparison: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]

class ValueBench:
    """Value-Bench 基准评估系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 基准配置
        self.scoring_precision = config.get('scoring_precision', 2)
        self.min_participants_for_comparison = config.get('min_participants_for_comparison', 3)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        
        # 评估权重配置
        self.dimension_weights = {
            EvaluationDimension.MARKET_INTELLIGENCE: 0.15,
            EvaluationDimension.USER_INSIGHT_GENERATION: 0.15,
            EvaluationDimension.PRODUCT_DEFINITION: 0.15,
            EvaluationDimension.STRATEGY_FORMULATION: 0.15,
            EvaluationDimension.DOCUMENTATION_QUALITY: 0.10,
            EvaluationDimension.INNOVATION_CREATIVITY: 0.10,
            EvaluationDimension.BUSINESS_ACUMEN: 0.10,
            EvaluationDimension.TECHNICAL_FEASIBILITY: 0.10
        }
        
        # 数据存储
        self.benchmark_suites: Dict[str, BenchmarkSuite] = {}
        self.participants: Dict[str, ParticipantProfile] = {}
        self.evaluation_results: List[EvaluationResult] = []
        self.comparison_reports: List[ComparisonReport] = []
        
        # 初始化基准套件
        self._initialize_benchmark_suites()
        
    def _initialize_benchmark_suites(self):
        """初始化基准测试套件"""
        # 创建不同复杂度的测试套件
        self._create_basic_suite()
        self._create_intermediate_suite()
        self._create_advanced_suite()
        self._create_expert_suite()
    
    def _create_basic_suite(self):
        """创建基础测试套件"""
        tasks = []
        
        # 市场情报基础任务
        market_intel_task = BenchmarkTask(
            task_id="basic_market_001",
            name="基础市场分析",
            description="分析给定的市场数据，识别关键趋势和机会",
            dimension=EvaluationDimension.MARKET_INTELLIGENCE,
            complexity=TaskComplexity.BASIC,
            input_data={
                "market_reports": ["AI市场报告2024", "竞争对手分析"],
                "financial_data": {"market_size": 387000000000, "growth_rate": 0.37},
                "time_period": "2024-2026"
            },
            expected_outputs=["市场规模分析", "竞争格局评估", "增长机会识别"],
            evaluation_criteria=[
                EvaluationMetric.ACCURACY,
                EvaluationMetric.COMPLETENESS,
                EvaluationMetric.COHERENCE
            ],
            time_limit=60,
            scoring_weights={
                EvaluationMetric.ACCURACY: 0.4,
                EvaluationMetric.COMPLETENESS: 0.3,
                EvaluationMetric.COHERENCE: 0.3
            },
            reference_solution=None,
            metadata={"domain": "general_ai", "difficulty": 1}
        )
        tasks.append(market_intel_task)
        
        # 用户洞察基础任务
        user_insight_task = BenchmarkTask(
            task_id="basic_user_001",
            name="用户痛点识别",
            description="基于用户反馈数据识别核心痛点和需求",
            dimension=EvaluationDimension.USER_INSIGHT_GENERATION,
            complexity=TaskComplexity.BASIC,
            input_data={
                "user_feedback": [
                    "AI工具太复杂，学习成本高",
                    "结果不够准确，需要人工验证",
                    "响应速度慢，影响工作效率"
                ],
                "usage_data": {"daily_active_users": 10000, "completion_rate": 0.65},
                "satisfaction_scores": [3.2, 3.8, 2.9, 4.1, 3.5]
            },
            expected_outputs=["痛点优先级列表", "用户需求分析", "改进建议"],
            evaluation_criteria=[
                EvaluationMetric.ACCURACY,
                EvaluationMetric.PRACTICALITY,
                EvaluationMetric.BUSINESS_VALUE
            ],
            time_limit=45,
            scoring_weights={
                EvaluationMetric.ACCURACY: 0.35,
                EvaluationMetric.PRACTICALITY: 0.35,
                EvaluationMetric.BUSINESS_VALUE: 0.3
            },
            reference_solution=None,
            metadata={"user_segment": "general", "data_quality": "high"}
        )
        tasks.append(user_insight_task)
        
        # 产品定义基础任务
        product_def_task = BenchmarkTask(
            task_id="basic_product_001",
            name="基础产品规格定义",
            description="基于市场需求定义基础的产品功能和规格",
            dimension=EvaluationDimension.PRODUCT_DEFINITION,
            complexity=TaskComplexity.BASIC,
            input_data={
                "market_need": "简化AI工具使用，提高用户采用率",
                "target_users": ["小企业用户", "非技术用户"],
                "constraints": {"budget": 500000, "timeline": "6个月", "team_size": 5}
            },
            expected_outputs=["核心功能列表", "用户界面设计要求", "技术规格概述"],
            evaluation_criteria=[
                EvaluationMetric.COMPLETENESS,
                EvaluationMetric.PRACTICALITY,
                EvaluationMetric.COHERENCE
            ],
            time_limit=90,
            scoring_weights={
                EvaluationMetric.COMPLETENESS: 0.4,
                EvaluationMetric.PRACTICALITY: 0.35,
                EvaluationMetric.COHERENCE: 0.25
            },
            reference_solution=None,
            metadata={"product_type": "software", "target_market": "smb"}
        )
        tasks.append(product_def_task)
        
        basic_suite = BenchmarkSuite(
            suite_id="basic_suite_001",
            name="基础产品管理能力测试",
            description="评估基础的产品管理核心技能",
            tasks=tasks,
            total_score=100.0,
            difficulty_level="Basic",
            estimated_duration=195,  # 总时间
            prerequisites=[]
        )
        
        self.benchmark_suites["basic"] = basic_suite
    
    def _create_intermediate_suite(self):
        """创建中级测试套件"""
        tasks = []
        
        # 战略制定中级任务
        strategy_task = BenchmarkTask(
            task_id="inter_strategy_001",
            name="产品战略制定",
            description="基于复杂市场环境制定综合产品战略",
            dimension=EvaluationDimension.STRATEGY_FORMULATION,
            complexity=TaskComplexity.INTERMEDIATE,
            input_data={
                "market_analysis": {
                    "competitors": 8,
                    "market_growth": 0.25,
                    "user_segments": 4,
                    "regulatory_changes": True
                },
                "product_portfolio": ["产品A", "产品B", "新产品C"],
                "resources": {"team": 15, "budget": 2000000, "timeline": "12个月"}
            },
            expected_outputs=[
                "产品定位策略",
                "竞争差异化方案",
                "资源分配计划",
                "风险缓解策略"
            ],
            evaluation_criteria=[
                EvaluationMetric.COHERENCE,
                EvaluationMetric.INNOVATION_LEVEL,
                EvaluationMetric.BUSINESS_VALUE,
                EvaluationMetric.PRACTICALITY
            ],
            time_limit=120,
            scoring_weights={
                EvaluationMetric.COHERENCE: 0.25,
                EvaluationMetric.INNOVATION_LEVEL: 0.25,
                EvaluationMetric.BUSINESS_VALUE: 0.25,
                EvaluationMetric.PRACTICALITY: 0.25
            },
            reference_solution=None,
            metadata={"complexity_factors": ["multi_product", "competitive", "resource_constrained"]}
        )
        tasks.append(strategy_task)
        
        # 技术可行性中级任务
        tech_feasibility_task = BenchmarkTask(
            task_id="inter_tech_001",
            name="技术架构可行性评估",
            description="评估复杂产品的技术可行性和架构方案",
            dimension=EvaluationDimension.TECHNICAL_FEASIBILITY,
            complexity=TaskComplexity.INTERMEDIATE,
            input_data={
                "product_requirements": {
                    "users": 100000,
                    "performance": "< 200ms响应",
                    "availability": "99.9%",
                    "security": "企业级",
                    "scalability": "10x增长支持"
                },
                "technical_constraints": {
                    "budget": 1000000,
                    "team_expertise": ["Python", "React", "AWS"],
                    "timeline": "8个月"
                },
                "integration_requirements": ["CRM", "ERP", "第三方API"]
            },
            expected_outputs=[
                "技术架构设计",
                "可行性风险评估",
                "实施计划",
                "技术选型建议"
            ],
            evaluation_criteria=[
                EvaluationMetric.ACCURACY,
                EvaluationMetric.PRACTICALITY,
                EvaluationMetric.COHERENCE,
                EvaluationMetric.CONSISTENCY
            ],
            time_limit=150,
            scoring_weights={
                EvaluationMetric.ACCURACY: 0.3,
                EvaluationMetric.PRACTICALITY: 0.3,
                EvaluationMetric.COHERENCE: 0.2,
                EvaluationMetric.CONSISTENCY: 0.2
            },
            reference_solution=None,
            metadata={"technical_depth": "architectural", "integration_complexity": "high"}
        )
        tasks.append(tech_feasibility_task)
        
        intermediate_suite = BenchmarkSuite(
            suite_id="inter_suite_001",
            name="中级产品管理综合测试",
            description="评估中级产品管理技能和战略思维",
            tasks=tasks,
            total_score=100.0,
            difficulty_level="Intermediate",
            estimated_duration=270,
            prerequisites=["basic"]
        )
        
        self.benchmark_suites["intermediate"] = intermediate_suite
    
    def _create_advanced_suite(self):
        """创建高级测试套件"""
        tasks = []
        
        # 创新创意高级任务
        innovation_task = BenchmarkTask(
            task_id="adv_innovation_001",
            name="突破性产品创新设计",
            description="在现有范式基础上设计突破性的产品创新方案",
            dimension=EvaluationDimension.INNOVATION_CREATIVITY,
            complexity=TaskComplexity.ADVANCED,
            input_data={
                "industry_context": {
                    "mature_market": True,
                    "technology_plateau": True,
                    "user_expectation_fatigue": True,
                    "regulatory_pressure": True
                },
                "constraints": {
                    "must_be_profitable": True,
                    "maintain_compatibility": True,
                    "environmental_responsibility": True
                },
                "emerging_technologies": ["量子计算", "脑机接口", "增强现实", "边缘AI"]
            },
            expected_outputs=[
                "突破性产品概念",
                "创新实现路径",
                "商业化可行性分析",
                "市场颠覆潜力评估"
            ],
            evaluation_criteria=[
                EvaluationMetric.INNOVATION_LEVEL,
                EvaluationMetric.BUSINESS_VALUE,
                EvaluationMetric.COHERENCE,
                EvaluationMetric.PRACTICALITY
            ],
            time_limit=180,
            scoring_weights={
                EvaluationMetric.INNOVATION_LEVEL: 0.35,
                EvaluationMetric.BUSINESS_VALUE: 0.25,
                EvaluationMetric.COHERENCE: 0.2,
                EvaluationMetric.PRACTICALITY: 0.2
            },
            reference_solution=None,
            metadata={"innovation_type": "disruptive", "market_impact": "transformational"}
        )
        tasks.append(innovation_task)
        
        # 商业敏锐度高级任务
        business_acumen_task = BenchmarkTask(
            task_id="adv_business_001",
            name="多市场商业模式优化",
            description="在多个市场环境下优化复杂的商业模式",
            dimension=EvaluationDimension.BUSINESS_ACUMEN,
            complexity=TaskComplexity.ADVANCED,
            input_data={
                "markets": {
                    "north_america": {"maturity": "mature", "growth": 0.15, "competition": "high"},
                    "europe": {"maturity": "mature", "growth": 0.12, "regulation": "strict"},
                    "asia_pacific": {"maturity": "emerging", "growth": 0.45, "fragmentation": "high"},
                    "latin_america": {"maturity": "developing", "growth": 0.32, "volatility": "high"}
                },
                "current_model": {
                    "revenue_streams": ["subscription", "transaction_fees", "premium_features"],
                    "cost_structure": {"fixed": 0.4, "variable": 0.6},
                    "margins": {"gross": 0.65, "operating": 0.15}
                },
                "challenges": ["currency_fluctuation", "regulatory_compliance", "local_competition"]
            },
            expected_outputs=[
                "优化商业模式方案",
                "市场进入策略",
                "风险管控机制",
                "财务预测模型"
            ],
            evaluation_criteria=[
                EvaluationMetric.BUSINESS_VALUE,
                EvaluationMetric.COHERENCE,
                EvaluationMetric.PRACTICALITY,
                EvaluationMetric.CONSISTENCY
            ],
            time_limit=240,
            scoring_weights={
                EvaluationMetric.BUSINESS_VALUE: 0.35,
                EvaluationMetric.COHERENCE: 0.25,
                EvaluationMetric.PRACTICALITY: 0.25,
                EvaluationMetric.CONSISTENCY: 0.15
            },
            reference_solution=None,
            metadata={"geographic_scope": "global", "business_complexity": "multi_model"}
        )
        tasks.append(business_acumen_task)
        
        advanced_suite = BenchmarkSuite(
            suite_id="adv_suite_001",
            name="高级产品管理专家测试",
            description="评估高级产品管理专业技能和创新能力",
            tasks=tasks,
            total_score=100.0,
            difficulty_level="Advanced",
            estimated_duration=420,
            prerequisites=["basic", "intermediate"]
        )
        
        self.benchmark_suites["advanced"] = advanced_suite
    
    def _create_expert_suite(self):
        """创建专家级测试套件"""
        tasks = []
        
        # 综合能力专家任务
        comprehensive_task = BenchmarkTask(
            task_id="expert_comp_001",
            name="端到端产品生态系统设计",
            description="设计完整的产品生态系统，包含多个相互关联的产品和服务",
            dimension=EvaluationDimension.STRATEGY_FORMULATION,
            complexity=TaskComplexity.EXPERT,
            input_data={
                "ecosystem_scope": {
                    "core_products": 3,
                    "complementary_services": 5,
                    "platform_components": 2,
                    "third_party_integrations": 10
                },
                "stakeholders": {
                    "end_users": ["consumers", "businesses", "developers"],
                    "partners": ["technology", "distribution", "content"],
                    "regulators": ["data_protection", "competition", "industry_specific"]
                },
                "constraints": {
                    "technology_evolution": "rapid",
                    "market_dynamics": "volatile",
                    "resource_limitations": "significant",
                    "competitive_pressure": "intense"
                }
            },
            expected_outputs=[
                "生态系统架构设计",
                "产品互动矩阵",
                "价值网络分析",
                "演进路线图",
                "风险管控策略"
            ],
            evaluation_criteria=[
                EvaluationMetric.COHERENCE,
                EvaluationMetric.INNOVATION_LEVEL,
                EvaluationMetric.BUSINESS_VALUE,
                EvaluationMetric.COMPLETENESS,
                EvaluationMetric.CONSISTENCY
            ],
            time_limit=360,
            scoring_weights={
                EvaluationMetric.COHERENCE: 0.25,
                EvaluationMetric.INNOVATION_LEVEL: 0.2,
                EvaluationMetric.BUSINESS_VALUE: 0.2,
                EvaluationMetric.COMPLETENESS: 0.2,
                EvaluationMetric.CONSISTENCY: 0.15
            },
            reference_solution=None,
            metadata={"complexity": "ecosystem", "scope": "enterprise", "innovation_requirement": "breakthrough"}
        )
        tasks.append(comprehensive_task)
        
        expert_suite = BenchmarkSuite(
            suite_id="expert_suite_001",
            name="专家级产品生态系统设计",
            description="评估专家级的产品生态系统设计和战略规划能力",
            tasks=tasks,
            total_score=100.0,
            difficulty_level="Expert",
            estimated_duration=360,
            prerequisites=["basic", "intermediate", "advanced"]
        )
        
        self.benchmark_suites["expert"] = expert_suite
    
    async def register_participant(self, 
                                 name: str,
                                 participant_type: ParticipantType,
                                 experience_level: str,
                                 specializations: List[str] = None,
                                 metadata: Dict[str, Any] = None) -> str:
        """注册参与者"""
        participant_id = str(uuid.uuid4())
        
        participant = ParticipantProfile(
            participant_id=participant_id,
            name=name,
            type=participant_type,
            experience_level=experience_level,
            specializations=specializations or [],
            previous_scores=[],
            metadata=metadata or {}
        )
        
        self.participants[participant_id] = participant
        self.logger.info(f"注册参与者: {name} ({participant_type.value})")
        
        return participant_id
    
    async def conduct_evaluation(self,
                               participant_id: str,
                               suite_name: str,
                               submissions: Dict[str, Dict[str, Any]],
                               reviewer_id: Optional[str] = None) -> List[EvaluationResult]:
        """进行评估"""
        if participant_id not in self.participants:
            raise ValueError(f"参与者未注册: {participant_id}")
        
        if suite_name not in self.benchmark_suites:
            raise ValueError(f"测试套件不存在: {suite_name}")
        
        suite = self.benchmark_suites[suite_name]
        participant = self.participants[participant_id]
        
        self.logger.info(f"开始评估 {participant.name} 在 {suite.name} 上的表现")
        
        results = []
        
        for task in suite.tasks:
            if task.task_id not in submissions:
                self.logger.warning(f"缺少任务提交: {task.task_id}")
                continue
            
            submission = submissions[task.task_id]
            result = await self._evaluate_single_task(
                task, participant, submission, reviewer_id
            )
            results.append(result)
            self.evaluation_results.append(result)
        
        # 更新参与者历史分数
        overall_scores = [r.overall_score for r in results]
        if overall_scores:
            avg_score = statistics.mean(overall_scores)
            participant.previous_scores.append(avg_score)
        
        self.logger.info(f"评估完成，共评估 {len(results)} 个任务")
        return results
    
    async def _evaluate_single_task(self,
                                   task: BenchmarkTask,
                                   participant: ParticipantProfile,
                                   submission: Dict[str, Any],
                                   reviewer_id: Optional[str]) -> EvaluationResult:
        """评估单个任务"""
        scores = {}
        feedback = []
        
        # 根据不同评估指标计算分数
        for metric in task.evaluation_criteria:
            score = await self._calculate_metric_score(metric, task, submission, participant)
            scores[metric] = score
        
        # 计算加权总分
        overall_score = sum(
            scores[metric] * task.scoring_weights.get(metric, 0)
            for metric in scores
        )
        
        # 生成反馈
        feedback = await self._generate_task_feedback(task, submission, scores)
        
        result = EvaluationResult(
            result_id=str(uuid.uuid4()),
            task_id=task.task_id,
            participant_id=participant.participant_id,
            participant_type=participant.type,
            submission=submission,
            scores=scores,
            overall_score=round(overall_score, self.scoring_precision),
            completion_time=submission.get('completion_time', task.time_limit),
            feedback=feedback,
            timestamp=datetime.now(),
            reviewer_id=reviewer_id
        )
        
        return result
    
    async def _calculate_metric_score(self,
                                    metric: EvaluationMetric,
                                    task: BenchmarkTask,
                                    submission: Dict[str, Any],
                                    participant: ParticipantProfile) -> float:
        """计算指标分数"""
        if metric == EvaluationMetric.ACCURACY:
            return await self._evaluate_accuracy(task, submission)
        elif metric == EvaluationMetric.COMPLETENESS:
            return await self._evaluate_completeness(task, submission)
        elif metric == EvaluationMetric.COHERENCE:
            return await self._evaluate_coherence(submission)
        elif metric == EvaluationMetric.INNOVATION_LEVEL:
            return await self._evaluate_innovation(submission)
        elif metric == EvaluationMetric.PRACTICALITY:
            return await self._evaluate_practicality(task, submission)
        elif metric == EvaluationMetric.EFFICIENCY:
            return await self._evaluate_efficiency(task, submission)
        elif metric == EvaluationMetric.CONSISTENCY:
            return await self._evaluate_consistency(submission)
        elif metric == EvaluationMetric.BUSINESS_VALUE:
            return await self._evaluate_business_value(task, submission)
        else:
            return 0.5  # 默认分数
    
    async def _evaluate_accuracy(self, task: BenchmarkTask, submission: Dict[str, Any]) -> float:
        """评估准确性"""
        # 模拟准确性评估逻辑
        outputs = submission.get('outputs', [])
        expected = task.expected_outputs
        
        if not outputs:
            return 0.0
        
        # 检查输出是否包含期望的关键元素
        accuracy_score = 0.0
        for expected_output in expected:
            for output in outputs:
                if isinstance(output, str) and expected_output.lower() in output.lower():
                    accuracy_score += 1.0
                    break
        
        return min(1.0, accuracy_score / len(expected))
    
    async def _evaluate_completeness(self, task: BenchmarkTask, submission: Dict[str, Any]) -> float:
        """评估完整性"""
        outputs = submission.get('outputs', [])
        expected_count = len(task.expected_outputs)
        actual_count = len(outputs)
        
        if expected_count == 0:
            return 1.0
        
        # 基础完整性分数
        base_score = min(1.0, actual_count / expected_count)
        
        # 内容质量调整
        quality_adjustment = 0.0
        for output in outputs:
            if isinstance(output, str):
                if len(output) > 50:  # 有实质内容
                    quality_adjustment += 0.1
        
        return min(1.0, base_score + quality_adjustment)
    
    async def _evaluate_coherence(self, submission: Dict[str, Any]) -> float:
        """评估连贯性"""
        outputs = submission.get('outputs', [])
        
        if len(outputs) < 2:
            return 0.8  # 单个输出默认分数
        
        # 模拟连贯性检查
        coherence_score = 0.75  # 基础分数
        
        # 检查输出间的逻辑一致性
        for i in range(len(outputs) - 1):
            if isinstance(outputs[i], str) and isinstance(outputs[i+1], str):
                # 简单的一致性检查
                if len(outputs[i]) > 0 and len(outputs[i+1]) > 0:
                    coherence_score += 0.05
        
        return min(1.0, coherence_score)
    
    async def _evaluate_innovation(self, submission: Dict[str, Any]) -> float:
        """评估创新水平"""
        # 创新关键词检测
        innovation_keywords = [
            "创新", "突破", "革命性", "颠覆", "原创", "独特",
            "新颖", "先进", "前沿", "尖端", "开创性"
        ]
        
        outputs = submission.get('outputs', [])
        innovation_score = 0.5  # 基础分数
        
        for output in outputs:
            if isinstance(output, str):
                for keyword in innovation_keywords:
                    if keyword in output:
                        innovation_score += 0.05
        
        # 检查解决方案的独特性
        approach = submission.get('approach', '')
        if '创新' in approach or '新方法' in approach:
            innovation_score += 0.1
        
        return min(1.0, innovation_score)
    
    async def _evaluate_practicality(self, task: BenchmarkTask, submission: Dict[str, Any]) -> float:
        """评估实用性"""
        constraints = task.input_data.get('constraints', {})
        solution = submission.get('solution', {})
        
        practicality_score = 0.7  # 基础分数
        
        # 检查是否考虑了约束条件
        if 'budget' in constraints:
            budget_mentioned = any('预算' in str(output) or '成本' in str(output) 
                                 for output in submission.get('outputs', []))
            if budget_mentioned:
                practicality_score += 0.1
        
        if 'timeline' in constraints:
            timeline_mentioned = any('时间' in str(output) or '计划' in str(output)
                                   for output in submission.get('outputs', []))
            if timeline_mentioned:
                practicality_score += 0.1
        
        # 检查实施可行性
        if '实施' in str(solution) or '执行' in str(solution):
            practicality_score += 0.1
        
        return min(1.0, practicality_score)
    
    async def _evaluate_efficiency(self, task: BenchmarkTask, submission: Dict[str, Any]) -> float:
        """评估效率"""
        completion_time = submission.get('completion_time', task.time_limit)
        time_ratio = completion_time / task.time_limit
        
        if time_ratio <= 0.5:
            return 1.0
        elif time_ratio <= 0.75:
            return 0.9
        elif time_ratio <= 1.0:
            return 0.8
        else:
            return max(0.3, 1.0 - (time_ratio - 1.0))
    
    async def _evaluate_consistency(self, submission: Dict[str, Any]) -> float:
        """评估一致性"""
        outputs = submission.get('outputs', [])
        
        if len(outputs) < 2:
            return 0.8
        
        # 检查术语和概念的一致性
        consistency_score = 0.75
        
        # 简单的一致性检查逻辑
        key_terms = set()
        for output in outputs:
            if isinstance(output, str):
                words = output.split()
                key_terms.update([word for word in words if len(word) > 5])
        
        # 如果有重复使用的关键词，说明一致性较好
        if len(key_terms) > 0:
            consistency_score += 0.15
        
        return min(1.0, consistency_score)
    
    async def _evaluate_business_value(self, task: BenchmarkTask, submission: Dict[str, Any]) -> float:
        """评估商业价值"""
        business_keywords = [
            "收入", "利润", "成本", "ROI", "市场份额", "竞争优势",
            "用户价值", "商业模式", "盈利", "投资回报"
        ]
        
        outputs = submission.get('outputs', [])
        business_value_score = 0.5
        
        for output in outputs:
            if isinstance(output, str):
                for keyword in business_keywords:
                    if keyword in output:
                        business_value_score += 0.05
        
        # 检查是否有量化的商业指标
        if any(char.isdigit() and '%' in str(output) for output in outputs):
            business_value_score += 0.1
        
        return min(1.0, business_value_score)
    
    async def _generate_task_feedback(self,
                                    task: BenchmarkTask,
                                    submission: Dict[str, Any],
                                    scores: Dict[EvaluationMetric, float]) -> List[str]:
        """生成任务反馈"""
        feedback = []
        
        # 基于分数生成反馈
        for metric, score in scores.items():
            if score >= 0.8:
                feedback.append(f"{metric.value}: 表现优秀 ({score:.2f})")
            elif score >= 0.6:
                feedback.append(f"{metric.value}: 表现良好 ({score:.2f})")
            elif score >= 0.4:
                feedback.append(f"{metric.value}: 需要改进 ({score:.2f})")
            else:
                feedback.append(f"{metric.value}: 表现不佳 ({score:.2f})")
        
        # 针对性建议
        if EvaluationMetric.COMPLETENESS in scores and scores[EvaluationMetric.COMPLETENESS] < 0.6:
            feedback.append("建议: 确保回答涵盖所有要求的输出项目")
        
        if EvaluationMetric.INNOVATION_LEVEL in scores and scores[EvaluationMetric.INNOVATION_LEVEL] < 0.6:
            feedback.append("建议: 尝试提出更具创新性的解决方案")
        
        if EvaluationMetric.PRACTICALITY in scores and scores[EvaluationMetric.PRACTICALITY] < 0.6:
            feedback.append("建议: 更多考虑实际约束条件和实施可行性")
        
        return feedback
    
    async def generate_comparison_report(self,
                                       ai_participant_ids: List[str],
                                       human_participant_ids: List[str],
                                       task_ids: List[str] = None) -> ComparisonReport:
        """生成AI vs 人类对比报告"""
        self.logger.info("生成AI vs 人类对比报告")
        
        # 筛选相关评估结果
        ai_results = [r for r in self.evaluation_results 
                     if r.participant_id in ai_participant_ids]
        human_results = [r for r in self.evaluation_results 
                        if r.participant_id in human_participant_ids]
        
        if task_ids:
            ai_results = [r for r in ai_results if r.task_id in task_ids]
            human_results = [r for r in human_results if r.task_id in task_ids]
        
        # 性能对比分析
        performance_comparison = await self._analyze_performance_comparison(
            ai_results, human_results
        )
        
        # 统计分析
        statistical_analysis = await self._perform_statistical_analysis(
            ai_results, human_results
        )
        
        # 洞察生成
        insights = await self._generate_comparison_insights(
            performance_comparison, statistical_analysis
        )
        
        # 建议生成
        recommendations = await self._generate_comparison_recommendations(
            performance_comparison, insights
        )
        
        report = ComparisonReport(
            report_id=str(uuid.uuid4()),
            comparison_date=datetime.now(),
            ai_participants=ai_participant_ids,
            human_participants=human_participant_ids,
            tasks_evaluated=task_ids or [r.task_id for r in ai_results + human_results],
            performance_comparison=performance_comparison,
            statistical_analysis=statistical_analysis,
            insights=insights,
            recommendations=recommendations
        )
        
        self.comparison_reports.append(report)
        self.logger.info("对比报告生成完成")
        
        return report
    
    async def _analyze_performance_comparison(self,
                                            ai_results: List[EvaluationResult],
                                            human_results: List[EvaluationResult]) -> Dict[str, Any]:
        """分析性能对比"""
        comparison = {
            "overall_performance": {},
            "dimension_performance": {},
            "metric_performance": {},
            "efficiency_comparison": {},
            "consistency_comparison": {}
        }
        
        # 总体性能对比
        ai_scores = [r.overall_score for r in ai_results]
        human_scores = [r.overall_score for r in human_results]
        
        comparison["overall_performance"] = {
            "ai_average": statistics.mean(ai_scores) if ai_scores else 0,
            "human_average": statistics.mean(human_scores) if human_scores else 0,
            "ai_median": statistics.median(ai_scores) if ai_scores else 0,
            "human_median": statistics.median(human_scores) if human_scores else 0,
            "ai_std": statistics.stdev(ai_scores) if len(ai_scores) > 1 else 0,
            "human_std": statistics.stdev(human_scores) if len(human_scores) > 1 else 0
        }
        
        # 按维度对比
        for dimension in EvaluationDimension:
            ai_dim_results = [r for r in ai_results 
                             if self._get_task_dimension(r.task_id) == dimension]
            human_dim_results = [r for r in human_results 
                               if self._get_task_dimension(r.task_id) == dimension]
            
            if ai_dim_results or human_dim_results:
                ai_dim_scores = [r.overall_score for r in ai_dim_results]
                human_dim_scores = [r.overall_score for r in human_dim_results]
                
                comparison["dimension_performance"][dimension.value] = {
                    "ai_average": statistics.mean(ai_dim_scores) if ai_dim_scores else 0,
                    "human_average": statistics.mean(human_dim_scores) if human_dim_scores else 0,
                    "sample_size_ai": len(ai_dim_scores),
                    "sample_size_human": len(human_dim_scores)
                }
        
        # 效率对比
        ai_times = [r.completion_time for r in ai_results]
        human_times = [r.completion_time for r in human_results]
        
        comparison["efficiency_comparison"] = {
            "ai_average_time": statistics.mean(ai_times) if ai_times else 0,
            "human_average_time": statistics.mean(human_times) if human_times else 0,
            "time_advantage": "ai" if (ai_times and human_times and 
                                    statistics.mean(ai_times) < statistics.mean(human_times)) else "human"
        }
        
        return comparison
    
    async def _perform_statistical_analysis(self,
                                          ai_results: List[EvaluationResult],
                                          human_results: List[EvaluationResult]) -> Dict[str, Any]:
        """执行统计分析"""
        # 简化的统计分析
        ai_scores = [r.overall_score for r in ai_results]
        human_scores = [r.overall_score for r in human_results]
        
        analysis = {
            "sample_sizes": {
                "ai": len(ai_scores),
                "human": len(human_scores)
            },
            "descriptive_stats": {
                "ai": {
                    "mean": statistics.mean(ai_scores) if ai_scores else 0,
                    "median": statistics.median(ai_scores) if ai_scores else 0,
                    "std": statistics.stdev(ai_scores) if len(ai_scores) > 1 else 0
                },
                "human": {
                    "mean": statistics.mean(human_scores) if human_scores else 0,
                    "median": statistics.median(human_scores) if human_scores else 0,
                    "std": statistics.stdev(human_scores) if len(human_scores) > 1 else 0
                }
            },
            "effect_size": 0.0,  # 简化实现
            "confidence_interval": [0.0, 0.0],  # 简化实现
            "significance_level": 0.05
        }
        
        # 计算效应大小 (Cohen's d 的简化版本)
        if ai_scores and human_scores and len(ai_scores) > 1 and len(human_scores) > 1:
            ai_mean = statistics.mean(ai_scores)
            human_mean = statistics.mean(human_scores)
            pooled_std = (statistics.stdev(ai_scores) + statistics.stdev(human_scores)) / 2
            
            if pooled_std > 0:
                analysis["effect_size"] = (ai_mean - human_mean) / pooled_std
        
        return analysis
    
    async def _generate_comparison_insights(self,
                                          performance_comparison: Dict[str, Any],
                                          statistical_analysis: Dict[str, Any]) -> List[str]:
        """生成对比洞察"""
        insights = []
        
        # 总体性能洞察
        ai_avg = performance_comparison["overall_performance"]["ai_average"]
        human_avg = performance_comparison["overall_performance"]["human_average"]
        
        if ai_avg > human_avg:
            insights.append(f"AI系统总体表现优于人类专家 ({ai_avg:.2f} vs {human_avg:.2f})")
        elif human_avg > ai_avg:
            insights.append(f"人类专家总体表现优于AI系统 ({human_avg:.2f} vs {ai_avg:.2f})")
        else:
            insights.append("AI系统与人类专家表现基本相当")
        
        # 效率洞察
        ai_time = performance_comparison["efficiency_comparison"]["ai_average_time"]
        human_time = performance_comparison["efficiency_comparison"]["human_average_time"]
        
        if ai_time < human_time:
            time_saving = ((human_time - ai_time) / human_time) * 100
            insights.append(f"AI系统在效率方面显著领先，平均节省{time_saving:.1f}%的时间")
        
        # 一致性洞察
        ai_std = performance_comparison["overall_performance"]["ai_std"]
        human_std = performance_comparison["overall_performance"]["human_std"]
        
        if ai_std < human_std:
            insights.append("AI系统表现更加稳定一致，人类专家表现存在较大变异性")
        elif human_std < ai_std:
            insights.append("人类专家表现更加稳定，AI系统结果存在一定变异性")
        
        # 维度特定洞察
        for dimension, stats in performance_comparison["dimension_performance"].items():
            ai_score = stats["ai_average"]
            human_score = stats["human_average"]
            
            if abs(ai_score - human_score) > 0.1:  # 显著差异
                if ai_score > human_score:
                    insights.append(f"在{dimension}维度，AI系统表现明显优于人类")
                else:
                    insights.append(f"在{dimension}维度，人类专家表现明显优于AI系统")
        
        return insights
    
    async def _generate_comparison_recommendations(self,
                                                 performance_comparison: Dict[str, Any],
                                                 insights: List[str]) -> List[str]:
        """生成对比建议"""
        recommendations = []
        
        ai_avg = performance_comparison["overall_performance"]["ai_average"]
        human_avg = performance_comparison["overall_performance"]["human_average"]
        
        if ai_avg > human_avg:
            recommendations.append("考虑在产品管理流程中更多采用AI系统以提升整体效率")
            recommendations.append("建立AI-人类协作机制，发挥各自优势")
        else:
            recommendations.append("继续重视人类专家在产品管理中的核心作用")
            recommendations.append("针对AI系统的弱项进行改进和优化")
        
        # 基于维度表现的建议
        for dimension, stats in performance_comparison["dimension_performance"].items():
            ai_score = stats["ai_average"]
            human_score = stats["human_average"]
            
            if ai_score > human_score + 0.1:
                recommendations.append(f"在{dimension}领域，优先使用AI系统")
            elif human_score > ai_score + 0.1:
                recommendations.append(f"在{dimension}领域，依赖人类专家判断")
        
        # 效率建议
        ai_time = performance_comparison["efficiency_comparison"]["ai_average_time"]
        human_time = performance_comparison["efficiency_comparison"]["human_average_time"]
        
        if ai_time < human_time * 0.5:
            recommendations.append("利用AI系统的速度优势，处理大量常规性产品管理任务")
        
        recommendations.append("建立持续的评估机制，跟踪AI vs 人类表现的变化趋势")
        recommendations.append("投资于AI系统和人类专家的能力提升")
        
        return recommendations
    
    def _get_task_dimension(self, task_id: str) -> Optional[EvaluationDimension]:
        """获取任务维度"""
        for suite in self.benchmark_suites.values():
            for task in suite.tasks:
                if task.task_id == task_id:
                    return task.dimension
        return None
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """获取基准测试摘要"""
        return {
            "benchmark_suites": len(self.benchmark_suites),
            "total_tasks": sum(len(suite.tasks) for suite in self.benchmark_suites.values()),
            "registered_participants": len(self.participants),
            "completed_evaluations": len(self.evaluation_results),
            "comparison_reports": len(self.comparison_reports),
            "available_suites": list(self.benchmark_suites.keys()),
            "evaluation_dimensions": [dim.value for dim in EvaluationDimension],
            "evaluation_metrics": [metric.value for metric in EvaluationMetric],
            "last_updated": datetime.now().isoformat()
        }
    
    def get_participant_performance(self, participant_id: str) -> Dict[str, Any]:
        """获取参与者表现摘要"""
        if participant_id not in self.participants:
            return {}
        
        participant = self.participants[participant_id]
        participant_results = [r for r in self.evaluation_results 
                              if r.participant_id == participant_id]
        
        if not participant_results:
            return {
                "participant_info": asdict(participant),
                "evaluations_completed": 0,
                "average_score": 0,
                "performance_trend": []
            }
        
        scores = [r.overall_score for r in participant_results]
        
        return {
            "participant_info": asdict(participant),
            "evaluations_completed": len(participant_results),
            "average_score": statistics.mean(scores),
            "best_score": max(scores),
            "latest_score": scores[-1],
            "performance_trend": scores,
            "strong_dimensions": self._identify_strong_dimensions(participant_results),
            "improvement_areas": self._identify_improvement_areas(participant_results)
        }
    
    def _identify_strong_dimensions(self, results: List[EvaluationResult]) -> List[str]:
        """识别强项维度"""
        dimension_scores = {}
        
        for result in results:
            dimension = self._get_task_dimension(result.task_id)
            if dimension:
                if dimension not in dimension_scores:
                    dimension_scores[dimension] = []
                dimension_scores[dimension].append(result.overall_score)
        
        strong_dimensions = []
        for dimension, scores in dimension_scores.items():
            if statistics.mean(scores) >= 0.8:
                strong_dimensions.append(dimension.value)
        
        return strong_dimensions
    
    def _identify_improvement_areas(self, results: List[EvaluationResult]) -> List[str]:
        """识别改进领域"""
        dimension_scores = {}
        
        for result in results:
            dimension = self._get_task_dimension(result.task_id)
            if dimension:
                if dimension not in dimension_scores:
                    dimension_scores[dimension] = []
                dimension_scores[dimension].append(result.overall_score)
        
        improvement_areas = []
        for dimension, scores in dimension_scores.items():
            if statistics.mean(scores) < 0.6:
                improvement_areas.append(dimension.value)
        
        return improvement_areas