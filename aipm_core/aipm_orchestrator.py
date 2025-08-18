"""
AI Product Manager Framework Orchestrator
Main coordination system that integrates all AI-PM components
基于论文的完整AI产品管理器框架编排器
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

# Import all AI-PM components
from .market_intelligence_agent import MarketIntelligenceAgent, MarketSegment
from .user_insight_generator import UserInsightGenerator, ProductDirection
from .product_definition_agent import ProductDefinitionAgent, DefinitionStatus
from .product_strategy_advisor import ProductStrategyAdvisor, ValidationDimension
from .product_documentation_agent import ProductDocumentationAgent, DocumentType
from .value_bench import ValueBench, ParticipantType
from .evaluation_framework import AIProductManagerEvaluator, EvaluationMode

class WorkflowStage(Enum):
    """工作流阶段"""
    INITIALIZATION = "initialization"
    MARKET_INTELLIGENCE = "market_intelligence"
    USER_INSIGHT_GENERATION = "user_insight_generation"
    PRODUCT_DEFINITION = "product_definition"
    STRATEGY_FORMULATION = "strategy_formulation"
    DOCUMENTATION_GENERATION = "documentation_generation"
    EVALUATION_ASSESSMENT = "evaluation_assessment"
    COMPLETION = "completion"

class ExecutionMode(Enum):
    """执行模式"""
    FULL_PIPELINE = "full_pipeline"          # 完整流水线
    INCREMENTAL = "incremental"              # 增量执行
    INTERACTIVE = "interactive"              # 交互式
    BATCH_PROCESSING = "batch_processing"    # 批量处理
    EVALUATION_ONLY = "evaluation_only"      # 仅评估

class Priority(Enum):
    """优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class WorkflowConfiguration:
    """工作流配置"""
    config_id: str
    name: str
    description: str
    execution_mode: ExecutionMode
    enabled_stages: List[WorkflowStage]
    stage_timeouts: Dict[WorkflowStage, int]  # seconds
    retry_policies: Dict[WorkflowStage, int]  # max retries
    quality_gates: Dict[WorkflowStage, float]  # minimum quality scores
    notification_settings: Dict[str, Any]
    output_formats: List[str]

@dataclass
class StageResult:
    """阶段结果"""
    stage: WorkflowStage
    start_time: datetime
    end_time: datetime
    success: bool
    result_data: Dict[str, Any]
    quality_score: float
    execution_time: float
    error_message: Optional[str]
    warnings: List[str]
    metadata: Dict[str, Any]

@dataclass
class WorkflowExecution:
    """工作流执行记录"""
    execution_id: str
    workflow_config: WorkflowConfiguration
    start_time: datetime
    end_time: Optional[datetime]
    current_stage: WorkflowStage
    stage_results: List[StageResult]
    overall_success: bool
    total_execution_time: float
    quality_metrics: Dict[str, float]
    output_artifacts: Dict[str, Any]
    execution_summary: str

@dataclass
class AIProductManagerRequest:
    """AI产品管理器请求"""
    request_id: str
    user_id: str
    priority: Priority
    execution_mode: ExecutionMode
    input_data: Dict[str, Any]
    requirements: Dict[str, Any]
    constraints: Dict[str, Any]
    expected_outputs: List[str]
    deadline: Optional[datetime]
    callback_url: Optional[str]

@dataclass
class AIProductManagerResponse:
    """AI产品管理器响应"""
    request_id: str
    execution_id: str
    status: str
    results: Dict[str, Any]
    quality_assessment: Dict[str, float]
    execution_metadata: Dict[str, Any]
    recommendations: List[str]
    next_steps: List[str]
    generated_documents: List[str]

class AIProductManagerOrchestrator:
    """AI产品管理器编排器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化所有组件
        self._initialize_components(config)
        
        # 工作流管理
        self.workflow_configurations: Dict[str, WorkflowConfiguration] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        
        # 系统状态
        self.system_status = "ready"
        self.performance_metrics = {}
        self.last_health_check = datetime.now()
        
        # 初始化工作流配置
        self._initialize_workflow_configurations()
        
        # 启动后台任务
        self._start_background_tasks()
    
    def _initialize_components(self, config: Dict[str, Any]):
        """初始化所有组件"""
        self.logger.info("初始化AI产品管理器组件")
        
        # 初始化各个AI-PM组件
        self.market_intelligence = MarketIntelligenceAgent(config.get('market_intelligence', {}))
        self.user_insight_generator = UserInsightGenerator(config.get('user_insight', {}))
        self.product_definition_agent = ProductDefinitionAgent(config.get('product_definition', {}))
        self.strategy_advisor = ProductStrategyAdvisor(config.get('strategy_advisor', {}))
        self.documentation_agent = ProductDocumentationAgent(config.get('documentation', {}))
        self.value_bench = ValueBench(config.get('value_bench', {}))
        self.evaluator = AIProductManagerEvaluator(config.get('evaluation', {}))
        
        # 注册AI系统为Value-Bench参与者
        self.ai_participant_id = None
        asyncio.create_task(self._register_ai_participant())
        
        self.logger.info("组件初始化完成")
    
    async def _register_ai_participant(self):
        """注册AI系统为Value-Bench参与者"""
        try:
            self.ai_participant_id = await self.value_bench.register_participant(
                name="AI-Product-Manager",
                participant_type=ParticipantType.AI_SYSTEM,
                experience_level="expert",
                specializations=["product_management", "market_analysis", "strategy_formulation"],
                metadata={
                    "version": self.config.get('version', '1.0.0'),
                    "capabilities": ["market_intelligence", "user_insights", "product_definition", 
                                   "strategy_advisory", "documentation_generation"]
                }
            )
            self.logger.info(f"AI系统已注册为Value-Bench参与者: {self.ai_participant_id}")
        except Exception as e:
            self.logger.error(f"注册AI参与者失败: {e}")
    
    def _initialize_workflow_configurations(self):
        """初始化工作流配置"""
        # 标准完整流水线配置
        full_pipeline_config = WorkflowConfiguration(
            config_id="full_pipeline_001",
            name="完整AI产品管理流水线",
            description="执行完整的AI产品管理工作流，从市场分析到文档生成",
            execution_mode=ExecutionMode.FULL_PIPELINE,
            enabled_stages=[
                WorkflowStage.MARKET_INTELLIGENCE,
                WorkflowStage.USER_INSIGHT_GENERATION,
                WorkflowStage.PRODUCT_DEFINITION,
                WorkflowStage.STRATEGY_FORMULATION,
                WorkflowStage.DOCUMENTATION_GENERATION,
                WorkflowStage.EVALUATION_ASSESSMENT
            ],
            stage_timeouts={
                WorkflowStage.MARKET_INTELLIGENCE: 300,      # 5分钟
                WorkflowStage.USER_INSIGHT_GENERATION: 600,  # 10分钟
                WorkflowStage.PRODUCT_DEFINITION: 900,       # 15分钟
                WorkflowStage.STRATEGY_FORMULATION: 600,     # 10分钟
                WorkflowStage.DOCUMENTATION_GENERATION: 1200, # 20分钟
                WorkflowStage.EVALUATION_ASSESSMENT: 300     # 5分钟
            },
            retry_policies={
                WorkflowStage.MARKET_INTELLIGENCE: 2,
                WorkflowStage.USER_INSIGHT_GENERATION: 2,
                WorkflowStage.PRODUCT_DEFINITION: 1,
                WorkflowStage.STRATEGY_FORMULATION: 2,
                WorkflowStage.DOCUMENTATION_GENERATION: 1,
                WorkflowStage.EVALUATION_ASSESSMENT: 1
            },
            quality_gates={
                WorkflowStage.MARKET_INTELLIGENCE: 0.7,
                WorkflowStage.USER_INSIGHT_GENERATION: 0.7,
                WorkflowStage.PRODUCT_DEFINITION: 0.75,
                WorkflowStage.STRATEGY_FORMULATION: 0.8,
                WorkflowStage.DOCUMENTATION_GENERATION: 0.75,
                WorkflowStage.EVALUATION_ASSESSMENT: 0.7
            },
            notification_settings={
                "email_alerts": True,
                "progress_updates": True,
                "error_notifications": True
            },
            output_formats=["json", "pdf", "html"]
        )
        self.workflow_configurations["full_pipeline"] = full_pipeline_config
        
        # 快速分析配置
        quick_analysis_config = WorkflowConfiguration(
            config_id="quick_analysis_001",
            name="快速产品分析",
            description="快速的市场和用户洞察分析",
            execution_mode=ExecutionMode.INCREMENTAL,
            enabled_stages=[
                WorkflowStage.MARKET_INTELLIGENCE,
                WorkflowStage.USER_INSIGHT_GENERATION,
                WorkflowStage.EVALUATION_ASSESSMENT
            ],
            stage_timeouts={
                WorkflowStage.MARKET_INTELLIGENCE: 180,
                WorkflowStage.USER_INSIGHT_GENERATION: 300,
                WorkflowStage.EVALUATION_ASSESSMENT: 120
            },
            retry_policies={stage: 1 for stage in [WorkflowStage.MARKET_INTELLIGENCE, 
                                                 WorkflowStage.USER_INSIGHT_GENERATION, 
                                                 WorkflowStage.EVALUATION_ASSESSMENT]},
            quality_gates={stage: 0.6 for stage in [WorkflowStage.MARKET_INTELLIGENCE, 
                                                   WorkflowStage.USER_INSIGHT_GENERATION, 
                                                   WorkflowStage.EVALUATION_ASSESSMENT]},
            notification_settings={"progress_updates": True},
            output_formats=["json"]
        )
        self.workflow_configurations["quick_analysis"] = quick_analysis_config
        
        # 评估专用配置
        evaluation_only_config = WorkflowConfiguration(
            config_id="evaluation_only_001",
            name="产品策略评估",
            description="仅执行产品策略评估和文档生成",
            execution_mode=ExecutionMode.EVALUATION_ONLY,
            enabled_stages=[
                WorkflowStage.STRATEGY_FORMULATION,
                WorkflowStage.DOCUMENTATION_GENERATION,
                WorkflowStage.EVALUATION_ASSESSMENT
            ],
            stage_timeouts={
                WorkflowStage.STRATEGY_FORMULATION: 600,
                WorkflowStage.DOCUMENTATION_GENERATION: 900,
                WorkflowStage.EVALUATION_ASSESSMENT: 300
            },
            retry_policies={stage: 1 for stage in [WorkflowStage.STRATEGY_FORMULATION, 
                                                 WorkflowStage.DOCUMENTATION_GENERATION, 
                                                 WorkflowStage.EVALUATION_ASSESSMENT]},
            quality_gates={stage: 0.75 for stage in [WorkflowStage.STRATEGY_FORMULATION, 
                                                    WorkflowStage.DOCUMENTATION_GENERATION, 
                                                    WorkflowStage.EVALUATION_ASSESSMENT]},
            notification_settings={"email_alerts": True, "progress_updates": True},
            output_formats=["json", "pdf"]
        )
        self.workflow_configurations["evaluation_only"] = evaluation_only_config
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 启动性能监控
        asyncio.create_task(self._performance_monitoring_task())
        
        # 启动健康检查
        asyncio.create_task(self._health_check_task())
        
        # 启动执行清理任务
        asyncio.create_task(self._execution_cleanup_task())
    
    async def execute_workflow(self, 
                             request: AIProductManagerRequest,
                             workflow_config_name: str = "full_pipeline") -> AIProductManagerResponse:
        """执行AI产品管理工作流"""
        self.logger.info(f"开始执行工作流: {request.request_id}")
        
        if workflow_config_name not in self.workflow_configurations:
            raise ValueError(f"工作流配置不存在: {workflow_config_name}")
        
        workflow_config = self.workflow_configurations[workflow_config_name]
        execution_id = str(uuid.uuid4())
        
        # 创建工作流执行记录
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_config=workflow_config,
            start_time=datetime.now(),
            end_time=None,
            current_stage=WorkflowStage.INITIALIZATION,
            stage_results=[],
            overall_success=False,
            total_execution_time=0.0,
            quality_metrics={},
            output_artifacts={},
            execution_summary=""
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            # 执行工作流各阶段
            stage_data = {}  # 阶段间数据传递
            
            for stage in workflow_config.enabled_stages:
                execution.current_stage = stage
                
                stage_result = await self._execute_stage(
                    stage, request.input_data, stage_data, workflow_config
                )
                
                execution.stage_results.append(stage_result)
                
                # 检查质量门槛
                if not self._check_quality_gate(stage_result, workflow_config):
                    self.logger.warning(f"阶段 {stage.value} 未通过质量门槛")
                    if stage_result.quality_score < 0.5:  # 严重质量问题
                        raise Exception(f"阶段 {stage.value} 质量分数过低: {stage_result.quality_score}")
                
                # 更新阶段间数据
                stage_data[stage.value] = stage_result.result_data
                
                self.logger.info(f"阶段 {stage.value} 完成，质量分数: {stage_result.quality_score:.3f}")
            
            # 计算总体指标
            execution.end_time = datetime.now()
            execution.total_execution_time = (execution.end_time - execution.start_time).total_seconds()
            execution.overall_success = True
            execution.quality_metrics = self._calculate_workflow_quality_metrics(execution.stage_results)
            execution.output_artifacts = self._collect_output_artifacts(execution.stage_results)
            execution.execution_summary = self._generate_execution_summary(execution)
            
            # 生成响应
            response = AIProductManagerResponse(
                request_id=request.request_id,
                execution_id=execution_id,
                status="completed",
                results=execution.output_artifacts,
                quality_assessment=execution.quality_metrics,
                execution_metadata={
                    "total_time": execution.total_execution_time,
                    "stages_completed": len(execution.stage_results),
                    "workflow_config": workflow_config_name
                },
                recommendations=self._generate_recommendations(execution),
                next_steps=self._generate_next_steps(execution),
                generated_documents=self._list_generated_documents(execution)
            )
            
            self.logger.info(f"工作流执行成功: {execution_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"工作流执行失败: {e}")
            execution.overall_success = False
            execution.end_time = datetime.now()
            execution.execution_summary = f"执行失败: {str(e)}"
            
            # 生成错误响应
            response = AIProductManagerResponse(
                request_id=request.request_id,
                execution_id=execution_id,
                status="failed",
                results={},
                quality_assessment={},
                execution_metadata={"error": str(e)},
                recommendations=[f"修复错误: {str(e)}"],
                next_steps=["检查输入数据", "重新执行工作流"],
                generated_documents=[]
            )
            
            return response
            
        finally:
            # 移动到历史记录
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            self.execution_history.append(execution)
    
    async def _execute_stage(self, 
                           stage: WorkflowStage,
                           input_data: Dict[str, Any],
                           stage_data: Dict[str, Any],
                           config: WorkflowConfiguration) -> StageResult:
        """执行单个工作流阶段"""
        start_time = datetime.now()
        stage_result = StageResult(
            stage=stage,
            start_time=start_time,
            end_time=start_time,
            success=False,
            result_data={},
            quality_score=0.0,
            execution_time=0.0,
            error_message=None,
            warnings=[],
            metadata={}
        )
        
        try:
            # 设置超时
            timeout = config.stage_timeouts.get(stage, 600)
            
            if stage == WorkflowStage.MARKET_INTELLIGENCE:
                result = await asyncio.wait_for(
                    self._execute_market_intelligence_stage(input_data, stage_data),
                    timeout=timeout
                )
            elif stage == WorkflowStage.USER_INSIGHT_GENERATION:
                result = await asyncio.wait_for(
                    self._execute_user_insight_stage(input_data, stage_data),
                    timeout=timeout
                )
            elif stage == WorkflowStage.PRODUCT_DEFINITION:
                result = await asyncio.wait_for(
                    self._execute_product_definition_stage(input_data, stage_data),
                    timeout=timeout
                )
            elif stage == WorkflowStage.STRATEGY_FORMULATION:
                result = await asyncio.wait_for(
                    self._execute_strategy_formulation_stage(input_data, stage_data),
                    timeout=timeout
                )
            elif stage == WorkflowStage.DOCUMENTATION_GENERATION:
                result = await asyncio.wait_for(
                    self._execute_documentation_stage(input_data, stage_data),
                    timeout=timeout
                )
            elif stage == WorkflowStage.EVALUATION_ASSESSMENT:
                result = await asyncio.wait_for(
                    self._execute_evaluation_stage(input_data, stage_data),
                    timeout=timeout
                )
            else:
                raise ValueError(f"未知阶段: {stage}")
            
            stage_result.result_data = result
            stage_result.success = True
            stage_result.quality_score = await self._assess_stage_quality(stage, result)
            
        except asyncio.TimeoutError:
            stage_result.error_message = f"阶段 {stage.value} 执行超时"
            stage_result.quality_score = 0.0
            self.logger.error(stage_result.error_message)
        except Exception as e:
            stage_result.error_message = str(e)
            stage_result.quality_score = 0.0
            self.logger.error(f"阶段 {stage.value} 执行失败: {e}")
        finally:
            stage_result.end_time = datetime.now()
            stage_result.execution_time = (stage_result.end_time - stage_result.start_time).total_seconds()
        
        return stage_result
    
    async def _execute_market_intelligence_stage(self, 
                                               input_data: Dict[str, Any],
                                               stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行市场情报阶段"""
        self.logger.info("执行市场情报收集阶段")
        
        market_context = input_data.get('market_context', {})
        target_segment = input_data.get('target_segment')
        
        # 转换target_segment为MarketSegment枚举
        if isinstance(target_segment, str):
            try:
                target_segment = MarketSegment(target_segment.lower())
            except ValueError:
                target_segment = None
        
        # 选择竞品和成功案例
        competitors, success_cases = await self.market_intelligence.select_competitors_and_success_cases(
            market_context, target_segment
        )
        
        # 收集补充市场数据
        supplementary_data = await self.market_intelligence.gather_supplementary_market_data(
            competitors, success_cases
        )
        
        return {
            "competitors": [asdict(comp) for comp in competitors],
            "success_cases": [asdict(case) for case in success_cases],
            "supplementary_data": supplementary_data,
            "market_summary": self.market_intelligence.get_market_intelligence_summary()
        }
    
    async def _execute_user_insight_stage(self,
                                        input_data: Dict[str, Any],
                                        stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行用户洞察阶段"""
        self.logger.info("执行用户洞察生成阶段")
        
        market_context = input_data.get('market_context', {})
        user_data = input_data.get('user_data', {})
        
        # 如果有市场情报数据，合并到市场上下文
        if 'market_intelligence' in stage_data:
            market_intelligence_data = stage_data['market_intelligence']
            market_context.update({
                "competitor_analysis": market_intelligence_data.get('competitors', []),
                "market_trends": market_intelligence_data.get('supplementary_data', {}).get('market_trends', {}),
                "ai_adoption_growing": True,
                "ai_infrastructure_mature": True
            })
        
        # 生成综合用户洞察
        insights = await self.user_insight_generator.generate_comprehensive_insights(
            market_context, user_data
        )
        
        return {
            "comprehensive_insights": insights,
            "insight_summary": self.user_insight_generator.get_insight_summary()
        }
    
    async def _execute_product_definition_stage(self,
                                              input_data: Dict[str, Any],
                                              stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行产品定义阶段"""
        self.logger.info("执行产品定义阶段")
        
        # 准备产品定义输入数据
        market_analysis = {}
        product_concept = input_data.get('product_concept', {})
        strategy_context = input_data.get('strategy_context', {})
        
        # 合并前面阶段的数据
        if 'market_intelligence' in stage_data:
            market_analysis.update(stage_data['market_intelligence'])
        
        if 'user_insight_generation' in stage_data:
            user_insights = stage_data['user_insight_generation']['comprehensive_insights']
            if 'recommended_concept' in user_insights:
                product_concept.update(user_insights['recommended_concept'])
            market_analysis.update({
                "user_insights": user_insights,
                "target_segment": "企业用户"  # 从洞察中提取
            })
        
        # 创建产品定义
        product_definition = await self.product_definition_agent.create_product_definition(
            market_analysis, product_concept, strategy_context
        )
        
        return {
            "product_definition": asdict(product_definition),
            "definition_summary": self.product_definition_agent.get_definition_summary()
        }
    
    async def _execute_strategy_formulation_stage(self,
                                                input_data: Dict[str, Any],
                                                stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行策略制定阶段"""
        self.logger.info("执行策略制定阶段")
        
        # 准备策略评估输入数据
        product_definition_data = {}
        market_context = input_data.get('market_context', {})
        user_research_data = input_data.get('user_research_data', {})
        
        # 从产品定义阶段获取数据
        if 'product_definition' in stage_data:
            product_definition_data = stage_data['product_definition']['product_definition']
        
        # 从市场情报阶段获取市场上下文
        if 'market_intelligence' in stage_data:
            market_context.update(stage_data['market_intelligence']['supplementary_data'])
        
        # 从用户洞察阶段获取用户研究数据
        if 'user_insight_generation' in stage_data:
            user_research_data.update(stage_data['user_insight_generation']['comprehensive_insights'])
        
        # 执行产品策略评估
        strategy_evaluation = await self.strategy_advisor.evaluate_product_strategy(
            product_definition_data, market_context, user_research_data
        )
        
        return {
            "strategy_evaluation": asdict(strategy_evaluation),
            "advisor_summary": self.strategy_advisor.get_advisor_summary()
        }
    
    async def _execute_documentation_stage(self,
                                         input_data: Dict[str, Any],
                                         stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行文档生成阶段"""
        self.logger.info("执行文档生成阶段")
        
        # 准备文档生成输入数据
        market_analysis = input_data.get('market_context', {})
        product_definition = {}
        strategy_evaluation = {}
        
        # 从前面阶段收集数据
        if 'market_intelligence' in stage_data:
            market_analysis.update(stage_data['market_intelligence'])
        
        if 'user_insight_generation' in stage_data:
            market_analysis.update({
                "user_insights": stage_data['user_insight_generation']['comprehensive_insights']
            })
        
        if 'product_definition' in stage_data:
            product_definition = stage_data['product_definition']['product_definition']
        
        if 'strategy_formulation' in stage_data:
            strategy_evaluation = stage_data['strategy_formulation']['strategy_evaluation']
        
        # 生成综合文档
        documents = await self.documentation_agent.generate_comprehensive_documentation(
            market_analysis, product_definition, strategy_evaluation
        )
        
        # 转换文档为可序列化格式
        serializable_documents = {}
        for doc_id, document in documents.items():
            serializable_documents[doc_id] = {
                "metadata": asdict(document.metadata),
                "content": document.content,
                "quality_metrics": asdict(document.quality_metrics),
                "structure": asdict(document.structure)
            }
        
        return {
            "generated_documents": serializable_documents,
            "documentation_summary": self.documentation_agent.get_documentation_summary()
        }
    
    async def _execute_evaluation_stage(self,
                                       input_data: Dict[str, Any],
                                       stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行评估阶段"""
        self.logger.info("执行评估阶段")
        
        # 构建测试场景
        test_scenario = {
            "name": f"Workflow_Evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "market_context": input_data.get('market_context', {}),
            "user_data": input_data.get('user_data', {}),
            "product_concept": input_data.get('product_concept', {}),
            "expected_outcome": input_data.get('expected_outcome', {})
        }
        
        # 添加阶段间数据
        if 'market_intelligence' in stage_data:
            test_scenario["market_intelligence_result"] = stage_data['market_intelligence']
        
        if 'user_insight_generation' in stage_data:
            test_scenario["user_insights_result"] = stage_data['user_insight_generation']
        
        if 'product_definition' in stage_data:
            test_scenario["product_definition_result"] = stage_data['product_definition']
        
        if 'strategy_formulation' in stage_data:
            test_scenario["strategy_evaluation_result"] = stage_data['strategy_formulation']
        
        # 执行综合评估
        system_evaluation = await self.evaluator.conduct_comprehensive_evaluation(
            [test_scenario], "comprehensive"
        )
        
        return {
            "system_evaluation": asdict(system_evaluation),
            "evaluation_summary": self.evaluator.get_evaluation_summary()
        }
    
    async def _assess_stage_quality(self, stage: WorkflowStage, result: Dict[str, Any]) -> float:
        """评估阶段质量"""
        # 基于不同阶段的特定质量指标
        if stage == WorkflowStage.MARKET_INTELLIGENCE:
            competitors_count = len(result.get('competitors', []))
            data_completeness = len(result.get('supplementary_data', {}))
            return min(1.0, (competitors_count * 0.2 + data_completeness * 0.1))
        
        elif stage == WorkflowStage.USER_INSIGHT_GENERATION:
            insights = result.get('comprehensive_insights', {})
            concepts_count = len(insights.get('divergent_concepts', []))
            has_recommendation = 'recommended_concept' in insights
            return min(1.0, 0.5 + concepts_count * 0.1 + (0.2 if has_recommendation else 0))
        
        elif stage == WorkflowStage.PRODUCT_DEFINITION:
            definition = result.get('product_definition', {})
            core_features = len(definition.get('core_features', []))
            user_personas = len(definition.get('user_personas', []))
            return min(1.0, 0.4 + core_features * 0.1 + user_personas * 0.15)
        
        elif stage == WorkflowStage.STRATEGY_FORMULATION:
            evaluation = result.get('strategy_evaluation', {})
            overall_score = evaluation.get('overall_score', 0)
            return overall_score
        
        elif stage == WorkflowStage.DOCUMENTATION_GENERATION:
            documents = result.get('generated_documents', {})
            doc_count = len(documents)
            avg_quality = 0.75  # 模拟平均质量分数
            return min(1.0, 0.3 + doc_count * 0.1 + avg_quality * 0.4)
        
        elif stage == WorkflowStage.EVALUATION_ASSESSMENT:
            evaluation = result.get('system_evaluation', {})
            aggregate_metrics = evaluation.get('aggregate_metrics', {})
            overall_score = aggregate_metrics.get('overall_score', 0)
            return overall_score
        
        return 0.7  # 默认质量分数
    
    def _check_quality_gate(self, stage_result: StageResult, config: WorkflowConfiguration) -> bool:
        """检查质量门槛"""
        threshold = config.quality_gates.get(stage_result.stage, 0.6)
        return stage_result.quality_score >= threshold
    
    def _calculate_workflow_quality_metrics(self, stage_results: List[StageResult]) -> Dict[str, float]:
        """计算工作流质量指标"""
        if not stage_results:
            return {}
        
        quality_scores = [result.quality_score for result in stage_results]
        execution_times = [result.execution_time for result in stage_results]
        success_count = sum(1 for result in stage_results if result.success)
        
        return {
            "overall_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "quality_consistency": 1.0 - (max(quality_scores) - min(quality_scores)),
            "success_rate": success_count / len(stage_results),
            "average_stage_time": sum(execution_times) / len(execution_times),
            "total_stages": len(stage_results)
        }
    
    def _collect_output_artifacts(self, stage_results: List[StageResult]) -> Dict[str, Any]:
        """收集输出工件"""
        artifacts = {}
        
        for result in stage_results:
            if result.success:
                artifacts[result.stage.value] = result.result_data
        
        return artifacts
    
    def _generate_execution_summary(self, execution: WorkflowExecution) -> str:
        """生成执行摘要"""
        if execution.overall_success:
            quality_score = execution.quality_metrics.get('overall_quality', 0)
            return f"工作流执行成功完成，总计{len(execution.stage_results)}个阶段，" \
                   f"总体质量分数: {quality_score:.3f}，执行时间: {execution.total_execution_time:.1f}秒"
        else:
            failed_stages = [result.stage.value for result in execution.stage_results if not result.success]
            return f"工作流执行失败，失败阶段: {', '.join(failed_stages)}"
    
    def _generate_recommendations(self, execution: WorkflowExecution) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于质量分数的建议
        quality_score = execution.quality_metrics.get('overall_quality', 0)
        if quality_score < 0.7:
            recommendations.append("整体质量需要改进，建议优化各阶段算法")
        
        # 基于执行时间的建议
        if execution.total_execution_time > 1800:  # 30分钟
            recommendations.append("执行时间较长，建议优化性能或采用并行处理")
        
        # 基于阶段结果的建议
        for result in execution.stage_results:
            if result.quality_score < 0.6:
                recommendations.append(f"改进{result.stage.value}阶段的算法和数据质量")
            
            if result.execution_time > 600:  # 10分钟
                recommendations.append(f"优化{result.stage.value}阶段的执行效率")
        
        # 基于成功率的建议
        success_rate = execution.quality_metrics.get('success_rate', 0)
        if success_rate < 1.0:
            recommendations.append("增强错误处理和重试机制")
        
        return recommendations[:5]  # 限制建议数量
    
    def _generate_next_steps(self, execution: WorkflowExecution) -> List[str]:
        """生成后续步骤"""
        next_steps = []
        
        if execution.overall_success:
            next_steps.extend([
                "审查生成的产品文档",
                "与利益相关者分享分析结果",
                "制定具体的实施计划",
                "设置后续评估里程碑"
            ])
            
            # 基于质量分数的后续步骤
            quality_score = execution.quality_metrics.get('overall_quality', 0)
            if quality_score > 0.8:
                next_steps.append("考虑推进到实施阶段")
            else:
                next_steps.append("进一步优化分析结果")
        else:
            next_steps.extend([
                "分析失败原因",
                "修复识别的问题",
                "重新执行工作流",
                "考虑调整输入参数"
            ])
        
        return next_steps
    
    def _list_generated_documents(self, execution: WorkflowExecution) -> List[str]:
        """列出生成的文档"""
        documents = []
        
        # 从文档生成阶段提取文档列表
        for result in execution.stage_results:
            if result.stage == WorkflowStage.DOCUMENTATION_GENERATION and result.success:
                generated_docs = result.result_data.get('generated_documents', {})
                documents.extend(generated_docs.keys())
        
        return documents
    
    # 后台任务
    async def _performance_monitoring_task(self):
        """性能监控任务"""
        while True:
            try:
                # 更新性能指标
                self.performance_metrics = {
                    "active_executions": len(self.active_executions),
                    "total_executions": len(self.execution_history),
                    "average_execution_time": self._calculate_average_execution_time(),
                    "success_rate": self._calculate_success_rate(),
                    "system_uptime": (datetime.now() - self.last_health_check).total_seconds()
                }
                
                # 等待下次监控
                await asyncio.sleep(60)  # 1分钟间隔
                
            except Exception as e:
                self.logger.error(f"性能监控任务出错: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_task(self):
        """健康检查任务"""
        while True:
            try:
                # 执行健康检查
                health_status = await self._perform_health_check()
                
                if health_status["status"] != "healthy":
                    self.logger.warning(f"系统健康状况异常: {health_status}")
                
                self.last_health_check = datetime.now()
                
                # 等待下次检查
                await asyncio.sleep(300)  # 5分钟间隔
                
            except Exception as e:
                self.logger.error(f"健康检查任务出错: {e}")
                await asyncio.sleep(300)
    
    async def _execution_cleanup_task(self):
        """执行清理任务"""
        while True:
            try:
                # 清理超过24小时的历史记录
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.execution_history = [
                    exec for exec in self.execution_history 
                    if exec.start_time > cutoff_time
                ]
                
                # 清理僵尸执行
                current_time = datetime.now()
                zombie_executions = []
                
                for exec_id, execution in self.active_executions.items():
                    if (current_time - execution.start_time).total_seconds() > 7200:  # 2小时超时
                        zombie_executions.append(exec_id)
                
                for exec_id in zombie_executions:
                    self.logger.warning(f"清理僵尸执行: {exec_id}")
                    execution = self.active_executions.pop(exec_id)
                    execution.overall_success = False
                    execution.execution_summary = "执行超时被清理"
                    self.execution_history.append(execution)
                
                # 等待下次清理
                await asyncio.sleep(3600)  # 1小时间隔
                
            except Exception as e:
                self.logger.error(f"执行清理任务出错: {e}")
                await asyncio.sleep(3600)
    
    # 辅助方法
    def _calculate_average_execution_time(self) -> float:
        """计算平均执行时间"""
        if not self.execution_history:
            return 0.0
        
        completed_executions = [exec for exec in self.execution_history if exec.overall_success]
        if not completed_executions:
            return 0.0
        
        total_time = sum(exec.total_execution_time for exec in completed_executions)
        return total_time / len(completed_executions)
    
    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        if not self.execution_history:
            return 0.0
        
        successful_count = sum(1 for exec in self.execution_history if exec.overall_success)
        return successful_count / len(self.execution_history)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "issues": []
        }
        
        # 检查组件健康状况
        try:
            # 市场情报组件检查
            market_summary = self.market_intelligence.get_market_intelligence_summary()
            health_status["components"]["market_intelligence"] = "healthy" if market_summary else "warning"
            
            # 用户洞察组件检查
            insight_summary = self.user_insight_generator.get_insight_summary()
            health_status["components"]["user_insight_generator"] = "healthy" if insight_summary else "warning"
            
            # 产品定义组件检查
            definition_summary = self.product_definition_agent.get_definition_summary()
            health_status["components"]["product_definition"] = "healthy" if definition_summary else "warning"
            
            # 策略顾问组件检查
            advisor_summary = self.strategy_advisor.get_advisor_summary()
            health_status["components"]["strategy_advisor"] = "healthy" if advisor_summary else "warning"
            
            # 文档代理组件检查
            doc_summary = self.documentation_agent.get_documentation_summary()
            health_status["components"]["documentation_agent"] = "healthy" if doc_summary else "warning"
            
            # Value-Bench组件检查
            bench_summary = self.value_bench.get_benchmark_summary()
            health_status["components"]["value_bench"] = "healthy" if bench_summary else "warning"
            
            # 评估器组件检查
            eval_summary = self.evaluator.get_evaluation_summary()
            health_status["components"]["evaluator"] = "healthy" if eval_summary else "warning"
            
        except Exception as e:
            health_status["issues"].append(f"组件检查失败: {e}")
            health_status["status"] = "unhealthy"
        
        # 检查系统资源
        if len(self.active_executions) > 10:
            health_status["issues"].append("活跃执行数量过多")
            health_status["status"] = "warning"
        
        # 检查错误率
        if self.execution_history:
            recent_executions = [exec for exec in self.execution_history 
                               if (datetime.now() - exec.start_time).total_seconds() < 3600]
            if recent_executions:
                recent_success_rate = sum(1 for exec in recent_executions if exec.overall_success) / len(recent_executions)
                if recent_success_rate < 0.8:
                    health_status["issues"].append(f"最近成功率偏低: {recent_success_rate:.2%}")
                    health_status["status"] = "warning"
        
        return health_status
    
    # 公共接口方法
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "status": self.system_status,
            "performance_metrics": self.performance_metrics,
            "active_executions": len(self.active_executions),
            "total_executions": len(self.execution_history),
            "workflow_configurations": list(self.workflow_configurations.keys()),
            "components_status": {
                "market_intelligence": "healthy",
                "user_insight_generator": "healthy",
                "product_definition": "healthy",
                "strategy_advisor": "healthy",
                "documentation_agent": "healthy",
                "value_bench": "healthy",
                "evaluator": "healthy"
            },
            "last_health_check": self.last_health_check.isoformat(),
            "ai_participant_id": self.ai_participant_id
        }
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """获取执行状态"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            return {
                "execution_id": execution_id,
                "status": "running",
                "current_stage": execution.current_stage.value,
                "completed_stages": len(execution.stage_results),
                "start_time": execution.start_time.isoformat(),
                "elapsed_time": (datetime.now() - execution.start_time).total_seconds()
            }
        
        # 在历史记录中查找
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return {
                    "execution_id": execution_id,
                    "status": "completed" if execution.overall_success else "failed",
                    "completed_stages": len(execution.stage_results),
                    "start_time": execution.start_time.isoformat(),
                    "end_time": execution.end_time.isoformat() if execution.end_time else None,
                    "total_time": execution.total_execution_time,
                    "quality_metrics": execution.quality_metrics
                }
        
        return {"error": "执行记录不存在"}
    
    def list_workflow_configurations(self) -> List[Dict[str, Any]]:
        """列出工作流配置"""
        configs = []
        for config_name, config in self.workflow_configurations.items():
            configs.append({
                "name": config_name,
                "display_name": config.name,
                "description": config.description,
                "execution_mode": config.execution_mode.value,
                "enabled_stages": [stage.value for stage in config.enabled_stages],
                "estimated_duration": sum(config.stage_timeouts.values()),
                "output_formats": config.output_formats
            })
        return configs