"""
Product Definition Agent
Based on Section 3.2.2 of the AI-Product Manager paper
将市场分析和产品构思转化为详细的产品定义、用户体验设计和技术需求
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid

class DefinitionStatus(Enum):
    """定义状态"""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    VALIDATED = "validated"
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"

class ComponentType(Enum):
    """组件类型"""
    CORE_FEATURE = "core_feature"
    SUPPORTING_FEATURE = "supporting_feature"
    INTEGRATION = "integration"
    INFRASTRUCTURE = "infrastructure"
    UI_COMPONENT = "ui_component"

class PriorityLevel(Enum):
    """优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class TechnicalRequirement:
    """技术需求"""
    requirement_id: str
    name: str
    description: str
    category: str  # Performance, Security, Scalability, etc.
    priority: PriorityLevel
    acceptance_criteria: List[str]
    dependencies: List[str]
    estimated_effort: str  # person-days
    risk_level: str
    validation_method: str

@dataclass
class ProductFeature:
    """产品功能"""
    feature_id: str
    name: str
    description: str
    component_type: ComponentType
    priority: PriorityLevel
    user_stories: List[str]
    acceptance_criteria: List[str]
    technical_requirements: List[str]
    ui_specifications: Dict[str, Any]
    business_value: str
    estimated_effort: str
    dependencies: List[str]
    risks: List[str]

@dataclass
class UserPersona:
    """用户画像"""
    persona_id: str
    name: str
    demographics: Dict[str, Any]
    goals: List[str]
    pain_points: List[str]
    behaviors: List[str]
    technical_proficiency: str
    preferred_channels: List[str]
    motivations: List[str]

@dataclass
class UserJourney:
    """用户旅程"""
    journey_id: str
    persona_id: str
    scenario: str
    touchpoints: List[Dict[str, Any]]
    emotions: List[str]
    pain_points: List[str]
    opportunities: List[str]
    success_metrics: List[str]

@dataclass
class ValueProposition:
    """价值主张"""
    proposition_id: str
    target_segment: str
    core_value: str
    benefits: List[str]
    differentiators: List[str]
    proof_points: List[str]
    messaging: str
    success_metrics: List[str]

@dataclass
class ProductDefinition:
    """产品定义"""
    definition_id: str
    product_name: str
    version: str
    status: DefinitionStatus
    overview: str
    target_market: str
    value_propositions: List[ValueProposition]
    user_personas: List[UserPersona]
    user_journeys: List[UserJourney]
    core_features: List[ProductFeature]
    supporting_features: List[ProductFeature]
    technical_requirements: List[TechnicalRequirement]
    architecture_overview: Dict[str, Any]
    integration_requirements: List[str]
    security_requirements: List[str]
    performance_requirements: Dict[str, Any]
    scalability_requirements: Dict[str, Any]
    compliance_requirements: List[str]
    success_metrics: List[str]
    launch_criteria: List[str]
    created_at: datetime
    updated_at: datetime
    validation_notes: List[str]
    modification_log: List[Dict[str, Any]]

class ProductDefinitionAgent:
    """产品定义代理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 工作空间配置
        self.workspace_path = config.get('workspace_path', './product_definitions')
        self.validation_enabled = config.get('validation_enabled', True)
        
        # 定义标准
        self.definition_standards = {
            'min_core_features': 3,
            'min_user_personas': 2,
            'min_user_journeys': 2,
            'min_technical_requirements': 5,
            'min_acceptance_criteria_per_feature': 3
        }
        
        # 数据存储
        self.product_definitions: Dict[str, ProductDefinition] = {}
        self.validation_history: List[Dict[str, Any]] = []
        
    async def create_product_definition(self,
                                      market_analysis: Dict[str, Any],
                                      product_concept: Dict[str, Any],
                                      strategy_context: Dict[str, Any]) -> ProductDefinition:
        """
        创建产品定义
        基于市场分析和产品构思创建详细的产品定义
        """
        self.logger.info("开始创建产品定义")
        
        definition_id = str(uuid.uuid4())
        
        # 1. 创建基础产品定义框架
        base_definition = self._create_base_definition(
            definition_id, product_concept, market_analysis
        )
        
        # 2. 生成用户画像和旅程
        user_personas = await self._generate_user_personas(
            market_analysis, product_concept
        )
        user_journeys = await self._generate_user_journeys(
            user_personas, product_concept
        )
        
        # 3. 设计价值主张
        value_propositions = await self._design_value_propositions(
            product_concept, user_personas, market_analysis
        )
        
        # 4. 定义产品功能
        core_features, supporting_features = await self._define_product_features(
            product_concept, user_personas, user_journeys
        )
        
        # 5. 制定技术需求
        technical_requirements = await self._define_technical_requirements(
            core_features, supporting_features, strategy_context
        )
        
        # 6. 设计系统架构
        architecture_overview = await self._design_system_architecture(
            core_features, technical_requirements
        )
        
        # 7. 定义非功能需求
        performance_reqs = self._define_performance_requirements(product_concept)
        security_reqs = self._define_security_requirements(product_concept)
        scalability_reqs = self._define_scalability_requirements(product_concept)
        compliance_reqs = self._define_compliance_requirements(product_concept)
        
        # 8. 设定成功指标和发布标准
        success_metrics = self._define_success_metrics(value_propositions, user_journeys)
        launch_criteria = self._define_launch_criteria(core_features, technical_requirements)
        
        # 构建完整产品定义
        product_definition = ProductDefinition(
            definition_id=definition_id,
            product_name=base_definition["name"],
            version="1.0.0",
            status=DefinitionStatus.DRAFT,
            overview=base_definition["overview"],
            target_market=base_definition["target_market"],
            value_propositions=value_propositions,
            user_personas=user_personas,
            user_journeys=user_journeys,
            core_features=core_features,
            supporting_features=supporting_features,
            technical_requirements=technical_requirements,
            architecture_overview=architecture_overview,
            integration_requirements=self._define_integration_requirements(product_concept),
            security_requirements=security_reqs,
            performance_requirements=performance_reqs,
            scalability_requirements=scalability_reqs,
            compliance_requirements=compliance_reqs,
            success_metrics=success_metrics,
            launch_criteria=launch_criteria,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            validation_notes=[],
            modification_log=[]
        )
        
        # 初始验证
        if self.validation_enabled:
            validation_result = await self._validate_product_definition(product_definition)
            product_definition.validation_notes.extend(validation_result["notes"])
            if not validation_result["is_valid"]:
                product_definition.status = DefinitionStatus.NEEDS_REVISION
        
        # 保存到工作空间
        self.product_definitions[definition_id] = product_definition
        await self._save_to_workspace(product_definition)
        
        self.logger.info(f"产品定义创建完成: {product_definition.product_name}")
        return product_definition
    
    def _create_base_definition(self,
                              definition_id: str,
                              product_concept: Dict[str, Any],
                              market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建基础产品定义框架"""
        return {
            "name": product_concept.get("name", "新AI产品"),
            "overview": product_concept.get("description", "创新AI解决方案"),
            "target_market": market_analysis.get("target_segment", "企业用户")
        }
    
    async def _generate_user_personas(self,
                                    market_analysis: Dict[str, Any],
                                    product_concept: Dict[str, Any]) -> List[UserPersona]:
        """生成用户画像"""
        self.logger.info("生成用户画像")
        
        personas = []
        
        # 基于市场分析生成主要用户画像
        target_users = product_concept.get("target_users", ["企业用户", "个人用户"])
        
        persona_templates = {
            "企业用户": {
                "name": "企业决策者",
                "demographics": {
                    "age_range": "35-50",
                    "role": "中高级管理者",
                    "company_size": "100-1000人",
                    "industry": "科技、金融、制造"
                },
                "goals": [
                    "提高团队工作效率",
                    "降低运营成本",
                    "获得竞争优势",
                    "数字化转型"
                ],
                "pain_points": [
                    "现有工具效率低下",
                    "缺乏数据洞察",
                    "人力成本上升",
                    "技术集成复杂"
                ],
                "technical_proficiency": "中等"
            },
            "个人用户": {
                "name": "专业工作者",
                "demographics": {
                    "age_range": "25-40",
                    "role": "知识工作者",
                    "education": "本科以上",
                    "income": "中高收入"
                },
                "goals": [
                    "提升个人效率",
                    "学习新技能",
                    "获得职业发展",
                    "简化工作流程"
                ],
                "pain_points": [
                    "工作重复性高",
                    "时间管理困难",
                    "信息过载",
                    "工具使用复杂"
                ],
                "technical_proficiency": "中高"
            },
            "技术用户": {
                "name": "技术专家",
                "demographics": {
                    "age_range": "28-45",
                    "role": "开发者/架构师",
                    "experience": "5年以上",
                    "specialization": "AI/ML/软件开发"
                },
                "goals": [
                    "提高开发效率",
                    "实现技术创新",
                    "优化系统性能",
                    "降低维护成本"
                ],
                "pain_points": [
                    "技术集成复杂",
                    "性能优化困难",
                    "文档不完善",
                    "学习成本高"
                ],
                "technical_proficiency": "高"
            }
        }
        
        for i, user_type in enumerate(target_users[:3]):  # 最多3个主要用户画像
            template = persona_templates.get(user_type, persona_templates["个人用户"])
            
            persona = UserPersona(
                persona_id=f"persona_{i+1:03d}",
                name=template["name"],
                demographics=template["demographics"],
                goals=template["goals"],
                pain_points=template["pain_points"],
                behaviors=[
                    "定期使用专业工具",
                    "关注行业趋势",
                    "重视效率提升",
                    "愿意尝试新技术"
                ],
                technical_proficiency=template["technical_proficiency"],
                preferred_channels=["官网", "专业社区", "同行推荐", "线上演示"],
                motivations=[
                    "解决实际问题",
                    "提升工作效率",
                    "获得竞争优势",
                    "简化工作流程"
                ]
            )
            
            personas.append(persona)
        
        return personas
    
    async def _generate_user_journeys(self,
                                    user_personas: List[UserPersona],
                                    product_concept: Dict[str, Any]) -> List[UserJourney]:
        """生成用户旅程"""
        self.logger.info("生成用户旅程")
        
        journeys = []
        
        for persona in user_personas:
            # 为每个用户画像创建主要使用场景的旅程
            journey = UserJourney(
                journey_id=f"journey_{persona.persona_id}",
                persona_id=persona.persona_id,
                scenario=f"{persona.name}使用{product_concept.get('name', '产品')}解决日常工作问题",
                touchpoints=self._generate_touchpoints(persona, product_concept),
                emotions=["好奇", "期待", "满意", "信任", "推荐"],
                pain_points=[
                    "初次使用时的学习成本",
                    "功能发现的困难",
                    "结果理解的挑战"
                ],
                opportunities=[
                    "优化首次使用体验",
                    "增强功能引导",
                    "提供智能推荐",
                    "改进结果展示"
                ],
                success_metrics=[
                    "完成核心任务的时间",
                    "功能发现率",
                    "用户满意度",
                    "重复使用率"
                ]
            )
            
            journeys.append(journey)
        
        return journeys
    
    def _generate_touchpoints(self,
                            persona: UserPersona,
                            product_concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成接触点"""
        return [
            {
                "stage": "认知",
                "touchpoint": "官方网站/营销材料",
                "action": "了解产品价值",
                "emotion": "好奇",
                "pain_point": "信息过载"
            },
            {
                "stage": "试用",
                "touchpoint": "注册/试用流程",
                "action": "注册账户并开始试用",
                "emotion": "期待",
                "pain_point": "注册流程复杂"
            },
            {
                "stage": "首次使用",
                "touchpoint": "产品界面",
                "action": "完成首个核心任务",
                "emotion": "困惑到理解",
                "pain_point": "学习成本高"
            },
            {
                "stage": "深度使用",
                "touchpoint": "高级功能",
                "action": "探索并使用高级功能",
                "emotion": "满意",
                "pain_point": "功能发现困难"
            },
            {
                "stage": "持续使用",
                "touchpoint": "日常工作流",
                "action": "将产品集成到工作流程",
                "emotion": "信任",
                "pain_point": "集成复杂"
            },
            {
                "stage": "推荐",
                "touchpoint": "社区/同事",
                "action": "向他人推荐产品",
                "emotion": "倡导",
                "pain_point": "缺乏推荐激励"
            }
        ]
    
    async def _design_value_propositions(self,
                                       product_concept: Dict[str, Any],
                                       user_personas: List[UserPersona],
                                       market_analysis: Dict[str, Any]) -> List[ValueProposition]:
        """设计价值主张"""
        self.logger.info("设计价值主张")
        
        value_propositions = []
        
        for persona in user_personas:
            # 为每个用户画像设计针对性价值主张
            core_value = self._extract_core_value_for_persona(persona, product_concept)
            
            value_prop = ValueProposition(
                proposition_id=f"vp_{persona.persona_id}",
                target_segment=persona.name,
                core_value=core_value,
                benefits=self._identify_benefits_for_persona(persona, product_concept),
                differentiators=self._identify_differentiators(product_concept, market_analysis),
                proof_points=self._generate_proof_points(product_concept),
                messaging=self._create_messaging(core_value, persona),
                success_metrics=[
                    "用户采用率",
                    "功能使用深度",
                    "推荐意愿度",
                    "客户满意度"
                ]
            )
            
            value_propositions.append(value_prop)
        
        return value_propositions
    
    def _extract_core_value_for_persona(self,
                                      persona: UserPersona,
                                      product_concept: Dict[str, Any]) -> str:
        """为用户画像提取核心价值"""
        concept_value = product_concept.get("value_proposition", "提升效率")
        
        if "企业" in persona.name:
            return f"为企业{concept_value}，降低成本并获得竞争优势"
        elif "技术" in persona.name:
            return f"为技术专家{concept_value}，简化复杂任务"
        else:
            return f"为专业人士{concept_value}，提升工作质量"
    
    def _identify_benefits_for_persona(self,
                                     persona: UserPersona,
                                     product_concept: Dict[str, Any]) -> List[str]:
        """为用户画像识别收益"""
        key_features = product_concept.get("key_features", [])
        
        benefits = []
        for goal in persona.goals[:3]:  # 主要目标
            if "效率" in goal:
                benefits.append("显著提升工作效率")
            elif "成本" in goal:
                benefits.append("降低运营成本")
            elif "竞争" in goal:
                benefits.append("获得竞争优势")
            elif "简化" in goal:
                benefits.append("简化复杂流程")
        
        # 基于功能添加收益
        for feature in key_features:
            if "智能" in feature:
                benefits.append("智能化决策支持")
            elif "自动" in feature:
                benefits.append("自动化处理能力")
        
        return list(set(benefits))[:5]  # 去重并限制数量
    
    def _identify_differentiators(self,
                                product_concept: Dict[str, Any],
                                market_analysis: Dict[str, Any]) -> List[str]:
        """识别差异化因素"""
        return [
            "先进的AI技术栈",
            "用户友好的界面设计",
            "企业级安全和合规",
            "灵活的集成能力",
            "持续的AI模型优化"
        ]
    
    def _generate_proof_points(self, product_concept: Dict[str, Any]) -> List[str]:
        """生成证明点"""
        return [
            "基于最新AI研究成果",
            "经过大规模用户验证",
            "获得行业专家认可",
            "符合国际安全标准",
            "持续的技术创新投入"
        ]
    
    def _create_messaging(self, core_value: str, persona: UserPersona) -> str:
        """创建消息传递"""
        return f"专为{persona.name}设计，{core_value}，让工作更智能、更高效。"
    
    async def _define_product_features(self,
                                     product_concept: Dict[str, Any],
                                     user_personas: List[UserPersona],
                                     user_journeys: List[UserJourney]) -> Tuple[List[ProductFeature], List[ProductFeature]]:
        """定义产品功能"""
        self.logger.info("定义产品功能")
        
        core_features = []
        supporting_features = []
        
        # 基于产品概念的核心功能
        concept_features = product_concept.get("key_features", [])
        
        for i, feature_name in enumerate(concept_features):
            feature = ProductFeature(
                feature_id=f"core_feat_{i+1:03d}",
                name=feature_name,
                description=self._generate_feature_description(feature_name),
                component_type=ComponentType.CORE_FEATURE,
                priority=PriorityLevel.HIGH if i < 3 else PriorityLevel.MEDIUM,
                user_stories=self._generate_user_stories(feature_name, user_personas),
                acceptance_criteria=self._generate_acceptance_criteria(feature_name),
                technical_requirements=self._generate_feature_tech_requirements(feature_name),
                ui_specifications=self._generate_ui_specifications(feature_name),
                business_value=self._assess_business_value(feature_name, user_personas),
                estimated_effort=self._estimate_effort(feature_name),
                dependencies=self._identify_dependencies(feature_name, concept_features),
                risks=self._identify_feature_risks(feature_name)
            )
            
            core_features.append(feature)
        
        # 基于用户旅程的支持功能
        supporting_feature_templates = [
            "用户引导系统",
            "帮助与文档",
            "用户反馈收集",
            "性能监控",
            "数据导入导出"
        ]
        
        for i, feature_name in enumerate(supporting_feature_templates):
            feature = ProductFeature(
                feature_id=f"supp_feat_{i+1:03d}",
                name=feature_name,
                description=self._generate_feature_description(feature_name),
                component_type=ComponentType.SUPPORTING_FEATURE,
                priority=PriorityLevel.MEDIUM,
                user_stories=self._generate_user_stories(feature_name, user_personas),
                acceptance_criteria=self._generate_acceptance_criteria(feature_name),
                technical_requirements=self._generate_feature_tech_requirements(feature_name),
                ui_specifications=self._generate_ui_specifications(feature_name),
                business_value=self._assess_business_value(feature_name, user_personas),
                estimated_effort=self._estimate_effort(feature_name),
                dependencies=[],
                risks=self._identify_feature_risks(feature_name)
            )
            
            supporting_features.append(feature)
        
        return core_features, supporting_features
    
    def _generate_feature_description(self, feature_name: str) -> str:
        """生成功能描述"""
        descriptions = {
            "智能推荐": "基于用户行为和偏好的个性化内容推荐系统",
            "自动化处理": "自动执行重复性任务，减少人工干预",
            "数据分析": "深度数据挖掘和洞察分析能力",
            "用户引导系统": "帮助新用户快速上手的交互式引导",
            "帮助与文档": "完整的产品使用说明和帮助系统",
            "用户反馈收集": "收集和分析用户反馈的系统"
        }
        
        return descriptions.get(feature_name, f"{feature_name}的详细功能实现")
    
    def _generate_user_stories(self, feature_name: str, personas: List[UserPersona]) -> List[str]:
        """生成用户故事"""
        stories = []
        
        for persona in personas[:2]:  # 主要用户画像
            if "智能" in feature_name:
                stories.append(f"作为{persona.name}，我希望获得智能建议，以便提高决策质量")
            elif "自动" in feature_name:
                stories.append(f"作为{persona.name}，我希望自动处理重复任务，以便节省时间")
            elif "分析" in feature_name:
                stories.append(f"作为{persona.name}，我希望深入分析数据，以便发现洞察")
            else:
                stories.append(f"作为{persona.name}，我希望使用{feature_name}，以便{persona.goals[0]}")
        
        return stories
    
    def _generate_acceptance_criteria(self, feature_name: str) -> List[str]:
        """生成验收标准"""
        base_criteria = [
            "功能按预期工作",
            "性能满足要求",
            "用户界面友好",
            "错误处理正确"
        ]
        
        specific_criteria = {
            "智能推荐": [
                "推荐准确率 > 85%",
                "响应时间 < 200ms",
                "支持个性化设置"
            ],
            "自动化处理": [
                "自动化成功率 > 95%",
                "支持异常处理",
                "提供执行日志"
            ],
            "数据分析": [
                "支持多种数据格式",
                "分析结果可视化",
                "支持导出报告"
            ]
        }
        
        return base_criteria + specific_criteria.get(feature_name, [])
    
    def _generate_feature_tech_requirements(self, feature_name: str) -> List[str]:
        """生成功能技术需求"""
        return [
            f"{feature_name}_api_design",
            f"{feature_name}_data_model",
            f"{feature_name}_performance_req",
            f"{feature_name}_security_req"
        ]
    
    def _generate_ui_specifications(self, feature_name: str) -> Dict[str, Any]:
        """生成UI规范"""
        return {
            "layout": "responsive",
            "components": ["header", "main_content", "actions", "footer"],
            "interactions": ["click", "hover", "keyboard"],
            "accessibility": "WCAG 2.1 AA",
            "responsive_breakpoints": ["mobile", "tablet", "desktop"],
            "design_system": "modern_minimal"
        }
    
    def _assess_business_value(self, feature_name: str, personas: List[UserPersona]) -> str:
        """评估商业价值"""
        if "智能" in feature_name:
            return "高价值：提升用户决策质量，增加用户粘性"
        elif "自动" in feature_name:
            return "高价值：显著提升效率，降低操作成本"
        elif "分析" in feature_name:
            return "中高价值：提供数据洞察，支持业务优化"
        else:
            return "中等价值：改善用户体验，提升满意度"
    
    def _estimate_effort(self, feature_name: str) -> str:
        """估算工作量"""
        complexity_mapping = {
            "智能推荐": "20-30人天",
            "自动化处理": "15-25人天",
            "数据分析": "25-35人天",
            "用户引导系统": "10-15人天",
            "帮助与文档": "5-10人天"
        }
        
        return complexity_mapping.get(feature_name, "10-20人天")
    
    def _identify_dependencies(self, feature_name: str, all_features: List[str]) -> List[str]:
        """识别依赖关系"""
        dependencies = {
            "智能推荐": ["数据收集", "用户模型"],
            "自动化处理": ["工作流引擎", "任务调度"],
            "数据分析": ["数据存储", "计算引擎"]
        }
        
        return dependencies.get(feature_name, [])
    
    def _identify_feature_risks(self, feature_name: str) -> List[str]:
        """识别功能风险"""
        return [
            "技术实现复杂度",
            "性能优化挑战",
            "用户接受度不确定",
            "集成兼容性问题"
        ]
    
    async def _define_technical_requirements(self,
                                           core_features: List[ProductFeature],
                                           supporting_features: List[ProductFeature],
                                           strategy_context: Dict[str, Any]) -> List[TechnicalRequirement]:
        """定义技术需求"""
        self.logger.info("定义技术需求")
        
        requirements = []
        
        # 系统级技术需求
        system_requirements = [
            {
                "name": "高可用性要求",
                "description": "系统需要保持99.9%的可用性",
                "category": "Availability",
                "priority": PriorityLevel.CRITICAL,
                "criteria": ["99.9%正常运行时间", "故障恢复时间<5分钟", "自动故障转移"]
            },
            {
                "name": "响应性能要求",
                "description": "API响应时间和页面加载性能要求",
                "category": "Performance",
                "priority": PriorityLevel.HIGH,
                "criteria": ["API响应<200ms", "页面加载<3秒", "并发支持>1000用户"]
            },
            {
                "name": "数据安全要求",
                "description": "用户数据和系统安全保护",
                "category": "Security",
                "priority": PriorityLevel.CRITICAL,
                "criteria": ["数据加密传输", "访问权限控制", "审计日志记录"]
            },
            {
                "name": "可扩展性要求",
                "description": "系统水平扩展能力",
                "category": "Scalability",
                "priority": PriorityLevel.HIGH,
                "criteria": ["支持水平扩展", "负载均衡", "数据库分片"]
            },
            {
                "name": "数据备份要求",
                "description": "数据备份和恢复机制",
                "category": "Reliability",
                "priority": PriorityLevel.HIGH,
                "criteria": ["每日自动备份", "异地备份", "快速恢复能力"]
            }
        ]
        
        for i, req_data in enumerate(system_requirements):
            requirement = TechnicalRequirement(
                requirement_id=f"tech_req_{i+1:03d}",
                name=req_data["name"],
                description=req_data["description"],
                category=req_data["category"],
                priority=req_data["priority"],
                acceptance_criteria=req_data["criteria"],
                dependencies=self._identify_requirement_dependencies(req_data["name"]),
                estimated_effort=self._estimate_requirement_effort(req_data["category"]),
                risk_level=self._assess_requirement_risk(req_data["priority"]),
                validation_method=self._define_validation_method(req_data["category"])
            )
            
            requirements.append(requirement)
        
        return requirements
    
    def _identify_requirement_dependencies(self, req_name: str) -> List[str]:
        """识别需求依赖"""
        dependencies = {
            "高可用性要求": ["响应性能要求", "数据备份要求"],
            "响应性能要求": ["可扩展性要求"],
            "数据安全要求": ["数据备份要求"]
        }
        
        return dependencies.get(req_name, [])
    
    def _estimate_requirement_effort(self, category: str) -> str:
        """估算需求工作量"""
        effort_mapping = {
            "Performance": "15-25人天",
            "Security": "20-30人天",
            "Scalability": "25-35人天",
            "Availability": "20-30人天",
            "Reliability": "15-25人天"
        }
        
        return effort_mapping.get(category, "10-20人天")
    
    def _assess_requirement_risk(self, priority: PriorityLevel) -> str:
        """评估需求风险"""
        if priority == PriorityLevel.CRITICAL:
            return "高风险"
        elif priority == PriorityLevel.HIGH:
            return "中等风险"
        else:
            return "低风险"
    
    def _define_validation_method(self, category: str) -> str:
        """定义验证方法"""
        methods = {
            "Performance": "性能测试+监控",
            "Security": "安全审计+渗透测试",
            "Scalability": "负载测试+压力测试",
            "Availability": "可用性监控+故障演练",
            "Reliability": "备份恢复测试+数据完整性检查"
        }
        
        return methods.get(category, "功能测试+集成测试")
    
    async def _design_system_architecture(self,
                                        core_features: List[ProductFeature],
                                        technical_requirements: List[TechnicalRequirement]) -> Dict[str, Any]:
        """设计系统架构"""
        self.logger.info("设计系统架构")
        
        return {
            "architecture_pattern": "微服务架构",
            "components": {
                "api_gateway": {
                    "description": "API网关，统一入口",
                    "technologies": ["Kong", "Nginx"],
                    "responsibilities": ["路由", "认证", "限流", "监控"]
                },
                "user_service": {
                    "description": "用户管理服务",
                    "technologies": ["Node.js", "PostgreSQL"],
                    "responsibilities": ["用户认证", "权限管理", "用户画像"]
                },
                "ai_service": {
                    "description": "AI核心服务",
                    "technologies": ["Python", "TensorFlow", "Redis"],
                    "responsibilities": ["智能推荐", "数据分析", "模型推理"]
                },
                "automation_service": {
                    "description": "自动化处理服务",
                    "technologies": ["Java", "RabbitMQ", "MongoDB"],
                    "responsibilities": ["任务调度", "流程执行", "结果处理"]
                },
                "data_service": {
                    "description": "数据管理服务",
                    "technologies": ["Python", "Apache Spark", "ElasticSearch"],
                    "responsibilities": ["数据收集", "数据处理", "数据存储"]
                }
            },
            "data_flow": [
                "用户请求 -> API网关 -> 对应服务",
                "服务间通信 -> 消息队列 -> 异步处理",
                "数据收集 -> 数据服务 -> AI训练/推理",
                "结果返回 -> API网关 -> 用户界面"
            ],
            "deployment": {
                "platform": "Kubernetes",
                "cloud_provider": "AWS/Azure/GCP",
                "ci_cd": "GitLab CI/Jenkins",
                "monitoring": "Prometheus + Grafana",
                "logging": "ELK Stack"
            },
            "security": {
                "authentication": "OAuth 2.0 + JWT",
                "authorization": "RBAC",
                "data_encryption": "TLS 1.3 + AES-256",
                "vulnerability_scanning": "定期安全扫描"
            }
        }
    
    def _define_performance_requirements(self, product_concept: Dict[str, Any]) -> Dict[str, Any]:
        """定义性能需求"""
        return {
            "response_time": {
                "api_response": "< 200ms (95th percentile)",
                "page_load": "< 3 seconds",
                "ai_inference": "< 500ms"
            },
            "throughput": {
                "concurrent_users": "> 1000",
                "requests_per_second": "> 10000",
                "data_processing": "> 1GB/minute"
            },
            "resource_utilization": {
                "cpu_usage": "< 70% average",
                "memory_usage": "< 80% average",
                "disk_io": "< 80% capacity"
            }
        }
    
    def _define_security_requirements(self, product_concept: Dict[str, Any]) -> List[str]:
        """定义安全需求"""
        return [
            "数据传输加密 (TLS 1.3)",
            "静态数据加密 (AES-256)",
            "用户认证和授权 (OAuth 2.0)",
            "API访问控制和限流",
            "安全审计日志记录",
            "定期安全漏洞扫描",
            "数据隐私保护合规 (GDPR)",
            "安全事件响应机制"
        ]
    
    def _define_scalability_requirements(self, product_concept: Dict[str, Any]) -> Dict[str, Any]:
        """定义可扩展性需求"""
        return {
            "horizontal_scaling": {
                "auto_scaling": "基于CPU/内存使用率",
                "load_balancing": "支持多实例负载均衡",
                "service_discovery": "自动服务发现和注册"
            },
            "data_scaling": {
                "database_sharding": "支持数据库水平分片",
                "caching": "多层缓存策略",
                "cdn": "静态资源CDN分发"
            },
            "capacity_planning": {
                "growth_projection": "支持10x用户增长",
                "resource_monitoring": "实时资源使用监控",
                "capacity_alerts": "容量预警机制"
            }
        }
    
    def _define_compliance_requirements(self, product_concept: Dict[str, Any]) -> List[str]:
        """定义合规需求"""
        return [
            "GDPR数据保护合规",
            "SOC 2 Type II认证",
            "ISO 27001信息安全管理",
            "行业特定合规要求",
            "数据本地化存储要求",
            "审计追踪和报告",
            "隐私影响评估",
            "第三方安全评估"
        ]
    
    def _define_integration_requirements(self, product_concept: Dict[str, Any]) -> List[str]:
        """定义集成需求"""
        return [
            "RESTful API接口",
            "GraphQL查询支持",
            "Webhook事件通知",
            "SSO单点登录集成",
            "第三方身份提供商集成",
            "ERP/CRM系统集成",
            "数据导入/导出接口",
            "实时数据同步"
        ]
    
    def _define_success_metrics(self,
                              value_propositions: List[ValueProposition],
                              user_journeys: List[UserJourney]) -> List[str]:
        """定义成功指标"""
        return [
            "用户活跃度 (DAU/MAU)",
            "功能采用率",
            "用户满意度评分",
            "任务完成率",
            "响应时间达标率",
            "系统可用性",
            "用户留存率",
            "收入增长率",
            "客户获取成本",
            "净推荐值(NPS)"
        ]
    
    def _define_launch_criteria(self,
                              core_features: List[ProductFeature],
                              technical_requirements: List[TechnicalRequirement]) -> List[str]:
        """定义发布标准"""
        return [
            "所有核心功能完成开发和测试",
            "性能要求全部达标",
            "安全审计通过",
            "用户验收测试通过",
            "文档和培训材料完成",
            "监控和告警系统就绪",
            "备份和恢复机制验证",
            "负载测试通过",
            "合规性检查通过",
            "上线应急预案准备"
        ]
    
    async def _validate_product_definition(self, definition: ProductDefinition) -> Dict[str, Any]:
        """验证产品定义"""
        self.logger.info("验证产品定义")
        
        validation_result = {
            "is_valid": True,
            "notes": [],
            "warnings": [],
            "errors": []
        }
        
        # 完整性检查
        if len(definition.core_features) < self.definition_standards['min_core_features']:
            validation_result["errors"].append(
                f"核心功能数量不足，需要至少{self.definition_standards['min_core_features']}个"
            )
            validation_result["is_valid"] = False
        
        if len(definition.user_personas) < self.definition_standards['min_user_personas']:
            validation_result["errors"].append(
                f"用户画像数量不足，需要至少{self.definition_standards['min_user_personas']}个"
            )
            validation_result["is_valid"] = False
        
        if len(definition.technical_requirements) < self.definition_standards['min_technical_requirements']:
            validation_result["warnings"].append(
                f"技术需求可能不完整，建议至少{self.definition_standards['min_technical_requirements']}个"
            )
        
        # 一致性检查
        for feature in definition.core_features:
            if len(feature.acceptance_criteria) < self.definition_standards['min_acceptance_criteria_per_feature']:
                validation_result["warnings"].append(
                    f"功能 {feature.name} 的验收标准不够详细"
                )
        
        # 可行性检查
        high_priority_features = [f for f in definition.core_features 
                                if f.priority == PriorityLevel.CRITICAL]
        if len(high_priority_features) > 5:
            validation_result["warnings"].append(
                "关键优先级功能过多，可能影响发布计划"
            )
        
        if validation_result["is_valid"]:
            validation_result["notes"].append("产品定义验证通过")
        
        return validation_result
    
    async def _save_to_workspace(self, definition: ProductDefinition):
        """保存到工作空间"""
        try:
            import os
            os.makedirs(self.workspace_path, exist_ok=True)
            
            filename = f"{definition.definition_id}_{definition.product_name}.json"
            filepath = os.path.join(self.workspace_path, filename)
            
            # 转换为可序列化格式
            definition_dict = asdict(definition)
            definition_dict['created_at'] = definition.created_at.isoformat()
            definition_dict['updated_at'] = definition.updated_at.isoformat()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(definition_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"产品定义已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存产品定义失败: {e}")
    
    async def update_product_definition(self,
                                      definition_id: str,
                                      updates: Dict[str, Any],
                                      reason: str) -> ProductDefinition:
        """更新产品定义"""
        if definition_id not in self.product_definitions:
            raise ValueError(f"产品定义不存在: {definition_id}")
        
        definition = self.product_definitions[definition_id]
        
        # 记录修改日志
        modification_log = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "changes": updates
        }
        definition.modification_log.append(modification_log)
        
        # 应用更新
        for key, value in updates.items():
            if hasattr(definition, key):
                setattr(definition, key, value)
        
        definition.updated_at = datetime.now()
        definition.status = DefinitionStatus.IN_REVIEW
        
        # 重新验证
        if self.validation_enabled:
            validation_result = await self._validate_product_definition(definition)
            definition.validation_notes.extend(validation_result["notes"])
            if not validation_result["is_valid"]:
                definition.status = DefinitionStatus.NEEDS_REVISION
        
        # 保存更新
        await self._save_to_workspace(definition)
        
        self.logger.info(f"产品定义已更新: {definition.product_name}")
        return definition
    
    def get_definition_summary(self) -> Dict[str, Any]:
        """获取定义摘要"""
        return {
            "total_definitions": len(self.product_definitions),
            "status_distribution": {
                status.value: len([d for d in self.product_definitions.values() 
                                 if d.status == status])
                for status in DefinitionStatus
            },
            "average_features_per_product": np.mean([
                len(d.core_features) + len(d.supporting_features) 
                for d in self.product_definitions.values()
            ]) if self.product_definitions else 0,
            "validation_pass_rate": len([
                d for d in self.product_definitions.values() 
                if d.status not in [DefinitionStatus.NEEDS_REVISION]
            ]) / len(self.product_definitions) if self.product_definitions else 0,
            "workspace_path": self.workspace_path,
            "last_updated": datetime.now().isoformat()
        }