"""
User Insight Generator
Based on Section 3.1.2 of the AI-Product Manager paper
实现发散-收敛产品构思框架，超越既定范式的产品创新
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import random

class InsightCategory(Enum):
    """洞察类别"""
    USER_PAIN_POINTS = "user_pain_points"
    MARKET_GAPS = "market_gaps"
    TECHNOLOGY_OPPORTUNITIES = "technology_opportunities"
    BEHAVIORAL_PATTERNS = "behavioral_patterns"
    EMERGING_NEEDS = "emerging_needs"

class ProductDirection(Enum):
    """产品方向"""
    AI_ENHANCEMENT = "ai_enhancement"
    AUTOMATION_FOCUS = "automation_focus"
    PERSONALIZATION = "personalization"
    COLLABORATION = "collaboration"
    ANALYTICS_INSIGHTS = "analytics_insights"

@dataclass
class UserPainPoint:
    """用户痛点"""
    pain_point_id: str
    description: str
    severity: float  # 0-1
    frequency: float  # 0-1
    user_segments: List[str]
    current_solutions: List[str]
    solution_limitations: List[str]
    opportunity_score: float
    timestamp: datetime

@dataclass
class ProductConcept:
    """产品概念"""
    concept_id: str
    name: str
    direction: ProductDirection
    description: str
    target_pain_points: List[str]
    key_features: List[str]
    user_journey: List[str]
    value_proposition: str
    target_users: List[str]
    market_potential_score: float
    user_value_score: float
    commercial_feasibility_score: float
    innovation_level: float
    overall_score: float
    timestamp: datetime

@dataclass
class MarketOpportunity:
    """市场机会"""
    opportunity_id: str
    title: str
    description: str
    market_size: float
    growth_rate: float
    competition_level: float
    entry_barriers: List[str]
    success_factors: List[str]
    time_to_market: int  # months
    confidence_score: float

@dataclass
class UserInsight:
    """用户洞察"""
    insight_id: str
    category: InsightCategory
    title: str
    description: str
    evidence: List[str]
    implications: List[str]
    actionable_recommendations: List[str]
    confidence_level: float
    business_impact: float
    timestamp: datetime

class UserInsightGenerator:
    """用户洞察生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 发散-收敛框架配置
        self.divergent_concepts_count = config.get('divergent_concepts_count', 5)
        self.convergent_threshold = config.get('convergent_threshold', 0.7)
        
        # 评估权重
        self.evaluation_weights = {
            'market_potential': 0.35,
            'user_value': 0.35, 
            'commercial_feasibility': 0.30
        }
        
        # 创新指标权重
        self.innovation_weights = {
            'paradigm_shift': 0.4,
            'technology_novelty': 0.3,
            'user_experience_innovation': 0.3
        }
        
        # 数据存储
        self.user_insights: List[UserInsight] = []
        self.pain_points: List[UserPainPoint] = []
        self.market_opportunities: List[MarketOpportunity] = []
        self.product_concepts: List[ProductConcept] = []
        
    async def generate_comprehensive_insights(self, 
                                           market_context: Dict[str, Any],
                                           user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成综合用户洞察
        实现Section 3.1.2的完整框架
        """
        self.logger.info("开始生成综合用户洞察")
        
        # 1. 用户痛点识别
        pain_points = await self._identify_user_pain_points(user_data, market_context)
        
        # 2. 市场空白识别
        market_gaps = await self._identify_market_gaps(market_context, pain_points)
        
        # 3. 发散式产品构思
        divergent_concepts = await self._divergent_product_ideation(
            pain_points, market_gaps, market_context
        )
        
        # 4. 收敛式评估与选择
        converged_concept = await self._convergent_evaluation(
            divergent_concepts, market_context
        )
        
        # 5. 深度洞察生成
        deep_insights = await self._generate_deep_insights(
            converged_concept, pain_points, market_gaps
        )
        
        comprehensive_insights = {
            "pain_points_analysis": [asdict(pp) for pp in pain_points],
            "market_opportunities": [asdict(mo) for mo in market_gaps],
            "divergent_concepts": [asdict(dc) for dc in divergent_concepts],
            "recommended_concept": asdict(converged_concept),
            "deep_insights": [asdict(di) for di in deep_insights],
            "innovation_assessment": self._assess_innovation_level(converged_concept),
            "implementation_roadmap": self._generate_implementation_roadmap(converged_concept),
            "risk_assessment": self._assess_concept_risks(converged_concept)
        }
        
        self.logger.info("综合用户洞察生成完成")
        return comprehensive_insights
    
    async def _identify_user_pain_points(self, 
                                       user_data: Dict[str, Any],
                                       market_context: Dict[str, Any]) -> List[UserPainPoint]:
        """识别用户痛点"""
        self.logger.info("识别用户痛点")
        
        pain_points = []
        
        # 从用户反馈中提取痛点
        user_feedback = user_data.get('feedback_data', [])
        satisfaction_scores = user_data.get('satisfaction_scores', {})
        usage_patterns = user_data.get('usage_patterns', {})
        
        # 痛点模板（基于常见AI产品场景）
        pain_point_templates = [
            {
                "description": "AI推荐结果不够精准，经常推荐不相关内容",
                "severity": 0.8,
                "user_segments": ["内容消费者", "电商用户"],
                "current_solutions": ["手动筛选", "多平台对比"],
                "limitations": ["耗时耗力", "效果不佳", "体验差"]
            },
            {
                "description": "AI工具学习成本高，普通用户难以上手",
                "severity": 0.7,
                "user_segments": ["中小企业用户", "非技术用户"],
                "current_solutions": ["培训课程", "技术支持"],
                "limitations": ["成本高", "时间长", "效果不稳定"]
            },
            {
                "description": "AI输出结果缺乏可解释性，用户不信任",
                "severity": 0.75,
                "user_segments": ["企业决策者", "专业用户"],
                "current_solutions": ["人工验证", "多模型对比"],
                "limitations": ["效率低", "成本高", "主观性强"]
            },
            {
                "description": "AI服务响应速度慢，影响用户体验",
                "severity": 0.6,
                "user_segments": ["实时应用用户", "移动端用户"],
                "current_solutions": ["等待", "缓存机制"],
                "limitations": ["用户流失", "体验差", "竞争劣势"]
            },
            {
                "description": "AI产品功能单一，无法满足复杂业务需求",
                "severity": 0.65,
                "user_segments": ["企业客户", "专业服务机构"],
                "current_solutions": ["多工具组合", "定制开发"],
                "limitations": ["集成复杂", "成本高", "维护困难"]
            }
        ]
        
        for i, template in enumerate(pain_point_templates):
            # 根据实际数据调整痛点严重程度
            adjusted_severity = template["severity"]
            if satisfaction_scores:
                avg_satisfaction = np.mean(list(satisfaction_scores.values()))
                if avg_satisfaction < 3.0:  # 低满意度
                    adjusted_severity = min(1.0, adjusted_severity * 1.2)
            
            # 计算频率
            frequency = np.random.uniform(0.4, 0.8)
            
            # 计算机会分数
            opportunity_score = (adjusted_severity * 0.6 + frequency * 0.4)
            
            pain_point = UserPainPoint(
                pain_point_id=f"pain_{i+1:03d}",
                description=template["description"],
                severity=adjusted_severity,
                frequency=frequency,
                user_segments=template["user_segments"],
                current_solutions=template["current_solutions"],
                solution_limitations=template["limitations"],
                opportunity_score=opportunity_score,
                timestamp=datetime.now()
            )
            
            pain_points.append(pain_point)
        
        # 按机会分数排序
        pain_points.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        self.pain_points = pain_points
        return pain_points[:3]  # 返回前3个最重要的痛点
    
    async def _identify_market_gaps(self, 
                                  market_context: Dict[str, Any],
                                  pain_points: List[UserPainPoint]) -> List[MarketOpportunity]:
        """识别市场空白"""
        self.logger.info("识别市场空白")
        
        market_gaps = []
        
        # 基于痛点识别市场机会
        opportunity_templates = [
            {
                "title": "智能个性化推荐引擎",
                "description": "基于多模态数据的精准个性化推荐系统，解决推荐不准确问题",
                "market_size": 15000000000,  # $15B
                "growth_rate": 0.35,
                "competition_level": 0.7,
                "time_to_market": 18
            },
            {
                "title": "零代码AI应用平台",
                "description": "让非技术用户轻松构建和部署AI应用的平台",
                "market_size": 8000000000,   # $8B
                "growth_rate": 0.45,
                "competition_level": 0.5,
                "time_to_market": 24
            },
            {
                "title": "可解释AI决策系统",
                "description": "提供透明、可解释的AI决策支持系统",
                "market_size": 12000000000,  # $12B
                "growth_rate": 0.28,
                "competition_level": 0.4,
                "time_to_market": 30
            },
            {
                "title": "边缘AI加速平台",
                "description": "优化AI模型在边缘设备上的运行性能",
                "market_size": 6000000000,   # $6B
                "growth_rate": 0.52,
                "competition_level": 0.6,
                "time_to_market": 21
            },
            {
                "title": "企业AI集成中台",
                "description": "统一管理和协调企业内多个AI服务的中台系统",
                "market_size": 18000000000,  # $18B
                "growth_rate": 0.31,
                "competition_level": 0.3,
                "time_to_market": 36
            }
        ]
        
        for i, template in enumerate(opportunity_templates):
            # 计算入市门槛
            entry_barriers = self._calculate_entry_barriers(template)
            
            # 计算成功因素
            success_factors = self._identify_success_factors(template)
            
            # 计算信心分数
            confidence_score = self._calculate_opportunity_confidence(template, market_context)
            
            opportunity = MarketOpportunity(
                opportunity_id=f"opp_{i+1:03d}",
                title=template["title"],
                description=template["description"],
                market_size=template["market_size"],
                growth_rate=template["growth_rate"],
                competition_level=template["competition_level"],
                entry_barriers=entry_barriers,
                success_factors=success_factors,
                time_to_market=template["time_to_market"],
                confidence_score=confidence_score
            )
            
            market_gaps.append(opportunity)
        
        self.market_opportunities = market_gaps
        return market_gaps
    
    def _calculate_entry_barriers(self, opportunity_template: Dict[str, Any]) -> List[str]:
        """计算入市门槛"""
        barriers = []
        
        if opportunity_template["market_size"] > 10000000000:  # 大市场
            barriers.extend(["资金需求高", "技术门槛高"])
        
        if opportunity_template["competition_level"] > 0.6:  # 竞争激烈
            barriers.extend(["市场竞争激烈", "客户获取成本高"])
        
        if opportunity_template["time_to_market"] > 24:  # 开发周期长
            barriers.append("开发周期长")
        
        # 通用门槛
        barriers.extend(["人才获取", "监管合规"])
        
        return barriers[:4]  # 最多返回4个门槛
    
    def _identify_success_factors(self, opportunity_template: Dict[str, Any]) -> List[str]:
        """识别成功因素"""
        factors = ["技术创新", "用户体验", "市场营销", "团队执行力"]
        
        if "智能" in opportunity_template["title"]:
            factors.append("算法优势")
        
        if "平台" in opportunity_template["title"]:
            factors.append("生态建设")
        
        if "企业" in opportunity_template["title"]:
            factors.append("销售网络")
        
        return factors[:5]
    
    def _calculate_opportunity_confidence(self, 
                                        opportunity_template: Dict[str, Any],
                                        market_context: Dict[str, Any]) -> float:
        """计算机会信心分数"""
        base_confidence = 0.7
        
        # 市场增长率影响
        if opportunity_template["growth_rate"] > 0.4:
            base_confidence += 0.1
        elif opportunity_template["growth_rate"] < 0.2:
            base_confidence -= 0.1
        
        # 竞争水平影响
        if opportunity_template["competition_level"] < 0.4:
            base_confidence += 0.15
        elif opportunity_template["competition_level"] > 0.7:
            base_confidence -= 0.1
        
        # 市场趋势影响
        if market_context.get("ai_adoption_growing", True):
            base_confidence += 0.05
        
        return min(0.95, max(0.3, base_confidence))
    
    async def _divergent_product_ideation(self, 
                                        pain_points: List[UserPainPoint],
                                        market_gaps: List[MarketOpportunity],
                                        market_context: Dict[str, Any]) -> List[ProductConcept]:
        """
        发散式产品构思
        生成5个概念上不同的产品方向
        """
        self.logger.info("开始发散式产品构思")
        
        concepts = []
        
        # 5个不同的产品方向
        directions = [
            ProductDirection.AI_ENHANCEMENT,
            ProductDirection.AUTOMATION_FOCUS,
            ProductDirection.PERSONALIZATION,
            ProductDirection.COLLABORATION,
            ProductDirection.ANALYTICS_INSIGHTS
        ]
        
        for i, direction in enumerate(directions):
            concept = await self._generate_product_concept(
                f"concept_{i+1:03d}",
                direction,
                pain_points,
                market_gaps,
                market_context
            )
            concepts.append(concept)
        
        self.product_concepts = concepts
        return concepts
    
    async def _generate_product_concept(self,
                                      concept_id: str,
                                      direction: ProductDirection,
                                      pain_points: List[UserPainPoint],
                                      market_gaps: List[MarketOpportunity],
                                      market_context: Dict[str, Any]) -> ProductConcept:
        """生成单个产品概念"""
        
        # 根据方向生成概念
        if direction == ProductDirection.AI_ENHANCEMENT:
            concept_data = {
                "name": "智能增强助手",
                "description": "通过AI技术增强用户现有工具和工作流程的智能助手",
                "key_features": ["智能提示", "自动优化", "预测分析", "个性化建议"],
                "value_proposition": "让现有工具更智能，提升用户工作效率200%"
            }
        elif direction == ProductDirection.AUTOMATION_FOCUS:
            concept_data = {
                "name": "全流程自动化平台",
                "description": "端到端的业务流程自动化平台，减少人工干预",
                "key_features": ["流程自动化", "智能决策", "异常处理", "性能监控"],
                "value_proposition": "自动化复杂业务流程，降低操作成本60%"
            }
        elif direction == ProductDirection.PERSONALIZATION:
            concept_data = {
                "name": "超个性化体验引擎",
                "description": "基于深度学习的个性化用户体验定制引擎",
                "key_features": ["行为分析", "偏好学习", "动态适配", "实时优化"],
                "value_proposition": "为每个用户提供独一无二的个性化体验"
            }
        elif direction == ProductDirection.COLLABORATION:
            concept_data = {
                "name": "AI驱动协作平台",
                "description": "智能化团队协作平台，优化团队沟通和项目管理",
                "key_features": ["智能调度", "协作优化", "知识共享", "冲突解决"],
                "value_proposition": "用AI重新定义团队协作，提升团队效率50%"
            }
        else:  # ANALYTICS_INSIGHTS
            concept_data = {
                "name": "深度洞察分析平台",
                "description": "基于AI的商业数据深度分析和洞察平台",
                "key_features": ["数据挖掘", "趋势预测", "异常检测", "决策支持"],
                "value_proposition": "从数据中发现隐藏价值，驱动数据决策"
            }
        
        # 评估分数
        market_potential = self._evaluate_market_potential(concept_data, market_gaps)
        user_value = self._evaluate_user_value(concept_data, pain_points)
        commercial_feasibility = self._evaluate_commercial_feasibility(concept_data, market_context)
        innovation_level = self._evaluate_innovation_level(concept_data, direction)
        
        # 计算总分
        overall_score = (
            market_potential * self.evaluation_weights['market_potential'] +
            user_value * self.evaluation_weights['user_value'] +
            commercial_feasibility * self.evaluation_weights['commercial_feasibility']
        )
        
        concept = ProductConcept(
            concept_id=concept_id,
            name=concept_data["name"],
            direction=direction,
            description=concept_data["description"],
            target_pain_points=[pp.pain_point_id for pp in pain_points[:2]],
            key_features=concept_data["key_features"],
            user_journey=self._generate_user_journey(concept_data),
            value_proposition=concept_data["value_proposition"],
            target_users=self._identify_target_users(direction, pain_points),
            market_potential_score=market_potential,
            user_value_score=user_value,
            commercial_feasibility_score=commercial_feasibility,
            innovation_level=innovation_level,
            overall_score=overall_score,
            timestamp=datetime.now()
        )
        
        return concept
    
    def _evaluate_market_potential(self, concept_data: Dict[str, Any], 
                                 market_gaps: List[MarketOpportunity]) -> float:
        """评估市场潜力"""
        base_score = 0.7
        
        # 根据相关市场机会调整
        relevant_gaps = [gap for gap in market_gaps 
                        if any(keyword in gap.title.lower() 
                              for keyword in concept_data["name"].lower().split())]
        
        if relevant_gaps:
            avg_market_size = np.mean([gap.market_size for gap in relevant_gaps])
            avg_growth_rate = np.mean([gap.growth_rate for gap in relevant_gaps])
            
            # 市场规模影响
            if avg_market_size > 10000000000:  # >$10B
                base_score += 0.2
            elif avg_market_size > 5000000000:   # >$5B
                base_score += 0.1
            
            # 增长率影响
            if avg_growth_rate > 0.4:
                base_score += 0.1
            elif avg_growth_rate > 0.3:
                base_score += 0.05
        
        return min(1.0, base_score)
    
    def _evaluate_user_value(self, concept_data: Dict[str, Any],
                           pain_points: List[UserPainPoint]) -> float:
        """评估用户价值"""
        base_score = 0.6
        
        # 根据解决的痛点严重程度调整
        relevant_pain_points = [pp for pp in pain_points 
                               if any(keyword in pp.description.lower() 
                                     for keyword in concept_data["description"].lower().split())]
        
        if relevant_pain_points:
            avg_severity = np.mean([pp.severity for pp in relevant_pain_points])
            avg_frequency = np.mean([pp.frequency for pp in relevant_pain_points])
            
            base_score += avg_severity * 0.25 + avg_frequency * 0.15
        
        # 价值主张强度
        value_prop = concept_data["value_proposition"].lower()
        if "200%" in value_prop or "60%" in value_prop or "50%" in value_prop:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _evaluate_commercial_feasibility(self, concept_data: Dict[str, Any],
                                       market_context: Dict[str, Any]) -> float:
        """评估商业可行性"""
        base_score = 0.65
        
        # 技术复杂度
        complex_features = ["深度学习", "预测分析", "智能决策", "自动化"]
        complexity_count = sum(1 for feature in concept_data["key_features"] 
                              if any(complex_term in feature for complex_term in complex_features))
        
        if complexity_count <= 2:
            base_score += 0.1  # 技术复杂度适中
        elif complexity_count >= 4:
            base_score -= 0.1  # 技术复杂度高
        
        # 市场成熟度
        if market_context.get("ai_infrastructure_mature", True):
            base_score += 0.15
        
        # 商业模式清晰度
        if "平台" in concept_data["name"]:
            base_score += 0.05  # 平台模式相对成熟
        
        return min(1.0, base_score)
    
    def _evaluate_innovation_level(self, concept_data: Dict[str, Any],
                                 direction: ProductDirection) -> float:
        """评估创新水平"""
        base_score = 0.7
        
        # 范式转换程度
        paradigm_shift_indicators = ["重新定义", "革命性", "颠覆", "突破"]
        if any(indicator in concept_data["value_proposition"] 
               for indicator in paradigm_shift_indicators):
            base_score += 0.15
        
        # 技术新颖性
        if direction in [ProductDirection.AI_ENHANCEMENT, ProductDirection.PERSONALIZATION]:
            base_score += 0.1  # 相对新颖的方向
        
        # 用户体验创新
        ux_innovation_indicators = ["个性化", "智能", "自动", "优化"]
        innovation_count = sum(1 for feature in concept_data["key_features"]
                              if any(indicator in feature for indicator in ux_innovation_indicators))
        
        if innovation_count >= 3:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _generate_user_journey(self, concept_data: Dict[str, Any]) -> List[str]:
        """生成用户旅程"""
        journey_templates = {
            "智能增强助手": [
                "用户安装助手插件",
                "助手分析用户工作模式",
                "提供智能建议和优化",
                "用户体验效率提升",
                "助手持续学习优化"
            ],
            "全流程自动化平台": [
                "用户配置业务流程",
                "系统自动执行任务",
                "智能处理异常情况",
                "生成执行报告",
                "持续优化流程"
            ],
            "超个性化体验引擎": [
                "收集用户行为数据",
                "分析用户偏好模式",
                "生成个性化内容",
                "用户交互反馈",
                "动态调整个性化策略"
            ],
            "AI驱动协作平台": [
                "团队成员加入平台",
                "AI分析协作模式",
                "智能安排任务分配",
                "实时协作优化",
                "生成团队效率报告"
            ],
            "深度洞察分析平台": [
                "导入业务数据",
                "AI自动分析模式",
                "生成洞察报告",
                "提供决策建议",
                "跟踪决策效果"
            ]
        }
        
        name = concept_data["name"]
        for template_name in journey_templates:
            if template_name in name:
                return journey_templates[template_name]
        
        # 默认用户旅程
        return [
            "用户注册和配置",
            "体验核心功能",
            "获得价值反馈",
            "深度使用产品",
            "推荐给他人"
        ]
    
    def _identify_target_users(self, direction: ProductDirection,
                             pain_points: List[UserPainPoint]) -> List[str]:
        """识别目标用户"""
        direction_users = {
            ProductDirection.AI_ENHANCEMENT: ["知识工作者", "专业人士", "创意工作者"],
            ProductDirection.AUTOMATION_FOCUS: ["企业管理者", "运营人员", "IT管理员"],
            ProductDirection.PERSONALIZATION: ["消费者", "内容创作者", "营销人员"],
            ProductDirection.COLLABORATION: ["团队领导", "项目经理", "远程工作者"],
            ProductDirection.ANALYTICS_INSIGHTS: ["数据分析师", "商业决策者", "产品经理"]
        }
        
        base_users = direction_users.get(direction, ["通用用户"])
        
        # 从痛点中提取额外用户群体
        for pain_point in pain_points:
            base_users.extend(pain_point.user_segments)
        
        # 去重并返回前5个
        unique_users = list(set(base_users))
        return unique_users[:5]
    
    async def _convergent_evaluation(self, 
                                   concepts: List[ProductConcept],
                                   market_context: Dict[str, Any]) -> ProductConcept:
        """
        收敛式评估与选择
        选择最有前景的概念进行全面发展
        """
        self.logger.info("开始收敛式评估")
        
        # 多维度评估
        evaluation_results = []
        
        for concept in concepts:
            # 重新评估，考虑更多因素
            enhanced_score = self._enhanced_concept_evaluation(concept, market_context)
            evaluation_results.append((concept, enhanced_score))
        
        # 排序选择最佳概念
        evaluation_results.sort(key=lambda x: x[1], reverse=True)
        best_concept = evaluation_results[0][0]
        
        # 更新最佳概念的分数
        best_concept.overall_score = evaluation_results[0][1]
        
        self.logger.info(f"选择概念: {best_concept.name}, 得分: {best_concept.overall_score:.3f}")
        
        return best_concept
    
    def _enhanced_concept_evaluation(self, concept: ProductConcept,
                                   market_context: Dict[str, Any]) -> float:
        """增强概念评估"""
        base_score = concept.overall_score
        
        # 创新性加分
        if concept.innovation_level > 0.8:
            base_score *= 1.1
        
        # 市场时机
        if market_context.get("ai_adoption_accelerating", True):
            if concept.direction in [ProductDirection.AI_ENHANCEMENT, ProductDirection.AUTOMATION_FOCUS]:
                base_score *= 1.05
        
        # 实现复杂度
        complex_features = len([f for f in concept.key_features 
                               if any(term in f.lower() 
                                     for term in ["智能", "自动", "深度", "预测"])])
        if complex_features <= 2:
            base_score *= 1.02  # 实现相对简单
        elif complex_features >= 4:
            base_score *= 0.98  # 实现复杂
        
        # 竞争优势
        if concept.innovation_level > 0.75 and concept.commercial_feasibility_score > 0.7:
            base_score *= 1.08  # 既创新又可行
        
        return min(1.0, base_score)
    
    async def _generate_deep_insights(self,
                                    best_concept: ProductConcept,
                                    pain_points: List[UserPainPoint],
                                    market_gaps: List[MarketOpportunity]) -> List[UserInsight]:
        """生成深度洞察"""
        self.logger.info("生成深度洞察")
        
        insights = []
        
        # 用户痛点洞察
        pain_insight = UserInsight(
            insight_id="insight_pain_001",
            category=InsightCategory.USER_PAIN_POINTS,
            title="用户核心痛点：AI工具易用性差距",
            description="用户在使用AI工具时面临学习成本高、结果不可预测等核心问题",
            evidence=[pp.description for pp in pain_points[:2]],
            implications=[
                "需要降低AI工具使用门槛",
                "提升AI输出的可理解性",
                "增强用户对AI的信任"
            ],
            actionable_recommendations=[
                "设计直观的用户界面",
                "提供智能引导和提示",
                "增加结果解释功能"
            ],
            confidence_level=0.85,
            business_impact=0.8,
            timestamp=datetime.now()
        )
        insights.append(pain_insight)
        
        # 市场机会洞察
        market_insight = UserInsight(
            insight_id="insight_market_001",
            category=InsightCategory.MARKET_GAPS,
            title="市场空白：企业级AI集成需求未满足",
            description="企业需要统一的AI服务管理和集成平台，现有解决方案碎片化严重",
            evidence=[gap.description for gap in market_gaps[:2]],
            implications=[
                "企业AI集成是巨大市场机会",
                "需要提供端到端解决方案",
                "标准化和互操作性是关键"
            ],
            actionable_recommendations=[
                "开发企业级AI中台",
                "建立AI服务标准",
                "提供专业服务支持"
            ],
            confidence_level=0.78,
            business_impact=0.9,
            timestamp=datetime.now()
        )
        insights.append(market_insight)
        
        # 技术机会洞察
        tech_insight = UserInsight(
            insight_id="insight_tech_001",
            category=InsightCategory.TECHNOLOGY_OPPORTUNITIES,
            title="技术趋势：多模态AI成为新增长点",
            description="多模态AI技术成熟，为产品创新提供新机会",
            evidence=[
                "多模态模型性能显著提升",
                "用户对多媒体内容处理需求增长",
                "竞争对手开始布局多模态能力"
            ],
            implications=[
                "多模态能力将成为竞争优势",
                "需要重新设计产品架构",
                "用户体验将发生根本改变"
            ],
            actionable_recommendations=[
                "投资多模态技术研发",
                "重新设计用户交互方式",
                "构建多模态数据处理能力"
            ],
            confidence_level=0.82,
            business_impact=0.85,
            timestamp=datetime.now()
        )
        insights.append(tech_insight)
        
        self.user_insights = insights
        return insights
    
    def _assess_innovation_level(self, concept: ProductConcept) -> Dict[str, Any]:
        """评估创新水平"""
        return {
            "overall_innovation_score": concept.innovation_level,
            "paradigm_shift_potential": concept.innovation_level * 0.9,
            "technology_novelty": concept.innovation_level * 0.8,
            "user_experience_innovation": concept.innovation_level * 0.85,
            "market_disruption_potential": concept.overall_score * concept.innovation_level,
            "innovation_categories": [
                "技术创新" if concept.innovation_level > 0.8 else "渐进式改进",
                "用户体验创新" if "个性化" in concept.name else "功能优化",
                "商业模式创新" if "平台" in concept.name else "产品创新"
            ]
        }
    
    def _generate_implementation_roadmap(self, concept: ProductConcept) -> Dict[str, Any]:
        """生成实施路线图"""
        return {
            "phase_1": {
                "duration": "3-6个月",
                "goals": ["MVP开发", "核心功能验证"],
                "deliverables": ["原型产品", "用户测试报告"],
                "resources_needed": ["研发团队", "测试用户"]
            },
            "phase_2": {
                "duration": "6-12个月",
                "goals": ["产品优化", "市场验证"],
                "deliverables": ["beta版本", "市场反馈分析"],
                "resources_needed": ["产品团队", "营销资源"]
            },
            "phase_3": {
                "duration": "12-18个月",
                "goals": ["规模化部署", "市场扩张"],
                "deliverables": ["正式产品", "商业化运营"],
                "resources_needed": ["全功能团队", "市场预算"]
            },
            "success_metrics": [
                f"用户满意度 > {concept.user_value_score * 100:.0f}%",
                f"市场份额 > {concept.market_potential_score * 10:.1f}%",
                f"收入目标达成率 > {concept.commercial_feasibility_score * 100:.0f}%"
            ],
            "risk_mitigation": [
                "技术风险：建立技术评审机制",
                "市场风险：持续用户调研",
                "竞争风险：差异化定位"
            ]
        }
    
    def _assess_concept_risks(self, concept: ProductConcept) -> Dict[str, Any]:
        """评估概念风险"""
        return {
            "technical_risks": {
                "complexity_risk": 1.0 - concept.commercial_feasibility_score,
                "scalability_risk": 0.3,
                "security_risk": 0.2,
                "mitigation_strategies": [
                    "分阶段开发降低复杂度",
                    "云原生架构确保可扩展性",
                    "安全优先设计原则"
                ]
            },
            "market_risks": {
                "adoption_risk": 1.0 - concept.user_value_score,
                "competition_risk": 0.4,
                "timing_risk": 0.3,
                "mitigation_strategies": [
                    "用户驱动的产品设计",
                    "差异化竞争策略",
                    "市场时机监控"
                ]
            },
            "business_risks": {
                "monetization_risk": 1.0 - concept.commercial_feasibility_score,
                "resource_risk": 0.35,
                "regulatory_risk": 0.25,
                "mitigation_strategies": [
                    "多元化商业模式",
                    "资源优化配置",
                    "合规性监控"
                ]
            },
            "overall_risk_level": "中等" if concept.overall_score > 0.7 else "较高"
        }
    
    def get_insight_summary(self) -> Dict[str, Any]:
        """获取洞察摘要"""
        return {
            "insights_generated": len(self.user_insights),
            "pain_points_identified": len(self.pain_points),
            "market_opportunities": len(self.market_opportunities),
            "product_concepts": len(self.product_concepts),
            "average_innovation_score": np.mean([c.innovation_level for c in self.product_concepts]) if self.product_concepts else 0,
            "highest_potential_concept": max(self.product_concepts, key=lambda x: x.overall_score).name if self.product_concepts else None,
            "key_insights": [insight.title for insight in self.user_insights],
            "last_updated": datetime.now().isoformat()
        }