"""
Market Intelligence Acquisition Agent
Based on Section 3.1.1 of the AI-Product Manager paper
负责竞品与成功案例选择、补充市场数据收集
"""

import asyncio
import json
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from enum import Enum

class MarketSegment(Enum):
    """市场细分"""
    CONSUMER_AI = "consumer_ai"
    ENTERPRISE_AI = "enterprise_ai"
    CONTENT_GENERATION = "content_generation"
    VERTICAL_INDUSTRY = "vertical_industry"
    ECOMMERCE = "ecommerce"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"

@dataclass
class CompetitorProfile:
    """竞品档案"""
    company_id: str
    name: str
    market_segment: MarketSegment
    market_activity_score: float  # 市场活跃度 0-1
    user_reputation_score: float  # 用户口碑 0-1
    report_quality_score: float   # 报告质量 0-1
    market_relevance_score: float # 市场相关性 0-1
    business_impact_score: float  # 商业影响力 0-1
    overall_score: float
    products: List[str]
    financial_data: Dict[str, Any]
    user_feedback: List[str]
    recent_updates: List[str]
    timestamp: datetime

@dataclass
class SuccessCase:
    """成功案例"""
    case_id: str
    product_name: str
    company: str
    market_segment: MarketSegment
    success_metrics: Dict[str, float]
    key_innovations: List[str]
    business_model: str
    user_adoption_rate: float
    revenue_growth: float
    market_penetration: float
    lessons_learned: List[str]
    timestamp: datetime

@dataclass
class MarketReport:
    """市场报告"""
    report_id: str
    title: str
    source: str
    market_segment: MarketSegment
    key_findings: List[str]
    market_size_data: Dict[str, float]
    growth_projections: Dict[str, float]
    competitive_landscape: List[str]
    emerging_trends: List[str]
    confidence_score: float
    publication_date: datetime

class MarketIntelligenceAgent:
    """市场情报获取代理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.min_quality_threshold = config.get('min_quality_threshold', 0.7)
        self.max_candidates = config.get('max_candidates', 20)
        self.target_selections = config.get('target_selections', 5)
        
        # 权重配置 (Section 3.1.1)
        self.selection_weights = {
            'market_activity': 0.25,      # 市场活跃度
            'user_reputation': 0.20,      # 用户口碑
            'report_quality': 0.20,       # 报告质量
            'market_relevance': 0.20,     # 市场相关性
            'business_impact': 0.15       # 商业影响力
        }
        
        # 数据存储
        self.competitor_database: List[CompetitorProfile] = []
        self.success_cases: List[SuccessCase] = []
        self.market_reports: List[MarketReport] = []
        
        # 初始化市场数据库
        self._initialize_market_database()
    
    def _initialize_market_database(self):
        """初始化市场数据库（模拟真实数据）"""
        # 根据论文，使用2022-2024年间的AI产品案例
        self._create_competitor_profiles()
        self._create_success_cases()
        self._create_market_reports()
    
    def _create_competitor_profiles(self):
        """创建竞品档案"""
        competitors_data = [
            {
                "company_id": "comp_001",
                "name": "OpenAI",
                "market_segment": MarketSegment.CONSUMER_AI,
                "products": ["ChatGPT", "GPT-4", "DALL-E", "Codex"],
                "market_activity": 0.95,
                "user_reputation": 0.88,
                "business_impact": 0.92,
                "financial_data": {"valuation": 90000000000, "revenue_2023": 1600000000}
            },
            {
                "company_id": "comp_002", 
                "name": "Anthropic",
                "market_segment": MarketSegment.ENTERPRISE_AI,
                "products": ["Claude", "Constitutional AI"],
                "market_activity": 0.87,
                "user_reputation": 0.85,
                "business_impact": 0.78,
                "financial_data": {"valuation": 15000000000, "revenue_2023": 200000000}
            },
            {
                "company_id": "comp_003",
                "name": "Midjourney",
                "market_segment": MarketSegment.CONTENT_GENERATION,
                "products": ["Midjourney AI Art"],
                "market_activity": 0.83,
                "user_reputation": 0.89,
                "business_impact": 0.75,
                "financial_data": {"valuation": 5000000000, "revenue_2023": 200000000}
            },
            {
                "company_id": "comp_004",
                "name": "Stability AI",
                "market_segment": MarketSegment.CONTENT_GENERATION,
                "products": ["Stable Diffusion", "StableCode"],
                "market_activity": 0.79,
                "user_reputation": 0.82,
                "business_impact": 0.71,
                "financial_data": {"valuation": 4000000000, "revenue_2023": 50000000}
            },
            {
                "company_id": "comp_005",
                "name": "Hugging Face",
                "market_segment": MarketSegment.ENTERPRISE_AI,
                "products": ["Transformers", "Datasets", "Spaces"],
                "market_activity": 0.91,
                "user_reputation": 0.87,
                "business_impact": 0.68,
                "financial_data": {"valuation": 4500000000, "revenue_2023": 70000000}
            },
            {
                "company_id": "comp_006",
                "name": "Cohere",
                "market_segment": MarketSegment.ENTERPRISE_AI,
                "products": ["Command", "Embed", "Classify"],
                "market_activity": 0.72,
                "user_reputation": 0.76,
                "business_impact": 0.65,
                "financial_data": {"valuation": 2200000000, "revenue_2023": 35000000}
            }
        ]
        
        for comp_data in competitors_data:
            profile = CompetitorProfile(
                company_id=comp_data["company_id"],
                name=comp_data["name"],
                market_segment=comp_data["market_segment"],
                market_activity_score=comp_data["market_activity"],
                user_reputation_score=comp_data["user_reputation"],
                report_quality_score=np.random.uniform(0.7, 0.9),  # 模拟报告质量
                market_relevance_score=np.random.uniform(0.75, 0.95),
                business_impact_score=comp_data["business_impact"],
                overall_score=0.0,  # 将在筛选时计算
                products=comp_data["products"],
                financial_data=comp_data["financial_data"],
                user_feedback=self._generate_user_feedback(comp_data["name"]),
                recent_updates=self._generate_recent_updates(comp_data["name"]),
                timestamp=datetime.now()
            )
            self.competitor_database.append(profile)
    
    def _create_success_cases(self):
        """创建成功案例"""
        success_cases_data = [
            {
                "case_id": "success_001",
                "product_name": "ChatGPT",
                "company": "OpenAI",
                "market_segment": MarketSegment.CONSUMER_AI,
                "business_model": "Freemium + Subscription",
                "user_adoption_rate": 0.89,
                "revenue_growth": 15.2,
                "market_penetration": 0.67
            },
            {
                "case_id": "success_002",
                "product_name": "Midjourney",
                "company": "Midjourney Inc",
                "market_segment": MarketSegment.CONTENT_GENERATION,
                "business_model": "Subscription",
                "user_adoption_rate": 0.78,
                "revenue_growth": 8.5,
                "market_penetration": 0.45
            },
            {
                "case_id": "success_003",
                "product_name": "GitHub Copilot",
                "company": "GitHub/Microsoft",
                "market_segment": MarketSegment.ENTERPRISE_AI,
                "business_model": "Subscription B2B",
                "user_adoption_rate": 0.72,
                "revenue_growth": 12.3,
                "market_penetration": 0.38
            }
        ]
        
        for case_data in success_cases_data:
            success_case = SuccessCase(
                case_id=case_data["case_id"],
                product_name=case_data["product_name"],
                company=case_data["company"],
                market_segment=case_data["market_segment"],
                success_metrics={
                    "user_satisfaction": np.random.uniform(0.8, 0.95),
                    "market_share": np.random.uniform(0.1, 0.4),
                    "revenue_per_user": np.random.uniform(5, 50)
                },
                key_innovations=[
                    "Large-scale transformer architecture",
                    "Constitutional AI training",
                    "Human feedback optimization"
                ],
                business_model=case_data["business_model"],
                user_adoption_rate=case_data["user_adoption_rate"],
                revenue_growth=case_data["revenue_growth"],
                market_penetration=case_data["market_penetration"],
                lessons_learned=[
                    "User experience is crucial for adoption",
                    "Freemium model accelerates growth",
                    "Community engagement drives retention"
                ],
                timestamp=datetime.now()
            )
            self.success_cases.append(success_case)
    
    def _create_market_reports(self):
        """创建市场报告"""
        reports_data = [
            {
                "report_id": "report_001",
                "title": "AI Market Analysis 2024",
                "source": "McKinsey & Company",
                "market_segment": MarketSegment.CONSUMER_AI,
                "market_size": 387000000000,  # $387B
                "growth_rate": 0.37
            },
            {
                "report_id": "report_002",
                "title": "Enterprise AI Adoption Report",
                "source": "Gartner",
                "market_segment": MarketSegment.ENTERPRISE_AI,
                "market_size": 156000000000,  # $156B
                "growth_rate": 0.42
            },
            {
                "report_id": "report_003",
                "title": "Generative AI Content Market",
                "source": "IDC",
                "market_segment": MarketSegment.CONTENT_GENERATION,
                "market_size": 45000000000,   # $45B
                "growth_rate": 0.58
            }
        ]
        
        for report_data in reports_data:
            report = MarketReport(
                report_id=report_data["report_id"],
                title=report_data["title"],
                source=report_data["source"],
                market_segment=report_data["market_segment"],
                key_findings=[
                    "AI adoption accelerating across industries",
                    "User experience becoming key differentiator",
                    "Regulatory considerations growing importance"
                ],
                market_size_data={
                    "total_addressable_market": report_data["market_size"],
                    "serviceable_addressable_market": report_data["market_size"] * 0.3,
                    "serviceable_obtainable_market": report_data["market_size"] * 0.05
                },
                growth_projections={
                    "2024": report_data["growth_rate"],
                    "2025": report_data["growth_rate"] * 0.8,
                    "2026": report_data["growth_rate"] * 0.6
                },
                competitive_landscape=[
                    "Market consolidation around major players",
                    "Emerging startups focusing on specialized niches",
                    "Open source solutions gaining traction"
                ],
                emerging_trends=[
                    "Multimodal AI capabilities",
                    "Edge computing integration", 
                    "Responsible AI frameworks"
                ],
                confidence_score=np.random.uniform(0.85, 0.95),
                publication_date=datetime.now()
            )
            self.market_reports.append(report)
    
    def _generate_user_feedback(self, company_name: str) -> List[str]:
        """生成用户反馈（模拟）"""
        feedback_templates = [
            f"{company_name}的产品用户体验很好，功能强大",
            f"对{company_name}的AI技术印象深刻，但价格略高",
            f"{company_name}的客服支持需要改进",
            f"使用{company_name}的产品显著提高了工作效率"
        ]
        return feedback_templates[:2]  # 返回部分反馈
    
    def _generate_recent_updates(self, company_name: str) -> List[str]:
        """生成最近更新（模拟）"""
        updates = [
            f"{company_name} 发布新版本，增强性能和安全性",
            f"{company_name} 扩展API功能，支持更多集成",
            f"{company_name} 获得新一轮融资，估值上升"
        ]
        return updates[:2]
    
    async def select_competitors_and_success_cases(self, 
                                                 market_context: Dict[str, Any],
                                                 target_segment: MarketSegment = None) -> Tuple[List[CompetitorProfile], List[SuccessCase]]:
        """
        竞品与成功案例选择
        根据论文Section 3.1.1的筛选算法
        """
        self.logger.info("开始竞品与成功案例选择")
        
        # 1. 筛选相关竞品
        relevant_competitors = self._filter_relevant_competitors(market_context, target_segment)
        
        # 2. 评估和排序竞品
        scored_competitors = self._score_and_rank_competitors(relevant_competitors, market_context)
        
        # 3. 选择最高质量的竞品
        selected_competitors = scored_competitors[:self.target_selections]
        
        # 4. 选择相关成功案例
        selected_success_cases = self._select_success_cases(market_context, target_segment)
        
        self.logger.info(f"选择了{len(selected_competitors)}个竞品和{len(selected_success_cases)}个成功案例")
        
        return selected_competitors, selected_success_cases
    
    def _filter_relevant_competitors(self, market_context: Dict[str, Any], 
                                   target_segment: MarketSegment = None) -> List[CompetitorProfile]:
        """筛选相关竞品"""
        relevant_competitors = []
        
        for competitor in self.competitor_database:
            # 市场细分匹配
            if target_segment and competitor.market_segment != target_segment:
                continue
            
            # 质量阈值筛选
            if competitor.market_activity_score < self.min_quality_threshold:
                continue
                
            # 市场相关性检查
            if self._check_market_relevance(competitor, market_context):
                relevant_competitors.append(competitor)
        
        return relevant_competitors
    
    def _check_market_relevance(self, competitor: CompetitorProfile, 
                               market_context: Dict[str, Any]) -> bool:
        """检查市场相关性"""
        # 根据市场上下文判断相关性
        context_keywords = market_context.get('keywords', [])
        competitor_keywords = [p.lower() for p in competitor.products]
        
        # 关键词匹配
        keyword_match = any(keyword.lower() in ' '.join(competitor_keywords) 
                           for keyword in context_keywords)
        
        # 市场规模匹配
        target_market_size = market_context.get('target_market_size', 0)
        competitor_market_impact = competitor.business_impact_score
        
        size_match = (target_market_size == 0 or  # 无特定要求
                     (target_market_size > 1000000000 and competitor_market_impact > 0.7) or  # 大市场
                     (target_market_size <= 1000000000 and competitor_market_impact > 0.5))   # 小市场
        
        return keyword_match or size_match
    
    def _score_and_rank_competitors(self, competitors: List[CompetitorProfile],
                                  market_context: Dict[str, Any]) -> List[CompetitorProfile]:
        """评分和排序竞品"""
        weights = self.selection_weights
        
        for competitor in competitors:
            # 计算综合得分 (Section 3.1.1)
            overall_score = (
                competitor.market_activity_score * weights['market_activity'] +
                competitor.user_reputation_score * weights['user_reputation'] +
                competitor.report_quality_score * weights['report_quality'] +
                competitor.market_relevance_score * weights['market_relevance'] +
                competitor.business_impact_score * weights['business_impact']
            )
            
            # 上下文调整
            if market_context.get('prioritize_innovation', False):
                # 如果优先考虑创新，提高新兴公司分数
                if competitor.business_impact_score < 0.8:  # 新兴公司
                    overall_score *= 1.1
            
            if market_context.get('prioritize_stability', False):
                # 如果优先考虑稳定性，提高大公司分数
                if competitor.business_impact_score > 0.8:  # 大公司
                    overall_score *= 1.15
            
            competitor.overall_score = overall_score
        
        # 按得分排序
        return sorted(competitors, key=lambda x: x.overall_score, reverse=True)
    
    def _select_success_cases(self, market_context: Dict[str, Any],
                            target_segment: MarketSegment = None) -> List[SuccessCase]:
        """选择成功案例"""
        relevant_cases = []
        
        for case in self.success_cases:
            # 市场细分匹配
            if target_segment and case.market_segment != target_segment:
                continue
            
            # 成功指标阈值
            if case.user_adoption_rate >= 0.7 and case.revenue_growth >= 5.0:
                relevant_cases.append(case)
        
        # 按综合成功指标排序
        scored_cases = []
        for case in relevant_cases:
            success_score = (
                case.user_adoption_rate * 0.4 +
                (case.revenue_growth / 20.0) * 0.3 +  # 标准化到0-1
                case.market_penetration * 0.3
            )
            scored_cases.append((case, success_score))
        
        # 排序并返回前N个
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        return [case for case, score in scored_cases[:3]]
    
    async def gather_supplementary_market_data(self, 
                                             selected_competitors: List[CompetitorProfile],
                                             selected_cases: List[SuccessCase]) -> Dict[str, Any]:
        """
        补充市场数据收集
        为每个筛选出的案例收集补充数据
        """
        self.logger.info("开始收集补充市场数据")
        
        supplementary_data = {
            "competitor_deep_analysis": {},
            "success_case_analysis": {},
            "market_trends": {},
            "user_sentiment": {},
            "financial_insights": {},
            "technology_analysis": {}
        }
        
        # 竞品深度分析
        for competitor in selected_competitors:
            deep_analysis = await self._deep_analyze_competitor(competitor)
            supplementary_data["competitor_deep_analysis"][competitor.company_id] = deep_analysis
        
        # 成功案例分析
        for case in selected_cases:
            case_analysis = await self._analyze_success_case(case)
            supplementary_data["success_case_analysis"][case.case_id] = case_analysis
        
        # 市场趋势分析
        supplementary_data["market_trends"] = await self._analyze_market_trends(
            selected_competitors, selected_cases
        )
        
        # 用户情感分析
        supplementary_data["user_sentiment"] = await self._analyze_user_sentiment(
            selected_competitors
        )
        
        # 财务洞察
        supplementary_data["financial_insights"] = await self._analyze_financial_data(
            selected_competitors, selected_cases
        )
        
        # 技术分析
        supplementary_data["technology_analysis"] = await self._analyze_technology_trends(
            selected_competitors, selected_cases
        )
        
        self.logger.info("补充市场数据收集完成")
        return supplementary_data
    
    async def _deep_analyze_competitor(self, competitor: CompetitorProfile) -> Dict[str, Any]:
        """深度分析竞品"""
        return {
            "business_model_analysis": {
                "revenue_streams": self._identify_revenue_streams(competitor),
                "cost_structure": self._analyze_cost_structure(competitor),
                "value_proposition": self._extract_value_proposition(competitor)
            },
            "product_feature_analysis": {
                "core_features": competitor.products,
                "feature_gaps": self._identify_feature_gaps(competitor),
                "innovation_level": competitor.business_impact_score
            },
            "market_positioning": {
                "target_segments": self._identify_target_segments(competitor),
                "competitive_advantages": self._identify_competitive_advantages(competitor),
                "market_share_estimate": competitor.business_impact_score * 0.5
            },
            "user_feedback_analysis": {
                "sentiment_score": np.mean([0.7, 0.8, 0.75]),  # 模拟情感分析
                "key_complaints": ["pricing", "complexity"],
                "key_praises": ["functionality", "performance"]
            },
            "financial_performance": competitor.financial_data,
            "recent_developments": competitor.recent_updates
        }
    
    async def _analyze_success_case(self, case: SuccessCase) -> Dict[str, Any]:
        """分析成功案例"""
        return {
            "success_factors": {
                "product_innovation": case.key_innovations,
                "market_timing": "optimal",
                "execution_quality": "high",
                "team_expertise": "excellent"
            },
            "growth_trajectory": {
                "user_adoption_curve": self._model_adoption_curve(case),
                "revenue_growth_pattern": case.revenue_growth,
                "market_expansion_strategy": "geographic + vertical"
            },
            "business_model_effectiveness": {
                "model_type": case.business_model,
                "monetization_efficiency": case.revenue_growth / case.user_adoption_rate,
                "scalability_score": case.market_penetration
            },
            "lessons_applicable": case.lessons_learned,
            "replication_feasibility": {
                "technical_complexity": "medium",
                "resource_requirements": "high",
                "market_barriers": "medium"
            }
        }
    
    async def _analyze_market_trends(self, competitors: List[CompetitorProfile],
                                   cases: List[SuccessCase]) -> Dict[str, Any]:
        """分析市场趋势"""
        return {
            "emerging_technologies": [
                "Multimodal AI",
                "Edge AI computing",
                "Federated learning",
                "Neural architecture search"
            ],
            "market_consolidation": {
                "trend": "increasing",
                "major_players": [comp.name for comp in competitors[:3]],
                "acquisition_activity": "high"
            },
            "user_behavior_shifts": [
                "Preference for AI-native applications",
                "Demand for explainable AI",
                "Privacy-conscious usage patterns"
            ],
            "regulatory_landscape": [
                "AI Act in EU",
                "Privacy regulations",
                "Algorithmic accountability"
            ],
            "investment_patterns": {
                "total_investment_2023": 50000000000,  # $50B
                "focus_areas": ["LLMs", "Computer Vision", "Robotics"],
                "geographic_distribution": {"US": 0.6, "China": 0.2, "EU": 0.15, "Others": 0.05}
            }
        }
    
    async def _analyze_user_sentiment(self, competitors: List[CompetitorProfile]) -> Dict[str, Any]:
        """分析用户情感"""
        sentiment_data = {}
        
        for competitor in competitors:
            sentiment_data[competitor.name] = {
                "overall_sentiment": competitor.user_reputation_score,
                "sentiment_distribution": {
                    "positive": competitor.user_reputation_score * 0.8,
                    "neutral": 0.2,
                    "negative": 1.0 - competitor.user_reputation_score - 0.2
                },
                "key_themes": {
                    "performance": competitor.user_reputation_score,
                    "ease_of_use": competitor.user_reputation_score * 0.9,
                    "value_for_money": competitor.user_reputation_score * 0.7,
                    "customer_support": competitor.user_reputation_score * 0.8
                },
                "recommendation_rate": competitor.user_reputation_score * 0.85
            }
        
        return sentiment_data
    
    async def _analyze_financial_data(self, competitors: List[CompetitorProfile],
                                    cases: List[SuccessCase]) -> Dict[str, Any]:
        """分析财务数据"""
        return {
            "valuation_trends": {
                "average_valuation": np.mean([comp.financial_data.get("valuation", 0) 
                                            for comp in competitors]),
                "valuation_multiples": {
                    "revenue_multiple": 15.5,
                    "user_multiple": 1200
                }
            },
            "revenue_patterns": {
                "average_growth_rate": np.mean([case.revenue_growth for case in cases]),
                "revenue_diversification": "increasing",
                "monetization_models": ["subscription", "usage-based", "enterprise"]
            },
            "funding_landscape": {
                "average_round_size": 45000000,  # $45M
                "funding_stages": ["Series A", "Series B", "Series C"],
                "investor_types": ["VCs", "Strategic", "Government"]
            },
            "profitability_outlook": {
                "time_to_profitability": "24-36 months",
                "margin_expectations": "15-25%",
                "key_cost_drivers": ["R&D", "Infrastructure", "Talent"]
            }
        }
    
    async def _analyze_technology_trends(self, competitors: List[CompetitorProfile],
                                       cases: List[SuccessCase]) -> Dict[str, Any]:
        """分析技术趋势"""
        return {
            "core_technologies": [
                "Large Language Models",
                "Computer Vision",
                "Reinforcement Learning",
                "Neural Architecture Search"
            ],
            "technology_maturity": {
                "LLMs": "mature",
                "Computer Vision": "mature",
                "Robotics": "emerging",
                "Quantum ML": "research"
            },
            "infrastructure_trends": [
                "Cloud-native AI platforms",
                "Edge computing adoption",
                "Specialized AI chips",
                "MLOps standardization"
            ],
            "innovation_hotspots": [
                "Multimodal AI",
                "AI Safety & Alignment",
                "Efficient AI Training",
                "AI Democratization"
            ],
            "competitive_dynamics": {
                "open_source_vs_proprietary": "balanced",
                "api_vs_on_premise": "api_dominant",
                "horizontal_vs_vertical": "horizontal_growing"
            }
        }
    
    # 辅助方法
    def _identify_revenue_streams(self, competitor: CompetitorProfile) -> List[str]:
        """识别收入来源"""
        if competitor.market_segment == MarketSegment.CONSUMER_AI:
            return ["Subscription", "Premium Features", "API Access"]
        elif competitor.market_segment == MarketSegment.ENTERPRISE_AI:
            return ["Enterprise Licenses", "Professional Services", "Support"]
        else:
            return ["Usage-based Pricing", "Subscriptions", "Marketplace"]
    
    def _analyze_cost_structure(self, competitor: CompetitorProfile) -> Dict[str, float]:
        """分析成本结构"""
        return {
            "R&D": 0.40,
            "Infrastructure": 0.25,
            "Sales & Marketing": 0.20,
            "General & Administrative": 0.15
        }
    
    def _extract_value_proposition(self, competitor: CompetitorProfile) -> str:
        """提取价值主张"""
        value_props = {
            MarketSegment.CONSUMER_AI: "Easy-to-use AI for everyone",
            MarketSegment.ENTERPRISE_AI: "Enterprise-grade AI solutions",
            MarketSegment.CONTENT_GENERATION: "Creative AI for content creators",
            MarketSegment.VERTICAL_INDUSTRY: "Industry-specific AI solutions"
        }
        return value_props.get(competitor.market_segment, "Advanced AI technology")
    
    def _identify_feature_gaps(self, competitor: CompetitorProfile) -> List[str]:
        """识别功能差距"""
        return ["Mobile optimization", "Multi-language support", "Advanced analytics"]
    
    def _identify_target_segments(self, competitor: CompetitorProfile) -> List[str]:
        """识别目标细分"""
        segment_mapping = {
            MarketSegment.CONSUMER_AI: ["Individual users", "Small businesses"],
            MarketSegment.ENTERPRISE_AI: ["Large enterprises", "Government"],
            MarketSegment.CONTENT_GENERATION: ["Content creators", "Marketing agencies"],
            MarketSegment.VERTICAL_INDUSTRY: ["Healthcare", "Finance", "Education"]
        }
        return segment_mapping.get(competitor.market_segment, ["General market"])
    
    def _identify_competitive_advantages(self, competitor: CompetitorProfile) -> List[str]:
        """识别竞争优势"""
        return ["Technical expertise", "Market position", "User base", "Financial resources"]
    
    def _model_adoption_curve(self, case: SuccessCase) -> List[float]:
        """模拟采用曲线"""
        # S曲线增长模拟
        months = 24
        curve = []
        for month in range(months):
            adoption = case.user_adoption_rate * (1 / (1 + np.exp(-0.3 * (month - 12))))
            curve.append(adoption)
        return curve
    
    def get_market_intelligence_summary(self) -> Dict[str, Any]:
        """获取市场情报摘要"""
        return {
            "database_size": {
                "competitors": len(self.competitor_database),
                "success_cases": len(self.success_cases),
                "market_reports": len(self.market_reports)
            },
            "data_quality": {
                "average_competitor_score": np.mean([comp.overall_score for comp in self.competitor_database if comp.overall_score > 0]),
                "data_freshness": "current",
                "coverage_completeness": "high"
            },
            "market_segments_covered": list(set([comp.market_segment.value for comp in self.competitor_database])),
            "last_updated": datetime.now().isoformat()
        }