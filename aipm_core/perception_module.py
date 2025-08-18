"""
感知模块 (Perception Module)
负责多渠道数据收集，包括用户需求洞察、市场数据、用户反馈等
"""

import asyncio
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
import numpy as np

@dataclass
class UserFeedback:
    """用户反馈数据结构"""
    user_id: str
    feedback_type: str  # 功能问题、界面设计、性能等
    content: str
    rating: float
    timestamp: datetime
    channel: str  # 产品内、社交媒体、邮件等

@dataclass
class MarketData:
    """市场数据结构"""
    source: str
    data_type: str  # 竞品分析、行业趋势、用户画像等
    content: Dict[str, Any]
    timestamp: datetime
    confidence: float

@dataclass
class UserBehavior:
    """用户行为数据结构"""
    user_id: str
    action: str
    session_duration: float
    page_views: int
    conversion_rate: float
    timestamp: datetime

class PerceptionModule:
    """感知模块实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_sources = config.get('data_sources', {})
        self.nlp_model = config.get('nlp_model', 'gpt-3.5-turbo')
        
        # 数据存储
        self.user_feedbacks: List[UserFeedback] = []
        self.market_data: List[MarketData] = []
        self.user_behaviors: List[UserBehavior] = []
        
    async def collect_user_needs(self, channels: List[str] = None) -> Dict[str, Any]:
        """
        用户需求洞察 - 从多渠道收集用户需求数据
        """
        if channels is None:
            channels = ['product_feedback', 'social_media', 'survey', 'support_tickets']
            
        collected_data = {}
        
        for channel in channels:
            try:
                if channel == 'product_feedback':
                    data = await self._collect_product_feedback()
                elif channel == 'social_media':
                    data = await self._collect_social_media_data()
                elif channel == 'survey':
                    data = await self._collect_survey_data()
                elif channel == 'support_tickets':
                    data = await self._collect_support_tickets()
                else:
                    data = []
                    
                collected_data[channel] = data
                self.logger.info(f"从{channel}收集到{len(data)}条数据")
                
            except Exception as e:
                self.logger.error(f"从{channel}收集数据失败: {e}")
                collected_data[channel] = []
                
        # 使用NLP处理和分析收集的数据
        processed_data = await self._process_user_needs(collected_data)
        return processed_data
    
    async def _collect_product_feedback(self) -> List[UserFeedback]:
        """收集产品内反馈数据"""
        # 模拟产品内反馈收集
        feedbacks = []
        
        # 这里应该连接到实际的产品反馈系统API
        mock_feedbacks = [
            {
                "user_id": "user_001",
                "feedback_type": "功能问题",
                "content": "推荐算法不够精准，经常推荐不相关内容",
                "rating": 2.5,
                "channel": "product_feedback"
            },
            {
                "user_id": "user_002", 
                "feedback_type": "界面设计",
                "content": "界面操作不够直观，需要优化用户体验",
                "rating": 3.0,
                "channel": "product_feedback"
            }
        ]
        
        for fb in mock_feedbacks:
            feedback = UserFeedback(
                user_id=fb["user_id"],
                feedback_type=fb["feedback_type"],
                content=fb["content"],
                rating=fb["rating"],
                timestamp=datetime.now(),
                channel=fb["channel"]
            )
            feedbacks.append(feedback)
            
        self.user_feedbacks.extend(feedbacks)
        return feedbacks
    
    async def _collect_social_media_data(self) -> List[UserFeedback]:
        """收集社交媒体数据"""
        # 模拟社交媒体数据收集
        social_data = []
        
        # 关键词监控
        keywords = ["AI产品", "智能推荐", "机器学习应用"]
        
        mock_social_data = [
            {
                "user_id": "social_user_001",
                "feedback_type": "功能需求",
                "content": "希望AI产品能更好理解用户意图",
                "rating": 4.0,
                "channel": "social_media"
            }
        ]
        
        for data in mock_social_data:
            feedback = UserFeedback(
                user_id=data["user_id"],
                feedback_type=data["feedback_type"], 
                content=data["content"],
                rating=data["rating"],
                timestamp=datetime.now(),
                channel=data["channel"]
            )
            social_data.append(feedback)
            
        return social_data
    
    async def _collect_survey_data(self) -> List[UserFeedback]:
        """收集用户调研数据"""
        # 模拟用户调研数据
        return []
    
    async def _collect_support_tickets(self) -> List[UserFeedback]:
        """收集客服工单数据"""
        # 模拟客服工单数据
        return []
    
    async def _process_user_needs(self, raw_data: Dict[str, List]) -> Dict[str, Any]:
        """使用NLP处理用户需求数据"""
        processed_data = {
            "pain_points": [],
            "feature_requests": [],
            "satisfaction_scores": {},
            "priority_needs": [],
            "user_segments": {}
        }
        
        all_feedbacks = []
        for channel, feedbacks in raw_data.items():
            all_feedbacks.extend(feedbacks)
        
        # 提取痛点
        pain_points = []
        feature_requests = []
        
        for feedback in all_feedbacks:
            if hasattr(feedback, 'rating') and feedback.rating < 3.0:
                pain_points.append({
                    "content": feedback.content,
                    "type": feedback.feedback_type,
                    "severity": 3.0 - feedback.rating
                })
            elif hasattr(feedback, 'feedback_type') and "需求" in feedback.feedback_type:
                feature_requests.append({
                    "content": feedback.content,
                    "priority": feedback.rating if hasattr(feedback, 'rating') else 3.0
                })
        
        processed_data["pain_points"] = pain_points
        processed_data["feature_requests"] = feature_requests
        
        # 计算满意度分数
        channel_scores = {}
        for channel, feedbacks in raw_data.items():
            if feedbacks:
                ratings = [fb.rating for fb in feedbacks if hasattr(fb, 'rating')]
                if ratings:
                    channel_scores[channel] = np.mean(ratings)
        
        processed_data["satisfaction_scores"] = channel_scores
        
        return processed_data
    
    async def collect_market_data(self) -> Dict[str, Any]:
        """市场数据收集"""
        market_insights = {
            "competitor_analysis": await self._analyze_competitors(),
            "industry_trends": await self._collect_industry_trends(),
            "market_size": await self._estimate_market_size(),
            "user_demographics": await self._analyze_user_demographics()
        }
        
        return market_insights
    
    async def _analyze_competitors(self) -> List[Dict[str, Any]]:
        """竞品分析"""
        # 模拟竞品分析数据
        competitors = [
            {
                "name": "竞品A",
                "market_share": 0.25,
                "strengths": ["技术先进", "用户体验好"],
                "weaknesses": ["价格较高", "功能复杂"],
                "features": ["智能推荐", "个性化定制", "数据分析"],
                "pricing": "premium"
            },
            {
                "name": "竞品B", 
                "market_share": 0.15,
                "strengths": ["价格便宜", "易于使用"],
                "weaknesses": ["功能有限", "技术落后"],
                "features": ["基础推荐", "简单界面"],
                "pricing": "budget"
            }
        ]
        
        return competitors
    
    async def _collect_industry_trends(self) -> Dict[str, Any]:
        """行业趋势收集"""
        trends = {
            "ai_adoption_rate": 0.78,
            "growth_rate": 0.25,
            "emerging_technologies": ["大模型", "多模态AI", "边缘计算"],
            "regulatory_changes": ["数据隐私法规", "AI伦理准则"],
            "market_drivers": ["数字化转型", "成本优化需求", "用户体验提升"]
        }
        
        return trends
    
    async def _estimate_market_size(self) -> Dict[str, float]:
        """市场规模估算"""
        return {
            "total_addressable_market": 10000000,  # TAM
            "serviceable_addressable_market": 2000000,  # SAM  
            "serviceable_obtainable_market": 200000  # SOM
        }
    
    async def _analyze_user_demographics(self) -> Dict[str, Any]:
        """用户画像分析"""
        demographics = {
            "age_groups": {
                "18-25": 0.20,
                "26-35": 0.45,
                "36-45": 0.25,
                "46+": 0.10
            },
            "industries": {
                "技术": 0.40,
                "金融": 0.25,
                "教育": 0.20,
                "其他": 0.15
            },
            "company_sizes": {
                "小型(<50人)": 0.30,
                "中型(50-500人)": 0.45,
                "大型(500+人)": 0.25
            }
        }
        
        return demographics
    
    async def collect_user_behavior(self) -> Dict[str, Any]:
        """用户行为数据收集"""
        behavior_data = {
            "usage_patterns": await self._analyze_usage_patterns(),
            "feature_adoption": await self._track_feature_adoption(),
            "retention_metrics": await self._calculate_retention_metrics(),
            "conversion_funnel": await self._analyze_conversion_funnel()
        }
        
        return behavior_data
    
    async def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """分析使用模式"""
        return {
            "daily_active_users": 5000,
            "avg_session_duration": 25.5,  # 分钟
            "peak_usage_hours": [9, 10, 14, 15, 20],
            "feature_usage_frequency": {
                "搜索": 0.85,
                "推荐": 0.65,
                "个人设置": 0.30
            }
        }
    
    async def _track_feature_adoption(self) -> Dict[str, float]:
        """功能采用率跟踪"""
        return {
            "新功能A": 0.45,
            "新功能B": 0.32,
            "新功能C": 0.78
        }
    
    async def _calculate_retention_metrics(self) -> Dict[str, float]:
        """计算留存指标"""
        return {
            "day_1_retention": 0.75,
            "day_7_retention": 0.45,
            "day_30_retention": 0.25,
            "monthly_churn_rate": 0.08
        }
    
    async def _analyze_conversion_funnel(self) -> Dict[str, float]:
        """转化漏斗分析"""
        return {
            "awareness_to_interest": 0.65,
            "interest_to_trial": 0.40,
            "trial_to_purchase": 0.25,
            "purchase_to_advocate": 0.15
        }
    
    async def detect_badcases(self) -> List[Dict[str, Any]]:
        """Badcase检测和分析"""
        badcases = []
        
        # 基于用户反馈检测问题
        for feedback in self.user_feedbacks:
            if feedback.rating < 2.0:
                badcase = {
                    "type": "用户反馈问题",
                    "severity": "高" if feedback.rating < 1.5 else "中",
                    "description": feedback.content,
                    "category": feedback.feedback_type,
                    "timestamp": feedback.timestamp,
                    "user_id": feedback.user_id
                }
                badcases.append(badcase)
        
        # 基于系统监控检测问题
        system_badcases = await self._detect_system_badcases()
        badcases.extend(system_badcases)
        
        return badcases
    
    async def _detect_system_badcases(self) -> List[Dict[str, Any]]:
        """系统级Badcase检测"""
        # 模拟系统监控数据
        return [
            {
                "type": "性能问题",
                "severity": "中",
                "description": "API响应时间超过500ms",
                "category": "技术",
                "timestamp": datetime.now()
            }
        ]
    
    def get_perception_summary(self) -> Dict[str, Any]:
        """获取感知模块汇总数据"""
        return {
            "data_collection_status": {
                "user_feedbacks": len(self.user_feedbacks),
                "market_data_points": len(self.market_data),
                "user_behaviors": len(self.user_behaviors)
            },
            "last_update": datetime.now().isoformat(),
            "data_quality_score": self._calculate_data_quality_score()
        }
    
    def _calculate_data_quality_score(self) -> float:
        """计算数据质量分数"""
        # 简化的数据质量评估
        total_data = len(self.user_feedbacks) + len(self.market_data) + len(self.user_behaviors)
        if total_data == 0:
            return 0.0
        
        # 基于数据完整性和时效性计算分数
        completeness_score = min(total_data / 100, 1.0)  # 假设100条数据为完整
        freshness_score = 1.0  # 简化处理，假设数据都是新鲜的
        
        return (completeness_score * 0.7 + freshness_score * 0.3)