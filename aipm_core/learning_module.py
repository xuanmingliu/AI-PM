"""
学习模块 (Learning Module)
负责持续优化能力，通过强化学习等算法不断改进决策策略
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from enum import Enum
import pickle

class RewardType(Enum):
    """奖励类型"""
    USER_SATISFACTION = "user_satisfaction"
    BUSINESS_METRICS = "business_metrics"
    TECHNICAL_PERFORMANCE = "technical_performance"
    BADCASE_REDUCTION = "badcase_reduction"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"

class ActionType(Enum):
    """动作类型"""
    PRIORITY_ADJUSTMENT = "priority_adjustment"
    MODEL_SELECTION = "model_selection"
    SCENARIO_SELECTION = "scenario_selection"
    PARAMETER_TUNING = "parameter_tuning"
    RESOURCE_ALLOCATION = "resource_allocation"

@dataclass
class Experience:
    """经验数据结构"""
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    action_type: ActionType
    timestamp: datetime
    context: Dict[str, Any] = None

@dataclass
class LearningMetrics:
    """学习指标"""
    total_experiences: int
    average_reward: float
    improvement_rate: float
    exploration_ratio: float
    convergence_score: float
    last_update: datetime

@dataclass
class OptimizationRule:
    """优化规则"""
    rule_id: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    confidence: float
    usage_count: int
    success_rate: float
    created_at: datetime

class ReinforcementLearner:
    """强化学习器"""
    
    def __init__(self, action_space_size: int, state_space_size: int, 
                 learning_rate: float = 0.01, discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q表初始化
        self.q_table = np.random.uniform(low=-1, high=1, 
                                       size=(state_space_size, action_space_size))
        
        # 经验回放缓冲区
        self.experience_buffer = deque(maxlen=10000)
        
    def select_action(self, state_index: int) -> int:
        """选择动作（ε-贪婪策略）"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(0, self.action_space_size)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[state_index])
    
    def update_q_table(self, state: int, action: int, reward: float, next_state: int):
        """更新Q表"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def add_experience(self, state: int, action: int, reward: float, next_state: int):
        """添加经验到回放缓冲区"""
        self.experience_buffer.append((state, action, reward, next_state))
    
    def experience_replay(self, batch_size: int = 32):
        """经验回放训练"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # 随机采样batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        for idx in indices:
            state, action, reward, next_state = self.experience_buffer[idx]
            self.update_q_table(state, action, reward, next_state)

class LearningModule:
    """学习模块实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 学习配置
        self.learning_rate = config.get('learning_rate', 0.01)
        self.discount_factor = config.get('discount_factor', 0.95)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.batch_size = config.get('batch_size', 32)
        
        # 经验存储
        self.experiences: List[Experience] = []
        self.max_experiences = config.get('max_experiences', 10000)
        
        # 奖励权重
        self.reward_weights = config.get('reward_weights', {
            RewardType.USER_SATISFACTION: 0.3,
            RewardType.BUSINESS_METRICS: 0.3,
            RewardType.TECHNICAL_PERFORMANCE: 0.2,
            RewardType.BADCASE_REDUCTION: 0.1,
            RewardType.EFFICIENCY_IMPROVEMENT: 0.1
        })
        
        # 优化规则库
        self.optimization_rules: List[OptimizationRule] = []
        
        # 性能指标
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # 强化学习器
        self.rl_learners: Dict[ActionType, ReinforcementLearner] = {}
        
        # 初始化强化学习器
        self._initialize_rl_learners()
        
        # 存储路径
        self.storage_path = config.get('storage_path', './aipm_learning')
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 加载历史数据
        self._load_learning_data()
    
    def _initialize_rl_learners(self):
        """初始化强化学习器"""
        for action_type in ActionType:
            self.rl_learners[action_type] = ReinforcementLearner(
                action_space_size=10,  # 可配置
                state_space_size=20,   # 可配置
                learning_rate=self.learning_rate,
                discount_factor=self.discount_factor,
                epsilon=self.exploration_rate
            )
    
    def record_experience(self, state: Dict[str, Any], action: Dict[str, Any],
                         reward_signals: Dict[RewardType, float],
                         next_state: Dict[str, Any], action_type: ActionType,
                         context: Dict[str, Any] = None):
        """记录经验"""
        # 计算综合奖励
        total_reward = self._calculate_total_reward(reward_signals)
        
        experience = Experience(
            state=state,
            action=action,
            reward=total_reward,
            next_state=next_state,
            action_type=action_type,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        self.experiences.append(experience)
        
        # 限制经验数量
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)
        
        # 更新强化学习器
        self._update_rl_learner(experience)
        
        self.logger.info(f"记录经验: {action_type.value}, 奖励: {total_reward:.3f}")
    
    def _calculate_total_reward(self, reward_signals: Dict[RewardType, float]) -> float:
        """计算总奖励"""
        total_reward = 0.0
        
        for reward_type, value in reward_signals.items():
            weight = self.reward_weights.get(reward_type, 0.0)
            total_reward += weight * value
        
        return total_reward
    
    def _update_rl_learner(self, experience: Experience):
        """更新强化学习器"""
        if experience.action_type not in self.rl_learners:
            return
        
        learner = self.rl_learners[experience.action_type]
        
        # 将状态和动作转换为索引（简化处理）
        state_index = self._state_to_index(experience.state)
        action_index = self._action_to_index(experience.action)
        next_state_index = self._state_to_index(experience.next_state)
        
        # 添加经验并更新Q表
        learner.add_experience(state_index, action_index, experience.reward, next_state_index)
        learner.update_q_table(state_index, action_index, experience.reward, next_state_index)
    
    def _state_to_index(self, state: Dict[str, Any]) -> int:
        """将状态转换为索引（简化实现）"""
        # 这是一个简化的状态编码，实际应用中需要更复杂的方法
        state_str = json.dumps(state, sort_keys=True)
        return hash(state_str) % 20  # 假设状态空间大小为20
    
    def _action_to_index(self, action: Dict[str, Any]) -> int:
        """将动作转换为索引（简化实现）"""
        action_str = json.dumps(action, sort_keys=True)
        return hash(action_str) % 10  # 假设动作空间大小为10
    
    def suggest_optimization(self, current_state: Dict[str, Any],
                           action_type: ActionType) -> Dict[str, Any]:
        """建议优化动作"""
        if action_type not in self.rl_learners:
            return {}
        
        learner = self.rl_learners[action_type]
        state_index = self._state_to_index(current_state)
        
        # 使用强化学习器选择动作
        action_index = learner.select_action(state_index)
        
        # 将动作索引转换为具体动作（需要根据实际情况实现）
        suggested_action = self._index_to_action(action_index, action_type, current_state)
        
        self.logger.info(f"建议{action_type.value}优化: {suggested_action}")
        return suggested_action
    
    def _index_to_action(self, action_index: int, action_type: ActionType,
                        current_state: Dict[str, Any]) -> Dict[str, Any]:
        """将动作索引转换为具体动作"""
        if action_type == ActionType.PRIORITY_ADJUSTMENT:
            # 优先级调整动作
            adjustment_factor = (action_index - 5) * 0.1  # -0.5 到 0.5
            return {
                "type": "priority_adjustment",
                "adjustment_factor": adjustment_factor,
                "reason": f"基于历史经验的优先级调整 ({adjustment_factor:+.1f})"
            }
        
        elif action_type == ActionType.MODEL_SELECTION:
            # 模型选择动作
            model_preferences = [
                "performance_optimized", "cost_optimized", "latency_optimized",
                "accuracy_optimized", "balanced", "experimental"
            ]
            preference = model_preferences[action_index % len(model_preferences)]
            return {
                "type": "model_selection",
                "preference": preference,
                "reason": f"基于学习经验推荐{preference}模型"
            }
        
        elif action_type == ActionType.SCENARIO_SELECTION:
            # 场景选择动作
            scenarios = [
                "high_traffic", "cost_sensitive", "innovation_focused",
                "stability_focused", "growth_oriented"
            ]
            scenario = scenarios[action_index % len(scenarios)]
            return {
                "type": "scenario_selection",
                "recommended_scenario": scenario,
                "reason": f"基于历史数据推荐{scenario}场景"
            }
        
        elif action_type == ActionType.PARAMETER_TUNING:
            # 参数调优动作
            return {
                "type": "parameter_tuning",
                "learning_rate_adjustment": (action_index - 5) * 0.001,
                "batch_size_adjustment": (action_index - 5) * 2,
                "reason": "基于性能反馈的参数调优建议"
            }
        
        elif action_type == ActionType.RESOURCE_ALLOCATION:
            # 资源分配动作
            allocation_strategies = [
                "compute_heavy", "memory_heavy", "balanced", "minimal",
                "peak_performance", "cost_efficient"
            ]
            strategy = allocation_strategies[action_index % len(allocation_strategies)]
            return {
                "type": "resource_allocation",
                "strategy": strategy,
                "reason": f"推荐{strategy}资源分配策略"
            }
        
        return {"type": "no_action", "reason": "未知动作类型"}
    
    def analyze_badcases(self, badcases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析Bad Case并生成改进建议"""
        self.logger.info(f"分析{len(badcases)}个Bad Case")
        
        # 按类型分组Bad Case
        badcase_groups = defaultdict(list)
        for badcase in badcases:
            category = badcase.get('category', 'unknown')
            badcase_groups[category].append(badcase)
        
        analysis_results = {}
        improvement_suggestions = []
        
        for category, cases in badcase_groups.items():
            # 分析每个类别的Bad Case
            category_analysis = self._analyze_badcase_category(category, cases)
            analysis_results[category] = category_analysis
            
            # 生成改进建议
            suggestions = self._generate_improvement_suggestions(category, cases)
            improvement_suggestions.extend(suggestions)
        
        # 更新优化规则
        self._update_optimization_rules(badcases, improvement_suggestions)
        
        return {
            "analysis_results": analysis_results,
            "improvement_suggestions": improvement_suggestions,
            "total_badcases": len(badcases),
            "categories_affected": len(badcase_groups),
            "priority_actions": self._prioritize_actions(improvement_suggestions)
        }
    
    def _analyze_badcase_category(self, category: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析特定类别的Bad Case"""
        severities = [case.get('severity', 'medium') for case in cases]
        severity_counts = {
            'high': severities.count('high'),
            'medium': severities.count('medium'),
            'low': severities.count('low')
        }
        
        # 找出常见模式
        descriptions = [case.get('description', '') for case in cases]
        common_keywords = self._extract_common_keywords(descriptions)
        
        return {
            "case_count": len(cases),
            "severity_distribution": severity_counts,
            "common_keywords": common_keywords,
            "trend": self._calculate_trend(category),
            "impact_score": self._calculate_impact_score(cases)
        }
    
    def _extract_common_keywords(self, descriptions: List[str]) -> List[str]:
        """提取常见关键词"""
        word_counts = defaultdict(int)
        
        for desc in descriptions:
            words = desc.lower().split()
            for word in words:
                if len(word) > 3:  # 过滤短词
                    word_counts[word] += 1
        
        # 返回出现频率最高的关键词
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:5]]
    
    def _calculate_trend(self, category: str) -> str:
        """计算趋势"""
        # 简化的趋势计算
        recent_count = len([exp for exp in self.experiences[-100:] 
                          if exp.context and exp.context.get('badcase_category') == category])
        historical_count = len([exp for exp in self.experiences[:-100] 
                              if exp.context and exp.context.get('badcase_category') == category])
        
        if recent_count > historical_count * 1.2:
            return "increasing"
        elif recent_count < historical_count * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_impact_score(self, cases: List[Dict[str, Any]]) -> float:
        """计算影响分数"""
        severity_weights = {'high': 3, 'medium': 2, 'low': 1}
        total_impact = sum(severity_weights.get(case.get('severity', 'low'), 1) for case in cases)
        return min(total_impact / 10.0, 1.0)  # 标准化到0-1
    
    def _generate_improvement_suggestions(self, category: str, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成改进建议"""
        suggestions = []
        
        if category == "performance":
            suggestions.append({
                "type": "optimization",
                "action": "optimize_algorithms",
                "priority": "high",
                "description": "优化核心算法以提升性能",
                "estimated_impact": 0.8
            })
        
        elif category == "user_experience":
            suggestions.append({
                "type": "ui_improvement",
                "action": "simplify_interface",
                "priority": "medium",
                "description": "简化用户界面，提升易用性",
                "estimated_impact": 0.6
            })
        
        elif category == "data_quality":
            suggestions.append({
                "type": "data_enhancement",
                "action": "improve_data_collection",
                "priority": "high",
                "description": "改进数据收集和清洗流程",
                "estimated_impact": 0.7
            })
        
        return suggestions
    
    def _prioritize_actions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优先级排序动作"""
        priority_order = {"high": 3, "medium": 2, "low": 1}
        
        sorted_suggestions = sorted(
            suggestions,
            key=lambda x: (priority_order.get(x.get('priority', 'low'), 1), 
                          x.get('estimated_impact', 0)),
            reverse=True
        )
        
        return sorted_suggestions[:5]  # 返回前5个优先动作
    
    def _update_optimization_rules(self, badcases: List[Dict[str, Any]], 
                                 suggestions: List[Dict[str, Any]]):
        """更新优化规则库"""
        for suggestion in suggestions:
            # 创建新的优化规则
            rule = OptimizationRule(
                rule_id=f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                condition={
                    "badcase_category": suggestion.get('type'),
                    "min_case_count": 3,
                    "severity_threshold": "medium"
                },
                action=suggestion,
                confidence=suggestion.get('estimated_impact', 0.5),
                usage_count=0,
                success_rate=0.0,
                created_at=datetime.now()
            )
            
            self.optimization_rules.append(rule)
    
    def continuous_learning(self, performance_metrics: Dict[str, float],
                          user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """持续学习过程"""
        # 分析性能趋势
        trend_analysis = self._analyze_performance_trends(performance_metrics)
        
        # 处理用户反馈
        feedback_insights = self._process_user_feedback(user_feedback)
        
        # 调整学习参数
        learning_adjustments = self._adjust_learning_parameters(trend_analysis, feedback_insights)
        
        # 执行经验回放
        self._perform_experience_replay()
        
        # 更新模型
        model_updates = self._update_learning_models()
        
        return {
            "trend_analysis": trend_analysis,
            "feedback_insights": feedback_insights,
            "learning_adjustments": learning_adjustments,
            "model_updates": model_updates,
            "learning_metrics": self._get_learning_metrics()
        }
    
    def _analyze_performance_trends(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """分析性能趋势"""
        trends = {}
        
        for metric_name, current_value in metrics.items():
            # 记录历史性能
            self.performance_history[metric_name].append(current_value)
            
            # 保持历史记录长度
            if len(self.performance_history[metric_name]) > 100:
                self.performance_history[metric_name].pop(0)
            
            # 计算趋势
            if len(self.performance_history[metric_name]) >= 10:
                recent_avg = np.mean(self.performance_history[metric_name][-10:])
                historical_avg = np.mean(self.performance_history[metric_name][-20:-10])
                
                if recent_avg > historical_avg * 1.05:
                    trend = "improving"
                elif recent_avg < historical_avg * 0.95:
                    trend = "declining"
                else:
                    trend = "stable"
                
                trends[metric_name] = {
                    "trend": trend,
                    "current_value": current_value,
                    "recent_average": recent_avg,
                    "historical_average": historical_avg,
                    "improvement_rate": (recent_avg - historical_avg) / historical_avg
                }
        
        return trends
    
    def _process_user_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户反馈"""
        satisfaction_score = feedback.get('satisfaction_score', 3.0)
        feedback_text = feedback.get('feedback_text', '')
        
        # 情感分析（简化）
        sentiment = self._analyze_sentiment(feedback_text)
        
        # 提取改进点
        improvement_areas = self._extract_improvement_areas(feedback_text)
        
        return {
            "satisfaction_score": satisfaction_score,
            "sentiment": sentiment,
            "improvement_areas": improvement_areas,
            "feedback_trend": self._calculate_feedback_trend()
        }
    
    def _analyze_sentiment(self, text: str) -> str:
        """情感分析（简化实现）"""
        positive_words = ["好", "棒", "优秀", "满意", "喜欢", "推荐"]
        negative_words = ["差", "糟糕", "不满", "问题", "错误", "失望"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_improvement_areas(self, text: str) -> List[str]:
        """提取改进领域"""
        improvement_keywords = {
            "性能": ["慢", "卡", "延迟", "响应"],
            "界面": ["界面", "UI", "操作", "体验"],
            "功能": ["功能", "特性", "需求", "建议"],
            "准确性": ["错误", "准确", "精度", "结果"]
        }
        
        areas = []
        for area, keywords in improvement_keywords.items():
            if any(keyword in text for keyword in keywords):
                areas.append(area)
        
        return areas
    
    def _calculate_feedback_trend(self) -> str:
        """计算反馈趋势"""
        # 简化的反馈趋势计算
        recent_experiences = self.experiences[-50:]
        if not recent_experiences:
            return "stable"
        
        recent_rewards = [exp.reward for exp in recent_experiences]
        avg_reward = np.mean(recent_rewards)
        
        if avg_reward > 0.6:
            return "improving"
        elif avg_reward < 0.4:
            return "declining"
        else:
            return "stable"
    
    def _adjust_learning_parameters(self, trend_analysis: Dict[str, Any],
                                  feedback_insights: Dict[str, Any]) -> Dict[str, Any]:
        """调整学习参数"""
        adjustments = {}
        
        # 根据性能趋势调整探索率
        declining_metrics = sum(1 for trend in trend_analysis.values() 
                              if trend.get('trend') == 'declining')
        
        if declining_metrics > len(trend_analysis) * 0.5:
            # 性能下降，增加探索
            new_epsilon = min(self.exploration_rate * 1.2, 0.3)
            adjustments['exploration_rate'] = new_epsilon
            for learner in self.rl_learners.values():
                learner.epsilon = new_epsilon
        
        # 根据用户满意度调整学习率
        satisfaction = feedback_insights.get('satisfaction_score', 3.0)
        if satisfaction < 3.0:
            # 满意度低，增加学习率
            new_lr = min(self.learning_rate * 1.1, 0.1)
            adjustments['learning_rate'] = new_lr
            for learner in self.rl_learners.values():
                learner.learning_rate = new_lr
        
        return adjustments
    
    def _perform_experience_replay(self):
        """执行经验回放"""
        for action_type, learner in self.rl_learners.items():
            learner.experience_replay(self.batch_size)
    
    def _update_learning_models(self) -> Dict[str, Any]:
        """更新学习模型"""
        updates = {}
        
        for action_type, learner in self.rl_learners.items():
            # 计算Q表统计信息
            q_mean = np.mean(learner.q_table)
            q_std = np.std(learner.q_table)
            
            updates[action_type.value] = {
                "q_table_mean": q_mean,
                "q_table_std": q_std,
                "experience_count": len(learner.experience_buffer),
                "epsilon": learner.epsilon,
                "learning_rate": learner.learning_rate
            }
        
        return updates
    
    def _get_learning_metrics(self) -> LearningMetrics:
        """获取学习指标"""
        if not self.experiences:
            return LearningMetrics(
                total_experiences=0,
                average_reward=0.0,
                improvement_rate=0.0,
                exploration_ratio=0.0,
                convergence_score=0.0,
                last_update=datetime.now()
            )
        
        recent_rewards = [exp.reward for exp in self.experiences[-100:]]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        
        # 计算改进率
        if len(self.experiences) >= 200:
            old_rewards = [exp.reward for exp in self.experiences[-200:-100]]
            old_avg = np.mean(old_rewards) if old_rewards else 0.0
            improvement_rate = (avg_reward - old_avg) / max(abs(old_avg), 0.1)
        else:
            improvement_rate = 0.0
        
        return LearningMetrics(
            total_experiences=len(self.experiences),
            average_reward=avg_reward,
            improvement_rate=improvement_rate,
            exploration_ratio=self.exploration_rate,
            convergence_score=self._calculate_convergence_score(),
            last_update=datetime.now()
        )
    
    def _calculate_convergence_score(self) -> float:
        """计算收敛分数"""
        if len(self.experiences) < 50:
            return 0.0
        
        recent_rewards = [exp.reward for exp in self.experiences[-50:]]
        reward_variance = np.var(recent_rewards)
        
        # 方差越小，收敛性越好
        convergence_score = max(0.0, 1.0 - reward_variance)
        return convergence_score
    
    def save_learning_data(self):
        """保存学习数据"""
        try:
            # 保存经验数据
            experiences_data = [asdict(exp) for exp in self.experiences]
            with open(os.path.join(self.storage_path, 'experiences.json'), 'w', encoding='utf-8') as f:
                json.dump(experiences_data, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存Q表
            for action_type, learner in self.rl_learners.items():
                q_table_path = os.path.join(self.storage_path, f'q_table_{action_type.value}.npy')
                np.save(q_table_path, learner.q_table)
            
            # 保存优化规则
            rules_data = [asdict(rule) for rule in self.optimization_rules]
            with open(os.path.join(self.storage_path, 'optimization_rules.json'), 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info("学习数据保存成功")
            
        except Exception as e:
            self.logger.error(f"保存学习数据失败: {e}")
    
    def _load_learning_data(self):
        """加载历史学习数据"""
        try:
            # 加载经验数据
            experiences_path = os.path.join(self.storage_path, 'experiences.json')
            if os.path.exists(experiences_path):
                with open(experiences_path, 'r', encoding='utf-8') as f:
                    experiences_data = json.load(f)
                
                # 重建经验对象（简化版）
                for exp_data in experiences_data[-1000:]:  # 只加载最近的1000条经验
                    if 'timestamp' in exp_data:
                        exp_data['timestamp'] = datetime.fromisoformat(exp_data['timestamp'])
                    if 'action_type' in exp_data:
                        exp_data['action_type'] = ActionType(exp_data['action_type'])
                    
                    self.experiences.append(Experience(**exp_data))
            
            # 加载Q表
            for action_type in ActionType:
                q_table_path = os.path.join(self.storage_path, f'q_table_{action_type.value}.npy')
                if os.path.exists(q_table_path):
                    self.rl_learners[action_type].q_table = np.load(q_table_path)
            
            self.logger.info(f"加载了{len(self.experiences)}条历史经验")
            
        except Exception as e:
            self.logger.warning(f"加载历史学习数据失败: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习模块摘要"""
        metrics = self._get_learning_metrics()
        
        return {
            "learning_metrics": asdict(metrics),
            "active_rules": len(self.optimization_rules),
            "performance_trends": len(self.performance_history),
            "recent_improvements": self._get_recent_improvements(),
            "learning_status": "active" if len(self.experiences) > 0 else "inactive"
        }
    
    def _get_recent_improvements(self) -> List[Dict[str, Any]]:
        """获取最近的改进"""
        improvements = []
        
        # 分析最近的正奖励经验
        recent_positive_experiences = [
            exp for exp in self.experiences[-50:] if exp.reward > 0.5
        ]
        
        for exp in recent_positive_experiences[-5:]:  # 最近5个正面经验
            improvements.append({
                "action_type": exp.action_type.value,
                "reward": exp.reward,
                "timestamp": exp.timestamp.isoformat(),
                "improvement_area": exp.context.get('improvement_area', 'general') if exp.context else 'general'
            })
        
        return improvements