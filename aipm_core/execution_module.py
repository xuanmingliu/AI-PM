"""
执行模块 (Execution Module)
负责具体任务执行，包括数据集构建、评价体系运行、产品迭代优化、运营活动策划等
"""

import asyncio
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import requests
import time

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    """任务类型"""
    DATA_COLLECTION = "data_collection"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"
    OPTIMIZATION = "optimization"
    MARKETING = "marketing"

@dataclass
class ExecutionTask:
    """执行任务数据结构"""
    task_id: str
    name: str
    task_type: TaskType
    description: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class DatasetSpec:
    """数据集规格"""
    dataset_name: str
    data_sources: List[str]
    target_size: int
    quality_requirements: Dict[str, float]
    annotation_rules: Dict[str, Any]
    validation_split: float = 0.2

@dataclass
class Campaign:
    """营销活动"""
    campaign_id: str
    name: str
    type: str  # email, social, ads等
    target_audience: List[str]
    content: Dict[str, Any]
    budget: float
    start_date: datetime
    end_date: datetime
    metrics: Dict[str, float] = None

class ExecutionModule:
    """执行模块实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 任务队列和状态跟踪
        self.task_queue: List[ExecutionTask] = []
        self.running_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: Dict[str, ExecutionTask] = {}
        
        # 执行器配置
        self.max_concurrent_tasks = config.get('max_concurrent_tasks', 5)
        self.data_storage_path = config.get('data_storage_path', './aipm_data')
        self.model_storage_path = config.get('model_storage_path', './aipm_models')
        
        # 创建存储目录
        os.makedirs(self.data_storage_path, exist_ok=True)
        os.makedirs(self.model_storage_path, exist_ok=True)
        
        # API连接配置
        self.api_configs = config.get('api_configs', {})
        
    async def execute_task(self, task: ExecutionTask) -> ExecutionTask:
        """执行单个任务"""
        self.logger.info(f"开始执行任务: {task.name} ({task.task_id})")
        
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        task.progress = 0.0
        
        self.running_tasks[task.task_id] = task
        
        try:
            # 根据任务类型分派执行
            if task.task_type == TaskType.DATA_COLLECTION:
                result = await self._execute_data_collection(task)
            elif task.task_type == TaskType.DATA_PROCESSING:
                result = await self._execute_data_processing(task)
            elif task.task_type == TaskType.MODEL_TRAINING:
                result = await self._execute_model_training(task)
            elif task.task_type == TaskType.EVALUATION:
                result = await self._execute_evaluation(task)
            elif task.task_type == TaskType.DEPLOYMENT:
                result = await self._execute_deployment(task)
            elif task.task_type == TaskType.OPTIMIZATION:
                result = await self._execute_optimization(task)
            elif task.task_type == TaskType.MARKETING:
                result = await self._execute_marketing(task)
            else:
                raise ValueError(f"未知任务类型: {task.task_type}")
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.end_time = datetime.now()
            
            self.logger.info(f"任务完成: {task.name}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.end_time = datetime.now()
            self.logger.error(f"任务执行失败: {task.name}, 错误: {e}")
        
        finally:
            # 从运行队列移除，添加到完成队列
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
        return task
    
    async def _execute_data_collection(self, task: ExecutionTask) -> Dict[str, Any]:
        """执行数据收集任务"""
        params = task.parameters
        data_sources = params.get('data_sources', [])
        target_size = params.get('target_size', 1000)
        
        collected_data = []
        
        for i, source in enumerate(data_sources):
            self.logger.info(f"从数据源收集数据: {source}")
            
            # 模拟数据收集过程
            source_data = await self._collect_from_source(source, target_size // len(data_sources))
            collected_data.extend(source_data)
            
            # 更新进度
            task.progress = (i + 1) / len(data_sources) * 0.8
            
        # 数据清洗和验证
        self.logger.info("开始数据清洗和验证")
        cleaned_data = await self._clean_and_validate_data(collected_data)
        task.progress = 0.9
        
        # 保存数据集
        dataset_path = os.path.join(self.data_storage_path, f"{task.task_id}_dataset.json")
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        task.progress = 1.0
        
        return {
            "dataset_path": dataset_path,
            "total_samples": len(cleaned_data),
            "data_quality_score": self._calculate_data_quality(cleaned_data),
            "collection_summary": {
                "sources_used": len(data_sources),
                "samples_collected": len(collected_data),
                "samples_after_cleaning": len(cleaned_data)
            }
        }
    
    async def _collect_from_source(self, source: str, sample_count: int) -> List[Dict[str, Any]]:
        """从指定数据源收集数据"""
        # 模拟数据收集，实际应用中需要连接真实数据源
        await asyncio.sleep(1)  # 模拟网络延迟
        
        mock_data = []
        for i in range(sample_count):
            mock_data.append({
                "id": f"{source}_{i}",
                "source": source,
                "content": f"来自{source}的示例数据{i}",
                "timestamp": datetime.now().isoformat(),
                "quality_score": np.random.uniform(0.6, 1.0)
            })
            
        return mock_data
    
    async def _clean_and_validate_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """数据清洗和验证"""
        cleaned_data = []
        
        for item in raw_data:
            # 数据质量检查
            if item.get('quality_score', 0) > 0.7:  # 质量阈值
                # 数据清洗
                cleaned_item = {
                    "id": item["id"],
                    "source": item["source"],
                    "content": item["content"].strip(),
                    "timestamp": item["timestamp"],
                    "quality_score": item["quality_score"]
                }
                cleaned_data.append(cleaned_item)
        
        return cleaned_data
    
    def _calculate_data_quality(self, data: List[Dict[str, Any]]) -> float:
        """计算数据质量分数"""
        if not data:
            return 0.0
            
        quality_scores = [item.get('quality_score', 0) for item in data]
        return np.mean(quality_scores)
    
    async def _execute_data_processing(self, task: ExecutionTask) -> Dict[str, Any]:
        """执行数据处理任务"""
        params = task.parameters
        input_path = params.get('input_path')
        processing_steps = params.get('processing_steps', [])
        
        # 加载数据
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = data
        
        for i, step in enumerate(processing_steps):
            self.logger.info(f"执行处理步骤: {step}")
            processed_data = await self._apply_processing_step(processed_data, step)
            task.progress = (i + 1) / len(processing_steps)
        
        # 保存处理后的数据
        output_path = os.path.join(self.data_storage_path, f"{task.task_id}_processed.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        return {
            "output_path": output_path,
            "original_size": len(data),
            "processed_size": len(processed_data),
            "processing_steps": processing_steps
        }
    
    async def _apply_processing_step(self, data: List[Dict[str, Any]], step: str) -> List[Dict[str, Any]]:
        """应用数据处理步骤"""
        if step == "deduplication":
            # 去重
            seen_ids = set()
            deduplicated = []
            for item in data:
                if item['id'] not in seen_ids:
                    deduplicated.append(item)
                    seen_ids.add(item['id'])
            return deduplicated
        
        elif step == "normalization":
            # 标准化
            for item in data:
                item['content'] = item['content'].lower().strip()
            return data
        
        elif step == "feature_extraction":
            # 特征提取（模拟）
            for item in data:
                item['features'] = {
                    "length": len(item['content']),
                    "word_count": len(item['content'].split()),
                    "source_type": item['source']
                }
            return data
        
        return data
    
    async def _execute_model_training(self, task: ExecutionTask) -> Dict[str, Any]:
        """执行模型训练任务"""
        params = task.parameters
        dataset_path = params.get('dataset_path')
        model_type = params.get('model_type', 'classification')
        training_config = params.get('training_config', {})
        
        self.logger.info(f"开始训练{model_type}模型")
        
        # 模拟模型训练过程
        epochs = training_config.get('epochs', 10)
        
        training_history = []
        
        for epoch in range(epochs):
            # 模拟训练一个epoch
            await asyncio.sleep(0.5)  # 模拟训练时间
            
            # 模拟训练指标
            loss = 1.0 - (epoch / epochs) * 0.8 + np.random.normal(0, 0.05)
            accuracy = (epoch / epochs) * 0.9 + np.random.normal(0, 0.02)
            
            training_history.append({
                "epoch": epoch + 1,
                "loss": max(0.1, loss),
                "accuracy": min(0.95, max(0.1, accuracy))
            })
            
            task.progress = (epoch + 1) / epochs
            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # 保存模型
        model_path = os.path.join(self.model_storage_path, f"{task.task_id}_model.json")
        model_info = {
            "model_type": model_type,
            "training_config": training_config,
            "training_history": training_history,
            "final_metrics": training_history[-1] if training_history else {},
            "created_at": datetime.now().isoformat()
        }
        
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        return {
            "model_path": model_path,
            "training_history": training_history,
            "final_loss": training_history[-1]["loss"] if training_history else 0,
            "final_accuracy": training_history[-1]["accuracy"] if training_history else 0,
            "training_time": epochs * 0.5  # 模拟训练时间
        }
    
    async def _execute_evaluation(self, task: ExecutionTask) -> Dict[str, Any]:
        """执行评估任务"""
        params = task.parameters
        model_path = params.get('model_path')
        test_data_path = params.get('test_data_path')
        evaluation_metrics = params.get('metrics', ['accuracy', 'precision', 'recall'])
        
        self.logger.info("开始模型评估")
        
        # 加载模型和测试数据
        with open(model_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 模拟评估过程
        evaluation_results = {}
        
        for i, metric in enumerate(evaluation_metrics):
            # 模拟计算评估指标
            if metric == 'accuracy':
                score = np.random.uniform(0.75, 0.95)
            elif metric == 'precision':
                score = np.random.uniform(0.70, 0.90)
            elif metric == 'recall':
                score = np.random.uniform(0.65, 0.85)
            elif metric == 'f1_score':
                precision = evaluation_results.get('precision', 0.8)
                recall = evaluation_results.get('recall', 0.75)
                score = 2 * (precision * recall) / (precision + recall)
            else:
                score = np.random.uniform(0.6, 0.9)
            
            evaluation_results[metric] = score
            task.progress = (i + 1) / len(evaluation_metrics)
            
            await asyncio.sleep(0.2)  # 模拟计算时间
        
        # 生成评估报告
        report = {
            "model_info": model_info,
            "test_data_size": len(test_data),
            "evaluation_metrics": evaluation_results,
            "evaluation_timestamp": datetime.now().isoformat(),
            "overall_score": np.mean(list(evaluation_results.values()))
        }
        
        return report
    
    async def _execute_deployment(self, task: ExecutionTask) -> Dict[str, Any]:
        """执行部署任务"""
        params = task.parameters
        model_path = params.get('model_path')
        deployment_config = params.get('deployment_config', {})
        
        self.logger.info("开始模型部署")
        
        # 模拟部署过程
        deployment_steps = [
            "准备部署环境",
            "加载模型文件",
            "配置API服务",
            "运行健康检查",
            "启动服务"
        ]
        
        for i, step in enumerate(deployment_steps):
            self.logger.info(f"执行部署步骤: {step}")
            await asyncio.sleep(0.5)  # 模拟部署时间
            task.progress = (i + 1) / len(deployment_steps)
        
        # 生成部署信息
        deployment_info = {
            "service_url": f"http://api.aipm.com/model/{task.task_id}",
            "deployment_id": f"deploy_{task.task_id}",
            "status": "active",
            "deployment_time": datetime.now().isoformat(),
            "configuration": deployment_config,
            "health_check_url": f"http://api.aipm.com/health/{task.task_id}"
        }
        
        return deployment_info
    
    async def _execute_optimization(self, task: ExecutionTask) -> Dict[str, Any]:
        """执行优化任务"""
        params = task.parameters
        target_metrics = params.get('target_metrics', [])
        optimization_config = params.get('optimization_config', {})
        
        self.logger.info("开始系统优化")
        
        optimization_results = {}
        
        for i, metric in enumerate(target_metrics):
            self.logger.info(f"优化指标: {metric}")
            
            # 模拟优化过程
            before_value = np.random.uniform(0.6, 0.8)
            improvement = np.random.uniform(0.1, 0.3)
            after_value = min(1.0, before_value + improvement)
            
            optimization_results[metric] = {
                "before": before_value,
                "after": after_value,
                "improvement": improvement,
                "improvement_percentage": (improvement / before_value) * 100
            }
            
            task.progress = (i + 1) / len(target_metrics)
            await asyncio.sleep(0.3)
        
        return {
            "optimization_results": optimization_results,
            "overall_improvement": np.mean([r["improvement_percentage"] for r in optimization_results.values()]),
            "optimization_timestamp": datetime.now().isoformat()
        }
    
    async def _execute_marketing(self, task: ExecutionTask) -> Dict[str, Any]:
        """执行营销任务"""
        params = task.parameters
        campaign_type = params.get('campaign_type', 'email')
        target_audience = params.get('target_audience', [])
        content = params.get('content', {})
        budget = params.get('budget', 0)
        
        self.logger.info(f"执行{campaign_type}营销活动")
        
        # 模拟营销活动执行
        if campaign_type == 'email':
            result = await self._execute_email_campaign(target_audience, content, budget)
        elif campaign_type == 'social':
            result = await self._execute_social_campaign(target_audience, content, budget)
        elif campaign_type == 'ads':
            result = await self._execute_ads_campaign(target_audience, content, budget)
        else:
            result = await self._execute_generic_campaign(target_audience, content, budget)
        
        task.progress = 1.0
        return result
    
    async def _execute_email_campaign(self, audience: List[str], content: Dict[str, Any], budget: float) -> Dict[str, Any]:
        """执行邮件营销活动"""
        # 模拟邮件发送
        total_emails = len(audience)
        sent_emails = int(total_emails * 0.95)  # 95%发送成功率
        opened_emails = int(sent_emails * 0.25)  # 25%打开率
        clicked_emails = int(opened_emails * 0.15)  # 15%点击率
        
        return {
            "campaign_type": "email",
            "total_audience": total_emails,
            "emails_sent": sent_emails,
            "emails_opened": opened_emails,
            "emails_clicked": clicked_emails,
            "open_rate": opened_emails / sent_emails if sent_emails > 0 else 0,
            "click_rate": clicked_emails / opened_emails if opened_emails > 0 else 0,
            "budget_used": budget * 0.8,
            "cost_per_click": (budget * 0.8) / clicked_emails if clicked_emails > 0 else 0
        }
    
    async def _execute_social_campaign(self, audience: List[str], content: Dict[str, Any], budget: float) -> Dict[str, Any]:
        """执行社交媒体营销活动"""
        reach = int(len(audience) * 2.5)  # 社交媒体扩散效应
        impressions = int(reach * 1.8)
        engagements = int(impressions * 0.05)
        conversions = int(engagements * 0.10)
        
        return {
            "campaign_type": "social",
            "reach": reach,
            "impressions": impressions,
            "engagements": engagements,
            "conversions": conversions,
            "engagement_rate": engagements / impressions if impressions > 0 else 0,
            "conversion_rate": conversions / engagements if engagements > 0 else 0,
            "budget_used": budget * 0.9,
            "cost_per_conversion": (budget * 0.9) / conversions if conversions > 0 else 0
        }
    
    async def _execute_ads_campaign(self, audience: List[str], content: Dict[str, Any], budget: float) -> Dict[str, Any]:
        """执行广告营销活动"""
        impressions = int(budget * 100)  # 假设每元预算100次展示
        clicks = int(impressions * 0.02)  # 2%点击率
        conversions = int(clicks * 0.05)  # 5%转化率
        
        return {
            "campaign_type": "ads",
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "click_rate": clicks / impressions if impressions > 0 else 0,
            "conversion_rate": conversions / clicks if clicks > 0 else 0,
            "budget_used": budget,
            "cost_per_click": budget / clicks if clicks > 0 else 0,
            "cost_per_conversion": budget / conversions if conversions > 0 else 0
        }
    
    async def _execute_generic_campaign(self, audience: List[str], content: Dict[str, Any], budget: float) -> Dict[str, Any]:
        """执行通用营销活动"""
        return {
            "campaign_type": "generic",
            "audience_size": len(audience),
            "budget_used": budget * 0.85,
            "estimated_reach": int(len(audience) * 1.5),
            "estimated_conversions": int(len(audience) * 0.03)
        }
    
    def build_dataset(self, dataset_spec: DatasetSpec) -> ExecutionTask:
        """构建数据集"""
        task = ExecutionTask(
            task_id=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"构建数据集: {dataset_spec.dataset_name}",
            task_type=TaskType.DATA_COLLECTION,
            description=f"构建{dataset_spec.dataset_name}数据集",
            parameters={
                "data_sources": dataset_spec.data_sources,
                "target_size": dataset_spec.target_size,
                "quality_requirements": dataset_spec.quality_requirements,
                "annotation_rules": dataset_spec.annotation_rules
            }
        )
        
        self.task_queue.append(task)
        return task
    
    def create_evaluation_task(self, model_path: str, test_data_path: str, metrics: List[str]) -> ExecutionTask:
        """创建评估任务"""
        task = ExecutionTask(
            task_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="模型评估",
            task_type=TaskType.EVALUATION,
            description="评估模型性能",
            parameters={
                "model_path": model_path,
                "test_data_path": test_data_path,
                "metrics": metrics
            }
        )
        
        self.task_queue.append(task)
        return task
    
    def create_optimization_task(self, target_metrics: List[str], config: Dict[str, Any]) -> ExecutionTask:
        """创建优化任务"""
        task = ExecutionTask(
            task_id=f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="系统优化",
            task_type=TaskType.OPTIMIZATION,
            description="优化系统性能",
            parameters={
                "target_metrics": target_metrics,
                "optimization_config": config
            }
        )
        
        self.task_queue.append(task)
        return task
    
    def launch_marketing_campaign(self, campaign: Campaign) -> ExecutionTask:
        """启动营销活动"""
        task = ExecutionTask(
            task_id=f"campaign_{campaign.campaign_id}",
            name=f"营销活动: {campaign.name}",
            task_type=TaskType.MARKETING,
            description=f"执行{campaign.type}营销活动",
            parameters={
                "campaign_type": campaign.type,
                "target_audience": campaign.target_audience,
                "content": campaign.content,
                "budget": campaign.budget
            }
        )
        
        self.task_queue.append(task)
        return task
    
    async def process_task_queue(self):
        """处理任务队列"""
        while self.task_queue and len(self.running_tasks) < self.max_concurrent_tasks:
            task = self.task_queue.pop(0)
            asyncio.create_task(self.execute_task(task))
    
    def get_task_status(self, task_id: str) -> Optional[ExecutionTask]:
        """获取任务状态"""
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            for task in self.task_queue:
                if task.task_id == task_id:
                    return task
        return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行模块运行摘要"""
        return {
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len([t for t in self.completed_tasks.values() if t.status == TaskStatus.FAILED]),
            "average_completion_time": self._calculate_average_completion_time(),
            "resource_utilization": len(self.running_tasks) / self.max_concurrent_tasks
        }
    
    def _calculate_average_completion_time(self) -> float:
        """计算平均完成时间"""
        completed_tasks = [t for t in self.completed_tasks.values() 
                          if t.status == TaskStatus.COMPLETED and t.start_time and t.end_time]
        
        if not completed_tasks:
            return 0.0
        
        total_time = sum([(t.end_time - t.start_time).total_seconds() for t in completed_tasks])
        return total_time / len(completed_tasks)