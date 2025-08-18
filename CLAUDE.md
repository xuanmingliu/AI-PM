# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-Researcher is a comprehensive autonomous scientific research system that automates the complete research lifecycle from literature review to paper generation. The system operates as a multi-agent framework with specialized agents for different research tasks.

## Key Architecture Components

### 1. Research Agent Framework (`research_agent/`)
- **Main entry points**: `run_infer_idea.py` (reference-based ideation), `run_infer_plan.py` (detailed idea implementation)
- **Core system**: `inno/` directory contains the main agent architecture
- **Agent types**:
  - `idea_agent.py`: Generates research ideas from references
  - `plan_agent.py`: Creates implementation plans
  - `survey_agent.py`: Conducts literature reviews
  - `ml_agent.py`: Implements machine learning solutions
  - `exp_analyser.py`: Analyzes experimental results
  - `judge_agent.py`: Evaluates research quality

### 2. Paper Writing System (`paper_agent/`)
- **Main module**: `writing.py` - Automated academic paper generation
- **Section composers**: Individual modules for different paper sections (abstract, introduction, methodology, experiments, conclusion)
- **Template system**: Domain-specific writing templates in subdirectories (gnn/, vq/, etc.)

### 3. Environment Management
- **Docker integration**: `docker/` directory with containerized execution environment
- **Environment variables**: Configured via `.env` file with support for multiple LLM providers
- **Web interface**: `web_ai_researcher.py` provides Gradio-based GUI

## Common Development Commands

### Running Research Agents

#### Level 1 Tasks (Detailed Idea Description):
```bash
cd research_agent
python run_infer_plan.py \
  --instance_path ../benchmark/final/${category}/${instance_id}.json \
  --container_name paper_eval \
  --task_level task1 \
  --model claude-3-5-sonnet-20241022 \
  --workplace_name workplace \
  --cache_path cache \
  --port 12372 \
  --max_iter_times 0 \
  --category ${category}
```

#### Level 2 Tasks (Reference-Based Ideation):
```bash
cd research_agent
python run_infer_idea.py \
  --instance_path ../benchmark/final/${category}/${instance_id}.json \
  --container_name paper_eval \
  --model claude-3-5-sonnet-20241022 \
  --workplace_name workplace \
  --cache_path cache \
  --port 12372 \
  --max_iter_times 0 \
  --category ${category}
```

### Paper Generation
```bash
cd paper_agent
python writing.py --research_field ${research_field} --instance_id ${instance_id}
```

### Web Interface
```bash
python web_ai_researcher.py
```
Launches on port 7039 with Gradio interface for easier interaction.

### Docker Environment
```bash
# Pull the research environment image
docker pull tjbtech1/airesearcher:v1

# Or build from source
cd docker && docker build -t tjbtech1/airesearcher:v1 .
```

## Environment Configuration

### Required Environment Variables
Create `.env` file with:
```bash
# LLM Configuration
COMPLETION_MODEL=claude-3-5-sonnet-20241022
CHEEP_MODEL=claude-3-5-haiku-20241022

# Container Configuration
BASE_IMAGES=tjbtech1/airesearcher:v1
DOCKER_WORKPLACE_NAME=workplace_paper
CONTAINER_NAME=paper_eval
WORKPLACE_NAME=workplace
CACHE_PATH=cache
PORT=7020
GPUS='"device=0"'

# Task Configuration
CATEGORY=vq  # Options: vq, gnn, diffu_flow, reasoning, recommendation
INSTANCE_ID=one_layer_vq
TASK_LEVEL=task1  # task1 or task2
MAX_ITER_TIMES=0

# API Keys (configure as needed)
OPENAI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
GITHUB_AI_TOKEN=your_token
```

## Project Structure Understanding

### Research Domains
The system supports 5 research categories:
- `vq`: Vector Quantization methods
- `gnn`: Graph Neural Networks
- `diffu_flow`: Diffusion and Flow Matching
- `reasoning`: Reasoning systems
- `recommendation`: Recommendation systems

### Memory and Tool Systems
- **Memory management**: `research_agent/inno/memory/` - Handles paper, code, and RAG memories
- **Tool ecosystem**: `research_agent/inno/tools/` - Specialized tools for research tasks including arXiv search, code search, and web tools
- **Environment abstraction**: `research_agent/inno/environment/` - Docker and browser environment management

### Benchmark System
- **Benchmark data**: `benchmark/final/` contains evaluation datasets
- **Collection tools**: `benchmark_collection/` for creating new benchmarks
- **Example outputs**: `examples/` directory shows generated research projects

## Development Guidelines

### Working with Research Agents
1. Always use the appropriate environment variables for your research domain
2. The system expects Docker to be available for code execution
3. LLM models can be configured through environment variables
4. Use the web interface for interactive development and debugging

### Adding New Research Domains
1. Create templates in `paper_agent/{domain}/writing_templates/`
2. Add benchmark data in `benchmark/final/{domain}/`
3. Update metaprompt.py files for new domain processing

### Memory and State Management
- The system maintains state through `global_state.py`
- Research sessions are persistent through the workflow system
- Tool memory helps avoid redundant operations

## Testing and Validation
- Use benchmark datasets in `benchmark/final/` for evaluation
- Each research domain has specific evaluation metrics
- Generated papers and code are automatically validated

## Important Notes
- The system requires significant computational resources for full operation
- Docker environment is essential for secure code execution
- LLM API costs can be substantial during full research runs
- Generated research outputs are stored in domain-specific directories