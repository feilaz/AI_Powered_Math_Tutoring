# AI-Powered Math Tutoring Platform

An AI-powered mathematics tutoring system designed to provide personalized, adaptive, and guided learning experiences. This platform implements the multi-agent architecture detailed in our research paper:
> **AI-Powered Math Tutoring: Platform for Personalized and Adaptive Education** (Anonymous Authors, *Submitted for Anonymous Review*).

## 🌟 Key Features

- **Multi-Agent Architecture**: Built with LangGraph for robust agent collaboration
- **Intelligent Content Retrieval**: GraphRAG-powered textbook access for accurate information recall
- **Conversational Tutoring**: Natural dialogue-based learning experience
- **Dual Memory System**: Long-term student profiles and working memory for session context
- **Structured Learning Paths**: DAG-based course planning for coherent educational progression
- **Automated Task Generation**: Dynamic creation of problems tailored to student needs
- **Symbolic Math Integration**: Built-in mathematical solving capabilities via SymPy
- **Visual Learning Support**: Function plotting and visualization tools with Matplotlib

## 📋 System Architecture

Our platform uses a sophisticated multi-agent system where specialized agents collaborate to deliver a comprehensive tutoring experience:

| Agent | Role |
|-------|------|
| **TutorAgent** | Primary student interface, coordinates other agents |
| **MemoryDispatcherAgent** | Routes memory operations to appropriate memory systems |
| **LongTermMemoryAgent** | Maintains persistent student profiles and learning histories |
| **WorkingMemoryAgent** | Handles session-specific context and short-term information |
| **TaskCreationAgent** | Generates customized practice problems and assessments |
| **Course Creation Agents** | Collaborative team that designs structured learning paths |

## 🚀 Installation

### Setup
1. Configure the `config.yaml` file:
```yaml
# API Keys
openai_api_key: "<your key>"
langsmith_api_key: "<your key>"  # optional
api_base_url: "<your url>"
llm_model: "<your model>"
small_llm_model: "<your smaller model>"

# LangSmith Configuration
langsmith_configuration:
  LANGCHAIN_TRACING_V2: True
  LANGCHAIN_PROJECT: "Tutoring Project"  # Change project name if needed
```

2. Initialize GraphRAG:
   - Follow the instructions at https://microsoft.github.io/graphrag/get_started/
   - Place the generated files in the `rag` folder

3. Run the application:


## 📁 Project Structure

```
ai-math-tutor/
├── code/
│   ├── main.py                # Application entry point
│   ├── agents.py              # Core agent definitions
│   ├── tools.py               # SymPy and Matplotlib integrations
│   ├── __init__.py
│   ├── course/                # Course creation & management
│   │   ├── course_agents.py   # Course creation agent team
│   │   ├── course_graph.py    # DAG-based course structure
│   │   └── __init__.py
│   ├── prompts/               # LLM prompt templates
│   ├── rag/                   # Knowledge retrieval system
│   │   ├── graphrag.py        # GraphRAG implementation
│   │   ├── settings.yaml      # GraphRAG configuration
│   │   └── __init__.py
│   └── config/                # System configuration
│       ├── config.py          # Configuration logic
│       ├── config.yaml        # Default configuration
│       └── __init__.py
├── data/                      # Educational content
└── README.md                  # This file
```

## 📚 Citation

If you use this platform in your research, please cite our paper:

```bibtex
@article{anonymous2025aimath,
  title={AI-Powered Math Tutoring: Platform for Personalized and Adaptive Education},
  author={Anonymous},
  journal={Under Review},
  year={2025}
}
```
