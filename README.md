# 🧰 AI Agent Service Toolkit for smolagents with history handlings

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml) [![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit) [![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

A full toolkit for running an AI agent service built with Smolagents, FastAPI and Streamlit.

It includes a [Smolagents](https://github.com/huggingface/smolagents) agent, a [FastAPI](https://fastapi.tiangolo.com/) service to serve it, a client to interact with the service, and a [Streamlit](https://streamlit.io/) app that uses the client to provide a chat interface. Data structures and settings are built with [Pydantic](https://github.com/pydantic/pydantic).

This project offers a template for you to easily build and run your own agents using the LangGraph framework. It demonstrates a complete setup from agent definition to user interface, making it easier to get started with LangGraph-based projects by providing a full, robust toolkit.

This application is inspired by the [Agent Service Toolkit](https://github.com/JoshuaC215/agent-service-toolkit) framework, providing a robust foundation with LangGraph, FastAPI, and Streamlit.


### Quickstart

Run directly in python

```sh
# At least one LLM API key is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# uv is recommended but "pip install ." also works
pip install uv
uv sync --frozen
# "uv sync" creates .venv automatically
source .venv/bin/activate
python src/run_service.py

# In another shell
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

### Key Files

The repository is structured as follows:

- `src/agents/`: Defines several agents with different capabilities
- `src/schema/`: Defines the protocol schema
- `src/core/`: Core modules including LLM definition and settings
- `src/service/service.py`: FastAPI service to serve the agents
- `src/client/client.py`: Client to interact with the agent service
- `src/streamlit_app.py`: Streamlit app providing a chat interface
- `tests/`: Unit and integration tests

## Setup and Usage

1. Clone the repository:

   ```sh
   git clone https://github.com/YannH14/streamlit_with_history.git
   cd agent-service-toolkit
   ```

2. Set up environment variables:
   Create a `.env` file in the root directory. At least one LLM API key or configuration is required. See the [`.env.example` file](./.env.example) for a full list of available environment variables, including a variety of model provider API keys, header-based authentication, LangSmith tracing, testing and development modes, and OpenWeatherMap API key.

3. You can now run the agent service and the Streamlit app locally, either with Docker or just using Python. 

### Additional setup for specific AI providers

- [Setting up Ollama](docs/Ollama.md)

### Building or customizing your own agent

To customize the agent for your own use case:

1. Add your new agent to the `src/agents` directory. You can copy `research_assistant.py` or `chatbot.py` and modify it to change the agent's behavior and tools.
1. Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`. Your agent can be called by `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
1. Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

### Local development without Docker

You can also run the agent service and the Streamlit app locally without Docker, just using a Python virtual environment.

1. Create a virtual environment and install dependencies:

   ```sh
   pip install uv
   uv sync --frozen
   source .venv/bin/activate
   ```

2. Run the FastAPI server:

   ```sh
   python src/run_service.py
   ```

3. In a separate terminal, run the Streamlit app:

   ```sh
   streamlit run src/streamlit_app.py
   ```

4. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
