# Memgraph AI Toolkit

[![PyPI - memgraph-toolbox](https://img.shields.io/pypi/v/memgraph-toolbox?label=memgraph-toolbox)](https://pypi.org/project/memgraph-toolbox/)
[![PyPI - langchain-memgraph](https://img.shields.io/pypi/v/langchain-memgraph?label=langchain-memgraph)](https://pypi.org/project/langchain-memgraph/)
[![PyPI - mcp-memgraph](https://img.shields.io/pypi/v/mcp-memgraph?label=mcp-memgraph)](https://pypi.org/project/mcp-memgraph/)
[![PyPI - unstructured2graph](https://img.shields.io/pypi/v/unstructured2graph?label=unstructured2graph)](https://pypi.org/project/unstructured2graph/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289da)](https://discord.gg/memgraph)

Build powerful AI applications with graph-powered RAG using [Memgraph](https://memgraph.com/). This toolkit provides everything you need to integrate knowledge graphs into your GenAI workflows.

## 🚀 Quick Setup

### Start Memgraph

```bash
docker run -p 7687:7687 \
  --name memgraph \
  memgraph/memgraph-mage:latest \
  --schema-info-enabled=true
```

### Install Packages

```bash
# Core toolbox
pip install memgraph-toolbox

# LangChain integration
pip install langchain-memgraph

# MCP server
pip install mcp-memgraph

# Unstructured to Graph
pip install unstructured2graph
```

---

## 📚 Usage Examples

### unstructured2graph - Build Knowledge Graphs from Documents

Transform PDFs, URLs, and documents into queryable knowledge graphs:

```python
import asyncio
from memgraph_toolbox.api.memgraph import Memgraph
from lightrag_memgraph import MemgraphLightRAGWrapper
from unstructured2graph import from_unstructured, create_index

async def main():
    memgraph = Memgraph()
    create_index(memgraph, "Chunk", "hash")

    lightrag = MemgraphLightRAGWrapper()
    await lightrag.initialize(working_dir="./lightrag_storage")

    # Ingest documents from URLs or local files
    await from_unstructured(
        sources=["https://example.com/doc.pdf", "./local_file.md"],
        memgraph=memgraph,
        lightrag_wrapper=lightrag,
        link_chunks=True,
    )
    await lightrag.afinalize()

asyncio.run(main())
```

👉 [Full Documentation](https://memgraph.com/docs/ai-ecosystem/unstructured2graph) | [Examples](/unstructured2graph/examples/)

---

### langchain-memgraph - LangChain Integration

#### Natural Language Queries with MemgraphQAChain

```python
from langchain_memgraph.graphs.memgraph import MemgraphLangChain
from langchain_memgraph.chains.graph_qa import MemgraphQAChain
from langchain_openai import ChatOpenAI

graph = MemgraphLangChain(url="bolt://localhost:7687")

chain = MemgraphQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    model_name="gpt-4-turbo",
    allow_dangerous_requests=True,
)

response = chain.invoke("Who are the main characters in the dataset?")
print(response["result"])
```

#### Build Agents with MemgraphToolkit

```python
from langchain.chat_models import init_chat_model
from langchain_memgraph import MemgraphToolkit
from langchain_memgraph.graphs.memgraph import MemgraphLangChain
from langgraph.prebuilt import create_react_agent

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
db = MemgraphLangChain(url="bolt://localhost:7687")
toolkit = MemgraphToolkit(db=db, llm=llm)

agent = create_react_agent(llm, toolkit.get_tools())
events = agent.stream({"messages": [("user", "Find all Person nodes")]})
```

👉 [Full Documentation](https://memgraph.com/docs/ai-ecosystem/integrations#langchain) | [Notebooks](/integrations/langchain-memgraph/docs/)

---

### mcp-memgraph - Model Context Protocol Server

Expose Memgraph to LLMs via MCP. Run with Docker:

```bash
# HTTP mode (recommended)
docker run --rm -p 8000:8000 mcp-memgraph:latest

# Stdio mode for MCP clients
docker run --rm -i -e MCP_TRANSPORT=stdio mcp-memgraph:latest
```

**Available Tools:**

| Tool                    | Description                |
| ----------------------- | -------------------------- |
| `run_query`             | Execute Cypher queries     |
| `get_schema`            | Fetch graph schema         |
| `get_page_rank`         | Compute PageRank scores    |
| `get_node_neighborhood` | Find nodes within distance |
| `search_node_vectors`   | Vector similarity search   |

👉 [Full Documentation](https://memgraph.com/docs/ai-ecosystem/integrations#model-context-protocol-mcp)

---

### sql2graph Agent - Automated Database Migration

Migrate from MySQL/PostgreSQL to Memgraph with AI assistance:

```bash
cd agents/sql2graph
uv run main.py
```

👉 [Full Documentation](https://memgraph.com/docs/ai-ecosystem/agents#sql2graph-agent)

---

## 🛠️ Packages Overview

| Package                                                 | Description                  | Install                          |
| ------------------------------------------------------- | ---------------------------- | -------------------------------- |
| [memgraph-toolbox](/memgraph-toolbox/)                  | Core utilities for Memgraph  | `pip install memgraph-toolbox`   |
| [langchain-memgraph](/integrations/langchain-memgraph/) | LangChain tools and chains   | `pip install langchain-memgraph` |
| [mcp-memgraph](/integrations/mcp-memgraph/)             | MCP server for LLMs          | `pip install mcp-memgraph`       |
| [unstructured2graph](/unstructured2graph/)              | Document to graph conversion | `pip install unstructured2graph` |
| [sql2graph](/agents/sql2graph/)                         | Database migration agent     | See docs                         |

---

## ❓ FAQ

**Which databases are supported?**  
Memgraph is the primary target. The sql2graph agent supports MySQL and PostgreSQL as source databases.

**Do I need an LLM API key?**  
Yes, for features like entity extraction (unstructured2graph) and natural language queries (langchain-memgraph).

**Can I use local LLMs?**  
Yes! LangChain integration supports any LangChain-compatible model, including Ollama.

---

## 🤝 Community

- [GitHub Issues](https://github.com/memgraph/ai-toolkit/issues)
- [Discord](https://discord.gg/memgraph)
- [Documentation](https://memgraph.com/docs/ai-ecosystem)

⭐ If you find this toolkit helpful, please star the repository!

---

## 🧪 Developing Locally

You can build and test each package directly from your repo.

### Core tests

```bash
uv pip install -e memgraph-toolbox[test]
pytest -s memgraph-toolbox/src/memgraph_toolbox/tests
```

### LangChain integration tests

Create a `.env` file with your `OPENAI_API_KEY`, as the tests depend on LLM calls:

```bash
uv pip install -e integrations/langchain-memgraph[test]
pytest -s integrations/langchain-memgraph/tests
```

### MCP integration tests

```bash
uv pip install -e integrations/mcp-memgraph[test]
pytest -s integrations/mcp-memgraph/tests
```

### Agent integration tests

```bash
uv pip install -e integrations/agents[test]
pytest -s integrations/agents/tests
```

To run a complete migration workflow with the agent:

```bash
cd integrations/agents
uv run main.py
```

**Note:** The agent requires both MySQL and Memgraph connections. Set up your environment variables in `.env` based on `.env.example`.

If you are running any test on macOS in zsh, add `""` to the command:

```bash
uv pip install -e memgraph-toolbox"[test]"
```
