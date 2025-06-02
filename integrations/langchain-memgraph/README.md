# 🦜️🔗 LangChain Memgraph

This package contains the LangChain integration with [Memgraph](https://memgraph.com/) graph database.

## 📦 Installation

In order to start running the examples or tests you need to install the LangChain integration.

You can do it via pip:

```bash
pip install -U langchain-memgraph
```

Before running the examples below, make sure to start Memgraph, you can do it via following command:

```bash
docker run -p 7687:7687 \
  --name memgraph \
  memgraph/memgraph-mage:latest \
  --schema-info-enabled=true
```

## 💻 Integration features

### Memgraph

The `Memgraph` class is a wrapper around the database client that supports the
query operation.

```python
import os
from langchain_memgraph.graphs.memgraph import MemgraphLangChain

url = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
username = os.getenv("MEMGRAPH_USER", "")
password = os.getenv("MEMGRAPH_PASSWORD", "")

graph = MemgraphLangChain(url=url, username=username, password=password)
results = graph.query("MATCH (n) RETURN n LIMIT 1")
print(results)
```

### MemgraphQAChain

The `MemgraphQAChain` class enables natural language interactions with a Memgraph database.
It uses an LLM and the database's schema to translate a user's question into a Cypher query, which is executed against the database.
The resulting data is then sent along with the user's question to the LLM to generate a natural language response.

For the example below you need to install an extra dependency the `lanchain_openai`, you can do it by running:

```bash
pip install lanchain_openai
```

```python
import os
from langchain_memgraph.graphs.memgraph import MemgraphLangChain
from langchain_memgraph.chains.graph_qa import MemgraphQAChain
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
url = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
username = os.getenv("MEMGRAPH_USER", "")
password = os.getenv("MEMGRAPH_PASSWORD", "")

graph = MemgraphLangChain(url=url, username=username, password=password, refresh_schema=False)

chain = MemgraphQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    model_name="gpt-4-turbo",
    allow_dangerous_requests=True,
)
response = chain.invoke("Is there a any Person node in the dataset?")
result = response["result"].lower()
print(result)
```

### Memgraph toolkit

The `MemgraphToolkit` contains different tools agents can leverage to perform specific tasks the user has given them. Toolkit
needs a database object and LLM access since different tools leverage different operations.

Currently supported tools:

1. **QueryMemgraphTool** - Basic Cypher query execution tool

```python
import os
import pytest
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_memgraph import MemgraphToolkit
from langchain_memgraph.graphs.memgraph import MemgraphLangChain
from langgraph.prebuilt import create_react_agent

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
url = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
username = os.getenv("MEMGRAPH_USER", "")
password = os.getenv("MEMGRAPH_PASSWORD", "")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

db = MemgraphLangChain(url=url, username=username, password=password)
toolkit = MemgraphToolkit(db=db, llm=llm)

agent_executor = create_react_agent(
    llm, toolkit.get_tools(), prompt="You will get a cypher query, try to execute it on the Memgraph database."
)

example_query = "MATCH (n) WHERE n.name = 'Jon Snow' RETURN n"
events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)

last_event = None
for event in events:
    last_event = event
    event["messages"][-1].pretty_print()

print(last_event)

```

## 🧪 Test

Install the test dependencies to run the tests:

1. Install dependencies

```bash
poetry install --with test,test_integration
```

2. Start Memgraph in the background.
3. Create an `.env` file that points to Memgraph and OpenAI API

```
MEMGRAPH_URL=bolt://localhost:7687
MEMGRAPH_USER=
MEMGRAPH_PASSWORD=
OPENAI_API_KEY=your_openai_api_key
```

### Run tests

Run the unit tests using:

```bash
make tests
```

Run the integration test using:

```bash
make integration_tests
```

## 🧹 Code Formatting and Linting

Install the `codespell`, `lint`, and typing dependencies to lint and format your code:

```bash
poetry install --with codespell,lint,typing
```

To format your code, run:

```bash
make format
```

To lint it, run:

```bash
make lint
```
