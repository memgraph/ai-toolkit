import os
from getpass import getpass

import pytest
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from langchain_memgraph import MemgraphToolkit
from memgraph_toolbox.api.memgraph import Memgraph

# Load environment variables
load_dotenv()


@pytest.fixture(scope="module")
def memgraph_connection():
    """Setup Memgraph connection fixture."""
    url = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
    username = os.getenv("MEMGRAPH_USER", "")
    password = os.getenv("MEMGRAPH_PASSWORD", "")

    graph = Memgraph(url=url, username=username, password=password)
    yield graph

    # Cleanup: clear the database after test
    graph.query("MATCH (n) DETACH DELETE n")


@pytest.fixture(scope="module")
def memgraph_agent():
    url = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
    username = os.getenv("MEMGRAPH_USER", "")
    password = os.getenv("MEMGRAPH_PASSWORD", "")

    """Set up Memgraph agent with React pattern."""
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("Enter API key for OpenAI: ")

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    db = Memgraph(url=url, username=username, password=password)
    toolkit = MemgraphToolkit(db=db, llm=llm)

    agent_executor = create_react_agent(
        llm,
        toolkit.get_tools(),
        prompt="You will get a cypher query, try to execute it on the Memgraph database.",
    )
    return agent_executor


def test_seed_graph(memgraph_connection):
    """Test to seed the graph with Jon Snow node"""
    query = """
       CREATE (c:Character {name: 'Jon Snow', house: 'Stark', title: 'King in the North'})
    """
    memgraph_connection.query(query)

    # Verify nodes exist
    game_exists = memgraph_connection.query(
        """MATCH (c:Character {name: 'Jon Snow'}) RETURN c"""
    )
    assert len(game_exists) > 0, "Game node not created!"


def test_memgraph_agent(memgraph_agent):
    """Test Memgraph agent executes a Cypher query correctly."""
    example_query = "MATCH (n) WHERE n.name = 'Jon Snow' RETURN n"
    events = memgraph_agent.stream(
        {"messages": [("user", example_query)]},
        stream_mode="values",
    )

    last_event = None
    for event in events:
        last_event = event
        event["messages"][-1].pretty_print()

    assert last_event, "Agent did not return any results!"
    assert "Jon Snow" in str(
        last_event["messages"][-1]
    ), "Expected 'Jon Snow' in the final result!"
