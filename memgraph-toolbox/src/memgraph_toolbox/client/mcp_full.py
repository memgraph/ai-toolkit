import asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import ElicitResult
from rich.console import Console

# https://github.com/prompt-toolkit/python-prompt-toolkit is also very interesting.
import typer
from litellm import acompletion
from tools import atool_calls

LLM_MODEL = "openai/gpt-4o"
app = typer.Typer(help="MCP Terminal Client (Streaming HTTP)")
console = Console()


async def describe_mcp_server(client: Client):
    tools = await client.list_tools()
    prompts = await client.list_prompts()
    tool_names = [tool.name for tool in tools]
    console.print(f"[bold green]Available Tools:[/bold green] {tool_names}")
    prompt_names = [prompt.name for prompt in prompts]
    console.print(f"[bold green]Available Prompts:[/bold green] {prompt_names}")
    console.print(f"Getting all the prompts...")
    for prompt_name in prompt_names:
        result = await client.get_prompt(prompt_name)
        console.print(f"  Prompt: [yellow]{prompt_name}[/yellow]")
        for message in result.messages:
            console.print(f"    Role: [blue]{message.role}[/blue]")
            console.print(f"    Content Text:")
            console.print(f"    [blue]{message.content.text}[/blue]")
    return tools, prompts


async def my_elicitation_handler(message: str, response_type: type, params, context):
    """
    User adds additonal feedback to the failed query.
    Actions are: accept, decline, cancel.
    """
    console.print(f"[bold yellow]Server asks:[/bold yellow] {message}")
    console.print(f"[dim]Response type: {response_type}[/dim]")
    # Use an editor to get user input
    content = typer.edit(message)
    # Decline
    if not content or content.strip() == "":
        console.print("[red]Declining elicitation request[/red]")
        return ElicitResult(action="decline")
    # Accept
    console.print(f"[green]Sending response:[/green] {content}")
    # NOTE: None of the below pass the pydantic validation for some reason.
    #   return response_type(**{"data": content})
    #   return ElicitResult(action="accept", content=response_type(**{"data": content}))
    #   return ElicitResult(action="accept", content={"data": content})
    return {"data": content}


async def my_sampling_handler(message, params, context) -> str:
    """
    Iterate in a loop and try to generate valid responses.
    """
    return "Generated sample response"


def convert_mcp_tools_to_litellm(mcp_tools):
    """Convert MCP tools to litellm format"""
    litellm_tools = []
    for tool in mcp_tools:
        litellm_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "parameters": tool.inputSchema or {},
            },
        }
        litellm_tools.append(litellm_tool)
    return litellm_tools


@app.command()
def chat(
    base_url: str = typer.Option(
        "http://localhost:8000/mcp", help="MCP server base URL"
    ),
    interactivity_type: str = typer.Option(
        "llm-use",  # TODO: There should be one-shot, iterate, etc...
        help="Interactivity type (llm-use, no-llm-use)",
        case_sensitive=False,
        metavar="INTERACTIVITY_TYPE",
    ),
):
    config = {
        "mcpServers": {
            "memgraph": {"url": base_url},
        }
    }
    client = Client(
        config,
        elicitation_handler=my_elicitation_handler,
        sampling_handler=my_sampling_handler,
    )
    print(f"Interactivity type: {interactivity_type}")

    async def _run():
        async with client:
            tools, prompts = await describe_mcp_server(client)
            litellm_tools = convert_mcp_tools_to_litellm(tools)
            # TODO(gitbuda): Fill this in from the templates.
            chat_messages = [
                {
                    "role": "user",
                    "content": """
                        You are an expert in Cypher and Memgraph graph databases.
                        You have access to the following tools:
                        - `run_query`: Run a Cypher query on the Memgraph database.
                                       Input is a Cypher query string. Output is a list of records or an error message.
                        - `get_schema`: Get the schema information of the Memgraph database. Input is empty.
                        You should first call `get_schema` to understand the database structure,
                        then use `run_query` to execute Cypher queries.
                        IMPORTANT: Do not call run_query if you don't have the schema information.
                    """,
                }
            ]

            while True:
                user_input = input("\n> ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break
                # TODO(gitbuda): Expose prompts user can pick from.

                if interactivity_type == "llm-use":
                    console.print(f"\n[bold yellow]Prompt:[/bold yellow] {user_input}")
                    # Convert MCP tools to litellm format
                    chat_messages.append({"role": "user", "content": user_input})
                    pick_tool_resp = await acompletion(
                        LLM_MODEL, chat_messages, tools=litellm_tools
                    )
                    # print(pick_tool_resp)
                    # TODO(gitbuda): Here should be a prompt to allow tool call.
                    tool_call_msg = await atool_calls(
                        pick_tool_resp, chat_messages, client
                    )
                    print(tool_call_msg)
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": """Based on the given data, generate the run_query call.""",
                        }
                    )
                    the_query_resp = await acompletion(
                        LLM_MODEL, tool_call_msg, tools=litellm_tools
                    )
                    print(the_query_resp)
                    tool_call_msg = await atool_calls(
                        the_query_resp, chat_messages, client
                    )
                    mg_data = tool_call_msg[-1]["content"]
                    console.print(
                        f"\n[bold yellow]Query Result:[/bold yellow] {mg_data}"
                    )

                if interactivity_type == "no-llm-use":
                    query_result = await client.call_tool(
                        "run_query", {"query": user_input}
                    )
                    console.print(
                        f"\n[bold yellow]Query Result:[/bold yellow] {query_result.data}"
                    )
                    # Print any additional content if available
                    for content in query_result.content:
                        if hasattr(content, "text"):
                            console.print(
                                f"[bold blue]Content:[/bold blue] {content.text}"
                            )
                        elif hasattr(content, "data"):
                            console.print(
                                f"[bold blue]Data:[/bold blue] {content.data}"
                            )

                # TODO(gitbuda): If something fails or empty, don't take user input -> rerun the whole thing again (make sure the train history is there).

    asyncio.run(_run())


if __name__ == "__main__":
    app()
