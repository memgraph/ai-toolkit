import httpx
import asyncio
import orjson
from rich.console import Console
from rich.markdown import Markdown

# https://github.com/prompt-toolkit/python-prompt-toolkit is also very interesting.
import typer
import random

from fastmcp import Client
from fastmcp.exceptions import ToolError

# from fastmcp.client.elicitation import ElicitResult
# from fastmcp.types import ElicitResult
from mcp.types import ElicitResult


console = Console()


async def my_elicitation_handler(message: str, response_type: type, params, context):
    console.print(f"[bold yellow]Server asks:[/bold yellow] {message}")
    console.print(f"[dim]Response type: {response_type}[/dim]")
    console.print(f"[dim]Params: {params}[/dim]")
    console.print(f"[dim]Context: {context}[/dim]")
    # Use typer.edit to get user input
    content = typer.edit(message)
    # Decline
    if not content or content.strip() == "":
        console.print("[red]Declining elicitation request[/red]")
        return ElicitResult(action="decline")
    # Accept
    console.print(f"[green]Sending response:[/green] {content}")
    # NOTE: None of the below pass the pydantic validation for some reason.
    #   return response_type(**{"data": content})
    #  return ElicitResult(action="accept", content=response_type(**{"data": content}))
    # return ElicitResult(action="accept", content={"data": content})
    return {"data": content}


def _elicit_prompt(user_input: str, strategy: str = "socratic") -> str:
    if strategy == "socratic":
        return f"Let's reason step by step. {user_input}"
    elif strategy == "decompose":
        return f"Break this problem into smaller parts before solving:\n{user_input}"
    elif strategy == "reflective":
        return f"Consider alternative interpretations or solutions:\n{user_input}"
    return user_input


async def _sample_responses(client: Client, prompt: str, n: int = 3):
    return []
    samples = []
    for i in range(n):
        console.print(f"\n[bold yellow]Sample {i+1}[/bold yellow]:")
        modified_prompt = (
            f"{prompt}\n\n(Variation #{i+1}, random seed={random.randint(0,9999)})"
        )
        # TODO: Implement the interaction await client.stream_chat(modified_prompt)
        samples.append(modified_prompt)
    return samples


app = typer.Typer(help="MCP Terminal Client (Streaming HTTP)")


@app.command()
def chat(
    base_url: str = typer.Option(
        "http://localhost:8000/mcp", help="MCP server base URL"
    ),
    elicitation: str = typer.Option(
        "none", help="Elicitation strategy (none, socratic, decompose, reflective)"
    ),
    samples: int = typer.Option(1, help="Number of samples to generate"),
):
    config = {
        "mcpServers": {
            "memgraph": {"url": base_url},
        }
    }
    client = Client(config, elicitation_handler=my_elicitation_handler)

    async def _run():
        async with client:
            while True:
                user_input = input("\n> ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break
                # prompt = elicit_prompt(user_input, strategy=elicitation)
                console.print(f"\n[bold yellow]Prompt:[/bold yellow] {user_input}")

                try:
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
                except ToolError as e:
                    console.print(f"[red]Tool Error:[/red] {e}")
                except Exception as e:
                    console.print(f"[red]Unexpected Error:[/red] {e}")

                if samples > 1:
                    await _sample_responses(client, user_input, samples)
                else:
                    continue
                    await client.call_tool("run_query", {"query": user_input})

    asyncio.run(_run())


if __name__ == "__main__":
    app()
