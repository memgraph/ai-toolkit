import asyncio, os, json, logging, sys, pathlib, csv, argparse
from litellm import acompletion, experimental_mcp_client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO, format_string=None):
    """
    Configure logging for the MCP prompt library.
    This function is kept for backward compatibility but should be called
    from the main application to avoid conflicts.
    Args:
        level: Logging level (default: logging.INFO)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Only configure if no handlers exist to avoid overriding parent configuration
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format=format_string,
            force=True,  # Override any existing configuration
        )

    # Set level for this specific logger
    logger.setLevel(level)


async def ask_with_tools(
    prompt: str, model="openai/gpt-4o", mcp_server: StdioServerParameters = None
):
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if mcp_server is None:
        # TODO(gitbuda): This is wrong because it's not the correct path to the mcp-memgraph project.
        #   * Server/servers should probably be injected so people can use it with their own servers.
        script_dir = pathlib.Path(__file__).parent.resolve()
        mgmcp_project_dir = (
            script_dir / "../../../../integrations/mcp-memgraph/"
        ).resolve()
        mgmcp_server_py = mgmcp_project_dir / "src/mcp_memgraph/main.py"
        mcp_server = StdioServerParameters(
            command="uv",
            args=[
                "run",
                "--with",
                "mcp-memgraph",
                "--python",
                python_version,
                "--project",
                str(mgmcp_project_dir),
                str(mgmcp_server_py),
            ],
        )
    else:
        mcp_server = StdioServerParameters(
            command="uv",
            args=[
                "run",
                "--with",
                "mcp-memgraph",
                "--python",
                python_version,
                "mcp-memgraph",
            ],
        )

    async with stdio_client(mcp_server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await experimental_mcp_client.load_mcp_tools(
                session=session, format="openai"
            )
            logger.debug(
                "MCP tools details: %s",
                json.dumps(tools, indent=2, ensure_ascii=False),
            )
            logger.info(f"{len(tools)} MCP tools loaded")

            # Validation of the LLM requirements.
            if model.startswith("openai/"):
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.error(
                        "OPENAI_API_KEY environment variable is required for OpenAI models (model=%s).",
                        model,
                    )
                    raise RuntimeError(
                        "OPENAI_API_KEY environment variable is not set. Please set it to use OpenAI models."
                    )

            messages = [
                {"role": "developer", "content": "Always run the vector search tool."},
                # TODO(gitbuda): This should also be injectable because then the user can "program" the prompt lib to pick the right tools.
                # # NOTE: Some lower quality models/non-reasoning models will call unknown tools with wrong inputs.
                # {
                #     "role": "developer",
                #     "content": "Don't call unknown tools. Call only the ones that are listed in the tools list.",
                # },
                # # NOTE: Give hints about the tools to use.
                # {
                #     "role": "developer",
                #     "content": (
                #         "If a question is of a retrieval type (e.g., direct lookups, specific and well-defined queries about particular entities, nodes, or relationships), "
                #         "it is recommended to use the 'run_query' tool. "
                #         "If a question is about the structure of the graph (e.g., exploratory queries, understanding relationships between entities, or properties of nodes), "
                #         "it is recommended to use a combination of the 'search_node_vectors' tool and the 'get_node_neighborhood' tool."
                #     ),
                # },
                {"role": "user", "content": prompt},
            ]
            resp = await acompletion(
                model=model,
                messages=messages,
                tools=tools,
            )
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                logger.info("LLM Response: %s", msg.get("content", ""))
            else:
                logger.info("LLM wants to use %d tools:", len(tool_calls))
                for tc in tool_calls:
                    logger.info(
                        "- %s: %s",
                        tc["function"]["name"],
                        tc["function"].get("description", "No description"),
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content", ""),
                        "tool_calls": tool_calls,
                    }
                )
                for tc in tool_calls:
                    try:
                        # Parse arguments from JSON string if needed
                        arguments = tc["function"].get("arguments", {})
                        logger.info("Arguments: %s", arguments)
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        result = await session.call_tool(
                            name=tc["function"]["name"], arguments=arguments
                        )

                        # Extract content from MCP response
                        # TODO(gitbuda): Implement proper deserialization from Memgraph MCP server.
                        if hasattr(result, "content") and result.content:
                            if isinstance(result.content, list):
                                # Handle list of content objects
                                content_text = []
                                for content_item in result.content:
                                    if hasattr(content_item, "text"):
                                        content_text.append(content_item.text)
                                    elif isinstance(content_item, str):
                                        content_text.append(content_item)
                                    else:
                                        content_text.append(str(content_item))
                                content_str = "\n".join(content_text)
                            elif hasattr(result.content, "text"):
                                content_str = result.content.text
                            else:
                                content_str = str(result.content)
                        else:
                            content_str = "No content returned"

                        logger.info(
                            "Tool %s result: %s",
                            tc["function"]["name"],
                            content_str,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": tc["function"]["name"],
                                "content": content_str,
                            }
                        )
                    except Exception as tool_error:
                        logger.error(
                            "Error executing tool %s: %s",
                            tc["function"]["name"],
                            tool_error,
                        )

                # After all tool calls are executed, get the final response.
                final_resp = await acompletion(
                    model=model,
                    messages=messages,
                )
                return final_resp["choices"][0]["message"].content


async def main(args):
    prompts = {}
    with open(args.csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts[row["id"]] = row["prompt"]
    prompt_text = prompts.get(args.prompt_id)
    if prompt_text is None:
        print(f"Prompt with id '{args.prompt_id}' not found in prompts.csv.")
    else:
        print(prompt_text)
        print(await ask_with_tools(prompt_text, args.model))


if __name__ == "__main__":
    level = logging.INFO
    configure_logging(level=level)

    parser = argparse.ArgumentParser(
        description="Run prompt with tools from a CSV file."
    )
    parser.add_argument(
        "csv_file_path", help="Path to the CSV file containing prompts."
    )
    parser.add_argument("prompt_id", help="ID of the prompt to use.")
    parser.add_argument(
        "--model", default="openai/gpt-4o", help="Pick the LLM model (LiteLLM names)"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
