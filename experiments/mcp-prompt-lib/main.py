import asyncio, os, json, logging, sys
from litellm import acompletion, experimental_mcp_client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO, format_string=None):
    """
    Configure logging for the MCP prompt library.
    Args:
        level: Logging level (default: logging.INFO)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_string,
        force=True,  # Override any existing configuration
    )
    logger.setLevel(level)


def ask_with_tools(prompt, model="openai/gpt-4o"):
    async def __call(prompt):
        server = StdioServerParameters(
            command="uv",
            args=[
                "run",
                "--with",
                "mcp-memgraph",
                "--python",
                "3.13",
                "--project",
                "/Users/buda/Workspace/code/memgraph/ai-toolkit/integrations/mcp-memgraph/",
                "/Users/buda/Workspace/code/memgraph/ai-toolkit/integrations/mcp-memgraph/src/mcp_memgraph/main.py",
            ],
        )

        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await experimental_mcp_client.load_mcp_tools(
                    session=session, format="openai"
                )
                logger.info("Successfully loaded MCP tools:")
                logger.info(
                    "MCP tools details: %s",
                    json.dumps(tools, indent=2, ensure_ascii=False),
                )
                logger.info("MCP tools listed")

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

                messages = [{"role": "user", "content": prompt}]
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
                            logger.info(
                                "Tool %s result: %s",
                                tc["function"]["name"],
                                json.dumps(result.content, indent=2),
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "name": tc["function"]["name"],
                                    "content": json.dumps(
                                        result.content, ensure_ascii=False
                                    ),
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

    try:
        return asyncio.run(__call(prompt))
    except ExceptionGroup as eg:
        logger.error(f"Caught an exception from asyncio.run: {eg}")
    except Exception as e:
        logger.error(f"Caught a different type of exception: {e} (type: {type(e)})")


if __name__ == "__main__":
    level = logging.WARNING
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        level = getattr(logging, arg, logging.WARNING)
    configure_logging(level=level)

    # ask_with_tools("What's the list of tool that you have?")
    print(ask_with_tools("What's the most important entity in my dataset?"))
    # print(ask_with_tools("Find me discussions related to the AI topics and give me a summary of connected entities."))
