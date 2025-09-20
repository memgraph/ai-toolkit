import json, logging, sys, os
from typing import Optional, List, Dict

import asyncio
from litellm import acompletion, experimental_mcp_client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


async def prompt_with_tools(
    prompt: str,
    tool_selection_model_name: str = "openai/gpt-4o",
    tool_selection_system_messages: List[Dict[str, str]] = [],
    response_model_name: str = "openai/gpt-4o",
    response_system_messages: List[Dict[str, str]] = [],
    mcp_server_params: Optional[StdioServerParameters] = None,
):
    """
    Make an LLM prompt with tools.
    First there is a tool selection phase where the LLM is asked to select the tools to use.
    Second there is a response phase where the LLM is asked to generate a response.
    """
    # Configure MCP server connection.
    if mcp_server_params is None:  # Use Memgraph MCP server by default.
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        server_params = StdioServerParameters(
            command="uv",
            args=[
                "run",
                "--with",
                "mcp-memgraph",
                "--python",
                python_version,
                "mcp-memgraph",
            ],
            env={
                "MEMGRAPH_URL": os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
                # NOTE: If you want to control where uv pulls the dependencies from inject UV_PYTHON or VIRTUAL_ENV.
            },
        )
    else:
        server_params = mcp_server_params

    # Establish MCP server connection.
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # https://docs.litellm.ai/docs/mcp#litellm-python-sdk-mcp-bridge
            tools = await experimental_mcp_client.load_mcp_tools(
                # NOTE: format="openai" works with everything tested, "mcp" does NOT work.
                session=session,
                format="openai",
            )
            logger.debug(
                "MCP tools details: %s",
                json.dumps(tools, indent=2, ensure_ascii=False),
            )
            logger.info(f"{len(tools)} MCP tools loaded")

            # Tool selection by an LLM.
            messages = tool_selection_system_messages + [
                {"role": "user", "content": prompt},
            ]
            resp = await acompletion(
                model=tool_selection_model_name,
                messages=messages,
                tools=tools,
            )
            msg = resp["choices"][0]["message"]
            # Tool calls by the MCP server.
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
                # Ensure all tool call IDs are within OpenAI's 40-character limit
                for tc in tool_calls:
                    if len(tc["id"]) > 40:
                        logger.warning(
                            "Tool call ID is too long, truncating to 40 characters: %s",
                            tc["id"],
                        )
                        tc["id"] = tc["id"][:40]
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content", ""),
                        "tool_calls": tool_calls,
                    }
                )
                for tc in tool_calls:
                    try:
                        arguments = tc["function"].get("arguments", {})
                        logger.info("Arguments: %s", arguments)
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        result = await session.call_tool(
                            name=tc["function"]["name"], arguments=arguments
                        )
                        if hasattr(result, "content") and result.content:
                            if isinstance(result.content, list):
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
                        logger.debug(
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

                # Generating the final response using LLM.
                messages = messages + response_system_messages
                final_resp = await acompletion(
                    model=response_model_name,
                    messages=messages,
                )
                return final_resp["choices"][0]["message"].content


async def __async_context():
    response = await prompt_with_tools(
        "Tell me anything using tools :pray:",
        tool_selection_model_name="openai/gpt-4o",
        # tool_selection_model_name="ollama/llama3.2",
        # tool_selection_model_name="ollama/deepseek-r1:1.5b",
        # tool_selection_model_name="azure/gpt-4o",
        tool_selection_system_messages=[
            {
                "role": "system",
                "content": "If you choose to run a query tool, use Cypher. Use LIMIT 1.",
            }
        ],
        response_model_name="openai/gpt-4o",
        response_system_messages=[],
    )
    print(response)


if __name__ == "__main__":
    # NOTE: Chaning how mcp-memgraph logs from here is not easy because the
    # server is started in another process.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(__async_context())
