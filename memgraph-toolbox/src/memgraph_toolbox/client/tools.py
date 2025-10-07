import json, logging

logger = logging.getLogger(__name__)


async def atool_calls(resp, messages, session):
    """
    Execute tool calls by the MCP server.
    """
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
            except Exception as e:
                logger.error(str(e))
    return messages
