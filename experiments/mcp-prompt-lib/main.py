import asyncio, os, json
from litellm import acompletion, experimental_mcp_client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ssl


async def main():
    # Configure SSL context to handle certificate issues
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Set environment variables to handle SSL issues
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["SSL_VERIFY"] = "false"
    os.environ["PYTHONHTTPSVERIFY"] = "0"

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

            # 1) Load MCP tools in OpenAI format
            tools = await experimental_mcp_client.load_mcp_tools(
                session=session, format="openai"
            )

            print("Successfully loaded MCP tools:")
            print(json.dumps(tools, indent=2, ensure_ascii=False))

            # Check if we have an API key
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("\nNo OPENAI_API_KEY found. MCP tools loaded successfully!")
                print(
                    "To use with an actual LLM, set the OPENAI_API_KEY environment variable."
                )
                return

            # 2) Test with LLM if API key is available
            messages = [
                {"role": "user", "content": "List me the tools you are aware of."}
            ]

            resp = await acompletion(
                model="gpt-4o",
                api_key=api_key,
                messages=messages,
                tools=tools,
            )
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                print(f"\nLLM Response: {msg.get('content', '')}")
            else:
                print(f"\nLLM wants to use {len(tool_calls)} tools:")
                for tc in tool_calls:
                    print(
                        f"- {tc['function']['name']}: {tc['function'].get('description', 'No description')}"
                    )

                # Execute tool calls
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content", ""),
                        "tool_calls": tool_calls,
                    }
                )
                for tc in tool_calls:
                    try:
                        result = await experimental_mcp_client.call_openai_tool(
                            session=session, tool_call=tc
                        )
                        print(
                            f"\nTool {tc['function']['name']} result: {json.dumps(result, indent=2)}"
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": tc["function"]["name"],
                                "content": json.dumps(result, ensure_ascii=False),
                            }
                        )
                    except Exception as tool_error:
                        print(
                            f"Error executing tool {tc['function']['name']}: {tool_error}"
                        )


asyncio.run(main())
