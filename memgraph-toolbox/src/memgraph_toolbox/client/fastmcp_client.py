import asyncio
from fastmcp.client import Client
from fastmcp.client.elicitation import ElicitResult


async def handle_elicitation(message, response_type, *_):
    print(f"Server asks: {message}")
    comment = input("INPUT:")
    return ElicitResult(action="accept", content=response_type(**{"comment": comment}))


async def main():
    async with Client(
        "http://localhost:8000/mcp", elicitation_handler=handle_elicitation
    ) as c:
        result = await c.call_tool("ask_feedback", {})
        print("Tool result:", result.data)


if __name__ == "__main__":
    asyncio.run(main())
