from fastmcp import FastMCP, Context
from pydantic import BaseModel

mcp = FastMCP("demo-server")


class Feedback(BaseModel):
    comment: str


@mcp.tool
async def ask_feedback(ctx: Context) -> str:
    result = await ctx.elicit("Please write a comment:", response_type=Feedback)
    print(f"Result: {result}")
    if result.action == "accept":
        return f"You said: {result.data.comment}"
    else:
        return "You declined to provide feedback."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
