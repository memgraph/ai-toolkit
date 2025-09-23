import pytest

from ..client.mcp_prompt import prompt_with_tools


@pytest.mark.asyncio
async def test_mcp_prompt():
    """Test the MCP prompt."""
    prompt = "What's the number of entities in my database?"
    response = await prompt_with_tools(prompt)
    assert response is not None
    assert response != ""
