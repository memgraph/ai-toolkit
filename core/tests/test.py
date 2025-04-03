
from api.tool import BaseTool
from api.toolkit import Toolkit

from typing import Dict, List
import asyncio


class CalculateSumTool(BaseTool):
    def __init__(self):
        input_schema = {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
        super().__init__(name="calculate_sum", description="Add two numbers together", input_schema=input_schema)

    async def call(self, arguments: Dict[str, any]) -> List[any]:
        a = arguments.get("a")
        b = arguments.get("b")
        result = a + b
        return [result]

async def main():
    
    toolkit = Toolkit()

    calc_tool = CalculateSumTool()

    toolkit.add(calc_tool)
    
    print("Added tools:", toolkit.list_tools())
    
    tool = toolkit.get_tool("calculate_sum")
    result = await tool.call({"a": 10, "b": 5})
    print(f"Result from '{tool.name}':", result)

if __name__ == "__main__":
    asyncio.run(main())
