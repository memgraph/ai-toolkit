from memgraph_toolbox.tools.trigger import ShowTriggersTool
from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.toolbox import MemgraphToolbox

# Connect to Memgraph
db = Memgraph(url="bolt://localhost:7687", username="", password="")

# Show available tools
toolbox = MemgraphToolbox(db)
for tool in toolbox.get_all_tools():
    print(f"Tool Name: {tool.name}, Description: {tool.description}")

# Use the ShowTriggersTool
tool = ShowTriggersTool(db)
triggers = tool.call({})
print(triggers)
