# Here we will create a custom tool for the crewai framework

from crewai_tools import BaseTool

class CustomTool(BaseTool):
    """A custom tool for the crewai framework."""
    name: str = "custom_tool"
    description: str = "A custom tool for the crewai framework."
    
    def _run(self, query: str) -> str:
        return "Hello, world!"