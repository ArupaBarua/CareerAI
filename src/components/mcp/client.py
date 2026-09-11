from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.constants.settings import settings

web_search_client = MultiServerMCPClient(
    {
        "exa": {
            "transport": "http",
            "url": settings.EXA_MCP_URL
        }
    }
)

async def get_web_search_tools() -> list[BaseTool]:
    return await web_search_client.get_tools()