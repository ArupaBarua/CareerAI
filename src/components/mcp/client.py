from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.constants.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

mcp_client = MultiServerMCPClient(
    {
        "exa": {
            "transport": "streamable_http",
            "url": settings.EXA_MCP_URL
        }
    }
)

async def get_exa_search_tools() -> BaseTool:

    tools = await mcp_client.get_tools(
        server_name="exa"
    )

    for tool in tools:
        if tool.name == "web_search_exa":
            return tool

    logger.warning("Exa MCP tool 'web_search_exa' was not found.")

    return None