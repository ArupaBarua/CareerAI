from neo4j import AsyncDriver, AsyncGraphDatabase

from src.constants.settings import settings
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

neo4j_driver: AsyncDriver = AsyncGraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(
        settings.NEO4J_USERNAME,
        settings.NEO4J_PASSWORD
    )
)

async def verify_neo4j_connection() -> None:
    await neo4j_driver.verify_connectivity()

    logger.info("Neo4j connection verified.")


async def close_neo4j_connection() -> None:
    await neo4j_driver.close()

    logger.info("Neo4j connection closed.")
