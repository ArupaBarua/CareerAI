from src.components.graphrag.connection import neo4j_driver
from src.constants.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


CONSTRAINT_QUERIES = [
    # Candidate
    """
    CREATE CONSTRAINT candidate_user_id_unique IF NOT EXISTS
    FOR (c:Candidate)
    REQUIRE c.user_id IS UNIQUE
    """,

    # Resume
    """
    CREATE CONSTRAINT resume_id_unique IF NOT EXISTS
    FOR (r:Resume)
    REQUIRE r.resume_id IS UNIQUE
    """,

    # Global entities
    """
    CREATE CONSTRAINT skill_name_unique IF NOT EXISTS
    FOR (e:Skill)
    REQUIRE e.normalized_name IS UNIQUE
    """,

    """
    CREATE CONSTRAINT role_name_unique IF NOT EXISTS
    FOR (e:Role)
    REQUIRE e.normalized_name IS UNIQUE
    """,

    """
    CREATE CONSTRAINT company_name_unique IF NOT EXISTS
    FOR (e:Company)
    REQUIRE e.normalized_name IS UNIQUE
    """,

    """
    CREATE CONSTRAINT organization_name_unique IF NOT EXISTS
    FOR (e:Organization)
    REQUIRE e.normalized_name IS UNIQUE
    """,

    """
    CREATE CONSTRAINT institution_name_unique IF NOT EXISTS
    FOR (e:Institution)
    REQUIRE e.normalized_name IS UNIQUE
    """,

    # User-scoped entities
    """
    CREATE CONSTRAINT experience_owner_name_unique IF NOT EXISTS
    FOR (e:Experience)
    REQUIRE (
        e.owner_user_id,
        e.normalized_name
    ) IS UNIQUE
    """,

    """
    CREATE CONSTRAINT project_owner_name_unique IF NOT EXISTS
    FOR (e:Project)
    REQUIRE (
        e.owner_user_id,
        e.normalized_name
    ) IS UNIQUE
    """,

    """
    CREATE CONSTRAINT research_owner_name_unique IF NOT EXISTS
    FOR (e:Research)
    REQUIRE (
        e.owner_user_id,
        e.normalized_name
    ) IS UNIQUE
    """,

    """
    CREATE CONSTRAINT activity_owner_name_unique IF NOT EXISTS
    FOR (e:Activity)
    REQUIRE (
        e.owner_user_id,
        e.normalized_name
    ) IS UNIQUE
    """,
]


async def ensure_neo4j_schema() -> None:

    async with neo4j_driver.session(
        database=settings.NEO4J_DATABASE
    ) as session:

        for query in CONSTRAINT_QUERIES:
            result = await session.run(query)
            await result.consume()

    logger.info("Neo4j schema constraints verified.")