from pydantic import BaseModel
from neo4j import AsyncManagedTransaction

from src.components.graphrag.connection import neo4j_driver
from src.constants.settings import settings

class GraphFact(BaseModel):
    source_type: str
    source_name: str
    relationship: str
    target_type: str
    target_name: str

RESUME_GRAPH_QUERY = """
MATCH (candidate:Candidate {
    user_id: $user_id
})-[:HAS_RESUME]->(resume:Resume {
    resume_id: $resume_id
})

MATCH (resume)-[:MENTIONS]->(target)

MATCH (candidate)-[relationship]->(target)

WHERE type(relationship) <> 'HAS_RESUME'

RETURN
    'Candidate' AS source_type,
    COALESCE(candidate.name, 'Candidate') AS source_name,
    type(relationship) AS relationship,
    labels(target)[0] AS target_type,
    target.name AS target_name

UNION

MATCH (candidate:Candidate {
    user_id: $user_id
})-[:HAS_RESUME]->(resume:Resume {
    resume_id: $resume_id
})

MATCH (resume)-[:MENTIONS]->(source)

MATCH (source)-[relationship]->(target)

WHERE EXISTS {
    MATCH (resume)-[:MENTIONS]->(target)
}

RETURN
    labels(source)[0] AS source_type,
    source.name AS source_name,
    type(relationship) AS relationship,
    labels(target)[0] AS target_type,
    target.name AS target_name
"""

async def _retrieve_resume_graph(
    tx: AsyncManagedTransaction,
    user_id: int,
    resume_id: int,
) -> list[GraphFact]:

    result = await tx.run(
        RESUME_GRAPH_QUERY,
        user_id=user_id,
        resume_id=resume_id,
    )

    records = await result.data()

    return [
        GraphFact(**record)
        for record in records
    ]


async def retrieve_resume_graph(
    user_id: int,
    resume_id: int,
) -> list[GraphFact]:

    async with neo4j_driver.session(
        database=settings.NEO4J_DATABASE
    ) as session:

        return await session.execute_read(
            _retrieve_resume_graph,
            user_id,
            resume_id,
        )