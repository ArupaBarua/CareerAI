import re

from neo4j import AsyncManagedTransaction

from src.components.graphrag.connection import neo4j_driver
from src.components.graphrag.extractor import (
    ExtractedKnowledgeGraph,
    GraphEntity,
)
from src.constants.settings import settings


ENTITY_LABELS = {
    "skill": "Skill",
    "role": "Role",
    "company": "Company",
    "organization": "Organization",
    "institution": "Institution",
    "experience": "Experience",
    "project": "Project",
    "research": "Research",
    "publication": "Publication",
    "activity": "Activity",
    "certification": "Certification",
}


# These entities can reasonably be reused by multiple candidates.
GLOBAL_ENTITY_TYPES = {
    "skill",
    "role",
    "company",
    "organization",
    "institution",
    "certification",
}


# These belong to a specific candidate.
USER_SCOPED_ENTITY_TYPES = {
    "experience",
    "project",
    "research",
    "publication",
    "activity",
}


RELATIONSHIP_TYPES = {
    "HAS_SKILL",
    "HAS_ROLE",
    "HAS_EXPERIENCE",
    "WORKED_AT",
    "STUDIED_AT",
    "WORKED_ON",
    "CONDUCTED_RESEARCH",
    "AUTHORED",
    "PARTICIPATED_IN",
    "MEMBER_OF",
    "HAS_CERTIFICATION",
    "USES_SKILL",
    "ROLE_AT",
    "RELATED_TO",
}

def normalize_name(
    name: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        name,
    ).strip().casefold()


async def run_query(
    tx: AsyncManagedTransaction,
    query: str,
    **parameters
) -> None:
    
    result = await tx.run(
        query,
        **parameters
    )
    await result.consume()


async def merge_entity(
    tx: AsyncManagedTransaction,
    entity: GraphEntity,
    user_id: int
) -> None:
    
    label = ENTITY_LABELS[entity.type]

    normalized_name = normalize_name(entity.name)

    if entity.type in GLOBAL_ENTITY_TYPES:

        query = f"""
        MERGE (e:{label} {{
        normalized_name: $normalized_name
        }})

        ON CREATE SET
            e.name = $name

        SET
            e.category = COALESCE(
                $category,
                e.category
            )
        """

        await run_query(
            tx,
            query,
            normalized_name,
            name=entity.name,
            category=entity.category
        )

        return

    query = f"""
    MERGE (e:{label} {{
        owner_user_id: $user_id,
        normalized_name: $normalized_name
    }})

    ON CREATE SET
        e.name = $name

    SET
        e.category = COALESCE(
            $category,
            e.category
        )
    """

    await run_query(
        tx,
        query,
        user_id=user_id,
        normalized_name=normalized_name,
        name=entity.name,
        category=entity.category
    )


async def merge_resume_mention(
    tx: AsyncManagedTransaction,
    resume_id: int,
    user_id: int,
    entity: GraphEntity
) -> None:

    label = ENTITY_LABELS[entity.type]

    normalized_name = normalize_name(entity.name)

    if entity.type in GLOBAL_ENTITY_TYPES:
        query = f"""
        MATCH (r:Resume {{
            resume_id: $resume_id
        }})

        MATCH (e:{label} {{
            normalized_name: $normalized_name
        }})

        MERGE (r)-[:MENTIONS]->(e)
        """

        await run_query(
            tx,
            query,
            resume_id=resume_id,
            normalized_name=normalized_name
        )

        return

    query = f"""
    MATCH (r:Resume {{
        resume_id: $resume_id
    }})

    MATCH (e:{label} {{
        owner_user_id: $user_id,
        normalized_name: $normalized_name
    }})

    MERGE (r)-[:MENTIONS]->(e)
    """

    await run_query(
        tx,
        query,
        resume_id=resume_id,
        user_id=user_id,
        normalized_name=normalized_name
    )


def build_entity_match(
    variable: str,
    entity: GraphEntity
) -> tuple[str, dict]:

    label = ENTITY_LABELS[entity.type]

    if entity.type in GLOBAL_ENTITY_TYPES:
        clause = f"""
        MATCH ({variable}:{label} {{
            normalized_name: ${variable}_name
        }})
        """

        parameters = {
            f"{variable}_name": normalize_name(entity.name)
        }

        return clause, parameters

    clause = f"""
    MATCH ({variable}:{label} {{
        owner_user_id: $user_id,
        normalized_name: ${variable}_name
    }})
    """
    parameters = {
        f"{variable}_name": normalize_name(
            entity.name
        )
    }

    return clause, parameters


async def write_relationship(
    tx: AsyncManagedTransaction,
    relationship_type: str,
    source_entity: GraphEntity | None,
    target_entity: GraphEntity,
    user_id: int
) -> None:

    if relationship_type not in RELATIONSHIP_TYPES:
        return

    target_match, target_parameters = build_entity_match("target", target_entity)

    #Candidate -> Entity
    if source_entity is None:
        query = f"""
        MATCH (source:Candidate {{
            user_id: $user_id
        }})

        {target_match}

        MERGE (source)-[:{relationship_type}]->(target)
        """

        await run_query(
            tx,
            query,
            user_id=user_id,
            **target_parameters
        )

        return

    source_match, source_parameters = build_entity_match("source", source_entity)

    query = f"""
    {source_match}

    {target_match}

    MERGE (source)-[:{relationship_type}]->(target)
    """

    await run_query(
        tx,
        query,
        user_id=user_id,
        **source_parameters,
        **target_parameters
    )


async def _write_resume_graph(
    tx: AsyncManagedTransaction,
    user_id: int,
    resume_id: int,
    filename: str,
    graph: ExtractedKnowledgeGraph
) -> None:

    #Candidate
    await run_query(
        tx,
        """
        MERGE (c:Candidate {
            user_id: $user_id
        })

        SET c.name = COALESCE(
            $candidate_name,
            c.name
        )
        """,
        user_id=user_id,
        candidate_name=graph.candidate_name
    )

    #Resume
    await run_query(
        tx,
        """
        MATCH (c:Candidate {
            user_id: $user_id
        })

        MERGE (r:Resume {
            resume_id: $resume_id
        })

        SET
            r.user_id = $user_id,
            r.filename = $filename

        MERGE (c)-[:HAS_RESUME]->(r)
        """,
        user_id=user_id,
        resume_id=resume_id,
        filename=filename
    )

    #Entities

    entity_map: dict[str, GraphEntity] = {}

    for entity in graph.entities:
        entity_map[entity.id] = entity

        await merge_entity(
            tx=tx,
            entity=entity,
            user_id=user_id,
        )

        await merge_resume_mention(
            tx=tx,
            resume_id=resume_id,
            user_id=user_id,
            entity=entity,
        )

    for relationship in graph.relationships:

        if (relationship.type not in RELATIONSHIP_TYPES):
            continue

        target_entity = entity_map.get(relationship.target)

        if target_entity is None:
            continue

        if relationship.source == "candidate":
            await write_relationship(
                tx=tx,
                relationship_type=relationship.type,
                source_entity=None,
                target_entity=target_entity,
                user_id=user_id
            )

            continue

        source_entity = entity_map.get(relationship.source)

        if source_entity is None:
            continue

        await write_relationship(
            tx=tx,
            relationship_type=relationship.type,
            source_entity=source_entity,
            target_entity=target_entity,
            user_id=user_id
        )


async def save_resume_graph(
    user_id: int,
    resume_id: int,
    filename: str,
    graph: ExtractedKnowledgeGraph
) -> None:
    
    async with neo4j_driver.session(
        database=settings.NEO4J_DATABASE
    ) as session:

        await session.execute_write(
            _write_resume_graph,
            user_id,
            resume_id,
            filename,
            graph
        )