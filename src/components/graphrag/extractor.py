from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.components.llm.model import llm

class GraphEntity(BaseModel):
    id: str = Field(
        description="Temporary unique identifier for this entity within this extraction, such as e1, e2, e3."
    )
    name: str = Field(
        description="Canonical human-readable name of the entity such as Python, AWS, RAG."
    )
    type: Literal[
        "skill",
        "role",
        "company",
        "organization",
        "institution",
        "experience",
        "project",
        "research",
        "publication",
        "activity",
        "certification",
    ]
    category: str | None = Field(
        default=None,
        description=(
            "Optional category or subtype. For example, a skill "
            "may have category: 'language', 'ml_dl', 'genai', "
            "'database', 'mlops_devops', or 'cloud'. An activity "
            "may have category: 'leadership', 'volunteering', "
            "'competition', or 'extracurricular'."
        ),
    )


class GraphRelationship(BaseModel):
    source: str = Field(
        description=("Temporary entity ID of the relationship source. "
            "Use the special value 'candidate' when the "
            "relationship starts from the resume owner."
        )
    )
    target: str = Field(
        description="Temporary entity ID of the relationship target."
    )
    type: Literal[
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
    ]


class ExtractedKnowledgeGraph(BaseModel):
    candidate_name: str | None = Field(
        default=None,
        description="Full name of the resume owner if explicitly present in the resume. Otherwise null."
    )
    entities: list[GraphEntity]
    relationships: list[GraphRelationship]


graph_extractor_llm = llm.with_structured_output(ExtractedKnowledgeGraph)


graph_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the resume entity-and-relationship extraction component
of CareerAI.

CareerAI is an AI-powered career assistant that performs resume
analysis, skill-gap analysis, job recommendations, job search,
and career guidance.

Your task is to analyze the supplied resume text and extract
career-related entities and relationships as structured data.

The extracted entities and relationships will be used to construct
and update CareerAI's Neo4j knowledge graph.

Do not create a Candidate entity.
Use the special identifier "candidate" as the relationship source
when the resume owner is the source of the relationship.

Extract only information explicitly supported by the resume.

ENTITY TYPES

skill:
Any competency, language, framework, library, platform, tool,
technical concept, methodology, or professional skill.

Examples:
Python
PyTorch
RAG
LangGraph
Docker
AWS
Data Structures & Algorithms
Leadership

Use category to preserve useful classifications such as:
language
core_cs
ml_dl
genai_agentic_ai
vector_database
database
mlops_devops
cloud
backend
data_analysis
soft_skill

role:
Professional, academic, research, organizational, or leadership role.

Examples:
AI Engineer
Research Assistant
President
Team Lead

company:
Commercial employer or company.

organization:
Club, association, research group, society, nonprofit,
professional organization, or other organization.

institution:
University, college, school, research institution, or
other educational institution.

experience:
A meaningful professional, internship, leadership, volunteer,
or other experience described in the resume.

project:
A project built or contributed to by the candidate.

research:
A research work, research topic, thesis, or research project.

publication:
A paper, journal article, conference publication, poster,
or other scholarly publication.

activity:
An extracurricular, competition, volunteering, club,
community, or other activity.

Use category when useful, such as:
leadership
competition
volunteering
extracurricular

certification:
A professional or academic certification.

RELATIONSHIPS

candidate -> HAS_SKILL -> skill

candidate -> HAS_ROLE -> role

candidate -> HAS_EXPERIENCE -> experience

candidate -> WORKED_AT -> company

candidate -> STUDIED_AT -> institution

candidate -> WORKED_ON -> project

candidate -> CONDUCTED_RESEARCH -> research

candidate -> AUTHORED -> publication

candidate -> PARTICIPATED_IN -> activity

candidate -> MEMBER_OF -> organization

candidate -> HAS_CERTIFICATION -> certification

experience/project/research -> USES_SKILL -> skill

role -> ROLE_AT -> company/organization/institution/activity

RELATED_TO may be used between extracted entities only when
the relationship is explicitly supported by the resume.

IMPORTANT RULES

- Never invent information.
- Never infer skills merely because they are commonly associated
  with another technology, role, project, or field.
- If a skill is explicitly listed in a Technical Skills section,
  it may be connected directly to the candidate with HAS_SKILL.
- Treat tools and technologies such as PyTorch, Docker, AWS,
  LangGraph, PostgreSQL, and FastAPI as skills. Use category to
  describe their subtype rather than creating a separate
  technology entity type.
- Create one entity for each meaningful skill rather than one
  entity representing an entire resume section.
- Leadership positions should normally be represented as roles,
  organizations, experiences, or activities rather than merely
  converting the entire experience into a "leadership" skill.
- Research works and publications are separate concepts.
- Do not assume that being related to a project means the
  candidate has every possible skill associated with that project.
- Each extracted entity must have a temporary ID such as
  e1, e2, e3.
- Relationship source and target values must reference those
  temporary IDs, except that source may be "candidate".
- Normalize obvious naming variations when appropriate.
- Return only the structured graph."""
        ),
        (
            "human",
            """
Resume content:

{content}"""
        )
    ]
)


graph_extraction_chain = graph_extraction_prompt | graph_extractor_llm