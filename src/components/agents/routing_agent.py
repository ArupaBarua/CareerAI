from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.components.graph.state import CareerAIState
from src.components.llm.model import llm
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class RouterDecision(BaseModel):
    intent: Literal[
        "job_search",
        "resume_analysis",
        "skill_gap",
        "job_recommendation",
        "general"
    ] = Field(
        description="The CareerAI workflow that should handle the user's request."
    )

router_llm = llm.with_structured_output(RouterDecision)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the routing agent for CareerAI.

CareerAI is an AI-powered career assistant chatbot designed to help users
with job searching, resume analysis, job-role recommendations, skill-gap
analysis, and general career-related questions.

Your only responsibility is to determine which CareerAI workflow should
handle the user's latest request.

Available intents:

1. job_search:
Use when the user primarily wants to find current job openings,
vacancies, hiring companies, or live job listings.

Do NOT use job_search merely because a role or job is mentioned.
If the user asks whether they are suitable, qualified, a good fit,
what they are missing, or how their background compares with a role,
use skill_gap instead.

2. resume_analysis:
Use when the user asks to review, analyze, summarize, or ask factual
questions about their own resume, candidate profile, professional
background, skills, projects, education, work experience, publications,
research, certifications, achievements, or other information that may
come from their selected resume.

Also use this intent when the user asks whether CareerAI has access to,
knows, or can use their selected resume.

3. skill_gap:
Use when the user asks whether they are suitable, qualified, or a
good fit for a particular role or job, or asks what skills,
experience, or qualifications they are missing.

This includes comparisons between the user's resume/profile and:
- a job description,
- a named role,
- a target profession,
- expected requirements for that role.

Examples:
"Am I suitable for an AI Engineer role?"
→ skill_gap

"Do I qualify for ML Engineer positions?"
→ skill_gap

"What skills am I missing for an AI Engineer role?"
→ skill_gap

"Compare my resume with this job description."
→ skill_gap

"Find remote AI Engineer jobs."
→ job_search

If a request contains both job-search language and a request to assess
the user's suitability, qualifications, fit, or skill gaps, classify it
as skill_gap unless the user explicitly asks to find actual current
openings as the main task.

4. job_recommendation
Use when the user wants CareerAI to recommend suitable job roles or career
opportunities based on their resume, skills, experience, education,
projects, or background.

5. general
Use for career-related questions or any general question that do not require job search, resume
analysis, skill-gap analysis, or personalized job recommendations.

Choose exactly one intent based on the user's actual request.

Do not answer the user's question.
Do not perform the requested task.
Only classify the request into the most appropriate workflow."""
        ),
        (
            "human",
            "{user_query}"
        )
    ]
)

router_chain = router_prompt | router_llm

async def routing_node(state: CareerAIState) -> dict:
    user_query = state["messages"][-1].content

    decision = await router_chain.ainvoke(
        {
            "user_query": user_query
        }
    )

    return {
        "intent": decision.intent
    }