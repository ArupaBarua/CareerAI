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

1. job_search
Use when the user wants CareerAI to search for current job opportunities,
open positions, internships, or vacancies for a particular role, skill set,
location, company, or other criteria.

2. resume_analysis
Use when the user wants CareerAI to review, analyze, evaluate, improve,
or provide feedback on their resume or CV.

3. skill_gap
Use when the user wants CareerAI to compare their resume, experience, or
skills against a specific job description or role and identify missing,
weak, or required skills.

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