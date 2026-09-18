from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore

from src.components.agents.candidate_career_agent import candidate_career_agent
from src.components.agents.general_agent import general_agent
from src.components.agents.job_search_agent import job_search_agent
from src.components.agents.routing_agent import routing_node
from src.components.graph.state import CareerAIState
from src.components.memory.ltm_graph import ltm_graph
from src.components.memory.stm_node import summarize_conversation_node


def get_latest_user_message(
    state: CareerAIState,
) -> str:

    for message in reversed(state["messages"]):

        if isinstance(message, HumanMessage):
            return str(message.content)

    return ""


# Candidate Career Agent

async def candidate_career_node(
    state: CareerAIState,
    config: RunnableConfig,
) -> dict:

    user_query = get_latest_user_message(state)

    # Candidate-centric workflows require a resume.
    if state["resume_id"] is None:

        response = (
            "I need a resume to analyze your background "
            "for this request. Please upload resume first."
        )

        return {
            "final_response": response,
            "job_results": [],
            "messages": [AIMessage(content=response)]
        }

    result = await candidate_career_agent.ainvoke(
        {
            "query": user_query,

            "intent": state["intent"],

            "user_id": state["user_id"],

            "resume_id": state["resume_id"],

            "messages": state["messages"],

            "conversation_summary": state.get(
                "conversation_summary",
                "",
            ),

            "long_term_memories": state.get(
                "long_term_memories",
                [],
            ),
        },

        config=config,
    )

    final_response = result["final_response"]

    return {
        "final_response": final_response,

        # Clear results belonging to an older
        # job-search turn.
        "job_results": [],

        "messages": [AIMessage(content=final_response)]
    }

# Job Search Agent

async def job_search_node(
    state: CareerAIState,
    config: RunnableConfig,
) -> dict:

    user_query = get_latest_user_message(state)

    result = await job_search_agent.ainvoke(
        {
            "query": user_query,

            "messages": state[
                "messages"
            ],

            "conversation_summary": state.get(
                "conversation_summary",
                "",
            ),

            "long_term_memories": state.get(
                "long_term_memories",
                [],
            ),
        },

        config=config,
    )

    final_response = result["final_response"]

    return {
        "job_results": result.get(
            "job_results",
            [],
        ),

        "final_response": final_response,

        "messages": [AIMessage(content=final_response)]
    }


# General Agent

async def general_node(
    state: CareerAIState,
    config: RunnableConfig,
) -> dict:

    user_query = get_latest_user_message(state)

    result = await general_agent.ainvoke(
        {
            "query": user_query,

            "messages": state[
                "messages"
            ],

            "conversation_summary": state.get(
                "conversation_summary",
                "",
            ),

            "long_term_memories": state.get(
                "long_term_memories",
                [],
            ),
        },

        config=config,
    )

    final_response = result["final_response"]

    return {
        "final_response": final_response,

        "job_results": [],

        "messages": [AIMessage(content=final_response)]
    }


# Routing

def route_to_agent(
    state: CareerAIState,
) -> str:

    intent = state["intent"]

    if intent in {
        "resume_analysis",
        "skill_gap",
        "job_recommendation",
    }:
        return "candidate_career"

    if intent == "job_search":
        return "job_search"

    return "general"

# Building Main Graph

def build_career_ai_graph(
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
):

    builder = StateGraph(CareerAIState)

    builder.add_node(
        "long_term_memory",
        ltm_graph,
    )

    builder.add_node(
        "summarize_conversation",
        summarize_conversation_node,
    )

    # Router
    
    builder.add_node(
        "routing",
        routing_node,
    )

    builder.add_node(
        "candidate_career",
        candidate_career_node,
    )

    builder.add_node(
        "job_search",
        job_search_node,
    )

    builder.add_node(
        "general",
        general_node,
    )

    builder.add_edge(
        START,
        "long_term_memory",
    )

    builder.add_edge(
        "long_term_memory",
        "summarize_conversation",
    )

    builder.add_edge(
        "summarize_conversation",
        "routing",
    )

    builder.add_conditional_edges(
        "routing",
        route_to_agent,
        {
            "candidate_career": "candidate_career",

            "job_search": "job_search",

            "general": "general",
        },
    )


    builder.add_edge(
        "candidate_career",
        END,
    )

    builder.add_edge(
        "job_search",
        END,
    )

    builder.add_edge(
        "general",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )