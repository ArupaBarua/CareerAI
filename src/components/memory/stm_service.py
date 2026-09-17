from langchain_core.messages import BaseMessage

from src.constants.settings import settings

summary_threshold = settings.SUMMARY_THRESHOLD
recent_messages_to_keep = settings.RECENT_MESSAGES_TO_KEEP

def should_summarize(messages: list[BaseMessage]) -> bool:

    return len(messages) >= summary_threshold


def split_messages_for_summary(
    messages: list[BaseMessage],
) -> tuple[
    list[BaseMessage],
    list[BaseMessage],
]:

    messages_to_summarize = messages[:-recent_messages_to_keep]

    recent_messages = messages[-recent_messages_to_keep:]

    return (
        messages_to_summarize,
        recent_messages,
    )


def format_messages_for_summary(
    messages: list[BaseMessage],
) -> str:

    formatted_messages = []

    for message in messages:

        formatted_messages.append(
            f"{message.type}: {message.content}"
        )

    return "\n".join(formatted_messages)


def get_thread_config(
    conversation_id: int,
) -> dict:

    return {
        "configurable": {
            "thread_id": str(conversation_id)
        }
    }