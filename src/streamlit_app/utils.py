import streamlit as st
from client import AgentClient, AgentClientError
from schema.schema import ChatMessage


def format_message_for_display(msg_dict) -> ChatMessage:
    # msg_dict comes from your SQLite rows
    msg_type = "human" if msg_dict["sender"] == "User" else "ai"
    return ChatMessage(
        type=msg_type,
        content=msg_dict["content"],
        run_id=msg_dict["id"],
    )


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)
    # If the feedback value or run ID has changed, send a new feedback record
    if (
        feedback is not None
        and (latest_run_id, feedback) != st.session_state.last_feedback
    ):
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        print(f"Recording feedback: {feedback} for run ID: {latest_run_id}")
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")
