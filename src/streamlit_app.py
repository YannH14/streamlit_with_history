import asyncio
import os
import uuid
from collections.abc import AsyncGenerator
import time
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import streamlit as st
from client import AgentClient, AgentClientError
from schema.schema import ChatHistory, ChatMessage
from streamlit_app.init_db import init_database
from streamlit_app.message_generation import draw_messages
from streamlit_app.models import ChatDatabase
from streamlit_app.utils import format_message_for_display, handle_feedback

# Load environment variables from .env file
load_dotenv()

# Constants
APP_TITLE = "Chatbot Epargne Demo"
APP_ICON = "🤖"
USER_ID_COOKIE = "user_id"
DB_PATH = "chat_database.db"
API_KEY = os.getenv("OPENAI_API_KEY", "")


# Initialize LLM and conversation chain
def get_llm_chain():
    llm = ChatOpenAI(api_key=API_KEY)
    memory = ConversationBufferMemory()
    return ConversationChain(llm=llm, memory=memory)


# Streamlit UI
async def main():
    # Initialize database
    init_database()
    db = ChatDatabase()

    # Page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
    )

    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_info = None
        st.session_state.conversations = []
        st.session_state.current_thread_id = None
        st.session_state.messages = []
        st.session_state.llm_chain = None

    # Login screen
    if not st.session_state.logged_in:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.subheader("Login")

        username = st.text_input("Username")
        if st.button("Login") and username:
            # Get or create user record
            user_info = db.get_or_create_user(username)
            print(f"User info: {user_info}")
            # Set session state
            st.session_state.logged_in = True
            st.session_state.user_info = user_info
            st.session_state.conversations = db.list_conversations(user_info["id"])
            st.session_state.llm_chain = get_llm_chain()

            st.rerun()

        st.stop()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()

    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(
                    thread_id=thread_id
                ).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Main app layout
    st.title(f"{APP_ICON} {APP_TITLE}")

    # Sidebar
    with st.sidebar:
        st.header(f"👤 {st.session_state.user_info['username']}")

        # Logout button
        if st.button("🚪 Log out", use_container_width=True):
            for key in [
                "logged_in",
                "user_info",
                "conversations",
                "current_thread_id",
                "messages",
                "llm_chain",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # New chat button
        if st.button("➕ New Chat", use_container_width=True):
            new_conv = db.create_conversation(st.session_state.user_info["id"])
            st.session_state.conversations.insert(0, new_conv)
            st.session_state.current_thread_id = new_conv["thread_id"]
            st.session_state.messages = []
            st.session_state.llm_chain = get_llm_chain()  # Reset conversation memory
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox(
                "LLM to use", options=agent_client.info.models, index=model_idx
            )
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)

            # Display user ID (for debugging or user information)
            st.text_input(
                "User ID (read-only)",
                value=st.session_state.user_info["id"],
                disabled=True,
            )
        st.divider()

        # List existing conversationss
        if st.session_state.conversations:
            st.subheader("Your Chats")

            for i, conv in enumerate(st.session_state.conversations):
                title = conv["title"] or "Untitled Chat"
                if st.sidebar.button(
                    f"{title}", key=f"convo_{i}", use_container_width=True
                ):
                    st.session_state.current_thread_id = conv["thread_id"]
                    messages = db.get_conversation(
                        conv["thread_id"], st.session_state.user_info["id"]
                    )
                    st.session_state.messages = [
                        format_message_for_display(msg) for msg in messages
                    ]
                    st.session_state.llm_chain = (
                        get_llm_chain()
                    )  # Reset conversation memory

                    # Load conversation history into the LLM memory
                    for msg in messages:
                        if msg["sender"] == "User":
                            st.session_state.llm_chain.memory.chat_memory.add_user_message(
                                msg["content"]
                            )
                        else:
                            st.session_state.llm_chain.memory.chat_memory.add_ai_message(
                                msg["content"]
                            )

                    st.rerun()

                # Delete button with confirmation
                with st.sidebar.expander("Delete", expanded=False):
                    if st.button(
                        "Confirm Delete", key=f"delete_{i}", use_container_width=True
                    ):
                        db.delete_conversation(conv["thread_id"])
                        st.session_state.conversations.pop(i)
                        if st.session_state.current_thread_id == conv["thread_id"]:
                            st.session_state.current_thread_id = None
                            st.session_state.messages = []
                        st.rerun()

    # Main chat area
    if not st.session_state.current_thread_id:
        st.info(
            "Select 'New Chat' or choose an existing conversation from the sidebar."
        )
        st.stop()

    # Get current conversation title
    current_title = next(
        (
            c["title"]
            for c in st.session_state.conversations
            if c["thread_id"] == st.session_state.current_thread_id
        ),
        "Chat",
    )

    # Title with edit option
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.subheader(f"💬 {current_title}")
    with col2:
        if st.button("Edit Title", key="edit_title"):
            st.session_state.editing_title = True

    # Title edit form
    if st.session_state.get("editing_title", False):
        with st.form(key="title_form"):
            new_title = st.text_input("New Title", value=current_title)
            submit = st.form_submit_button("Save")

            if submit and new_title:
                db.update_conversation_title(
                    st.session_state.current_thread_id, new_title
                )

                for conv in st.session_state.conversations:
                    if conv["thread_id"] == st.session_state.current_thread_id:
                        conv["title"] = new_title
                        break
                st.session_state.editing_title = False
                st.rerun()

    messages: list[ChatMessage] = st.session_state.messages

    # Display messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter(), show_human = True)

    user_input = st.chat_input("Type your message...")

    if user_input:
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        message_id = str(uuid.uuid4())

        db.save_message(
            message_id, st.session_state.current_thread_id, "User", user_input
        )

        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=st.session_state.user_info["id"],
                )
                await draw_messages(stream, is_new=True)
                response = st.session_state.messages[-1]
                print(f"Response: {response}")
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=st.session_state.user_info["id"],
                )

                messages.append(response)
                st.chat_message("ai").write(response.content)
            
            message_id = str(uuid.uuid4())
            
            db.save_message(
                message_id, st.session_state.current_thread_id, "AI", response.content
            )
            
            print(response, message_id)
            
 
            current_conversation = next(
                (
                    c
                    for c in st.session_state.conversations
                    if c["thread_id"] == st.session_state.current_thread_id
                ),
                None,
            )

            if current_conversation and (
                current_conversation["title"] == "New Chat"
                or not current_conversation["title"]
            ):
                new_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
                db.update_conversation_title(
                    st.session_state.current_thread_id, new_title
                )

                for conv in st.session_state.conversations:
                    if conv["thread_id"] == st.session_state.current_thread_id:
                        conv["title"] = new_title
                        break
             # Give time for the message to be displayed
            st.rerun()
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()


    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) > 0 and st.session_state.last_message:
        await handle_feedback()

    # Clean up database connection
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
