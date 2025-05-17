import streamlit as st
import sqlite3
import uuid
import os
from datetime import datetime
from dotenv import load_dotenv
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from client import AgentClient, AgentClientError
from schema.schema import ChatMessage,ChatHistory
from schema.task_data import TaskData, TaskDataStatus
from collections.abc import AsyncGenerator
from pydantic import ValidationError


# Load environment variables from .env file
load_dotenv()

# Constants
APP_TITLE = "Chatbot Epargne Demo"
APP_ICON = "🤖"
USER_ID_COOKIE = "user_id"
DB_PATH = "chat_database.db"
API_KEY = os.getenv("OPENAI_API_KEY", "")

async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        
        if not isinstance(msg, ChatMessage) and not isinstance(msg, dict):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()
            
        if isinstance(msg, dict):
            # Convert dict to ChatMessage
            try:
                msg = ChatMessage.model_validate(msg)
            except ValidationError:
                st.error("Unexpected message format received from agent")
                st.write(msg)
                st.stop()
                
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    print("Handling feedback...")
    latest_run_id = st.session_state.messages[-1].get('run_id')
    print(f"Latest run ID: {latest_run_id}")
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
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


# Initialize the database
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Conversations table
    c.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Messages table
    c.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT,
        sender TEXT,
        content TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Database operations
class ChatDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.c = self.conn.cursor()
    
    def close(self):
        self.conn.close()
    
    def get_or_create_user(self, username):
        # Check if user exists
        self.c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = self.c.fetchone()
        
        if user:
            return dict(user)
        
        # Create new user
        user_id = str(uuid.uuid4())
        self.c.execute(
            'INSERT INTO users (id, username) VALUES (?, ?)',
            (user_id, username)
        )
        self.conn.commit()
        
        return {"id": user_id, "username": username}
    
    def create_conversation(self, user_id, title="New Chat"):
        conversation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        self.c.execute(
            'INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)',
            (conversation_id, user_id, title, now, now)
        )
        self.conn.commit()
        
        return {
            "thread_id": conversation_id,
            "title": title,
            "created_at": now
        }
    
    def list_conversations(self, user_id):
        self.c.execute(
            '''SELECT id as thread_id, title, created_at, updated_at 
               FROM conversations 
               WHERE user_id = ? 
               ORDER BY updated_at DESC''',
            (user_id,)
        )
        
        return [dict(row) for row in self.c.fetchall()]
    
    def get_conversation(self, thread_id, user_id=None):
        # Verify the conversation belongs to the user if user_id is provided
        if user_id:
            self.c.execute(
                'SELECT user_id FROM conversations WHERE id = ?',
                (thread_id,)
            )
            owner = self.c.fetchone()
            if not owner or owner['user_id'] != user_id:
                return None
        
        # Get all messages for this conversation
        self.c.execute(
            '''SELECT id ,sender, content, created_at 
               FROM messages 
               WHERE conversation_id = ? 
               ORDER BY created_at''',
            (thread_id,)
        )
        
        return [dict(row) for row in self.c.fetchall()]
    
    def save_message(self, message_id,conversation_id, sender, content):
        now = datetime.now().isoformat()
        
        self.c.execute(
            'INSERT INTO messages (id, conversation_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?)',
            (message_id, conversation_id, sender, content, now)
        )
        
        # Update the conversation's updated_at timestamp
        self.c.execute(
            'UPDATE conversations SET updated_at = ? WHERE id = ?',
            (now, conversation_id)
        )
        
        self.conn.commit()
        return message_id
    
    def update_conversation_title(self, thread_id, title):
        self.c.execute(
            'UPDATE conversations SET title = ? WHERE id = ?',
            (title, thread_id)
        )
        self.conn.commit()
    
    def delete_conversation(self, thread_id):
        # Delete all messages in the conversation
        self.c.execute('DELETE FROM messages WHERE conversation_id = ?', (thread_id,))
        
        # Delete the conversation
        self.c.execute('DELETE FROM conversations WHERE id = ?', (thread_id,))
        
        self.conn.commit()

# Initialize LLM and conversation chain
def get_llm_chain():
    llm = ChatOpenAI(api_key=API_KEY)
    memory = ConversationBufferMemory()
    return ConversationChain(llm=llm, memory=memory)

# Helper function to convert between UI and database message formats
def format_message_for_display(msg_dict) -> ChatMessage:
    # msg_dict comes from your SQLite rows
    msg_type = "human" if msg_dict["sender"] == "User" else "ai"
    return ChatMessage(
        type=msg_type,
        content=msg_dict["content"],
        run_id=msg_dict["id"],
    )

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
            
    if "thread_id" not in st.session_state:
            thread_id = st.query_params.get("thread_id")
            if not thread_id:
                thread_id = str(uuid.uuid4())
                messages = []
            else:
                try:
                    messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
                except AgentClientError:
                    st.error("No message history found for this Thread ID.")
                    messages = []
            st.session_state.messages = messages
            st.session_state.thread_id = thread_id    
    
    agent_client: AgentClient = st.session_state.agent_client
    
    # Main app layout
    st.title(f"{APP_ICON} {APP_TITLE}")
    
    # Sidebar
    with st.sidebar:
        st.header(f"👤 {st.session_state.user_info['username']}")
        
        # Logout button
        if st.button("🚪 Log out", use_container_width=True):
            for key in ["logged_in", "user_info", "conversations", "current_thread_id", "messages", "llm_chain"]:
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
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)

            # Display user ID (for debugging or user information)
            st.text_input("User ID (read-only)", value=st.session_state.user_info["id"], disabled=True)
        st.divider()
        
        # List existing conversationss
        if st.session_state.conversations:
            st.subheader("Your Chats")
            
            for i, conv in enumerate(st.session_state.conversations):
                title = conv["title"] or "Untitled Chat"
                if st.sidebar.button(f"{title}", key=f"convo_{i}", use_container_width=True):
                    st.session_state.current_thread_id = conv["thread_id"]
                    messages = db.get_conversation(conv["thread_id"], st.session_state.user_info["id"])
                    st.session_state.messages = [format_message_for_display(msg) for msg in messages]
                    st.session_state.llm_chain = get_llm_chain()  # Reset conversation memory
                    
                    # Load conversation history into the LLM memory
                    for msg in messages:
                        if msg["sender"] == "User":
                            st.session_state.llm_chain.memory.chat_memory.add_user_message(msg["content"])
                        else:
                            st.session_state.llm_chain.memory.chat_memory.add_ai_message(msg["content"])
                    
                    st.rerun()
                
                # Delete button with confirmation
                with st.sidebar.expander("Delete", expanded=False):
                    if st.button("Confirm Delete", key=f"delete_{i}", use_container_width=True):
                        db.delete_conversation(conv["thread_id"])
                        st.session_state.conversations.pop(i)
                        if st.session_state.current_thread_id == conv["thread_id"]:
                            st.session_state.current_thread_id = None
                            st.session_state.messages = []
                        st.rerun()
    
    # Main chat area
    if not st.session_state.current_thread_id:
        st.info("Select 'New Chat' or choose an existing conversation from the sidebar.")
        st.stop()
    
    # Get current conversation title
    current_title = next(
        (c["title"] for c in st.session_state.conversations 
         if c["thread_id"] == st.session_state.current_thread_id),
        "Chat"
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
                db.update_conversation_title(st.session_state.current_thread_id, new_title)

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

    await draw_messages(amessage_iter())
            
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        message_id = str(uuid.uuid4())

        db.save_message(message_id,
                        st.session_state.current_thread_id, 
                        "User", 
                        user_input
        )
        
        st.session_state.messages.append(
            ChatMessage(type="human", 
                        content=user_input,
                        run_id=message_id)
        )
        
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=st.session_state.user_info["id"],
                )
                await draw_messages(stream, is_new=False)
                response = st.session_state.messages[-1]
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=st.session_state.user_info["id"],
                )
                
                messages.append(response)
                st.chat_message("ai").write(response.content)
                
            print(f"Response: {response}")
            message_id = str(uuid.uuid4())
            db.save_message(
                message_id,
                st.session_state.current_thread_id, 
                "AI", 
                response.content
            )
            
            st.session_state.messages.append({
                "type": "ai",
                "content": response.content,
                "run_id": message_id
            })

            current_conversation = next(
            (c for c in st.session_state.conversations 
             if c["thread_id"] == st.session_state.current_thread_id), 
            None
        )
        
            if current_conversation and (current_conversation["title"] == "New Chat" or not current_conversation["title"]):
                # Generate a title (first 30 chars of user message)
                new_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
                db.update_conversation_title(st.session_state.current_thread_id, new_title)
                
                # Update title in session state
                for conv in st.session_state.conversations:
                    if conv["thread_id"] == st.session_state.current_thread_id:
                        conv["title"] = new_title
                        break
                    
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()
    
    # Clean up database connection
    db.close()

if __name__ == "__main__":
    asyncio.run(main())