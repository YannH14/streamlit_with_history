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

# Load environment variables from .env file
load_dotenv()

# Constants
APP_TITLE = "Chatbot Epargne Demo"
APP_ICON = "🤖"
USER_ID_COOKIE = "user_id"
DB_PATH = "chat_database.db"
API_KEY = os.getenv("OPENAI_API_KEY", "")

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
            '''SELECT sender, content, created_at 
               FROM messages 
               WHERE conversation_id = ? 
               ORDER BY created_at''',
            (thread_id,)
        )
        
        return [dict(row) for row in self.c.fetchall()]
    
    def save_message(self, conversation_id, sender, content):
        message_id = str(uuid.uuid4())
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
def format_message_for_display(msg):
    sender_type = "human" if msg["sender"] == "User" else "ai"
    return {"type": sender_type, "content": msg["content"]}

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
            
            # Set session state
            st.session_state.logged_in = True
            st.session_state.user_info = user_info
            st.session_state.conversations = db.list_conversations(user_info["id"])
            st.session_state.llm_chain = get_llm_chain()
            
            st.rerun()
        
        st.stop()
    
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
        
        st.divider()
        
        # List existing conversations
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
                # Update title in session state
                for conv in st.session_state.conversations:
                    if conv["thread_id"] == st.session_state.current_thread_id:
                        conv["title"] = new_title
                        break
                st.session_state.editing_title = False
                st.rerun()
    
    # Display messages
    if st.session_state.messages:
        for msg in st.session_state.messages:
            with st.chat_message(msg["type"]):
                st.write(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    if user_input:
        # Show user message immediately
        st.chat_message("human").write(user_input)
        
        # Save user message to database
        db.save_message(st.session_state.current_thread_id, "User", user_input)
        
        # Add message to session state
        st.session_state.messages.append({"type": "human", "content": user_input})
        
        # Generate AI response
        with st.spinner("Thinking..."):
            response = st.session_state.llm_chain.predict(input=user_input)
        
        # Show AI response
        st.chat_message("ai").write(response)
        
        # Save AI response to database
        db.save_message(st.session_state.current_thread_id, "AI", response)
        
        # Add AI message to session state
        st.session_state.messages.append({"type": "ai", "content": response})
        
        # If this is a new chat with a generic title, update the title based on the first message
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
        
        st.rerun()
    
    # Clean up database connection
    db.close()

if __name__ == "__main__":
    asyncio.run(main())