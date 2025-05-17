import os
import sqlite3
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.environ["DB_PATH"]
print(f"DB_PATH: {DB_PATH}")
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
        self.c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = self.c.fetchone()

        if user:
            return dict(user)

        # Create new user
        user_id = str(uuid.uuid4())
        self.c.execute(
            "INSERT INTO users (id, username) VALUES (?, ?)", (user_id, username)
        )
        self.conn.commit()

        return {"id": user_id, "username": username}

    def create_conversation(self, user_id, title="New Chat"):
        conversation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        self.c.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, user_id, title, now, now),
        )
        self.conn.commit()

        return {"thread_id": conversation_id, "title": title, "created_at": now}

    def list_conversations(self, user_id):
        self.c.execute(
            """SELECT id as thread_id, title, created_at, updated_at 
               FROM conversations 
               WHERE user_id = ? 
               ORDER BY updated_at DESC""",
            (user_id,),
        )

        return [dict(row) for row in self.c.fetchall()]

    def get_conversation(self, thread_id, user_id=None):
        # Verify the conversation belongs to the user if user_id is provided
        if user_id:
            self.c.execute(
                "SELECT user_id FROM conversations WHERE id = ?", (thread_id,)
            )
            owner = self.c.fetchone()
            if not owner or owner["user_id"] != user_id:
                return None

        # Get all messages for this conversation
        self.c.execute(
            """SELECT id ,sender, content, created_at 
               FROM messages 
               WHERE conversation_id = ? 
               ORDER BY created_at""",
            (thread_id,),
        )

        return [dict(row) for row in self.c.fetchall()]

    def save_message(self, message_id, conversation_id, sender, content):
        now = datetime.now().isoformat()

        self.c.execute(
            "INSERT INTO messages (id, conversation_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, conversation_id, sender, content, now),
        )

        # Update the conversation's updated_at timestamp
        self.c.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )

        self.conn.commit()
        return message_id

    def update_conversation_title(self, thread_id, title):
        self.c.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, thread_id)
        )
        self.conn.commit()

    def delete_conversation(self, thread_id):
        # Delete all messages in the conversation
        self.c.execute("DELETE FROM messages WHERE conversation_id = ?", (thread_id,))

        # Delete the conversation
        self.c.execute("DELETE FROM conversations WHERE id = ?", (thread_id,))

        self.conn.commit()
