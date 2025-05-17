-- schema/chat_schema.sql
CREATE TABLE users (
    username TEXT PRIMARY KEY  -- if we want to keep a user table (optional)
);

CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    username TEXT REFERENCES users(username),
    title TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id),
    role TEXT CHECK (role IN ('user','assistant')),
    content TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);
