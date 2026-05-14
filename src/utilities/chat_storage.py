"""
chat_storage.py - Conversation History Manager (JSONL version)

This module handles the persistence of chat sessions using JSON Lines (.jsonl).
Each file starts with a metadata line, followed by individual message objects
with sequential IDs.
"""
import os
import json
import uuid
from datetime import datetime

SESSION_DIR = "./chat_sessions"

def _init_dir():
    """Ensures the chat session storage directory exists."""
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR, exist_ok=True)

def list_sessions() -> list:
    """
    Scans the session directory and returns a list of all saved chat sessions.
    Optimized: Only reads the first line (metadata) of each .jsonl file.
    
    Returns:
        list: A list of dictionaries containing session ID, title, timestamp, and file path,
              sorted by most recent first.
    """
    _init_dir()
    sessions = []
    if not os.path.exists(SESSION_DIR):
        return []
        
    for f in os.listdir(SESSION_DIR):
        if f.endswith(".jsonl"):
            path = os.path.join(SESSION_DIR, f)
            try:
                with open(path, "r") as file:
                    # Read only the first line for metadata
                    first_line = file.readline()
                    if first_line:
                        data = json.loads(first_line)
                        sessions.append({
                            "id": data.get("id"),
                            "title": data.get("title", "Untitled Chat"),
                            "timestamp": data.get("timestamp", 0),
                            "path": path
                        })
            except Exception:
                pass
    return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

def load_session(session_id: str) -> list:
    """
    Loads the message history for a specific session from its JSONL file.
    
    Args:
        session_id: The unique identifier for the session.
        
    Returns:
        list: A list of message dictionaries (role, content, and id).
    """
    _init_dir()
    path = os.path.join(SESSION_DIR, f"{session_id}.jsonl")
    messages = []
    if os.path.exists(path):
        try:
            with open(path, "r") as file:
                lines = file.readlines()
                # Skip the first line (metadata)
                for line in lines[1:]:
                    if line.strip():
                        messages.append(json.loads(line))
        except Exception:
            pass
    return messages

def save_session(session_id: str, messages: list):
    """
    Saves or updates a chat session to a JSONL file.
    Automatically generates a title based on the first user message.
    Injects sequential IDs (1, 2, 3...) into messages.
    
    Args:
        session_id: The unique identifier for the session.
        messages: The list of message dictionaries to persist.
    """
    _init_dir()
    if not messages:
        return
        
    # Auto-generate a title from the first user message if possible
    title = "New Chat"
    for m in messages:
        if m["role"] == "user":
            title = m["content"][:30] + ("..." if len(m["content"]) > 30 else "")
            break
            
    # Add sequential IDs to messages
    for i, msg in enumerate(messages, 1):
        msg["id"] = i

    metadata = {
        "type": "metadata",
        "id": session_id,
        "title": title,
        "timestamp": datetime.now().timestamp()
    }

    path = os.path.join(SESSION_DIR, f"{session_id}.jsonl")
    with open(path, "w") as file:
        # Line 1: Metadata
        file.write(json.dumps(metadata) + "\n")
        # Subsequent lines: Messages
        for msg in messages:
            file.write(json.dumps(msg) + "\n")

def generate_session_id() -> str:
    """Generates a unique random UUID for a new chat session."""
    return str(uuid.uuid4())
