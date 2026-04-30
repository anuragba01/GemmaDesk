import os
import json
import uuid
from datetime import datetime

SESSION_DIR = "./chat_sessions"

def _init_dir():
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR, exist_ok=True)

def list_sessions() -> list:
    _init_dir()
    sessions = []
    for f in os.listdir(SESSION_DIR):
        if f.endswith(".json"):
            path = os.path.join(SESSION_DIR, f)
            try:
                with open(path, "r") as file:
                    data = json.load(file)
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
    _init_dir()
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as file:
                return json.load(file).get("messages", [])
        except Exception:
            pass
    return []

def save_session(session_id: str, messages: list):
    _init_dir()
    if not messages:
        return
        
    title = "New Chat"
    for m in messages:
        if m["role"] == "user":
            title = m["content"][:30] + ("..." if len(m["content"]) > 30 else "")
            break
            
    data = {
        "id": session_id,
        "title": title,
        "timestamp": datetime.now().timestamp(),
        "messages": messages
    }
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    with open(path, "w") as file:
        json.dump(data, file, indent=2)

def generate_session_id() -> str:
    return str(uuid.uuid4())
