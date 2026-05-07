import json
import os
import logging

log = logging.getLogger("rag.profile")

PROFILE_FILE = "user_profile.json"

def load_profile() -> dict:
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load profile: {e}")
    return {}

def save_profile(profile_data: dict):
    try:
        with open(PROFILE_FILE, "w") as f:
            json.dump(profile_data, f, indent=4)
        log.info("User profile saved.")
    except Exception as e:
        log.error(f"Failed to save profile: {e}")

def has_profile() -> bool:
    return os.path.exists(PROFILE_FILE)
