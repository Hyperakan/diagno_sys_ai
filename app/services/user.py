import json
import logging
from pathlib import Path
from typing import Dict

from fastapi import HTTPException

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin + Firestore
cred_path = Path(__file__).parent.parent / "firebase_and_users/firebase-creds.json"
if not firebase_admin._apps:
    cred = credentials.Certificate(str(cred_path))
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Path for local JSON cache
JSON_PATH = Path(__file__).parent.parent / "firebase_and_users/users_cache.json"

# Ensure the cache directory exists
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_cache() -> Dict[str, dict]:
    """
    Load the user cache from disk. Returns an empty dict if the file doesn't exist.
    """
    if JSON_PATH.exists():
        logging.debug(f"Loading cache from {JSON_PATH.resolve()}")
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    logging.debug(f"No cache file found at {JSON_PATH.resolve()}, starting fresh.")
    return {}


def save_cache(cache: Dict[str, dict]) -> None:
    """
    Save the cache to disk, converting any datetime-like objects into ISO strings,
    and creating the file if it doesn't exist.
    """
    # (Parent directory was already created at import time)
    logging.debug(f"Saving cache to {JSON_PATH.resolve()}")
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            cache,
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o),
        )


def fetch_user(uid: str) -> list[dict]:
    """
    Fetch all allergy documents for a given user UID.
    Raises 404 HTTPException if the user has no allergy records.
    """
    logging.info(f"Fetching user {uid} allergies from Firestore")
    snapshots = db.collection("users") \
                  .document(uid) \
                  .collection("allergies") \
                  .get()

    # If there are no documents, raise a 404
    if not snapshots:
        raise HTTPException(
            status_code=404,
            detail=f"No allergy records found for user {uid}"
        )

    # Option A: list comprehension
    allergies = [snap.to_dict() for snap in snapshots]

    # Option B: built-in map (less common for dict construction)
    # allergies = list(map(lambda snap: snap.to_dict(), snapshots))

    return allergies


async def get_users(uid: str):
    """
    Retrieve user info, first checking local cache; otherwise fetch from Firestore
    and update the cache.
    Returns a dict with 'user' and 'source' keys.
    """
    cache = load_cache()

    if uid in cache:
        logging.info(f"Cache'den kullanıcı bilgileri alınıyor: {uid}")
        return {"user": cache[uid], "source": "cache"}

    logging.info(f"Cache'de kullanıcı bilgileri bulunamadı: {uid}")
    user_info = fetch_user(uid)
    logging.info(f"Firebase'den kullanıcı bilgileri alındı: {uid}")

    # Update cache and save to disk (this will create the JSON file if needed)
    cache[uid] = user_info
    save_cache(cache)
    user_allergires = "".join([f"{allergy['name']} " for allergy in user_info])
    return {"user_allergies": user_allergires}
