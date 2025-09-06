# server_fastapi/database.py

import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set.")

client = AsyncIOMotorClient(MONGO_URI)
database = client.educursusDB
profile_collection = database.get_collection("userprofiles")