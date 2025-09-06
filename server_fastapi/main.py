# server_fastapi/main.py

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
import socketio
import json
from bson import ObjectId
from .database import profile_collection
from .models import UserProfileBase, ProgressUpdate

from .groq_client import call_sync_groq

app = FastAPI(title="Educursus Real-Time API")
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
# The ASGI app wraps the FastAPI app and the Socket.IO server
socket_app = socketio.ASGIApp(sio)
app.mount("/", socket_app)

# --- Connection Events ---
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

# --- Profile Events ---
@sio.on('profile:create')
async def create_profile(sid, data):
    """Creates a new user profile and stores it in MongoDB."""
    try:
        profile_data = UserProfileBase(**data)
        result = await profile_collection.insert_one(profile_data.model_dump(by_alias=True))
        new_profile = await profile_collection.find_one({"_id": result.inserted_id})
        new_profile["_id"] = str(new_profile["_id"]) # Convert ObjectId to string for client
        await sio.emit('profile:created', data=new_profile, to=sid)
    except Exception as e:
        await sio.emit('error', data={'message': f"Profile creation failed: {e}"}, to=sid)


@sio.on('profile:get')
async def get_profile(sid, data):
    """Fetches a user profile from MongoDB by its ID."""
    try:
        profile_id = data.get('id')
        if not profile_id or not ObjectId.is_valid(profile_id):
            raise ValueError("Invalid or missing profile ID")

        profile = await profile_collection.find_one({"_id": ObjectId(profile_id)})
        if profile:
            profile["_id"] = str(profile["_id"])
            await sio.emit('profile:data', data=profile, to=sid)
        else:
            await sio.emit('error', data={'message': "Profile not found"}, to=sid)
    except Exception as e:
        await sio.emit('error', data={'message': f"Failed to get profile: {e}"}, to=sid)


@sio.on('profile:update_progress')
async def update_progress(sid, data):
    """Updates the completion status of a skill in a user's roadmap."""
    try:
        update_data = ProgressUpdate(**data)
        profile_id = ObjectId(update_data.profileId)
        # Use dot notation to update a specific field in the 'progress' map
        update_result = await profile_collection.update_one(
            {"_id": profile_id},
            {"$set": {f"progress.{update_data.skill}": update_data.completed}}
        )
        if update_result.matched_count:
            # Send back the full updated profile
            updated_profile = await profile_collection.find_one({"_id": profile_id})
            updated_profile["_id"] = str(updated_profile["_id"])
            await sio.emit('profile:data', data=updated_profile, to=sid)
        else:
            await sio.emit('error', data={'message': 'Profile not found for progress update'}, to=sid)
    except Exception as e:
        await sio.emit('error', data={'message': f"Progress update failed: {e}"}, to=sid)


# --- AI Events ---
@sio.on('ai:get_suggestions')
async def get_ai_suggestions(sid, data):
    """Generates initial career suggestions based on a user profile."""
    profile = data.get('profile', {})
    system_prompt = (
        "You are an expert AI career counselor for Indian students. Your analysis must be practical, insightful, and grounded in the Indian job market. "
        "Analyze the provided user data to suggest three diverse and suitable career paths. "
        "For each career, provide a concise, compelling description (2-3 sentences) explaining why it's a good fit for the user. "
        "Your entire response MUST be a single, valid JSON array of 3 objects. Each object must have three string keys: 'id', 'title', and 'description'. Do not include any text outside the JSON array."
    )
    user_prompt = f"Generate career suggestions for the following user profile:\n{json.dumps(profile, indent=2)}"
    try:
        response_str = await run_in_threadpool(call_sync_groq, system_prompt=system_prompt, user_prompt=user_prompt, is_json=True)
        suggestions = json.loads(response_str)
        await sio.emit('ai:suggestions_data', data=suggestions, to=sid)
    except Exception as e:
        await sio.emit('error', data={'message': f"AI suggestion failed: {e}"}, to=sid)


@sio.on('ai:get_roadmap')
async def get_ai_roadmap(sid, data):
    """Generates a detailed learning roadmap for a chosen career."""
    career_title = data.get('careerTitle', 'the selected career')
    profile = data.get('profile', {})
    system_prompt = (
        "You are an AI curriculum developer creating a personalized learning roadmap for an Indian student. "
        "The roadmap should be structured, actionable, and broken down into logical phases (e.g., 'Phase 1: Foundational Knowledge'). "
        "Each phase must contain a list of specific, learnable skills or topics. "
        "Your entire response MUST be a single, valid JSON object. The root object should have keys that represent phase titles (e.g., 'Phase 1: ...'). "
        "The value for each phase key must be an array of strings, where each string is a skill or topic. "
        "Example format: {\"Phase 1: Foundation (Months 1-3)\": [\"Skill A\", \"Skill B\"], \"Phase 2: Core Skills (Months 4-9)\": [\"Skill C\", \"Skill D\"]}"
    )
    user_prompt = f"Create a detailed learning roadmap for a student aspiring to become a '{career_title}'. Personalize it based on their profile:\n{json.dumps(profile, indent=2)}"
    try:
        response_str = await run_in_threadpool(call_sync_groq, system_prompt=system_prompt, user_prompt=user_prompt, is_json=True)
        roadmap = json.loads(response_str)
        await sio.emit('ai:roadmap_data', data={'career': career_title, 'roadmap': roadmap}, to=sid)
    except Exception as e:
        await sio.emit('error', data={'message': f"AI roadmap generation failed: {e}"}, to=sid)


@sio.on('chat:send_message')
async def handle_chat_message(sid, data):
    """Handles a conversational chat message from the user, acting as an AI advisor."""
    user_message = data.get('message')
    profile = data.get('profile', {})
    system_prompt = (
        "You are 'Educursus AI', a friendly and professional career advisor for students in India. "
        "Your tone should be encouraging, supportive, and knowledgeable. "
        "Answer the user's questions concisely and directly, using their profile for context when relevant. "
        "Keep your answers to a few sentences unless the user asks for detail. Do not respond in JSON."
    )
    user_prompt = f"Here is my profile for context:\n{json.dumps(profile, indent=2)}\n\nMy question is: \"{user_message}\""
    try:
        # Chat responses are not expected to be JSON
        response_text = await run_in_threadpool(call_sync_groq, system_prompt=system_prompt, user_prompt=user_prompt, is_json=False)
        await sio.emit('chat:new_message', data={'sender': 'ai', 'text': response_text}, to=sid)
    except Exception as e:
        await sio.emit('error', data={'message': f"AI chat failed: {e}"}, to=sid)