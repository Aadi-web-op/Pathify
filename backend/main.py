import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Annotated, Any, Dict
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from jose import jwt, JWTError
from passlib.context import CryptContext
import motor.motor_asyncio
from beanie import Document, Indexed, init_beanie, PydanticObjectId
import requests 
import httpx 
import json
import random
import joblib
import pandas as pd
import numpy as np

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
# [NEW] Path to your model files
MODEL_DIR = BASE_DIR / "prediction_model"

# --- 1. CONFIGURATION ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR.parent / ".env", extra='ignore')
    
    DATABASE_URL: str
    SECRET_KEY: str
    
    GOOGLE_CLIENT_ID: str = ""
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    LINKEDIN_CLIENT_ID: str = ""
    LINKEDIN_CLIENT_SECRET: str = ""
    
    GEMINI_API_KEY: str
    
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

settings = Settings()

# --- 2. DATABASE MODELS ---
class AssessmentData(BaseModel):
    timestamp: str
    original_responses: dict
    model_predictions: Any
    selected_careers: List[Any]

class UserPreferences(BaseModel):
    theme: str = "light"

class UserProgress(BaseModel):
    xp: int = 0
    level: int = 1
    completed_courses: int = 0
    certificates_earned: int = 0
    streak_days: int = 0

class User(Document):
    name: str
    email: Annotated[EmailStr, Indexed(unique=True)]
    hashed_password: str
    assessment: Optional[AssessmentData] = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    progress: UserProgress = Field(default_factory=UserProgress)
    certifications: List[Any] = [] # Stores user certification progress
    
    class Settings:
        name = "users"

class ChatMessage(Document):
    user_name: str
    user_initial: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    community_id: str
    
    class Settings:
        name = "community_messages"

class Community(Document):
    name: str
    description: str
    icon: str
    color: str
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    members_count: int = 1
    online_count: int = 0

    class Settings:
        name = "communities"

class Certification(Document):
    title: str
    description: str
    icon: str 
    color: str 
    level: str
    duration: str
    enrolled_count: int = 0
    
    class Settings:
        name = "certifications"

# --- 3. API SCHEMAS ---
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str = Field(min_length=8, max_length=72)

class Token(BaseModel):
    access_token: str
    token_type: str

class GoogleToken(BaseModel):
    token: str

class GoogleLoginResponse(Token):
    is_new_user: bool

class UserSchema(BaseModel):
    id: str
    email: EmailStr
    name: str
    assessment: Optional[AssessmentData] = None
    preferences: UserPreferences
    progress: UserProgress
    certifications: List[Any] = []
    
    class Config:
        from_attributes = True

class StaticAnswers(BaseModel):
    age: str
    education: str
    experience: str
    curiosity: int

class DynamicQuestion(BaseModel):
    id: int
    title: str
    subtitle: str
    type: str 
    options: Optional[List[str]] = None

class DynamicQuestionList(BaseModel):
    questions: List[DynamicQuestion]

class CareerMatch(BaseModel):
    title: str
    confidence: float
    description: str
    skills_match: List[str]

class CareerMatchResponse(BaseModel):
    careers: List[CareerMatch]

class RoadmapRequest(BaseModel):
    career_title: str

class RoadmapStep(BaseModel):
    title: str
    description: str
    skills: List[str]
    duration: str
    status: str 

class RoadmapPlan(BaseModel):
    career_title: str
    steps: List[RoadmapStep]

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

class CommunityMessageRequest(BaseModel):
    community_id: str
    content: str

class CommunityMessageResponse(BaseModel):
    user_name: str
    user_initial: str
    content: str
    timestamp: str
    is_own: bool

class CommunityStats(BaseModel):
    active_guilds: int
    total_members: int
    messages_today: int
    user_guilds: int

class CommunityInfo(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    color: str
    online_count: int

class CommunityCreate(BaseModel):
    name: str
    description: str
    icon: str
    color: str

class CertificationInfo(BaseModel):
    id: str
    title: str
    description: str
    icon: str
    color: str
    level: str
    duration: str
    enrolled_count: int
    is_enrolled: bool = False
    is_completed: bool = False

# [NEW] Company Challenge Model
class CompanyChallenge(BaseModel):
    id: str
    company_name: str
    role: str
    title: str
    description: str
    difficulty: str 
    xp_reward: int
    time_estimate: str
    attempts: int
    icon: str 
    color: str 

# [NEW] ML Prediction Schema
class MLPredictionRequest(BaseModel):
    # Define fields based on your model's training data columns
    # Example fields based on typical career inputs:
    certifications: Optional[str] = None
    workshops: Optional[str] = None
    memory_capability: Optional[str] = None
    interested_subjects: Optional[str] = None
    career_area_interest: Optional[str] = None
    type_of_company: Optional[str] = None
    management_technical: Optional[str] = None
    # Add other fields your model expects here as optional or required

# --- 4. DATABASE & MODEL INIT ---
async def seed_certifications():
    if await Certification.count() == 0:
        certs = [
            Certification(title="Python Programming", description="Master Python fundamentals.", icon="fab fa-python", color="green", level="Beginner", duration="12 Weeks", enrolled_count=2847),
            Certification(title="JavaScript Mastery", description="Modern ES6+ and DOM manipulation.", icon="fab fa-js", color="yellow", level="Intermediate", duration="10 Weeks", enrolled_count=1923),
            Certification(title="Data Science & AI", description="Machine Learning & Neural Networks.", icon="fas fa-brain", color="purple", level="Advanced", duration="16 Weeks", enrolled_count=1234),
        ]
        await Certification.insert_many(certs)

# [NEW] Career Prediction Model Wrapper
class CareerPredictor:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.target_encoder = None
        self.label_encoders = {}

    def load(self):
        try:
            # Update filenames if they differ in your folder
            model_path = MODEL_DIR / "career_prediction_model_v80_20251130_131358.pkl"
            encoders_path = MODEL_DIR / "model_encoders_20251130_131358.pkl"
            
            if model_path.exists() and encoders_path.exists():
                print("Loading ML Model...")
                self.model = joblib.load(model_path)
                encoder_data = joblib.load(encoders_path)
                
                self.target_encoder = encoder_data.get('target_encoder')
                self.label_encoders = encoder_data.get('label_encoders', {})
                print("ML Model Loaded Successfully.")
            else:
                print("ML Model files not found. Skipping load.")
        except Exception as e:
            print(f"Error loading ML Model: {e}")

    def predict(self, input_data: dict):
        if not self.model:
            return None
        
        try:
            # Create DataFrame
            df = pd.DataFrame([input_data])
            
            # Preprocess / Encode inputs
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    # Handle unknown labels gracefully or use encoder
                    # For simplicity, we try to transform; in prod, handle unknowns
                    try:
                         df[col] = encoder.transform(df[col])
                    except Exception:
                         # Fallback if label not seen during training: use first class or -1
                         print(f"Warning: Unknown label in {col}")
                         df[col] = 0 

            # Ensure all expected columns are present (fill missing with defaults)
            # This step depends on exactly what columns your model expects.
            # `self.model.feature_names_in_` might be available if sklearn >= 1.0
            if hasattr(self.model, 'feature_names_in_'):
                for col in self.model.feature_names_in_:
                    if col not in df.columns:
                        df[col] = 0 # Default value for missing features

                # Reorder columns to match training
                df = df[self.model.feature_names_in_]

            # Predict
            prediction_idx = self.model.predict(df)[0]
            
            # Decode prediction
            if self.target_encoder:
                prediction_label = self.target_encoder.inverse_transform([prediction_idx])[0]
                return prediction_label
            return str(prediction_idx)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None

career_predictor = CareerPredictor()

async def init_db():
    print("Connecting to MongoDB...")
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.DATABASE_URL, 
            serverSelectionTimeoutMS=30000
        )
        await client.admin.command('ping')
        db_name = "pathify_db" 
        database = client[db_name]
        await init_beanie(database=database, document_models=[User, ChatMessage, Community, Certification])
        await seed_certifications()
        print(f"Successfully connected to MongoDB: {db_name}")
    except Exception as e:
        print(f"FATAL: MongoDB Connection Error: {e}")

# --- 5. APP SETUP ---
app = FastAPI(title="Pathify API", version="2.0.0")

@app.on_event("startup")
async def on_startup():
    await init_db()
    career_predictor.load() # [NEW] Load ML Model

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    images_dir = FRONTEND_DIR / "images"
    if images_dir.exists():
        app.mount("/images", StaticFiles(directory=images_dir), name="images")

# --- 6. AUTH HELPERS ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login/")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "sub": data["email"]})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

async def get_user_by_email(email: str) -> Optional[User]:
    return await User.find_one(User.email == email)

async def create_social_user(email: str, name: str) -> User:
    db_user = User(email=email, name=name, hashed_password="SOCIAL_LOGIN")
    await db_user.create()
    return db_user

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None: raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid token")
    
    user = await get_user_by_email(email)
    if user is None: raise HTTPException(401, "User not found")
    return user

# --- 7. GEMINI AI HELPER (Improved) ---
async def call_gemini(system_prompt: str, user_prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
    if not settings.GEMINI_API_KEY:
        return {"status": "error", "message": "Gemini API Key missing"}
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={settings.GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    
    if schema:
        # Pass raw schema dictionary directly
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    
    async with httpx.AsyncClient() as client:
        try:
            # Increased timeout to 60s for complex reasoning
            resp = await client.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=60.0)
            if resp.status_code == 200:
                result = resp.json()
                try:
                    text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    return {"status": "success", "content": text}
                except (KeyError, IndexError):
                    return {"status": "error", "message": "Invalid Gemini response format"}
            
            print(f"Gemini API Error ({resp.status_code}): {resp.text}")
            return {"status": "error", "message": f"API Error {resp.status_code}"}
        except Exception as e:
            print(f"Gemini Connection Exception: {e}")
            return {"status": "error", "message": str(e)}

# ==========================================
#              API ENDPOINTS
# ==========================================

@app.post("/api/register/", response_model=Token)
async def register(user: UserCreate):
    if await get_user_by_email(user.email):
        raise HTTPException(400, "Email registered")
    new_user = User(email=user.email, name=user.name, hashed_password=hash_password(user.password))
    await new_user.create()
    return {"access_token": create_access_token({"email": new_user.email}), "token_type": "bearer"}

@app.post("/api/login/", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Incorrect credentials")
    return {"access_token": create_access_token({"email": user.email}), "token_type": "bearer"}

# --- SOCIAL AUTH ---
@app.get("/api/auth/google-client-id")
async def google_id():
    return {"client_id": settings.GOOGLE_CLIENT_ID}

@app.post("/api/auth/google", response_model=GoogleLoginResponse)
async def google_login(token: GoogleToken):
    try:
        resp = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {token.token}"})
        if resp.status_code != 200: raise ValueError("Invalid Token")
        data = resp.json()
        user = await get_user_by_email(data['email'])
        is_new = False
        if not user:
            is_new = True
            user_name = data.get('name', data['email'].split('@')[0])
            user = await create_social_user(data['email'], user_name)
        return {"access_token": create_access_token({"email": user.email}), "token_type": "bearer", "is_new_user": is_new}
    except Exception as e:
        raise HTTPException(401, f"Google Auth Failed: {str(e)}")

def token_html_response(token: str, is_new: bool):
    redirect = "/questions" if is_new else "/dashboard"
    return Response(content=f"""<html><script>
        localStorage.setItem('userToken', '{token}');
        window.location.href = '{redirect}';
    </script></html>""", media_type="text/html")

@app.get("/api/auth/github/login")
async def gh_login():
    return RedirectResponse(f"https://github.com/login/oauth/authorize?client_id={settings.GITHUB_CLIENT_ID}&scope=user:email")

@app.get("/api/auth/github/callback")
async def gh_callback(code: str):
    try:
        token_resp = requests.post("https://github.com/login/oauth/access_token", 
            params={"client_id": settings.GITHUB_CLIENT_ID, "client_secret": settings.GITHUB_CLIENT_SECRET, "code": code},
            headers={"Accept": "application/json"}).json()
        if 'error' in token_resp: return RedirectResponse("/login.html?error=github-auth-failed")
        user_resp = requests.get("https://api.github.com/user", headers={"Authorization": f"token {token_resp['access_token']}"}).json()
        email = user_resp.get("email")
        if not email:
            emails = requests.get("https://api.github.com/user/emails", headers={"Authorization": f"token {token_resp['access_token']}"}).json()
            email = next((e['email'] for e in emails if e['primary']), None)
        if not email: return RedirectResponse("/login.html?error=github-no-email")
        name = user_resp.get("name") or user_resp.get("login")
        user = await get_user_by_email(email)
        is_new = False
        if not user:
            is_new = True
            user = await create_social_user(email, name)
        return token_html_response(create_access_token({"email": user.email}), is_new)
    except Exception: return RedirectResponse("/login.html?error=github-login-failed")

@app.get("/api/auth/linkedin/login")
async def li_login():
    return RedirectResponse(f"https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id={settings.LINKEDIN_CLIENT_ID}&redirect_uri=http://127.0.0.1:8000/api/auth/linkedin/callback&scope=email%20profile%20openid")

@app.get("/api/auth/linkedin/callback")
async def li_callback(code: str):
    try:
        token_resp = requests.post("https://www.linkedin.com/oauth/v2/accessToken",
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": "http://127.0.0.1:8000/api/auth/linkedin/callback", "client_id": settings.LINKEDIN_CLIENT_ID, "client_secret": settings.LINKEDIN_CLIENT_SECRET},
            headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
        if 'error' in token_resp: return RedirectResponse("/login.html?error=linkedin-auth-failed")
        user_info = requests.get("https://api.linkedin.com/v2/userinfo", headers={"Authorization": f"Bearer {token_resp['access_token']}"}).json()
        user = await get_user_by_email(user_info['email'])
        is_new = False
        if not user:
            is_new = True
            user = await create_social_user(user_info['email'], user_info['name'])
        return token_html_response(create_access_token({"email": user.email}), is_new)
    except Exception: return RedirectResponse("/login.html?error=linkedin-login-failed")

# --- USER DATA ---
@app.get("/api/users/me", response_model=UserSchema)
async def get_profile(user: User = Depends(get_current_user)):
    return {
        "id": str(user.id), 
        "email": user.email, 
        "name": user.name, 
        "assessment": user.assessment,
        "preferences": user.preferences,
        "progress": user.progress,
        "certifications": user.certifications
    }

@app.put("/api/user/preferences")
async def update_prefs(prefs: UserPreferences, user: User = Depends(get_current_user)):
    user.preferences = prefs
    await user.save()
    return {"status": "success"}

# --- AI GENERATION (Questions & Matches) ---
@app.post("/api/generate-dynamic-questions", response_model=DynamicQuestionList)
async def dynamic_questions(answers: StaticAnswers, user: User = Depends(get_current_user)):
    # Manual Schema
    schema = {
        "type": "OBJECT",
        "properties": {
            "questions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "title": {"type": "STRING"},
                        "subtitle": {"type": "STRING"},
                        "type": {"type": "STRING"},
                        "options": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["id", "title", "subtitle", "type"]
                }
            }
        }
    }
    prompt = f"Generate 5 follow-up career questions for: Age {answers.age}, Ed {answers.education}, Exp {answers.experience}, Curiosity {answers.curiosity}. Return JSON."
    res = await call_gemini("Career Counselor. Output JSON.", prompt, schema)
    if res['status'] == 'success': 
        try: return DynamicQuestionList.model_validate_json(res['content'])
        except Exception: pass
    
    # FALLBACK QUESTIONS
    return DynamicQuestionList(questions=[
        {"id": 1, "title": "Work Style", "subtitle": "I prefer working independently.", "type": "scale"},
        {"id": 2, "title": "Creativity", "subtitle": "I enjoy designing new things.", "type": "scale"},
        {"id": 3, "title": "Logic", "subtitle": "I like solving complex puzzles.", "type": "scale"},
        {"id": 4, "title": "Leadership", "subtitle": "I am comfortable leading a team.", "type": "scale"},
        {"id": 5, "title": "Tech Interest", "subtitle": "I follow the latest tech trends.", "type": "scale"}
    ])

# [NEW] ML Prediction Endpoint
@app.post("/api/predict-career-ml", response_model=CareerMatchResponse)
async def predict_career_ml(answers: Dict[str, Any], user: User = Depends(get_current_user)):
    """
    Uses the locally trained ML model to predict careers based on input data.
    Falls back to Gemini or hardcoded data if ML prediction fails.
    """
    # Convert answers to flat dict expected by model (you might need to map keys)
    # For now, we pass the answers dict directly. Ensure frontend keys match model features.
    prediction = career_predictor.predict(answers)
    
    if prediction:
        # Create a structured response based on the single label prediction
        return CareerMatchResponse(careers=[
            {"title": prediction, "confidence": 0.9, "description": f"Predicted based on your profile using our ML model.", "skills_match": ["ML Match"]}
        ])
    
    # Fallback to Gemini if ML fails
    return await matches(answers, user)


@app.post("/api/generate-matches", response_model=CareerMatchResponse)
async def matches(answers: Dict[str, Any], user: User = Depends(get_current_user)):
    # Manual Schema
    schema = {
        "type": "OBJECT",
        "properties": {
            "careers": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                        "confidence": {"type": "NUMBER"},
                        "description": {"type": "STRING"},
                        "skills_match": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["title", "confidence", "description", "skills_match"]
                }
            }
        }
    }
    
    prompt = f"Analyze answers and suggest 4 careers: {answers}. Return JSON."
    res = await call_gemini("Career Match Engine. Output JSON.", prompt, schema)
    if res['status'] == 'success': 
        try: return CareerMatchResponse.model_validate_json(res['content'])
        except Exception: pass
    
    # FALLBACK MATCHES
    return CareerMatchResponse(careers=[
        {"title": "Software Engineer", "confidence": 0.95, "description": "Architect digital solutions.", "skills_match": ["Logic", "Code"]},
        {"title": "Product Manager", "confidence": 0.88, "description": "Bridge tech and business.", "skills_match": ["Strategy", "Comms"]},
        {"title": "Data Scientist", "confidence": 0.82, "description": "Extract insights from data.", "skills_match": ["Math", "Analysis"]},
        {"title": "UX Designer", "confidence": 0.79, "description": "Design user experiences.", "skills_match": ["Creativity", "Empathy"]}
    ])

@app.post("/api/assessment")
async def save_assessment(data: AssessmentData, user: User = Depends(get_current_user)):
    user.assessment = data
    user.progress.xp += 100
    user.progress.completed_courses += 1
    await user.save()
    return {"status": "success"}

# [CRITICAL] Fallback-enabled Roadmap Generation with CLEAN SCHEMA
@app.post("/api/generate-roadmap-json", response_model=RoadmapPlan)
async def roadmap_json(req: RoadmapRequest, user: User = Depends(get_current_user)):
    print(f"Generating roadmap for {req.career_title}...")
    
    context = user.assessment.original_responses if user.assessment else "General interest"
    
    # Manual Schema to avoid $defs error
    schema = {
        "type": "OBJECT",
        "properties": {
            "career_title": {"type": "STRING"},
            "steps": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                        "description": {"type": "STRING"},
                        "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "duration": {"type": "STRING"},
                        "status": {"type": "STRING", "enum": ["done", "current", "upcoming"]}
                    },
                    "required": ["title", "description", "skills", "duration", "status"]
                }
            }
        },
        "required": ["career_title", "steps"]
    }
    
    prompt = f"""
    Create a structured learning roadmap for: '{req.career_title}'.
    User Context: {context}
    
    Generate 5 phases.
    Step 1 status: 'done'.
    Step 2 status: 'current'.
    Steps 3-5 status: 'upcoming'.
    """
    
    res = await call_gemini("Expert Career Coach. Output strictly JSON.", prompt, schema)
    
    if res['status'] == 'success': 
        try:
            return RoadmapPlan.model_validate_json(res['content'])
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            
    print("Using Fallback Roadmap Data due to AI failure.")
    
    # FALLBACK ROADMAP (Prevents UI crash)
    return RoadmapPlan(
        career_title=req.career_title,
        steps=[
            RoadmapStep(title="Foundations", description=f"Core concepts of {req.career_title}", skills=["Basics", "Principles"], duration="4 weeks", status="done"),
            RoadmapStep(title="Skill Building", description="Learning key tools and technologies", skills=["Tools", "Frameworks"], duration="6 weeks", status="current"),
            RoadmapStep(title="Real-world Projects", description="Apply knowledge to practical tasks", skills=["Project Mgmt", "Implementation"], duration="5 weeks", status="upcoming"),
            RoadmapStep(title="Advanced Specialization", description="Mastering complex topics", skills=["Advanced Tech", "Optimization"], duration="4 weeks", status="upcoming"),
            RoadmapStep(title="Career Launch", description="Portfolio and Interview Prep", skills=["Interviewing", "Resume"], duration="Ongoing", status="upcoming")
        ]
    )

@app.post("/api/chat")
async def chat_with_ksarth(req: ChatRequest, user: User = Depends(get_current_user)):
    context_str = f"User: {user.name}. "
    if user.assessment and user.assessment.selected_careers:
        careers = [c['title'] for c in user.assessment.selected_careers]
        context_str += f"Interested in: {', '.join(careers)}. "
    
    system_prompt = f"You are Ksarth, a helpful AI career assistant. {context_str} Keep answers concise."
    res = await call_gemini(system_prompt, req.message)
    
    if res['status'] == 'success': return {"reply": res['content']}
    return {"reply": "I'm currently offline for maintenance, but keep pushing forward on your roadmap!"}

# --- COMMUNITY ---
@app.get("/api/communities", response_model=List[CommunityInfo])
async def get_communities(user: User = Depends(get_current_user)):
    communities = await Community.find_all().to_list()
    if not communities:
        return [
            CommunityInfo(id="pandava", name="Pandava", description="Warriors of Career Growth", icon="🏹", color="orange", online_count=random.randint(80, 120)),
            CommunityInfo(id="engineers", name="Engineers", description="Building Tomorrow", icon="⚙️", color="blue", online_count=random.randint(130, 200)),
            CommunityInfo(id="doctors", name="Med Minds", description="Healing & Science", icon="⚕️", color="green", online_count=random.randint(40, 90)),
            CommunityInfo(id="business", name="Biz Leaders", description="Entrepreneurial Spirits", icon="💼", color="purple", online_count=random.randint(100, 150))
        ]
    return [
        CommunityInfo(
            id=str(c.id), name=c.name, description=c.description, icon=c.icon, color=c.color,
            online_count=c.online_count + random.randint(5, 20)
        ) for c in communities
    ]

@app.post("/api/communities", response_model=CommunityInfo)
async def create_community(req: CommunityCreate, user: User = Depends(get_current_user)):
    new_community = Community(
        name=req.name, description=req.description, icon=req.icon, color=req.color,
        created_by=user.email, online_count=1
    )
    await new_community.create()
    return CommunityInfo(
        id=str(new_community.id), name=new_community.name, description=new_community.description,
        icon=new_community.icon, color=new_community.color, online_count=1
    )

@app.get("/api/community/stats", response_model=CommunityStats)
async def get_community_stats(user: User = Depends(get_current_user)):
    total_msgs = await ChatMessage.count()
    active_guilds = await Community.count() or 4
    return CommunityStats(
        active_guilds=active_guilds, total_members=2847 + active_guilds * 10,
        messages_today=total_msgs + random.randint(10, 50), user_guilds=3
    )

@app.get("/api/community/{community_id}/messages", response_model=List[CommunityMessageResponse])
async def get_messages(community_id: str, user: User = Depends(get_current_user)):
    messages = await ChatMessage.find(ChatMessage.community_id == community_id).sort("-timestamp").limit(50).to_list()
    return [
        CommunityMessageResponse(
            user_name=msg.user_name, user_initial=msg.user_initial, content=msg.content,
            timestamp=msg.timestamp.strftime("%H:%M"), is_own=(msg.user_name == user.name)
        ) for msg in reversed(messages)
    ]

@app.post("/api/community/messages")
async def send_message(req: CommunityMessageRequest, user: User = Depends(get_current_user)):
    msg = ChatMessage(user_name=user.name, user_initial=user.name[0].upper(), content=req.content, community_id=req.community_id)
    await msg.create()
    return {"status": "success"}

@app.get("/api/certifications", response_model=List[CertificationInfo])
async def get_certifications(user: User = Depends(get_current_user)):
    certs = await Certification.find_all().to_list()
    user_cert_ids = {c['cert_id'] for c in user.certifications}
    return [CertificationInfo(id=str(c.id), title=c.title, description=c.description, icon=c.icon, color=c.color, level=c.level, duration=c.duration, enrolled_count=c.enrolled_count, is_enrolled=(str(c.id) in user_cert_ids)) for c in certs]

@app.post("/api/certifications/{cert_id}/enroll")
async def enroll_certification(cert_id: str, user: User = Depends(get_current_user)):
    if any(c['cert_id'] == cert_id for c in user.certifications): return {"status": "already_enrolled"}
    cert = await Certification.get(PydanticObjectId(cert_id))
    if not cert: raise HTTPException(404, "Certificate not found")
    user.certifications.append({"cert_id": str(cert.id), "title": cert.title, "status": "in-progress", "progress": 0})
    await user.save()
    return {"status": "success"}

@app.post("/api/certifications/{cert_id}/complete")
async def complete_certification(cert_id: str, user: User = Depends(get_current_user)):
    found = False
    for c in user.certifications:
        if c['cert_id'] == cert_id:
            c['status'] = 'completed'
            c['progress'] = 100
            c['score'] = 95.0
            c['completed_at'] = datetime.utcnow().isoformat()
            c['certificate_uid'] = f"PTH-{datetime.utcnow().year}-{str(user.id)[-4:]}-{cert_id[-4:]}".upper()
            user.progress.certificates_earned += 1
            user.progress.xp += 500
            found = True
            break
            
    if not found:
        # Auto enroll and complete if testing
        cert = await Certification.get(PydanticObjectId(cert_id))
        if cert:
            new_cert = {
                "cert_id": str(cert.id),
                "title": cert.title,
                "status": "completed",
                "progress": 100,
                "score": 98.0,
                "completed_at": datetime.utcnow().isoformat(),
                "certificate_uid": f"PTH-{datetime.utcnow().year}-{str(user.id)[-4:]}-{cert_id[-4:]}".upper()
            }
            user.certifications.append(new_cert)
            user.progress.certificates_earned += 1
        else:
             raise HTTPException(404, "Certificate not found")
             
    await user.save()
    return {"status": "success"}

@app.get("/api/certifications/{cert_id}/pdf")
async def generate_certificate_pdf(cert_id: str, user: User = Depends(get_current_user)):
    # Check if user actually completed it
    user_cert = next((c for c in user.certifications if c['cert_id'] == cert_id and c['status'] == 'completed'), None)
    if not user_cert:
        raise HTTPException(400, "Certificate not completed or found")
        
    dummy_pdf_content = f"""
    PATHIFY CERTIFICATE OF COMPLETION
    ---------------------------------
    Awarded To: {user.name}
    Course: {user_cert['title']}
    Date: {user_cert['completed_at']}
    Score: {user_cert['score']}%
    ID: {user_cert['certificate_uid']}
    
    Verified by Pathify AI.
    """
    return Response(content=dummy_pdf_content, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={user.name}_Certificate.pdf"})


@app.get("/api/company/challenges", response_model=List[CompanyChallenge])
async def get_company_challenges(user: User = Depends(get_current_user)):
    return [
        CompanyChallenge(id="google-search", company_name="Google", role="Software Engineering", title="Search Algo Optimization", description="Design a search algorithm handling 10M+ queries/second with sub-100ms latency.", difficulty="Hard", xp_reward=150, time_estimate="3-4 hours", attempts=1247, icon="fab fa-google", color="blue"),
        CompanyChallenge(id="microsoft-azure", company_name="Microsoft", role="Cloud Architecture", title="Azure Cost Optimization", description="Reduce infrastructure costs by 40% for a generic Fortune 500 client.", difficulty="Medium", xp_reward=100, time_estimate="2-3 hours", attempts=892, icon="fab fa-microsoft", color="green"),
        CompanyChallenge(id="tesla-autopilot", company_name="Tesla", role="AI / Computer Vision", title="Autopilot Path Planning", description="Develop real-time path planning logic for urban environments with dynamic obstacles.", difficulty="Expert", xp_reward=300, time_estimate="6 hours", attempts=450, icon="fas fa-car-battery", color="red"),
        CompanyChallenge(id="apple-privacy", company_name="Apple", role="iOS Security", title="Zero-Knowledge Analytics", description="Architect a user analytics system that preserves complete user privacy.", difficulty="Medium", xp_reward=120, time_estimate="3 hours", attempts=643, icon="fab fa-apple", color="gray"),
        CompanyChallenge(id="amazon-logistics", company_name="Amazon", role="Operations Research", title="Last-Mile Delivery", description="Optimize delivery routes for a fleet of 500 drones in a metropolitan area.", difficulty="Hard", xp_reward=200, time_estimate="4-5 hours", attempts=920, icon="fab fa-amazon", color="orange"),
        CompanyChallenge(id="netflix-stream", company_name="Netflix", role="Backend Engineering", title="Global CDN Balancing", description="Balance streaming load across 3 continents during a major premiere.", difficulty="Hard", xp_reward=180, time_estimate="4 hours", attempts=780, icon="fas fa-play", color="red")
    ]

@app.post("/api/company/challenges/{challenge_id}/start")
async def start_challenge(challenge_id: str, user: User = Depends(get_current_user)):
    user.progress.xp += 10
    await user.save()
    return {"status": "success", "message": "Challenge started! Good luck."}

# --- STATIC FILES ---
@app.get("/")
async def root(): return RedirectResponse("/landingpage")

@app.get("/{page}")
async def serve_html(page: str):
    page_name = page.replace(".html", "")
    valid_pages = ["login", "dashboard", "questions", "community", "gamification", "ksarth-chat", "workthrough", "landingpage", "certifications", "company_chat"]
    if page_name in valid_pages:
        file_path = FRONTEND_DIR / f"{page_name}.html"
        if file_path.exists(): return FileResponse(file_path)
    raise HTTPException(status_code=404, detail=f"Page '{page}' not found")

@app.get("/style.css")
async def serve_css():
    if (FRONTEND_DIR / "style.css").exists(): return FileResponse(FRONTEND_DIR / "style.css")
    raise HTTPException(404)
    
@app.get("/app.js")
async def serve_js():
    if (FRONTEND_DIR / "app.js").exists(): return FileResponse(FRONTEND_DIR / "app.js")
    raise HTTPException(404)