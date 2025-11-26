import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Annotated, Any, Dict
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # [NEW] Import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from jose import jwt, JWTError
from passlib.context import CryptContext
import motor.motor_asyncio
from beanie import Document, Indexed, init_beanie, PydanticObjectId
import requests 
import httpx 

# --- 1. CONFIGURATION ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')
    
    DATABASE_URL: str
    SECRET_KEY: str
    
    # Auth Keys
    GOOGLE_CLIENT_ID: str = ""
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    LINKEDIN_CLIENT_ID: str = ""
    LINKEDIN_CLIENT_SECRET: str = ""
    
    # AI Keys
    GEMINI_API_KEY: str
    
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

settings = Settings()

# --- 2. DATABASE MODELS ---

# Sub-models for User
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

# Main User Document
class User(Document):
    name: str
    email: Annotated[EmailStr, Indexed(unique=True)]
    hashed_password: str
    assessment: Optional[AssessmentData] = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    progress: UserProgress = Field(default_factory=UserProgress)
    
    class Settings:
        name = "users"

# Community Message Document
class ChatMessage(Document):
    user_name: str
    user_initial: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    community_id: str # e.g., 'pandava', 'engineers'
    
    class Settings:
        name = "community_messages"

# --- 3. API SCHEMAS ---
# Auth Schemas
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

# User Profile Schema
class UserSchema(BaseModel):
    id: str
    email: EmailStr
    name: str
    assessment: Optional[AssessmentData] = None
    preferences: UserPreferences
    progress: UserProgress
    
    class Config:
        from_attributes = True

# Assessment Schemas
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

# Feature Request Schemas
class RoadmapRequest(BaseModel):
    career_title: str

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

# --- 4. DATABASE INIT ---
async def init_db():
    print("Connecting to MongoDB...")
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.DATABASE_URL)
        await client.admin.command('ping')
        db_name = "pathify_db" 
        database = client[db_name]
        # Register all document models here
        await init_beanie(database=database, document_models=[User, ChatMessage])
        print(f"Successfully connected to MongoDB: {db_name}")
    except Exception as e:
        print(f"FATAL: MongoDB Connection Error: {e}")
        raise e


# --- PATH SETUP ---
# Get the directory where main.py is located (e.g., .../Pathify/backend)
BASE_DIR = Path(__file__).resolve().parent

# Point to the frontend folder (sibling to backend)
FRONTEND_DIR = BASE_DIR.parent / "frontend"
# --- 5. APP SETUP ---
app = FastAPI(title="Pathify API", version="2.0.0")

@app.on_event("startup")
async def on_startup():
    await init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [NEW] Mount the frontend folder to serve static assets (CSS, JS, Images)
# This allows HTML files to reference "style.css" or "images/logo.png" directly.
# We check if the folder exists first to avoid errors during development setup.
# Mount the frontend folder
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    # Mount images specifically so /images/logo.png works
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

# --- 7. GEMINI AI HELPER ---
async def call_gemini(system_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Dict[str, Any]:
    if not settings.GEMINI_API_KEY:
        return {"status": "error", "message": "Gemini API Key missing"}
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={settings.GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    
    if schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": schema.model_json_schema()
        }
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30.0)
            if resp.status_code == 200:
                result = resp.json()
                text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return {"status": "success", "content": text}
            return {"status": "error", "message": f"API Error {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ==========================================
#              API ENDPOINTS
# ==========================================

# --- AUTHENTICATION ---

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

# --- SOCIAL AUTH (Google) ---
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
            user = await create_social_user(data['email'], data.get('name', 'User'))
            
        return {"access_token": create_access_token({"email": user.email}), "token_type": "bearer", "is_new_user": is_new}
    except Exception as e:
        raise HTTPException(401, f"Google Auth Failed: {e}")

# --- SOCIAL AUTH CALLBACKS (GitHub/LinkedIn) ---
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
    # (Simplified flow: Exchange code -> Get User -> Login)
    # Note: Error handling abbreviated for speed
    token_resp = requests.post("https://github.com/login/oauth/access_token", 
        params={"client_id": settings.GITHUB_CLIENT_ID, "client_secret": settings.GITHUB_CLIENT_SECRET, "code": code},
        headers={"Accept": "application/json"}).json()
    
    user_resp = requests.get("https://api.github.com/user", headers={"Authorization": f"token {token_resp['access_token']}"}).json()
    
    # Fetch email if private
    email = user_resp.get("email")
    if not email:
        emails = requests.get("https://api.github.com/user/emails", headers={"Authorization": f"token {token_resp['access_token']}"}).json()
        email = next((e['email'] for e in emails if e['primary']), None)

    name = user_resp.get("name") or user_resp.get("login")
    
    # 4. Login/Register
    user = await get_user_by_email(email)
    is_new = False
    if not user:
        is_new = True
        user = await create_social_user(email, name)
        
    return token_html_response(create_access_token({"email": user.email}), is_new)

@app.get("/api/auth/linkedin/login")
async def li_login():
    return RedirectResponse(f"https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id={settings.LINKEDIN_CLIENT_ID}&redirect_uri=http://127.0.0.1:8000/api/auth/linkedin/callback&scope=email%20profile%20openid")

@app.get("/api/auth/linkedin/callback")
async def li_callback(code: str):
    token_resp = requests.post("https://www.linkedin.com/oauth/v2/accessToken",
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": "http://127.0.0.1:8000/api/auth/linkedin/callback", "client_id": settings.LINKEDIN_CLIENT_ID, "client_secret": settings.LINKEDIN_CLIENT_SECRET},
        headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
        
    user_info = requests.get("https://api.linkedin.com/v2/userinfo", headers={"Authorization": f"Bearer {token_resp['access_token']}"}).json()
    
    user = await get_user_by_email(user_info['email'])
    is_new = False
    if not user:
        is_new = True
        user = await create_social_user(user_info['email'], user_info['name'])
        
    return token_html_response(create_access_token({"email": user.email}), is_new)

# --- USER DATA & SETTINGS ---

@app.get("/api/users/me", response_model=UserSchema)
async def get_profile(user: User = Depends(get_current_user)):
    return {
        "id": str(user.id), 
        "email": user.email, 
        "name": user.name, 
        "assessment": user.assessment,
        "preferences": user.preferences,
        "progress": user.progress
    }

@app.put("/api/user/preferences")
async def update_prefs(prefs: UserPreferences, user: User = Depends(get_current_user)):
    user.preferences = prefs
    await user.save()
    return {"status": "success"}

# --- ASSESSMENT & CAREER MATCHING ---

@app.post("/api/generate-dynamic-questions", response_model=DynamicQuestionList)
async def dynamic_questions(answers: StaticAnswers, user: User = Depends(get_current_user)):
    prompt = f"Generate 5 follow-up career questions for: Age {answers.age}, Ed {answers.education}, Exp {answers.experience}, Curiosity {answers.curiosity}. Return JSON."
    res = await call_gemini("Career Counselor. Output JSON.", prompt, DynamicQuestionList)
    if res['status'] == 'success': return DynamicQuestionList.model_validate_json(res['content'])
    raise HTTPException(500, res['message'])

@app.post("/api/generate-matches", response_model=CareerMatchResponse)
async def matches(answers: Dict[str, Any], user: User = Depends(get_current_user)):
    prompt = f"Analyze answers and suggest 4 careers: {answers}. Return JSON."
    res = await call_gemini("Career Match Engine. Output JSON.", prompt, CareerMatchResponse)
    if res['status'] == 'success': return CareerMatchResponse.model_validate_json(res['content'])
    raise HTTPException(500, res['message'])

@app.post("/api/assessment")
async def save_assessment(data: AssessmentData, user: User = Depends(get_current_user)):
    user.assessment = data
    
    # [GAMIFICATION] Award XP for completing assessment
    user.progress.xp += 100
    user.progress.completed_courses += 1
    await user.save()
    
    return {"status": "success"}

@app.post("/api/generate-roadmap")
async def roadmap(req: RoadmapRequest, user: User = Depends(get_current_user)):
    if not user.assessment: raise HTTPException(400, "No assessment")
    prompt = f"Create a markdown learning roadmap for '{req.career_title}' based on profile: {user.assessment.original_responses}"
    res = await call_gemini("Expert Career Coach. Return Markdown.", prompt)
    if res['status'] == 'success': return {"roadmap": res['content']}
    raise HTTPException(500, res['message'])

# --- KSARTH CHAT AI ---
@app.post("/api/chat")
async def chat_with_ksarth(req: ChatRequest, user: User = Depends(get_current_user)):
    """
    Chat with Ksarth AI. Includes user context if available.
    """
    context_str = ""
    if user.assessment:
        careers = [c['title'] for c in user.assessment.selected_careers]
        context_str = f"User is interested in: {', '.join(careers)}. "
    
    system_prompt = f"You are Ksarth, a helpful and encouraging AI career assistant. {context_str} Keep answers concise and helpful."
    res = await call_gemini(system_prompt, req.message)
    
    if res['status'] == 'success': return {"reply": res['content']}
    raise HTTPException(500, res['message'])

# --- COMMUNITY FEATURES ---

@app.get("/api/community/{community_id}/messages", response_model=List[CommunityMessageResponse])
async def get_messages(community_id: str, user: User = Depends(get_current_user)):
    # Fetch last 50 messages for this community
    messages = await ChatMessage.find(ChatMessage.community_id == community_id).sort("-timestamp").limit(50).to_list()
    
    # Format for frontend
    return [
        CommunityMessageResponse(
            user_name=msg.user_name,
            user_initial=msg.user_initial,
            content=msg.content,
            timestamp=msg.timestamp.strftime("%H:%M"),
            is_own=(msg.user_name == user.name)
        ) for msg in reversed(messages)
    ]

@app.post("/api/community/messages")
async def send_message(req: CommunityMessageRequest, user: User = Depends(get_current_user)):
    msg = ChatMessage(
        user_name=user.name,
        user_initial=user.name[0].upper(),
        content=req.content,
        community_id=req.community_id
    )
    await msg.create()
    
    # [GAMIFICATION] Small XP reward for engagement
    user.progress.xp += 2
    await user.save()
    
    return {"status": "success"}

# --- STATIC FILES ---
@app.get("/")
async def root(): return RedirectResponse("/landingpage")

@app.get("/{page}")
async def serve_html(page: str):
    # Handle requests for "login.html" by stripping the extension
    page_name = page.replace(".html", "")
    
    valid_pages = ["login", "dashboard", "questions", "community", "gamification", "ksarth-chat", "workthrough", "landingpage"]
    
    if page_name in valid_pages:
        file_path = FRONTEND_DIR / f"{page_name}.html"
        if file_path.exists():
            return FileResponse(file_path)
            
    raise HTTPException(status_code=404, detail=f"Page '{page}' not found")

@app.get("/style.css")
async def serve_css():
    file_path = FRONTEND_DIR / "style.css"
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/app.js")
async def serve_js():
    file_path = FRONTEND_DIR / "app.js"
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="JS file not found")

# Note: Image serving is handled by the mounted /images static folder