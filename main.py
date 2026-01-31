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

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
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
    
    # API Key set directly for preview environment
    GEMINI_API_KEY: str = "" 
    
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
    certifications: List[Any] = [] 
    
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

# [NEW] Rule-Based Prediction Schema (Matches the 19 questions)
class RuleBasedPredictionRequest(BaseModel):
    logical_quotient_rating: int
    hackathons: int
    coding_skills_rating: int
    public_speaking_points: int
    self_learning_capability: str 
    extra_courses_did: str 
    taken_inputs_from_seniors_or_elders: str 
    worked_in_teams_ever: str 
    introvert: str 
    management_or_technical: str 
    hard_smart_worker: str 
    certifications: str
    workshops: str
    reading_and_writing_skills: str 
    memory_capability_score: str 
    interested_subjects: str
    interested_career_area: str
    type_of_company_want_to_settle_in: str
    interested_type_of_books: str

# --- 4. DATABASE & PREDICTOR INIT ---
async def seed_certifications():
    if await Certification.count() == 0:
        certs = [
            Certification(title="Python Programming", description="Master Python fundamentals.", icon="fab fa-python", color="green", level="Beginner", duration="12 Weeks", enrolled_count=2847),
            Certification(title="JavaScript Mastery", description="Modern ES6+ and DOM manipulation.", icon="fab fa-js", color="yellow", level="Intermediate", duration="10 Weeks", enrolled_count=1923),
            Certification(title="Data Science & AI", description="Machine Learning & Neural Networks.", icon="fas fa-brain", color="purple", level="Advanced", duration="16 Weeks", enrolled_count=1234),
        ]
        await Certification.insert_many(certs)

# [NEW] Rule-Based Expert System Logic
class RuleBasedPredictor:
    def predict(self, data: RuleBasedPredictionRequest) -> str:
        """
        Deterministic Rules based on CSV patterns.
        We calculate a 'score' for each possible role based on the 19 inputs.
        """
        # Initialize scores for all potential roles
        scores = {
            "Software Engineer": 0,
            "Database Developer": 0,
            "Network Security Engineer": 0,
            "Web Developer": 0,
            "Technical Support": 0,
            "UX Designer": 0,
            "CRM Technical Developer": 0,
            "Mobile Applications Developer": 0,
            "Systems Security Administrator": 0,
            "Software Developer": 0,
            "Applications Developer": 0,
            "Software Quality Assurance (QA) / Testing": 0
        }

        # 1. Weighting based on Career Area Interest (Strong Indicator)
        area = data.interested_career_area.lower()
        if "system developer" in area or "testing" in area:
            scores["Software Engineer"] += 2
            scores["Software Developer"] += 2
            scores["Mobile Applications Developer"] += 1
        if "security" in area:
            scores["Network Security Engineer"] += 3
            scores["Systems Security Administrator"] += 3
        if "business process" in area:
            scores["CRM Technical Developer"] += 3
            scores["UX Designer"] += 1
        if "cloud" in area:
            scores["Applications Developer"] += 2
            scores["Web Developer"] += 1

        # 2. Weighting based on Certifications (Specific Skills)
        cert = data.certifications.lower()
        if "python" in cert or "full stack" in cert:
            scores["Software Engineer"] += 3
            scores["Web Developer"] += 3
            scores["Mobile Applications Developer"] += 2
        if "information security" in cert:
            scores["Network Security Engineer"] += 4
            scores["Systems Security Administrator"] += 4
        if "hadoop" in cert or "r programming" in cert:
            scores["Database Developer"] += 4
        if "distro making" in cert:
            scores["Systems Security Administrator"] += 3
            scores["Technical Support"] += 2
        if "machine learning" in cert:
            scores["Software Engineer"] += 1
            scores["UX Designer"] += 2  # Pattern from CSV

        # 3. Weighting based on Workshops
        workshop = data.workshops.lower()
        if "database" in workshop:
            scores["Database Developer"] += 3
        if "web" in workshop:
            scores["Web Developer"] += 3
        if "hacking" in workshop:
            scores["Network Security Engineer"] += 2
            scores["Systems Security Administrator"] += 2
        if "cloud" in workshop:
            scores["Applications Developer"] += 2

        # 4. Weighting based on Subjects
        subj = data.interested_subjects.lower()
        if "software engineering" in subj or "programming" in subj:
            scores["Software Engineer"] += 2
            scores["Software Developer"] += 2
        if "networks" in subj:
            scores["Network Security Engineer"] += 2
        if "hacking" in subj:
            scores["Systems Security Administrator"] += 2
        if "parallel computing" in subj: # Pattern from CSV for UX
            scores["UX Designer"] += 3
        if "data engineering" in subj:
            scores["Database Developer"] += 3

        # 5. Technical vs Management
        if data.management_or_technical.lower() == "management":
             scores["CRM Technical Developer"] += 2
             scores["Technical Support"] += 2
             scores["UX Designer"] += 2
        else: # Technical
             scores["Software Engineer"] += 1
             scores["Database Developer"] += 1
             scores["Network Security Engineer"] += 1

        # 6. Coding Skills Impact
        if data.coding_skills_rating > 7:
            scores["Software Engineer"] += 2
            scores["Mobile Applications Developer"] += 2
            scores["Web Developer"] += 2
        elif data.coding_skills_rating < 4:
            scores["Technical Support"] += 2
            scores["UX Designer"] += 1

        # 7. Type of Company preference
        comp = data.type_of_company_want_to_settle_in.lower()
        if "saas" in comp:
            scores["Software Developer"] += 1
            scores["Cloud Services"] += 1 # Mapping Applications Developer
        if "finance" in comp:
            scores["Database Developer"] += 1
            scores["Systems Security Administrator"] += 1

        # 8. Logic Quotient
        if data.logical_quotient_rating > 7:
            scores["Software Engineer"] += 1
            scores["Database Developer"] += 1

        # Return the role with the highest score
        predicted_role = max(scores, key=scores.get)
        return predicted_role

career_predictor = RuleBasedPredictor()

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
    # Use the provided API key here. 
    api_key = "" 
    
    if not api_key:
        return {"status": "error", "message": "Gemini API Key missing"}
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    
    if schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=60.0)
            if resp.status_code == 200:
                result = resp.json()
                try:
                    text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    return {"status": "success", "content": text}
                except (KeyError, IndexError):
                    return {"status": "error", "message": "Invalid Gemini response format"}
            return {"status": "error", "message": f"API Error {resp.status_code}"}
        except Exception as e:
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
    if not user:
        raise HTTPException(401, "Incorrect credentials")

    if user.hashed_password == "SOCIAL_LOGIN":
        raise HTTPException(400, "Use social login for this account")

    if not verify_password(form_data.password, user.hashed_password):
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

# [NEW] RULE-BASED PREDICTION ENDPOINT
@app.post("/api/predict-career-rule-based", response_model=CareerMatchResponse)
async def predict_career_rule_based(answers: RuleBasedPredictionRequest, user: User = Depends(get_current_user)):
    """
    Receives 19 data points from questions.html.
    Uses the RuleBasedPredictor to calculate the best career match.
    Returns the result in a format that the frontend can display.
    """
    predicted_role = career_predictor.predict(answers)
    
    # Create descriptions for the result
    descriptions = {
        "Software Engineer": "Architects and builds software solutions. Requires strong coding and logical skills.",
        "Web Developer": "Builds and maintains websites and web applications. Needs HTML, CSS, JS skills.",
        "Network Security Engineer": "Protects networks from cyber threats. Requires knowledge of security protocols.",
        "Database Developer": "Designs and manages database systems. Needs SQL and data engineering skills.",
        "Technical Support": "Assists users with technical issues. Requires good communication and problem-solving.",
        "UX Designer": "Designs user interfaces and experiences. Needs creativity and empathy.",
        "Systems Security Administrator": "Manages system security and access. Critical for data protection.",
        "Mobile Applications Developer": "Builds apps for iOS and Android devices.",
        "Applications Developer": "Develops software applications for various platforms.",
        "CRM Technical Developer": "Specializes in Customer Relationship Management software customization.",
        "Software Developer": "Generalist role involving coding, testing, and maintaining software.",
        "Software Quality Assurance (QA) / Testing": "Ensures software quality through rigorous testing."
    }
    
    desc = descriptions.get(predicted_role, "A role matching your specific skills and interests.")
    
    match_response = CareerMatchResponse(careers=[
        {
            "title": predicted_role, 
            "confidence": 0.95, 
            "description": desc, 
            "skills_match": [answers.certifications, answers.workshops, answers.interested_subjects]
        }
    ])

    # Save to DB
    user.assessment = AssessmentData(
        timestamp=datetime.utcnow().isoformat(),
        original_responses=answers.dict(),
        model_predictions=predicted_role,
        selected_careers=[c.dict() for c in match_response.careers]
    )
    user.progress.xp += 150
    await user.save()
    
    return match_response

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