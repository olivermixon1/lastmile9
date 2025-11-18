from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# -----------------------------------------------------
# AI CONFIG (Step 4)
# -----------------------------------------------------
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (.env file)
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt(prompt: str) -> str:
    """
    Helper function to call GPT and return plain text.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a logistics AI assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Internal Error: {str(e)}"


# -----------------------------------------------------
# FastAPI App Setup
# -----------------------------------------------------
app = FastAPI(
    title="Last-Mile Logistics API",
    description="Backend API for connecting autonomous trucking companies with local truckers.",
    version="3.0.0"
)

# Allow access from iOS app + HTML dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open for development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


# -----------------------------------------------------
# AUTH CONFIG
# -----------------------------------------------------
JWT_SECRET = "SUPER_SECRET_KEY_CHANGE_THIS"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60 * 24  # 1 day token

security = HTTPBearer()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def auth_driver(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    driver_id = payload.get("driver_id")

    for d in drivers:
        if d.id == driver_id:
            return d

    raise HTTPException(status_code=404, detail="Driver not found")


# -----------------------------------------------------
# DATA MODELS
# -----------------------------------------------------
class Driver(BaseModel):
    id: int
    name: str
    email: str
    password_hash: str
    phone: Optional[str] = None
    vehicle_type: Optional[str] = None
    current_location: Optional[str] = None


class DriverCreate(BaseModel):
    name: str
    email: str
    password: str
    phone: Optional[str] = None
    vehicle_type: Optional[str] = None
    current_location: Optional[str] = None


class DriverLogin(BaseModel):
    email: str
    password: str


class Job(BaseModel):
    id: int
    pickup_location: str
    dropoff_location: str
    load_description: str
    status: str  # available, assigned, completed
    driver_id: Optional[int] = None

    # Optional job details
    weight: Optional[str] = None
    equipment_type: Optional[str] = None
    delivery_window: Optional[str] = None
    contact_phone: Optional[str] = None
    price_offered: Optional[float] = None


class JobCreate(BaseModel):
    pickup_location: str
    dropoff_location: str
    load_description: str
    weight: Optional[str] = None
    equipment_type: Optional[str] = None
    delivery_window: Optional[str] = None
    contact_phone: Optional[str] = None
    price_offered: Optional[float] = None


# -----------------------------------------------------
# AI MODELS (Step 5)
# -----------------------------------------------------

# -------- AI Job Matching --------
class AIDriver(BaseModel):
    id: int
    name: str
    rating: float
    distance_from_job: float
    reliability_score: float
    equipment: str

class AIJobDetails(BaseModel):
    job_id: int
    pickup: str
    dropoff: str
    load_type: str
    scheduled_time: str
    weight: float

class MatchRequest(BaseModel):
    job: AIJobDetails
    drivers: List[AIDriver]

class MatchResponse(BaseModel):
    result: str


# -------- AI Route Optimization --------
class RouteRequest(BaseModel):
    pickup: str
    dropoff: str
    priority: str  # fastest, cheapest, safe, no-traffic

class RouteResponse(BaseModel):
    optimized_route: str


# -------- AI ETA Prediction --------
class ETARequest(BaseModel):
    distance_miles: float
    current_traffic: str
    weather: str
    driver_rating: float

class ETAResponse(BaseModel):
    eta: str


# -------- AI Chatbot --------
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


# -----------------------------------------------------
# IN-MEMORY DATA STORES
# -----------------------------------------------------
drivers: List[Driver] = []
next_driver_id = 1

jobs: List[Job] = [
    Job(id=1, pickup_location="Amazon SODO", dropoff_location="Bellevue Downtown",
        load_description="Pallet of boxes", status="available"),
    Job(id=2, pickup_location="Port of Seattle Terminal 18", dropoff_location="Redmond Microsoft Campus",
        load_description="Electronics container", status="available"),
]

next_job_id = 3


# -----------------------------------------------------
# ROOT
# -----------------------------------------------------
@app.get("/")
def home():
    return {"message": "Last-Mile Logistics API is live!"}


# -----------------------------------------------------
# AUTH ENDPOINTS
# -----------------------------------------------------
@app.post("/drivers/signup")
def signup(driver: DriverCreate):
    global next_driver_id

    for d in drivers:
        if d.email == driver.email:
            raise HTTPException(status_code=400, detail="Email already registered")

    new_driver = Driver(
        id=next_driver_id,
        name=driver.name,
        email=driver.email,
        password_hash=hash_password(driver.password),
        phone=driver.phone,
        vehicle_type=driver.vehicle_type,
        current_location=driver.current_location or "Unknown",
    )

    drivers.append(new_driver)
    next_driver_id += 1

    token = create_token({"driver_id": new_driver.id, "email": new_driver.email})

    return {
        "token": token,
        "driver": {"id": new_driver.id, "name": new_driver.name, "email": new_driver.email}
    }


@app.post("/drivers/login")
def login(credentials: DriverLogin):
    for d in drivers:
        if d.email == credentials.email:
            if verify_password(credentials.password, d.password_hash):
                token = create_token({"driver_id": d.id, "email": d.email})
                return {"token": token, "driver": {"id": d.id, "name": d.name, "email": d.email}}
            raise HTTPException(status_code=401, detail="Incorrect password")
    raise HTTPException(status_code=404, detail="Driver not found")


@app.get("/drivers/me")
def get_current_driver(current_driver: Driver = Depends(auth_driver)):
    return current_driver


# -----------------------------------------------------
# DRIVER LIST (UNPROTECTED)
# -----------------------------------------------------
@app.get("/drivers", response_model=List[Driver])
def get_drivers():
    return drivers


# -----------------------------------------------------
# JOBS
# -----------------------------------------------------
@app.get("/jobs", response_model=List[Job])
def get_jobs():
    return jobs


@app.post("/jobs", response_model=Job)
def create_job(job: JobCreate):
    global next_job_id

    new_job = Job(
        id=next_job_id,
        pickup_location=job.pickup_location,
        dropoff_location=job.dropoff_location,
        load_description=job.load_description,
        status="available",
        weight=job.weight,
        equipment_type=job.equipment_type,
        delivery_window=job.delivery_window,
        contact_phone=job.contact_phone,
        price_offered=job.price_offered
    )

    jobs.append(new_job)
    next_job_id += 1
    return new_job


@app.post("/assign/{job_id}/{driver_id}")
def assign_driver(job_id: int, driver_id: int):
    for job in jobs:
        if job.id == job_id:
            job.status = "assigned"
            job.driver_id = driver_id
            return {"message": "Driver assigned", "job": job}
    raise HTTPException(status_code=404, detail="Job not found")


@app.post("/complete/{job_id}")
def complete_job(job_id: int):
    for job in jobs:
        if job.id == job_id:
            job.status = "completed"
            return {"message": "Job completed", "job": job}
    raise HTTPException(status_code=404, detail="Job not found")


# -----------------------------------------------------
# AI ENDPOINTS (FULL PHASE 1)
# -----------------------------------------------------
@app.post("/ai/match-driver", response_model=MatchResponse)
async def ai_match_driver(data: MatchRequest):
    prompt = f"""
    You are an AI logistics dispatcher.

    JOB:
    {data.job}

    DRIVERS:
    {data.drivers}

    Select:
    - Best driver ID
    - Ranked list of all drivers
    - Short explanation

    Consider:
    distance, reliability, rating, equipment, load requirements.
    """

    result = call_gpt(prompt)
    return MatchResponse(result=result)


@app.post("/ai/optimize-route", response_model=RouteResponse)
async def ai_route_optimize(data: RouteRequest):
    prompt = f"""
    Optimize a route.

    Pickup: {data.pickup}
    Dropoff: {data.dropoff}
    Priority: {data.priority}

    Provide:
    - Strategy
    - Rough ETA
    - Truck restrictions
    - Traffic considerations

    Keep it short and useful for drivers.
    """

    result = call_gpt(prompt)
    return RouteResponse(optimized_route=result)


@app.post("/ai/eta", response_model=ETAResponse)
async def ai_eta(data: ETARequest):
    prompt = f"""
    Estimate ETA.

    Distance: {data.distance_miles} miles
    Traffic: {data.current_traffic}
    Weather: {data.weather}
    Driver rating: {data.driver_rating}

    Give:
    - ETA in minutes
    - Explanation
    """

    result = call_gpt(prompt)
    return ETAResponse(eta=result)


@app.post("/ai/chat", response_model=ChatResponse)
async def ai_chat(data: ChatMessage):
    prompt = f"""
    You are a helpful assistant for drivers and shippers.

    User message:
    {data.message}

    Keep answer clear and friendly.
    """

    result = call_gpt(prompt)
    return ChatResponse(reply=result)


# -----------------------------------------------------
# HTML DASHBOARDS
# -----------------------------------------------------
@app.get("/shipper")
def shipper_dashboard():
    return FileResponse("static/shipper_dashboard.html")


@app.get("/admin")
def admin_dashboard():
    return FileResponse("static/admin_dashboard.html")
