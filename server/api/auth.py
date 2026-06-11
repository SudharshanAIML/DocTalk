from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from db.mongo import users_col
from auth.auth_utils import hash_password, verify_password
from auth.jwt_handler import create_access_token
from auth.dependencies import get_current_user
from storage.session_manager import cleanup_user_session
import uuid

router = APIRouter(prefix="/auth", tags=["Auth"])

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/register")
def register(data: RegisterRequest):
    if users_col.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user = {
        "user_id": str(uuid.uuid4()),
        "email": data.email,
        "password": hash_password(data.password)
    }
    users_col.insert_one(user)

    return {"message": "User registered successfully"}

@router.post("/login")
def login(data: LoginRequest):
    user = users_col.find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"user_id": user["user_id"]})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/demo")
def demo_login():
    """Login with a temporary demo account (auto-created if not exists)"""
    demo_email = "demo@doctalk.app"
    demo_password = "Demo@12345"

    user = users_col.find_one({"email": demo_email})
    if not user:
        user = {
            "user_id": str(uuid.uuid4()),
            "email": demo_email,
            "password": hash_password(demo_password),
            "is_demo": True,
        }
        users_col.insert_one(user)

    token = create_access_token({"user_id": user["user_id"]})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/logout")
def logout(user: dict = Depends(get_current_user)):
    """
    Logout user and clean up their session.
    This will delete:
    - FAISS index from cloud storage
    - Documents metadata from database
    - Chunks metadata from database
    - Chat history from database
    """
    user_id = user["user_id"]
    
    # Clean up all user data
    success = cleanup_user_session(user_id)
    
    if success:
        return {"message": "Logged out successfully. All session data has been cleaned up."}
    else:
        raise HTTPException(status_code=500, detail="Error during logout cleanup")
