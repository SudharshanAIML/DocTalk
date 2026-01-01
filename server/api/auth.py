from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.mongo import users_col
from auth.auth_utils import hash_password, verify_password
from auth.jwt_handler import create_access_token
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
