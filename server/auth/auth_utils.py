import hashlib
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _pre_hash(password: str) -> str:
    """
    Pre-hashes the password using SHA-256 to bypass the 72-byte limit of bcrypt.
    """
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def hash_password(password: str) -> str:
    return pwd_context.hash(_pre_hash(password))

def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(_pre_hash(password), hashed_password)
