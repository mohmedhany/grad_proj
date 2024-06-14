from pydantic import BaseModel


class User(BaseModel):
    id: int
    name: str
    phone: int
    email: str
    password: str = None


class UserCreate(BaseModel):
    id: int
    name: str
    phone: int
    email: str
    password: str
