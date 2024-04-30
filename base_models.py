from pydantic import BaseModel


class User(BaseModel):
    id: int
    name: str
    phone: int
    email: str
    password: str


class UserCreate(BaseModel):
    id: int
    name: str
    phone: int
    email: str
    password: str