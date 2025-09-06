# server_fastapi/models.py

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, *args, **kwargs):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        schema.update(type='string')
        return schema


class UserProfileBase(BaseModel):
    name: str = Field(...)
    age: int = Field(..., gt=10, lt=100)
    gender: Optional[str] = None
    stream: Optional[str] = None # e.g., 'Science', 'Commerce', 'Arts'
    interests: List[str] = Field(default_factory=list)
    learningStyle: Optional[str] = None # e.g., 'Visual', 'Auditory', 'Kinesthetic'
    workEnv: Optional[str] = None # e.g., 'Collaborative', 'Independent'
    progress: Dict[str, bool] = Field(default_factory=dict)

class UserProfileInDB(UserProfileBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True


class ProgressUpdate(BaseModel):
    profileId: str
    skill: str
    completed: bool