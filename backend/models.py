from typing import Optional
from pydantic import BaseModel

class FormValidate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    salary: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    ai_used: Optional[str] = None
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    employer: Optional[str] = None

class TextPosting(BaseModel):
    """Request model for text-based job posting validation."""
    posting_text: str