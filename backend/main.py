from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import csv
import pandas as pd
import os 
from pathlib import Path
from models import FormValidate, TextPosting
import pickle
import numpy as np
from textbase_classifier_loader import get_predictor

app = FastAPI()

# Configure CORS to allow frontend requests
origins = [
    "http://localhost:5173",      # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",      # Alternative frontend port
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load transparency model on startup
transparency_model = None
transparency_vectorizer = None
required_fields = None

@app.on_event("startup")
async def load_model():
    """Load the trained transparency model before the app starts."""
    global transparency_model, transparency_vectorizer, required_fields
    
    backend_dir = Path(__file__).parent
    model_path = backend_dir / "transparency_model.pkl"
    
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            transparency_model = model_data['model']
            transparency_vectorizer = model_data['vectorizer']
            required_fields = model_data['required_fields']
            print(f"✓ Transparency model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading transparency model: {e}")
    else:
        print(f"⚠ Warning: transparency_model.pkl not found at {model_path}")




# This function was generated with assistance from Copilot (Nov 2025 Version).
# Prompt: "Write a Python function to read a file from a relative path."
# Modifications: Wrap inside a fastapi endpiont and return the content.

@app.get("/jobs-postings/")
async def read_root():
    invalid_job = {}
    valid_job = {}
    
    backend_dir = Path(__file__).parent
    raw_postings_dir = backend_dir.parent / "dataset" / "raw_postings"
    
    dataset = os.listdir(raw_postings_dir)
    for file in dataset:
        file_path = raw_postings_dir / file
        with open(file_path, 'r') as f:
            content = f.read()
            
        if "invalid" in file:
            invalid_job[file] = content
        else:
            valid_job[file] = content
            
    return {"valid_job": valid_job, "invalid_job": invalid_job}

@app.post("/jobs-postings/text-base")
async def validate_jobs_posting(job_posting: TextPosting):
    """
    Validate job posting using text-based classification model.
    
    Accepts raw job posting text and classifies it as VALID or INVALID
    based on rule-based checks and ML model prediction.
    """
    text_classifier = get_predictor()
    
    if not text_classifier.is_loaded:
        return {
            "classification": "ERROR",
            "confidence": 0.0,
            "message": "Model not loaded"
        }
    
    result = text_classifier.predict(job_posting.posting_text)
    
    return {
        "classification": result.get("classification", "ERROR"),
        "confidence": result.get("confidence", 0.0)
    }

@app.post("/jobs-postings/field-base")
async def validate_jobs_posting_fields(job_posting: FormValidate):
    """
    Validate job posting fields for completeness and transparency compliance.
    
    Performs two checks:
    1. Field-based validation: checks if all required fields are present
    2. ML-based transparency check: uses trained model to predict compliance score
    
    Returns validation status, missing fields, and transparency compliance score.
    """
    required_fields_list = ["title", "description", "salary", "location", "employment_type", "ai_used", "requirements", "benefits", "employer"]
    job_posting_dict = job_posting.dict()
    missing_fields = [field for field in required_fields_list if not job_posting_dict.get(field)]
    
    # Determine classification based on missing fields
    classification = "VALID" if not missing_fields else "INVALID"
    transparency_score = 0.0
    confidence = 0.0
    
    # Add ML-based transparency check if model is loaded
    if transparency_model is not None:
        try:
            # Extract text features
            text = str(job_posting_dict.get('title', '')) + ' ' + str(job_posting_dict.get('description', '')) + ' ' + str(job_posting_dict.get('requirements', ''))
            
            # Field completeness
            field_completeness = sum([1 for field in required_fields_list if pd.notna(job_posting_dict.get(field)) and job_posting_dict.get(field) != '']) / len(required_fields_list)
            
            # Employment type encoding
            employment_type = str(job_posting_dict.get('employment_type', 'Unknown')).lower()
            employment_encoded = 1 if employment_type in ['full-time', 'full time', 'fulltime'] else 0
            
            # AI used encoding
            ai_used = str(job_posting_dict.get('ai_used', 'Unknown')).lower()
            ai_encoded = 1 if ai_used == 'yes' else (0 if ai_used == 'no' else -1)
            
            # Salary presence
            salary_present = 1 if pd.notna(job_posting_dict.get('salary')) and job_posting_dict.get('salary') != '' else 0
            
            # Vectorize and predict
            X_text = transparency_vectorizer.transform([text]).toarray()
            X_numeric = np.array([[field_completeness, employment_encoded, ai_encoded, salary_present]])
            X = np.hstack((X_text, X_numeric))
            
            prediction = transparency_model.predict(X)[0]
            proba = transparency_model.predict_proba(X)
            confidence = float(np.max(proba) * 100)
            
            # Transparency score is the probability of class 1 (compliant/transparent)
            transparency_score = float(proba[0][1] * 100) if len(proba[0]) > 1 else 0.0
        except Exception as e:
            print(f"Error in transparency model prediction: {e}")
            confidence = 0.0
            transparency_score = 0.0
    
    return {
        "classification": classification,
        "confidence": confidence,
        "missing_fields": missing_fields,
        "transparency_score": transparency_score
    }