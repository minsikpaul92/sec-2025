from fastapi import FastAPI
import csv
import pandas as pd
import os 
from pathlib import Path

app = FastAPI()


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
async def validate_jobs_posting(job_posting: dict):
    return {"status": "Job posting is valid", "data": job_posting}

@app.post("/jobs-postings/field-base")
async def validate_jobs_posting_fields(job_posting: dict):
    required_fields = ["title", "description", "salary", "location", "employment_type", "ai_used", "requirements", "benefits", "employer"]
    missing_fields = [field for field in required_fields if field not in job_posting]
    
    if missing_fields:
        return {"status": "Invalid job posting", "missing_fields": missing_fields}
    
    return {"status": "Valid job posting", "data": job_posting}