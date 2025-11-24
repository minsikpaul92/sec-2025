import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import re

dataset = pd.read_csv('../dataset/job_postings_dataset.csv')

# Define required fields based on the dataset structure
REQUIRED_FIELDS = ["id", "title", "description", "salary", "location", "employment_type", "ai_used", "requirements", "benefits", "employer"]

def validate_job_posting_fields(job_posting):
    """
    Validates that a job posting contains all required fields.
    
    Args:
        job_posting (dict): Job posting data to validate
        
    Returns:
        dict: Status of validation with any missing fields
    """
    missing_fields = [field for field in REQUIRED_FIELDS if field not in job_posting or pd.isna(job_posting.get(field))]
    
    if missing_fields:
        return {"status": "Invalid job posting", "missing_fields": missing_fields}
    
    return {"status": "Valid job posting", "data": job_posting}


def preprocess_data(dataset):
    """
    Preprocesses job posting dataset by normalizing text fields.
    
    Args:
        dataset (pd.DataFrame): Raw job postings dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Drop id columns
    dataset = dataset.drop(columns=['id'])
    
    # Fill missing values
    dataset = dataset.fillna('')
    
    # Normalize text columns
    dataset["title"] = dataset["title"].str.lower()
    dataset["description"] = dataset["description"].str.lower()
    dataset["location"] = dataset["location"].str.lower()
    dataset["employer"] = dataset["employer"].str.lower()
    dataset["requirements"] = dataset["requirements"].str.lower()
    dataset["benefits"] = dataset["benefits"].str.lower()
    
    return dataset


HOURS_PER_YEAR = 40 * 52   # 2080 hours


def hourly_to_yearly(salary_str):
    """
    Convert salary strings like "$30-$36/hour" or "$17/hour"
    into yearly salary range (min, max).

    Args:
        salary_str (str): Salary string in hourly format

    Returns:
        (yearly_min, yearly_max) as integers
        or (None, None) if invalid/missing.
    """

    if not salary_str or not isinstance(salary_str, str):
        return None, None

    s = salary_str.lower().replace(" ", "")

    # Detect if it's hourly
    if "/hour" not in s and "/hr" not in s:
        return None, None

    # Extract all numeric values
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return None, None

    nums = [float(n) for n in nums]

    # For "$17/hour"
    if len(nums) == 1:
        hourly = nums[0]
        yearly = int(hourly * HOURS_PER_YEAR)
        return yearly, yearly

    # For "$30-$36/hour"
    if len(nums) >= 2:
        hourly_min = min(nums)
        hourly_max = max(nums)
        yearly_min = int(hourly_min * HOURS_PER_YEAR)
        yearly_max = int(hourly_max * HOURS_PER_YEAR)
        return yearly_min, yearly_max

    return None, None


def validate_all_postings(dataset):
    """
    Validates all job postings in the dataset based on required fields.
    
    Args:
        dataset (pd.DataFrame): Job postings dataset
        
    Returns:
        dict: Validation results with valid and invalid postings
    """
    valid_postings = []
    invalid_postings = []
    
    for idx, row in dataset.iterrows():
        job_posting = row.to_dict()
        validation_result = validate_job_posting_fields(job_posting)
        
        if validation_result["status"] == "Valid job posting":
            valid_postings.append(job_posting)
        else:
            invalid_postings.append({
                "posting": job_posting,
                "missing_fields": validation_result["missing_fields"]
            })
    
    return {
        "total": len(dataset),
        "valid": len(valid_postings),
        "invalid": len(invalid_postings),
        "valid_postings": valid_postings,
        "invalid_postings": invalid_postings
    }


# Main execution
if __name__ == "__main__":
    # Validate all postings
    validation_results = validate_all_postings(dataset)
    
    print(f"Total job postings: {validation_results['total']}")
    print(f"Valid postings: {validation_results['valid']}")
    print(f"Invalid postings: {validation_results['invalid']}")
    
    if validation_results['invalid_postings']:
        print("\nInvalid postings details:")
        for invalid in validation_results['invalid_postings']:
            print(f"  - Missing fields: {invalid['missing_fields']}")
