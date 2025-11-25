"""
This function was generated with assistance from Copilot (Nov 2025 Version).
Prompt: Write a Python class called TextClassifierPredictor that loads a pre-trained text-based job posting classifier from a pickle file. The class should:
        1. Load a saved model from '../model_textbase/textbase_classifier.pkl' 
        2. Extract rule-based features from job posting text including:
            - Basic field presence (title, description, location, employment type, requirements, employer)
            - Salary information and range validation
            - AI usage disclosure detection
            - Vacancy type disclosure
            - Canadian experience requirement detection
        3. Provide a method to get reasons why a posting is invalid based on hard rules
        4. Make predictions on whether a job posting is VALID or INVALID using both rule-based features and ML model
        5. Handle edge cases like short text and provide confidence scores
        6. Include a global predictor instance with getter function

        The predict method should return a dictionary with status, classification, and confidence. Include proper error handling and informative print statements.

"""
import pickle
import re
from typing import List, Tuple
from pathlib import Path


class TextClassifierPredictor:
    """Load and use the trained text-based job posting classifier."""
    
    def __init__(self, model_path='../model_textbase/textbase_classifier.pkl'):
        """
        Load the trained model from pickle file.
        
        Args:
            model_path (str): Path to the saved model pickle file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.model = self.model_data['model']
            self.label_names = self.model_data['label_names']
            self.is_loaded = True
            print(f"✓ Text-based classifier loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading text classifier: {e}")
            self.is_loaded = False
    
    def extract_rule_features(self, text: str) -> dict:
        """Extract rule-based features from posting text."""
        t = text.lower()
        
        # Basic field presence checks
        has_title = bool(re.search(r"\b(position|job title|title)\s*:", t))
        has_description = any(
            kw in t for kw in ["description:", "overview:", "responsibilities:", "about the role"]
        )
        has_location = "location:" in t
        has_employment_type = any(
            kw in t for kw in ["full-time", "part-time", "contract", "temporary", "intern", "permanent"]
        )
        has_requirements = any(
            kw in t for kw in ["requirements:", "qualifications:"]
        )
        has_employer = any(
            kw in t for kw in ["company:", "employer:", "organization:"]
        )
        
        # Salary information
        has_salary_keyword = any(
            kw in t for kw in ["salary", "compensation", "pay", "per hour", "per year", "$"]
        )
        
        # Salary range detection
        range_pattern = r"\$?\s*(\d[\d,]*)\s*[-–]\s*\$?\s*(\d[\d,]*)"
        range_matches = re.findall(range_pattern, t)
        
        has_salary_range = False
        range_exceeds_50k = False
        
        for a, b in range_matches:
            try:
                v1 = int(a.replace(",", ""))
                v2 = int(b.replace(",", ""))
                has_salary_range = True
                if abs(v2 - v1) > 50000:
                    range_exceeds_50k = True
            except ValueError:
                continue
        
        # AI usage disclosure
        has_ai_keyword = any(
            kw in t for kw in ["ai", "artificial intelligence", "automated system", "algorithm"]
        )
        has_ai_disclosure = any(
            phrase in t for phrase in [
                "we use ai", "ai is used", "artificial intelligence is used", 
                "this posting is screened using ai"
            ]
        )
        
        # Vacancy disclosure
        has_vacancy_disclosure = any(
            phrase in t for phrase in [
                "existing vacancy", "filling an existing role", "new position", "new role", "newly created role"
            ]
        )
        
        # Canadian experience requirement
        mentions_canadian_experience = "canadian experience" in t
        
        return {
            "has_title": int(has_title),
            "has_description": int(has_description),
            "has_location": int(has_location),
            "has_employment_type": int(has_employment_type),
            "has_requirements": int(has_requirements),
            "has_employer": int(has_employer),
            "has_salary_keyword": int(has_salary_keyword),
            "has_salary_range": int(has_salary_range),
            "range_exceeds_50k": int(range_exceeds_50k),
            "has_ai_keyword": int(has_ai_keyword),
            "has_ai_disclosure": int(has_ai_disclosure),
            "has_vacancy_disclosure": int(has_vacancy_disclosure),
            "mentions_canadian_experience": int(mentions_canadian_experience),
        }
    
    def get_invalid_reasons(self, text: str) -> List[str]:
        """Return a list of reasons why a posting is invalid according to hard rules."""
        f = self.extract_rule_features(text)
        reasons = []
        
        # Required fields
        if f["has_title"] == 0:
            reasons.append("Missing title.")
        if f["has_description"] == 0:
            reasons.append("Missing description or responsibilities section.")
        if f["has_salary_keyword"] == 0:
            reasons.append("Missing salary or compensation information.")
        if f["has_location"] == 0:
            reasons.append("Missing location information.")
        if f["has_employment_type"] == 0:
            reasons.append("Missing employment type.")
        if f["has_requirements"] == 0:
            reasons.append("Missing requirements or qualifications section.")
        if f["has_employer"] == 0:
            reasons.append("Missing employer information.")
        
        # Salary range validation
        if f["range_exceeds_50k"] == 1:
            reasons.append("Salary range is wider than $50,000.")
        
        # AI disclosure
        if f["has_ai_keyword"] == 0 or f["has_ai_disclosure"] == 0:
            reasons.append("AI usage and disclosure are required but not clearly stated.")
        
        # Vacancy type disclosure
        if f["has_vacancy_disclosure"] == 0:
            reasons.append("Vacancy type (existing vs new role) is not disclosed.")
        
        # Canadian experience requirement
        if f["mentions_canadian_experience"] == 1:
            reasons.append("Posting explicitly demands Canadian experience.")
        
        return reasons
    
    def predict(self, text: str):
        """
        Predict if a job posting is VALID or INVALID.
        
        Args:
            text (str): Job posting text content
            
        Returns:
            dict: Prediction result with classification and confidence
        """
        if not self.is_loaded:
            return {"error": "Model not loaded", "status": "error"}
        
        try:
            # 1) Check if text is too short
            if len(text.strip()) < 50:
                return {
                    "status": "success",
                    "classification": "INVALID",
                    "confidence": 100.0,
                    "reason": "Text is too short to classify"
                }
            
            # 2) Build simple feature vector
            length = len(text)
            word_count = len(text.split())
            has_salary = 1 if "$" in text or "salary" in text.lower() else 0
            has_location = 1 if any(loc in text.lower() for loc in ["toronto", "vancouver", "montreal", "calgary", "ottawa", "remote"]) else 0
            has_requirements = 1 if any(req in text.lower() for req in ["experience", "requirements", "qualifications", "skills"]) else 0
            
            X = [[length, word_count, has_salary, has_location, has_requirements]]
            
            # 3) Run ML model
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            model_label = self.label_names[int(pred)]
            model_confidence = float(max(proba)) * 100
            
            return {
                "status": "success",
                "classification": model_label,
                "confidence": float(model_confidence)
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}


# Initialize predictor when module is imported
predictor = None

def get_predictor():
    """Get or initialize the text classifier predictor instance."""
    global predictor
    if predictor is None:
        predictor = TextClassifierPredictor()
    return predictor
