import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import pickle
import os

dataset = pd.read_csv('../dataset/job_postings_dataset.csv')

# Define required fields for job posting transparency
REQUIRED_FIELDS = ["title", "description", "salary", "location", "employment_type", "ai_used", "requirements", "benefits", "employer"]

def create_transparency_labels(dataset):
    """
    Create binary labels for transparency compliance (1 = transparent, 0 = not transparent).
    A posting is considered transparent if all required fields have values.
    """
    labels = []
    for idx, row in dataset.iterrows():
        missing_count = 0
        for field in REQUIRED_FIELDS:
            value = row.get(field, '')
            if pd.isna(value) or value == '':
                missing_count += 1
        
        # 1 if all fields present (fully transparent), 0 if any missing
        label = 1 if missing_count == 0 else 0
        labels.append(label)
    
    return np.array(labels)


def extract_features(dataset):
    """
    Extract features from job postings for ML model training.
    Features include: text features (TF-IDF), field completeness, and categorical features.
    """
    features_list = []
    
    for idx, row in dataset.iterrows():
        # Text features: combine title, description, and requirements
        text = str(row.get('title', '')) + ' ' + str(row.get('description', '')) + ' ' + str(row.get('requirements', ''))
        
        # Field completeness features
        field_completeness = sum([1 for field in REQUIRED_FIELDS if pd.notna(row.get(field)) and row.get(field) != '']) / len(REQUIRED_FIELDS)
        
        # Employment type encoding
        employment_type = str(row.get('employment_type', 'Unknown')).lower()
        employment_encoded = 1 if employment_type in ['full-time', 'full time'] else 0
        
        # AI used encoding
        ai_used = str(row.get('ai_used', 'Unknown')).lower()
        ai_encoded = 1 if ai_used == 'yes' else (0 if ai_used == 'no' else -1)
        
        # Salary presence
        salary_present = 1 if pd.notna(row.get('salary')) and row.get('salary') != '' else 0
        
        # Store features as dictionary for later processing
        features_list.append({
            'text': text,
            'field_completeness': field_completeness,
            'employment_encoded': employment_encoded,
            'ai_encoded': ai_encoded,
            'salary_present': salary_present
        })
    
    return features_list


# Prepare data
print("=" * 70)
print("MACHINE LEARNING MODEL: JOB POSTING TRANSPARENCY COMPLIANCE")
print("=" * 70)

print("\n[1/5] Creating transparency labels...")
y = create_transparency_labels(dataset)

print("[2/5] Extracting features...")
features_list = extract_features(dataset)

print("[3/5] Vectorizing text features...")
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X_text = vectorizer.fit_transform([f['text'] for f in features_list]).toarray()

# Combine text features with numeric features
X_numeric = np.array([
    [f['field_completeness'], f['employment_encoded'], f['ai_encoded'], f['salary_present']]
    for f in features_list
])

X = np.hstack((X_text, X_numeric))

print("[4/5] Training/testing split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("[5/5] Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Transparent', 'Transparent']))

# Overall dataset prediction
print("\n" + "=" * 70)
print("TRANSPARENCY COMPLIANCE SCORE")
print("=" * 70)

y_pred_all = model.predict(X)
y_pred_proba_all = model.predict_proba(X)

transparent_count = np.sum(y_pred_all == 1)
compliance_score = (transparent_count / len(y_pred_all)) * 100
avg_confidence = np.mean(np.max(y_pred_proba_all, axis=1)) * 100

print(f"\nDataset Transparency Compliance Score: {compliance_score:.2f}%")
print(f"Model Confidence Level: {avg_confidence:.2f}%")
print(f"Compliant Postings: {transparent_count}/{len(y_pred_all)}")
print("=" * 70)

# Save the trained model and vectorizer
print("\n[SAVING] Saving trained model and vectorizer...")
model_data = {
    'model': model,
    'vectorizer': vectorizer,
    'required_fields': REQUIRED_FIELDS
}

model_path = 'transparency_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"✓ Model saved to: {model_path}")
print("=" * 70)

# Function to predict on new job posting
def predict_transparency(job_posting):
    """
    Predict transparency compliance for a single job posting.
    Returns: (prediction, confidence_score)
    prediction: 1 = transparent, 0 = not transparent
    confidence_score: 0-100% confidence in the prediction
    """
    # Extract features for single posting
    text = str(job_posting.get('title', '')) + ' ' + str(job_posting.get('description', '')) + ' ' + str(job_posting.get('requirements', ''))
    
    field_completeness = sum([1 for field in REQUIRED_FIELDS if pd.notna(job_posting.get(field)) and job_posting.get(field) != '']) / len(REQUIRED_FIELDS)
    employment_type = str(job_posting.get('employment_type', 'Unknown')).lower()
    employment_encoded = 1 if employment_type in ['full-time', 'full time'] else 0
    ai_used = str(job_posting.get('ai_used', 'Unknown')).lower()
    ai_encoded = 1 if ai_used == 'yes' else (0 if ai_used == 'no' else -1)
    salary_present = 1 if pd.notna(job_posting.get('salary')) and job_posting.get('salary') != '' else 0
    
    # Vectorize text
    X_text_single = vectorizer.transform([text]).toarray()
    X_numeric_single = np.array([[field_completeness, employment_encoded, ai_encoded, salary_present]])
    X_single = np.hstack((X_text_single, X_numeric_single))
    
    # Predict
    prediction = model.predict(X_single)[0]
    confidence = np.max(model.predict_proba(X_single)) * 100
    return prediction, confidence