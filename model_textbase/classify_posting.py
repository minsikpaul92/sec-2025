import os
import re
from typing import List, Tuple
import pickle

from sklearn.linear_model import LogisticRegression


# Directory containing training .txt files
DATA_DIR = "dataset/raw_postings"
TEST_DIR = "dataset/test_postings"

# This function was generated with assistance from Copilot (Nov 2025 Version).
# Prompt: "Write a function to load training postings from a directory"
def load_postings(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load training postings from the given directory.

    File names must contain either 'valid' or 'invalid' to determine the label.
    - 'valid'   -> label 1
    - 'invalid' -> label 0
    """
    texts: List[str] = []
    labels: List[int] = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        lower_name = filename.lower()

        if "invalid" in lower_name:
            label = 0
        elif "valid" in lower_name:
            label = 1
        else:
            print(f"Skipping file without label keyword: {filename}")
            continue

        file_path = os.path.join(data_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="cp949") as f:
                text = f.read()

        texts.append(text)
        labels.append(label)

    return texts, labels

# This function was generated with assistance from Copilot (Nov 2025 Version).
# Prompt: "Write a function to get reasons why a posting is invalid based on hard rules"
def get_invalid_reasons(text: str) -> List[str]:
    """
    Return a list of human-readable reasons why a posting is invalid.
    For plain text classification, we only check basic length.
    Detailed classification is done by the ML model.
    """
    reasons: List[str] = []
    
    # Only check if text is too short
    if len(text.strip()) < 50:
        reasons.append("Posting text is too short to classify.")
    
    return reasons


def rule_based_invalid(text: str) -> bool:
    """
    Apply minimal hard rules. For plain text, we mostly rely on the ML model.
    Only reject if text is too short.
    """
    reasons = get_invalid_reasons(text)
    return len(reasons) > 0


def build_feature_matrix(texts: List[str]):
    """
    Convert a list of texts into a simple feature matrix.
    For plain text classification, we use basic text statistics.
    """
    X = []
    for txt in texts:
        length = len(txt)
        word_count = len(txt.split())
        has_salary = 1 if "$" in txt or "salary" in txt.lower() else 0
        has_location = 1 if any(loc in txt.lower() for loc in ["toronto", "vancouver", "montreal", "calgary", "ottawa", "remote"]) else 0
        has_requirements = 1 if any(req in txt.lower() for req in ["experience", "requirements", "qualifications", "skills"]) else 0
        
        X.append([length, word_count, has_salary, has_location, has_requirements])
    return X

# This function was generated with assistance from Copilot (Nov 2025 Version).
# Prompt: "Write a PostingClassifier class that combines hard rules with an ML model"
class PostingClassifier:
    """
    A classifier that combines hard rule checks with a simple ML model.

    - If hard rules indicate the posting is invalid, we immediately return INVALID.
    - Otherwise, we pass rule-based features into a LogisticRegression model
      trained on labeled examples.
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.label_names = {0: "INVALID", 1: "VALID"}

    def train(self, texts: List[str], labels: List[int]) -> None:
        X = build_feature_matrix(texts)
        self.model.fit(X, labels)

    def predict_text(self, text: str) -> Tuple[str, float, List[str]]:
        # 1) Run rule-based analysis
        reasons = get_invalid_reasons(text)
        rule_invalid = len(reasons) > 0

        X = build_feature_matrix([text])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        model_label = self.label_names[int(pred)]
        model_confidence = float(max(proba))
        model_confidence_pct = model_confidence * 100

        print("Rule-based analysis:")
        if rule_invalid:
            print("  -> INVALID")
            for r in reasons:
                print(f"   - {r}")
        else:
            print("  -> PASSED (no rule violations)")

        print("Model-based prediction:")
        print(f"  -> {model_label} (confidence: {model_confidence_pct:.2f}%)")

        if rule_invalid:
            final_label = "INVALID"
            final_confidence_pct = 100.0
            final_reasons = reasons
        else:
            final_label = model_label
            final_confidence_pct = model_confidence_pct
            if final_label == "INVALID":
                final_reasons = [f"Model classified as INVALID (confidence: {model_confidence_pct:.2f}%)"]
            else:
                final_reasons = []

        return final_label, final_confidence_pct, final_reasons
    # This function was generated with assistance from Copilot (Nov 2025 Version).
    # Prompt: "Write a function to predict transparency compliance for a text file"
    def predict_file(self, path: str) -> Tuple[str, float, List[str]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="cp949") as f:
                text = f.read()

        return self.predict_text(text)


def main():
    print("Loading training postings from directory:", DATA_DIR)
    texts, labels = load_postings(DATA_DIR)

    if len(texts) == 0:
        print("No training texts found. Please check the raw_postings directory.")
        return

    print(f"Number of training samples: {len(texts)}")

    classifier = PostingClassifier()
    classifier.train(texts, labels)
    print("Model training completed.\n")

    # Save the trained model
    print("[SAVING] Saving trained text-based classification model...")
    model_data = {
        'model': classifier.model,
        'label_names': classifier.label_names
    }
    
    model_path = 'textbase_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)


if __name__ == "__main__":
    main()
