import os
import re
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression


# Directory containing training .txt files
DATA_DIR = "dataset/raw_postings"
TEST_DIR = "dataset/test_postings"


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
            # Skip files that do not clearly indicate a label
            print(f"Skipping file without label keyword: {filename}")
            continue

        file_path = os.path.join(data_dir, filename)

        # Read the file with a simple encoding fallback
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="cp949") as f:
                text = f.read()

        texts.append(text)
        labels.append(label)

    return texts, labels


def extract_rule_features(text: str) -> dict:
    """
    Extract rule-based features from a posting.

    These features are based on simple rules like:
    - whether salary information is present
    - whether a salary range is present and not too wide
    - whether AI usage is mentioned and disclosed
    - whether vacancy type is disclosed
    - whether 'Canadian experience' is explicitly required
    """
    t = text.lower()

    # 0) Basic field presence checks
    has_title = bool(re.search(r"\b(position|job title|title)\s*:", t))
    has_description = any(
        kw in t
        for kw in [
            "description:",
            "overview:",
            "responsibilities:",
            "about the role",
        ]
    )

    has_location = "location:" in t

    has_employment_type = any(
        kw in t
        for kw in [
            "full-time",
            "part-time",
            "contract",
            "temporary",
            "intern",
            "permanent",
        ]
    )

    has_requirements = any(
        kw in t
        for kw in [
            "requirements:",
            "qualifications:",
        ]
    )

    has_employer = any(
        kw in t
        for kw in [
            "company:",
            "employer:",
            "organization:",
        ]
    )

    # 1) Salary / pay information
    has_salary_keyword = any(
        kw in t
        for kw in [
            "salary",
            "compensation",
            "pay",
            "per hour",
            "per year",
            "$",
        ]
    )

    # Simple numeric range detection, e.g. "20 - 30", "50000-70000"
    # Allow optional $ before each number (e.g., "$65,000 - $80,000")
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

    # 2) AI usage disclosure
    has_ai_keyword = any(
        kw in t
        for kw in [
            "ai",
            "artificial intelligence",
            "automated system",
            "algorithm",
        ]
    )

    has_ai_disclosure = any(
        phrase in t
        for phrase in [
            "we use ai",
            "ai is used",
            "artificial intelligence is used",
            "this posting is screened using ai",
        ]
    )

    # 3) Vacancy disclosure (existing vs new role)
    has_vacancy_disclosure = any(
        phrase in t
        for phrase in [
            "existing vacancy",
            "filling an existing role",
            "new position",
            "new role",
            "newly created role",
        ]
    )

    # 4) Canadian experience requirement
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


def get_invalid_reasons(text: str) -> List[str]:
    """
    Return a list of human-readable reasons why a posting is invalid
    according to the hard rules. If the list is empty, the posting
    passes the rule-based checks.
    """
    f = extract_rule_features(text)
    reasons: List[str] = []

    # 0) Required fields presence
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

    # 1) No salary range or too wide
    if f["has_salary_keyword"] == 0:
        reasons.append("Missing salary range.")
    elif f["range_exceeds_50k"] == 1:
        reasons.append("Salary range is wider than $50,000.")

    # 2) AI usage must be explicitly disclosed
    if f["has_ai_keyword"] == 0 or f["has_ai_disclosure"] == 0:
        reasons.append("AI usage and disclosure are required but not clearly stated.")

    # 3) Vacancy type not disclosed -> invalid
    if f["has_vacancy_disclosure"] == 0:
        reasons.append("Vacancy type (existing vs new role) is not disclosed.")

    # 4) Posting explicitly demands 'Canadian experience' -> invalid
    if f["mentions_canadian_experience"] == 1:
        reasons.append("Posting explicitly demands Canadian experience.")

    return reasons


def rule_based_invalid(text: str) -> bool:
    """
    Apply hard rules to determine if a posting is clearly invalid.

    If any of these rules are triggered, we return True (invalid),
    without asking the ML model.
    """
    reasons = get_invalid_reasons(text)
    return len(reasons) > 0


def build_feature_matrix(texts: List[str]):
    """
    Convert a list of texts into a simple feature matrix
    using the rule-based features defined above. The model
    intentionally excludes vacancy disclosure because the
    rule-based layer already enforces it and it was skewing
    the learned decision boundary.
    """
    X = []
    for txt in texts:
        f = extract_rule_features(txt)
        X.append(
            [
                f["has_title"],
                f["has_description"],
                f["has_location"],
                f["has_employment_type"],
                f["has_requirements"],
                f["has_employer"],
                f["has_salary_keyword"],
                f["has_salary_range"],
                f["range_exceeds_50k"],
                f["has_ai_keyword"],
                f["has_ai_disclosure"],
                f["mentions_canadian_experience"],
            ]
        )
    return X


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

        # 2) Run ML model on the same text
        X = build_feature_matrix([text])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        model_label = self.label_names[int(pred)]
        model_confidence = float(max(proba))
        model_confidence_pct = model_confidence * 100

        # 3) Print detailed analysis
        print("Rule-based analysis:")
        if rule_invalid:
            print("  -> INVALID")
            for r in reasons:
                print(f"   - {r}")
        else:
            print("  -> PASSED (no rule violations)")

        print("Model-based prediction:")
        print(f"  -> {model_label} (confidence: {model_confidence_pct:.2f}%)")

        # 4) Final decision:
        #    If any rule violation exists, we still treat it as INVALID,
        #    but we have also shown the model's opinion above.
        if rule_invalid:
            final_label = "INVALID"
            final_confidence_pct = 100.0
            final_reasons = reasons
        else:
            final_label = model_label
            final_confidence_pct = model_confidence_pct
            if final_label == "INVALID":
                # Model judged INVALID even though rules passed
                final_reasons = [f"Model classified as INVALID (confidence: {model_confidence_pct:.2f}%)"]
            else:
                final_reasons = []

        return final_label, final_confidence_pct, final_reasons

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

    test_files = sorted(
        [f for f in os.listdir(TEST_DIR) if f.endswith(".txt")]
    )
    num_tests = len(test_files)

    print("Classify new posting files as VALID or INVALID.")
    print("You can:")
    print(f"  - Enter a .txt file path (e.g., {TEST_DIR}/posting_01.txt)")
    if num_tests > 0:
        print(f"  - Or just enter a test index number 1-{num_tests}")
    print("Type 'q' to quit.\n")

    while True:
        prompt_idx = f"1-{num_tests}" if num_tests > 0 else ""
        user_input = input(
            f"Path to .txt file or test index ({prompt_idx}): " if prompt_idx else "Path to .txt file: "
        ).strip()
        if user_input.lower() == "q":
            print("Exiting.")
            break

        # If the user enters a digit, map it to sorted test files
        if user_input.isdigit():
            idx = int(user_input)
            if idx < 1 or idx > num_tests:
                print(f"Index out of range. Please enter 1-{num_tests}.\n")
                continue
            filename = test_files[idx - 1]
            path = os.path.join(TEST_DIR, filename)
        else:
            path = user_input

        if not path.endswith(".txt"):
            print("Only .txt files are supported.\n")
            continue

        try:
            label, confidence, reasons = classifier.predict_file(path)
            print(f"Prediction: {label} (confidence: {confidence:.2f}%)")
            if label == "INVALID":
                if reasons:
                    print("Reasons:")
                    for r in reasons:
                        print(f" - {r}")
                else:
                    print("Reasons: Model flagged INVALID (no rule violations).")
            print()
        except FileNotFoundError as e:
            print(e, "\n")
        except Exception as e:
            print("Error during prediction:", e, "\n")


if __name__ == "__main__":
    main()
