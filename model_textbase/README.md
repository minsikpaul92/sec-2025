# Seneca Software Engineering Competition 2025

Welcome to the Seneca Software Engineering Competition! This repository contains the competition dataset and all the information you need to get started. We recommend following these steps:

1. Read the rest of this README.md
2. Read the tutorial linked in `info/githubHowToUseGithubDesktop.txt`
3. Read the tutorial linked in `info/githubHowToFork.txt`
4. Fork this repository
5. Clone your forked repository locally on your machine
6. Add the organizers as collaborators (@ChloeQuijano, @MARYANNE67, @masuma131)
7. Start coding!

## Getting Help

If you need assistance during the competition, please follow these steps:

1. Check the included documentation for problem details and scoring questions
2. Check the Discord for previously answered questions
3. If you have done all of the above, ask the organizers for help

## Submission Requirements

**Important:** In your team's README.md, you must specify:

- How to run your code
- What programming language and version your code uses (e.g., Python 3.11)
- A list of required packages with versions if needed (e.g., `pytorch==2.1.116`). A `requirements.txt` file is also acceptable.
- If applicable, what operating system your code should be run on. Any specifications not included in your README cannot be assumed to be available on the organizers' machine(s).
- If your code requires a model, include it in the repository (with links if not located on the main branch) that need to be downloaded to run your code.

## Dataset

This repository contains a dataset of job postings located in the `dataset/` folder. This data is intended to support analysis, generation, or enhancement of job postings during the competition.

There are two types of data formats available for you to use:

### 1. Structured CSV Data

File provided:

`job_postings_dataset.csv`

This file contains a tabular set of job postings with fields such as:

- Job title  
- Description  
- Salary (may be missing)  
- Location  
- Employment type  
- AI usage in the hiring process  
- Requirements  
- Benefits  
- Employer name  

You may use this CSV directly for tasks such as parsing, scoring, visualization, insight generation, or generating improved job postings.

### 2. Raw Job Posting Text Files

Folder provided:

`raw_postings/`

This folder contains individual job postings as text files. Some are valid, while others contain missing, vague, or unclear information, such as:

- Missing salary or benefits  
- Unclear working conditions  
- Misleading descriptions (e.g., vague scheduling, unclear pay)  
- Undisclosed AI usage  
- Incomplete requirements  

Use of these files is optional, but they may help if you want to:

- Test your tool against messy, realistic text inputs  
- Build analysis from unstructured data  
- Support file upload or scanning features  
- Demonstrate extensibility beyond the CSV

### Dataset Usage Notes

- You may use either or both dataset formats depending on your solution design.
- You may generate additional sample postings only if they are created during the competition.
- If you use generative AI or external tools to expand or clean the data, you must cite them clearly in your documentation and code.

## Example: Loading the Dataset

### Python Example

#### Loading the CSV File

```python
import pandas as pd

# Load the structured dataset
df = pd.read_csv("dataset/job_postings_dataset.csv")

# Display the first few entries
print(df.head())
```

#### Loading Raw Text Files

```python
import os

folder_path = "dataset/raw_postings/"

# Loop through and read each text file
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            print(f"--- {filename} ---")
            print(content)
            print()
```

## How to Cite AI Usage

If you use generative AI (GitHub Copilot) to help generate code, text, or ideas for your project, you are required to properly cite it in your:

- Code comments
- README documentation
- Presentation slides (if referenced visually or verbally)

### Minimum Citation Requirements

Every use of AI should include:

1. **Which tool was used** (GitHub Copilot)
2. **What it was used for** (e.g., code generation, documentation, debugging)
3. **How much of the result was modified** (e.g., "minor edits", "significant modifications", "used as-is")
4. **Where it appears in your project** (file paths, sections, etc.)

### Example Code Citation

```python
# This function was generated with assistance from Copilot (Feb 2025 Version).
# Prompt: "Write a Python function to flag missing job posting fields."
# Modifications: Added error handling and adjusted return values manually.

def check_missing_fields(posting):
    required_fields = ["title", "salary", "location", "employment_type"]
    return [field for field in required_fields if not posting.get(field)]
```

### Example README Citation

Some code for parsing job postings was assisted using Copilot (Feb 2025).
The model was prompted to "generate Python code to parse job posting fields."
Logic for error handling and formatting was modified manually by the team.

### Example Presentation Citation

**Slide Text:**
- Feature extraction logic initially generated using Copilot
- Team modified output formatting, error handling, and validity checks

