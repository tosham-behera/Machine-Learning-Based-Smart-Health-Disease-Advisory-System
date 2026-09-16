# Smart Health Advisory: Heart Disease Prediction

I built this project to practice the complete machine learning workflow with a healthcare-related dataset that I sourced from Kaggle. I wanted to work through the process of loading data, checking its quality, cleaning it, training different models, and comparing their results.

The project predicts the `Heart_Disease` target using features such as age, BMI, blood pressure, cholesterol, smoking status, alcohol intake, stress level, family history, and previous heart attack history.

The Kaggle dataset is synthetic, so I created this project for learning, practice, and portfolio demonstration. It is not a medical diagnostic system and should not be used to make medical decisions.

## Why I Built This Project

I built this project to improve my practical understanding of:

- working with structured healthcare-style data
- checking data quality before training a model
- cleaning incorrect and missing values
- performing exploratory data analysis
- creating and correcting features
- preparing categorical and numerical data
- training and comparing classification models
- evaluating model performance
- documenting a project so it can be reproduced on another computer

## Important Data Cleaning Step: Correcting BMI

One of the main problems I found in the dataset was the BMI column. The BMI values in some rows did not match the person's weight and height.

I checked the BMI values by calculating BMI again using this formula:

```text
BMI = Weight / (Height in metres)^2
```

Because the height values were stored in centimetres, I converted height to metres before calculating BMI:

```python
df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
```

I rounded the corrected BMI values to two decimal places and used the corrected values for the later analysis and model training. This was an important part of the data cleaning process because using incorrect BMI values could affect the model results.

## What I Did in the Project

1. Loaded the synthetic dataset I sourced from Kaggle using pandas.
2. Checked the number of rows, columns, data types, and column names.
3. Checked for missing values and duplicate rows.
4. Checked the target distribution for `Heart_Disease`.
5. Explored numerical and categorical features.
6. Created charts for missing values, class distribution, outliers, and correlations.
7. Checked whether the original BMI values matched weight and height.
8. Corrected the BMI column using weight and height.
9. Replaced missing `Alcohol_Intake` values with `Unknown`.
10. Removed duplicate rows.
11. Checked the cleaned dataset again for missing values and duplicates.
12. Saved the cleaned data as `heart_disease_cleaned.csv`.
13. Separated the features from the `Heart_Disease` target column.
14. Encoded categorical columns using pandas.
15. Split the data into training and testing sets using a stratified split.
16. Scaled numerical columns for Logistic Regression.
17. Trained Logistic Regression, Random Forest, and Gradient Boosting models.
18. Evaluated the models using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and classification reports.
19. Compared the model results and plotted the performance comparison.
20. Checked the feature importance from the Random Forest model.

## Dataset

The dataset contains 50,000 synthetic records and 21 columns.

The features include:

- Age
- Gender
- Weight
- Height
- BMI
- Smoking status
- Alcohol intake
- Physical activity
- Diet
- Stress level
- Hypertension
- Diabetes
- Hyperlipidemia
- Family history
- Previous heart attack
- Systolic and diastolic blood pressure
- Heart rate
- Fasting blood sugar
- Total cholesterol
- `Heart_Disease`, the target column

## Models Used

I trained and compared three classification models:

- **Logistic Regression:** Used as a baseline model and requires scaled numerical features.
- **Random Forest:** Used to capture non-linear relationships using multiple decision trees.
- **Gradient Boosting:** Used as another tree-based ensemble model for comparison.

## Model Results

These are the results currently saved in the notebook after the stratified train-test split:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9239 | 0.9180 | 0.9178 | 0.9179 | 0.9822 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The Random Forest and Gradient Boosting models scored perfectly on this test split. Since the dataset is synthetic, I would need to test the models on real and independent data before making any conclusions about their performance in a real healthcare setting.

## Project Structure

```text
.
├── Smart_Health_Advisory_Heart_Disease_ML_Project.ipynb  # Main notebook
├── synthetic_heart_disease_dataset.csv                   # Original dataset
├── heart_disease_cleaned.csv                              # Cleaned dataset
├── README.md                                              # Project documentation
├── requirements.txt                                       # Python dependencies
├── pyproject.toml                                         # Project configuration
├── uv.lock                                                # Locked dependency versions
└── .gitignore                                             # Ignored local files
```

## How to Run This Project on Another Computer

The following steps explain how I would set up and run this project on another computer.

### Step 1: Install Python

I used Python 3.13 or a newer compatible Python version.

Download Python from [python.org](https://www.python.org/downloads/) and make sure Python is added to the system PATH during installation.

Check the installation:

```bash
python --version
```

### Step 2: Install uv

I used uv to create the virtual environment and install the project packages.

On Windows PowerShell, I can install uv with:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Then I check that uv is available:

```bash
uv --version
```

### Step 3: Clone the repository

I clone the project from GitHub and move into the project folder:

```bash
git clone https://github.com/tosham-behera/Machine-Learning-Based-Smart-Health-Disease-Advisory-System.git
cd <project-folder>
```

I replace `<your-repository-link>` with the URL of my GitHub repository and `<project-folder>` with the cloned folder name.

### Step 4: Create the environment and install packages

Because the repository contains `pyproject.toml` and `uv.lock`, I can install the project dependencies with:

```bash
uv sync
```

This creates a `.venv` folder and installs the required packages, including pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

### Step 5: Activate the environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 6: Install notebook support

I open the project in VS Code and install the Python and Jupyter extensions if they are not already installed.

If I want to run the notebook from a terminal, I can also install Jupyter in the active environment:

```bash
uv pip install jupyter ipykernel
```

### Step 7: Open the notebook

I open:

```text
Smart_Health_Advisory_Heart_Disease_ML_Project.ipynb
```

In VS Code, I select the Python interpreter from the project `.venv` as the notebook kernel.

### Step 8: Run the notebook

I run the notebook cells from top to bottom. The source dataset must stay in the same folder as the notebook:

```text
synthetic_heart_disease_dataset.csv
```

The notebook creates or updates:

```text
heart_disease_cleaned.csv
```

Running the cells in order is important because later cells use variables created in earlier cells.

## Requirements File Setup

If I do not use uv, I can create a virtual environment and install the packages from `requirements.txt`:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then I install the packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Technologies I Used

- Python
- pandas for data loading and data cleaning
- NumPy for numerical operations
- Matplotlib for visualizations
- Seaborn for statistical visualizations
- scikit-learn for preprocessing, model training, and evaluation
- Jupyter Notebook for documenting the analysis
- uv for environment and dependency management
- Git and GitHub for version control

## What I Learned

This project helped me understand that checking the quality of a dataset is just as important as choosing a machine learning model. In particular, finding and correcting the incorrect BMI values showed me how a data-cleaning problem can affect the features used by a model.

I also practiced exploratory data analysis, categorical encoding, numerical scaling, stratified train-test splitting, model comparison, and evaluation using several metrics instead of relying only on accuracy.

## Limitations and Future Improvements

The data is synthetic and the current results come from one train-test split. The perfect scores from the ensemble models should therefore be interpreted carefully.

For a future version, I would like to:

- use cross-validation and hyperparameter tuning
- test the workflow on an independent dataset
- investigate the reasons for the perfect ensemble scores
- add feature-importance and model-explainability analysis
- add automated tests for the cleaning and modeling steps
- create a separate prediction script or demonstration application
- use an LLM to provide general health-related suggestions based on the model output, with strong privacy protections, medically reviewed guidance, clear uncertainty, and a reminder that it cannot replace a qualified healthcare professional
