
# ============================================================
# CODEALPHA INTERNSHIP - TASK 2
# EXPLORATORY DATA ANALYSIS (EDA)
# Dataset: Titanic
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ------------------------------------------------------------
# 1. CREATE REQUIRED FOLDERS
# ------------------------------------------------------------

os.makedirs("dataset", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

sns.set_theme(style="whitegrid")

print("\nCODEALPHA - TASK 2: EXPLORATORY DATA ANALYSIS")

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
dataset_path = "dataset/titanic.csv"

try:
    if os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 0:
        df = pd.read_csv(dataset_path)

        if df.empty or len(df.columns) == 0:
            raise ValueError("Dataset file is empty or invalid.")

    else:
        print("\nDataset missing or empty. Downloading...")
        df = pd.read_csv(url)
        df.to_csv(dataset_path, index=False)

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

except Exception as e:
    print("Error loading dataset:", e)
    raise

# ------------------------------------------------------------
# 3. UNDERSTANDING THE DATA
# ------------------------------------------------------------

print("\n========== DATASET OVERVIEW ==========")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))

# ------------------------------------------------------------
# 4. DATA QUALITY ANALYSIS
# ------------------------------------------------------------

print("\n========== DATA QUALITY ANALYSIS ==========")

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

quality_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage,
    "Data Type": df.dtypes
})

print("\nMissing Values:")
print(quality_report)

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nUnique Values:")
print(df.nunique())

quality_report.to_csv(
    "results/data_quality_report.csv"
)

# ------------------------------------------------------------
# 5. DATA CLEANING
# ------------------------------------------------------------

print("\n========== DATA CLEANING ==========")

df_clean = df.copy()

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()

# Fill missing numerical values with median
numerical_columns = df_clean.select_dtypes(
    include=np.number
).columns

for col in numerical_columns:
    df_clean[col] = df_clean[col].fillna(
        df_clean[col].median()
    )

# Fill missing categorical values with mode
categorical_columns = df_clean.select_dtypes(
    include=["object", "category", "bool"]
).columns

for col in categorical_columns:
    if df_clean[col].isnull().any():
        mode_value = df_clean[col].mode()

        if not mode_value.empty:
            df_clean[col] = df_clean[col].fillna(
                mode_value[0]
            )

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

print("\nCleaned Dataset Shape:")
print(df_clean.shape)

df_clean.to_csv(
    "results/cleaned_dataset.csv",
    index=False
)

# ------------------------------------------------------------
# 6. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

print("\nSurvival Count:")
print(df_clean["survived"].value_counts())

print("\nSurvival Rate:")
print(df_clean["survived"].value_counts(normalize=True) * 100)

print("\nSurvival by Gender:")
print(pd.crosstab(df_clean["sex"], df_clean["survived"]))

print("\nSurvival by Passenger Class:")
print(pd.crosstab(df_clean["class"], df_clean["survived"]))

print("\nAverage Age:")
print(df_clean["age"].mean())

print("\nAverage Fare:")
print(df_clean["fare"].mean())

# ------------------------------------------------------------
# 7. UNIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n========== UNIVARIATE ANALYSIS ==========")

# Numerical columns
numerical_columns = ["age", "fare", "sibsp", "parch"]

for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_clean[col], kde=True)
    plt.title("Distribution of " + col)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/distribution_" + col + ".png")
    plt.close()

# Categorical columns
categorical_columns = ["sex", "class", "embarked", "who"]

for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_clean, x=col)
    plt.title("Distribution of " + col)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("plots/category_" + col + ".png")
    plt.close()

# ------------------------------------------------------------
# 8. BIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n========== BIVARIATE ANALYSIS ==========")

# Survival count
plt.figure(figsize=(7, 5))
sns.countplot(data=df_clean, x="survived")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/survival_count.png")
plt.close()

# Survival by gender
plt.figure(figsize=(7, 5))
sns.countplot(data=df_clean, x="sex", hue="survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/survival_by_gender.png")
plt.close()

# Survival by passenger class
plt.figure(figsize=(7, 5))
sns.countplot(data=df_clean, x="class", hue="survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/survival_by_class.png")
plt.close()

# Age versus fare
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df_clean,
    x="age",
    y="fare",
    hue="survived"
)
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("plots/age_vs_fare.png")
plt.close()

# ------------------------------------------------------------
# 9. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n========== CORRELATION ANALYSIS ==========")

numeric_df = df_clean.select_dtypes(include=np.number)

correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

correlation_matrix.to_csv(
    "results/correlation_matrix.csv"
)

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# ------------------------------------------------------------
# 10. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\n========== OUTLIER DETECTION ==========")

outlier_results = []

for col in numerical_columns:

    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df_clean[
        (df_clean[col] < lower_limit) |
        (df_clean[col] > upper_limit)
    ]

    outlier_results.append({
        "Column": col,
        "Outlier Count": len(outliers),
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit
    })

    print(col, "Outliers:", len(outliers))

    plt.figure(figsize=(7, 5))
    sns.boxplot(y=df_clean[col])
    plt.title("Outliers in " + col)
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig("plots/outliers_" + col + ".png")
    plt.close()

outlier_report = pd.DataFrame(outlier_results)

outlier_report.to_csv(
    "results/outlier_report.csv",
    index=False
)

# ------------------------------------------------------------
# 11. HYPOTHESIS TESTING - CHI-SQUARE TEST
# ------------------------------------------------------------

print("\n========== HYPOTHESIS TESTING ==========")

contingency_table = pd.crosstab(
    df_clean["sex"],
    df_clean["survived"]
)

chi2, p_value, dof, expected = chi2_contingency(
    contingency_table
)

print("\nChi-Square Test: Gender vs Survival")
print("Chi-Square Value:", chi2)
print("P-Value:", p_value)
print("Degrees of Freedom:", dof)

if p_value < 0.05:
    hypothesis_result = (
        "There is a significant relationship "
        "between gender and survival."
    )
else:
    hypothesis_result = (
        "There is no significant relationship "
        "between gender and survival."
    )

print(hypothesis_result)

# ------------------------------------------------------------
# 12. AUTOMATIC KEY FINDINGS
# ------------------------------------------------------------

print("\n========== KEY FINDINGS ==========")

survival_rate = df_clean["survived"].mean() * 100

gender_survival = df_clean.groupby("sex")["survived"].mean() * 100

class_survival = df_clean.groupby("class")["survived"].mean() * 100

print("Overall Survival Rate:", round(survival_rate, 2), "%")

print("\nSurvival Rate by Gender:")
print(gender_survival)

print("\nSurvival Rate by Passenger Class:")
print(class_survival)

print("\nAverage Passenger Age:", round(df_clean["age"].mean(), 2))

print("Average Ticket Fare:", round(df_clean["fare"].mean(), 2))

# ------------------------------------------------------------
# 13. GENERATE EDA REPORT
# ------------------------------------------------------------

print("\n========== GENERATING EDA REPORT ==========")

with open("results/EDA_Report.txt", "w", encoding="utf-8") as report:

    report.write("CODEALPHA - TASK 2\n")
    report.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    report.write("=" * 50 + "\n\n")

    report.write("1. DATASET OVERVIEW\n")
    report.write("Original Dataset Shape: " + str(df.shape) + "\n")
    report.write("Cleaned Dataset Shape: " + str(df_clean.shape) + "\n\n")

    report.write("2. DATA QUALITY\n")
    report.write(str(quality_report) + "\n\n")

    report.write("3. SURVIVAL ANALYSIS\n")
    report.write("Overall Survival Rate: " + str(round(survival_rate, 2)) + "%\n\n")

    report.write("Survival Rate by Gender:\n")
    report.write(str(gender_survival) + "\n\n")

    report.write("Survival Rate by Passenger Class:\n")
    report.write(str(class_survival) + "\n\n")

    report.write("4. CORRELATION MATRIX\n")
    report.write(str(correlation_matrix) + "\n\n")

    report.write("5. OUTLIER ANALYSIS\n")
    report.write(str(outlier_report) + "\n\n")

    report.write("6. HYPOTHESIS TESTING\n")
    report.write("Chi-Square Value: " + str(chi2) + "\n")
    report.write("P-Value: " + str(p_value) + "\n")
    report.write(hypothesis_result + "\n\n")

    report.write("7. KEY FINDINGS\n")
    report.write(
        "Overall Survival Rate: "
        + str(round(survival_rate, 2)) + "%\n"
    )
    report.write(
        "Average Age: "
        + str(round(df_clean["age"].mean(), 2)) + "\n"
    )
    report.write(
        "Average Fare: "
        + str(round(df_clean["fare"].mean(), 2)) + "\n"
    )

print("\nEDA Report saved successfully!")

# ------------------------------------------------------------
# 14. FINAL COMPLETION MESSAGE
# ------------------------------------------------------------

print("\nGenerated files:")
print("Dataset: dataset/titanic.csv")
print("Plots: plots/")
print("Results: results/")

print("\nEDA PROJECT COMPLETED SUCCESSFULLY!")
