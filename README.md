# 🏭 Manufacturing Defect Analysis & Optimization

## 📌 Project Overview
This project analyzes defect rates in a simulated manufacturing environment to identify key factors influencing defects and proposes data-driven optimizations. It is designed to demonstrate proficient SQL data manipulation, visualization, and insights relevant to industry standards.

## 🎯 Objective
- Identify key factors affecting defect rates in manufacturing.
- Optimize production efficiency by reducing defects using SQL and machine learning.

## 🔍 Approach & Methodology
### 1️⃣ SQL Data Processing & Cleaning
- Creating an environment in Jupyter to use PostgreSQL and connect to the database. 
- Load dataset into a SQL database (PostgreSQL via Jupyter Notebook).
- Handle missing/inconsistent data (handle nulls, ensure correct data types). 

### 2️⃣ Exploratory Data Analysis (EDA) with Advanced SQL Queries
- Simple queries to extract data, providing a general overview of the statistics (min, max, avg).
- Complex queries to extract the relationship between defect percentage and other variables.
- Identify and visualize significant trends from data analysis output using Python.

### 3️⃣ Model Prediction and Validation
- Implement a simple Machine Learning model (Logistic Regression, Decision Tree, or Random Forest) to classify defect risk based on SQL-extracted features.
- Optimize model for improved accuracy.
- Conduct post-analysis and review observations.

## 📊 Key Findings
1. **Random Forest is the Best Predictor**
   - Among Logistic Regression, Decision Tree, and Random Forest, **Random Forest had the highest accuracy (0.95).**
   - Suggests defect status is influenced by complex, non-linear relationships among multiple factors.

2. **Key Features Influencing Defects**
   - Production Volume, Supplier Quality, Maintenance Hours, and Worker Productivity had the strongest impact on defect status.
   - Higher safety incidents correlated with a lower defect rate, possibly due to stricter quality control.

3. **Safety Incidents & Defect Rate Anomaly**
   - Higher safety incidents unexpectedly correlated with lower defect percentages.
   - Possible explanations:
     - Safety incidents trigger corrective actions (e.g., more inspections, improved protocols).
     - Factories with frequent safety incidents may also have better documentation and process monitoring.
     - Other confounding factors like production scale or process efficiency could be at play.

## 📈 Model Performance Metrics
### Logistic Regression:
- Precision: 0.8630
- Recall: 0.8765
- F1 Score: 0.8604

### Decision Tree:
- Precision: 0.9009
- Recall: 0.8997
- F1 Score: 0.9003

### Random Forest:
- Precision: 0.9506
- Recall: 0.9506
- F1 Score: 0.9479
- **Best Parameters**: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
- **Best Accuracy**: 0.9614

## 🚀 How to Install and Run the Project
### ✅ Prerequisites
Ensure you have the following installed:
- Python 3
- Jupyter Notebook
- PostgreSQL

### 📥 Installation Steps
1. Clone the Repository:
   ```bash
   git clone https://github.com/ScottTorzewski/Manufacturing-Defect-ML-Project.git
   ```
2. Navigate to the Project Directory:
   ```bash
   cd Manufacture-Defect-ML-Project
   ```
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open the `.ipynb` file and run the cells sequentially.

## 🎯 How to Use the Project
1. Run all notebook cells sequentially to preprocess data, analyze trends, and generate visualizations.
2. Review SQL query results for defect rate analysis.
3. Adjust SQL queries and machine learning hyperparameters to explore further optimizations.
4. Experiment with alternative ML models.

## 📂 Dataset
🔗 Original Dataset: [Kaggle - Predicting Manufacturing Defects Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset/data)

## 📜 License
This project is licensed under the **GNU General Public License v3.0**. See the LICENSE file for details.
```

