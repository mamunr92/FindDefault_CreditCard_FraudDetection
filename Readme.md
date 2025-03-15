# **FindDefault: Credit Card Fraud Detection**


## **Project Overview**

Credit card fraud is a major challenge for financial institutions, leading to significant financial losses. The goal of this project is to build a **Machine Learning model** that accurately detects fraudulent transactions using a real-world dataset.

This project follows a structured data science workflow, including **data cleaning, feature engineering, model selection, evaluation, and deployment**.

---

## 📊 **Dataset Details**

- **Source:** European credit card transactions (September 2013)
- **Total Transactions:** 284,807
- **Fraud Cases:** 492 (**0.172% of data, highly imbalanced**)
- **Features:**
  - `Time`: Seconds since the first transaction
  - `Amount`: Transaction amount
  - `V1 - V28`: Anonymized transaction features (PCA-transformed)
  - `Class`: (0 = Legitimate, 1 = Fraudulent)

---

## 🛠 **Project Workflow**

### **2. Data Preprocessing**

- Checked and removed **missing values** and **duplicates**.
- Handled **outliers** using Z-score method.
- Applied **StandardScaler** to normalize `Amount`.

### **2. Feature Engineering**

- Extracted **‘Hour of the Day’** from `Time`.
- Removed **highly correlated features (V21–V28)**.

### **3. Handling Class Imbalance**

- Since fraud cases were very few (**highly imbalanced dataset**), we applied **undersampling** to balance the dataset.

### **4. Model Selection & Training**

- **Baseline Model:** Logistic Regression
- **Advanced Model:** Random Forest
- **Optimization:** Hyperparameter tuning using `GridSearchCV`

### **5. Model Evaluation**

| Model                   |   | Precision | Recall | F1-Score |
| ----------------------- | - | --------- | ------ | -------- |
| **Logistic Regression** |   | 80.0%     | 80.0%  | 80.0%    |
| **Random Forest**       |   | 70.0%     | 70.0%  | 70.0%    |

### **6. Feature Importance Analysis**

- Identified **key features influencing fraud detection**.
- Visualized feature importance using **Logistic Regression & Random Forest**.

### **7. Model Saving & Deployment** (Optional)

- Saved the trained model as `fraud_detector.pkl` using **joblib**.

---

##  **Project Folder Structure**

```
finddefault-project/
│-- notebooks/         # Jupyter Notebooks
│-- data/              # Raw & Processed Data
│   │-- creditcard.csv  # Original dataset
│   │-- processed_data.csv  # Cleaned & Preprocessed dataset
│-- models/            # Trained Models (.pkl files)
│   │-- fraud_detector.pkl  # Final trained model
│-- visuals/           # Saved graphs & images
│-- README.md          # Project Documentation
│-- requirements.txt   # Python Dependencies
```

---

##  **How to Run the Project**

### **1. Install Dependencies**

Run the following command to install required Python libraries:

```bash
pip install -r requirements.txt
```

### **2. Run the Jupyter Notebook**

Launch Jupyter Notebook and open `FindDefault.ipynb`:

```bash
jupyter notebook
```

Run each cell step by step to execute **data processing, model training, and evaluation**.

### **3. Save and Load the Model**

To save the trained model:

```python
import joblib
joblib.dump(grid_search.best_estimator_, "models/fraud_detector_machine.pkl")
```

To load the saved model:

```python
model = joblib.load("models/fraud_detector_machine.pkl")
```
