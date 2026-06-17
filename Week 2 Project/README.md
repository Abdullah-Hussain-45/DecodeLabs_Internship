Markdown
# Data Classification Using Artificial Intelligence

An introductory machine learning project implementing a **K-Nearest Neighbors (KNN)** classifier to identify flower species using the classic **Iris Dataset**. This script walks through the complete data science pipeline: dataset initialization, feature scaling, model training, predictive inference, and rigorous statistical validation.

---

## 🛠️ System Workflow & Architecture

The project implements a 4-step structured machine learning pipeline:

[ 1. Load Iris Dataset ] ──> [ 2. Feature Scaling (StandardScaler) ]
│
▼
[ 4. Performance Validation ] <── [ 3. KNN Model Training & Prediction ]


1. **Dataset Loading & Exploration:** Ingests the raw biological metrics and constructs a structured data frame using `pandas`.
2. **Feature Standardization:** Transforms spatial dimensions using `StandardScaler` to ensure equal geometric weighting.
3. **Classification Model:** Maps features using localized multidimensional distance metrics via `KNeighborsClassifier`.
4. **Statistical Validation:** Generates precise metrics and multi-class confusion matrices to measure performance[cite: 1].

---

## 📊 Core Concepts Explained

### 1. K-Nearest Neighbors (KNN)
KNN is a non-parametric, instance-based learning algorithm. It classifies a test sample based on the majority vote of its $k$ closest neighbors within the multi-dimensional feature space. 
In this implementation, **$k=5$**, meaning the model identifies the 5 training samples nearest to the unknown data point using **Euclidean distance**[cite: 1]:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

### 2. Feature Scaling (Standardization)
Because KNN relies entirely on distance metrics, features with larger raw numeric ranges could dominate the model's geometric distance calculations. `StandardScaler` eliminates this bias by scaling features to achieve a mean of 0 ($\mu = 0$) and a standard deviation of 1 ($\sigma = 1$)[cite: 1]:

$$z = \frac{x - \mu}{\sigma}$$

### 3. Model Evaluation Metrics
* **Accuracy:** The percentage of instances correctly predicted[cite: 1].
* **Confusion Matrix:** An explicit grid mapping actual vs. predicted classifications to visualize cross-class misidentifications[cite: 1].
* **True Positives (TP) / False Positives (FP) / False Negatives (FN) / True Negatives (TN):** Computed systematically for each target flower category (*Setosa*, *Versicolor*, *Virginica*)[cite: 1].

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ and the necessary libraries installed:

```bash
pip install pandas scikit-learn
Installation & Execution
Clone or download this repository.

Save the main script as Data_Classification_Using_AI.py[cite: 1].

Execute the script via your terminal:

Bash
python Data_Classification_Using_AI.py
📋 Sample Terminal Output Preview
When executed successfully, the script displays structured tabular updates tracking the pipeline's lifecycle[cite: 1]:

Plaintext
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0

Total Samples:  150
Training Samples:  120
Testing Samples:  30

Scaled Training Data: 
 [[-1.47393699  1.22037978 -1.56391443 -1.30948358]
  [-0.13868612  3.09000842 -1.2774797  -1.04292196]]

Actual Labels:     [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
Predicted Labels:  [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]

Accuracy: 100.00%

Category of Flower: SETOSA
                     Predicted Positive  Predicted Negative
Actual Positive (Y)                  10                   0
Actual Negative (N)                   0                  20

Classification Report: 
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11
📂 Codebase Breakdown
load_iris(): Ingests 150 total samples of iris measurements (3 distinct species, 50 samples each)[cite: 1].

train_test_split(): Allocates 80% of the dataset for model training (120 samples) and reserves 20% for out-of-sample evaluation (30 samples), setting a fixed random_state=42 to guarantee reproducible splits[cite: 1].

KNeighborsClassifier(n_neighbors=5): Instantiates the primary classification algorithm[cite: 1].

classification_report(): Generates high-level statistical indices breaking down Precision, Recall, and F1-Score per taxonomic category[cite: 1].
