# Explainable Fake Review Detection using Ensemble Machine Learning and SHAP

An end-to-end NLP and tabular machine learning framework designed to detect computer-generated (CG) versus original (OR) consumer reviews. The pipeline integrates text preprocessing, structural and linguistic metadata extraction, hybrid feature vectorization (TF-IDF + statistical features), multi-model ensemble classification, and post-hoc interpretability via SHAP (SHapley Additive exPlanations).

---

## 📌 Project Overview

E-commerce platforms face severe trust and integrity challenges due to deceptive, computer-generated reviews. This project develops an end-to-end classification system that:
1. Distinguishes authentic customer reviews (`OR`) from synthetic/deceptive reviews (`CG`).
2. Extracts domain-specific linguistic signals (e.g., lexical diversity, punctuation anomalies, capital ratios, ratings).
3. Evaluates diverse classical classifiers, ensemble methods (Voting, Stacking, AdaBoost, Random Forest), and gradient boosting (XGBoost).
4. Explains individual model decisions using SHAP value attribution.
5. Deploys an interactive inference interface via Gradio.

---

## 📊 Dataset Description

The pipeline utilizes `fake_reviews_dataset.csv`, consisting of **40,432 raw records** across 10 diverse retail categories:

| Column | Type | Description |
| :--- | :--- | :--- |
| `category` | Categorical (10 classes) | Product domain (e.g., `Home_and_Kitchen_5`, `Sports_and_Outdoors_5`) |
| `rating` | Numeric (1.0 to 5.0) | Customer rating (Mean: ~4.26) |
| `label` | Categorical (`CG`, `OR`) | Target ground truth: Computer-Generated (`CG`) vs. Original (`OR`) |
| `text` | String | Raw customer review content |

### Class Balance & Preprocessing
* **Class Distribution:** Perfectly balanced at **50.00% Fake (`CG`)** and **50.00% Real (`OR`)** (20,210 samples per class after initial deduplication).
* **Target Encoding:**
  $$\text{target} = \begin{cases} 0, & \text{if } \text{label} = \text{'CG'} \text{ (Fake)} \\ 1, & \text{if } \text{label} = \text{'OR'} \text{ (Real)} \end{cases}$$

---

## 🛠 Feature Engineering Pipeline

The model does not rely on text tokens alone; it combines n-gram representations with metadata and engineered statistical signals:

### 1. Linguistic & Surface-Level Features
* **`num_characters`**: Total raw character count per review.
* **`num_words`**: Token count derived via `nltk.word_tokenize`.
* **`num_sentences`**: Sentence count derived via `nltk.sent_tokenize`.
* **`capital_ratio`**: Proportion of uppercase characters $\left( \frac{\sum [c \in \text{Upper}]}{\text{length}(x) + 1} \right)$ to capture aggressive emphasis or synthetic casing.
* **`exclamation_count`**: Frequency of exclamation marks (`!`).

### 2. Tabular Metadata
* **`category_encoded`**: Integer encoding of the 10 categorical product verticals via `LabelEncoder`.
* **`rating`**: Normalized numerical product rating (scaled using `MinMaxScaler` alongside metadata).

### 3. Text Vectorization
* **Preprocessing:** Lowercasing, punctuation stripping, alphanumeric filtering, English stopword removal (`nltk.corpus.stopwords`), and lemmatization via `WordNetLemmatizer`.
* **Feature Extraction:** Sub-linear term frequency-inverse document frequency (`TfidfVectorizer`).
* **Feature Union:** Sparse concatenation (`scipy.sparse.hstack`) combining the TF-IDF feature matrix with scaled numerical/metadata feature vectors.

---

## 🔬 Model Architecture & Benchmarking

The training pipeline evaluates a diverse suite of classical, tree-based, and ensemble architectures:

* **Linear Classifiers:** Logistic Regression (L2 regularized)
* **Probabilistic Models:** Multinomial Naive Bayes (`MultinomialNB`)
* **Instance-Based Models:** $k$-Nearest Neighbors (`KNeighborsClassifier`)
* **Tree-Based Models:** Decision Trees, Random Forest (`RandomForestClassifier`)
* **Boosting:** AdaBoost (`AdaBoostClassifier`), XGBoost (`XGBClassifier`)
* **Ensemble Methods:** 
  * `VotingClassifier` (Soft/Hard probability aggregation)
  * `StackingClassifier` (Meta-classifier stacking over heterogeneous base estimators)

### Evaluation Metrics
Models are validated using stratified train-test splits and cross-validation across:
* Classification Accuracy, Precision, Recall, and Macro $F_1$-score
* Confusion Matrices (False Positive / False Negative trade-offs)
* Receiver Operating Characteristic – Area Under Curve (ROC-AUC)

---

## 🔍 Model Explainability (SHAP)

To prevent black-box decision-making, the project leverages **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:
* **Global Interpretability:** Summary beeswarm plots and feature importance bar plots identify whether synthetic reviews are driven more by surface metrics (e.g., rating extremity, exclamation marks, short length) or specific lexical tokens.
* **Local Interpretability:** Force plots and waterfall plots visualize token-level positive and negative contributions for individual review predictions.

---

## 🚀 Interactive UI (Gradio)

The notebook includes an interactive local web app built with **Gradio**:
* Allows real-time user input of review text, rating, and product category.
* Runs the end-to-end preprocessing, feature scaling, and ensemble inference.
* Displays the deception probability score and predicted classification (`Real` vs. `Fake`).

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/Rimi-351/fake-review-detection.git](https://github.com/Rimi-351/fake-review-detection.git)
cd fake-review-detection
