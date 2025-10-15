# Probabilistic Reasoning for Medical Diagnosis 🩺

This project explores the application of probabilistic reasoning to the complex task of medical diagnosis. [cite_start]It implements and compares multiple models—**Bernoulli Naive Bayes**, **Random Forest**, and a **Bayesian Network**—to predict diseases from a set of patient-reported symptoms[cite: 13, 90, 124]. [cite_start]The primary goal is to demonstrate how to frame diagnosis as a problem of probabilistic inference under uncertainty and to highlight the critical importance of data quality in building effective predictive models[cite: 12, 14, 128].

[cite_start]The project culminates in an interactive web application built with Streamlit, where users can select symptoms and receive a ranked list of likely diseases from each model[cite: 96].



## ✨ Key Features

* [cite_start]**Probabilistic Modeling**: Implements both a simple Naive Bayes classifier and a more expressive Bayesian Network to compute the probability of diseases given symptoms ($P(Disease | Symptoms)$)[cite: 13, 22].
* **Comparative Analysis**: Trains a Random Forest classifier as a powerful machine learning baseline to compare against the probabilistic models.
* [cite_start]**Rigorous Data Cleaning**: Features a robust data preprocessing pipeline that identifies and resolves critical dataset issues like duplicate entries and "perfect predictors" to ensure the models learn meaningful patterns rather than memorizing data flaws[cite: 26, 27, 29, 128].
* [cite_start]**Interactive UI**: A user-friendly web application built with Streamlit (`app.py`) allows for real-time disease prediction by selecting from a list of 51 common symptoms[cite: 31, 96, 97].
* [cite_start]**Comprehensive Evaluation**: Models are evaluated on a held-out test set using multiple metrics, including Accuracy, Top-3 Accuracy, and F1-Score, to assess their real-world diagnostic utility[cite: 88, 91].

---

## 🧠 Methodology and Approach

[cite_start]Medical diagnosis is a classic challenge of reasoning under uncertainty[cite: 18]. [cite_start]Symptoms can be ambiguous, and information is often incomplete[cite: 19, 20]. This project tackles the problem methodically.

### The Challenge: Data Integrity

[cite_start]The initial phase of the project immediately revealed a critical lesson in machine learning: **data quality is paramount**[cite: 14, 128]. [cite_start]The original dataset led to perfect accuracy scores, a clear red flag indicating that the model wasn't learning—it was exploiting flaws in the data[cite: 26].

Two major issues were discovered using the `analyze_data.py` script and rectified in `train.py`:

1.  [cite_start]**Duplicate Data**: The dataset contained thousands of duplicate rows which were removed[cite: 27]. This prevents the models from being biased towards the over-represented samples.
2.  [cite_start]**"Perfect Predictors"**: An analysis showed that 80 symptoms were uniquely associated with a single disease[cite: 29]. These symptoms make the prediction task trivial and unrealistic. For example, if 'receiving unsterile injections' only ever appeared for 'Hepatitis C' in the dataset, the model learns a simple rule, not a probabilistic relationship. [cite_start]These symptoms were excluded to create a more challenging and realistic problem space where the models must learn from the overlapping relationships between **51 common symptoms and 41 diseases**[cite: 31].

### Modeling Approaches

With a clean dataset, we explored three different modeling techniques.

#### 1. Bernoulli Naive Bayes

* [cite_start]**Foundation**: This model is based on Bayes' Theorem ($P(D|E) \propto P(E|D)P(D)$) and serves as an excellent probabilistic baseline[cite: 35, 36].
* [cite_start]**The "Naive" Assumption**: Its core assumption is that all symptoms are **conditionally independent** of each other, given the disease ($P(S_{1},S_{2},...|D)=\prod_{i=1}^{n}P(S_{i}|D)$)[cite: 40, 41].
* [cite_start]**Analogy**: As described in the project presentation, this model acts like an "evidence collector" where each symptom provides an independent vote for a disease, without understanding that symptoms like 'fever' and 'chills' are often related[cite: 49, 50].

#### 2. Random Forest

* **Foundation**: An ensemble learning method that builds a multitude of decision trees and outputs the mode of their predictions.
* **Purpose**: Included as a powerful, non-probabilistic baseline to see how a standard, high-performance classifier compares to our probabilistic models. It is known for its high accuracy and robustness.

#### 3. Bayesian Network

* [cite_start]**Goal**: To move beyond the "naive" assumption and create a model that can capture the real-world dependencies between symptoms[cite: 54]. For instance, a fever can cause chills.
* **Implementation**: The `train.py` script implements a Bayesian Network where the disease is the parent node to all symptom nodes. This structure explicitly models the probability of a symptom appearing given a specific disease. The `pgmpy` library is then used to perform inference, calculating the posterior probability of each disease given the evidence (selected symptoms).
* [cite_start]**Reasoning**: This network can understand context[cite: 82]. [cite_start]For example, the evidence of 'chills' is interpreted differently depending on whether 'fever' is also present, leading to a more sophisticated form of reasoning[cite: 83].

---

## 📂 Project Structure

The repository is organized as follows:
```bash
├── 🐍 analyze_data.py       # Script to inspect the dataset for issues
├── 🐍 app.py                # The Streamlit web application
├── 🐍 train.py              # Main script for data preprocessing, model training, and evaluation
├── 📦 model_bernoulli_nb.joblib # Saved Naive Bayes model
├── 📦 model_rf.joblib       # Saved Random Forest model
├── 📦 model_bn.pkl          # Saved Bayesian Network model
├── 📦 label_encoder.pkl      # Saved label encoder for disease names
├── 📦 feature_names.pkl     # Saved list of symptom features
├── 📄 disease_symptoms.csv  # The raw dataset
└── 📄 README.md             # This README file
 ```
---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.8+
* `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install pandas numpy scikit-learn pgmpy streamlit
    ```

---

## ⚙️ How to Run

Execute the scripts in the following order.

### Step 1: Analyze the Dataset (Optional)

Run this script to see the analysis of the raw data, including the discovery of perfect predictors and duplicate rows.

```bash
python analyze_data.py
```
### Step 2: Train the Models
This is the main script that will preprocess the data, train all three models, evaluate them, and save the final model artifacts to disk.

```bash
python train.py
```
After training, you will see evaluation metrics for each model printed to the console.
### Step 3: Launch the Prediction App
Once the models are trained and saved, launch the Streamlit web application.

```bash
streamlit run app.py
```
Your web browser should open with the interactive application, ready for you to select symptoms and see the predictions.
## 📊 Results and Evaluation

[cite_start]The models were trained on 80% of the cleaned data and evaluated on the remaining 20% unseen test set[cite: 88]. [cite_start]The key results demonstrate high performance and prove the models are effective as a diagnostic support tool[cite: 91, 130].

| Model                   | Test Set Accuracy | Test Set Top-3 Accuracy |
| :---------------------- | :---------------: | :---------------------: |
| **Bernoulli Naive Bayes** |      91.80%       |         95.08%          |
| **Random Forest** |      95.53%       |          ~98%           |
| **Bayesian Network** |    **94.26%** |         **~99%** |

[cite_start]The Bayesian Network achieved superior accuracy, confirming that modeling inter-symptom dependencies provides a more robust and realistic model of the problem domain[cite: 124].

**Application Interface:**



---

## 💡 Conclusion

This project successfully demonstrates that:

* [cite_start]**Data Quality is Everything**: The most crucial step was data investigation and cleaning, which transformed the project from a memorization task into a true reasoning problem[cite: 128].
* [cite_start]**Model Complexity Matters**: There is a clear benefit to moving from the strong independence assumptions of Naive Bayes to a Bayesian Network that can capture more complex relationships, resulting in higher accuracy[cite: 129].
* [cite_start]**Probabilistic Reasoning is Effective**: The final models effectively handle the ambiguity and uncertainty of medical diagnosis, providing a ranked list of potential diseases that can serve as a valuable differential diagnosis support tool[cite: 130].

---

## 🛠️ Technologies Used

* **Python**: Core programming language
* **Pandas & NumPy**: Data manipulation and numerical operations
* **Scikit-learn**: For training Naive Bayes, Random Forest, and model evaluation
* **pgmpy**: For creating, training, and running inference on the Bayesian Network
* **Streamlit**: For building the interactive web application
* **Joblib & Pickle**: For saving and loading trained models
