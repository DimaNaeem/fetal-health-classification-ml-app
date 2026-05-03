# Fetal Health Classification System

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

A machine learning system for predicting fetal health status using Cardiotocography (CTG) data. The system classifies fetal health into three categories:
- **Normal** (Class 1)
- **Suspect** (Class 2) 
- **Pathological** (Class 3)

The model achieves **93% accuracy** and identifies critical risk factors from fetal heart rate patterns.

## 📊 Dataset

- **Source**: UCI Machine Learning Repository - Cardiotocography Dataset (ID: 193)
- **Samples**: 2,126 fetal cardiotocograms
- **Features**: 21 numerical features from FHR and UC signals
- **Target**: NSP (Fetal State Classification)

### Class Distribution
| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| 1 | Normal | 1,655 | 77.8% |
| 2 | Suspect | 295 | 13.9% |
| 3 | Pathological | 176 | 8.3% |

## 🏆 Model Performance

### Random Forest Classifier (Optimized)
### Key Features for Prediction
1. **ASTV** (Abnormal Short Term Variability) - Most important
2. **ALTV** (Abnormal Long Term Variability)
3. **Mean** (Mean FHR value)
4. **MSTV** (Mean Short Term Variability)
5. **DP** (Progressive Decelerations)

## 🚀 Features

- **Interactive Web Application**: User-friendly Streamlit interface
- **Real-time Predictions**: Instant fetal health classification
- **Probability Scores**: Confidence levels for each class
- **Medical Guidance**: Interpretive insights for clinical decisions
- **Feature Importance**: Visualization of key risk factors

## 📁 Project Structure
fetal-health-classification-ml-app/
├── analysis.ipynb # Jupyter notebook with EDA & modeling
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── models/ # Saved model files
│ ├── fetal_health_best_model.pkl
│ └── scaler.pkl
├── data/ # Data directory (fetched dynamically)
├── feature_importance.csv # Feature importance rankings
└── README.md # Project documentation

text

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/DimaNaeem/fetal-health-classification-ml-app.git
cd fetal-health-classification-ml-app
2. Create virtual environment
bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac
3. Install dependencies
bash
pip install -r requirements.txt
4. Run the Streamlit app
bash
streamlit run app.py
💻 Usage
Using the Web App
Open your browser at http://localhost:8501

Enter the 21 CTG features in the input form

Click "Predict Fetal Health"

View the prediction result and probability scores

Using the Jupyter Notebook
Launch Jupyter: jupyter notebook

Open analysis.ipynb

Run cells sequentially to:

Load and explore data

Train models

Evaluate performance

Save the best model

📈 Technical Details
Data Preprocessing
Standard scaling of features

Train-test split (80-20) with stratification

Handling of class imbalance with weights

Models Evaluated
Random Forest (Best performer: 93% accuracy)

Gradient Boosting

Logistic Regression

Performance Metrics
Accuracy: 93%

Macro F1-Score: 0.87

Pathological Class Recall: 91%

🧪 Test Cases
Input Condition	Expected Output
Normal FHR pattern	Normal (Class 1)
High ASTV, ALTV	Suspect/Pathological
High decelerations (DP)	Pathological
Healthy accelerations (AC)	Normal
🩺 Clinical Significance
This system helps healthcare providers:

Early Detection: Identify fetal distress before delivery

Risk Stratification: Classify cases by urgency level

Decision Support: Provide objective analysis of CTG patterns

Monitoring Guidance: Recommend follow-up based on risk level

Note: This tool is for clinical decision support only. Always consult with healthcare professionals.



imbalanced-learn 0.10+


📝 License
This project is licensed under the MIT License.

👩‍⚕️ Author
Dima Naeem

GitHub: @DimaNaeem

🙏 Acknowledgments
UCI Machine Learning Repository for the CTG dataset

Streamlit for the web framework

scikit-learn community for ML tools

📧 Contact
For questions or collaboration, please open an issue on GitHub.

Built with ❤️ for fetal health monitoring




