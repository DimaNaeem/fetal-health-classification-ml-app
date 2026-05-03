# Fetal Health Prediction System

Machine learning system for predicting fetal health states using Cardiotocography (CTG) data.

## Dataset
- **Source**: UCI Machine Learning Repository - Cardiotocography Dataset
- **Instances**: 2,126
- **Features**: 21 numeric features from FHR and UC signals
- **Classes**: Normal (1), Suspect (2), Pathologic (3)

## Project Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for model development
- `src/`: Source code for data preprocessing and model training
- `app/`: Streamlit web application
- `models/`: Saved model files

## Setup Instructions
1. Clone repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment
4. Install requirements: `pip install -r requirements.txt`
5. Download dataset from UCI repository
6. Run `python src/models/train_model.py` to train the model
7. Launch app: `streamlit run app/streamlit_app.py`
