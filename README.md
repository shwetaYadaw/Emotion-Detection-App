# Emotion Detection App

## Overview
Emotion Detection App is a Machine Learning and Natural Language Processing (NLP) based web application that predicts emotions from user-entered text. The application is built using Streamlit and can classify text into multiple emotions such as Joy, Sadness, Anger, Fear, Love, and Surprise.

## Features
- Real-time emotion prediction from text input
- Interactive and user-friendly Streamlit interface
- Text preprocessing and cleaning
- Emotion classification using Machine Learning models
- Supports multiple emotion categories
- Fast and accurate predictions

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- TF-IDF Vectorization
- Machine Learning

## Dataset
The model is trained on an emotion-labeled text dataset containing sentences categorized into:
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

## Machine Learning Workflow
1. Data Collection
2. Text Preprocessing
3. Feature Extraction using TF-IDF
4. Model Training
5. Model Evaluation
6. Deployment using Streamlit

## Project Structure

```
Emotion-Detection-App/
│
├── app.py
├── train.txt
├── requirements.txt
├── README.md
├── model.pkl
├── vectorizer.pkl
└── assets/
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/shwetaYadaw/Emotion-Detection-App.git
```

### Navigate to the Project Directory

```bash
cd Emotion-Detection-App
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

## Sample Input

```text
I am feeling very happy today because I achieved my goal.
```

### Output

```text
Predicted Emotion: Joy
```

## Future Enhancements
- Deep Learning based emotion classification
- Emotion probability visualization
- Multi-language support
- Voice-to-text emotion detection
- Emotion analytics dashboard

## Author

Shweta Yadav

GitHub: https://github.com/shwetaYadaw
