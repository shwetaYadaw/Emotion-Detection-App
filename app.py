import os
import re
import string
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', header=None, names=['text', 'emotion'])
    df = df.dropna().reset_index(drop=True)
    return df


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ''

    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(char for char in text if char.isascii())
    text = ' '.join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)
    return text.strip()


def create_banner(path: str = 'emotion_banner.png') -> str:
    if os.path.exists(path):
        return path

    width, height = 900, 300
    img = Image.new('RGB', (width, height), (255, 238, 238))
    draw = ImageDraw.Draw(img)

    colors = [(130, 210, 55), (244, 195, 74), (232, 70, 55)]
    centers = [(180, 160), (450, 160), (720, 160)]
    radius = 110

    for index, (cx, cy) in enumerate(centers):
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colors[index])

        left_eye = (cx - 40, cy - 40)
        right_eye = (cx + 20, cy - 40)
        for ex, ey in [left_eye, right_eye]:
            draw.ellipse([ex, ey, ex + 35, ey + 35], fill='white')
            draw.ellipse([ex + 10, ey + 10, ex + 25, ey + 25], fill='black')

        if index == 0:
            draw.arc([cx - 50, cy - 10, cx + 50, cy + 80], start=0, end=180, fill='black', width=10)
        elif index == 1:
            draw.line([cx - 40, cy + 35, cx + 40, cy + 35], fill='black', width=10)
        else:
            draw.arc([cx - 50, cy + 10, cx + 50, cy + 90], start=180, end=360, fill='black', width=10)
            draw.line([cx - 70, cy - 70, cx - 30, cy - 95], fill='black', width=12)
            draw.line([cx + 70, cy - 70, cx + 30, cy - 95], fill='black', width=12)

    img.save(path)
    return path


def build_model(data_path: str):
    df = load_data(data_path)
    df['text'] = df['text'].apply(preprocess_text)

    emotion_labels = sorted(df['emotion'].unique())
    label_to_index = {label: idx for idx, label in enumerate(emotion_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    df['emotion_id'] = df['emotion'].map(label_to_index)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['emotion_id'],
        test_size=0.20,
        random_state=42,
        stratify=df['emotion_id'],
    )

    model = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_features=8000)),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')),
    ])
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return model, index_to_label, score, df


def main():
    st.set_page_config(page_title='Emotion Classifier', page_icon='😊')

    st.markdown(
        """
        <style>
            .stApp {
                background-color: #efdfc5;
            }
            .css-18e3th9, .css-1d391kg, .css-1v3fvcr, .css-1outpf7 {
                background-color: #efdfc5;
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                background-color: #f3e2c8;
                border-radius: 18px;
                box-shadow: 0 0 30px rgba(0, 0, 0, 0.08);
            }
            .stButton>button {
                background-color: #b5651d;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    banner_path = create_banner()
    st.image(banner_path, width=900)
    st.title('Emotion Detection App')
    st.write('Enter a sentence below and click Predict to see the emotion and confidence scores.')

    model, index_to_label, test_score, df = build_model('train.txt')

    user_input = st.text_area('Input text', value='', height=200)
    if st.button('Predict emotion'):
        if not user_input.strip():
            st.warning('Please enter a sentence before predicting.')
        else:
            cleaned = preprocess_text(user_input)
            if not cleaned:
                st.warning('The text does not contain enough words after preprocessing. Please try another sentence.')
            else:
                predicted_index = model.predict([cleaned])[0]
                emotion = index_to_label[predicted_index]
                probabilities = model.predict_proba([cleaned])[0]
                prob_df = pd.DataFrame(
                    {'emotion': [index_to_label[i] for i in range(len(probabilities))], 'probability': probabilities}
                ).sort_values('probability', ascending=False).set_index('emotion')

                st.success(f'Predicted emotion: **{emotion}**')
                st.write(f'**Model accuracy:** {test_score:.2f}')
                st.write('---')
                st.write('Processed text:')
                st.write(cleaned)
                st.write('### Prediction confidence')
                st.bar_chart(prob_df)


if __name__ == '__main__':
    main()
