# NLP Emotion Classification

Classifies text into one of six emotions using **TF-IDF + Logistic Regression**, served through a **Streamlit** app for real-time prediction.

| 😔 Sadness (0) | 😠 Anger (1) | ❤️ Love (2) | 😮 Surprise (3) | 😨 Fear (4) | 😊 Joy (5) |
|:-:|:-:|:-:|:-:|:-:|:-:|

## Model Performance

| Dataset | Split | Features | Classifier | Classes | Test Accuracy |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 16,000 samples | 80 / 20 (`random_state=42`) | TF-IDF | Logistic Regression | 6 | **86.16%** |

> Accuracy is measured on a held-out test set not used during training.

## Pipeline

`Raw Text` → `Lowercase` → `Remove Punctuation` → `Remove Numbers` → `Remove Non-ASCII` → `Remove Stopwords` → `TF-IDF` → `Logistic Regression` → `Emotion`

The same preprocessing and trained vectorizer are reused at inference time.

## Tech Stack

| Language | Data | ML | Serialization | App | Dev |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Python | pandas | scikit-learn | joblib | Streamlit | Jupyter, VS Code |

## Project Structure

| File | Description |
|---|---|
| `app.py` | Streamlit app for emotion prediction |
| `emotion_model.pkl` | Trained model, TF-IDF vectorizer and config |
| `nlp-emotion-classification.ipynb` | Model development and evaluation |
| `train.txt` | Labelled text dataset |
| `requirements.txt` | Python dependencies |

## Run Locally

```bash
git clone https://github.com/arshiafreen090/NLP-Emotion-Classification.git
cd NLP-Emotion-Classification
pip install -r requirements.txt
streamlit run app.py
```

## App Features

Text input • Predicted emotion • Confidence score • Probability distribution across all classes • Processed text view • Model info

## Limitations

Built for NLP learning and demonstration. Emotion is subjective, and the model may miss context or nuance. Confidence scores are not a measure of a person's actual emotional state.

---
Built with Python, scikit-learn and Streamlit.
