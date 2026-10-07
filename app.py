import streamlit as st
import joblib
import pandas as pd
import string


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Emotion Classification",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# SMALL TYPOGRAPHY + BUTTON ADJUSTMENTS
# =========================================================

st.markdown(
    """
    <style>

    /* Slightly larger normal text */
    .stMarkdown p {
        font-size: 1rem;
    }

    /* Slightly larger sidebar text */
    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 1rem;
    }

    section[data-testid="stSidebar"] .stCaption {
        font-size: 0.9rem;
    }

    /* Keep prediction button blue */
    div.stButton > button[kind="primary"] {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 7px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("emotion_model.pkl")


artifact = load_model()

vectorizer = artifact["vectorizer"]
model = artifact["model"]
stop_words = artifact["stop_words"]
emotion_names = artifact["emotion_names"]
accuracy = artifact["accuracy"]


# =========================================================
# EMOTION EMOJIS
# =========================================================

emotion_emojis = {
    "sadness": "😔",
    "anger": "😠",
    "love": "❤️",
    "surprise": "😮",
    "fear": "😨",
    "joy": "😊"
}


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = str(text).lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = "".join(
        char for char in text
        if not char.isdigit()
    )

    # Remove non-ASCII characters
    text = "".join(
        char for char in text
        if char.isascii()
    )

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word.lower() not in stop_words
    ]

    return " ".join(words)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧠 Model Summary")

    st.write("")

    st.caption("Test Accuracy")

    st.markdown(
        f"### {accuracy:.2%}"
    )

    st.caption("Emotion Classes")

    st.markdown(
        f"### {len(emotion_names)}"
    )

    st.caption("Model")

    st.markdown(
        "**Logistic Regression**"
    )

    st.divider()

    st.subheader("Emotions")

    for emotion in emotion_names:

        emoji = emotion_emojis.get(
            emotion,
            "•"
        )

        st.write(
            f"{emoji} {emotion.capitalize()}"
        )

    st.divider()

    st.caption(
        "Feature extraction: TF-IDF"
    )


# =========================================================
# HEADER
# =========================================================

st.title("🧠 Emotion Classification")

st.write(
    "A simple NLP model that analyzes text and predicts "
    "the emotion it most closely represents."
)


# =========================================================
# EMOTION OVERVIEW
# =========================================================

st.subheader("What does it predict?")

st.write(
    "The model classifies text into six emotion categories:"
)

emotion_cols = st.columns(6)

for col, emotion in zip(
    emotion_cols,
    emotion_names
):

    with col:

        emoji = emotion_emojis.get(
            emotion,
            "•"
        )

        st.markdown(
            f"**{emoji} {emotion.capitalize()}**"
        )


# =========================================================
# PREDICTION SECTION
# =========================================================

st.divider()

st.subheader("🔍 Predict an Emotion")

st.write(
    "Enter a sentence below and the model will analyze "
    "the text and predict the most likely emotion."
)


text = st.text_area(
    "Enter your text",
    placeholder="Example: I am really excited about my new project!",
    height=140
)


predict_button = st.button(
    "Predict Emotion",
    type="primary"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if not text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        # ---------------------------------------------
        # Preprocess
        # ---------------------------------------------

        cleaned_text = preprocess_text(text)


        # ---------------------------------------------
        # TF-IDF transformation
        # ---------------------------------------------

        text_vector = vectorizer.transform(
            [cleaned_text]
        )


        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        prediction = model.predict(
            text_vector
        )[0]


        # ---------------------------------------------
        # Probabilities
        # ---------------------------------------------

        probabilities = model.predict_proba(
            text_vector
        )[0]


        # ---------------------------------------------
        # Emotion
        # ---------------------------------------------

        emotion = emotion_names[prediction]

        confidence = probabilities[prediction]

        emoji = emotion_emojis.get(
            emotion,
            "🧠"
        )


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.info(
                f"{emoji} **Predicted Emotion: "
                f"{emotion.capitalize()}**"
            )

        with result_col2:

            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )


        # =================================================
        # PROBABILITIES
        # =================================================

        st.subheader("Emotion Probabilities")

        probability_df = pd.DataFrame(
            {
                "Emotion": [
                    f"{emotion_emojis.get(e, '•')} "
                    f"{e.capitalize()}"
                    for e in emotion_names
                ],
                "Probability": probabilities * 100
            }
        )

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Emotion")[
                "Probability"
            ]
        )


        # =================================================
        # PROCESSED TEXT
        # =================================================

        with st.expander("View processed text"):

            st.write(
                "The input is processed using the same "
                "preprocessing steps used during training."
            )

            st.code(cleaned_text)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

with st.expander("⚙️ How does the model work?"):

    st.markdown(
        """
**1. Text preprocessing**

Lowercasing → punctuation removal → number removal →
non-ASCII character removal → stopword removal

**2. TF-IDF**

The cleaned text is converted into numerical features
using TF-IDF.

**3. Logistic Regression**

The TF-IDF features are passed to a Logistic Regression
classifier.

**4. Prediction**

The model predicts one of the six emotion categories
and provides a probability for each class.
"""
    )


with st.expander("📊 Model details & evaluation"):

    st.write(
        "**Model:** Logistic Regression"
    )

    st.write(
        "**Feature extraction:** TF-IDF"
    )

    st.write(
        f"**Emotion classes:** {len(emotion_names)}"
    )

    st.write(
        f"**Test accuracy:** {accuracy:.2%}"
    )

    st.write(
        "The model was evaluated on a held-out test set "
        "separate from the training data."
    )


with st.expander("🔄 Project pipeline"):

    st.code(
        """
Raw Text
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Emotion Prediction
        """.strip()
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Python · Scikit-learn · Streamlit"
)