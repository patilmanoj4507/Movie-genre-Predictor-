
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CinePredict | Movie Genre AI",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path(__file__).parent / "movie_genre_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.info("Make sure movie_genre_model.pkl is in the project folder.")
    st.stop()

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101322 0%, #171b30 100%);
    color: #f5f5f5;
}
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
}
.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(110deg, #262c49, #42285a);
    border: 1px solid #41476a;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #d5d8ec;
    font-size: 16px;
}
div.stButton > button {
    background: #8b5cf6;
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 45px;
    font-weight: bold;
}
div.stButton > button:hover {
    background: #7044d4;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 CinePredict AI</h1>
    <p>
        Discover a movie's genre using machine learning.
        Enter a plot summary and let our AI model analyze the story.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🎥 CinePredict")
    st.write("Movie Genre Classification")
    st.divider()
    st.subheader("How it works")
    st.write("1. Enter a movie plot.")
    st.write("2. TF-IDF converts text into numerical features.")
    st.write("3. Logistic Regression predicts the genre.")
    st.divider()
    st.caption("Built with Python, Streamlit and Scikit-learn.")

# ---------------- INPUT ----------------
st.subheader("Analyze a movie plot")
st.write("Describe the story in a few sentences.")

examples = {
    "Choose an example": "",
    "Action": (
        "A brave soldier fights a group of criminals "
        "to rescue hostages from a dangerous city."
    ),
    "Comedy": (
        "Two friends get into hilarious situations "
        "while organizing a surprise birthday party."
    ),
    "Horror": (
        "A family moves into an abandoned house where "
        "strange noises and terrifying events occur every night."
    ),
    "Drama": (
        "A young woman struggles to rebuild her life "
        "after losing her job and reconnects with her family."
    )
}

selected_example = st.selectbox(
    "Try a sample plot",
    list(examples.keys())
)

plot = st.text_area(
    "Movie plot summary",
    value=examples[selected_example],
    height=150,
    placeholder="Example: A detective investigates a mysterious disappearance..."
)

predict_button = st.button(
    "✨ Predict Movie Genre",
    use_container_width=True
)

# ---------------- PREDICTION ----------------
if predict_button:
    if not plot.strip():
        st.warning("Please enter a movie plot first.")

    elif len(plot.strip()) < 20:
        st.warning("Please enter a more descriptive plot (at least 20 characters).")

    else:
        with st.spinner("Analyzing the movie plot..."):
            try:
                predicted_genre = model.predict([plot])[0]

                st.divider()
                st.subheader("Prediction Results")

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(110deg, #253a43, #26334f);
                        border: 1px solid #46616d;
                        padding: 25px;
                        border-radius: 15px;
                        text-align: center;">
                        <p style="color:#c4d5e5;font-size:16px;">
                            PREDICTED GENRE
                        </p>
                        <h1 style="color:#7ee7c2;font-size:36px;">
                            {predicted_genre.title()}
                        </h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Show the model's relative class scores
                probabilities = model.predict_proba([plot])[0]
                classes = model.classes_

                results = pd.DataFrame({
                    "Genre": classes,
                    "Score": probabilities
                }).sort_values("Score", ascending=False)

                top_three = results.head(3).copy()

                st.subheader("Top 3 Genre Scores")
                st.caption(
                    "These are model confidence scores, not guaranteed "
                    "probabilities that the genre is objectively correct."
                )

                chart_data = top_three.set_index("Genre")
                st.bar_chart(chart_data["Score"])

                for _, row in top_three.iterrows():
                    st.write(
                        f"**{row['Genre'].title()}** — "
                        f"{row['Score'] * 100:.2f}% model score"
                    )

                st.info(
                    "This model predicts among the genres included "
                    "during training. A prediction may be incorrect, "
                    "especially for genres it was not trained to recognize."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
