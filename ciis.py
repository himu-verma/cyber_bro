import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
from transformers import pipeline
import os

# -------------------------------
# Load Hugging Face Toxicity Classifier
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

toxic_model = load_model()

# -------------------------------
# File for saving history
# -------------------------------
HISTORY_FILE = "analysis_history.csv"

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["Post", "Sentiment", "Toxicity"])

def save_history(df_new):
    df_history = load_history()
    df_all = pd.concat([df_history, df_new], ignore_index=True)
    df_all.to_csv(HISTORY_FILE, index=False)
    return df_all

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.set_page_config(page_title="Cyberbro Analyzer", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #141E30, #243B55);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }
    h1, h2, h3 {
        color: #FFD700;
        text-shadow: 2px 2px 5px #000000;
    }
    .stRadio > div {
        background: #FF6F61;
        padding: 12px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        box-shadow: 2px 4px 12px rgba(0,0,0,0.5);
    }
    .stButton>button {
        background-color: #FF6F61;
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e55b50;
        box-shadow: 2px 6px 14px rgba(0,0,0,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚀 Cyberbro Social Media Impact Analyzer")

# -------------------------------
# Tabs for Analyzer & History
# -------------------------------
tab1, tab2 = st.tabs(["🧪 Analyzer", "📜 History"])

# -------------------------------
# Tab 1 - Analyzer
# -------------------------------
with tab1:
    st.subheader("📥 Choose Input Method")
    posts = []

    option = st.radio("", ("✍️ Enter Text", "📂 Upload CSV"))

    if option == "✍️ Enter Text":
        user_input = st.text_area("Paste social media posts (one per line):")

        if st.button("🔎 Analyze Posts"):
            if user_input.strip():
                posts = user_input.strip().split("\n")
            else:
                st.warning("⚠️ Please enter some posts to analyze.")

    elif option == "📂 Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file with a column named 'post'", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "post" not in df.columns:
                st.error("❌ CSV must contain a column named 'post'")
            else:
                posts = df["post"].dropna().tolist()

    # Run Analysis
    if posts:
        results = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        toxic_counts = {"Toxic": 0, "Safe": 0}

        for post in posts:
            # Sentiment
            sentiment = TextBlob(post).sentiment.polarity
            if sentiment > 0.2:
                sentiment_label = "Positive"
                sentiment_counts["Positive"] += 1
            elif sentiment < -0.2:
                sentiment_label = "Negative"
                sentiment_counts["Negative"] += 1
            else:
                sentiment_label = "Neutral"
                sentiment_counts["Neutral"] += 1

            # Toxicity
            predictions = toxic_model(post)[0]
            toxic_score = [p for p in predictions if p["label"] == "toxic"][0]["score"]
            if toxic_score > 0.5:
                toxicity = f"⚠️ Toxic ({toxic_score:.2f})"
                toxic_counts["Toxic"] += 1
            else:
                toxicity = f"✅ Safe ({toxic_score:.2f})"
                toxic_counts["Safe"] += 1

            results.append((post, sentiment_label, toxicity))

        # Results DataFrame
        results_df = pd.DataFrame(results, columns=["Post", "Sentiment", "Toxicity"])

        # Save to history
        save_history(results_df)

        # Show Results
        st.subheader("📊 Analysis Results")
        st.dataframe(results_df)

        # Sentiment Pie Chart
        st.subheader("📈 Sentiment Distribution")
        fig1 = px.pie(
            names=list(sentiment_counts.keys()),
            values=list(sentiment_counts.values()),
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Sentiment Breakdown"
        )
        st.plotly_chart(fig1)

        # Toxicity Bar Chart
        st.subheader("⚠️ Toxic vs Safe Posts")
        fig2 = px.bar(
            x=list(toxic_counts.keys()),
            y=list(toxic_counts.values()),
            color=list(toxic_counts.keys()),
            color_discrete_sequence=["#FF6F61", "#66BB6A"],
            text=list(toxic_counts.values())
        )
        fig2.update_traces(marker=dict(line=dict(color="black", width=1)))
        st.plotly_chart(fig2)

# -------------------------------
# Tab 2 - History
# -------------------------------
with tab2:
    st.subheader("📜 Full Analysis History")
    history_df = load_history()

    if not history_df.empty:
        st.dataframe(history_df)

        # Download full history
        st.download_button(
            label="⬇️ Download Full History as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="cyberbro_full_history.csv",
            mime="text/csv",
        )
    else:
        st.info("ℹ️ No history found yet. Run an analysis to start building history.")
