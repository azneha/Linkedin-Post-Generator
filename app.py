import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Mini AI Co-Writer", page_icon="✍️")
st.title("✍️ Mini AI Co-Writer / LinkedIn Assistant")
st.write("Generate summaries, LinkedIn posts, and cover letters instantly!")

# -----------------------------
# Task Selection
# -----------------------------
task = st.selectbox(
    "Choose your task:",
    ["Summarize", "Paraphrase", "Write Cover Letter", "Write LinkedIn Post"]
)

prompt = st.text_area("✍️ Enter your text, prompt, or job info:")

# Optional cover letter inputs
job_title = st.text_input("🎯 Job Title (for cover letters)")
company = st.text_input("🏢 Company Name (for cover letters)")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    sentiment_pipe = pipeline("sentiment-analysis")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return generator, sentiment_pipe, embed_model

generator, sentiment_pipe, embed_model = load_models()

# -----------------------------
# Hashtag Pool
# -----------------------------
hashtags_pool = [
    "#AI", "#MachineLearning", "#DataScience", "#Python", "#NLP", "#DeepLearning",
    "#Innovation", "#Tech", "#Productivity", "#Coding", "#LinkedInTips", "#Career"
]
hashtags_embeddings = embed_model.encode(hashtags_pool, convert_to_tensor=True)

# -----------------------------
# Generate Button
# -----------------------------
if st.button("🚀 Generate"):
    if not prompt.strip():
        st.warning("Please enter some text first!")
    else:
        if task == "Summarize":
            input_text = f"Summarize this text in a clear, professional way:\n{prompt}"
        elif task == "Paraphrase":
            input_text = f"Rewrite the following text in a unique, fluent way:\n{prompt}"
        elif task == "Write Cover Letter":
            input_text = (
                f"Write a professional, concise cover letter for the position of {job_title} at {company}. "
                f"Base it on the following user information:\n{prompt}"
            )
        elif task == "Write LinkedIn Post":
            input_text = (
                f"Write an engaging and inspiring LinkedIn post with a conversational tone, "
                f"based on this information:\n{prompt}"
            )

        # Generate text
        response = generator(input_text, max_length=200, num_beams=4)
        generated_text = response[0]["generated_text"].strip()

        st.subheader("✨ Generated Text:")
        st.write(generated_text)

        # Sentiment
        sentiment = sentiment_pipe(generated_text)[0]
        st.subheader("💭 Sentiment Analysis:")
        st.write(f"{sentiment['label']} ({sentiment['score']:.2f})")

        # Hashtags
        text_embedding = embed_model.encode(generated_text, convert_to_tensor=True)
        similarities = util.cos_sim(text_embedding, hashtags_embeddings)
        top_indices = similarities[0].topk(5).indices.tolist()
        suggested_hashtags = [hashtags_pool[i] for i in top_indices]

        st.subheader("🏷️ Suggested Hashtags:")
        st.write(" ".join(suggested_hashtags))

        # LinkedIn Preview
        st.subheader("🔗 LinkedIn Preview:")
        st.write(f"{generated_text}\n\n{' '.join(suggested_hashtags)}")
