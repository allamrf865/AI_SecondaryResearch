import streamlit as st
import fitz  # PyMuPDF for PDF handling
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download necessary NLTK data once, if not already available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up page configurations and basic styles
st.set_page_config(page_title="Literature Review Quality Analyzer", layout="centered")

# Custom styling with HTML/CSS for lightweight animations
st.markdown("""
    <style>
    body { font-family: 'Comic Sans MS', sans-serif; }
    .header { font-size: 32px; color: #333; font-weight: bold; text-align: center; margin-bottom: 5px; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #c3cfe2; color: #333; text-align: center; padding: 10px; font-size: 16px; font-weight: bold; }
    .loader { border: 8px solid #f3f3f3; border-radius: 50%; border-top: 8px solid #3498db; width: 40px; height: 40px; animation: spin 2s linear infinite; margin: auto; display: block; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>📚 Literature Review Quality Analyzer 📚</div>", unsafe_allow_html=True)
st.write("Upload a PDF file, and let the magic of AI do the rest! ✨")

# Footer Branding
st.markdown("<div class='footer'>AI by Allam Rafi FKUI 2022</div>", unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

if uploaded_file is not None:
    st.markdown("<div class='loader'></div>", unsafe_allow_html=True)  # Display loader during processing
    st.write("Analyzing... Please wait a moment!")

    # Extract text from the uploaded PDF
    def extract_text_from_pdf(file):
        text = ""
        file_bytes = file.read()  # Read the file as bytes
        with fitz.open("pdf", file_bytes) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)
    
    # Text preprocessing: tokenization and stopword removal
    sentences = sent_tokenize(text)
    words = [word.lower() for word in word_tokenize(text) if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Readability Analysis
    def readability_score(text):
        fk_score = textstat.flesch_kincaid_grade(text)
        return 100 if fk_score <= 10 else 70 if fk_score <= 14 else 30, fk_score

    # Structure Completeness Analysis
    def structure_completeness(text):
        sections = ["introduction", "methods", "results", "discussion", "conclusion"]
        found_sections = {section: (section in text.lower()) for section in sections}
        return sum(found_sections.values()) / len(sections) * 100, found_sections

    # Cohesion Analysis using cosine similarity
    def cohesion_analysis(sentences):
        vectorizer = CountVectorizer().fit_transform(sentences)
        vectors = vectorizer.toarray()
        avg_cohesion = cosine_similarity(vectors).mean()
        return 100 if avg_cohesion >= 0.5 else 70 if avg_cohesion >= 0.3 else 30, avg_cohesion

    # Keyword Relevance with TF-IDF
    def keyword_relevance(text):
        tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        keywords = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(keywords)
        return sum(tfidf_matrix.toarray()[0]) / len(tfidf_matrix.toarray()[0]) * 100, keywords, wordcloud

    # Perform analysis on text
    readability_score, fk_score = readability_score(text)
    structure_score, found_sections = structure_completeness(text)
    cohesion_score, avg_cohesion = cohesion_analysis(sentences)
    keyword_score, keywords, wordcloud = keyword_relevance(text)

    # Calculate final quality score as an average of individual scores
    final_score = (readability_score + structure_score + cohesion_score + keyword_score) / 4

    # Determine quality level based on the new cutoff standards
    if final_score > 85:
        quality_level = "High"
    elif 60 <= final_score <= 85:
        quality_level = "Standard"
    else:
        quality_level = "Low"

    # Display Results
    st.subheader("Analysis Results:")
    st.markdown(f"**Overall Quality Level**: {quality_level} ({final_score:.2f}/100)")
    st.write("### Detailed Quality Metrics")
    st.write(f"**Readability Score**: {readability_score} (Flesch-Kincaid Grade: {fk_score})")
    st.write(f"**Structure Completeness**: {structure_score}% - {found_sections}")
    st.write(f"**Cohesion Score**: {cohesion_score} (Avg Similarity: {avg_cohesion:.2f})")
    st.write("### Keyword Relevance")
    st.pyplot(wordcloud.to_image())
