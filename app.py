import streamlit as st
import pdfplumber
import textstat
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

    # Extract text from the uploaded PDF using pdfplumber
    def extract_text_from_pdf(file):
        text = ""
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add non-empty pages
                        text += page_text + "\n"
        except Exception as e:
            st.error("Error reading the PDF file.")
            return None
        return text.strip()  # Remove any extra whitespace

    # Extract and clean text from PDF
    text = extract_text_from_pdf(uploaded_file)
    if not text:
        st.error("No text extracted from PDF.")
    else:
        # Simple sentence tokenization using regex
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        words = [word.lower() for word in re.findall(r'\b\w+\b', text) if word.isalnum()]

        # Readability Analysis with refined scoring
        def readability_score(text):
            fk_score = textstat.flesch_kincaid_grade(text)
            score = 100 if fk_score <= 10 else 70 if fk_score <= 14 else 30
            return score, fk_score

        # Structure Completeness Analysis
        def structure_completeness(text):
            sections = ["introduction", "methods", "results", "discussion", "conclusion"]
            found_sections = {section: (section in text.lower()) for section in sections}
            completeness_score = sum(found_sections.values()) / len(sections) * 100
            return completeness_score, found_sections

        # Cohesion Analysis using cosine similarity
        def cohesion_analysis(sentences):
            try:
                vectorizer = CountVectorizer().fit_transform(sentences)
                vectors = vectorizer.toarray()
                avg_cohesion = cosine_similarity(vectors).mean()
                score = 100 if avg_cohesion >= 0.5 else 70 if avg_cohesion >= 0.3 else 30
            except Exception:
                avg_cohesion = 0
                score = 0  # Default score if analysis fails
            return score, avg_cohesion

        # Keyword Relevance with TF-IDF and WordCloud generation
        def keyword_relevance(text):
            if not text.strip():  # Check if text is empty
                return 0, {}, None

            try:
                tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
                tfidf_matrix = tfidf_vectorizer.fit_transform([text])
                
                # Check if TF-IDF matrix is empty
                if tfidf_matrix.shape[1] == 0:
                    return 0, {}, None

                keywords = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
                
                # Generate WordCloud
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(keywords)
                
                keyword_score = sum(tfidf_matrix.toarray()[0]) / len(tfidf_matrix.toarray()[0]) * 100
            except Exception:
                keywords = {}
                keyword_score = 0
                wordcloud = None
            return keyword_score, keywords, wordcloud

        # Convert WordCloud to matplotlib figure
        def plot_wordcloud(wordcloud):
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            return fig

        # Perform analysis on text
        readability_score, fk_score = readability_score(text)
        structure_score, found_sections = structure_completeness(text)
        cohesion_score, avg_cohesion = cohesion_analysis(sentences)
        keyword_score, keywords, wordcloud = keyword_relevance(text)

        # Calculate final quality score as an average of individual scores
        final_score = (readability_score + structure_score + cohesion_score + keyword_score) / 4

        # Determine quality level based on the cutoff standards
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
        
        # Display the WordCloud if available
        if wordcloud:
            st.write("### Keyword Relevance")
            fig = plot_wordcloud(wordcloud)
            st.pyplot(fig)
        else:
            st.write("Keyword relevance analysis failed or insufficient data.")
