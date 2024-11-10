import streamlit as st
import fitz  # PyMuPDF for PDF handling
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Custom HTML and CSS for animations and styling
st.markdown("""
    <style>
    /* Background with subtle animations */
    body {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        animation: gradientAnimation 15s ease infinite;
        color: #333333;
        font-family: 'Comic Sans MS', sans-serif;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Header styling */
    .header {
        font-size: 32px;
        color: #333;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #c3cfe2;
        color: #333;
        text-align: center;
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
    }

    /* Loading animation */
    .loader {
        border: 8px solid #f3f3f3;
        border-radius: 50%;
        border-top: 8px solid #3498db;
        width: 40px;
        height: 40px;
        animation: spin 2s linear infinite;
        margin: auto;
        display: block;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Custom cursor */
    * { cursor: url('https://cdn-icons-png.flaticon.com/512/1686/1686176.png'), auto; }

    /* Character animation */
    .character {
        position: fixed;
        bottom: 10px;
        right: 10px;
        animation: float 4s ease-in-out infinite;
        width: 100px;
    }

    @keyframes float {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<div class='header'>📚 Literature Review Quality Analyzer 📚</div>", unsafe_allow_html=True)
st.write("Upload a PDF file, and let the magic of AI do the rest! ✨")

# Animated Character Sticker (Mascot)
st.markdown("""
    <img src="https://media.giphy.com/media/l0HUpt2s9Pclgt9Vm/giphy.gif" class="character" alt="Character Animation">
""", unsafe_allow_html=True)

# Footer Branding
st.markdown("<div class='footer'>AI by Allam Rafi FKUI 2022</div>", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

# Analyze the document if a file is uploaded
if uploaded_file is not None:
    # Show loading animation while processing
    st.markdown("<div class='loader'></div>", unsafe_allow_html=True)
    st.write("Analyzing... Please wait a moment!")

    # Extract text from uploaded PDF
    def extract_text_from_pdf(file):
        text = ""
        with fitz.open(stream=file, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    text = extract_text_from_pdf(uploaded_file)
    
    # Preprocess text
    sentences = sent_tokenize(text)
    words = [word.lower() for word in word_tokenize(text) if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Quality Analysis Functions
    def readability_score(text):
        fk_score = textstat.flesch_kincaid_grade(text)
        return 100 if fk_score <= 10 else 70 if fk_score <= 14 else 30, fk_score

    def structure_completeness(text):
        sections = ["introduction", "methods", "results", "discussion", "conclusion"]
        found_sections = {section: (section in text.lower()) for section in sections}
        return sum(found_sections.values()) / len(sections) * 100, found_sections

    def cohesion_analysis(sentences):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = CountVectorizer().fit_transform(sentences)
        vectors = vectorizer.toarray()
        avg_cohesion = cosine_similarity(vectors).mean()
        return 100 if avg_cohesion >= 0.5 else 70 if avg_cohesion >= 0.3 else 30, avg_cohesion

    def sentiment_consistency(text):
        sections = ["introduction", "methods", "results", "discussion", "conclusion"]
        section_sentiments = {section: TextBlob(text[text.lower().find(section):]).sentiment.polarity
                              for section in sections if section in text.lower()}
        avg_sentiment = sum(section_sentiments.values()) / len(section_sentiments)
        max_variance = max(abs(sent - avg_sentiment) for sent in section_sentiments.values())
        return 100 if max_variance <= 0.2 else 70 if max_variance <= 0.5 else 30, section_sentiments, max_variance

    def keyword_relevance(text):
        tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        keywords = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(keywords)
        return sum(tfidf_matrix.toarray()[0]) / len(tfidf_matrix.toarray()[0]) * 100, keywords, wordcloud

    def topic_depth(text):
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
        lda = LDA(n_components=3, random_state=42)
        lda.fit(tfidf_matrix)
        depth_score = lda.components_.shape[0] / 3 * 100
        topics = {i: [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
                  for i, topic in enumerate(lda.components_)}
        return depth_score, topics

    # Perform Analyses
    readability_score, fk_score = readability_score(text)
    structure_score, found_sections = structure_completeness(text)
    cohesion_score, avg_cohesion = cohesion_analysis(sentences)
    sentiment_score, section_sentiments, max_variance = sentiment_consistency(text)
    keyword_score, keywords, wordcloud = keyword_relevance(text)
    topic_score, topics = topic_depth(text)

    # Overall Quality Score
    final_score = 0.2 * readability_score + 0.2 * structure_score + 0.2 * cohesion_score + 0.15 * sentiment_score + 0.15 * keyword_score + 0.1 * topic_score
    quality_level = "🌟 High Quality 🌟" if final_score >= 85 else "Standard Quality" if final_score >= 65 else "⚠️ Low Quality ⚠️"

    # Display Results
    st.subheader("Analysis Results:")
    st.markdown(f"**Overall Quality Level**: {quality_level} ({final_score:.2f}/100)")
    st.write("### Detailed Quality Metrics")
    st.write(f"**Readability Score**: {readability_score} (Flesch-Kincaid Grade: {fk_score})")
    st.write(f"**Structure Completeness**: {structure_score}% - {found_sections}")
    st.write(f"**Cohesion Score**: {cohesion_score} (Avg Similarity: {avg_cohesion:.2f})")
    st.write(f"**Sentiment Consistency Score**: {sentiment_score} (Max Variance: {max_variance:.2f})")
    st.write("### Keyword Relevance")
    st.pyplot(wordcloud.to_image())
    st.write(f"**Topic Depth**: {topic_score}% - {topics}")
