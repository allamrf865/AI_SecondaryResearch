import streamlit as st
import pdfplumber
import docx
import re
import textstat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from wordcloud import WordCloud

# Set up page configurations
st.set_page_config(page_title="Review Quality Analyzer", layout="centered")

# Custom styling with HTML/CSS
st.markdown("""
    <style>
    body { font-family: 'Comic Sans MS', sans-serif; }
    .header { font-size: 32px; color: #333; font-weight: bold; text-align: center; margin-bottom: 5px; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #c3cfe2; color: #333; text-align: center; padding: 10px; font-size: 16px; font-weight: bold; }
    .watermark { position: absolute; bottom: 20px; right: 20px; font-size: 10px; color: rgba(0, 0, 0, 0.1); font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Header and instructions
st.markdown("<h1 class='header'>Review Quality Analyzer</h1>", unsafe_allow_html=True)
st.write("Upload a PDF or Word file, select the review type, and get a detailed quality analysis! ✨")

# Footer Branding
st.markdown("<div class='footer'>AI by Allam Rafi FKUI 2022</div>", unsafe_allow_html=True)

# Watermark
st.markdown("<div class='watermark'>Watermark: Allam Rafi FKUI 2022</div>", unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF or Word document", type=["pdf", "docx"])

# Document type selection
review_type = st.selectbox("Select the review type:", 
                           ["Select Review Type", "Literature Review", "Systematic Review", "Meta Analysis", "Network Meta Analysis", "EBCP", "EBCR"])

# Enable analysis button only after file is uploaded and review type is selected
analyze_button = st.button("Analyze", disabled=(not uploaded_file or review_type == "Select Review Type"))

if analyze_button:
    st.write("Analyzing... Please wait a moment!")

    # Function to extract text from PDF
    def extract_text_from_pdf(file):
        text = ""
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading the PDF file: {e}")
        return text.strip()

    # Function to extract text from Word document
    def extract_text_from_docx(file):
        text = ""
        try:
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            st.error(f"Error reading the Word file: {e}")
        return text.strip()

    # Extract text from the uploaded file
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)

    # Basic checks for empty text
    if not text:
        st.error("No text extracted from the document.")
    else:
        # Process text (lowercasing, removing unwanted characters)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        words = [word.lower() for word in re.findall(r'\b\w+\b', text) if word.isalnum()]

        # Readability analysis
        def readability_score(text: str) -> Tuple[int, float]:
            fk_score = textstat.flesch_kincaid_grade(text)
            score = 100 if fk_score <= 10 else 70 if fk_score <= 14 else 30
            return score, fk_score

        # Complexity analysis
        def complexity_analysis(text: str) -> Tuple[int, float]:
            sentences = re.split(r'[.!?]', text)
            avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])
            word_complexity = np.mean([len(word) for word in re.findall(r'\b\w+\b', text)])  # Average word length
            return avg_sentence_length, word_complexity

        # Thematic Consistency (Detecting themes based on frequent words)
        def thematic_consistency(text: str) -> Tuple[int, str]:
            vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20)
            X = vectorizer.fit_transform([text])
            features = vectorizer.get_feature_names_out()
            freq_terms = sorted(zip(features, X.sum(axis=0).A1), key=lambda x: x[1], reverse=True)[:5]
            themes = [term[0] for term in freq_terms]
            consistency_score = len(set(themes))  # Number of distinct themes
            return consistency_score, ', '.join(themes)

        # Textual Diversity (unique words / total words)
        def textual_diversity(text: str) -> float:
            unique_words = set(re.findall(r'\b\w+\b', text.lower()))
            total_words = len(re.findall(r'\b\w+\b', text.lower()))
            diversity_score = len(unique_words) / total_words
            return diversity_score

        # Scientific Rigor (checking if certain scientific terms are present)
        def scientific_rigor(text: str) -> int:
            scientific_terms = ["hypothesis", "methodology", "control", "data", "statistical", "result", "significance"]
            score = sum(term in text.lower() for term in scientific_terms)
            return score

        # Bias Analysis: Publication and selection bias (very simplified)
        def bias_analysis(text: str) -> Tuple[str, str]:
            bias_score = "Low" if "systematic" in text.lower() else "High"
            publication_bias = "Possible" if "publication bias" in text.lower() else "Unlikely"
            return bias_score, publication_bias

        # Scope of Research: Checking if the scope is well-defined
        def scope_of_research(text: str) -> str:
            if "limitations" in text.lower():
                return "Wide Scope, but Some Gaps"
            return "Scope Seems Limited"

        # Structural Completeness based on the review type
        def structure_completeness(text: str, review_type: str) -> Tuple[int, dict]:
            sections = {
                "Literature Review": ["introduction", "literature review", "methodology", "discussion", "conclusion"],
                "Systematic Review": ["introduction", "methods", "results", "discussion", "conclusion", "references"],
                "Meta Analysis": ["introduction", "methods", "results", "discussion", "references"],
                "Network Meta Analysis": ["introduction", "methods", "results", "discussion", "references"],
                "EBCP": ["introduction", "methodology", "clinical recommendations", "discussion", "conclusion"],
                "EBCR": ["introduction", "methods", "results", "conclusion"]
            }
            found_sections = {section: (section in text.lower()) for section in sections.get(review_type, [])}
            completeness = sum(found_sections.values()) / len(sections.get(review_type, [])) * 100
            return completeness, found_sections

        # Apply all functions
        readability_result, fk_score = readability_score(text)
        avg_sentence_length, word_complexity = complexity_analysis(text)
        thematic_score, themes = thematic_consistency(text)
        diversity_score = textual_diversity(text)
        scientific_score = scientific_rigor(text)
        bias_score, publication_bias = bias_analysis(text)
        scope_result = scope_of_research(text)
        structure_result, found_sections = structure_completeness(text, review_type)

        # Calculating final quality score based on different factors
        final_score = (readability_result + thematic_score * 10 + diversity_score * 100 + scientific_score * 10 + structure_result) / 5

        # Quality level based on score
        quality_level = "Q1" if final_score > 85 else "Q2" if final_score > 70 else "Q3" if final_score > 50 else "Q4"
        sinta_level = "Sinta 1" if final_score > 85 else "Sinta 2" if final_score > 70 else "Sinta 3" if final_score > 50 else "Sinta 4"

        # Display results
        st.subheader("Analysis Results:")
        st.write(f"**Readability Score (FKG)**: {fk_score} (Grade: {readability_result})")
        st.write(f"**Thematic Consistency Score**: {thematic_score} (Themes: {themes})")
        st.write(f"**Textual Diversity**: {diversity_score:.2f}")
        st.write(f"**Scientific Rigor**: {scientific_score}")
        st.write(f"**Bias in Selection**: {bias_score}")
        st.write(f"**Publication Bias**: {publication_bias}")
        st.write(f"**Scope of Research**: {scope_result}")
        st.write(f"**Structural Completeness**: {structure_result}% - {found_sections}")
        st.write(f"**Final Quality Score**: {final_score:.2f}")
        st.write(f"**Journal Recommendation**: {quality_level} ({sinta_level})")

        # Visualizations

        # 1. Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wordcloud.to_array(), caption="Word Cloud", use_column_width=True)

        # 2. Sentence Length Distribution
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        plt.figure(figsize=(8, 5))
        plt.hist(sentence_lengths, bins=20, color='lightblue', edgecolor='black')
        plt.title("Sentence Length Distribution")
        plt.xlabel("Sentence Length (Words)")
        plt.ylabel("Frequency")
        st.pyplot()

        # 3. Complexity and Diversity Graph
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Sentence Complexity
        ax[0].bar(["Avg Sentence Length", "Avg Word Length"], [avg_sentence_length, word_complexity], color='purple')
        ax[0].set_title("Text Complexity Analysis")
        ax[0].set_ylabel("Length / Complexity")

        # Textual Diversity
        ax[1].bar(["Diversity Score"], [diversity_score], color='green')
        ax[1].set_title("Textual Diversity")
        ax[1].set_ylabel("Score")

        st.pyplot()
