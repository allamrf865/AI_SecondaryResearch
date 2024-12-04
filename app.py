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
        except Exception:
            st.error("Error reading the PDF file.")
        return text.strip()

    # Function to extract text from Word document
    def extract_text_from_docx(file):
        text = ""
        try:
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception:
            st.error("Error reading the Word file.")
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

       # Function to analyze evidence (references)
def evidence_analysis(text: str) -> Tuple[int, float]:
    # Menangkap referensi URL dan DOI
    url_doi_references = re.findall(r'\b(?:https?|www)\S+|doi:\S+', text)
    
    # Menangkap referensi berupa penulis dan tahun (Author, Year format)
    author_year_references = re.findall(r'[A-Za-z]+(?:\s[A-Za-z]+)*,\s?\d{4}', text)
    
    # Menangkap referensi dalam format [1], [2], [3], dll.
    number_references = re.findall(r'\[\d+\]', text)

    # Gabungkan semua jenis referensi menjadi satu set untuk mendapatkan referensi unik
    all_references = set(url_doi_references + author_year_references + number_references)

    # Menghitung jumlah referensi unik
    num_unique_references = len(all_references)
    
    # Skor keberagaman referensi
    diversity_score = min(100, num_unique_references * 10)  # Skor maksimal 100 berdasarkan referensi unik
    
    return num_unique_references, diversity_score

        # Methodology analysis: Check for methodological references
        def methodology_analysis(text: str) -> Tuple[int, str]:
            methodology_terms = ["methodology", "statistical method", "sampling", "effect size"]
            methodology_score = sum(term in text.lower() for term in methodology_terms)
            methodology_quality = "Good" if methodology_score >= 2 else "Average"
            return methodology_score, methodology_quality

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
            completeness_score = sum(found_sections.values()) / len(found_sections) * 100
            return completeness_score, found_sections

        # Cohesion analysis: Using Cosine Similarity
        def cohesion_analysis(sentences: List[str]) -> Tuple[int, float]:
            try:
                vectorizer = CountVectorizer().fit_transform(sentences)
                vectors = vectorizer.toarray()
                avg_cohesion = cosine_similarity(vectors).mean()
                score = 100 if avg_cohesion >= 0.5 else 70 if avg_cohesion >= 0.3 else 30
            except Exception:
                avg_cohesion = 0
                score = 0
            return score, avg_cohesion

        # Perform analysis on document
        readability_result, fk_score = readability_score(text)
        evidence_count, evidence_diversity = evidence_analysis(text)
        methodology_score, methodology_quality = methodology_analysis(text)
        bias_score, publication_bias = bias_analysis(text)
        scope_result = scope_of_research(text)
        structure_result, found_sections = structure_completeness(text, review_type)
        cohesion_result, avg_cohesion = cohesion_analysis(sentences)

        # Final score
        final_score = (readability_result + evidence_diversity + methodology_score + cohesion_result) / 4
        quality_level = "High" if final_score > 85 else "Standard" if final_score >= 70 else "Low"

        # Display results
        st.subheader("Analysis Results:")
        st.write(f"**Readability Score (FKG)**: {fk_score} (Grade: {readability_result})")
        st.write(f"**Number of References**: {evidence_count}")
        st.write(f"**Evidence Diversity Score**: {evidence_diversity}")
        st.write(f"**Methodology Quality**: {methodology_quality}")
        st.write(f"**Bias in Selection**: {bias_score}")
        st.write(f"**Publication Bias**: {publication_bias}")
        st.write(f"**Research Scope**: {scope_result}")
        st.write(f"**Structure Completeness**: {structure_result}% - {found_sections}")
        st.write(f"**Cohesion Score**: {avg_cohesion:.2f}")
        st.write(f"**Final Quality Level**: {quality_level}")
        st.write("### Recommendations:")
        st.write("- Ensure clear explanation of methodology and effect size.")
        st.write("- Include diverse sources and references.")

        # Visualize results
        st.write("### Score Overview")
        metrics = ["Readability", "Evidence Diversity", "Methodology Quality", "Cohesion"]
        scores = [readability_result, evidence_diversity, methodology_score, cohesion_result]
        
        plt.figure(figsize=(8, 5))
        plt.bar(metrics, scores, color=['blue', 'green', 'red', 'purple'])
        plt.title("Document Analysis Scores")
        plt.ylabel("Score")
        plt.ylim(0, 100)
        st.pyplot()
