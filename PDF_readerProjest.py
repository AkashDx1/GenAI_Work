import streamlit as st
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK stopwords
nltk.download('stopwords')
default_stopwords = set(stopwords.words('english'))

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters, numbers, and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Function to remove stopwords
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in default_stopwords]
    return " ".join(filtered_words)

# Streamlit app
def main():
    st.title("PDF Text Extractor and WordCloud Generator")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        # Preprocess the text
        with st.spinner("Preprocessing text..."):
            processed_text = preprocess_text(text)

        # Remove stopwords
        with st.spinner("Removing stopwords..."):
            filtered_text = remove_stopwords(processed_text)

        # Display the preprocessed text (optional)
        if st.checkbox("Show extracted text"):
            st.subheader("Extracted Text")
            st.write(text)

        # Display the filtered text (optional)
        if st.checkbox("Show filtered text"):
            st.subheader("Filtered Text")
            st.write(filtered_text)

        # Generate WordCloud
        with st.spinner("Generating WordCloud..."):
            wordcloud = WordCloud(width=800, height=400, max_words=500, background_color='white').generate(filtered_text)

        # Display the WordCloud
        st.subheader("WordCloud")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
