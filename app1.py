import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import requests
from io import StringIO
import gdown

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def extract_file_id_from_url(url):
    """Extract file ID from Google Drive URL"""
    if 'drive.google.com' not in url:
        return None
    
    # Handle different Google Drive URL formats
    if '/file/d/' in url:
        file_id = url.split('/file/d/')[1].split('/')[0]
    elif 'id=' in url:
        file_id = url.split('id=')[1].split('&')[0]
    else:
        return None
    
    return file_id

def load_data_from_gdrive(url):
    """Load CSV data from Google Drive"""
    try:
        file_id = extract_file_id_from_url(url)
        if not file_id:
            raise ValueError("Invalid Google Drive URL")
        
        # Create download URL
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download file using gdown
        try:
            data = gdown.download(download_url, None, quiet=True)
            df = pd.read_csv(StringIO(data))
            return df, None
        except Exception as e:
            return None, f"Error downloading file: {str(e)}"
            
    except Exception as e:
        return None, f"Error processing Google Drive URL: {str(e)}"

class NewsDeduplicator:
    def __init__(self, similarity_threshold=0.6):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)

    def find_similar_articles(self, articles_df):
        # Preprocess all articles
        preprocessed_texts = [self.preprocess_text(text) for text in articles_df['content']]
        
        # Calculate TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find groups of similar articles
        groups = []
        used_indices = set()
        
        for i in range(len(articles_df)):
            if i in used_indices:
                continue
                
            group = [i]
            used_indices.add(i)
            
            for j in range(i + 1, len(articles_df)):
                if j not in used_indices and similarity_matrix[i][j] >= self.similarity_threshold:
                    group.append(j)
                    used_indices.add(j)
            
            if group:
                groups.append(group)
        
        return groups

    def aggregate_articles(self, articles_df, groups):
        aggregated_articles = []
        
        for group in groups:
            group_articles = articles_df.iloc[group]
            
            # Select the most recent article as the main article
            main_article = group_articles.iloc[group_articles['date'].argmax()]
            
            # Collect unique sources
            sources = list(group_articles['source'].unique())
            
            # Create aggregated article
            aggregated_article = {
                'title': main_article['title'],
                'content': main_article['content'],
                'main_source': main_article['source'],
                'date': main_article['date'],
                'additional_sources': [s for s in sources if s != main_article['source']],
                'article_count': len(group)
            }
            
            aggregated_articles.append(aggregated_article)
        
        return aggregated_articles

def process_dataframe(df, similarity_threshold):
    """Process the dataframe and return results"""
    with st.spinner("Processing articles..."):
        # Initialize deduplicator
        deduplicator = NewsDeduplicator(similarity_threshold=similarity_threshold)
        
        # Find similar articles
        groups = deduplicator.find_similar_articles(df)
        
        # Aggregate articles
        aggregated_articles = deduplicator.aggregate_articles(df, groups)
        
        return aggregated_articles

def main():
    st.title("News Deduplication System")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload CSV", "Google Drive Link"])
    
    with tab1:
        st.write("Upload your news articles dataset to find and aggregate similar articles.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                data_loaded = True
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                data_loaded = False
    
    with tab2:
        st.write("Enter Google Drive link to your CSV file")
        st.info("Make sure the file is publicly accessible or has view access")
        gdrive_url = st.text_input("Google Drive URL")
        
        if gdrive_url:
            df, error = load_data_from_gdrive(gdrive_url)
            if error:
                st.error(error)
                data_loaded = False
            else:
                data_loaded = True
                st.success("File loaded successfully from Google Drive!")
    
    if 'data_loaded' in locals() and data_loaded:
        # Check if required columns exist
        required_columns = ['title', 'content', 'source', 'date']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file must contain columns: title, content, source, and date")
            return
        
        st.subheader("Original Dataset")
        st.write(f"Total articles: {len(df)}")
        st.dataframe(df.head())

        # Similarity threshold slider
        similarity_threshold = st.slider(
            "Select similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1
        )

        if st.button("Process Articles"):
            # Process the dataframe
            aggregated_articles = process_dataframe(df, similarity_threshold)
            
            # Display results
            st.subheader("Aggregated Results")
            st.write(f"Number of unique topics: {len(aggregated_articles)}")
            
            for idx, article in enumerate(aggregated_articles, 1):
                with st.expander(f"Topic {idx}: {article['title']}"):
                    st.write(f"**Main Source:** {article['main_source']}")
                    st.write(f"**Date:** {article['date']}")
                    st.write(f"**Number of Related Articles:** {article['article_count']}")
                    if article['additional_sources']:
                        st.write("**Additional Sources:**")
                        for source in article['additional_sources']:
                            st.write(f"- {source}")
                    st.write("**Content:**")
                    st.write(article['content'])

if __name__ == "__main__":
    main()