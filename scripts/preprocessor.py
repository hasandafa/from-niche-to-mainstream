"""
Text Preprocessing Module for Sentiment Analysis
Handles cleaning, normalization, and feature extraction
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path
import logging

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles all text preprocessing for sentiment analysis"""
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize preprocessor
        
        Args:
            use_spacy: Whether to load spaCy model for advanced NLP
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model if requested
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        
        # Anime-specific terms to preserve
        self.anime_terms = {
            'anime', 'manga', 'otaku', 'weeb', 'weaboo', 'shonen', 'shojo',
            'seinen', 'josei', 'mecha', 'isekai', 'kawaii', 'senpai', 'sensei'
        }
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Setup directories
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text
            preserve_case: Whether to preserve original case
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Markdown links
        text = re.sub(r'u/\w+', '', text)  # User mentions
        text = re.sub(r'r/\w+', '', text)  # Subreddit mentions
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\']', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase unless specified
        if not preserve_case:
            text = text.lower()
        
        return text
    
    def remove_stopwords(self, text: str, preserve_anime_terms: bool = True) -> str:
        """
        Remove stopwords while preserving important terms
        
        Args:
            text: Input text
            preserve_anime_terms: Whether to keep anime-specific terminology
            
        Returns:
            Text with stopwords removed
        """
        tokens = word_tokenize(text)
        
        if preserve_anime_terms:
            filtered_tokens = [
                word for word in tokens 
                if word.lower() not in self.stop_words or word.lower() in self.anime_terms
            ]
        else:
            filtered_tokens = [
                word for word in tokens 
                if word.lower() not in self.stop_words
            ]
        
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text using NLTK
        
        Args:
            text: Input text
            
        Returns:
            Lemmatized text
        """
        tokens = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized)
    
    def spacy_process(self, text: str) -> Dict:
        """
        Advanced NLP processing with spaCy
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with processed features
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        
        return {
            'lemmatized': ' '.join([token.lemma_ for token in doc]),
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks]
        }
    
    def extract_features(self, text: str) -> Dict:
        """
        Extract text features for analysis
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text features
        """
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
    
    def preprocess_submission(
        self,
        submission: Dict,
        cleaning_level: str = 'moderate'
    ) -> Dict:
        """
        Preprocess a Reddit submission
        
        Args:
            submission: Submission dictionary
            cleaning_level: 'light', 'moderate', or 'aggressive'
            
        Returns:
            Processed submission dictionary
        """
        # Combine title and body
        full_text = f"{submission.get('title', '')} {submission.get('selftext', '')}"
        
        # Apply cleaning based on level
        if cleaning_level == 'light':
            cleaned = self.clean_text(full_text, preserve_case=True)
        elif cleaning_level == 'moderate':
            cleaned = self.clean_text(full_text)
            cleaned = self.remove_stopwords(cleaned)
        else:  # aggressive
            cleaned = self.clean_text(full_text)
            cleaned = self.remove_stopwords(cleaned)
            cleaned = self.lemmatize_text(cleaned)
        
        # Extract features
        features = self.extract_features(full_text)
        
        # Create processed submission
        processed = submission.copy()
        processed.update({
            'original_text': full_text,
            'cleaned_text': cleaned,
            'text_features': features
        })
        
        # Add spaCy features if available
        if self.nlp and cleaning_level in ['moderate', 'aggressive']:
            spacy_features = self.spacy_process(cleaned)
            processed['spacy_features'] = spacy_features
        
        return processed
    
    def process_dataset(
        self,
        input_file: Path,
        output_file: Path,
        cleaning_level: str = 'moderate'
    ):
        """
        Process entire dataset from JSON file
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            cleaning_level: Cleaning intensity
        """
        logger.info(f"Processing {input_file}")
        
        try:
            # Load data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process submissions
            processed_submissions = []
            submissions = data.get('submissions', [])
            
            for i, submission in enumerate(submissions):
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(submissions)} submissions")
                
                processed = self.preprocess_submission(submission, cleaning_level)
                processed_submissions.append(processed)
            
            # Update data
            data['submissions'] = processed_submissions
            data['preprocessing_level'] = cleaning_level
            
            # Save processed data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved processed data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise
    
    def process_all_raw_data(self, cleaning_level: str = 'moderate'):
        """
        Process all files in data/raw/ directory
        Skip empty files (< 1000 bytes)
        
        Args:
            cleaning_level: Cleaning intensity
        """
        raw_files = list(self.raw_dir.glob("*_data.json"))
        
        # Filter out small/empty files
        valid_files = [f for f in raw_files if f.stat().st_size > 1000]
        
        logger.info(f"Found {len(valid_files)} valid files to process (skipped {len(raw_files) - len(valid_files)} empty files)")
        
        for raw_file in valid_files:
            output_file = self.processed_dir / raw_file.name.replace('_data.json', '_processed.json')
            logger.info(f"Processing: {raw_file.name} ({raw_file.stat().st_size / 1024 / 1024:.2f} MB)")
            self.process_dataset(raw_file, output_file, cleaning_level)
    
    def create_combined_dataframe(self) -> pd.DataFrame:
        """
        Load and combine all processed files into pandas DataFrame
        
        Returns:
            Combined DataFrame
        """
        all_submissions = []
        processed_files = list(self.processed_dir.glob("*_processed.json"))
        
        for file in processed_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                submissions = data.get('submissions', [])
                all_submissions.extend(submissions)
        
        df = pd.DataFrame(all_submissions)
        
        # Convert timestamp to datetime
        if 'created_utc' in df.columns:
            df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
            df['year'] = df['created_datetime'].dt.year
            df['month'] = df['created_datetime'].dt.month
            df['quarter'] = df['created_datetime'].dt.quarter
        
        logger.info(f"Created DataFrame with {len(df)} submissions")
        
        return df


def main():
    """Example usage"""
    preprocessor = TextPreprocessor(use_spacy=True)
    
    # Process all raw data files
    preprocessor.process_all_raw_data(cleaning_level='moderate')
    
    # Create combined DataFrame
    df = preprocessor.create_combined_dataframe()
    
    # Save to CSV
    df.to_csv("data/processed/combined_processed.csv", index=False)
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
        
    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text
            preserve_case: Whether to preserve original case
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Markdown links
        text = re.sub(r'u/\w+', '', text)  # User mentions
        text = re.sub(r'r/\w+', '', text)  # Subreddit mentions
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\']', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase unless specified
        if not preserve_case:
            text = text.lower()
        
        return text
    
    def remove_stopwords(self, text: str, preserve_anime_terms: bool = True) -> str:
        """
        Remove stopwords while preserving important terms
        
        Args:
            text: Input text
            preserve_anime_terms: Whether to keep anime-specific terminology
            
        Returns:
            Text with stopwords removed
        """
        tokens = word_tokenize(text)
        
        if preserve_anime_terms:
            filtered_tokens = [
                word for word in tokens 
                if word.lower() not in self.stop_words or word.lower() in self.anime_terms
            ]
        else:
            filtered_tokens = [
                word for word in tokens 
                if word.lower() not in self.stop_words
            ]
        
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text using NLTK
        
        Args:
            text: Input text
            
        Returns:
            Lemmatized text
        """
        tokens = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized)
    
    def spacy_process(self, text: str) -> Dict:
        """
        Advanced NLP processing with spaCy
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with processed features
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        
        return {
            'lemmatized': ' '.join([token.lemma_ for token in doc]),
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks]
        }
    
    def extract_features(self, text: str) -> Dict:
        """
        Extract text features for analysis
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text features
        """
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
    
    def preprocess_submission(
        self,
        submission: Dict,
        cleaning_level: str = 'moderate'
    ) -> Dict:
        """
        Preprocess a Reddit submission
        
        Args:
            submission: Submission dictionary
            cleaning_level: 'light', 'moderate', or 'aggressive'
            
        Returns:
            Processed submission dictionary
        """
        # Combine title and body
        full_text = f"{submission.get('title', '')} {submission.get('selftext', '')}"
        
        # Apply cleaning based on level
        if cleaning_level == 'light':
            cleaned = self.clean_text(full_text, preserve_case=True)
        elif cleaning_level == 'moderate':
            cleaned = self.clean_text(full_text)
            cleaned = self.remove_stopwords(cleaned)
        else:  # aggressive
            cleaned = self.clean_text(full_text)
            cleaned = self.remove_stopwords(cleaned)
            cleaned = self.lemmatize_text(cleaned)
        
        # Extract features
        features = self.extract_features(full_text)
        
        # Create processed submission
        processed = submission.copy()
        processed.update({
            'original_text': full_text,
            'cleaned_text': cleaned,
            'text_features': features
        })
        
        # Add spaCy features if available
        if self.nlp and cleaning_level in ['moderate', 'aggressive']:
            spacy_features = self.spacy_process(cleaned)
            processed['spacy_features'] = spacy_features
        
        return processed
    
    def process_dataset(
        self,
        input_file: Path,
        output_file: Path,
        cleaning_level: str = 'moderate'
    ):
        """
        Process entire dataset from JSON file
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            cleaning_level: Cleaning intensity
        """
        logger.info(f"Processing {input_file}")
        
        try:
            # Load data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process submissions
            processed_submissions = []
            submissions = data.get('submissions', [])
            
            for i, submission in enumerate(submissions):
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(submissions)} submissions")
                
                processed = self.preprocess_submission(submission, cleaning_level)
                processed_submissions.append(processed)
            
            # Update data
            data['submissions'] = processed_submissions
            data['preprocessing_level'] = cleaning_level
            
            # Save processed data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved processed data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise
    
    def create_dataframe(self, json_files: List[Path]) -> pd.DataFrame:
        """
        Load and combine multiple JSON files into pandas DataFrame
        
        Args:
            json_files: List of JSON file paths
            
        Returns:
            Combined DataFrame
        """
        all_submissions = []
        
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                submissions = data.get('submissions', [])
                all_submissions.extend(submissions)
        
        df = pd.DataFrame(all_submissions)
        
        # Convert timestamp to datetime
        if 'created_utc' in df.columns:
            df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
            df['year'] = df['created_datetime'].dt.year
            df['month'] = df['created_datetime'].dt.month
            df['quarter'] = df['created_datetime'].dt.quarter
        
        return df


def main():
    """Example usage"""
    from config import DATA_DIR
    
    preprocessor = TextPreprocessor(use_spacy=True)
    
    # Process a single file
    input_file = Path(DATA_DIR) / "movies_2020_data.json"
    output_file = Path(DATA_DIR) / "movies_2020_processed.json"
    
    if input_file.exists():
        preprocessor.process_dataset(input_file, output_file, cleaning_level='moderate')
    else:
        logger.warning(f"File {input_file} not found")


if __name__ == "__main__":
    main()