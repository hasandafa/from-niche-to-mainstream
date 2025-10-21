"""
Configuration file for anime sentiment analysis project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Create directories if not exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Reddit API Credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "anime-sentiment-analyzer/1.0")

# Scraping Configuration
TARGET_SUBREDDITS = [
    "movies",
    "television", 
    "entertainment",
    "anime",
    "TrueFilm",
    "boxoffice"
]

# Anime-related keywords untuk filtering
ANIME_KEYWORDS = [
    "anime", "japanese animation", "manga",
    # Popular titles - tambahkan sesuai kebutuhan
    "attack on titan", "demon slayer", "your name",
    "spirited away", "naruto", "one piece", "ghibli",
    "makoto shinkai", "hayao miyazaki"
]

# Time Range
START_YEAR = 2010
END_YEAR = 2025

# Scraping Settings
MAX_POSTS_PER_QUERY = 1000  # PRAW limitation
COMMENTS_PER_POST = 100  # Max comments to fetch per post
RATE_LIMIT_SLEEP = 2  # seconds between requests

# Data Storage
METADATA_FILE = RAW_DATA_DIR / "scraping_metadata.json"
CHECKPOINT_FILE = RAW_DATA_DIR / "scraping_checkpoint.json"

# Preprocessing Settings
MIN_COMMENT_LENGTH = 10  # Minimum character length
REMOVE_DELETED = True
REMOVE_AUTOMODERATOR = True

# Sentiment Analysis
SENTIMENT_MODELS = {
    "vader": "vader",  # Rule-based
    "textblob": "textblob",  # Rule-based
    "bert": "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Transformer
}

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "scraper.log"