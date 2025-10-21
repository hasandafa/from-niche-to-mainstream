# From Niche to Mainstream: Anime Sentiment Evolution on Reddit

Tracking how Reddit's perception of Japanese anime evolved from "weird hobby" to box office dominator over the past 15 years. This NLP project scrapes historical Reddit data to analyze sentiment shifts, cultural turning points, and the mainstreaming of anime culture.

*spoiler: the weebs were right all along* 🎌

## 🎯 Project Overview

Remember when liking anime meant you were automatically labeled "weird" or "nerdy"? This project investigates how public sentiment toward Japanese anime has transformed over 10-15 years using Reddit as our data source.

**Key Questions:**
- How has the volume and sentiment of anime discussions changed over time?
- When did the cultural shift from "niche/otaku" to "mainstream" occur?
- What events (movie releases, streaming platforms, COVID-19) influenced these changes?
- How do general audiences vs. dedicated fans discuss anime differently?

## 🏗️ Project Structure

```
from-niche-to-mainstream/
├── scripts/                    # Console scripts for automation
│   ├── scraper.py             # Reddit data collection
│   ├── preprocessor.py        # Text cleaning & normalization
│   ├── config.py              # Project configuration
│   └── utils.py               # Helper functions
│
├── data/                       # Data storage (git-ignored)
│   ├── raw/                   # Raw scraped data
│   ├── processed/             # Cleaned & processed data
│   └── results/               # Analysis outputs
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_temporal_trends.ipynb
│   └── 04_visualization.ipynb
│
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Reddit API credentials ([Get them here](https://www.reddit.com/prefs/apps))
- Virtual environment (recommended)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/from-niche-to-mainstream.git
cd from-niche-to-mainstream
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
# Create .env file with your Reddit API credentials
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=anime-sentiment-analyzer/1.0
```

### Usage

**Scrape Reddit data:**
```bash
# Scrape specific subreddit
python scripts/scraper.py --subreddit movies --time-filter all

# Scrape all configured subreddits
python scripts/scraper.py --all

# Incremental update (only new posts)
python scripts/scraper.py --all --incremental
```

**Preprocess data:**
```bash
python scripts/preprocessor.py --input data/raw/ --output data/processed/
```

**Run analysis:**
```bash
jupyter notebook notebooks/
```

## 📊 Data Collection Strategy

### Target Subreddits
- General: r/movies, r/television, r/entertainment
- Film-focused: r/TrueFilm, r/boxoffice
- Dedicated: r/anime

### Timeline
- **2010-2015**: Pre-mainstream era
- **2015-2019**: Transition period (streaming boom)
- **2020-2025**: Mainstream acceptance

### Filtering Criteria
Posts/comments mentioning:
- Generic terms: "anime", "japanese animation", "manga"
- Popular titles: "Attack on Titan", "Your Name", "Demon Slayer"
- Studios/creators: "Studio Ghibli", "Makoto Shinkai"

## 🔬 Analysis Approach

### Sentiment Analysis Methods
1. **VADER** - Rule-based, good for social media
2. **TextBlob** - Simple polarity scoring
3. **BERT** - Transformer-based for nuanced understanding

### Key Metrics
- Sentiment distribution (positive/negative/neutral)
- Sentiment trends over time
- Volume of discussions
- Engagement metrics (upvotes, comments)
- Vocabulary evolution (word embeddings)

## 🎨 Expected Insights

- Visualization of sentiment evolution timeline
- Identification of cultural "turning points"
- Correlation analysis with major anime releases
- Comparison of sentiment across different communities
- Word clouds showing vocabulary shifts

## 📝 TODO

- [ ] Complete scraper implementation
- [ ] Build preprocessing pipeline
- [ ] Experiment with sentiment models
- [ ] Create temporal visualizations
- [ ] Document key findings
- [ ] Write final report

## 🤝 Contributing

This is currently a personal research project, but suggestions and feedback are welcome!

## 📄 License

MIT License - feel free to use this for your own analysis

## 🙏 Acknowledgments

- Reddit API via PRAW
- Pushshift for historical data access
- All the anime fans who've been right this whole time

---

*Last updated: October 2025*