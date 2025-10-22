"""
Utility Functions for Anime Sentiment Analysis Project
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def load_json(filepath: Path) -> Dict:
    """Load JSON file safely"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return {}


def save_json(data: Dict, filepath: Path, indent: int = 2):
    """Save data to JSON file"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {str(e)}")


def load_all_data(data_dir: Path = Path("data/processed"), pattern: str = "*_processed.json") -> pd.DataFrame:
    """
    Load all JSON files matching pattern into single DataFrame
    
    Args:
        data_dir: Directory containing data files
        pattern: File pattern to match
        
    Returns:
        Combined DataFrame
    """
    all_submissions = []
    
    for file in data_dir.glob(pattern):
        logger.info(f"Loading {file.name}")
        data = load_json(file)
        submissions = data.get('submissions', [])
        
        # Add metadata
        for sub in submissions:
            sub['source_file'] = file.name
        
        all_submissions.extend(submissions)
    
    df = pd.DataFrame(all_submissions)
    logger.info(f"Loaded {len(df)} submissions from {len(list(data_dir.glob(pattern)))} files")
    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features to DataFrame"""
    if 'created_utc' in df.columns:
        df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
        df['year'] = df['created_datetime'].dt.year
        df['month'] = df['created_datetime'].dt.month
        df['quarter'] = df['created_datetime'].dt.quarter
        df['day_of_week'] = df['created_datetime'].dt.dayofweek
        df['hour'] = df['created_datetime'].dt.hour
        df['date'] = df['created_datetime'].dt.date
    
    return df


def calculate_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate engagement metrics for submissions"""
    if 'score' in df.columns and 'num_comments' in df.columns:
        df['engagement_score'] = df['score'] + (df['num_comments'] * 2)
        df['comment_ratio'] = df['num_comments'] / (df['score'] + 1)
        df['engagement_category'] = pd.cut(
            df['engagement_score'],
            bins=[0, 10, 50, 200, float('inf')],
            labels=['low', 'medium', 'high', 'viral']
        )
    
    return df


def aggregate_by_period(
    df: pd.DataFrame,
    period: str = 'year',
    metrics: List[str] = None
) -> pd.DataFrame:
    """Aggregate data by time period"""
    if metrics is None:
        metrics = ['score', 'num_comments', 'engagement_score']
    
    if period not in df.columns:
        df = add_temporal_features(df)
    
    agg_dict = {metric: ['count', 'sum', 'mean', 'median', 'std'] for metric in metrics if metric in df.columns}
    
    result = df.groupby(period).agg(agg_dict)
    result.columns = ['_'.join(col).strip() for col in result.columns.values]
    
    return result.reset_index()


def filter_by_keywords(
    df: pd.DataFrame,
    keywords: List[str],
    text_column: str = 'full_text',
    case_sensitive: bool = False
) -> pd.DataFrame:
    """Filter DataFrame by keywords in text column"""
    if text_column not in df.columns:
        logger.warning(f"Column {text_column} not found in DataFrame")
        return df
    
    pattern = '|'.join(keywords)
    
    if case_sensitive:
        mask = df[text_column].str.contains(pattern, na=False, regex=True)
    else:
        mask = df[text_column].str.contains(pattern, na=False, regex=True, case=False)
    
    filtered_df = df[mask].copy()
    logger.info(f"Filtered {len(filtered_df)} rows matching keywords from {len(df)} total")
    
    return filtered_df


def detect_trending_topics(
    df: pd.DataFrame,
    text_column: str = 'cleaned_text',
    top_n: int = 20
) -> pd.DataFrame:
    """Extract trending topics/terms from text data"""
    all_words = []
    for text in df[text_column].dropna():
        words = str(text).lower().split()
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)
    
    return pd.DataFrame(top_words, columns=['term', 'frequency'])


def split_by_sentiment_threshold(
    df: pd.DataFrame,
    sentiment_column: str = 'sentiment_score',
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into positive, neutral, and negative sentiment groups"""
    positive_df = df[df[sentiment_column] > positive_threshold].copy()
    negative_df = df[df[sentiment_column] < negative_threshold].copy()
    neutral_df = df[
        (df[sentiment_column] >= negative_threshold) & 
        (df[sentiment_column] <= positive_threshold)
    ].copy()
    
    logger.info(
        f"Split into: {len(positive_df)} positive, "
        f"{len(neutral_df)} neutral, {len(negative_df)} negative"
    )
    
    return positive_df, neutral_df, negative_df


def calculate_sentiment_trend(
    df: pd.DataFrame,
    sentiment_column: str = 'sentiment_score',
    period: str = 'year'
) -> pd.DataFrame:
    """Calculate sentiment trends over time"""
    df = add_temporal_features(df)
    
    trend = df.groupby(period).agg({
        sentiment_column: ['mean', 'std', 'count'],
        'score': 'mean',
        'num_comments': 'mean'
    }).reset_index()
    
    trend.columns = ['_'.join(col).strip('_') for col in trend.columns.values]
    
    # Calculate rolling average
    if len(trend) > 2:
        trend[f'{sentiment_column}_rolling'] = trend[f'{sentiment_column}_mean'].rolling(
            window=3, center=True
        ).mean()
    
    return trend


def export_to_csv(df: pd.DataFrame, filepath: Path, columns: Optional[List[str]] = None):
    """Export DataFrame to CSV file"""
    try:
        if columns:
            df = df[columns]
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Exported {len(df)} rows to {filepath}")
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")


def get_data_summary(df: pd.DataFrame) -> Dict:
    """Generate comprehensive data summary statistics"""
    summary = {
        'total_rows': len(df),
        'date_range': {
            'start': df['created_datetime'].min().isoformat() if 'created_datetime' in df.columns else None,
            'end': df['created_datetime'].max().isoformat() if 'created_datetime' in df.columns else None
        },
        'subreddits': df['subreddit'].value_counts().to_dict() if 'subreddit' in df.columns else {},
        'yearly_distribution': df['year'].value_counts().sort_index().to_dict() if 'year' in df.columns else {},
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    
    return summary


def find_cultural_moments(
    df: pd.DataFrame,
    window_days: int = 30,
    spike_threshold: float = 2.0
) -> pd.DataFrame:
    """Identify potential cultural moments based on activity spikes"""
    df = add_temporal_features(df)
    
    # Daily post counts
    daily_counts = df.groupby('date').size().reset_index(name='post_count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    # Calculate rolling average
    daily_counts['rolling_avg'] = daily_counts['post_count'].rolling(
        window=window_days, center=True
    ).mean()
    
    # Detect spikes
    daily_counts['is_spike'] = (
        daily_counts['post_count'] > 
        daily_counts['rolling_avg'] * spike_threshold
    )
    
    spikes = daily_counts[daily_counts['is_spike']].copy()
    
    logger.info(f"Found {len(spikes)} activity spikes")
    
    return spikes


def match_with_anime_releases(
    spike_dates: List[datetime],
    release_data: Dict[str, datetime],
    tolerance_days: int = 14
) -> List[Dict]:
    """Match activity spikes with known anime releases"""
    matches = []
    
    for spike_date in spike_dates:
        for title, release_date in release_data.items():
            days_diff = abs((spike_date - release_date).days)
            
            if days_diff <= tolerance_days:
                matches.append({
                    'spike_date': spike_date,
                    'anime_title': title,
                    'release_date': release_date,
                    'days_difference': days_diff
                })
    
    return matches


def calculate_growth_rate(
    df: pd.DataFrame,
    metric: str = 'post_count',
    period: str = 'year'
) -> pd.DataFrame:
    """Calculate year-over-year or period-over-period growth rates"""
    df = add_temporal_features(df)
    
    period_metrics = df.groupby(period).size().reset_index(name=metric)
    period_metrics['growth_rate'] = period_metrics[metric].pct_change() * 100
    period_metrics['cumulative_growth'] = (
        (period_metrics[metric] / period_metrics[metric].iloc[0] - 1) * 100
    )
    
    return period_metrics


def sample_balanced_dataset(
    df: pd.DataFrame,
    label_column: str,
    samples_per_class: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """Create balanced sample for training/testing"""
    balanced_samples = []
    
    for label in df[label_column].unique():
        class_df = df[df[label_column] == label]
        sample_size = min(len(class_df), samples_per_class)
        sampled = class_df.sample(n=sample_size, random_state=random_state)
        balanced_samples.append(sampled)
    
    result = pd.concat(balanced_samples, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Created balanced dataset with {len(result)} samples")
    
    return result


def validate_data_quality(df: pd.DataFrame) -> Dict:
    """Check data quality and return validation report"""
    report = {
        'total_rows': len(df),
        'duplicate_ids': df.duplicated(subset=['id']).sum() if 'id' in df.columns else 0,
        'missing_text': df['full_text'].isnull().sum() if 'full_text' in df.columns else 0,
        'empty_text': (df['full_text'].str.strip() == '').sum() if 'full_text' in df.columns else 0,
        'missing_dates': df['created_utc'].isnull().sum() if 'created_utc' in df.columns else 0,
        'future_dates': 0,
        'invalid_scores': 0
    }
    
    # Check for future dates
    if 'created_utc' in df.columns:
        current_timestamp = datetime.now().timestamp()
        report['future_dates'] = (df['created_utc'] > current_timestamp).sum()
    
    # Check for invalid scores
    if 'score' in df.columns:
        report['invalid_scores'] = (df['score'] < 0).sum()
    
    # Quality score (0-100)
    issues = sum([
        report['duplicate_ids'],
        report['missing_text'],
        report['empty_text'],
        report['missing_dates'],
        report['future_dates'],
        report['invalid_scores']
    ])
    
    report['quality_score'] = max(0, 100 - (issues / max(len(df), 1)) * 100)
    report['status'] = 'excellent' if report['quality_score'] > 95 else \
                      'good' if report['quality_score'] > 80 else \
                      'fair' if report['quality_score'] > 60 else 'poor'
    
    return report


# Major anime releases for correlation analysis
MAJOR_ANIME_RELEASES = {
    'Your Name': datetime(2016, 8, 26),
    'Demon Slayer: Mugen Train': datetime(2020, 10, 16),
    'Jujutsu Kaisen 0': datetime(2021, 12, 24),
    'Spy x Family': datetime(2022, 4, 9),
    'Attack on Titan Final Season': datetime(2020, 12, 7),
    'My Hero Academia Season 1': datetime(2016, 4, 3),
    'One Punch Man': datetime(2015, 10, 5),
    'Chainsaw Man': datetime(2022, 10, 11),
    'The Boy and the Heron': datetime(2023, 7, 14),
    'Suzume': datetime(2022, 11, 11)
}


def main():
    """Example usage of utility functions"""
    # Load all processed data
    df = load_all_data(Path("data/processed"))
    
    # Add features
    df = add_temporal_features(df)
    df = calculate_engagement_metrics(df)
    
    # Generate summary
    summary = get_data_summary(df)
    print("Data Summary:", json.dumps(summary, indent=2, default=str))
    
    # Validate quality
    validation = validate_data_quality(df)
    print("\nData Quality Report:", json.dumps(validation, indent=2))
    
    # Find cultural moments
    spikes = find_cultural_moments(df)
    print(f"\nFound {len(spikes)} activity spikes")
    
    # Calculate growth
    growth = calculate_growth_rate(df)
    print("\nGrowth Rate by Year:")
    print(growth)
    
    # Save results
    export_to_csv(df, Path("data/results/combined_with_features.csv"))


if __name__ == "__main__":
    main()