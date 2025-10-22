"""
Reddit Data Scraper for Anime Sentiment Analysis
UPDATED: Works without Pushshift (uses PRAW only)
"""

import praw # type: ignore
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional
from config import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    SUBREDDITS, ANIME_KEYWORDS, START_DATE, END_DATE
)

# Setup logging (remove emoji for Windows compatibility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RedditScraper:
    """Handles Reddit data collection using PRAW only (Pushshift-free)"""
    
    def __init__(self):
        """Initialize Reddit API client"""
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        
        # Setup data directories
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("[OK] Reddit scraper initialized (PRAW-only mode)")
    
    def search_submissions(
        self,
        subreddit: str,
        keywords: List[str],
        time_filter: str = 'all',
        limit: int = 1000
    ) -> List[Dict]:
        """
        Search for submissions using PRAW search (no Pushshift needed)
        
        Args:
            subreddit: Subreddit name
            keywords: List of keywords to search for
            time_filter: 'all', 'year', 'month', 'week', 'day'
            limit: Maximum number of posts per keyword
            
        Returns:
            List of submission dictionaries
        """
        submissions = []
        subreddit_obj = self.reddit.subreddit(subreddit)
        
        for keyword in keywords:
            logger.info(f"Searching r/{subreddit} for '{keyword}' (time_filter={time_filter})")
            
            try:
                # Search using PRAW
                results = subreddit_obj.search(
                    keyword,
                    time_filter=time_filter,
                    limit=limit,
                    sort='relevance'
                )
                
                count = 0
                for post in results:
                    submission_data = {
                        'id': post.id,
                        'subreddit': subreddit,
                        'title': post.title,
                        'selftext': post.selftext,
                        'author': str(post.author) if post.author else '[deleted]',
                        'created_utc': post.created_utc,
                        'created_date': datetime.fromtimestamp(post.created_utc).isoformat(),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'url': post.url,
                        'permalink': f"https://reddit.com{post.permalink}",
                        'keyword': keyword,
                        'full_text': f"{post.title} {post.selftext}"
                    }
                    submissions.append(submission_data)
                    count += 1
                
                logger.info(f"[OK] Found {count} submissions for '{keyword}'")
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"[ERROR] Failed searching for '{keyword}' in r/{subreddit}: {str(e)}")
                continue
        
        return submissions
    
    def filter_by_date(self, submissions: List[Dict], start_year: int, end_year: int) -> List[Dict]:
        """
        Filter submissions by year range
        
        Args:
            submissions: List of submission dictionaries
            start_year: Start year
            end_year: End year
            
        Returns:
            Filtered submissions
        """
        filtered = []
        for sub in submissions:
            post_year = datetime.fromtimestamp(sub['created_utc']).year
            if start_year <= post_year <= end_year:
                filtered.append(sub)
        
        return filtered
    
    def get_comments(self, submission_id: str, limit: int = 100) -> List[Dict]:
        """
        Fetch top comments for a submission using PRAW
        
        Args:
            submission_id: Reddit submission ID
            limit: Maximum number of comments to retrieve
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body'):
                    comment_data = {
                        'id': comment.id,
                        'submission_id': submission_id,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'body': comment.body,
                        'created_utc': comment.created_utc,
                        'created_date': datetime.fromtimestamp(comment.created_utc).isoformat(),
                        'score': comment.score
                    }
                    comments.append(comment_data)
            
            time.sleep(2)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error fetching comments for submission {submission_id}: {str(e)}")
        
        return comments
    
    def scrape_subreddit(
        self,
        subreddit: str,
        keywords: List[str],
        time_filter: str = 'year',
        include_comments: bool = False
    ) -> Dict:
        """
        Scrape submissions from a subreddit
        
        Args:
            subreddit: Subreddit name
            keywords: List of keywords to search for
            time_filter: Time filter for search
            include_comments: Whether to fetch comments
            
        Returns:
            Dictionary with submissions and metadata
        """
        logger.info(f"[SCRAPING] r/{subreddit} with time_filter={time_filter}")
        
        submissions = self.search_submissions(
            subreddit=subreddit,
            keywords=keywords,
            time_filter=time_filter
        )
        
        # Remove duplicates
        unique_submissions = {}
        for sub in submissions:
            if sub['id'] not in unique_submissions:
                unique_submissions[sub['id']] = sub
        
        submissions = list(unique_submissions.values())
        
        result = {
            'subreddit': subreddit,
            'time_filter': time_filter,
            'scraped_date': datetime.now().isoformat(),
            'submissions': submissions,
            'total_submissions': len(submissions)
        }
        
        if include_comments and len(submissions) > 0:
            logger.info(f"Fetching comments for top {min(50, len(submissions))} posts...")
            all_comments = []
            for sub in submissions[:50]:  # Limit to first 50 to avoid rate limits
                comments = self.get_comments(sub['id'])
                all_comments.extend(comments)
            
            result['comments'] = all_comments
            result['total_comments'] = len(all_comments)
        
        return result
    
    def save_data(self, data: Dict, filename: str):
        """
        Save scraped data to JSON file in data/raw/
        
        Args:
            data: Data dictionary to save
            filename: Output filename
        """
        filepath = self.raw_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"[SAVED] Data saved to {filepath}")
        except Exception as e:
            logger.error(f"[ERROR] Failed saving data to {filepath}: {str(e)}")
    
    def scrape_all_recent(
        self,
        subreddits: List[str] = None,
        keywords: List[str] = None,
        time_filter: str = 'year',
        include_comments: bool = False
    ):
        """
        Scrape recent data from all subreddits (recommended method)
        
        Args:
            subreddits: List of subreddits (uses config default if None)
            keywords: List of keywords (uses config default if None)
            time_filter: 'year', 'month', 'all' (year = past 12 months)
            include_comments: Whether to fetch comments
        """
        if subreddits is None:
            subreddits = SUBREDDITS
        if keywords is None:
            keywords = ANIME_KEYWORDS
        
        logger.info(f"[START] Starting scrape of {len(subreddits)} subreddits")
        logger.info(f"        Time filter: {time_filter}")
        logger.info(f"        Keywords: {len(keywords)}")
        
        for subreddit in subreddits:
            try:
                data = self.scrape_subreddit(
                    subreddit=subreddit,
                    keywords=keywords,
                    time_filter=time_filter,
                    include_comments=include_comments
                )
                
                # Save with timestamp
                timestamp = datetime.now().strftime('%Y%m%d')
                filename = f"{subreddit}_{time_filter}_{timestamp}_data.json"
                self.save_data(data, filename)
                
                logger.info(
                    f"[COMPLETE] {subreddit}: "
                    f"{data['total_submissions']} submissions"
                )
                
            except Exception as e:
                logger.error(f"[ERROR] Failed scraping r/{subreddit}: {str(e)}")
                continue
        
        logger.info("[SUCCESS] Scraping completed!")
    
    def scrape_by_time_periods(
        self,
        subreddits: List[str] = None,
        keywords: List[str] = None
    ):
        """
        Scrape data using multiple time filters to get historical data
        (PRAW limitation: can't get exact years, but can get 'all' time)
        
        Args:
            subreddits: List of subreddits
            keywords: List of keywords
        """
        if subreddits is None:
            subreddits = SUBREDDITS
        if keywords is None:
            keywords = ANIME_KEYWORDS
        
        # Use 'all' time filter to get maximum historical data
        logger.info("[START] Starting historical scrape (all time)")
        
        for subreddit in subreddits:
            try:
                data = self.scrape_subreddit(
                    subreddit=subreddit,
                    keywords=keywords,
                    time_filter='all',
                    include_comments=False
                )
                
                timestamp = datetime.now().strftime('%Y%m%d')
                filename = f"{subreddit}_all_time_{timestamp}_data.json"
                self.save_data(data, filename)
                
                logger.info(
                    f"[COMPLETE] {subreddit}: "
                    f"{data['total_submissions']} submissions (all time)"
                )
                
            except Exception as e:
                logger.error(f"[ERROR] Failed scraping r/{subreddit}: {str(e)}")
                continue
        
        logger.info("[SUCCESS] Historical scraping completed!")


def main():
    """Main execution function"""
    scraper = RedditScraper()
    
    print("\n" + "="*70)
    print("REDDIT SCRAPER - PRAW ONLY MODE")
    print("="*70)
    print("\nNOTE: Pushshift is unavailable, using PRAW search instead.")
    print("      This means we can get recent data, but not specific historical years.")
    print("\nChoose an option:")
    print("  1. Scrape recent data (past year) - RECOMMENDED")
    print("  2. Scrape all available data (all time)")
    print("  3. Quick test (single subreddit)")
    print("="*70)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\n[START] Scraping recent data (past year)...")
        scraper.scrape_all_recent(
            time_filter='year',
            include_comments=False
        )
    
    elif choice == '2':
        print("\n[START] Scraping all available data...")
        print("WARNING: This may take a long time!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            scraper.scrape_by_time_periods()
        else:
            print("[CANCELLED] Operation cancelled by user.")
    
    elif choice == '3':
        print("\n[TEST] Running quick test on r/movies...")
        data = scraper.scrape_subreddit(
            subreddit='movies',
            keywords=['anime', 'Studio Ghibli', 'Your Name'],
            time_filter='month',
            include_comments=False
        )
        scraper.save_data(data, 'test_movies_data.json')
        print(f"\n[OK] Test complete! Found {data['total_submissions']} submissions")
    
    else:
        print("[ERROR] Invalid choice!")
        return
    
    print("\n" + "="*70)
    print("[SUCCESS] SCRAPING COMPLETE!")
    print(f"[INFO] Data saved in: {scraper.raw_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
    
    def search_submissions(
        self,
        subreddit: str,
        keywords: List[str],
        time_filter: str = 'all',
        limit: int = 1000
    ) -> List[Dict]:
        """
        Search for submissions using PRAW search (no Pushshift needed)
        
        Args:
            subreddit: Subreddit name
            keywords: List of keywords to search for
            time_filter: 'all', 'year', 'month', 'week', 'day'
            limit: Maximum number of posts per keyword
            
        Returns:
            List of submission dictionaries
        """
        submissions = []
        subreddit_obj = self.reddit.subreddit(subreddit)
        
        for keyword in keywords:
            logger.info(f"Searching r/{subreddit} for '{keyword}' (time_filter={time_filter})")
            
            try:
                # Search using PRAW
                results = subreddit_obj.search(
                    keyword,
                    time_filter=time_filter,
                    limit=limit,
                    sort='relevance'
                )
                
                count = 0
                for post in results:
                    submission_data = {
                        'id': post.id,
                        'subreddit': subreddit,
                        'title': post.title,
                        'selftext': post.selftext,
                        'author': str(post.author) if post.author else '[deleted]',
                        'created_utc': post.created_utc,
                        'created_date': datetime.fromtimestamp(post.created_utc).isoformat(),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'url': post.url,
                        'permalink': f"https://reddit.com{post.permalink}",
                        'keyword': keyword,
                        'full_text': f"{post.title} {post.selftext}"
                    }
                    submissions.append(submission_data)
                    count += 1
                
                logger.info(f"✅ Found {count} submissions for '{keyword}'")
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Error searching for '{keyword}' in r/{subreddit}: {str(e)}")
                continue
        
        return submissions
    
    def filter_by_date(self, submissions: List[Dict], start_year: int, end_year: int) -> List[Dict]:
        """
        Filter submissions by year range
        
        Args:
            submissions: List of submission dictionaries
            start_year: Start year
            end_year: End year
            
        Returns:
            Filtered submissions
        """
        filtered = []
        for sub in submissions:
            post_year = datetime.fromtimestamp(sub['created_utc']).year
            if start_year <= post_year <= end_year:
                filtered.append(sub)
        
        return filtered
    
    def get_comments(self, submission_id: str, limit: int = 100) -> List[Dict]:
        """
        Fetch top comments for a submission using PRAW
        
        Args:
            submission_id: Reddit submission ID
            limit: Maximum number of comments to retrieve
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body'):
                    comment_data = {
                        'id': comment.id,
                        'submission_id': submission_id,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'body': comment.body,
                        'created_utc': comment.created_utc,
                        'created_date': datetime.fromtimestamp(comment.created_utc).isoformat(),
                        'score': comment.score
                    }
                    comments.append(comment_data)
            
            time.sleep(2)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error fetching comments for submission {submission_id}: {str(e)}")
        
        return comments
    
    def scrape_subreddit(
        self,
        subreddit: str,
        keywords: List[str],
        time_filter: str = 'year',
        include_comments: bool = False
    ) -> Dict:
        """
        Scrape submissions from a subreddit
        
        Args:
            subreddit: Subreddit name
            keywords: List of keywords to search for
            time_filter: Time filter for search
            include_comments: Whether to fetch comments
            
        Returns:
            Dictionary with submissions and metadata
        """
        logger.info(f"📊 Scraping r/{subreddit} with time_filter={time_filter}")
        
        submissions = self.search_submissions(
            subreddit=subreddit,
            keywords=keywords,
            time_filter=time_filter
        )
        
        # Remove duplicates
        unique_submissions = {}
        for sub in submissions:
            if sub['id'] not in unique_submissions:
                unique_submissions[sub['id']] = sub
        
        submissions = list(unique_submissions.values())
        
        result = {
            'subreddit': subreddit,
            'time_filter': time_filter,
            'scraped_date': datetime.now().isoformat(),
            'submissions': submissions,
            'total_submissions': len(submissions)
        }
        
        if include_comments and len(submissions) > 0:
            logger.info(f"Fetching comments for top {min(50, len(submissions))} posts...")
            all_comments = []
            for sub in submissions[:50]:  # Limit to first 50 to avoid rate limits
                comments = self.get_comments(sub['id'])
                all_comments.extend(comments)
            
            result['comments'] = all_comments
            result['total_comments'] = len(all_comments)
        
        return result
    
    def save_data(self, data: Dict, filename: str):
        """
        Save scraped data to JSON file in data/raw/
        
        Args:
            data: Data dictionary to save
            filename: Output filename
        """
        filepath = self.raw_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Data saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving data to {filepath}: {str(e)}")
    
    def scrape_all_recent(
        self,
        subreddits: List[str] = None,
        keywords: List[str] = None,
        time_filter: str = 'year',
        include_comments: bool = False
    ):
        """
        Scrape recent data from all subreddits (recommended method)
        
        Args:
            subreddits: List of subreddits (uses config default if None)
            keywords: List of keywords (uses config default if None)
            time_filter: 'year', 'month', 'all' (year = past 12 months)
            include_comments: Whether to fetch comments
        """
        if subreddits is None:
            subreddits = SUBREDDITS
        if keywords is None:
            keywords = ANIME_KEYWORDS
        
        logger.info(f"🚀 Starting scrape of {len(subreddits)} subreddits")
        logger.info(f"   Time filter: {time_filter}")
        logger.info(f"   Keywords: {len(keywords)}")
        
        for subreddit in subreddits:
            try:
                data = self.scrape_subreddit(
                    subreddit=subreddit,
                    keywords=keywords,
                    time_filter=time_filter,
                    include_comments=include_comments
                )
                
                # Save with timestamp
                timestamp = datetime.now().strftime('%Y%m%d')
                filename = f"{subreddit}_{time_filter}_{timestamp}_data.json"
                self.save_data(data, filename)
                
                logger.info(
                    f"✅ Completed {subreddit}: "
                    f"{data['total_submissions']} submissions"
                )
                
            except Exception as e:
                logger.error(f"❌ Error scraping r/{subreddit}: {str(e)}")
                continue
        
        logger.info("🎉 Scraping completed!")
    
    def scrape_by_time_periods(
        self,
        subreddits: List[str] = None,
        keywords: List[str] = None
    ):
        """
        Scrape data using multiple time filters to get historical data
        (PRAW limitation: can't get exact years, but can get 'all' time)
        
        Args:
            subreddits: List of subreddits
            keywords: List of keywords
        """
        if subreddits is None:
            subreddits = SUBREDDITS
        if keywords is None:
            keywords = ANIME_KEYWORDS
        
        # Use 'all' time filter to get maximum historical data
        logger.info("🚀 Starting historical scrape (all time)")
        
        for subreddit in subreddits:
            try:
                data = self.scrape_subreddit(
                    subreddit=subreddit,
                    keywords=keywords,
                    time_filter='all',
                    include_comments=False
                )
                
                timestamp = datetime.now().strftime('%Y%m%d')
                filename = f"{subreddit}_all_time_{timestamp}_data.json"
                self.save_data(data, filename)
                
                logger.info(
                    f"✅ Completed {subreddit}: "
                    f"{data['total_submissions']} submissions (all time)"
                )
                
            except Exception as e:
                logger.error(f"❌ Error scraping r/{subreddit}: {str(e)}")
                continue
        
        logger.info("🎉 Historical scraping completed!")


def main():
    """Main execution function"""
    scraper = RedditScraper()
    
    print("\n" + "="*70)
    print("REDDIT SCRAPER - PRAW ONLY MODE")
    print("="*70)
    print("\n📌 Note: Pushshift is unavailable, using PRAW search instead.")
    print("   This means we can get recent data, but not specific historical years.")
    print("\nChoose an option:")
    print("  1. Scrape recent data (past year) - RECOMMENDED")
    print("  2. Scrape all available data (all time)")
    print("  3. Quick test (single subreddit)")
    print("="*70)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\n🚀 Scraping recent data (past year)...")
        scraper.scrape_all_recent(
            time_filter='year',
            include_comments=False
        )
    
    elif choice == '2':
        print("\n🚀 Scraping all available data...")
        print("⚠️  Warning: This may take a long time!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            scraper.scrape_by_time_periods()
    
    elif choice == '3':
        print("\n🧪 Running quick test on r/movies...")
        data = scraper.scrape_subreddit(
            subreddit='movies',
            keywords=['anime', 'Studio Ghibli', 'Your Name'],
            time_filter='month',
            include_comments=False
        )
        scraper.save_data(data, 'test_movies_data.json')
        print(f"\n✅ Test complete! Found {data['total_submissions']} submissions")
    
    else:
        print("❌ Invalid choice!")
    
    print("\n" + "="*70)
    print("✅ SCRAPING COMPLETE!")
    print(f"📁 Data saved in: {scraper.raw_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
    
    def search_submissions(
        self,
        subreddit: str,
        keywords: List[str],
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for submissions using Pushshift API
        
        Args:
            subreddit: Subreddit name
            keywords: List of keywords to search for
            start_date: Start date for search
            end_date: End date for search
            limit: Maximum number of posts to retrieve
            
        Returns:
            List of submission dictionaries
        """
        submissions = []
        
        for keyword in keywords:
            logger.info(f"Searching r/{subreddit} for '{keyword}' from {start_date} to {end_date}")
            
            try:
                # Convert dates to Unix timestamps
                after = int(start_date.timestamp())
                before = int(end_date.timestamp())
                
                # Search using Pushshift
                results = self.pushshift.search_submissions(
                    subreddit=subreddit,
                    q=keyword,
                    after=after,
                    before=before,
                    limit=limit,
                    filter=['id', 'title', 'selftext', 'author', 'created_utc', 
                           'score', 'num_comments', 'url', 'permalink']
                )
                
                # Convert to list and process
                for post in results:
                    submission_data = {
                        'id': post['id'],
                        'subreddit': subreddit,
                        'title': post.get('title', ''),
                        'selftext': post.get('selftext', ''),
                        'author': str(post.get('author', '[deleted]')),
                        'created_utc': post['created_utc'],
                        'created_date': datetime.fromtimestamp(post['created_utc']).isoformat(),
                        'score': post.get('score', 0),
                        'num_comments': post.get('num_comments', 0),
                        'url': post.get('url', ''),
                        'permalink': f"https://reddit.com{post.get('permalink', '')}",
                        'keyword': keyword,
                        'full_text': f"{post.get('title', '')} {post.get('selftext', '')}"
                    }
                    submissions.append(submission_data)
                
                logger.info(f"Found {len(list(results))} submissions for '{keyword}'")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error searching for '{keyword}' in r/{subreddit}: {str(e)}")
                continue
        
        return submissions
    
    def get_comments(self, submission_id: str, limit: int = 100) -> List[Dict]:
        """
        Fetch top comments for a submission using PRAW
        
        Args:
            submission_id: Reddit submission ID
            limit: Maximum number of comments to retrieve
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body'):
                    comment_data = {
                        'id': comment.id,
                        'submission_id': submission_id,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'body': comment.body,
                        'created_utc': comment.created_utc,
                        'created_date': datetime.fromtimestamp(comment.created_utc).isoformat(),
                        'score': comment.score
                    }
                    comments.append(comment_data)
            
            time.sleep(2)  # Rate limiting for PRAW
            
        except Exception as e:
            logger.error(f"Error fetching comments for submission {submission_id}: {str(e)}")
        
        return comments
    
    def scrape_by_year(
        self,
        subreddit: str,
        year: int,
        keywords: List[str],
        include_comments: bool = False
    ) -> Dict:
        """
        Scrape submissions for a specific year
        
        Args:
            subreddit: Subreddit name
            year: Year to scrape
            keywords: List of keywords to search for
            include_comments: Whether to fetch comments for each submission
            
        Returns:
            Dictionary with submissions and optionally comments
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        
        logger.info(f"Scraping r/{subreddit} for year {year}")
        
        submissions = self.search_submissions(
            subreddit=subreddit,
            keywords=keywords,
            start_date=start_date,
            end_date=end_date
        )
        
        result = {
            'subreddit': subreddit,
            'year': year,
            'submissions': submissions,
            'total_submissions': len(submissions)
        }
        
        if include_comments:
            all_comments = []
            for sub in submissions[:50]:  # Limit to first 50 for comments
                comments = self.get_comments(sub['id'])
                all_comments.extend(comments)
            
            result['comments'] = all_comments
            result['total_comments'] = len(all_comments)
        
        return result
    
    def save_data(self, data: Dict, filename: str):
        """
        Save scraped data to JSON file in data/raw/
        
        Args:
            data: Data dictionary to save
            filename: Output filename
        """
        filepath = self.raw_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
    
    def scrape_all(
        self,
        start_year: int = 2010,
        end_year: int = 2025,
        include_comments: bool = False
    ):
        """
        Main scraping function - collects data for all subreddits and years
        
        Args:
            start_year: Starting year for scraping
            end_year: Ending year for scraping
            include_comments: Whether to fetch comments
        """
        logger.info(f"Starting full scrape from {start_year} to {end_year}")
        
        for subreddit in SUBREDDITS:
            for year in range(start_year, end_year + 1):
                try:
                    data = self.scrape_by_year(
                        subreddit=subreddit,
                        year=year,
                        keywords=ANIME_KEYWORDS,
                        include_comments=include_comments
                    )
                    
                    filename = f"{subreddit}_{year}_data.json"
                    self.save_data(data, filename)
                    
                    logger.info(
                        f"Completed {subreddit} {year}: "
                        f"{data['total_submissions']} submissions"
                    )
                    
                except Exception as e:
                    logger.error(f"Error scraping r/{subreddit} for {year}: {str(e)}")
                    continue
        
        logger.info("Full scrape completed!")


def main():
    """Main execution function"""
    scraper = RedditScraper()
    
    # Option 1: Full scrape (all years)
    scraper.scrape_all(start_year=2010, end_year=2025, include_comments=False)
    
    # Option 2: Single year test
    # data = scraper.scrape_by_year('movies', 2020, ANIME_KEYWORDS, include_comments=True)
    # scraper.save_data(data, 'test_data.json')


if __name__ == "__main__":
    main()
    
    def search_submissions(
        self,
        subreddit: str,
        keywords: List[str],
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for submissions using Pushshift API
        
        Args:
            subreddit: Subreddit name
            keywords: List of keywords to search for
            start_date: Start date for search
            end_date: End date for search
            limit: Maximum number of posts to retrieve
            
        Returns:
            List of submission dictionaries
        """
        submissions = []
        
        for keyword in keywords:
            logger.info(f"Searching r/{subreddit} for '{keyword}' from {start_date} to {end_date}")
            
            try:
                # Convert dates to Unix timestamps
                after = int(start_date.timestamp())
                before = int(end_date.timestamp())
                
                # Search using Pushshift
                results = self.pushshift.search_submissions(
                    subreddit=subreddit,
                    q=keyword,
                    after=after,
                    before=before,
                    limit=limit,
                    filter=['id', 'title', 'selftext', 'author', 'created_utc', 
                           'score', 'num_comments', 'url', 'permalink']
                )
                
                # Convert to list and process
                for post in results:
                    submission_data = {
                        'id': post['id'],
                        'subreddit': subreddit,
                        'title': post.get('title', ''),
                        'selftext': post.get('selftext', ''),
                        'author': str(post.get('author', '[deleted]')),
                        'created_utc': post['created_utc'],
                        'created_date': datetime.fromtimestamp(post['created_utc']).isoformat(),
                        'score': post.get('score', 0),
                        'num_comments': post.get('num_comments', 0),
                        'url': post.get('url', ''),
                        'permalink': f"https://reddit.com{post.get('permalink', '')}",
                        'keyword': keyword,
                        'full_text': f"{post.get('title', '')} {post.get('selftext', '')}"
                    }
                    submissions.append(submission_data)
                
                logger.info(f"Found {len(list(results))} submissions for '{keyword}'")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error searching for '{keyword}' in r/{subreddit}: {str(e)}")
                continue
        
        return submissions
    
    def get_comments(self, submission_id: str, limit: int = 100) -> List[Dict]:
        """
        Fetch top comments for a submission using PRAW
        
        Args:
            submission_id: Reddit submission ID
            limit: Maximum number of comments to retrieve
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)  # Remove "load more comments"
            
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body'):
                    comment_data = {
                        'id': comment.id,
                        'submission_id': submission_id,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'body': comment.body,
                        'created_utc': comment.created_utc,
                        'created_date': datetime.fromtimestamp(comment.created_utc).isoformat(),
                        'score': comment.score
                    }
                    comments.append(comment_data)
            
            time.sleep(2)  # Rate limiting for PRAW
            
        except Exception as e:
            logger.error(f"Error fetching comments for submission {submission_id}: {str(e)}")
        
        return comments
    
    def scrape_by_year(
        self,
        subreddit: str,
        year: int,
        keywords: List[str],
        include_comments: bool = False
    ) -> Dict:
        """
        Scrape submissions for a specific year
        
        Args:
            subreddit: Subreddit name
            year: Year to scrape
            keywords: List of keywords to search for
            include_comments: Whether to fetch comments for each submission
            
        Returns:
            Dictionary with submissions and optionally comments
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        
        logger.info(f"Scraping r/{subreddit} for year {year}")
        
        submissions = self.search_submissions(
            subreddit=subreddit,
            keywords=keywords,
            start_date=start_date,
            end_date=end_date
        )
        
        result = {
            'subreddit': subreddit,
            'year': year,
            'submissions': submissions,
            'total_submissions': len(submissions)
        }
        
        if include_comments:
            all_comments = []
            for sub in submissions[:50]:  # Limit to first 50 for comments due to rate limits
                comments = self.get_comments(sub['id'])
                all_comments.extend(comments)
            
            result['comments'] = all_comments
            result['total_comments'] = len(all_comments)
        
        return result
    
    def save_data(self, data: Dict, filename: str):
        """
        Save scraped data to JSON file
        
        Args:
            data: Data dictionary to save
            filename: Output filename
        """
        filepath = self.data_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
    
    def scrape_all(
        self,
        start_year: int = 2010,
        end_year: int = 2025,
        include_comments: bool = False
    ):
        """
        Main scraping function - collects data for all subreddits and years
        
        Args:
            start_year: Starting year for scraping
            end_year: Ending year for scraping
            include_comments: Whether to fetch comments
        """
        logger.info(f"Starting full scrape from {start_year} to {end_year}")
        
        for subreddit in SUBREDDITS:
            for year in range(start_year, end_year + 1):
                try:
                    data = self.scrape_by_year(
                        subreddit=subreddit,
                        year=year,
                        keywords=ANIME_KEYWORDS,
                        include_comments=include_comments
                    )
                    
                    filename = f"{subreddit}_{year}_data.json"
                    self.save_data(data, filename)
                    
                    logger.info(
                        f"Completed {subreddit} {year}: "
                        f"{data['total_submissions']} submissions"
                    )
                    
                except Exception as e:
                    logger.error(f"Error scraping r/{subreddit} for {year}: {str(e)}")
                    continue
        
        logger.info("Full scrape completed!")


def main():
    """Main execution function"""
    scraper = RedditScraper()
    
    # Option 1: Full scrape (all years)
    scraper.scrape_all(start_year=2010, end_year=2025, include_comments=False)
    
    # Option 2: Single year test
    # data = scraper.scrape_by_year('movies', 2020, ANIME_KEYWORDS, include_comments=True)
    # scraper.save_data(data, 'test_data.json')


if __name__ == "__main__":
    main()