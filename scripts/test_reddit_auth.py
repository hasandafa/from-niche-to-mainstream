"""
Test Reddit API Credentials
Run this to verify your .env file is configured correctly
"""

import praw
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

print("="*70)
print("REDDIT API CREDENTIALS TEST")
print("="*70)

# Check if credentials exist
print("\n[1/5] Checking if credentials are set...")
if not REDDIT_CLIENT_ID or REDDIT_CLIENT_ID == '':
    print("    [ERROR] REDDIT_CLIENT_ID is missing!")
    print("    Please add it to your .env file")
    sys.exit(1)
else:
    print(f"    [OK] REDDIT_CLIENT_ID length: {len(REDDIT_CLIENT_ID)} chars")
    print(f"         Value: {REDDIT_CLIENT_ID}")

if not REDDIT_CLIENT_SECRET or REDDIT_CLIENT_SECRET == '':
    print("    [ERROR] REDDIT_CLIENT_SECRET is missing!")
    print("    Please add it to your .env file")
    sys.exit(1)
else:
    print(f"    [OK] REDDIT_CLIENT_SECRET length: {len(REDDIT_CLIENT_SECRET)} chars")
    print(f"         Starts with: {REDDIT_CLIENT_SECRET[:10]}...")

print(f"    [OK] REDDIT_USER_AGENT: {REDDIT_USER_AGENT}")

# Try to authenticate
print("\n[2/5] Testing Reddit API connection...")
try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        check_for_async=False
    )
    print("    [OK] PRAW initialized successfully")
except Exception as e:
    print(f"    [ERROR] Failed to initialize PRAW: {e}")
    sys.exit(1)

# Test authentication with user info
print("\n[3/5] Testing authentication...")
try:
    # This will trigger actual authentication
    user = reddit.user.me()
    if user:
        print(f"    [OK] Authenticated as user: {user.name}")
    else:
        print("    [INFO] Running in read-only mode (no user authentication)")
except Exception as e:
    print(f"    [INFO] Read-only mode (this is OK for scraping)")
    print(f"           Error details: {e}")

# Try to access a subreddit WITHOUT fetching posts
print("\n[4/5] Testing subreddit access (basic)...")
try:
    subreddit = reddit.subreddit('movies')
    print(f"    [OK] Subreddit object created: r/{subreddit.display_name}")
except Exception as e:
    print(f"    [ERROR] Failed to create subreddit object: {e}")
    sys.exit(1)

# Now try to actually fetch data
print("\n[5/5] Testing data fetching...")
try:
    print("    Attempting to fetch 1 hot post from r/movies...")
    
    posts_fetched = 0
    for post in reddit.subreddit('movies').hot(limit=1):
        print(f"    [OK] Successfully fetched post!")
        print(f"         Title: '{post.title[:60]}...'")
        print(f"         Score: {post.score}")
        print(f"         ID: {post.id}")
        posts_fetched += 1
        break
    
    if posts_fetched == 0:
        print("    [WARNING] No posts were fetched (might be rate limited)")
    
except Exception as e:
    error_msg = str(e).lower()
    print(f"    [ERROR] Failed to fetch data: {e}")
    print("\n    TROUBLESHOOTING:")
    
    if '401' in error_msg or 'unauthorized' in error_msg:
        print("    - Your CLIENT_ID or CLIENT_SECRET is INCORRECT")
        print("    - Double-check your .env file")
        print(f"    - CLIENT_ID should be: 1W9yFk4E4tWuyL10uwCMHQ")
        print(f"    - Your current CLIENT_ID: {REDDIT_CLIENT_ID}")
        print("\n    STEPS TO FIX:")
        print("    1. Go to: https://www.reddit.com/prefs/apps")
        print("    2. Look for 'from-niche-to-mainstream' app")
        print("    3. Copy the string UNDER the app name (below the icon)")
        print("    4. That's your CLIENT_ID (should be ~22 characters)")
        print("    5. Update your .env file with the correct value")
    elif '403' in error_msg or 'forbidden' in error_msg:
        print("    - Your app doesn't have permission")
        print("    - Make sure you selected 'script' type when creating app")
    elif '429' in error_msg or 'rate' in error_msg:
        print("    - You're being rate limited")
        print("    - Wait a few minutes and try again")
    else:
        print("    - Unknown error, check your internet connection")
    
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("[SUCCESS] All tests passed! Your credentials are working correctly.")
print("="*70)
print("\nYou can now run: python scripts/scraper.py")
print("="*70)