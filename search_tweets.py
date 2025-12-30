#!/usr/bin/env python3

import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import tweepy

# Load environment variables
load_dotenv()

# Twitter API credentials
api_key = os.getenv('API_KEY')
api_secret_key = os.getenv('API_SECRET_KEY')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

# Authenticate
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Calculate date from 3 years ago
three_years_ago = datetime.now() - timedelta(days=3*365)

print(f"Searching for tweets from @jreuben1 containing 'survey' since {three_years_ago.date()}")

results = []

try:
    # Search for tweets from jreuben1 containing "survey" (case insensitive)
    query = "from:jreuben1 survey"

    # Fetch tweets
    tweets = tweepy.Cursor(
        api.search_tweets,
        q=query,
        tweet_mode='extended',
        lang='en',
        count=100
    ).items(500)  # Fetch up to 500 tweets

    for tweet in tweets:
        # Check if tweet is within 3 years
        if tweet.created_at < three_years_ago:
            continue

        # Get tweet text
        text = tweet.full_text.replace('\n', ' ')

        # Extract URLs
        urls = []
        if hasattr(tweet, 'entities') and 'urls' in tweet.entities:
            urls = [url['expanded_url'] for url in tweet.entities['urls']]

        # Filter for paper links
        paper_links = [
            url for url in urls
            if 'arxiv.org' in url or 'huggingface.co' in url or 'paperswithcode.com' in url
        ]

        if paper_links:
            results.append({
                'text': text,
                'link': paper_links[0],
                'date': tweet.created_at.strftime('%Y-%m-%d')
            })

    print(f"Found {len(results)} tweets with paper links")

    # Write to markdown file
    with open('x_survey_search.md', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['text']} | {result['link']} | {result['date']}\n")

    # Save as JSON for analysis
    with open('x_survey_search.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Results saved to x_survey_search.md and x_survey_search.json")

except Exception as e:
    print(f"Error: {e}")
    print("Note: Twitter API v1.1 search might be limited. Consider using v2 API.")
