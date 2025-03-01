import tweepy
import praw
from GoogleNews import GoogleNews
from datetime import datetime, timedelta
from typing import List, Dict
from utils.config import Config
import os
import time
import logging
from newsapi import NewsApiClient
import json
import random

logger = logging.getLogger(__name__)

class TwitterScraper:
    def __init__(self):
        auth = tweepy.OAuth1UserHandler(
            consumer_key=Config.TWITTER_API_KEY,
            consumer_secret=Config.TWITTER_API_SECRET,
            access_token=Config.TWITTER_ACCESS_TOKEN,
            access_token_secret=Config.TWITTER_ACCESS_TOKEN_SECRET,
        )
        auth.set_access_token(
            Config.TWITTER_ACCESS_TOKEN,
            Config.TWITTER_ACCESS_TOKEN_SECRET
        )
        self.api = tweepy.API(auth, wait_on_rate_limit=True)

    def scrape_tweets(self) -> List[Dict]:
        """Scrape tweets related to Australian real estate."""
        tweets = []
        
        for keyword in Config.TWITTER_KEYWORDS:
            try:
                # Search tweets using Twitter API v1.1
                search_results = self.api.search_tweets(
                    q=f"{keyword} -filter:retweets",
                    lang="en",
                    count=1, #Config.TWITTER_MAX_RESULTS,
                    tweet_mode="extended"
                )
                
                for tweet in search_results:
                    tweets.append({
                        'content': tweet.full_text,
                        'date': tweet.created_at.isoformat(),
                        'source': 'twitter',
                        'url': f"https://twitter.com/twitter/status/{tweet.id}",
                        'metrics': {
                            'retweet_count': tweet.retweet_count,
                            'favorite_count': tweet.favorite_count,
                            'reply_count': getattr(tweet, 'reply_count', 0)
                        },
                        'keyword': keyword,
                        'user': tweet.user.screen_name
                    })
                
                print(f"Found {len(search_results)} tweets for keyword: {keyword}")
            
            except Exception as e:
                print(f"Error scraping Twitter for keyword {keyword}: {str(e)}")
                continue
        
        return tweets


class RedditScraper:
    def __init__(self):
        """Initialize Reddit API client."""
        self.reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent='RealEstateSentimentBot/1.0'
        )

    def scrape_subreddits(self) -> List[Dict]:
        """Scrape posts from real estate related subreddits."""
        posts = []
        
        for subreddit_name in Config.REDDIT_SUBREDDITS:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get posts from different time periods and sorting methods
                post_queries = [
                    subreddit.hot(limit=Config.REDDIT_POST_LIMIT),
                    subreddit.new(limit=Config.REDDIT_POST_LIMIT),
                    subreddit.top(time_filter='month', limit=Config.REDDIT_POST_LIMIT),
                    subreddit.top(time_filter='week', limit=Config.REDDIT_POST_LIMIT)
                ]
                
                for query in post_queries:
                    for post in query:
                        # Skip stickied posts
                        if post.stickied:
                            continue
                        
                        try:
                            # Convert UTC timestamp to datetime without timezone
                            post_date = datetime.fromtimestamp(post.created_utc)
                            
                            # Skip if post is too old
                            if post_date < datetime.now() - timedelta(days=730):  # 2 years
                                continue
                                
                            # Skip if post date is in future
                            if post_date > datetime.now():
                                continue
                            
                            posts.append({
                                'title': post.title,
                                'content': post.selftext,
                                'url': f"https://reddit.com{post.permalink}",
                                'date': post_date.isoformat(),
                                'subreddit': subreddit_name,
                                'source': 'reddit'
                            })
                            
                        except Exception as e:
                            logger.error(f"Error processing Reddit post date: {str(e)}")
                            continue
                
                logger.info(f"Successfully scraped {len(posts)} posts from r/{subreddit_name}")
                
            except Exception as e:
                logger.error(f"Error scraping subreddit {subreddit_name}: {str(e)}")
                continue
        
        return posts

   
class GoogleNewsScraper:
    def __init__(self):
        self.googlenews = GoogleNews()
        self.googlenews.set_lang('en')
        self.googlenews.set_encode('utf-8')
        self.time_periods = Config.GOOGLE_NEWS_PERIODS
        self.period_days = {
            '7d': 7,
            '30d': 30,
        }
        self.max_pages = Config.GOOGLE_NEWS_PAGES  # Number of pages to fetch

    def parse_relative_date(self, date_str: str, period_days: int) -> datetime:
        """Convert relative date string to datetime object."""
        try:
            # Handle relative dates like "1 day ago", "2 days ago", etc.
            if 'day' in date_str:
                days = int(date_str.split()[0])
                return datetime.now() - timedelta(days=days)
            
            # Handle relative dates like "1 hour ago", "2 hours ago", etc.
            elif 'hour' in date_str:
                hours = int(date_str.split()[0])
                return datetime.now() - timedelta(hours=hours)
            
            # Handle relative dates like "1 week ago", "2 weeks ago", etc.
            elif 'week' in date_str:
                weeks = int(date_str.split()[0])
                return datetime.now() - timedelta(weeks=weeks)
            
            # Handle relative dates like "1 month ago", "2 months ago", etc.
            elif 'month' in date_str:
                months = int(date_str.split()[0])
                return datetime.now() - timedelta(days=months * 30)
            
            else:
                # If can't parse, estimate based on the period
                logger.warning(f"Unhandled date format: {date_str}, estimating based on period")
                # Use a random point within the period to distribute articles
                random_days = random.randint(1, period_days)
                return datetime.now() - timedelta(days=random_days)
                
        except Exception as e:
            logger.error(f"Error parsing date '{date_str}': {str(e)}")
            return datetime.now() - timedelta(days=period_days // 2)

    def scrape_news(self) -> List[Dict]:
        """Scrape real estate related news from Google News."""
        articles = []
        
        try:
            # Iterate through time periods
            for period_key, period in self.time_periods.items():
                print(f"\nCollecting articles for period: {period_key} ({self.period_days[period_key]} days)")
                
                # Set the period for Google News
                self.googlenews.set_period(period)
                
                # Search with different queries
                for query in Config.GOOGLE_NEWS_QUERIES:
                    try:
                        # Clear previous results
                        self.googlenews.clear()
                        
                        print(f"Searching for: {query}")
                        self.googlenews.search(query)
                        
                        # Get multiple pages of results
                        for page in range(1, self.max_pages + 1):
                            try:
                                print(f"Fetching page {page}...")
                                self.googlenews.get_page(page)
                                results = self.googlenews.results()
                                
                                # Process results
                                for result in results:
                                    # Check if article already exists in our list
                                    if not any(a['url'] == result.get('link') for a in articles):
                                        # Parse the relative date
                                        date_str = result.get('date', '')
                                        article_date = self.parse_relative_date(date_str, self.period_days[period_key])
                                        
                                        articles.append({
                                            'title': result.get('title'),
                                            'content': result.get('desc'),
                                            'date': article_date,
                                            'source': result.get('site'),
                                            'url': result.get('link'),
                                            'period': period_key
                                        })
                                
                                print(f"Found {len(results)} articles on page {page}")
                                
                                # Add a small delay between pages
                                time.sleep(2)
                                
                            except Exception as e:
                                print(f"Error fetching page {page}: {str(e)}")
                                continue
                        
                        print(f"Total articles found for query '{query}' in period {period_key}: {len(articles)}")
                        
                    except Exception as e:
                        print(f"Error processing query '{query}' for period {period_key}: {str(e)}")
                        continue
            
            print(f"\nSuccessfully scraped {len(articles)} total news articles")
            
            # Sort articles by date
            articles.sort(key=lambda x: x['date'], reverse=True)
            
        except Exception as e:
            print(f"Error scraping Google News: {str(e)}")
        
        return articles


class NewsAPIScraper:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
        self.delay = 3  # Delay between requests in seconds

    def scrape_news(self) -> List[Dict]:
        """Scrape real estate related news from NewsAPI."""
        articles = []
        
        try:
            # Collect data for each time period
            for days in Config.NEWS_API_TIME_PERIODS:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                print(f"\nCollecting articles for the past {days} days")
                print(f"Date range: {start_date.date()} to {end_date.date()}")
                
                # Search with different queries
                for query in Config.NEWS_API_QUERIES:
                    try:
                        print(f"Searching NewsAPI for: {query}")
                        
                        response = self.newsapi.get_everything(
                            q=query,
                            language=Config.NEWS_API_LANGUAGE,
                            from_param=start_date.strftime('%Y-%m-%d'),
                            to=end_date.strftime('%Y-%m-%d'),
                            sort_by='relevancy'
                        )
                        #print(json.dumps(response, indent=2))
                        if response['status'] == 'ok':
                            for article in response['articles']:
                                #print(f"Article was created at: {article.get('publishedAt')}")
                                if not any(a['url'] == article['url'] for a in articles):
                                    articles.append({
                                        'title': article.get('title'),
                                        'content': article.get('description'),
                                        'date': article.get('publishedAt'),
                                        'source': article.get('source', {}).get('name'),
                                        'url': article.get('url'),
                                        'author': article.get('author')
                                    })
                            
                            print(f"Found {len(response['articles'])} articles for query: {query}")
                        else:
                            print(f"Error in NewsAPI response for query {query}: {response.get('message')}")
                        
                        # Add delay between queries
                        time.sleep(self.delay)
                        
                    except Exception as e:
                        print(f"Error processing NewsAPI query '{query}': {str(e)}")
                        continue
                
                print(f"Total articles for {days} day period: {len(articles)}")
            
            print(f"\nSuccessfully scraped {len(articles)} total articles from NewsAPI")
            
        except Exception as e:
            print(f"Error scraping NewsAPI: {str(e)}")
        # Print dates of all articles
        # print("\nDates of collected articles:")
        # for article in articles:
        #     print(f"Date: {article.get('date')}")
        return articles


class SocialMediaAggregator:
    def __init__(self):
        self.reddit_scraper = RedditScraper()
        self.google_scraper = GoogleNewsScraper()
        self.newsapi_scraper = NewsAPIScraper()

    def gather_all_data(self) -> Dict[str, List[Dict]]:
        """Gather data from all sources."""
        print("Gathering data from all sources...")

        # Get NewsAPI data
        newsapi_data = self.newsapi_scraper.scrape_news()
        #print(f"Successfully scraped {len(newsapi_data)} NewsAPI articles")

        # Get Google News data
        googleNews_data = self.google_scraper.scrape_news()
        print(f"Successfully scraped {len(googleNews_data)} Google News articles")

        # Get Reddit data
        reddit_data = self.reddit_scraper.scrape_subreddits()
        print(f"Successfully scraped {len(reddit_data)} Reddit posts")

        all_data = {
            'reddit': reddit_data,
            'google_news': googleNews_data,
            'newsapi': newsapi_data
        }
        # Print dates for articles from each source
        # for source, items in all_data.items():
        #     print(f"\nDates for {source} articles:")
        #     for item in items:
        #         print(f"Date: {item.get('date')}")

        total_items = sum(len(items) for items in all_data.values())
        
        print(f"Total items collected: {total_items}")
        
        return all_data


if __name__ == "__main__":
    print("Testing Social Media Scrapers...")
    
    # # Test Twitter Scraper
    # print("\n\n\n\n=== Testing Twitter Scraper ===")
    # try:
    #     twitter = TwitterScraper()
    #     tweets = twitter.scrape_tweets()
    #     print(f"Successfully scraped {len(tweets)} tweets")
    #     if tweets:
    #         print("Sample tweet:")
    #         sample_tweet = tweets[0]
    #         print(f"Content: {sample_tweet['content'][::100]}...")
    #         print(f"Date: {sample_tweet['date']}")
    #         print(f"Metrics: {sample_tweet['metrics']}")
    # except Exception as e:
    #     print(f"Error testing Twitter scraper: {str(e)}")

    #Test Reddit Scraper
    print("\n\n\n\n=== Testing Reddit Scraper ===")
    try:
        reddit = RedditScraper()
        posts = reddit.scrape_subreddits()
        print(f"Successfully scraped {len(posts)} Reddit posts")
        if posts:
            print("Sample post:")
            sample_post = posts[0]
            print(f"Title: {sample_post['title']}")
            print(f"Score: {sample_post['score']}")
            print(f"Content: {sample_post['content']}")
            print(f"Comments: {sample_post['num_comments']}")
            print(f"URL: {sample_post['url']}")
    except Exception as e:
        print(f"Error testing Reddit scraper: {str(e)}")

    # Test Google News Scraper
    print("\n\n\n\n=== Testing Google News Scraper ===")
    try:
        google = GoogleNewsScraper()
        articles = google.scrape_news()
        print(f"Successfully scraped {len(articles)} news articles")
        if articles:
            print("Sample article:")
            sample_article = articles[0]
            print(f"Title: {sample_article['title']}")
            print(f"Source: {sample_article['source']}")
            print(f"URL: {sample_article['url']}")
    except Exception as e:
        print(f"Error testing Google News scraper: {str(e)}")

    # Test NewsAPI Scraper
    print("\n\n\n\n=== Testing NewsAPI Scraper ===")
    try:
        newsapi = NewsAPIScraper()
        articles = newsapi.scrape_news()
        print(f"Successfully scraped {len(articles)} news articles from NewsAPI")
        if articles:
            print("Sample article:")
            sample_article = articles[0]
            print(f"Title: {sample_article['title']}")
            print(f"Source: {sample_article['source']}")
            print(f"URL: {sample_article['url']}")
    except Exception as e:
        print(f"Error testing NewsAPI scraper: {str(e)}")

    # Test Aggregator
    print("\n\n\n\n=== Testing Social Media Aggregator ===")
    try:
        aggregator = SocialMediaAggregator()
        all_data = aggregator.gather_all_data()
        print("\nData collected from all sources:")
        for source, data in all_data.items():
            print(f"{source}: {len(data)} items")
    except Exception as e:
        print(f"Error testing aggregator: {str(e)}") 