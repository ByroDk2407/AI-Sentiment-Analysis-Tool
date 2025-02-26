import tweepy
import praw
from GoogleNews import GoogleNews
from datetime import datetime, timedelta
from typing import List, Dict
from config import Config
import os

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
                
                # Get posts from different categories
                for category in ['hot', 'new', 'top']:
                    if category == 'top':
                        submission_list = subreddit.top(time_filter='week', limit=1)#limit=Config.REDDIT_POST_LIMIT)
                    elif category == 'hot':
                        submission_list = subreddit.hot(limit=1)#limit=Config.REDDIT_POST_LIMIT)
                    else:
                        submission_list = subreddit.new(limit=1)#limit=Config.REDDIT_POST_LIMIT)
                    
                    for submission in submission_list:
                        posts.append({
                            'title': submission.title,
                            'content': submission.selftext,
                            'date': datetime.fromtimestamp(submission.created_utc).isoformat(),
                            'source': f"reddit/r/{subreddit_name}",
                            'url': f"https://reddit.com{submission.permalink}",
                            'score': submission.score,
                            'num_comments': submission.num_comments
                        })
            
            except Exception as e:
                print(f"Error scraping subreddit {subreddit_name}: {str(e)}")
                continue
        
        return posts


class GoogleNewsScraper:
    def __init__(self):
        self.googlenews = GoogleNews(period=Config.GOOGLE_NEWS_PERIOD)
        self.googlenews.set_lang('en')
        self.googlenews.set_encode('utf-8')

    def scrape_news(self) -> List[Dict]:
        """Scrape real estate related news from Google News."""
        articles = []
        
        try:
            self.googlenews.clear()
            self.googlenews.search(Config.GOOGLE_NEWS_QUERY)
            results = self.googlenews.results()
            
            for result in results:
                articles.append({
                    'title': result.get('title'),
                    'content': result.get('desc'),
                    'date': result.get('datetime'),
                    'source': result.get('site'),
                    'url': result.get('link')
                })
        
        except Exception as e:
            print(f"Error scraping Google News: {str(e)}")
        
        return articles


class SocialMediaAggregator:
    def __init__(self):
        #self.twitter_scraper = TwitterScraper()
        self.reddit_scraper = RedditScraper()
        self.google_scraper = GoogleNewsScraper()

    def gather_all_data(self) -> Dict[str, List[Dict]]:
        """Gather data from all social media sources."""
        return {
            #'twitter': self.twitter_scraper.scrape_tweets(),
            'reddit': self.reddit_scraper.scrape_subreddits(),
            'google_news': self.google_scraper.scrape_news()
        }


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