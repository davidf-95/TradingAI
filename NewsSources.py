# NewsSources.py
import requests
import feedparser
from datetime import datetime
from typing import List, Dict
import streamlit as st
import pandas as pd

class NewsAnalyzer:
    @staticmethod
    def analyze_sentiment(text: str) -> float:
        text = text.lower()
        positive_words = {'bull', 'up', 'rise', 'growth', 'positive'}
        negative_words = {'bear', 'down', 'fall', 'drop', 'negative'}
        
        positive = sum(1 for word in positive_words if word in text)
        negative = sum(1 for word in negative_words if word in text)
        
        total = positive + negative
        return positive / total if total > 0 else 0.5

class BinanceNews:
    @classmethod
    @st.cache_data(ttl=3600)
    def fetch_news(cls, limit: int = 20) -> List[Dict]:
        try:
            url = "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query"
            params = {'catalogId': 48, 'pageNo': 1, 'pageSize': limit}
            response = requests.get(url, params=params, timeout=10)
            articles = response.json().get('data', {}).get('articles', [])
            
            return [{
                'source': 'Binance',
                'title': art['title'],
                'url': f"https://www.binance.com/en/support/announcement/{art['code']}",
                'date': datetime.fromtimestamp(art['releaseDate']/1000).strftime('%Y-%m-%d %H:%M'),
                'sentiment': 0.7
            } for art in articles]
        except Exception as e:
            st.error(f"Binance API Error: {str(e)}")
            return []

class CryptoPanicNews:
    @classmethod
    @st.cache_data(ttl=1800)
    def fetch_news(cls, api_key: str = "", limit: int = 20) -> List[Dict]:
        try:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {'auth_token': api_key, 'public': 'true', 'filter': 'hot'}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            news_items = []
            for post in data.get('results', [])[:limit]:
                positive = post.get('votes', {}).get('positive', 0)
                negative = post.get('votes', {}).get('negative', 1)
                sentiment = positive / max(1, positive + negative)
                
                news_items.append({
                    'source': 'CryptoPanic',
                    'title': post['title'],
                    'url': post['url'],
                    'date': post['created_at'][:19].replace('T', ' '),
                    'sentiment': sentiment
                })
            return news_items
        except Exception as e:
            st.error(f"CryptoPanic API Error: {str(e)}")
            return []

class CoinGeckoNews:
    @classmethod
    @st.cache_data(ttl=3600)
    def fetch_news(cls, limit: int = 15) -> List[Dict]:
        try:
            url = "https://api.coingecko.com/api/v3/news"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            return [{
                'source': 'CoinGecko',
                'title': item['title'],
                'url': item['url'],
                'date': datetime.fromtimestamp(item['updated_at']).strftime('%Y-%m-%d %H:%M'),
                'sentiment': NewsAnalyzer.analyze_sentiment(item['title'])
            } for item in data.get('data', [])[:limit]]
        except Exception as e:
            st.error(f"CoinGecko API Error: {str(e)}")
            return []

class NewsAggregator:
    SOURCES = {
        'Binance': BinanceNews,
        'CryptoPanic': CryptoPanicNews,
        'CoinGecko': CoinGeckoNews
    }

    @classmethod
    def get_all_news(cls, selected_sources: List[str], cryptopanic_key: str = "") -> pd.DataFrame:
        all_news = []
        
        for source in selected_sources:
            if source not in cls.SOURCES:
                continue
                
            try:
                if source == 'CryptoPanic':
                    news = cls.SOURCES[source].fetch_news(cryptopanic_key)
                else:
                    news = cls.SOURCES[source].fetch_news()
                
                all_news.extend(news)
            except Exception as e:
                st.error(f"Error fetching {source} news: {str(e)}")
                continue
                
        df = pd.DataFrame(all_news)
        if not df.empty:
            df = df.sort_values('date', ascending=False)
            df = df.drop_duplicates(subset=['title'])
        return df