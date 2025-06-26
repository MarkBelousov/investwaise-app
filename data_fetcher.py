import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataFetcher:
    """Class to handle fetching financial data from various sources"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
    
    @st.cache_data(ttl=300)
    def get_stock_data(_self, symbol, period="1y", interval="1d"):
        """
        Fetch stock data using yfinance
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                st.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            return data
        
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=600)
    def get_multiple_stocks(_self, symbols, period="1y"):
        """
        Fetch data for multiple stocks
        
        Args:
            symbols (list): List of stock symbols
            period (str): Data period
        
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}
        
        for symbol in symbols:
            data = _self.get_stock_data(symbol, period)
            if not data.empty:
                stock_data[symbol] = data
        
        return stock_data
    
    @st.cache_data(ttl=1800)
    def get_stock_info(_self, symbol):
        """
        Get detailed stock information
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        
        except Exception as e:
            st.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    
    @st.cache_data(ttl=3600)
    def get_popular_stocks(_self):
        """
        Get list of popular stocks for recommendations
        
        Returns:
            list: List of popular stock symbols
        """
        # Popular stocks across different sectors
        popular_stocks = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO'],
            'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'ETFs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND']
        }
        
        return popular_stocks
    
    def validate_stock_symbol(self, symbol):
        """
        Validate if a stock symbol exists and has data
        
        Args:
            symbol (str): Stock symbol to validate
        
        Returns:
            bool: True if symbol is valid and has data
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            return not hist.empty
        except:
            return False
    
    def search_stock_by_name(self, query):
        """
        Search for stocks by company name or symbol with autocorrect
        
        Args:
            query (str): Search query (company name or symbol)
        
        Returns:
            dict: Dictionary with 'exact_matches', 'suggestions', and 'corrected'
        """
        # Comprehensive stock database with company names
        stock_database = {
            # Technology
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            'ORCL': 'Oracle Corporation',
            'CRM': 'Salesforce Inc.',
            'ADBE': 'Adobe Inc.',
            'PYPL': 'PayPal Holdings Inc.',
            'SQ': 'Block Inc.',
            'ROKU': 'Roku Inc.',
            'ZM': 'Zoom Video Communications',
            'SPOT': 'Spotify Technology',
            'UBER': 'Uber Technologies',
            'SHOP': 'Shopify Inc.',
            'PLTR': 'Palantir Technologies',
            'RBLX': 'Roblox Corporation',
            'COIN': 'Coinbase Global',
            
            # Finance
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corp.',
            'WFC': 'Wells Fargo & Company',
            'GS': 'Goldman Sachs Group',
            'BRK-B': 'Berkshire Hathaway',
            'V': 'Visa Inc.',
            'MA': 'Mastercard Inc.',
            'C': 'Citigroup Inc.',
            'SCHW': 'Charles Schwab Corp.',
            'BLK': 'BlackRock Inc.',
            
            # Healthcare
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer Inc.',
            'UNH': 'UnitedHealth Group',
            'ABBV': 'AbbVie Inc.',
            'MRK': 'Merck & Co.',
            'TMO': 'Thermo Fisher Scientific',
            'DHR': 'Danaher Corporation',
            'AMGN': 'Amgen Inc.',
            'MRNA': 'Moderna Inc.',
            'BNTX': 'BioNTech SE',
            'GILD': 'Gilead Sciences',
            
            # Consumer
            'PG': 'Procter & Gamble',
            'KO': 'Coca-Cola Company',
            'WMT': 'Walmart Inc.',
            'HD': 'Home Depot Inc.',
            'MCD': 'McDonald\'s Corporation',
            'COST': 'Costco Wholesale',
            'NKE': 'Nike Inc.',
            'SBUX': 'Starbucks Corporation',
            'LOW': 'Lowe\'s Companies',
            
            # Communications
            'T': 'AT&T Inc.',
            'VZ': 'Verizon Communications',
            'CMCSA': 'Comcast Corporation',
            
            # Energy
            'XOM': 'Exxon Mobil Corporation',
            'CVX': 'Chevron Corporation',
            'COP': 'ConocoPhillips',
            'EOG': 'EOG Resources',
            'SLB': 'Schlumberger Limited',
            
            # Industrials
            'MMM': '3M Company',
            'CAT': 'Caterpillar Inc.',
            'GE': 'General Electric',
            'F': 'Ford Motor Company',
            'GM': 'General Motors',
            'DAL': 'Delta Air Lines',
            'UAL': 'United Airlines',
            'IBM': 'International Business Machines',
            
            # Utilities
            'NEE': 'NextEra Energy',
            'D': 'Dominion Energy',
            'SO': 'Southern Company',
            'DUK': 'Duke Energy',
            'AEP': 'American Electric Power',
            
            # REITs
            'AMT': 'American Tower Corp.',
            'PLD': 'Prologis Inc.',
            'CCI': 'Crown Castle Inc.',
            'EQIX': 'Equinix Inc.',
            'O': 'Realty Income Corp.',
            
            # ETFs
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'IWM': 'iShares Russell 2000 ETF',
            'VTI': 'Vanguard Total Stock Market',
            'VOO': 'Vanguard S&P 500 ETF',
            'BND': 'Vanguard Total Bond Market'
        }
        
        query_clean = query.upper().strip()
        
        # Exact symbol match
        if query_clean in stock_database:
            return {
                'exact_matches': [query_clean],
                'suggestions': [],
                'corrected': query_clean
            }
        
        # Search by company name or partial symbol
        exact_matches = []
        suggestions = []
        
        for symbol, company_name in stock_database.items():
            # Check if query matches symbol or is contained in company name
            if (query_clean in symbol or 
                query_clean.lower() in company_name.lower() or
                any(word in company_name.upper() for word in query_clean.split())):
                
                if query_clean == symbol:
                    exact_matches.append(symbol)
                else:
                    suggestions.append(symbol)
        
        # Fuzzy matching for typos
        if not exact_matches and not suggestions:
            for symbol in stock_database.keys():
                # Simple edit distance for autocorrect
                if self._similarity_score(query_clean, symbol) > 0.6:
                    suggestions.append(symbol)
        
        # Sort suggestions by relevance
        suggestions = suggestions[:10]
        
        return {
            'exact_matches': exact_matches,
            'suggestions': suggestions,
            'corrected': exact_matches[0] if exact_matches else (suggestions[0] if suggestions else None)
        }
    
    def _similarity_score(self, s1, s2):
        """Calculate similarity score between two strings"""
        if len(s1) == 0 or len(s2) == 0:
            return 0
        
        # Simple character overlap scoring
        s1_chars = set(s1.lower())
        s2_chars = set(s2.lower())
        
        intersection = len(s1_chars.intersection(s2_chars))
        union = len(s1_chars.union(s2_chars))
        
        return intersection / union if union > 0 else 0
    
    def get_market_indices(self):
        """
        Get major market indices data
        
        Returns:
            dict: Market indices data
        """
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        indices_data = {}
        for name, symbol in indices.items():
            data = self.get_stock_data(symbol, period="1y")
            if not data.empty:
                indices_data[name] = data
        
        return indices_data
    
    def calculate_returns(self, data, periods=[1, 7, 30, 90, 365]):
        """
        Calculate returns for different periods
        
        Args:
            data (pd.DataFrame): Stock data
            periods (list): List of periods in days
        
        Returns:
            dict: Returns for different periods
        """
        if data.empty or len(data) < max(periods):
            return {}
        
        returns = {}
        current_price = data['Close'].iloc[-1]
        
        for period in periods:
            if len(data) > period:
                past_price = data['Close'].iloc[-(period + 1)]
                period_return = (current_price - past_price) / past_price * 100
                returns[f'{period}d'] = period_return
        
        return returns
    
    def get_sector_performance(self):
        """
        Get sector ETF performance for diversification analysis
        
        Returns:
            dict: Sector performance data
        """
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU',
            'Communications': 'XLC'
        }
        
        sector_data = {}
        for sector, etf in sector_etfs.items():
            data = self.get_stock_data(etf, period="1y")
            if not data.empty:
                returns = self.calculate_returns(data)
                sector_data[sector] = {
                    'symbol': etf,
                    'data': data,
                    'returns': returns
                }
        
        return sector_data
