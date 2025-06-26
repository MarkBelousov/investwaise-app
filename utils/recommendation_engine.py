import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    """Class for generating investment recommendations based on user profile and market analysis"""
    
    def __init__(self):
        self.risk_weights = {
            'Conservative': {'stocks': 0.3, 'bonds': 0.6, 'cash': 0.1},
            'Moderate': {'stocks': 0.6, 'bonds': 0.3, 'cash': 0.1},
            'Aggressive': {'stocks': 0.8, 'bonds': 0.15, 'cash': 0.05}
        }
        
        self.time_horizon_weights = {
            '6 months': {'growth': 0.2, 'dividend': 0.8},
            '1 year': {'growth': 0.4, 'dividend': 0.6},
            '2 years': {'growth': 0.6, 'dividend': 0.4},
            '5 years': {'growth': 0.7, 'dividend': 0.3},
            '10 years': {'growth': 0.8, 'dividend': 0.2},
            '20+ years': {'growth': 0.9, 'dividend': 0.1}
        }
    
    def generate_recommendations(self, user_inputs):
        """
        Generate comprehensive investment recommendations
        
        Args:
            user_inputs (dict): User investment profile
        
        Returns:
            dict: Investment recommendations
        """
        try:
            # Analyze user profile
            risk_profile = self.analyze_risk_profile(user_inputs)
            
            # Get stock universe
            stock_universe = self.get_stock_universe()
            
            # Score and rank stocks
            scored_stocks = self.score_stocks(stock_universe, user_inputs, risk_profile)
            
            # Generate portfolio allocation
            allocation = self.generate_allocation(user_inputs, risk_profile)
            
            # Create final recommendations
            recommendations = {
                'risk_profile': risk_profile,
                'allocations': allocation,
                'stocks': scored_stocks[:10],  # Top 10 recommendations
                'etfs': self.get_etf_recommendations(user_inputs),
                'analysis': self.generate_analysis_summary(user_inputs, scored_stocks)
            }
            
            return recommendations
        
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return {}
    
    def analyze_risk_profile(self, user_inputs):
        """
        Analyze user's risk profile
        
        Args:
            user_inputs (dict): User investment profile
        
        Returns:
            dict: Risk analysis
        """
        risk_tolerance = user_inputs['risk_tolerance']
        investment_horizon = user_inputs['investment_horizon']
        monthly_investment = user_inputs['monthly_investment']
        initial_investment = user_inputs['initial_investment']
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(user_inputs)
        
        # Time horizon factor
        horizon_years = self.extract_years_from_horizon(investment_horizon)
        
        return {
            'tolerance': risk_tolerance,
            'score': risk_score,
            'horizon_years': horizon_years,
            'total_investment_capacity': initial_investment + (monthly_investment * 12 * horizon_years),
            'investment_frequency': 'Monthly' if monthly_investment > 0 else 'Lump Sum'
        }
    
    def calculate_risk_score(self, user_inputs):
        """
        Calculate numerical risk score (0-100)
        
        Args:
            user_inputs (dict): User investment profile
        
        Returns:
            int: Risk score
        """
        base_scores = {'Conservative': 30, 'Moderate': 60, 'Aggressive': 85}
        score = base_scores[user_inputs['risk_tolerance']]
        
        # Adjust for time horizon
        horizon_years = self.extract_years_from_horizon(user_inputs['investment_horizon'])
        if horizon_years >= 10:
            score += 10
        elif horizon_years >= 5:
            score += 5
        elif horizon_years < 2:
            score -= 15
        
        # Adjust for investment amount (larger amounts can handle more risk)
        total_investment = user_inputs['initial_investment'] + (user_inputs['monthly_investment'] * 12)
        if total_investment > 50000:
            score += 5
        elif total_investment < 10000:
            score -= 5
        
        return max(0, min(100, score))
    
    def extract_years_from_horizon(self, horizon):
        """Extract years from investment horizon string"""
        if '6 months' in horizon:
            return 0.5
        elif '1 year' in horizon:
            return 1
        elif '2 years' in horizon:
            return 2
        elif '5 years' in horizon:
            return 5
        elif '10 years' in horizon:
            return 10
        elif '20+' in horizon:
            return 25
        return 5  # default
    
    def get_stock_universe(self):
        """
        Get universe of stocks to analyze
        
        Returns:
            list: List of stock symbols
        """
        # Only verified, actively traded large-cap stocks with real company data
        stocks = {
            'large_cap_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ORCL', 'CRM', 'ADBE'],
            'large_cap_finance': ['JPM', 'BAC', 'WFC', 'GS', 'BRK.B', 'V', 'MA', 'C'],
            'large_cap_healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'AMGN'],
            'large_cap_consumer': ['PG', 'KO', 'WMT', 'HD', 'MCD', 'COST', 'NKE', 'SBUX'],
            'growth_stocks': ['TSLA', 'NFLX', 'PYPL', 'ROKU', 'ZM', 'UBER', 'SPOT'],
            'dividend_stocks': ['T', 'VZ', 'XOM', 'CVX', 'IBM', 'MMM', 'CAT', 'KMI'],
            'emerging_sectors': ['PLTR', 'MRNA', 'BNTX'],
            'utilities': ['NEE', 'D', 'SO', 'DUK', 'AEP', 'EXC', 'XEL', 'PPL'],
            'reits': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'WELL', 'AVB'],
            'housing_related': ['HD', 'LOW', 'WHR', 'LEN', 'DHI', 'NVR', 'PHM'],
            'retirement_focused': ['BLK', 'TROW', 'BEN', 'AMG', 'IVZ', 'SCHW']
        }
        
        all_stocks = []
        for category, symbols in stocks.items():
            all_stocks.extend(symbols)
        
        # Return all stocks - validation will happen at runtime
        return list(set(all_stocks))  # Remove duplicates
    
    def get_goal_specific_stocks(self, goal):
        """
        Get stocks specifically suited for a particular investment goal
        
        Args:
            goal (str): Investment goal
        
        Returns:
            list: List of stock symbols suited for the goal
        """
        goal_specific_stocks = {
            'Retirement': ['JNJ', 'PG', 'KO', 'T', 'VZ', 'XOM', 'CVX', 'MMM', 'CAT', 'O', 'SCHW', 'BLK'],
            'Emergency Fund': ['JNJ', 'PG', 'KO', 'WMT', 'COST', 'NEE', 'D', 'SO', 'T', 'VZ'],
            'Home Purchase': ['HD', 'LOW', 'WHR', 'LEN', 'DHI', 'NVR', 'PHM', 'AMT', 'PLD', 'EQIX'],
            'Education': ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'PFE', 'UNH', 'BRK.B', 'V', 'MA'],
            'General Wealth Building': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'V', 'MA']
        }
        
        # Return goal-specific stocks - validation handled at app level
        return goal_specific_stocks.get(goal, [])
    
    def score_stocks(self, stocks, user_inputs, risk_profile):
        """
        Score stocks based on user profile and market analysis
        
        Args:
            stocks (list): List of stock symbols
            user_inputs (dict): User investment profile
            risk_profile (dict): Risk analysis
        
        Returns:
            list: Scored and ranked stocks
        """
        scored_stocks = []
        
        for symbol in stocks:
            try:
                score = self.calculate_stock_score(symbol, user_inputs, risk_profile)
                if score > 0:
                    scored_stocks.append({
                        'symbol': symbol,
                        'score': score,
                        'recommendation': self.get_recommendation_strength(score)
                    })
            except Exception as e:
                print(f"Error scoring {symbol}: {e}")
                continue
        
        # Sort by score
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        return scored_stocks
    
    def calculate_stock_score(self, symbol, user_inputs, risk_profile):
        """
        Calculate score for individual stock
        
        Args:
            symbol (str): Stock symbol
            user_inputs (dict): User investment profile
            risk_profile (dict): Risk analysis
        
        Returns:
            float: Stock score
        """
        try:
            # Get stock data and info
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            info = ticker.info
            
            if hist.empty or not info:
                return 0
            
            score = 50  # Base score
            
            # Performance scoring
            returns_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) * 100 if len(hist) > 21 else 0
            returns_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-63] - 1) * 100 if len(hist) > 63 else 0
            returns_1y = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100 if len(hist) > 0 else 0
            
            # Positive performance bonus
            if returns_1y > 10:
                score += 15
            elif returns_1y > 0:
                score += 5
            elif returns_1y < -20:
                score -= 15
            
            # Volatility adjustment based on risk tolerance
            volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
            if risk_profile['tolerance'] == 'Conservative':
                if volatility < 20:
                    score += 10
                elif volatility > 40:
                    score -= 20
            elif risk_profile['tolerance'] == 'Aggressive':
                if volatility > 30:
                    score += 5
            
            # Market cap considerations
            market_cap = info.get('marketCap', 0)
            if market_cap > 100e9:  # Large cap
                score += 10
            elif market_cap > 10e9:  # Mid cap
                score += 5
            
            # Dividend yield for income-focused investors
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            horizon_years = risk_profile['horizon_years']
            if horizon_years < 2 and dividend_yield > 2:
                score += 15
            elif horizon_years < 5 and dividend_yield > 1:
                score += 10
            
            # P/E ratio consideration
            pe_ratio = info.get('trailingPE', 0)
            if pe_ratio and 10 < pe_ratio < 25:
                score += 5
            elif pe_ratio and pe_ratio > 50:
                score -= 10
            
            # Sector diversification bonus
            sector = info.get('sector', '')
            if sector in ['Technology', 'Healthcare', 'Financials']:
                score += 5
            
            return max(0, min(100, score))
        
        except Exception as e:
            print(f"Error calculating score for {symbol}: {e}")
            return 0
    
    def get_recommendation_strength(self, score):
        """
        Convert score to recommendation strength
        
        Args:
            score (float): Stock score
        
        Returns:
            str: Recommendation strength
        """
        if score >= 80:
            return "Strong Buy"
        elif score >= 70:
            return "Buy"
        elif score >= 60:
            return "Moderate Buy"
        elif score >= 50:
            return "Hold"
        elif score >= 40:
            return "Weak Hold"
        else:
            return "Avoid"
    
    def generate_allocation(self, user_inputs, risk_profile):
        """
        Generate portfolio allocation recommendations
        
        Args:
            user_inputs (dict): User investment profile
            risk_profile (dict): Risk analysis
        
        Returns:
            dict: Portfolio allocation
        """
        risk_tolerance = user_inputs['risk_tolerance']
        base_allocation = self.risk_weights[risk_tolerance].copy()
        
        # Adjust for time horizon
        horizon_years = risk_profile['horizon_years']
        if horizon_years >= 10:
            base_allocation['stocks'] = min(0.9, base_allocation['stocks'] + 0.1)
            base_allocation['bonds'] = max(0.05, base_allocation['bonds'] - 0.05)
        elif horizon_years < 2:
            base_allocation['stocks'] = max(0.2, base_allocation['stocks'] - 0.2)
            base_allocation['bonds'] = min(0.7, base_allocation['bonds'] + 0.15)
            base_allocation['cash'] = min(0.3, base_allocation['cash'] + 0.05)
        
        # Detailed stock allocation
        stock_allocation = base_allocation['stocks']
        detailed_allocation = {
            'Large Cap Stocks': stock_allocation * 0.6,
            'Mid Cap Stocks': stock_allocation * 0.2,
            'Small Cap Stocks': stock_allocation * 0.1,
            'International Stocks': stock_allocation * 0.1,
            'Bonds': base_allocation['bonds'],
            'Cash/Money Market': base_allocation['cash']
        }
        
        return detailed_allocation
    
    def get_etf_recommendations(self, user_inputs):
        """
        Get ETF recommendations for diversification
        
        Args:
            user_inputs (dict): User investment profile
        
        Returns:
            list: ETF recommendations
        """
        risk_tolerance = user_inputs['risk_tolerance']
        
        etf_recommendations = []
        
        if risk_tolerance == 'Conservative':
            etf_recommendations = [
                {'symbol': 'VTI', 'name': 'Total Stock Market ETF', 'allocation': 30},
                {'symbol': 'BND', 'name': 'Total Bond Market ETF', 'allocation': 50},
                {'symbol': 'VXUS', 'name': 'International Stocks ETF', 'allocation': 20}
            ]
        elif risk_tolerance == 'Moderate':
            etf_recommendations = [
                {'symbol': 'VOO', 'name': 'S&P 500 ETF', 'allocation': 40},
                {'symbol': 'VEA', 'name': 'Developed Markets ETF', 'allocation': 20},
                {'symbol': 'BND', 'name': 'Total Bond Market ETF', 'allocation': 30},
                {'symbol': 'VWO', 'name': 'Emerging Markets ETF', 'allocation': 10}
            ]
        else:  # Aggressive
            etf_recommendations = [
                {'symbol': 'QQQ', 'name': 'NASDAQ-100 ETF', 'allocation': 35},
                {'symbol': 'VOO', 'name': 'S&P 500 ETF', 'allocation': 30},
                {'symbol': 'VEA', 'name': 'Developed Markets ETF', 'allocation': 15},
                {'symbol': 'VWO', 'name': 'Emerging Markets ETF', 'allocation': 15},
                {'symbol': 'BND', 'name': 'Total Bond Market ETF', 'allocation': 5}
            ]
        
        return etf_recommendations
    
    def generate_analysis_summary(self, user_inputs, scored_stocks):
        """
        Generate analysis summary
        
        Args:
            user_inputs (dict): User investment profile
            scored_stocks (list): Scored stocks
        
        Returns:
            dict: Analysis summary
        """
        total_investment = user_inputs['initial_investment'] + (user_inputs['monthly_investment'] * 12)
        
        summary = {
            'total_annual_investment': total_investment,
            'recommended_stocks_count': len([s for s in scored_stocks if s['score'] >= 60]),
            'high_confidence_picks': len([s for s in scored_stocks if s['score'] >= 80]),
            'diversification_score': min(100, len(set([s['symbol'][:2] for s in scored_stocks[:10]])) * 10),
            'risk_adjusted_return_potential': self.estimate_return_potential(user_inputs, scored_stocks)
        }
        
        return summary
    
    def estimate_return_potential(self, user_inputs, scored_stocks):
        """
        Estimate potential returns based on portfolio
        
        Args:
            user_inputs (dict): User investment profile
            scored_stocks (list): Scored stocks
        
        Returns:
            dict: Return estimates
        """
        risk_tolerance = user_inputs['risk_tolerance']
        
        # Conservative estimates based on historical market performance
        base_returns = {
            'Conservative': {'min': 4, 'expected': 6, 'max': 8},
            'Moderate': {'min': 6, 'expected': 8, 'max': 12},
            'Aggressive': {'min': 8, 'expected': 10, 'max': 15}
        }
        
        return base_returns[risk_tolerance]
