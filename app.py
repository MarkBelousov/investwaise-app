import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time

# Import utility modules
from utils.data_fetcher import DataFetcher
from utils.technical_analysis import TechnicalAnalysis
from utils.recommendation_engine import RecommendationEngine
from utils.visualizations import Visualizations

# Page configuration
st.set_page_config(
    page_title="InvestWise - Smart Investment Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

# Initialize utility classes
@st.cache_resource
def get_utils():
    data_fetcher = DataFetcher()
    tech_analysis = TechnicalAnalysis()
    rec_engine = RecommendationEngine()
    visualizations = Visualizations()
    return data_fetcher, tech_analysis, rec_engine, visualizations

data_fetcher, tech_analysis, rec_engine, visualizations = get_utils()

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="main-title">📈 InvestWise</h1>
        <p class="subtitle">Smart Investment Advisor powered by Real Market Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme toggle
    col1, col2, col3 = st.columns([1, 1, 8])
    with col2:
        if st.button("🌙" if st.session_state.theme == 'light' else "☀️"):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.rerun()
    
    # Sidebar for user inputs
    with st.sidebar:
        st.markdown("### 💼 Investment Profile")
        
        # Investment parameters
        monthly_investment = st.number_input(
            "Monthly Investment Amount ($)",
            min_value=50,
            max_value=50000,
            value=1000,
            step=50,
            help="How much you plan to invest each month"
        )
        
        initial_investment = st.number_input(
            "Initial Investment ($)",
            min_value=0,
            max_value=1000000,
            value=5000,
            step=100,
            help="Amount you can invest right now"
        )
        
        investment_horizon = st.selectbox(
            "Investment Timeline",
            ["6 months", "1 year", "2 years", "5 years", "10 years", "20+ years"],
            index=2,
            help="How long you plan to keep your investments"
        )
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            index=1,
            help="Your comfort level with investment risk"
        )
        
        investment_goals = st.multiselect(
            "Investment Goals",
            ["Retirement", "Emergency Fund", "Home Purchase", "Education", "General Wealth Building"],
            default=["General Wealth Building"]
        )
        
        # Store user inputs
        st.session_state.user_inputs = {
            'monthly_investment': monthly_investment,
            'initial_investment': initial_investment,
            'investment_horizon': investment_horizon,
            'risk_tolerance': risk_tolerance,
            'investment_goals': investment_goals
        }
        
        # Generate recommendations button
        if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Analyzing market data and generating personalized recommendations..."):
                # Enhanced recommendations with goal-based filtering
                recommendations = rec_engine.generate_recommendations(st.session_state.user_inputs)
                
                # Add goal-specific stock analysis
                if recommendations and investment_goals:
                    goal_specific_analysis = {}
                    for goal in investment_goals:
                        goal_stocks = rec_engine.get_goal_specific_stocks(goal)
                        scored_goal_stocks = []
                        
                        for symbol in goal_stocks:
                            try:
                                score = rec_engine.calculate_stock_score(
                                    symbol, 
                                    st.session_state.user_inputs, 
                                    rec_engine.analyze_risk_profile(st.session_state.user_inputs)
                                )
                                if score > 0:
                                    scored_goal_stocks.append({
                                        'symbol': symbol,
                                        'score': score,
                                        'recommendation': rec_engine.get_recommendation_strength(score)
                                    })
                            except:
                                continue
                        
                        # Sort by score and take top 5
                        scored_goal_stocks.sort(key=lambda x: x['score'], reverse=True)
                        goal_specific_analysis[goal] = scored_goal_stocks[:5]
                    
                    recommendations['goal_specific'] = goal_specific_analysis
                
                st.session_state.recommendations = recommendations
                st.success("Personalized recommendations generated based on your goals!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Stock Search", "🎯 Recommendations", "💵 Return Simulator"])
    
    with tab1:
        if st.session_state.recommendations:
            display_recommendations()
        else:
            display_welcome_screen()
    
    with tab2:
        display_stock_search()
    
    with tab3:
        if st.session_state.recommendations:
            display_detailed_recommendations()
        else:
            st.info("Please complete your investment profile and generate recommendations first.")
    
    with tab4:
        display_return_simulator()

def display_welcome_screen():
    """Display welcome screen with market overview"""
    st.markdown("### 🌍 Market Overview")
    
    # Fetch major indices
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT"
    }
    
    # Display metrics
    cols = st.columns(len(indices))
    
    for i, (name, symbol) in enumerate(indices.items()):
        with cols[i]:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100
                    
                    color = "green" if change >= 0 else "red"
                    arrow = "▲" if change >= 0 else "▼"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{name}</h4>
                        <h3 style="color: {color};">{current_price:.2f}</h3>
                        <p style="color: {color};">{arrow} {change_pct:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading {name}: {str(e)}")
    
    # Market comparison chart
    st.markdown("##### Market Performance Today")
    try:
        market_data = {}
        for name, symbol in indices.items():
            data = data_fetcher.get_stock_data(symbol, period="1mo")
            if not data.empty:
                market_data[name] = data
        
        if market_data:
            fig_market = visualizations.create_comparison_chart(market_data, "Market Indices - 1 Month Performance")
            st.plotly_chart(fig_market, use_container_width=True, key="market_overview_chart")
    except:
        st.info("Market comparison chart temporarily unavailable")
    
    # Feature highlights
    st.markdown("### ✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Real-time Data</h4>
            <p>Access live market data and historical performance for informed decisions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Smart Recommendations</h4>
            <p>AI-powered investment suggestions based on your profile and market analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 Technical Analysis</h4>
            <p>Advanced charts with technical indicators and performance metrics</p>
        </div>
        """, unsafe_allow_html=True)

def display_recommendations():
    """Display investment recommendations and analysis"""
    recommendations = st.session_state.recommendations
    
    # Portfolio allocation
    st.markdown("### 🎯 Recommended Portfolio Allocation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pie chart for allocation
        if recommendations and 'allocations' in recommendations:
            fig = visualizations.create_allocation_chart(recommendations['allocations'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk assessment
        st.markdown("#### 📊 Risk Assessment")
        user_inputs = st.session_state.user_inputs
        risk_score = rec_engine.calculate_risk_score(user_inputs)
        
        st.markdown(f"""
        <div class="risk-card">
            <h4>Risk Level: {user_inputs['risk_tolerance']}</h4>
            <p>Score: {risk_score}/100</p>
            <div class="risk-bar">
                <div class="risk-fill" style="width: {risk_score}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Stock recommendations
    st.markdown("### 📈 Individual Stock Analysis")
    
    if recommendations and 'stocks' in recommendations:
        tabs = st.tabs([stock['symbol'] for stock in recommendations['stocks'][:5]])
        
        for i, stock in enumerate(recommendations['stocks'][:5]):
            with tabs[i]:
                display_stock_analysis(stock['symbol'])

def display_stock_analysis(symbol):
    """Display detailed analysis for a specific stock"""
    try:
        # Fetch stock data
        stock_data = data_fetcher.get_stock_data(symbol, period="1y")
        
        if stock_data.empty:
            st.error(f"No data available for {symbol}")
            return
        
        # Stock info
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = stock_data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
            change_pct = (change / stock_data['Close'].iloc[-2]) * 100
            st.metric("Daily Change", f"{change_pct:.2f}%", delta=f"${change:.2f}")
        
        with col3:
            volume = stock_data['Volume'].iloc[-1]
            st.metric("Volume", f"{volume:,.0f}")
        
        with col4:
            market_cap = info.get('marketCap', 0)
            if market_cap:
                st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
        
        # Time period selector
        period_options = ["1D", "1W", "1M", "3M", "1Y", "5Y"]
        selected_period = st.selectbox(f"Time Period for {symbol}", period_options, index=4, key=f"period_{symbol}")
        
        # Fetch data for selected period
        period_map = {"1D": "1d", "1W": "5d", "1M": "1mo", "3M": "3mo", "1Y": "1y", "5Y": "5y"}
        chart_data = data_fetcher.get_stock_data(symbol, period=period_map[selected_period])
        
        # Add technical indicators
        chart_data = tech_analysis.add_technical_indicators(chart_data)
        
        # Create charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main price chart with technical indicators
            fig = visualizations.create_stock_chart(chart_data, symbol)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Technical indicators summary
            st.markdown("#### Technical Indicators")
            
            if len(chart_data) > 0:
                rsi = chart_data['RSI'].iloc[-1] if 'RSI' in chart_data.columns else None
                if rsi:
                    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("RSI (14)", f"{rsi:.1f}", rsi_signal)
                
                # Moving averages
                if 'MA20' in chart_data.columns and 'MA50' in chart_data.columns:
                    ma20 = chart_data['MA20'].iloc[-1]
                    ma50 = chart_data['MA50'].iloc[-1]
                    trend = "Bullish" if ma20 > ma50 else "Bearish"
                    st.metric("20/50 MA Cross", trend)
        
        # Company information
        if info:
            st.markdown("#### Company Information")
            
            description = info.get('longBusinessSummary', 'No description available.')
            if len(description) > 500:
                description = description[:500] + "..."
            
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
            st.markdown(f"**Description:** {description}")
            
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")

def display_stock_search():
    """Display stock search functionality"""
    st.markdown("### 🔍 Stock Search & Analysis")
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for stocks by symbol or company name",
            placeholder="e.g., AAPL, Microsoft, Tesla",
            help="Enter stock symbol (AAPL) or company name"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Popular stocks section
    st.markdown("#### 📈 Popular Stocks by Sector")
    
    popular_stocks = data_fetcher.get_popular_stocks()
    
    # Create expandable sections for each sector
    for sector, stocks in popular_stocks.items():
        with st.expander(f"{sector} Stocks"):
            cols = st.columns(4)
            for i, symbol in enumerate(stocks):
                with cols[i % 4]:
                    if st.button(symbol, key=f"popular_{symbol}", use_container_width=True):
                        st.session_state.selected_search_stock = symbol
                        st.rerun()
    
    # Display search results or selected stock
    selected_stock = None
    if search_button and search_query:
        # Handle search query with autocorrect
        search_results = data_fetcher.search_stock_by_name(search_query)
        
        if search_results['exact_matches']:
            selected_stock = search_results['exact_matches'][0]
        elif search_results['suggestions']:
            st.info(f"Did you mean: {', '.join(search_results['suggestions'][:3])}?")
            # Auto-select the first suggestion
            selected_stock = search_results['suggestions'][0]
            st.success(f"Showing results for: {selected_stock}")
        else:
            st.error(f"Could not find any stocks matching '{search_query}'. Please check the symbol and try again.")
            
    elif hasattr(st.session_state, 'selected_search_stock'):
        selected_stock = st.session_state.selected_search_stock
    
    if selected_stock:
        st.markdown(f"### 📊 Analysis for {selected_stock}")
        
        # Validate and fetch stock data
        try:
            # First validate the stock symbol
            if not data_fetcher.validate_stock_symbol(selected_stock):
                st.error(f"Stock symbol '{selected_stock}' not found or has no data available.")
                return
            
            stock_data = data_fetcher.get_stock_data(selected_stock, period="1y")
            stock_info = data_fetcher.get_stock_info(selected_stock)
            
            if not stock_data.empty and stock_info:
                # Key metrics row
                col1, col2, col3, col4, col5 = st.columns(5)
                
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}", f"{change_pct:.2f}%")
                
                with col2:
                    volume = stock_data['Volume'].iloc[-1]
                    st.metric("Volume", f"{volume:,.0f}")
                
                with col3:
                    market_cap = stock_info.get('marketCap', 0)
                    if market_cap:
                        if market_cap > 1e12:
                            st.metric("Market Cap", f"${market_cap/1e12:.1f}T")
                        elif market_cap > 1e9:
                            st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
                        else:
                            st.metric("Market Cap", f"${market_cap/1e6:.1f}M")
                
                with col4:
                    pe_ratio = stock_info.get('trailingPE', 0)
                    if pe_ratio:
                        st.metric("P/E Ratio", f"{pe_ratio:.1f}")
                
                with col5:
                    dividend_yield = stock_info.get('dividendYield', 0)
                    if dividend_yield:
                        st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
                
                # Company information
                st.markdown("#### Company Information")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    company_name = stock_info.get('longName', selected_stock)
                    sector = stock_info.get('sector', 'N/A')
                    industry = stock_info.get('industry', 'N/A')
                    
                    st.markdown(f"**Company:** {company_name}")
                    st.markdown(f"**Sector:** {sector}")
                    st.markdown(f"**Industry:** {industry}")
                    
                    description = stock_info.get('longBusinessSummary', '')
                    if description:
                        if len(description) > 300:
                            description = description[:300] + "..."
                        st.markdown(f"**Description:** {description}")
                
                with col2:
                    # Investment suitability based on user profile
                    if st.session_state.user_inputs:
                        st.markdown("#### Investment Suitability")
                        suitability_score = calculate_stock_suitability(selected_stock, st.session_state.user_inputs)
                        
                        if suitability_score >= 80:
                            st.success(f"Excellent Match ({suitability_score}/100)")
                        elif suitability_score >= 60:
                            st.info(f"Good Match ({suitability_score}/100)")
                        elif suitability_score >= 40:
                            st.warning(f"Fair Match ({suitability_score}/100)")
                        else:
                            st.error(f"Poor Match ({suitability_score}/100)")
                
                # Comprehensive chart analysis
                chart_data = tech_analysis.add_technical_indicators(stock_data)
                
                # Main technical analysis chart
                st.markdown("##### 📊 Technical Analysis")
                fig_tech = visualizations.create_stock_chart(chart_data, selected_stock)
                st.plotly_chart(fig_tech, use_container_width=True)
                
                # Multi-chart dashboard
                chart_row1_col1, chart_row1_col2 = st.columns(2)
                
                with chart_row1_col1:
                    st.markdown("##### 📈 Growth Performance")
                    growth_data = data_fetcher.get_stock_data(selected_stock, period="5y")
                    if not growth_data.empty:
                        fig_growth = visualizations.create_growth_chart(growth_data, selected_stock)
                        st.plotly_chart(fig_growth, use_container_width=True)
                    else:
                        st.info("Growth data not available for this period")
                
                with chart_row1_col2:
                    st.markdown("##### 📊 Returns Distribution")
                    if not stock_data.empty:
                        returns = stock_data['Close'].pct_change().dropna()
                        fig_returns = visualizations.create_returns_chart(returns, selected_stock)
                        st.plotly_chart(fig_returns, use_container_width=True)
                
                # Additional analysis charts
                chart_row2_col1, chart_row2_col2 = st.columns(2)
                
                with chart_row2_col1:
                    st.markdown("##### 🎯 Performance Metrics")
                    # Calculate key metrics
                    returns = data_fetcher.calculate_returns(stock_data)
                    volatility = tech_analysis.calculate_volatility(stock_data)
                    
                    metrics = {
                        'Annual Return': returns.get(365, 0),
                        'Volatility': volatility.get('annual_volatility', 0),
                        'Sharpe Ratio': returns.get(365, 0) / max(volatility.get('annual_volatility', 1), 0.01),
                        'Max Drawdown': -abs(returns.get(365, 0) * 0.2)  # Simplified calculation
                    }
                    
                    fig_metrics = visualizations.create_metrics_dashboard(metrics, selected_stock)
                    st.plotly_chart(fig_metrics, use_container_width=True)
                
                with chart_row2_col2:
                    st.markdown("##### 🔄 Volume Analysis")
                    # Create volume trend chart
                    volume_data = stock_data[['Close', 'Volume']].tail(60)  # Last 60 days
                    if not volume_data.empty:
                        fig_volume = go.Figure()
                        
                        # Price line
                        fig_volume.add_trace(go.Scatter(
                            x=volume_data.index,
                            y=volume_data['Close'],
                            name='Price',
                            line=dict(color='#007AFF', width=2),
                            yaxis='y2'
                        ))
                        
                        # Volume bars with price-based colors
                        close_prices = volume_data['Close'].tolist()
                        colors = []
                        for i in range(len(close_prices)):
                            if i == 0:
                                colors.append('#34C759')  # First bar
                            else:
                                color = '#34C759' if close_prices[i] >= close_prices[i-1] else '#FF3B30'
                                colors.append(color)
                        
                        fig_volume.add_trace(go.Bar(
                            x=volume_data.index,
                            y=volume_data['Volume'],
                            name='Volume',
                            marker_color=colors,
                            yaxis='y1'
                        ))
                        
                        fig_volume.update_layout(
                            title=f'{selected_stock} - Price vs Volume (60 Days)',
                            xaxis_title='Date',
                            yaxis=dict(title='Volume', side='left'),
                            yaxis2=dict(title='Price ($)', side='right', overlaying='y'),
                            height=400,
                            showlegend=True,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_volume, use_container_width=True, key=f"volume_chart_{selected_stock}")
                
                # Technical signals
                signals = tech_analysis.generate_signals(chart_data)
                if signals:
                    st.markdown("#### Technical Signals")
                    cols = st.columns(len(signals))
                    
                    for i, (indicator, signal_data) in enumerate(signals.items()):
                        with cols[i]:
                            signal_type = signal_data['signal']
                            reason = signal_data['reason']
                            
                            if signal_type == 'BUY':
                                st.success(f"**{indicator}**: {signal_type}")
                            elif signal_type == 'SELL':
                                st.error(f"**{indicator}**: {signal_type}")
                            else:
                                st.info(f"**{indicator}**: {signal_type}")
                            
                            st.caption(reason)
                
            else:
                st.error(f"Could not find data for symbol: {selected_stock}")
                
        except Exception as e:
            st.error(f"Error fetching data for {selected_stock}: {str(e)}")

def calculate_stock_suitability(symbol, user_inputs):
    """Calculate how suitable a stock is for the user's profile"""
    try:
        score = rec_engine.calculate_stock_score(symbol, user_inputs, 
                                               rec_engine.analyze_risk_profile(user_inputs))
        return int(score)
    except:
        return 50  # Default neutral score

def display_detailed_recommendations():
    """Display detailed personalized recommendations based on user goals"""
    recommendations = st.session_state.recommendations
    user_inputs = st.session_state.user_inputs
    
    st.markdown("### 🎯 Personalized Investment Recommendations")
    
    # Investment goals analysis
    investment_goals = user_inputs.get('investment_goals', [])
    
    if investment_goals:
        st.markdown("#### Your Investment Goals")
        goal_cols = st.columns(len(investment_goals))
        
        for i, goal in enumerate(investment_goals):
            with goal_cols[i]:
                st.markdown(f"""
                <div class="goal-tag">
                    {goal}
                </div>
                """, unsafe_allow_html=True)
    
    # Portfolio overview charts
    if 'stocks' in recommendations and len(recommendations['stocks']) >= 3:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("##### Portfolio Allocation")
            allocation_data = {}
            for stock in recommendations['stocks'][:5]:
                allocation_data[stock['symbol']] = stock['score']
            fig_allocation = visualizations.create_allocation_chart(allocation_data)
            st.plotly_chart(fig_allocation, use_container_width=True, key="recommendations_allocation_chart")
        
        with chart_col2:
            st.markdown("##### Performance Comparison")
            comparison_data = {}
            for stock in recommendations['stocks'][:5]:
                try:
                    data = data_fetcher.get_stock_data(stock['symbol'], period="6mo")
                    if not data.empty:
                        comparison_data[stock['symbol']] = data
                except:
                    continue
            
            if comparison_data:
                fig_comparison = visualizations.create_comparison_chart(comparison_data, "Recommended Stocks - 6 Month Performance")
                st.plotly_chart(fig_comparison, use_container_width=True, key="recommendations_performance_chart")

    # Goal-specific recommendations using enhanced data
    st.markdown("#### Recommended Stocks Based on Your Goals")
    
    # Use enhanced goal-specific recommendations if available
    if 'goal_specific' in recommendations:
        goal_based_stocks = recommendations['goal_specific']
    else:
        goal_based_stocks = get_goal_based_recommendations(user_inputs, recommendations)
    
    for goal, stocks in goal_based_stocks.items():
        if stocks:
            st.markdown(f"##### 📋 Best Stocks for {goal}")
            
            # Goal-specific performance chart
            if len(stocks) >= 2:
                goal_data = {}
                for stock in stocks[:3]:
                    try:
                        # Validate stock exists before fetching data
                        if data_fetcher.validate_stock_symbol(stock['symbol']):
                            data = data_fetcher.get_stock_data(stock['symbol'], period="1y")
                            if not data.empty:
                                goal_data[stock['symbol']] = data
                    except:
                        continue
                
                if goal_data:
                    fig_goal = visualizations.create_comparison_chart(goal_data, f"{goal} - Stock Performance Comparison")
                    st.plotly_chart(fig_goal, use_container_width=True, key=f"goal_{goal.replace(' ', '_').lower()}_chart")
            
            cols = st.columns(min(3, len(stocks)))
            
            for i, stock in enumerate(stocks[:3]):
                with cols[i]:
                    symbol = stock['symbol']
                    score = stock.get('score', 0)
                    recommendation = stock.get('recommendation', 'Hold')
                    
                    # Get additional stock info
                    try:
                        stock_info = data_fetcher.get_stock_info(symbol)
                        company_name = stock_info.get('shortName', symbol)
                        current_price = 0
                        
                        # Get current price
                        stock_data = data_fetcher.get_stock_data(symbol, period="1d")
                        if not stock_data.empty:
                            current_price = stock_data['Close'].iloc[-1]
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{symbol}</h4>
                            <h5>{company_name}</h5>
                            <h3>${current_price:.2f}</h3>
                            <p><span class="recommendation-badge {recommendation.lower().replace(' ', '-')}">{recommendation}</span></p>
                            <p>Suitability Score: {score:.0f}/100</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"Analyze {symbol}", key=f"analyze_{goal}_{symbol}", use_container_width=True):
                            st.session_state.selected_search_stock = symbol
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error loading {symbol}: {str(e)}")
    
    # Sector Analysis Dashboard
    st.markdown("#### 🏢 Sector Analysis")
    
    sector_chart_col1, sector_chart_col2 = st.columns(2)
    
    with sector_chart_col1:
        st.markdown("##### Sector Performance (1 Year)")
        sector_stocks = {
            'Technology': 'AAPL',
            'Healthcare': 'JNJ', 
            'Finance': 'JPM',
            'Energy': 'XOM',
            'Consumer': 'KO'
        }
        
        sector_performance = {}
        for sector, symbol in sector_stocks.items():
            try:
                data = data_fetcher.get_stock_data(symbol, period="1y")
                if not data.empty:
                    sector_performance[sector] = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            except:
                continue
        
        if sector_performance:
            fig_sector = go.Figure(data=[
                go.Bar(
                    x=list(sector_performance.keys()),
                    y=list(sector_performance.values()),
                    marker_color=['#34C759' if v >= 0 else '#FF3B30' for v in sector_performance.values()]
                )
            ])
            fig_sector.update_layout(
                title="Major Sectors Annual Performance",
                xaxis_title="Sector",
                yaxis_title="Return (%)",
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_sector, use_container_width=True, key="sector_performance_chart")
    
    with sector_chart_col2:
        st.markdown("##### Sector Leaders Comparison")
        comparison_data = {}
        for sector, symbol in list(sector_stocks.items())[:3]:
            try:
                data = data_fetcher.get_stock_data(symbol, period="6mo")
                if not data.empty:
                    comparison_data[f"{sector}"] = data
            except:
                continue
        
        if comparison_data:
            fig_sector_comp = visualizations.create_comparison_chart(comparison_data, "Sector Leaders - 6 Month Performance")
            st.plotly_chart(fig_sector_comp, use_container_width=True, key="sector_comparison_chart")

    # Risk-adjusted portfolio suggestion
    st.markdown("#### 💼 Complete Portfolio Suggestion")
    
    if recommendations and 'stocks' in recommendations:
        portfolio_data = {}
        for stock in recommendations['stocks'][:5]:
            portfolio_data[stock['symbol']] = stock.get('score', 70)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### Portfolio Allocation")
            fig_allocation = visualizations.create_allocation_chart(portfolio_data)
            st.plotly_chart(fig_allocation, use_container_width=True, key="portfolio_allocation_chart")
        
        with col2:
            st.markdown("##### Top Stock Picks")
            for i, stock in enumerate(recommendations['stocks'][:5]):
                symbol = stock['symbol']
                score = stock.get('score', 0)
                recommendation = stock.get('recommendation', 'Hold')
                
                st.markdown(f"""
                <div class="data-row">
                    <span class="data-label">{i+1}. {symbol}</span>
                    <span class="data-value recommendation-badge {recommendation.lower().replace(' ', '-')}">{recommendation}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Risk Analysis with Gauge Chart
    st.markdown("#### ⚖️ Risk Analysis Dashboard")
    risk_score = rec_engine.calculate_risk_score(user_inputs)
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("##### Risk Score")
        fig_risk = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Investment Risk Level"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#007AFF"},
                'steps': [
                    {'range': [0, 30], 'color': "#34C759"},
                    {'range': [30, 70], 'color': "#FF9500"},
                    {'range': [70, 100], 'color': "#FF3B30"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        fig_risk.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig_risk, use_container_width=True, key="risk_gauge_chart")
    
    with risk_col2:
        st.markdown("##### Risk Assessment")
        if risk_score <= 30:
            st.success("Conservative Profile - Lower risk, stable returns expected")
        elif risk_score <= 70:
            st.warning("Moderate Profile - Balanced risk and growth potential")
        else:
            st.error("Aggressive Profile - Higher risk, higher growth potential")
        
        st.markdown("##### Key Factors")
        risk_factors = {
            "Time Horizon": user_inputs.get('investment_horizon', 'Unknown'),
            "Risk Tolerance": user_inputs.get('risk_tolerance', 'Unknown'),
            "Monthly Investment": f"${user_inputs.get('monthly_investment', 0):,}"
        }
        
        for factor, value in risk_factors.items():
            st.text(f"{factor}: {value}")

def get_goal_based_recommendations(user_inputs, recommendations):
    """Get stock recommendations tailored to specific investment goals"""
    investment_goals = user_inputs.get('investment_goals', [])
    all_stocks = recommendations.get('stocks', [])
    
    goal_stocks = {}
    
    for goal in investment_goals:
        filtered_stocks = []
        
        for stock in all_stocks:
            symbol = stock['symbol']
            
            try:
                stock_info = data_fetcher.get_stock_info(symbol)
                sector = stock_info.get('sector', '')
                dividend_yield = stock_info.get('dividendYield', 0) or 0
                
                # Goal-specific filtering
                if goal == 'Retirement':
                    # Prefer dividend stocks and stable large caps
                    if dividend_yield > 0.02 or sector in ['Utilities', 'Consumer Staples', 'Healthcare']:
                        filtered_stocks.append(stock)
                
                elif goal == 'Emergency Fund':
                    # Prefer stable, liquid stocks
                    if sector in ['Financials', 'Utilities', 'Consumer Staples']:
                        filtered_stocks.append(stock)
                
                elif goal == 'Home Purchase':
                    # Prefer growth stocks for medium-term goals
                    if sector in ['Technology', 'Healthcare', 'Consumer Discretionary']:
                        filtered_stocks.append(stock)
                
                elif goal == 'Education':
                    # Balanced approach
                    if stock.get('score', 0) >= 60:
                        filtered_stocks.append(stock)
                
                elif goal == 'General Wealth Building':
                    # Include top performing stocks
                    if stock.get('score', 0) >= 70:
                        filtered_stocks.append(stock)
                        
            except Exception:
                continue
        
        # Sort by score and take top 5
        filtered_stocks.sort(key=lambda x: x.get('score', 0), reverse=True)
        goal_stocks[goal] = filtered_stocks[:5]
    
    return goal_stocks

def create_goal_based_portfolio(user_inputs, stocks):
    """Create a portfolio allocation based on user goals"""
    # This is a simplified version - in practice, you'd use more sophisticated optimization
    risk_tolerance = user_inputs.get('risk_tolerance', 'Moderate')
    investment_horizon = user_inputs.get('investment_horizon', '5 years')
    
    # Adjust allocation based on goals and risk
    if 'Emergency Fund' in user_inputs.get('investment_goals', []):
        # More conservative for emergency funds
        return {
            'Large Cap Stocks': 0.4,
            'Bonds': 0.4,
            'Cash': 0.2
        }
    elif 'Retirement' in user_inputs.get('investment_goals', []):
        # Balanced long-term approach
        return {
            'Large Cap Stocks': 0.5,
            'International Stocks': 0.2,
            'Bonds': 0.2,
            'REITs': 0.1
        }
    else:
        # Standard allocation based on risk tolerance
        base_allocation = rec_engine.risk_weights.get(user_inputs.get('risk_tolerance', 'Moderate'), 
                                                     {'stocks': 0.6, 'bonds': 0.3, 'cash': 0.1})
        return {
            'Large Cap Stocks': base_allocation['stocks'] * 0.7,
            'Mid Cap Stocks': base_allocation['stocks'] * 0.2,
            'Small Cap Stocks': base_allocation['stocks'] * 0.1,
            'Bonds': base_allocation['bonds'],
            'Cash': base_allocation['cash']
        }

def display_return_simulator():
    """Display the return simulator for monthly return expectations and custom stock selection"""
    st.markdown("### 💵 Return Simulator")

    # If recommendations exist, use them and user's initial data automatically
    if st.session_state.recommendations and 'stocks' in st.session_state.recommendations and st.session_state.user_inputs:
        user_inputs = st.session_state.user_inputs
        recommended_stocks = st.session_state.recommendations['stocks']
        allocations = st.session_state.recommendations.get('allocations', None)
        default_annual_return = 0.07  # fallback

        # Use user input values
        initial_investment = user_inputs.get('initial_investment', 5000)
        monthly_contribution = user_inputs.get('monthly_investment', 1000)
        # Parse years from investment_horizon string
        horizon_map = {
            '6 months': 0.5,
            '1 year': 1,
            '2 years': 2,
            '5 years': 5,
            '10 years': 10,
            '20+ years': 25
        }
        horizon_years = horizon_map.get(user_inputs.get('investment_horizon', '5 years'), 5)

        # Prepare stock symbols and allocations
        stock_symbols = [s['symbol'] for s in recommended_stocks[:5]]
        if allocations:
            # Normalize allocations to sum to 1
            total_alloc = sum(allocations.get(sym, 0) for sym in stock_symbols)
            allocs = {sym: allocations.get(sym, 0)/total_alloc if total_alloc else 1/len(stock_symbols) for sym in stock_symbols}
        else:
            allocs = {sym: 1/len(stock_symbols) for sym in stock_symbols}

        # Calculate weighted expected annual return
        returns = []
        for sym in stock_symbols:
            try:
                data = data_fetcher.get_stock_data(sym, period="5y")
                if not data.empty:
                    start = data['Close'].iloc[0]
                    end = data['Close'].iloc[-1]
                    years = (data.index[-1] - data.index[0]).days / 365.25
                    ann_return = ((end / start) ** (1/years)) - 1 if years > 0 else default_annual_return
                else:
                    ann_return = default_annual_return
            except:
                ann_return = default_annual_return
            returns.append(ann_return * allocs[sym])
        weighted_annual_return = sum(returns)

        st.info(f"Weighted Expected Annual Return (from recommendations): {weighted_annual_return*100:.2f}%")

        # Calculate projection
        months = int(horizon_years * 12)
        r = weighted_annual_return / 12
        future_value = initial_investment * (1 + r) ** months + monthly_contribution * (((1 + r) ** months - 1) / r) if r != 0 else initial_investment + monthly_contribution * months
        total_invested = initial_investment + monthly_contribution * months
        total_gain = future_value - total_invested
        avg_monthly_return = total_gain / months if months > 0 else 0

        st.markdown(f"#### 📈 Projected Portfolio Value: **${future_value:,.2f}**")
        st.markdown(f"#### 💰 Expected Average Monthly Return: **${avg_monthly_return:,.2f}**")

        # Show growth chart
        values = []
        val = initial_investment
        for m in range(1, months + 1):
            val = val * (1 + r) + monthly_contribution
            values.append(val)
        chart_df = pd.DataFrame({"Month": list(range(1, months + 1)), "Portfolio Value": values})
        fig = px.line(chart_df, x="Month", y="Portfolio Value", title="Portfolio Growth Over Time (Recommended Portfolio)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 Simulate With Custom Stock Portfolio")
        # (Keep the custom portfolio section as fallback/optional)
    else:
        # User input for simulation (manual form)
        with st.form("return_sim_form"):
            initial_investment = st.number_input(
                "Initial Investment ($)",
                min_value=0,
                max_value=1_000_000,
                value=st.session_state.user_inputs.get('initial_investment', 5000),
                step=100
            )
            monthly_contribution = st.number_input(
                "Monthly Contribution ($)",
                min_value=0,
                max_value=50_000,
                value=st.session_state.user_inputs.get('monthly_investment', 1000),
                step=50
            )
            horizon_years = st.slider(
                "Investment Horizon (years)",
                min_value=1,
                max_value=30,
                value=5
            )
            default_annual_return = 0.07  # 7% typical market return
            expected_annual_return = st.number_input(
                "Expected Annual Return (%)",
                min_value=0.0,
                max_value=20.0,
                value=7.0,
                step=0.1,
                help="Estimate based on market or your selected stocks"
            )
            submitted = st.form_submit_button("Simulate")

        if submitted:
            # Calculate projection
            months = horizon_years * 12
            r = (expected_annual_return / 100) / 12
            future_value = initial_investment * (1 + r) ** months + monthly_contribution * (((1 + r) ** months - 1) / r) if r != 0 else initial_investment + monthly_contribution * months
            total_invested = initial_investment + monthly_contribution * months
            total_gain = future_value - total_invested
            avg_monthly_return = total_gain / months if months > 0 else 0

            st.markdown(f"#### 📈 Projected Portfolio Value: **${future_value:,.2f}**")
            st.markdown(f"#### 💰 Expected Average Monthly Return: **${avg_monthly_return:,.2f}**")

            # Show growth chart
            values = []
            val = initial_investment
            for m in range(1, months + 1):
                val = val * (1 + r) + monthly_contribution
                values.append(val)
            chart_df = pd.DataFrame({"Month": list(range(1, months + 1)), "Portfolio Value": values})
            fig = px.line(chart_df, x="Month", y="Portfolio Value", title="Portfolio Growth Over Time")
            st.plotly_chart(fig, use_container_width=True)

    # --- Custom Portfolio Section (always available) ---
    st.markdown("---")
    st.markdown("#### 🎯 Simulate With Custom Stock Portfolio")
    # Get available stocks (from recommendations or data_fetcher)
    available_stocks = []
    if st.session_state.recommendations and 'stocks' in st.session_state.recommendations:
        available_stocks = [s['symbol'] for s in st.session_state.recommendations['stocks']]
    else:
        # fallback: use popular stocks
        popular = data_fetcher.get_popular_stocks()
        for stocks in popular.values():
            available_stocks.extend(stocks)
        available_stocks = list(set(available_stocks))

    selected_stocks = st.multiselect(
        "Select Stocks to Invest In",
        options=available_stocks,
        default=available_stocks[:3] if len(available_stocks) >= 3 else available_stocks
    )

    if selected_stocks:
        st.markdown("##### Set Allocation (%) for Each Stock (must sum to 100%)")
        allocations = {}
        total_alloc = 0
        for symbol in selected_stocks:
            alloc = st.number_input(f"{symbol} Allocation (%)", min_value=0, max_value=100, value=int(100/len(selected_stocks)), key=f"alloc_{symbol}")
            allocations[symbol] = alloc
            total_alloc += alloc
        if total_alloc != 100:
            st.warning(f"Total allocation is {total_alloc}%. Please ensure allocations sum to 100%.")
        else:
            # Get historical annual returns for each stock
            returns = []
            for symbol in selected_stocks:
                try:
                    data = data_fetcher.get_stock_data(symbol, period="5y")
                    if not data.empty:
                        start = data['Close'].iloc[0]
                        end = data['Close'].iloc[-1]
                        years = (data.index[-1] - data.index[0]).days / 365.25
                        ann_return = ((end / start) ** (1/years)) - 1 if years > 0 else default_annual_return
                    else:
                        ann_return = default_annual_return
                except:
                    ann_return = default_annual_return
                returns.append(ann_return * allocations[symbol] / 100)
            weighted_annual_return = sum(returns)
            st.info(f"Weighted Expected Annual Return: {weighted_annual_return*100:.2f}%")
            # Recalculate projection
            months = horizon_years * 12 if 'horizon_years' in locals() else 60
            r = weighted_annual_return / 12
            future_value = initial_investment * (1 + r) ** months + monthly_contribution * (((1 + r) ** months - 1) / r) if r != 0 else initial_investment + monthly_contribution * months
            total_invested = initial_investment + monthly_contribution * months
            total_gain = future_value - total_invested
            avg_monthly_return = total_gain / months if months > 0 else 0
            st.markdown(f"#### 📈 Projected Portfolio Value: **${future_value:,.2f}**")
            st.markdown(f"#### 💰 Expected Average Monthly Return: **${avg_monthly_return:,.2f}**")
            # Show growth chart
            values = []
            val = initial_investment
            for m in range(1, months + 1):
                val = val * (1 + r) + monthly_contribution
                values.append(val)
            chart_df = pd.DataFrame({"Month": list(range(1, months + 1)), "Portfolio Value": values})
            fig = px.line(chart_df, x="Month", y="Portfolio Value", title="Custom Portfolio Growth Over Time")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
