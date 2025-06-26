# InvestWise - Smart Investment Advisor

## Overview

InvestWise is a modern financial application built with Streamlit that provides intelligent investment analysis and recommendations. The application combines real-time market data, technical analysis, and machine learning to offer personalized investment advice to users based on their risk profiles and investment goals.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Styling**: Custom CSS with Apple-inspired design language
- **Visualization**: Plotly for interactive financial charts and graphs
- **Layout**: Wide layout with expandable sidebar for navigation

### Backend Architecture
- **Core Logic**: Modular Python architecture with utility classes
- **Data Processing**: Pandas for data manipulation and analysis
- **Machine Learning**: Scikit-learn for recommendation algorithms
- **Technical Analysis**: Pandas-TA for financial indicators calculation

### Data Sources
- **Primary**: Yahoo Finance API via yfinance library
- **Real-time**: Market data with configurable intervals and periods
- **Caching**: Streamlit's built-in caching for performance optimization

## Key Components

### 1. Data Fetcher (`utils/data_fetcher.py`)
- Handles all external data retrieval from Yahoo Finance
- Implements caching strategies (5-minute TTL for stock data)
- Supports both single and multiple stock data fetching
- Error handling for invalid symbols and API failures

### 2. Technical Analysis (`utils/technical_analysis.py`)
- Calculates standard technical indicators (MA, EMA, MACD, RSI, Bollinger Bands)
- Uses pandas-ta library for reliable indicator calculations
- Provides comprehensive market analysis capabilities

### 3. Recommendation Engine (`utils/recommendation_engine.py`)
- Machine learning-powered investment recommendations
- Risk profile analysis with predefined allocation strategies:
  - Conservative: 30% stocks, 60% bonds, 10% cash
  - Moderate: 60% stocks, 30% bonds, 10% cash
  - Aggressive: 80% stocks, 15% bonds, 5% cash
- Time horizon-based growth vs dividend weighting

### 4. Visualizations (`utils/visualizations.py`)
- Interactive Plotly charts for financial data
- Apple-inspired color palette and design
- Multiple chart types: candlestick, line, area charts
- Technical indicator overlays and subplots

### 5. User Interface
- Clean, modern interface with Apple-inspired styling
- Responsive design with mobile-friendly layouts
- Theme support (light/dark modes)
- Session state management for user preferences

## Data Flow

1. **User Input**: Users provide investment preferences, risk tolerance, and target symbols
2. **Data Acquisition**: System fetches real-time market data via Yahoo Finance API
3. **Analysis Pipeline**: 
   - Technical indicators calculated on raw price data
   - Machine learning models analyze patterns and trends
   - Risk assessment based on user profile
4. **Recommendation Generation**: Algorithm combines technical analysis with user preferences
5. **Visualization**: Results presented through interactive charts and dashboards
6. **Caching**: Processed data cached to improve performance

## External Dependencies

### Core Libraries
- **yfinance**: Yahoo Finance API integration for market data
- **streamlit**: Web application framework
- **plotly**: Interactive visualization library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms

### Technical Analysis
- **pandas-ta**: Technical analysis indicators library

### Infrastructure
- **Python 3.11**: Runtime environment
- **Nix**: Package management and environment isolation
- **UV**: Python dependency management

## Deployment Strategy

### Platform
- **Target**: Replit Autoscale deployment
- **Port**: Application runs on port 5000
- **Command**: `streamlit run app.py --server.port 5000`

### Configuration
- Streamlit server configured for headless operation
- Custom theme with Apple-inspired color scheme
- Environment isolation using Nix packages

### Performance Optimizations
- Data caching with configurable TTL (5-10 minutes)
- Resource caching for utility class initialization
- Responsive design for various screen sizes

### Scalability Considerations
- Modular architecture allows for easy feature additions
- Caching reduces API calls and improves response times
- Stateless design facilitates horizontal scaling

## Changelog

```
Changelog:
- June 25, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```