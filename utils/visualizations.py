import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizations:
    """Class for creating financial visualizations and charts"""
    
    def __init__(self):
        # Apple-inspired color palette
        self.colors = {
            'primary': '#007AFF',
            'success': '#34C759',
            'danger': '#FF3B30',
            'warning': '#FF9500',
            'info': '#5AC8FA',
            'secondary': '#8E8E93',
            'background': '#F2F2F7',
            'surface': '#FFFFFF'
        }
        
        self.chart_config = {
            'displayModeBar': False,
            'responsive': True
        }
    
    def create_stock_chart(self, data, symbol, chart_type='candlestick'):
        """
        Create comprehensive stock chart with technical indicators
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            symbol (str): Stock symbol
            chart_type (str): Type of chart ('candlestick', 'line', 'area')
        
        Returns:
            plotly.graph_objects.Figure: Stock chart
        """
        if data.empty:
            return self.create_empty_chart("No data available")
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f'{symbol} Price Chart', 'Volume', 'RSI', 'MACD')
        )
        
        # Main price chart
        if chart_type == 'candlestick':
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol,
                    increasing_line_color=self.colors['success'],
                    decreasing_line_color=self.colors['danger']
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=f'{symbol} Price',
                    line=dict(color=self.colors['primary'], width=2),
                    fill='tonexty' if chart_type == 'area' else None
                ),
                row=1, col=1
            )
        
        # Moving averages
        if 'MA20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA20'],
                    mode='lines',
                    name='MA20',
                    line=dict(color=self.colors['warning'], width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        if 'MA50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA50'],
                    mode='lines',
                    name='MA50',
                    line=dict(color=self.colors['info'], width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color=self.colors['secondary'], width=1, dash='dash'),
                    opacity=0.5
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color=self.colors['secondary'], width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(142, 142, 147, 0.1)',
                    opacity=0.5
                ),
                row=1, col=1
            )
        
        # Volume chart
        colors = [self.colors['success'] if close >= open_price else self.colors['danger'] 
                 for close, open_price in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Volume moving average
        if 'Volume_MA' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Volume_MA'],
                    mode='lines',
                    name='Volume MA',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=2, col=1
            )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=3, col=1
            )
            
            # RSI levels - using shapes instead of hline for subplot compatibility
            fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=70, y1=70,
                         line=dict(color=self.colors['danger'], dash="dash", width=1),
                         opacity=0.5, row=3, col=1)
            fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=30, y1=30,
                         line=dict(color=self.colors['success'], dash="dash", width=1),
                         opacity=0.5, row=3, col=1)
            fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=50, y1=50,
                         line=dict(color=self.colors['secondary'], dash="dot", width=1),
                         opacity=0.3, row=3, col=1)
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color=self.colors['warning'], width=2)
                ),
                row=4, col=1
            )
            
            # MACD histogram
            if 'MACD_Histogram' in data.columns:
                colors = [self.colors['success'] if val >= 0 else self.colors['danger'] 
                         for val in data['MACD_Histogram']]
                
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['MACD_Histogram'],
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.6
                    ),
                    row=4, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_title='Date',
            template='plotly_white',
            font=dict(family='SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif'),
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Remove range slider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        return fig
    
    def create_allocation_chart(self, allocations):
        """
        Create portfolio allocation pie chart
        
        Args:
            allocations (dict): Portfolio allocations
        
        Returns:
            plotly.graph_objects.Figure: Allocation chart
        """
        if not allocations:
            return self.create_empty_chart("No allocation data")
        
        labels = list(allocations.keys())
        values = [allocations[label] * 100 for label in labels]  # Convert to percentages
        
        # Apple-inspired color palette
        colors = [
            '#007AFF', '#34C759', '#FF9500', 
            '#5AC8FA', '#FF3B30', '#AF52DE',
            '#FF2D92', '#FFCC00'
        ]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(
                    colors=colors[:len(labels)],
                    line=dict(color='#FFFFFF', width=2)
                ),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=12)
            )
        ])
        
        fig.update_layout(
            title='Recommended Portfolio Allocation',
            font=dict(family='SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif'),
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_comparison_chart(self, data_dict, title="Stock Comparison"):
        """
        Create comparison chart for multiple stocks
        
        Args:
            data_dict (dict): Dictionary with symbol as key and data as value
            title (str): Chart title
        
        Returns:
            plotly.graph_objects.Figure: Comparison chart
        """
        fig = go.Figure()
        
        colors = [self.colors['primary'], self.colors['success'], self.colors['warning'], 
                 self.colors['info'], self.colors['danger']]
        
        for i, (symbol, data) in enumerate(data_dict.items()):
            if not data.empty:
                # Normalize to percentage change from start
                normalized = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=normalized,
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=3),
                        hovertemplate=f'{symbol}: %{{y:.2f}}%<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Return (%)',
            template='plotly_white',
            font=dict(family='SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif'),
            height=500,
            hovermode='x unified'
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color=self.colors['secondary'], opacity=0.5)
        
        return fig
    
    def create_returns_chart(self, returns_data, symbol):
        """
        Create returns distribution chart
        
        Args:
            returns_data (pd.Series): Returns data
            symbol (str): Stock symbol
        
        Returns:
            plotly.graph_objects.Figure: Returns chart
        """
        if returns_data.empty:
            return self.create_empty_chart("No returns data")
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns_data * 100,  # Convert to percentage
                nbinsx=50,
                name='Returns Distribution',
                marker_color=self.colors['primary'],
                opacity=0.7
            )
        )
        
        # Add mean and std lines
        mean_return = returns_data.mean() * 100
        std_return = returns_data.std() * 100
        
        fig.add_vline(x=mean_return, line_dash="dash", line_color=self.colors['success'],
                     annotation_text=f"Mean: {mean_return:.2f}%")
        fig.add_vline(x=mean_return + std_return, line_dash="dot", line_color=self.colors['warning'],
                     annotation_text=f"+1σ: {mean_return + std_return:.2f}%")
        fig.add_vline(x=mean_return - std_return, line_dash="dot", line_color=self.colors['danger'],
                     annotation_text=f"-1σ: {mean_return - std_return:.2f}%")
        
        fig.update_layout(
            title=f'{symbol} Daily Returns Distribution',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            font=dict(family='SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif'),
            height=400
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix, title="Stock Correlations"):
        """
        Create correlation heatmap
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            title (str): Chart title
        
        Returns:
            plotly.graph_objects.Figure: Heatmap
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='Correlation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            font=dict(family='SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif'),
            height=500
        )
        
        return fig
    
    def create_metrics_dashboard(self, metrics, symbol):
        """
        Create metrics dashboard
        
        Args:
            metrics (dict): Financial metrics
            symbol (str): Stock symbol
        
        Returns:
            plotly.graph_objects.Figure: Metrics dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance', 'Risk Metrics', 'Technical Indicators', 'Valuation'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Performance metrics
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get('ytd_return', 0),
                title={'text': "YTD Return (%)"},
                gauge={'axis': {'range': [-50, 50]},
                       'bar': {'color': self.colors['primary']},
                       'steps': [{'range': [-50, 0], 'color': self.colors['danger']},
                                {'range': [0, 50], 'color': self.colors['success']}]}
            ),
            row=1, col=1
        )
        
        # Risk metrics
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get('volatility', 0),
                title={'text': "Volatility (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': self.colors['warning']}}
            ),
            row=1, col=2
        )
        
        # Technical indicators
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get('rsi', 50),
                title={'text': "RSI"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': self.colors['info']},
                       'steps': [{'range': [0, 30], 'color': self.colors['success']},
                                {'range': [30, 70], 'color': self.colors['secondary']},
                                {'range': [70, 100], 'color': self.colors['danger']}]}
            ),
            row=2, col=1
        )
        
        # Valuation
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics.get('pe_ratio', 0),
                title={'text': "P/E Ratio"},
                number={'font': {'color': self.colors['primary']}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title=f'{symbol} Key Metrics Dashboard',
            template='plotly_white',
            font=dict(family='SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif')
        )
        
        return fig
    
    def create_growth_chart(self, data, symbol, periods=['1M', '3M', '6M', '1Y', '2Y', '5Y']):
        """
        Create stock growth chart showing performance over multiple periods
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
            periods (list): List of time periods to show
        
        Returns:
            plotly.graph_objects.Figure: Growth chart
        """
        if data.empty:
            return self.create_empty_chart("No growth data available")
        
        # Calculate cumulative returns
        returns = data['Close'].pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # Create main growth chart
        fig = go.Figure()
        
        # Add cumulative return line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=cumulative_returns * 100,
                mode='lines',
                name=f'{symbol} Growth',
                line=dict(color=self.colors['primary'], width=3),
                fill='tonexty',
                fillcolor=f'rgba(0, 122, 255, 0.1)',
                hovertemplate=f'{symbol}: %{{y:.2f}}%<extra></extra>'
            )
        )
        
        # Add benchmark line
        fig.add_hline(y=0, line_dash="dash", line_color=self.colors['secondary'], 
                     opacity=0.5, annotation_text="Break Even")
        
        # Calculate period returns for display
        period_returns = {}
        current_price = data['Close'].iloc[-1]
        
        for period in periods:
            try:
                if period == '1M' and len(data) >= 21:
                    past_price = data['Close'].iloc[-21]
                elif period == '3M' and len(data) >= 63:
                    past_price = data['Close'].iloc[-63]
                elif period == '6M' and len(data) >= 126:
                    past_price = data['Close'].iloc[-126]
                elif period == '1Y' and len(data) >= 252:
                    past_price = data['Close'].iloc[-252]
                elif period == '2Y' and len(data) >= 504:
                    past_price = data['Close'].iloc[-504]
                elif period == '5Y' and len(data) >= 1260:
                    past_price = data['Close'].iloc[-1260]
                else:
                    continue
                
                period_return = ((current_price - past_price) / past_price) * 100
                period_returns[period] = period_return
            except:
                continue
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Growth Performance',
            xaxis_title='Date',
            yaxis_title='Total Return (%)',
            template='plotly_white',
            font=dict(family='SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif'),
            height=400,
            showlegend=False,
            annotations=[
                dict(
                    text=f"Period Returns: " + " | ".join([f"{k}: {v:+.1f}%" for k, v in period_returns.items()]),
                    xref="paper", yref="paper",
                    x=0.5, y=1.05, xanchor='center', yanchor='bottom',
                    showarrow=False,
                    font=dict(size=12, color=self.colors['secondary'])
                )
            ]
        )
        
        # Update fill color based on performance
        if cumulative_returns.iloc[-1] >= 0:
            fig.update_traces(fillcolor='rgba(52, 199, 89, 0.1)')  # Green for gains
        else:
            fig.update_traces(fillcolor='rgba(255, 59, 48, 0.1)')  # Red for losses
        
        return fig
    
    def create_empty_chart(self, message="No data available"):
        """
        Create empty chart with message
        
        Args:
            message (str): Message to display
        
        Returns:
            plotly.graph_objects.Figure: Empty chart
        """
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color=self.colors['secondary'])
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return fig
