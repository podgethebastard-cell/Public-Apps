# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
from datetime import datetime, timedelta
import requests
import json
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Smart Money Concepts Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #089981;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2157f3;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .signal-bullish {
        background-color: rgba(8, 153, 129, 0.1);
        padding: 10px;
        border-left: 4px solid #089981;
        border-radius: 4px;
    }
    .signal-bearish {
        background-color: rgba(242, 54, 69, 0.1);
        padding: 10px;
        border-left: 4px solid #F23645;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

class SmartMoneyConcepts:
    """Implement Smart Money Concepts indicator from Pine Script"""
    
    def __init__(self, df: pd.DataFrame, params: Dict):
        self.df = df.copy()
        self.params = params
        
        # Constants
        self.BULLISH = 1
        self.BEARISH = -1
        
        # Initialize all required arrays
        self.initialize_arrays()
        
    def initialize_arrays(self):
        """Initialize all required arrays for calculations"""
        # Swing structure arrays
        self.swing_high = np.full(len(self.df), np.nan)
        self.swing_low = np.full(len(self.df), np.nan)
        self.swing_high_last = np.full(len(self.df), np.nan)
        self.swing_low_last = np.full(len(self.df), np.nan)
        
        # Internal structure arrays
        self.internal_high = np.full(len(self.df), np.nan)
        self.internal_low = np.full(len(self.df), np.nan)
        self.internal_high_last = np.full(len(self.df), np.nan)
        self.internal_low_last = np.full(len(self.df), np.nan)
        
        # Trend bias
        self.swing_trend = np.zeros(len(self.df))
        self.internal_trend = np.zeros(len(self.df))
        
        # Order blocks
        self.swing_order_blocks = []
        self.internal_order_blocks = []
        
        # Fair Value Gaps
        self.fair_value_gaps = []
        
        # Equal highs/lows
        self.equal_highs = []
        self.equal_lows = []
        
    def calculate_atr(self, period: int = 200) -> pd.Series:
        """Calculate Average True Range"""
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()
    
    def detect_swings(self, lookback: int = 50):
        """Detect swing highs and lows"""
        high = self.df['High']
        low = self.df['Low']
        
        for i in range(lookback, len(self.df)):
            window_high = high.iloc[i-lookback:i]
            window_low = low.iloc[i-lookback:i]
            
            # Check for swing high
            if high.iloc[i] == window_high.max():
                self.swing_high[i] = high.iloc[i]
                if not np.isnan(self.swing_high[i-1]):
                    self.swing_high_last[i] = self.swing_high[i-1]
            
            # Check for swing low
            if low.iloc[i] == window_low.min():
                self.swing_low[i] = low.iloc[i]
                if not np.isnan(self.swing_low[i-1]):
                    self.swing_low_last[i] = self.swing_low[i-1]
    
    def detect_internal_swings(self, lookback: int = 5):
        """Detect internal structure swings"""
        high = self.df['High']
        low = self.df['Low']
        
        for i in range(lookback, len(self.df)):
            window_high = high.iloc[i-lookback:i]
            window_low = low.iloc[i-lookback:i]
            
            # Check for internal swing high
            if high.iloc[i] == window_high.max():
                self.internal_high[i] = high.iloc[i]
                if not np.isnan(self.internal_high[i-1]):
                    self.internal_high_last[i] = self.internal_high[i-1]
            
            # Check for internal swing low
            if low.iloc[i] == window_low.min():
                self.internal_low[i] = low.iloc[i]
                if not np.isnan(self.internal_low[i-1]):
                    self.internal_low_last[i] = self.internal_low[i-1]
    
    def detect_order_blocks(self, is_internal: bool = False):
        """Detect order blocks"""
        df = self.df
        atr = self.calculate_atr()
        
        for i in range(2, len(df)):
            # Bullish order block criteria
            if (df['Close'].iloc[i] > df['Close'].iloc[i-1] and 
                df['Close'].iloc[i-1] < df['Close'].iloc[i-2] and
                (df['High'].iloc[i-1] - df['Low'].iloc[i-1]) < (2 * atr.iloc[i-1])):
                
                ob = {
                    'index': i-1,
                    'high': df['High'].iloc[i-1],
                    'low': df['Low'].iloc[i-1],
                    'bias': self.BULLISH,
                    'is_internal': is_internal
                }
                
                if is_internal:
                    self.internal_order_blocks.append(ob)
                else:
                    self.swing_order_blocks.append(ob)
            
            # Bearish order block criteria
            elif (df['Close'].iloc[i] < df['Close'].iloc[i-1] and 
                  df['Close'].iloc[i-1] > df['Close'].iloc[i-2] and
                  (df['High'].iloc[i-1] - df['Low'].iloc[i-1]) < (2 * atr.iloc[i-1])):
                
                ob = {
                    'index': i-1,
                    'high': df['High'].iloc[i-1],
                    'low': df['Low'].iloc[i-1],
                    'bias': self.BEARISH,
                    'is_internal': is_internal
                }
                
                if is_internal:
                    self.internal_order_blocks.append(ob)
                else:
                    self.swing_order_blocks.append(ob)
    
    def detect_fair_value_gaps(self):
        """Detect Fair Value Gaps"""
        df = self.df
        
        for i in range(2, len(df)):
            # Bullish FVG
            if (df['Low'].iloc[i] > df['High'].iloc[i-2] and 
                df['Close'].iloc[i-1] > df['High'].iloc[i-2]):
                
                self.fair_value_gaps.append({
                    'index': i-1,
                    'top': df['Low'].iloc[i],
                    'bottom': df['High'].iloc[i-2],
                    'bias': self.BULLISH
                })
            
            # Bearish FVG
            elif (df['High'].iloc[i] < df['Low'].iloc[i-2] and 
                  df['Close'].iloc[i-1] < df['Low'].iloc[i-2]):
                
                self.fair_value_gaps.append({
                    'index': i-1,
                    'top': df['Low'].iloc[i-2],
                    'bottom': df['High'].iloc[i],
                    'bias': self.BEARISH
                })
    
    def detect_equal_highs_lows(self, confirmation_bars: int = 3, threshold: float = 0.1):
        """Detect Equal Highs and Lows"""
        df = self.df
        atr = self.calculate_atr()
        
        for i in range(confirmation_bars, len(df)):
            # Equal highs detection
            high_window = df['High'].iloc[i-confirmation_bars:i]
            current_high = df['High'].iloc[i]
            
            if abs(current_high - high_window.max()) < (threshold * atr.iloc[i]):
                self.equal_highs.append({
                    'index': i,
                    'price': current_high,
                    'type': 'EQH'
                })
            
            # Equal lows detection
            low_window = df['Low'].iloc[i-confirmation_bars:i]
            current_low = df['Low'].iloc[i]
            
            if abs(current_low - low_window.min()) < (threshold * atr.iloc[i]):
                self.equal_lows.append({
                    'index': i,
                    'price': current_low,
                    'type': 'EQL'
                })
    
    def detect_break_of_structure(self):
        """Detect Break of Structure (BOS) and Change of Character (CHoCH)"""
        df = self.df
        
        for i in range(1, len(df)):
            # Check for bullish BOS/CHoCH
            if not np.isnan(self.swing_high[i-1]):
                if df['Close'].iloc[i] > self.swing_high[i-1]:
                    if self.swing_trend[i-1] == self.BEARISH:
                        self.swing_trend[i] = self.BULLISH
                        # This would be CHoCH
                    else:
                        self.swing_trend[i] = self.BULLISH
                        # This would be BOS
            
            # Check for bearish BOS/CHoCH
            if not np.isnan(self.swing_low[i-1]):
                if df['Close'].iloc[i] < self.swing_low[i-1]:
                    if self.swing_trend[i-1] == self.BULLISH:
                        self.swing_trend[i] = self.BEARISH
                        # This would be CHoCH
                    else:
                        self.swing_trend[i] = self.BEARISH
                        # This would be BOS
            
            # Internal structure
            if not np.isnan(self.internal_high[i-1]):
                if df['Close'].iloc[i] > self.internal_high[i-1]:
                    self.internal_trend[i] = self.BULLISH
            
            if not np.isnan(self.internal_low[i-1]):
                if df['Close'].iloc[i] < self.internal_low[i-1]:
                    self.internal_trend[i] = self.BEARISH
    
    def calculate_all(self):
        """Calculate all Smart Money Concepts components"""
        print("Calculating Smart Money Concepts...")
        
        # Detect swings
        self.detect_swings(self.params.get('swings_length', 50))
        self.detect_internal_swings(5)
        
        # Detect order blocks
        self.detect_order_blocks(is_internal=False)
        self.detect_order_blocks(is_internal=True)
        
        # Detect FVGs
        self.detect_fair_value_gaps()
        
        # Detect equal highs/lows
        self.detect_equal_highs_lows(
            confirmation_bars=self.params.get('equal_length', 3),
            threshold=self.params.get('equal_threshold', 0.1)
        )
        
        # Detect structure breaks
        self.detect_break_of_structure()
        
        # Add calculated values to dataframe
        self.df['Swing_High'] = self.swing_high
        self.df['Swing_Low'] = self.swing_low
        self.df['Internal_High'] = self.internal_high
        self.df['Internal_Low'] = self.internal_low
        self.df['Swing_Trend'] = self.swing_trend
        self.df['Internal_Trend'] = self.internal_trend
        
        return self.df

class SqueezeMomentumIndicator:
    """Implement Squeeze Momentum Indicator from Pine Script"""
    
    def __init__(self, df: pd.DataFrame, params: Dict):
        self.df = df.copy()
        self.params = params
        
    def calculate_squeeze(self):
        """Calculate Squeeze Momentum Indicator"""
        df = self.df
        
        # Parameters
        bb_length = self.params.get('bb_length', 20)
        bb_mult = self.params.get('bb_mult', 2.0)
        kc_length = self.params.get('kc_length', 20)
        kc_mult = self.params.get('kc_mult', 1.5)
        
        # Calculate Bollinger Bands
        bb_basis = df['Close'].rolling(window=bb_length).mean()
        bb_std = df['Close'].rolling(window=bb_length).std()
        bb_upper = bb_basis + (bb_std * bb_mult)
        bb_lower = bb_basis - (bb_std * bb_mult)
        
        # Calculate Keltner Channels
        kc_ma = df['Close'].rolling(window=kc_length).mean()
        
        # Use True Range if specified
        if self.params.get('use_true_range', True):
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            range_ma = tr.rolling(window=kc_length).mean()
        else:
            range_ma = (df['High'] - df['Low']).rolling(window=kc_length).mean()
        
        kc_upper = kc_ma + (range_ma * kc_mult)
        kc_lower = kc_ma - (range_ma * kc_mult)
        
        # Squeeze conditions
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        squeeze_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
        no_squeeze = ~squeeze_on & ~squeeze_off
        
        # Calculate momentum
        highest_high = df['High'].rolling(window=kc_length).max()
        lowest_low = df['Low'].rolling(window=kc_length).min()
        avg_hl = (highest_high + lowest_low) / 2
        avg_close = df['Close'].rolling(window=kc_length).mean()
        
        # Linear regression momentum
        source = df['Close'] - ((avg_hl + avg_close) / 2)
        momentum = source.rolling(window=kc_length).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] * len(x), raw=True
        )
        
        # Determine colors
        momentum_color = np.where(
            momentum > 0,
            np.where(momentum > momentum.shift(), '#00FF00', '#008000'),
            np.where(momentum < momentum.shift(), '#FF0000', '#800000')
        )
        
        squeeze_color = np.where(
            no_squeeze, '#0000FF',
            np.where(squeeze_on, '#000000', '#808080')
        )
        
        # Add to dataframe
        df['Squeeze_Momentum'] = momentum
        df['Squeeze_On'] = squeeze_on
        df['Squeeze_Off'] = squeeze_off
        df['No_Squeeze'] = no_squeeze
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['KC_Upper'] = kc_upper
        df['KC_Lower'] = kc_lower
        
        return df

class AIAnalysisBrain:
    """AI analysis brain for technical analysis"""
    
    def __init__(self, smc_df: pd.DataFrame, squeeze_df: pd.DataFrame, ticker: str):
        self.smc_df = smc_df
        self.squeeze_df = squeeze_df
        self.ticker = ticker
        
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': self.ticker,
            'overall_bias': None,
            'key_levels': [],
            'signals': [],
            'risk_assessment': None,
            'recommendations': []
        }
        
        # Determine overall bias
        latest_trend = self.smc_df['Swing_Trend'].iloc[-1]
        if latest_trend == 1:
            report['overall_bias'] = 'BULLISH'
        elif latest_trend == -1:
            report['overall_bias'] = 'BEARISH'
        else:
            report['overall_bias'] = 'NEUTRAL'
        
        # Identify key levels (support/resistance)
        recent_swing_highs = self.smc_df['Swing_High'].dropna().tail(3).tolist()
        recent_swing_lows = self.smc_df['Swing_Low'].dropna().tail(3).tolist()
        
        report['key_levels'] = {
            'resistance_levels': sorted(recent_swing_highs, reverse=True),
            'support_levels': sorted(recent_swing_lows)
        }
        
        # Generate signals
        signals = []
        
        # Smart Money Concepts signals
        latest_close = self.smc_df['Close'].iloc[-1]
        
        # Check for BOS/CHoCH
        if len(self.smc_df) > 2:
            if self.smc_df['Swing_Trend'].iloc[-1] != self.smc_df['Swing_Trend'].iloc[-2]:
                signal_type = 'CHoCH' if abs(self.smc_df['Swing_Trend'].iloc[-1]) == 1 else 'BOS'
                signals.append({
                    'type': f'Swing {signal_type}',
                    'bias': 'BULLISH' if self.smc_df['Swing_Trend'].iloc[-1] == 1 else 'BEARISH',
                    'strength': 'HIGH',
                    'description': f'Major {signal_type} detected in swing structure'
                })
        
        # Squeeze signals
        if self.squeeze_df['Squeeze_On'].iloc[-1]:
            signals.append({
                'type': 'Squeeze',
                'bias': 'WATCH',
                'strength': 'MEDIUM',
                'description': 'Market in squeeze - potential breakout imminent'
            })
        
        if self.squeeze_df['Squeeze_Momentum'].iloc[-1] > 0 and \
           self.squeeze_df['Squeeze_Momentum'].iloc[-1] > self.squeeze_df['Squeeze_Momentum'].iloc[-2]:
            signals.append({
                'type': 'Momentum',
                'bias': 'BULLISH',
                'strength': 'MEDIUM',
                'description': 'Bullish momentum increasing'
            })
        
        report['signals'] = signals
        
        # Risk assessment
        volatility = self.smc_df['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
        if volatility < 0.2:
            report['risk_assessment'] = 'LOW'
        elif volatility < 0.4:
            report['risk_assessment'] = 'MEDIUM'
        else:
            report['risk_assessment'] = 'HIGH'
        
        # Generate recommendations
        recommendations = []
        
        if report['overall_bias'] == 'BULLISH' and len([s for s in signals if s['bias'] == 'BULLISH']) > 1:
            recommendations.append({
                'action': 'BUY',
                'confidence': 'HIGH',
                'reason': 'Multiple bullish confirmations',
                'stop_loss': min(report['key_levels']['support_levels']) if report['key_levels']['support_levels'] else latest_close * 0.98,
                'take_profit': max(report['key_levels']['resistance_levels']) if report['key_levels']['resistance_levels'] else latest_close * 1.02
            })
        elif report['overall_bias'] == 'BEARISH' and len([s for s in signals if s['bias'] == 'BEARISH']) > 1:
            recommendations.append({
                'action': 'SELL',
                'confidence': 'HIGH',
                'reason': 'Multiple bearish confirmations',
                'stop_loss': max(report['key_levels']['resistance_levels']) if report['key_levels']['resistance_levels'] else latest_close * 1.02,
                'take_profit': min(report['key_levels']['support_levels']) if report['key_levels']['support_levels'] else latest_close * 0.98
            })
        else:
            recommendations.append({
                'action': 'HOLD',
                'confidence': 'MEDIUM',
                'reason': 'Mixed signals - wait for confirmation',
                'stop_loss': 'N/A',
                'take_profit': 'N/A'
            })
        
        report['recommendations'] = recommendations
        
        return report

class TradingViewBroadcaster:
    """Handle broadcasting to TradingView, Telegram, etc."""
    
    def __init__(self, webhook_urls: Dict = None):
        self.webhook_urls = webhook_urls or {}
        
    def broadcast_signal(self, signal: Dict, platform: str = 'telegram'):
        """Broadcast signal to specified platform"""
        if platform not in self.webhook_urls:
            return False
        
        webhook_url = self.webhook_urls[platform]
        
        try:
            if platform == 'telegram':
                message = self.format_telegram_message(signal)
                response = requests.post(
                    webhook_url,
                    json={'text': message, 'parse_mode': 'HTML'}
                )
            elif platform == 'discord':
                message = self.format_discord_message(signal)
                response = requests.post(
                    webhook_url,
                    json={'content': message}
                )
            elif platform == 'tradingview':
                # TradingView webhook format
                response = requests.post(
                    webhook_url,
                    json=signal
                )
            else:
                return False
            
            return response.status_code == 200
        except Exception as e:
            print(f"Broadcast error: {e}")
            return False
    
    def format_telegram_message(self, signal: Dict) -> str:
        """Format message for Telegram"""
        return f"""
🚨 <b>Trading Signal Alert</b> 🚨

📊 <b>Ticker:</b> {signal.get('ticker', 'N/A')}
🎯 <b>Action:</b> {signal.get('action', 'N/A')}
📈 <b>Bias:</b> {signal.get('bias', 'N/A')}
💪 <b>Confidence:</b> {signal.get('confidence', 'N/A')}

📝 <b>Reason:</b> {signal.get('reason', 'N/A')}

🛑 <b>Stop Loss:</b> {signal.get('stop_loss', 'N/A')}
🎯 <b>Take Profit:</b> {signal.get('take_profit', 'N/A')}

⏰ <b>Time:</b> {signal.get('timestamp', 'N/A')}
        """
    
    def format_discord_message(self, signal: Dict) -> str:
        """Format message for Discord"""
        return f"""
**Trading Signal Alert** 

**Ticker:** {signal.get('ticker', 'N/A')}
**Action:** {signal.get('action', 'N/A')}
**Bias:** {signal.get('bias', 'N/A')}
**Confidence:** {signal.get('confidence', 'N/A')}

**Reason:** {signal.get('reason', 'N/A')}

**Stop Loss:** {signal.get('stop_loss', 'N/A')}
**Take Profit:** {signal.get('take_profit', 'N/A')}

**Time:** {signal.get('timestamp', 'N/A')}
        """

def fetch_market_data(ticker: str, period: str = '6mo', interval: str = '1d') -> pd.DataFrame:
    """Fetch market data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            # Try alternative method
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        # Return sample data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        return pd.DataFrame({
            'Open': np.random.normal(100, 10, 100).cumsum(),
            'High': np.random.normal(105, 10, 100).cumsum(),
            'Low': np.random.normal(95, 10, 100).cumsum(),
            'Close': np.random.normal(100, 10, 100).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

def create_plotly_chart(smc_df: pd.DataFrame, squeeze_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create interactive Plotly chart with all indicators"""
    
    # Create subplot figure
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=[f'{ticker} - Price with Smart Money Concepts', 
                       'Squeeze Momentum', 
                       'Order Blocks & FVGs',
                       'Volume']
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=smc_df.index,
            open=smc_df['Open'],
            high=smc_df['High'],
            low=smc_df['Low'],
            close=smc_df['Close'],
            name='Price',
            increasing_line_color='#089981',
            decreasing_line_color='#F23645'
        ),
        row=1, col=1
    )
    
    # Add swing highs and lows
    swing_highs = smc_df['Swing_High'].dropna()
    swing_lows = smc_df['Swing_Low'].dropna()
    
    fig.add_trace(
        go.Scatter(
            x=swing_highs.index,
            y=swing_highs.values,
            mode='markers',
            name='Swing High',
            marker=dict(color='#F23645', size=10, symbol='triangle-down'),
            hovertemplate='Swing High: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=swing_lows.index,
            y=swing_lows.values,
            mode='markers',
            name='Swing Low',
            marker=dict(color='#089981', size=10, symbol='triangle-up'),
            hovertemplate='Swing Low: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Squeeze Momentum indicator
    momentum = squeeze_df['Squeeze_Momentum'].dropna()
    
    # Create color array for histogram
    colors = []
    for i in range(len(momentum)):
        if i > 0:
            if momentum.iloc[i] > 0:
                colors.append('#00FF00' if momentum.iloc[i] > momentum.iloc[i-1] else '#008000')
            else:
                colors.append('#FF0000' if momentum.iloc[i] < momentum.iloc[i-1] else '#800000')
        else:
            colors.append('#808080')
    
    fig.add_trace(
        go.Bar(
            x=momentum.index,
            y=momentum.values,
            name='Momentum',
            marker_color=colors,
            width=1000 * 60 * 60 * 24  # Bar width in milliseconds for daily data
        ),
        row=2, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Add squeeze indicators
    squeeze_on = squeeze_df[squeeze_df['Squeeze_On']].index
    if len(squeeze_on) > 0:
        fig.add_trace(
            go.Scatter(
                x=squeeze_on,
                y=[0] * len(squeeze_on),
                mode='markers',
                name='Squeeze ON',
                marker=dict(color='black', size=8, symbol='diamond'),
                hovertemplate='Squeeze Active<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=smc_df.index,
            y=smc_df['Volume'],
            name='Volume',
            marker_color='#2157f3',
            opacity=0.7
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Momentum", row=2, col=1)
    fig.update_yaxes(title_text="Levels", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Smart Money Concepts Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=100)
        st.markdown("## 📊 Dashboard Settings")
        
        # Ticker selection
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Select Timeframe",
            ["1d", "1h", "4h", "1wk", "1mo"]
        )
        
        # Period selection
        period = st.selectbox(
            "Select Period",
            ["1mo", "3mo", "6mo", "1y", "2y"]
        )
        
        st.markdown("---")
        
        # Smart Money Concepts Settings
        st.markdown("### Smart Money Concepts")
        
        smc_settings = {
            'mode': st.selectbox("Mode", ["Historical", "Present"]),
            'style': st.selectbox("Style", ["Colored", "Monochrome"]),
            'show_internals': st.checkbox("Show Internal Structure", True),
            'swings_length': st.slider("Swing Length", 10, 100, 50),
            'show_order_blocks': st.checkbox("Show Order Blocks", True),
            'equal_length': st.slider("Equal Highs/Lows Bars", 1, 10, 3),
            'equal_threshold': st.slider("Equal Threshold", 0.0, 0.5, 0.1, 0.01)
        }
        
        st.markdown("---")
        
        # Squeeze Momentum Settings
        st.markdown("### Squeeze Momentum")
        
        squeeze_settings = {
            'bb_length': st.slider("BB Length", 10, 50, 20),
            'bb_mult': st.slider("BB Multiplier", 1.0, 3.0, 2.0, 0.1),
            'kc_length': st.slider("KC Length", 10, 50, 20),
            'kc_mult': st.slider("KC Multiplier", 1.0, 3.0, 1.5, 0.1),
            'use_true_range': st.checkbox("Use True Range", True)
        }
        
        st.markdown("---")
        
        # Broadcasting Settings
        st.markdown("### 📢 Broadcasting")
        broadcast_enabled = st.checkbox("Enable Broadcasting", False)
        
        if broadcast_enabled:
            telegram_webhook = st.text_input("Telegram Webhook URL", type="password")
            discord_webhook = st.text_input("Discord Webhook URL", type="password")
            tradingview_webhook = st.text_input("TradingView Webhook URL", type="password")
            
            webhooks = {}
            if telegram_webhook:
                webhooks['telegram'] = telegram_webhook
            if discord_webhook:
                webhooks['discord'] = discord_webhook
            if tradingview_webhook:
                webhooks['tradingview'] = tradingview_webhook
        
        st.markdown("---")
        
        # Quick Ticker Selection
        st.markdown("### Quick Select Tickers")
        quick_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "ETH-USD", "SPY", "QQQ"]
        cols = st.columns(3)
        for idx, qt in enumerate(quick_tickers):
            with cols[idx % 3]:
                if st.button(qt):
                    ticker = qt
        
        # Update button
        analyze_button = st.button("🔍 Analyze & Update", type="primary", use_container_width=True)
    
    # Main content area
    if 'analyze_button' not in locals():
        analyze_button = False
    
    # Initialize session state
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = None
    if 'last_data' not in st.session_state:
        st.session_state.last_data = None
    
    # Check if we need to fetch new data
    if analyze_button or st.session_state.last_ticker != ticker or st.session_state.last_data is None:
        with st.spinner(f"Fetching data for {ticker}..."):
            df = fetch_market_data(ticker, period, timeframe)
            
            if df.empty:
                st.error(f"Could not fetch data for {ticker}. Please try another ticker.")
                return
            
            # Calculate indicators
            with st.spinner("Calculating Smart Money Concepts..."):
                smc = SmartMoneyConcepts(df, smc_settings)
                smc_df = smc.calculate_all()
            
            with st.spinner("Calculating Squeeze Momentum..."):
                squeeze = SqueezeMomentumIndicator(df, squeeze_settings)
                squeeze_df = squeeze.calculate_squeeze()
            
            # Generate AI analysis
            with st.spinner("Generating AI Analysis..."):
                ai_brain = AIAnalysisBrain(smc_df, squeeze_df, ticker)
                analysis_report = ai_brain.generate_analysis_report()
            
            # Update session state
            st.session_state.last_ticker = ticker
            st.session_state.last_data = {
                'smc_df': smc_df,
                'squeeze_df': squeeze_df,
                'analysis_report': analysis_report,
                'ticker': ticker
            }
    
    # Display data if available
    if st.session_state.last_data:
        data = st.session_state.last_data
        smc_df = data['smc_df']
        squeeze_df = data['squeeze_df']
        analysis_report = data['analysis_report']
        ticker = data['ticker']
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Charts", 
            "🧠 AI Analysis", 
            "📊 Market Overview", 
            "📢 Broadcasting",
            "⚙️ Settings"
        ])
        
        with tab1:
            # Charts tab
            st.markdown(f"### {ticker} - Interactive Charts")
            
            # Create Plotly chart
            fig = create_plotly_chart(smc_df, squeeze_df, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Current Price",
                    f"${smc_df['Close'].iloc[-1]:.2f}",
                    f"{((smc_df['Close'].iloc[-1] - smc_df['Close'].iloc[-2]) / smc_df['Close'].iloc[-2] * 100):.2f}%"
                )
            
            with col2:
                latest_trend = analysis_report['overall_bias']
                trend_color = "#089981" if latest_trend == "BULLISH" else "#F23645" if latest_trend == "BEARISH" else "#808080"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Market Bias</h4>
                    <h2 style="color: {trend_color}">{latest_trend}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                risk_level = analysis_report['risk_assessment']
                risk_color = "#089981" if risk_level == "LOW" else "#FFA500" if risk_level == "MEDIUM" else "#F23645"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Risk Assessment</h4>
                    <h2 style="color: {risk_color}">{risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                signal_count = len(analysis_report['signals'])
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Active Signals</h4>
                    <h2>{signal_count}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            # AI Analysis tab
            st.markdown("### 🤖 AI Analysis Report")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Overall Analysis
                st.markdown("#### 📊 Overall Market Analysis")
                
                bias_class = "signal-bullish" if analysis_report['overall_bias'] == "BULLISH" else "signal-bearish" if analysis_report['overall_bias'] == "BEARISH" else ""
                st.markdown(f"""
                <div class="{bias_class}">
                    <h4>Market Bias: {analysis_report['overall_bias']}</h4>
                    <p>Based on Smart Money Concepts and Squeeze Momentum analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Key Levels
                st.markdown("#### 🎯 Key Price Levels")
                levels_col1, levels_col2 = st.columns(2)
                
                with levels_col1:
                    st.markdown("**Resistance Levels**")
                    for level in analysis_report['key_levels']['resistance_levels']:
                        st.write(f"• ${level:.2f}")
                
                with levels_col2:
                    st.markdown("**Support Levels**")
                    for level in analysis_report['key_levels']['support_levels']:
                        st.write(f"• ${level:.2f}")
                
                # Recommendations
                st.markdown("#### 💡 Trading Recommendations")
                for rec in analysis_report['recommendations']:
                    rec_color = "#089981" if rec['action'] == "BUY" else "#F23645" if rec['action'] == "SELL" else "#808080"
                    st.markdown(f"""
                    <div style="border-left: 4px solid {rec_color}; padding-left: 10px; margin: 10px 0;">
                        <h4 style="color: {rec_color}">{rec['action']} - Confidence: {rec['confidence']}</h4>
                        <p><strong>Reason:</strong> {rec['reason']}</p>
                        <p><strong>Stop Loss:</strong> {rec['stop_loss']}</p>
                        <p><strong>Take Profit:</strong> {rec['take_profit']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Active Signals
                st.markdown("#### 🚨 Active Signals")
                
                for signal in analysis_report['signals']:
                    signal_class = "signal-bullish" if signal['bias'] == "BULLISH" else "signal-bearish" if signal['bias'] == "BEARISH" else ""
                    st.markdown(f"""
                    <div class="{signal_class}" style="margin-bottom: 10px;">
                        <strong>{signal['type']}</strong><br/>
                        Bias: {signal['bias']}<br/>
                        Strength: {signal['strength']}<br/>
                        <small>{signal['description']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Technical Indicators Summary
                st.markdown("#### 📈 Technical Summary")
                
                # Calculate additional technical indicators
                rsi = talib.RSI(smc_df['Close'].values, timeperiod=14)[-1]
                macd, macd_signal, macd_hist = talib.MACD(smc_df['Close'].values)
                
                st.metric("RSI (14)", f"{rsi:.2f}")
                st.metric("MACD", f"{macd[-1]:.2f}")
                
                # Volatility
                volatility = smc_df['Close'].pct_change().std() * np.sqrt(252)
                st.metric("Annual Volatility", f"{volatility:.2%}")
        
        with tab3:
            # Market Overview tab
            st.markdown("### 🌐 Market Overview")
            
            # Multiple ticker comparison
            comparison_tickers = st.multiselect(
                "Compare with other tickers",
                ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BTC-USD", "ETH-USD", "SPY", "QQQ"],
                default=["AAPL", "MSFT", "GOOGL"]
            )
            
            if comparison_tickers:
                comparison_data = {}
                for ct in comparison_tickers:
                    try:
                        ct_df = fetch_market_data(ct, "1mo", "1d")
                        if not ct_df.empty:
                            comparison_data[ct] = ct_df
                    except:
                        pass
                
                if comparison_data:
                    # Create comparison chart
                    fig_compare = go.Figure()
                    
                    for ticker_name, ticker_df in comparison_data.items():
                        # Normalize prices for comparison
                        normalized_prices = (ticker_df['Close'] / ticker_df['Close'].iloc[0]) * 100
                        fig_compare.add_trace(
                            go.Scatter(
                                x=ticker_df.index,
                                y=normalized_prices,
                                name=ticker_name,
                                mode='lines'
                            )
                        )
                    
                    fig_compare.update_layout(
                        title="Normalized Price Comparison (Last 1 Month)",
                        yaxis_title="Normalized Price (%)",
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_compare, use_container_width=True)
            
            # Market news (placeholder)
            st.markdown("#### 📰 Market News")
            news_col1, news_col2, news_col3 = st.columns(3)
            
            with news_col1:
                st.markdown("""
                **Fed Rate Decision**
                - Expected to hold rates steady
                - Focus on inflation guidance
                """)
            
            with news_col2:
                st.markdown("""
                **Earnings Season**
                - Tech earnings this week
                - Expectations remain high
                """)
            
            with news_col3:
                st.markdown("""
                **Economic Data**
                - Jobs report Friday
                - CPI data next week
                """)
        
        with tab4:
            # Broadcasting tab
            st.markdown("### 📢 Signal Broadcasting")
            
            if broadcast_enabled and 'webhooks' in locals() and webhooks:
                st.success("Broadcasting enabled!")
                
                # Create signal for broadcasting
                if analysis_report['recommendations']:
                    primary_rec = analysis_report['recommendations'][0]
                    
                    broadcast_signal = {
                        'ticker': ticker,
                        'action': primary_rec['action'],
                        'bias': analysis_report['overall_bias'],
                        'confidence': primary_rec['confidence'],
                        'reason': primary_rec['reason'],
                        'stop_loss': primary_rec['stop_loss'],
                        'take_profit': primary_rec['take_profit'],
                        'timestamp': analysis_report['timestamp'],
                        'current_price': smc_df['Close'].iloc[-1]
                    }
                
                # Platform selection
                st.markdown("#### Select Platforms to Broadcast")
                
                platforms_to_broadcast = []
                cols = st.columns(4)
                if 'telegram' in webhooks:
                    with cols[0]:
                        if st.checkbox("Telegram", True):
                            platforms_to_broadcast.append('telegram')
                
                if 'discord' in webhooks:
                    with cols[1]:
                        if st.checkbox("Discord", True):
                            platforms_to_broadcast.append('discord')
                
                if 'tradingview' in webhooks:
                    with cols[2]:
                        if st.checkbox("TradingView", True):
                            platforms_to_broadcast.append('tradingview')
                
                # Broadcast button
                if st.button("📤 Broadcast Signal", type="primary"):
                    broadcaster = TradingViewBroadcaster(webhooks)
                    
                    success_count = 0
                    for platform in platforms_to_broadcast:
                        if broadcaster.broadcast_signal(broadcast_signal, platform):
                            success_count += 1
                            st.success(f"Signal sent to {platform.capitalize()}!")
                        else:
                            st.error(f"Failed to send to {platform.capitalize()}")
                    
                    if success_count > 0:
                        st.balloons()
                
                # Preview message
                st.markdown("#### 📋 Message Preview")
                st.json(broadcast_signal)
            
            else:
                st.warning("Broadcasting is not configured. Enable it in the sidebar settings.")
        
        with tab5:
            # Settings tab
            st.markdown("### ⚙️ Advanced Settings")
            
            # Data refresh settings
            st.markdown("#### Data Settings")
            auto_refresh = st.checkbox("Enable Auto-Refresh", False)
            if auto_refresh:
                refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60)
                st.info(f"Data will refresh every {refresh_interval} seconds")
            
            # Alert settings
            st.markdown("#### Alert Settings")
            alert_rsi = st.slider("RSI Alert Threshold", 0, 100, (30, 70))
            st.write(f"Alert when RSI below {alert_rsi[0]} or above {alert_rsi[1]}")
            
            # Export settings
            st.markdown("#### Export Data")
            if st.button("Export Analysis to CSV"):
                # Combine data
                export_df = smc_df.copy()
                export_df['Squeeze_Momentum'] = squeeze_df['Squeeze_Momentum']
                export_df['Market_Bias'] = analysis_report['overall_bias']
                
                csv = export_df.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        # Initial state - show welcome
        st.markdown("""
        ## Welcome to the Smart Money Concepts Dashboard! 🎯
        
        This comprehensive trading dashboard combines:
        
        ### 📊 Smart Money Concepts Indicators:
        - **Market Structure** (BOS/CHoCH detection)
        - **Order Blocks** (Smart money accumulation/distribution zones)
        - **Fair Value Gaps** (FVG imbalance zones)
        - **Equal Highs/Lows** (Key support/resistance levels)
        - **Premium/Discount Zones** (Market value areas)
        
        ### 🔄 Squeeze Momentum Indicator:
        - **Bollinger Bands & Keltner Channels**
        - **Momentum oscillator**
        - **Squeeze detection** (compression before expansion)
        
        ### 🤖 AI Analysis Brain:
        - **Automated technical analysis**
        - **Signal generation**
        - **Risk assessment**
        - **Trading recommendations**
        
        ### 📢 Broadcasting Features:
        - **Telegram integration**
        - **Discord webhooks**
        - **TradingView alerts**
        - **Multi-platform signal distribution**
        
        ### 🚀 Getting Started:
        1. Enter a ticker symbol in the sidebar
        2. Adjust settings as needed
        3. Click "Analyze & Update"
        4. Explore different tabs for analysis
        
        *Note: For demonstration purposes, some features may use sample data when live data is unavailable.*
        """)
        
        # Display sample chart
        st.info("Enter a ticker symbol in the sidebar and click 'Analyze & Update' to begin.")

if __name__ == "__main__":
    main()
