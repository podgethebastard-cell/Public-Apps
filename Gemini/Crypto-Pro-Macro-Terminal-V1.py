import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pycoingecko import CoinGeckoAPI
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

# --- Page Config ---
st.set_page_config(
    page_title="Crypto Pro Macro Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #262730;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .big-font { font-size: 26px !important; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- Robust API Session (Anti-Ban) ---
def get_retry_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

# Initialize APIs
cg = CoinGeckoAPI(session=get_retry_session())

# --- Constants ---
ASSETS_MAP = {
    'bitcoin': 'BTC-USD',
    'ethereum': 'ETH-USD',
    'solana': 'SOL-USD',
    'binancecoin': 'BNB-USD',
    'ripple': 'XRP-USD',
    'dogecoin': 'DOGE-USD',
    'cardano': 'ADA-USD',
    'avalanche-2': 'AVAX-USD',
    'shiba-inu': 'SHIB-USD',
    'chainlink': 'LINK-USD'
}

TRAD_ASSETS = {
    'DXY (Dollar Index)': 'DX=F', 
    'VIX (Volatility)': '^VIX', 
    'Gold Futures': 'GC=F', 
    'S&P 500': '^GSPC'
}

# --- Backend Functions ---

@st.cache_data(ttl=3600)
def get_fear_and_greed():
    """Fetches Fear & Greed Index safely"""
    try:
        session = get_retry_session()
        response = session.get("https://api.alternative.me/fng/", timeout=10)
        data = response.json()
        return data['data'][0]
    except:
        return {'value': 50, 'value_classification': 'Neutral'}

@st.cache_data(ttl=1800)
def fetch_cg_market_data(coin_id, days):
    """Fetches Volume/Market Cap from CoinGecko (Better for Macro/Volume)"""
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
        
        # Convert to DataFrames
        df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df_vol = pd.DataFrame(data['total_volume'], columns=['timestamp', 'volume'])
        df_mc = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
        
        # Merge
        df = pd.merge(df_price, df_vol, on='timestamp')
        df = pd.merge(df, df_mc, on='timestamp')
        
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        
        # Macro Indicators
        df['volume_7dma'] = df['volume'].rolling(window=7).mean()
        
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_parallel_prices(tickers, period="1y"):
    """
    Fetches adjusted close prices for multiple tickers in parallel using yfinance.
    """
    try:
        # Threads=True enables parallel fetching inside yfinance
        data = yf.download(tickers, period=period, group_by='ticker', progress=False, threads=True)
        
        # Extract just the 'Close' column for each ticker to build a comparison matrix
        close_df = pd.DataFrame()
        for t in tickers:
            try:
                # Handle different yfinance return structures (Single vs Multi index)
                if isinstance(data.columns, pd.MultiIndex):
                    if t in data:
                        close_df[t] = data[t]['Close']
                else:
                    if 'Close' in data:
                        close_df[t] = data['Close']
            except:
                continue
                
        return close_df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_technical_data(ticker, period="1y"):
    """Fetches OHLC data and calculates indicators using pandas_ta"""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: return df

        # --- Indicator Calculations (Pandas TA) ---
        # append=True adds the new columns directly to the dataframe
        
        # 1. RSI (14)
        df.ta.rsi(length=14, append=True)
        
        # 2. MACD (12, 26, 9)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # 3. Bollinger Bands (20, 2)
        df.ta.bbands(length=20, std=2, append=True)
        
        # 4. Z-Score (30)
        df.ta.zscore(length=30, append=True)

        return df
    except Exception:
        return pd.DataFrame()

# --- Sidebar ---
st.sidebar.title("🎛️ Terminal Controls")

# Global Timeframe
tf_label = st.sidebar.select_slider(
    "Global Timeframe", 
    options=["90", "365", "1095"], 
    value="365",
    format_func=lambda x: {"90": "3 Months", "365": "1 Year", "1095": "3 Years"}[x]
)
cg_days = tf_label
yf_period = "3mo" if tf_label == "90" else "1y" if tf_label == "365" else "5y"

st.sidebar.markdown("---")
st.sidebar.info("Data Sources:\n- **Volume/Macro:** CoinGecko (Spot)\n- **Technicals/OHLC:** Yahoo Finance\n- **Sentiment:** Alternative.me")

# --- Header Section ---
fng = get_fear_and_greed()
fng_val = int(fng['value'])
color = "#00FF00" if fng_val > 60 else "#FF4B4B" if fng_val < 40 else "#FFA500"

c1, c2, c3 = st.columns([0.6, 0.2, 0.2])
with c1:
    st.title("🦅 Crypto Macro Terminal")
    st.caption("Institutional-grade analysis: Volume Structure, Liquidity Flow, and Correlations.")
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #888; font-size: 12px;">SENTIMENT</div>
        <div class="big-font" style="color: {color};">{fng_val}</div>
        <div style="font-size: 12px;">{fng['value_classification']}</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    # Quick Fuel Check
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #888; font-size: 12px;">MACRO REGIME</div>
        <div class="big-font" style="color: #fff;">{'RISK ON' if fng_val > 45 else 'RISK OFF'}</div>
        <div style="font-size: 12px;">Est. Strategy</div>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# --- Main Tabs ---
tabs = st.tabs(["📊 Market Structure (Vol)", "📈 Deep Technicals", "🔗 Correlations & Z-Score", "⛽ Liquidity & Rotation", "🏛️ Macro Factors"])

# ==========================================
# TAB 1: MARKET STRUCTURE (VOLUME)
# ==========================================
with tabs[0]:
    st.subheader("Volume 7DMA Analysis (True Liquidity)")
    
    col_sel, col_chart = st.columns([1, 4])
    
    with col_sel:
        selected_macro_asset = st.radio("Select Asset", ["bitcoin", "ethereum", "solana", "binancecoin"], index=0)
        st.markdown("**Why 7DMA?**\nDaily volume is noisy. The 7-Day Moving Average shows the true trend of liquidity entering or leaving the asset.")
    
    with col_chart:
        with st.spinner(f"Fetching {selected_macro_asset} volume data..."):
            df_vol = fetch_cg_market_data(selected_macro_asset, cg_days)
        
        if not df_vol.empty:
            fig_vol = go.Figure()
            # Logarithmic Volume Bar
            fig_vol.add_trace(go.Bar(
                x=df_vol.index, y=df_vol['volume'], name='Daily Vol',
                marker_color='rgba(255, 255, 255, 0.1)'
            ))
            # 7DMA Line
            fig_vol.add_trace(go.Scatter(
                x=df_vol.index, y=df_vol['volume_7dma'], name='Volume 7DMA',
                line=dict(color='#00e676', width=3)
            ))
            # Price Overlay
            fig_vol.add_trace(go.Scatter(
                x=df_vol.index, y=df_vol['price'], name='Price',
                yaxis='y2', line=dict(color='#29b6f6', width=2, dash='dot')
            ))
            
            fig_vol.update_layout(
                title=f"<b>{selected_macro_asset.upper()} Liquidity Trend</b>",
                yaxis=dict(title="Volume", type="log"),
                yaxis2=dict(title="Price", overlaying="y", side="right"),
                template="plotly_dark",
                hovermode="x unified",
                height=500,
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_vol, use_container_width=True)

# ==========================================
# TAB 2: DEEP TECHNICALS (Pandas TA)
# ==========================================
with tabs[1]:
    st.subheader("Technical Deep Dive (Powered by Pandas TA)")
    
    # Asset Selector using Yahoo Tickers
    tech_asset_key = st.selectbox("Analyze Asset", list(ASSETS_MAP.keys()), format_func=lambda x: ASSETS_MAP[x])
    ticker = ASSETS_MAP[tech_asset_key]
    
    with st.spinner("Computing Indicators..."):
        df_ta = get_technical_data(ticker, yf_period)
    
    if not df_ta.empty:
        # Create Subplots
        fig_ta = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f"Price & Bollinger Bands", "MACD Momentum", "RSI Strength", "Statistical Z-Score")
        )

        # 1. Price + BBands
        fig_ta.add_trace(go.Candlestick(
            x=df_ta.index, open=df_ta['Open'], high=df_ta['High'], low=df_ta['Low'], close=df_ta['Close'], name='OHLC'
        ), row=1, col=1)
        
        # Dynamically find column names generated by pandas_ta
        # Usually BBU_20_2.0, BBL_20_2.0
        bb_upper = df_ta.columns[df_ta.columns.str.startswith('BBU')][0]
        bb_lower = df_ta.columns[df_ta.columns.str.startswith('BBL')][0]
        
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta[bb_upper], line=dict(color='gray', width=1), showlegend=False), row=1, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta[bb_lower], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)

        # 2. MACD
        # MACD returns 3 columns: MACD_..., MACDh_... (Hist), MACDs_... (Signal)
        macd_col = df_ta.columns[df_ta.columns.str.startswith('MACD_')][0] 
        hist_col = df_ta.columns[df_ta.columns.str.startswith('MACDh_')][0]
        sig_col = df_ta.columns[df_ta.columns.str.startswith('MACDs_')][0]
        
        colors = ['#00e676' if v >= 0 else '#ff5252' for v in df_ta[hist_col]]
        fig_ta.add_trace(go.Bar(x=df_ta.index, y=df_ta[hist_col], marker_color=colors, name='MACD Hist'), row=2, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta[macd_col], line=dict(color='#2979ff', width=1), name='MACD'), row=2, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta[sig_col], line=dict(color='#ff9100', width=1), name='Signal'), row=2, col=1)

        # 3. RSI
        rsi_col = df_ta.columns[df_ta.columns.str.startswith('RSI')][0]
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta[rsi_col], line=dict(color='#d500f9', width=2), name='RSI'), row=3, col=1)
        fig_ta.add_hline(y=70, line_dash="dot", row=3, col=1, line_color="red")
        fig_ta.add_hline(y=30, line_dash="dot", row=3, col=1, line_color="green")

        # 4. Z-Score
        # Column usually named Z_30
        z_col = df_ta.columns[df_ta.columns.str.startswith('ZS')][0]
        z_colors = ['red' if v > 2 else 'green' if v < -2 else 'gray' for v in df_ta[z_col]]
        fig_ta.add_trace(go.Bar(x=df_ta.index, y=df_ta[z_col], marker_color=z_colors, name='Z-Score'), row=4, col=1)
        fig_ta.add_hline(y=2, line_dash="dash", row=4, col=1, line_color="red")
        fig_ta.add_hline(y=-2, line_dash="dash", row=4, col=1, line_color="green")

        fig_ta.update_layout(height=1000, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_ta, use_container_width=True)

# ==========================================
# TAB 3: CORRELATIONS & Z-SCORE
# ==========================================
with tabs[2]:
    st.subheader("🔗 Market Correlation & Relative Value")
    
    # Parallel Fetch
    all_tickers = list(ASSETS_MAP.values())
    with st.spinner("Fetching data for Matrix (Parallel Processing)..."):
        df_corr = fetch_parallel_prices(all_tickers, period="6mo")
    
    if not df_corr.empty:
        col_heat, col_z = st.columns([1, 1])
        
        # 1. Heatmap
        with col_heat:
            corr_matrix = df_corr.corr()
            fig_heat = px.imshow(
                corr_matrix, 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale="RdBu",
                title="Correlation Matrix (Who is moving with BTC?)"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        # 2. Z-Score Deviation (Cheap vs Expensive)
        with col_z:
            # Calculate latest Z-Score for all assets
            # (Current Price - 50d Mean) / 50d StdDev
            z_scores = {}
            for col in df_corr.columns:
                series = df_corr[col].dropna()
                if len(series) > 50:
                    mean = series.rolling(50).mean().iloc[-1]
                    std = series.rolling(50).std().iloc[-1]
                    current = series.iloc[-1]
                    z = (current - mean) / std
                    z_scores[col] = z
            
            df_z = pd.DataFrame.from_dict(z_scores, orient='index', columns=['Z-Score'])
            df_z = df_z.sort_values(by='Z-Score', ascending=False)
            
            fig_z = px.bar(
                df_z, x='Z-Score', y=df_z.index, orientation='h',
                title="Relative Valuation (Z-Score Deviation from 50DMA)",
                color='Z-Score',
                color_continuous_scale="RdYlGn_r" # Red (High) to Green (Low)
            )
            st.plotly_chart(fig_z, use_container_width=True)

# ==========================================
# TAB 4: LIQUIDITY & ROTATION
# ==========================================
with tabs[3]:
    st.subheader("Liquidity Engine & Capital Rotation")
    
    col_liq, col_rot = st.columns(2)
    
    with col_liq:
        st.markdown("**1. Fuel Gauge (USDT Market Cap)**")
        df_usdt = fetch_cg_market_data("tether", cg_days)
        if not df_usdt.empty:
            fig_fuel = go.Figure()
            fig_fuel.add_trace(go.Scatter(x=df_usdt.index, y=df_usdt['market_cap'], fill='tozeroy', line=dict(color='#00C853')))
            fig_fuel.update_layout(height=400, template="plotly_dark", title="Total Stablecoin Liquidity", yaxis_title="Market Cap ($)")
            st.plotly_chart(fig_fuel, use_container_width=True)
            
    with col_rot:
        st.markdown("**2. Altseason Signal (ETH/BTC Ratio)**")
        # Reuse fetch to save time if possible, or fetch separate
        df_btc = fetch_cg_market_data("bitcoin", cg_days)
        df_eth = fetch_cg_market_data("ethereum", cg_days)
        
        if not df_btc.empty and not df_eth.empty:
            # Align
            df_ratio = pd.DataFrame(index=df_btc.index)
            df_eth_re = df_eth.reindex(df_btc.index, method='nearest')
            df_ratio['ratio'] = df_eth_re['price'] / df_btc['price']
            
            fig_rot = go.Figure()
            fig_rot.add_trace(go.Scatter(x=df_ratio.index, y=df_ratio['ratio'], line=dict(color='#651FFF', width=2)))
            fig_rot.update_layout(height=400, template="plotly_dark", title="ETH / BTC Ratio", yaxis_title="Ratio")
            st.plotly_chart(fig_rot, use_container_width=True)

# ==========================================
# TAB 5: MACRO FACTORS
# ==========================================
with tabs[4]:
    st.subheader("Global Macro Environment")
    
    # Fetch Trad Assets Parallel
    trad_tickers = list(TRAD_ASSETS.values())
    df_trad = fetch_parallel_prices(trad_tickers, period=yf_period)
    
    if not df_trad.empty:
        # Create 2x2 grid
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        cols = [c1, c2, c3, c4]
        
        for i, (name, ticker) in enumerate(TRAD_ASSETS.items()):
            # Handle potential ticker mismatches in the returned DF
            if ticker in df_trad.columns:
                with cols[i]:
                    series = df_trad[ticker].dropna()
                    fig_m = px.line(x=series.index, y=series.values, title=name)
                    fig_m.update_layout(template="plotly_dark", height=300, yaxis_title=None, xaxis_title=None)
                    # Color coding
                    if "DXY" in name: fig_m.update_traces(line_color="#EF5350")
                    elif "Gold" in name: fig_m.update_traces(line_color="#FFD700")
                    elif "S&P" in name: fig_m.update_traces(line_color="#42A5F5")
                    st.plotly_chart(fig_m, use_container_width=True)
