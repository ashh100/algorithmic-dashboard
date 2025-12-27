import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import xml.etree.ElementTree as ET

# 1. Page Setup
st.set_page_config(layout="wide", page_title="Ashwath's Pro Terminal V2.0")
st.title("⚡ Ashwath's Algorithmic Terminal V2.0")

# --- SESSION STATE INITIALIZATION ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
if 'show_ai' not in st.session_state:
    st.session_state.show_ai = False

# --- UTILITY FUNCTIONS ---
def add_to_watchlist(ticker):
    if ticker and ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)

def remove_from_watchlist(ticker):
    if ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(ticker)

def run_ai_analysis():
    st.session_state.show_ai = True

def go_back():
    st.session_state.show_ai = False

# --- PATTERN RECOGNITION (Custom Implementation) ---
def identify_patterns(df):
    # Doji: Open and Close are very close
    df['Doji'] = abs(df['Open'] - df['Close']) <= (df['High'] - df['Low']) * 0.1
    # Hammer: Small body, long lower wick, short upper wick
    body = abs(df['Open'] - df['Close'])
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Hammer'] = (lower_wick > 2 * body) & (upper_wick < body)
    return df

# --- BACKTESTING ENGINE (Vectorized) ---
def run_backtest(df, fast_ma, slow_ma, initial_capital=100000):
    df['Fast_MA'] = df['Close'].rolling(window=fast_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(window=slow_ma).mean()
    
    # Logic: Buy when Fast crosses above Slow, Sell when Fast crosses below Slow
    df['Signal'] = 0
    df['Signal'][fast_ma:] = np.where(df['Fast_MA'][fast_ma:] > df['Slow_MA'][fast_ma:], 1, 0)
    df['Position'] = df['Signal'].diff()
    
    # Calculate Returns
    df['Market_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Market_Returns'] * df['Signal'].shift(1)
    
    df['Portfolio_Value'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    
    total_return = df['Portfolio_Value'].iloc[-1] - initial_capital
    return df, total_return

# --- EXISTING ANALYTICS FUNCTIONS ---
def search_tickers(query):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        search_results = {}
        if 'quotes' in data:
            for quote in data['quotes']:
                if 'shortname' in quote and 'symbol' in quote:
                    label = f"{quote['symbol']} - {quote['shortname']}"
                    search_results[label] = quote['symbol']
        return search_results
    except:
        return {}

def is_support(df, i, n):
    for k in range(1, n+1):
        if i-k < 0 or i+k >= len(df): continue
        if df['Low'][i] >= df['Low'][i-k] or df['Low'][i] >= df['Low'][i+k]:
            return False
    return True

def is_resistance(df, i, n):
    for k in range(1, n+1): 
        if i-k < 0 or i+k >= len(df): continue
        if df['High'][i] <= df['High'][i-k] or df['High'][i] <= df['High'][i+k]:
            return False
    return True

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def count_levels(df, n, current_price):
    levels = []
    for i in range(n, df.shape[0]-n):
        if is_support(df, i, n):
            l = df['Low'][i]
            if np.sum([abs(l - x) < (current_price*0.02) for x in levels]) == 0:
                levels.append(l)
        elif is_resistance(df, i, n):
            l = df['High'][i]
            if np.sum([abs(l - x) < (current_price*0.02) for x in levels]) == 0:
                levels.append(l)
    return len(levels)

@st.cache_data(ttl=3600)
def get_stock_news(ticker):
    try:
        search_term = ticker.split('.')[0] 
        url = f"https://news.google.com/rss/search?q={search_term}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall('.//item')[:5]:
            title = item.find('title').text
            link = item.find('link').text
            source = item.find('source')
            publisher = source.text if source is not None else "Google News"
            news_items.append({'title': title, 'link': link, 'publisher': publisher})
        return news_items
    except Exception as e:
        return []

def get_ai_analysis(ticker, data, news_list):
    recent_data = data.tail(10).to_string()
    current_price = data['Close'].iloc[-1]
    
    if news_list:
        formatted_news = "\n".join([f"- {item['title']} ({item['publisher']})" for item in news_list])
    else:
        formatted_news = "No recent news."

    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
        else:
            return "⚠️ Error: GROQ_API_KEY missing."
    except:
        return "⚠️ Error: Secrets not loaded."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"Analyze {ticker}. Price: {current_price}. Data: {recent_data}. News: {formatted_news}. Give Trend, Reason, and Action."
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_data(ttl=300) 
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty: return df
    
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Pct_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Pct_Change'].rolling(window=20).std()
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])

    # Patterns
    df = identify_patterns(df)
    
    return df

@st.cache_data(ttl=3600)
def get_sector_correlation(ticker, period="1y"):
    # Compare with Nifty 50 (NSEI)
    try:
        tickers = [ticker, "^NSEI"]
        data = yf.download(tickers, period=period)['Close']
        correlation = data.corr().iloc[0, 1]
        return correlation, data
    except:
        return 0, None

# --- SIDEBAR: WATCHLIST & SEARCH ---
st.sidebar.header("🔍 Watchlist & Search")

# Watchlist Display
st.sidebar.subheader("My Portfolio")
for stock in st.session_state.watchlist:
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    if col1.button(stock, key=stock):
        st.session_state.selected_ticker = stock
    if col2.button("✖", key=f"del_{stock}"):
        remove_from_watchlist(stock)
        st.rerun()

# Search Input
search_query = st.sidebar.text_input("Add Stock", placeholder="e.g. Tata Motors")
if st.sidebar.button("Add to Watchlist"):
    results = search_tickers(search_query)
    if results:
        top_result = list(results.values())[0]
        add_to_watchlist(top_result)
        st.sidebar.success(f"Added {top_result}")
        st.rerun()

# Main Ticker Selection Logic
if 'selected_ticker' in st.session_state:
    ticker = st.session_state.selected_ticker
else:
    ticker = "RELIANCE.NS" # Default

st.sidebar.markdown("---")
period = st.sidebar.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
chart_type = st.sidebar.selectbox("Chart Style", ["Candlestick", "Line", "Area"])

# --- MAIN DASHBOARD LOGIC ---
if ticker:
    st.header(f"Analysis: {ticker}")
    df = get_stock_data(ticker, period) 
    
    if not df.empty:
        # Filter Volume
        if not df[df['Volume'] > 0].empty: df = df[df['Volume'] > 0]
        current_price = df['Close'].iloc[-1]
        
        # --- TOP METRICS & ALERTS PANEL ---
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        # Metrics
        col_m1.metric("Price", f"₹{current_price:.2f}", f"{df['Pct_Change'].iloc[-1]*100:.2f}%")
        col_m2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
        
        # Sector Correlation
        corr, corr_data = get_sector_correlation(ticker)
        col_m3.metric("Nifty Corr.", f"{corr:.2f}")

        # Live Signals (Logic)
        signal = "NEUTRAL"
        color = "off"
        if df['RSI'].iloc[-1] < 30: 
            signal = "OVERSOLD (BUY?)"
            color = "inverse" # Green-ish
        elif df['RSI'].iloc[-1] > 70: 
            signal = "OVERBOUGHT (SELL?)"
            color = "normal" # Red-ish warning
        
        col_m4.error(f"Signal: {signal}") if "SELL" in signal else col_m4.success(f"Signal: {signal}") if "BUY" in signal else col_m4.info("Signal: NEUTRAL")

        # --- AI & BACK BUTTON ---
        if st.sidebar.button("🤖 AI Analysis"):
            run_ai_analysis()
            
        if st.session_state.show_ai:
            if st.button("← Back to Charts"):
                go_back()
                st.rerun()
            with st.spinner("AI analyzing..."):
                news = get_stock_news(ticker)
                analysis = get_ai_analysis(ticker, df, news)
                st.info(analysis)

        # --- TABS FOR ANALYSIS ---
        tab_chart, tab_tech, tab_backtest = st.tabs(["📈 Pro Chart", "📊 Indicators", "🛠 Strategy Tester"])

        # 1. MAIN CHART
        with tab_chart:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            
            # Price Trace
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Price"), row=1, col=1)

            # Patterns (Markers)
            doji_dates = df[df['Doji']].index
            if len(doji_dates) > 0:
                fig.add_trace(go.Scatter(x=doji_dates, y=df.loc[doji_dates]['High'], mode='markers', marker=dict(symbol='diamond', size=5, color='yellow'), name="Doji"), row=1, col=1)
            
            hammer_dates = df[df['Hammer']].index
            if len(hammer_dates) > 0:
                fig.add_trace(go.Scatter(x=hammer_dates, y=df.loc[hammer_dates]['Low'], mode='markers', marker=dict(symbol='triangle-up', size=8, color='purple'), name="Hammer"), row=1, col=1)

            # Volume
            colors = ['green' if o-c <= 0 else 'red' for o, c in zip(df['Open'], df['Close'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
            
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # 2. INDICATORS TAB
        with tab_tech:
            st.subheader("Deep Dive Indicators")
            fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True)
            
            # RSI
            fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=1, col=1)
            fig2.add_hline(y=70, row=1, col=1, line_dash="dash", line_color="red")
            fig2.add_hline(y=30, row=1, col=1, line_dash="dash", line_color="green")
            
            # MACD
            fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"), row=2, col=1)
            fig2.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name="Signal"), row=2, col=1)
            fig2.add_trace(go.Bar(x=df.index, y=df['MACD']-df['Signal_Line'], name="Hist"), row=2, col=1)
            
            # Volatility
            fig2.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name="Volatility", line=dict(color='orange')), row=3, col=1)
            
            fig2.update_layout(height=800, template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

        # 3. BACKTEST TAB
        with tab_backtest:
            st.subheader("Strategy Lab: SMA Crossover")
            col_b1, col_b2 = st.columns(2)
            fast_ma = col_b1.number_input("Fast MA", 5, 50, 20)
            slow_ma = col_b2.number_input("Slow MA", 20, 200, 50)
            
            if st.button("Run Backtest"):
                res_df, total_ret = run_backtest(df.copy(), fast_ma, slow_ma)
                
                st.metric("Total Return", f"₹{total_ret:.2f}", f"{(total_ret/100000)*100:.1f}%")
                
                # Plot Equity Curve
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=res_df.index, y=res_df['Portfolio_Value'], fill='tozeroy', name="Portfolio Value"))
                fig_bt.update_layout(title="Equity Curve", template="plotly_dark")
                st.plotly_chart(fig_bt, use_container_width=True)

    else:
        st.error("Could not load data for this ticker.")
else:
    st.info("Select a stock from the Watchlist or Search to begin.")