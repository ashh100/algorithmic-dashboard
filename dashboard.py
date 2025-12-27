import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import xml.etree.ElementTree as ET

# 1. Page Setup
st.set_page_config(layout="wide", page_title="Ashwath's Pro Terminal")
st.title("⚡ Ashwath's Market Terminal")

# --- SESSION STATE ---
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

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
            return "⚠️ Error: GROQ_API_KEY missing in secrets."
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
    
    # Basic Indicators
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    
    return df

# --- SIDEBAR ---
st.sidebar.header("📁 Watchlist")

# 1. Watchlist Buttons
for stock in st.session_state.watchlist:
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    # Clicking the name sets the ticker
    if col1.button(stock, key=stock, use_container_width=True):
        st.session_state.selected_ticker = stock
    # Clicking X removes it
    if col2.button("✖", key=f"del_{stock}"):
        remove_from_watchlist(stock)
        st.rerun()

st.sidebar.markdown("---")
new_ticker = st.sidebar.text_input("Add Ticker", placeholder="e.g. TATAMOTORS.NS")
if st.sidebar.button("Add to List"):
    add_to_watchlist(new_ticker.upper())
    st.rerun()

# 2. Settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
period = st.sidebar.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
chart_style = st.sidebar.selectbox("Chart Style", ["Candlestick", "Line"])

# --- MAIN PAGE LOGIC ---
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = "RELIANCE.NS"

ticker = st.session_state.selected_ticker

# Fetch Data
df = get_stock_data(ticker, period)

if not df.empty:
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = ((current_price - prev_close)/prev_close)*100
    
    # --- HEADER SECTION (Clean Metrics) ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label=ticker, value=f"₹{current_price:.2f}", delta=f"{change:.2f}%")
    col2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
    col3.metric("Volume", f"{df['Volume'].iloc[-1]/1e6:.2f}M")
    
    # Signal Box (Fixed logic)
    rsi_val = df['RSI'].iloc[-1]
    if rsi_val < 30:
        col4.success("Signal: OVERSOLD (Buy?)")
    elif rsi_val > 70:
        col4.error("Signal: OVERBOUGHT (Sell?)")
    else:
        col4.info("Signal: NEUTRAL")

    # --- AI SECTION (Expandable) ---
    st.markdown("---")
    # Toggle button
    if st.button(f"🤖 Ask AI Analyst about {ticker}"):
        run_ai_analysis()
    
    if st.session_state.show_ai:
        if st.button("✖ Close Report"):
            go_back()
            st.rerun()
            
        with st.container():
            st.info("Generating analysis...")
            news = get_stock_news(ticker)
            analysis = get_ai_analysis(ticker, df, news)
            st.success(f"**AI Report:**\n\n{analysis}")

    # --- TABS (Organized Views) ---
    st.markdown("### Market Data")
    tab1, tab2 = st.tabs(["📈 Price Chart", "🛠 Strategy Tester"])
    
    with tab1:
        # Simple, Clean Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.1)
        
        if chart_style == "Candlestick":
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], fill='tozeroy', name="Price"), row=1, col=1)
            
        # Add EMA
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1), name="EMA 20"), row=1, col=1)
        
        # Volume
        colors = ['red' if row['Open'] - row['Close'] > 0 else 'green' for i, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("### Simple Moving Average Strategy")
        c1, c2 = st.columns(2)
        short_window = c1.number_input("Short Window", 10, 50, 20)
        long_window = c2.number_input("Long Window", 50, 200, 50)
        
        if st.button("Run Backtest"):
            # Simple vector backtest
            signals = pd.DataFrame(index=df.index)
            signals['signal'] = 0.0
            signals['short_mavg'] = df['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
            signals['long_mavg'] = df['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
            signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
            signals['positions'] = signals['signal'].diff()
            
            # Plot
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='gray', width=1)))
            fig_bt.add_trace(go.Scatter(x=signals.index, y=signals['short_mavg'], name="Short MA", line=dict(color='orange')))
            fig_bt.add_trace(go.Scatter(x=signals.index, y=signals['long_mavg'], name="Long MA", line=dict(color='blue')))
            
            # Buy/Sell markers
            buy_signals = signals.loc[signals.positions == 1.0]
            sell_signals = signals.loc[signals.positions == -1.0]
            
            fig_bt.add_trace(go.Scatter(x=buy_signals.index, y=df.loc[buy_signals.index]['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name="Buy"))
            fig_bt.add_trace(go.Scatter(x=sell_signals.index, y=df.loc[sell_signals.index]['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name="Sell"))
            
            fig_bt.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_bt, use_container_width=True)

else:
    st.error("No data found. Please check the ticker symbol.")