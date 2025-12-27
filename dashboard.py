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
st.title("Algorithmic Dashboard")

# --- SESSION STATE ---
if 'show_ai' not in st.session_state:
    st.session_state.show_ai = False

def run_ai_analysis():
    st.session_state.show_ai = True

def go_back():
    st.session_state.show_ai = False

# --- UTILITY FUNCTIONS ---
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

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- NEW FUNCTION: MARKET COMPARISON ---
@st.cache_data(ttl=3600)
def get_market_comparison(ticker, period):
    try:
        # Download both the stock and the Nifty 50 index
        tickers = [ticker, "^NSEI"]
        data = yf.download(tickers, period=period)['Close']
        
        # Calculate Percentage Change (Normalized Return)
        # Formula: (Price / Start_Price - 1) * 100
        normalized = (data / data.iloc[0] - 1) * 100
        
        # Calculate Correlation (0 to 1)
        correlation = data.corr().iloc[0, 1]
        
        return normalized, correlation
    except Exception as e:
        return None, 0

# --- NEWS FUNCTION ---
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

# --- AI ANALYST FUNCTION ---
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
            return "⚠️ Error: GROQ_API_KEY missing in .streamlit/secrets.toml"
    except Exception:
        return "⚠️ Error: Could not load secrets."

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
        return f"Connection Error: {str(e)}"

# --- DATA FETCHING ---
@st.cache_data(ttl=300) 
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty: return df
    
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
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
    
    return df

@st.cache_data(ttl=86400) 
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except:
        return None

# --- SIDEBAR ---
st.sidebar.header("Controls")
search_query = st.sidebar.text_input("Search Company", "Tata")

if search_query:
    results = search_tickers(search_query)
    if results:
        selected_label = st.sidebar.selectbox("Select Stock", options=results.keys())
        ticker = results[selected_label]
    else:
        st.sidebar.error("No stocks found.")
        ticker = None
else:
    ticker = None

period = st.sidebar.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

if ticker:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏢 Fundamentals")
    info = get_company_info(ticker)
    if info:
        st.sidebar.metric("Market Cap", f"₹{info.get('marketCap', 0)/1e9:.2f}B")
        st.sidebar.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Chart Display")
    chart_type = st.sidebar.selectbox("Chart Style", ["Candlestick", "Line", "Area"])
    
    st.sidebar.markdown("---")
    st.sidebar.button("Generate AI Report", on_click=run_ai_analysis)

# --- MAIN DASHBOARD ---
if ticker:
    df = get_stock_data(ticker, period) 
    if not df.empty:
        current_price = df['Close'].iloc[-1]
        
        # --- AI SECTION ---
        if st.session_state.show_ai:
            if st.button("← Back to Dashboard"):
                go_back()
                st.rerun()
            with st.spinner("Analyzing..."):
                news = get_stock_news(ticker)
                analysis = get_ai_analysis(ticker, df, news)
                st.info(analysis)
        
        # --- CHARTS ---
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"₹{current_price:.2f}")
            col2.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            col3.metric("High", f"₹{df['High'].max():.2f}")

            # --- UPDATE: Added "Sector Comparison" as the 3rd tab ---
            tab1, tab2, tab3 = st.tabs(["📈 Price Action", "📊 Indicators", "⚖️ Sector Comparison"])
            
            # Tab 1: Price Action
            with tab1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                if chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Price"), row=1, col=1)
                
                # Overlays
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='rgba(255,255,255,0.1)'), name='Upper Band'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='rgba(255,255,255,0.1)'), fill='tonexty', name='Lower Band'), row=1, col=1)
                
                # Volume
                colors = ['green' if o-c <= 0 else 'red' for o, c in zip(df['Open'], df['Close'])]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            # Tab 2: Technicals
            with tab2:
                fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True)
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=1, col=1)
                fig_macd.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                fig_macd.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=2, col=1)
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal'), row=2, col=1)
                fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD']-df['Signal_Line'], name='Hist'), row=2, col=1)
                
                fig_macd.update_layout(height=500, template="plotly_dark")
                st.plotly_chart(fig_macd, use_container_width=True)

            # Tab 3: Sector Comparison (New!)
            with tab3:
                st.subheader(f"Performance: {ticker} vs Nifty 50")
                comp_data, correlation = get_market_comparison(ticker, period)
                
                if comp_data is not None:
                    # Display Correlation Metric
                    c1, c2 = st.columns([1, 3])
                    c1.metric("Correlation with Market", f"{correlation:.2f}")
                    if correlation > 0.8:
                        c1.info("Moves with Market")
                    elif correlation < 0.3:
                        c1.warning("Moves Independently")
                    
                    # Plot Comparison Chart
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(x=comp_data.index, y=comp_data[ticker], name=ticker, line=dict(color='cyan', width=2)))
                    fig_comp.add_trace(go.Scatter(x=comp_data.index, y=comp_data['^NSEI'], name='Nifty 50', line=dict(color='gray', width=2, dash='dash')))
                    
                    fig_comp.update_layout(
                        title="Relative Return (%)", 
                        template="plotly_dark", 
                        yaxis_title="Percentage Change",
                        height=500
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.warning("Could not fetch Nifty 50 data for comparison.")

    else:
        st.error("Error loading data.")