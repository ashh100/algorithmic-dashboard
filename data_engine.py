import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import xml.etree.ElementTree as ET
from nsepython import nse_eq
from nselib import capital_market
from bs4 import BeautifulSoup
import database as db
# --- 1. NEW HELPER FUNCTION (Paste this above your main function) ---
def fetch_from_screener(symbol):
    """
    Fallback: Scrapes Screener.in if Yahoo fails (Good for Zomato/Swiggy)
    """
    try:
        # Screener uses pure symbols (ZOMATO, not ZOMATO.NS)
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
        url = f"https://www.screener.in/company/{clean_symbol}/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        ratios = soup.find('ul', {'id': 'top-ratios'})
        
        if not ratios:
            return None

        data = {}
        for li in ratios.find_all('li'):
            name = li.find('span', {'class': 'name'}).text.strip()
            val_span = li.find('span', {'class': 'number'})
            if val_span:
                # Remove commas (1,234 -> 1234)
                val_text = val_span.text.replace(',', '').strip()
                try:
                    val = float(val_text)
                    if "Market Cap" in name:
                        # Screener is in Cr, Dashboard likely expects raw or handles Cr
                        # Let's return raw for safety (Cr * 10^7)
                        data['marketCap'] = val * 10000000 
                    elif "Stock P/E" in name:
                        data['trailingPE'] = val
                    elif "Dividend Yield" in name:
                        data['dividendYield'] = val / 100 # Convert 0.35 to 0.0035
                    elif "Current Price" in name:
                        data['currentPrice'] = val
                except:
                    pass
        return data
    except:
        return None
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
                symbol = quote.get('symbol', '')
                if symbol.endswith('.NS') or symbol.endswith('.BO'):
                    shortname = quote.get('shortname', symbol)
                    label = f"{symbol} - {shortname}"
                    search_results[label] = symbol
        return search_results
    except:
        return {}
def get_nse_fundamentals(ticker):
    try:
        # --- YOUR EXISTING CODE STARTS HERE ---
        # 1. SUFFIX FIX 
        ticker = ticker.strip()
        
        if ticker.isdigit():
            yf_ticker_name = f"{ticker}.BO"
            symbol = ticker 
        elif not ticker.endswith(".NS") and not ticker.endswith(".BO"):
            yf_ticker_name = f"{ticker}.NS"
            symbol = ticker
        else:
            yf_ticker_name = ticker
            symbol = ticker.replace(".NS", "").replace(".BO", "")

        base_data = {
            "sector": "N/A",
            "marketCap": 0,
            "trailingPE": "N/A",
            "dividendYield": 0.0,
            "currentPrice": 0.0
        }

        # 2. FETCH FROM YFINANCE
        yf_obj = yf.Ticker(yf_ticker_name)
        
        # Market Cap
        try:
            mcap = yf_obj.fast_info.market_cap
            if mcap:
                base_data['marketCap'] = mcap
        except:
            pass

        # Fundamentals
        try:
            info = yf_obj.info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            base_data['currentPrice'] = price
            base_data['sector'] = info.get("sector", "N/A")

            pe = info.get("trailingPE")
            if pe is None:
                pe = info.get("forwardPE")
            if pe is not None:
                base_data['trailingPE'] = pe
            
            dy = info.get("dividendYield")
            if dy is None:
                dy = info.get("trailingAnnualDividendYield")
            if dy is not None:
                base_data['dividendYield'] = dy

        except Exception as e:
            print(f"YF Data Error: {e}")

        # NSE Fallback (Your existing sector backup)
        if base_data['sector'] == "N/A" and not ticker.isdigit():
            try:
                from nselib import capital_market
                data = capital_market.equity_list()
                row = data[data['SYMBOL'] == symbol]
                if not row.empty:
                    for col in row.columns:
                        if "SECTOR" in col.upper() or "INDUSTRY" in col.upper():
                            base_data['sector'] = row.iloc[0][col]
                            break
            except:
                pass
        # --- YOUR EXISTING CODE ENDS HERE ---

        # --- 3. THE NEW "SCREENER" BACKUP BLOCK ---
        # Only runs if Yahoo failed to get PE or Market Cap (Common for Swiggy/Zomato)
        # --- 3. THE NEW "SCREENER" BACKUP BLOCK ---
        # Runs if Yahoo failed to get PE, Market Cap, OR Dividend Yield
        if (base_data['trailingPE'] == "N/A" or 
            base_data['marketCap'] == 0 or 
            base_data['dividendYield'] == 0.0):
            
            screener_data = fetch_from_screener(symbol)
            
            if screener_data:
                # Update Market Cap only if Yahoo missed it
                if screener_data.get('marketCap') and base_data['marketCap'] == 0: 
                    base_data['marketCap'] = screener_data['marketCap']
                
                # Update PE only if Yahoo missed it
                if screener_data.get('trailingPE') and base_data['trailingPE'] == "N/A": 
                    base_data['trailingPE'] = screener_data['trailingPE']
                
                # Update Dividend only if Yahoo missed it (Crucial for your glitch!)
                if screener_data.get('dividendYield') and base_data['dividendYield'] == 0.0: 
                    # Divide by 100 to match Yahoo's decimal format (e.g. 1.5 -> 0.015)
                    base_data['dividendYield'] = screener_data['dividendYield'] / 100
                
                # Update Price only if Yahoo missed it
                if screener_data.get('currentPrice') and base_data['currentPrice'] == 0: 
                    base_data['currentPrice'] = screener_data['currentPrice']

        return base_data

    except Exception as e:
        return {"sector": "N/A", "marketCap": 0, "trailingPE": "N/A", "dividendYield": 0.0}
