# 3. Main Dashboard Logic
if ticker:
    df = get_stock_data(ticker, period) 
    
    if not df.empty:
        # Safety check for volume
        if not df[df['Volume'] > 0].empty:
            df = df[df['Volume'] > 0]
        
        current_price = df['Close'].iloc[-1]
        
        # --- AUTO SENSITIVITY ENGINE ---
        valid_n = []
        for n_scan in range(5, 45, 2): 
            count = count_levels(df, n_scan, current_price)
            if 3 <= count <= 6: valid_n.append(n_scan)
        
        if valid_n:
            optimal_n = int(sum(valid_n) / len(valid_n))
        else:
            optimal_n = 10

        # Sidebar Settings
        st.sidebar.caption("Overlays")
        show_ema = st.sidebar.checkbox("Show EMA (20/50)", value=True)
        show_support = st.sidebar.checkbox("Show Support", value=True)
        show_resistance = st.sidebar.checkbox("Show Resistance", value=True)
        show_bb = st.sidebar.checkbox('Show Bollinger Bands')
        sensitivity = st.sidebar.number_input("Sensitivity", 2, 50, optimal_n)
        
        # Metrics
        df['RSI'] = calculate_rsi(df['Close'])
        current_volatility = df['Volatility'].iloc[-1] * 100 if not np.isnan(df['Volatility'].iloc[-1]) else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"₹{current_price:.2f}", f"{((current_price - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100:.2f}%")
        col2.metric("High", f"₹{df['High'].max():.2f}")
        col3.metric("Low", f"₹{df['Low'].min():.2f}")
        col4.metric("Volatility", f"{current_volatility:.2f}%")

        # --- AI REPORT SECTION ---
        if run_ai:
            with st.spinner(f"Reading news & analyzing charts for {ticker}..."):
                news_headlines = get_stock_news(ticker)
                analysis = get_ai_analysis(ticker, df, news_headlines)
                st.info(f"**AI Analysis:**\n\n{analysis}")
                
                with st.expander("📰 Read the News Headlines Used by AI"):
                    if not news_headlines:
                        st.write("No specific news found.")
                    else:
                        for i, n in enumerate(news_headlines):
                            if isinstance(n, dict) and 'title' in n and 'link' in n:
                                st.markdown(f"### {i+1}. [{n['title']}]({n['link']})")
                                st.caption(f"Published by: {n.get('publisher', 'Unknown')}")
                                st.divider()
                            else:
                                st.write(n)

        # --- PLOTTING LOGIC (Ensuring this is OUTSIDE the AI block) ---
        tab1, tab2 = st.tabs(["📈 Price Action", "📊 Technical Indicators"])
        
        # --- TAB 1: MAIN PRICE CHART ---
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, row_heights=[0.7, 0.3])

            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                             low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            elif chart_type == "Line":
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', 
                                         name="Close", line=dict(color='#00e676')), row=1, col=1)
            elif chart_type == "Area":
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], fill='tozeroy', mode='lines', 
                                         name="Close", line=dict(color='#2962ff')), row=1, col=1)
            elif chart_type == "OHLC":
                fig.add_trace(go.Ohlc(x=df.index, open=df['Open'], high=df['High'], 
                                      low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            if show_ema:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#ff9100', width=1), name='EMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#2962ff', width=1), name='EMA 50'), row=1, col=1)

            if show_bb and 'Upper_Band' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='rgba(255, 255, 255, 0.1)'), name='Upper Band'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='rgba(255, 255, 255, 0.1)'), name='Lower Band', fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)'), row=1, col=1)

            if show_support or show_resistance:
                 levels = []
                 n = int(sensitivity)
                 for i in range(n, df.shape[0]-n):
                     if show_support and is_support(df, i, n):
                         l = df['Low'][i]
                         if np.sum([abs(l - x[1]) < (current_price*0.02) for x in levels]) == 0:
                             levels.append((df.index[i], l, "Support"))
                     elif show_resistance and is_resistance(df, i, n):
                         l = df['High'][i]
                         if np.sum([abs(l - x[1]) < (current_price*0.02) for x in levels]) == 0:
                             levels.append((df.index[i], l, "Resistance"))
                 for date, level, kind in levels:
                     color = "green" if kind == "Support" else "red"
                     fig.add_hline(y=level, line_dash="dot", line_color=color, row=1, col=1, opacity=0.5)

            colors = ['#00e676' if r['Open'] - r['Close'] <= 0 else '#ff1744' for i, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Momentum & Strength")
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5])
            
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#b388ff', width=2), name='RSI'), row=1, col=1)
            fig_macd.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig_macd.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            fig_macd.update_yaxes(title_text="RSI", row=1, col=1)
            
            if 'MACD' in df.columns:
                colors_macd = ['#00e676' if val >= 0 else '#ff1744' for val in df['MACD'] - df['Signal_Line']]
                fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal_Line'], marker_color=colors_macd, name='MACD Hist'), row=2, col=1)
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2962ff', width=1), name='MACD'), row=2, col=1)
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='#ff9100', width=1), name='Signal'), row=2, col=1)
                fig_macd.update_yaxes(title_text="MACD", row=2, col=1)

            fig_macd.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.error("Error loading data.")