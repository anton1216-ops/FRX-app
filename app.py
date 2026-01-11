
import streamlit as st
from streamlit_autorefresh import st_autorefresh
# --- Trade Signal Diary (in-memory for session) ---
import datetime
if 'trade_diary' not in st.session_state:
    st.session_state['trade_diary'] = []

# --- Pushbullet Notification Function ---
import requests
def send_push(title, body, api_key=None):
    if not api_key:
        return
    try:
        msg = {"type": "note", "title": title, "body": body}
        resp = requests.post('https://api.pushbullet.com/v2/pushes',
                            data=msg,
                            headers={'Access-Token': api_key})
        if resp.status_code != 200:
            st.warning(f"Pushbullet notification failed: {resp.text}")
    except Exception as e:
        st.warning(f"Pushbullet notification error: {e}")

# --- Imports ---
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import logging

# --- EMA/SMA/ATR Lengths (global defaults, adjustable later) ---
ema_fast_len = 9
ema_slow_len = 21
sma_len = 200
atr_len = 14
rsi_thresh = 50

# --- Logging ---
logging.basicConfig(filename='fx_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- Page Setup ---
st.set_page_config(page_title="Forex Confluence Workstation", layout="wide")
st.title("📊 Forex Confluence Workstation")

# --- Sidebar: Account & Risk ---


 # --- Auto-refresh interval ---
st.sidebar.header("App Auto-Refresh")
refresh_minutes = st.sidebar.slider("Refresh every (minutes)", 0, 30, 5, help="Set to 0 to disable auto-refresh.")
if refresh_minutes > 0:
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="autorefresh")

# --- Pushbullet API Key Persistence ---
import os
API_KEY_FILE = "pushbullet_api_key.txt"
if 'push_api_key' not in st.session_state:
    # Try to load from file
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as f:
            st.session_state['push_api_key'] = f.read().strip()
    else:
        st.session_state['push_api_key'] = ""

st.sidebar.header("Push Notifications")
push_enabled = st.sidebar.checkbox("Enable Push Alerts", value=False)
push_api_key = st.sidebar.text_input("Pushbullet API Key", type="password", value=st.session_state['push_api_key'])

# Save API key if changed
if push_api_key != st.session_state['push_api_key']:
    st.session_state['push_api_key'] = push_api_key
    with open(API_KEY_FILE, "w") as f:
        f.write(push_api_key)

# --- Test Pushbullet Notification Button ---
if st.sidebar.button("Send Test Notification"):
    if push_api_key:
        send_push("Test Notification", "This is a test from your Forex app.", api_key=push_api_key)
        st.sidebar.success("Test notification sent! Check your phone.")
    else:
        st.sidebar.error("Please enter your Pushbullet API key first.")


# --- Data/Signal Functions (must be defined before sidebar uses them) ---
@st.cache_data(ttl=300)
def get_data(pair, period):
    try:
        for interval in ["1h", "1d"]:
            df = yf.download(pair, period=period, interval=interval, multi_level_index=False)
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                df['ema_fast'] = ta.ema(df['close'], length=ema_fast_len)
                df['ema_slow'] = ta.ema(df['close'], length=ema_slow_len)
                df['sma_trend'] = ta.sma(df['close'], length=sma_len)
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
                # Always calculate RSI for confluence logic
                df['rsi'] = ta.rsi(df['close'], length=14)
                # Always calculate MACD
                macd = ta.macd(df['close'])
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                return df.dropna()
            else:
                logging.warning(f"No data for {pair} {period} {interval}")
        st.warning(f"No data returned from Yahoo Finance for {pair} ({period}). Tried 1h and 1d intervals.")
        return None
    except Exception as e:
        st.error(f"Data download failed: {e}")
        logging.error(f"Data download failed: {e}")
        return None

def analyze_signal(row):
    ema_buy = row['ema_fast'] > row['ema_slow']
    trend_up = row['close'] > row['sma_trend']
    ema_sell = row['ema_fast'] < row['ema_slow']
    trend_down = row['close'] < row['sma_trend']
    # --- STRICT FILTERS ---
    rsi_ok = ('rsi' in row and row['rsi'] > rsi_thresh)
    macd_ok = ('macd' in row and 'macd_signal' in row and row['macd'] > row['macd_signal'])
    buy = ema_buy and trend_up and rsi_ok and macd_ok
    sell = ema_sell and trend_down and rsi_ok and ('macd' in row and 'macd_signal' in row and row['macd'] < row['macd_signal'])
    return buy, sell, trend_up

# --- Sidebar: Categories Met for Each Currency Pair ---
st.sidebar.header("Signal Status (Backtest Logic)")
pairs = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"]
for pair in pairs:
    data = get_data(pair, "60d")
    status = "NO SIGNAL"
    if data is not None and len(data) > 0:
        last_row = data.iloc[-1]
        buy_signal, sell_signal, _ = analyze_signal(last_row)
        if buy_signal:
            status = "BUY"
        elif sell_signal:
            status = "SELL"
    st.sidebar.markdown(f"**{pair.replace('=X','')}: {status}**")



# --- Sidebar: Account & Risk ---

st.sidebar.header("Account & Risk")
start_bal = st.sidebar.number_input("Account Balance ($)", value=1000.0, min_value=0.0, key="account_balance")
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 3.0, key="risk_per_trade")
rr_ratio = st.sidebar.slider("Risk to Reward Ratio (1:X)", 1.0, 5.0, 1.66, key="risk_to_reward")

# --- Sidebar: TP & SL Calculator ---
st.sidebar.header("TP & SL Calculator")
entry_price = st.sidebar.number_input("Entry Price", value=0.0, min_value=0.0, format="%.5f", key="entry_price")
stop_loss_pips = st.sidebar.number_input("Stop Loss (pips)", value=20, min_value=1, key="stop_loss_pips")
risk_amount = start_bal * (risk_pct / 100)
lot_size = risk_amount / (stop_loss_pips * 10) if stop_loss_pips else 0
take_profit_pips = st.sidebar.number_input("Take Profit (pips)", value=float(stop_loss_pips*rr_ratio) if stop_loss_pips else 40.0, min_value=1.0, key="take_profit_pips")
if entry_price > 0 and stop_loss_pips > 0 and take_profit_pips > 0:
    sl_buy = entry_price - (stop_loss_pips * 0.0001)
    tp_buy = entry_price + (take_profit_pips * 0.0001)
    sl_sell = entry_price + (stop_loss_pips * 0.0001)
    tp_sell = entry_price - (take_profit_pips * 0.0001)
    st.sidebar.info(f"Buy: SL={sl_buy:.5f}, TP={tp_buy:.5f}")
    st.sidebar.info(f"Sell: SL={sl_sell:.5f}, TP={tp_sell:.5f}")
    st.sidebar.success(f"Lot Size: {lot_size:.2f}")
    st.sidebar.success(f"Risk Amount: ${risk_amount:.2f}")

# --- Chart plotting for this pair ---
advice = "NO SIGNAL"
buy_signal, sell_signal, trend_up = analyze_signal(last_row)
if buy_signal:
    advice = "BUY"
elif sell_signal:
    advice = "SELL"
c1, c2, c3 = st.columns(3)
if advice == "BUY":
    c1.success("### ✅ SIGNAL: BUY")
    if entry_price > 0 and stop_loss_pips > 0 and take_profit_pips > 0:
        c1.info(f"SL: {sl_buy:.5f} | TP: {tp_buy:.5f} | RR: {rr_ratio:.2f}")
elif advice == "SELL":
    c1.error("### ✅ SIGNAL: SELL")
    if entry_price > 0 and stop_loss_pips > 0 and take_profit_pips > 0:
        c1.info(f"SL: {sl_sell:.5f} | TP: {tp_sell:.5f} | RR: {rr_ratio:.2f}")
else:
    c1.warning("### ❌ STATUS: NO SIGNAL")

c2.metric("Current Price", round(last_row['close'], 5))
c3.metric("Trend (SMA)", "UP" if trend_up else "DOWN")

# Clean data for plotting
plot_data = data[['open', 'high', 'low', 'close']].dropna()
fig = go.Figure()
fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['open'], high=plot_data['high'], low=plot_data['low'], close=plot_data['close'], name="Price"))

# Auto-zoom y-axis to price range
price_min = plot_data['low'].min()
price_max = plot_data['high'].max()
y_margin = (price_max - price_min) * 0.05 if price_max > price_min else 0.001
fig.update_yaxes(range=[price_min - y_margin, price_max + y_margin])
fig.add_trace(go.Scatter(x=data.index, y=data['ema_fast'], name="EMA Fast", line=dict(color='#00d1ff')))
fig.add_trace(go.Scatter(x=data.index, y=data['ema_slow'], name="EMA Slow", line=dict(color='#ff9900')))
fig.add_trace(go.Scatter(x=data.index, y=data['sma_trend'], name="SMA Trend", line=dict(color='white', dash='dash')))

# --- Add Buy/Sell Signal Markers ---
buy_signals = []
sell_signals = []
for i, row in data.iterrows():
    buy, sell, _ = analyze_signal(row)
    if buy:
        buy_signals.append((i, row['low']))
    if sell:
        sell_signals.append((i, row['high']))
if buy_signals:
    buy_x, buy_y = zip(*buy_signals)
    fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', marker=dict(color='lime', size=12, symbol='triangle-up'), name='Buy Signal'))
if sell_signals:
    sell_x, sell_y = zip(*sell_signals)
    fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'), name='Sell Signal'))

if 'rsi' in data.columns:
    fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], name="RSI", line=dict(color='green')))
if 'macd' in data.columns:
    fig.add_trace(go.Scatter(x=data.index, y=data['macd'], name="MACD", line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name="MACD Signal", line=dict(color='orange')))
fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=True,
    height=900,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)
st.plotly_chart(fig, use_container_width=True)
st.sidebar.header("Account & Risk")
start_bal = st.sidebar.number_input("Account Balance ($)", value=1000.0, min_value=0.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 3.0)
rr_ratio = st.sidebar.slider("Risk to Reward Ratio (1:X)", 1.0, 5.0, 1.66)

# --- Sidebar: TP & SL Calculator ---
st.sidebar.header("TP & SL Calculator")
entry_price = st.sidebar.number_input("Entry Price", value=0.0, min_value=0.0, format="%.5f")
stop_loss_pips = st.sidebar.number_input("Stop Loss (pips)", value=20, min_value=1)
risk_amount = start_bal * (risk_pct / 100)
lot_size = risk_amount / (stop_loss_pips * 10) if stop_loss_pips else 0
take_profit_pips = st.sidebar.number_input("Take Profit (pips)", value=float(stop_loss_pips*rr_ratio) if stop_loss_pips else 40.0, min_value=1.0)
if entry_price > 0 and stop_loss_pips > 0 and take_profit_pips > 0:
    sl_buy = entry_price - (stop_loss_pips * 0.0001)
    tp_buy = entry_price + (take_profit_pips * 0.0001)
    sl_sell = entry_price + (stop_loss_pips * 0.0001)
    tp_sell = entry_price - (take_profit_pips * 0.0001)
    st.sidebar.info(f"Buy: SL={sl_buy:.5f}, TP={tp_buy:.5f}")
    st.sidebar.info(f"Sell: SL={sl_sell:.5f}, TP={tp_sell:.5f}")
    st.sidebar.success(f"Lot Size: {lot_size:.2f}")
    st.sidebar.success(f"Risk Amount: ${risk_amount:.2f}")



@st.cache_data(ttl=300)
def get_data(pair, period):
    try:
        for interval in ["1h", "1d"]:
            df = yf.download(pair, period=period, interval=interval, multi_level_index=False)
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                df['ema_fast'] = ta.ema(df['close'], length=ema_fast_len)
                df['ema_slow'] = ta.ema(df['close'], length=ema_slow_len)
                df['sma_trend'] = ta.sma(df['close'], length=sma_len)
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
                # Always calculate RSI for confluence logic
                df['rsi'] = ta.rsi(df['close'], length=14)
                # Always calculate MACD
                macd = ta.macd(df['close'])
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                return df.dropna()
            else:
                logging.warning(f"No data for {pair} {period} {interval}")
        st.warning(f"No data returned from Yahoo Finance for {pair} ({period}). Tried 1h and 1d intervals.")
        return None
    except Exception as e:
        st.error(f"Data download failed: {e}")
        logging.error(f"Data download failed: {e}")
        return None

def analyze_signal(row):
    ema_buy = row['ema_fast'] > row['ema_slow']
    trend_up = row['close'] > row['sma_trend']
    ema_sell = row['ema_fast'] < row['ema_slow']
    trend_down = row['close'] < row['sma_trend']
    # --- STRICT FILTERS ---
    rsi_ok = ('rsi' in row and row['rsi'] > rsi_thresh)
    macd_ok = ('macd' in row and 'macd_signal' in row and row['macd'] > row['macd_signal'])
    buy = ema_buy and trend_up and rsi_ok and macd_ok
    sell = ema_sell and trend_down and rsi_ok and ('macd' in row and 'macd_signal' in row and row['macd'] < row['macd_signal'])
    return buy, sell, trend_up

# --- REDUCE OVERTRADING: SIGNAL CONFIRMATION ---
def signal_confirmed(df, i, bars=2):
    # Require signal to persist for 'bars' consecutive candles
    if i < bars:
        return False
    for j in range(i-bars+1, i+1):
        buy, sell, _ = analyze_signal(df.iloc[j])
        if not (buy or sell):
            return False
    return True

def run_backtest(bt_data, start_bal, risk_pct, rr_ratio, atr_mult):
    balance = start_bal
    equity_curve = [start_bal]
    wins, losses = 0, 0
    total_p, total_l = 0, 0
    for i in range(200, len(bt_data)-20):
        # --- STRICT SIGNAL CONFIRMATION ---
        if not signal_confirmed(bt_data, i, bars=2):
            continue

        buy, sell, _ = analyze_signal(bt_data.iloc[i])
        if buy or sell:
            risk_amt = balance * (risk_pct / 100)
            entry = bt_data['close'].iloc[i]
            sl = bt_data['atr'].iloc[i] * atr_mult
            tp = sl * rr_ratio
            for j in range(i+1, min(i+100, len(bt_data))):
                res = None
                if buy:
                    if bt_data['high'].iloc[j] >= entry + tp: res = "W"
                    elif bt_data['low'].iloc[j] <= entry - sl: res = "L"
                else:
                    if bt_data['low'].iloc[j] <= entry - tp: res = "W"
                    elif bt_data['high'].iloc[j] >= entry + sl: res = "L"
                if res:
                    if res == "W":
                        pnl = (risk_amt * rr_ratio)
                        balance += pnl; total_p += pnl; wins += 1
                    else:
                        balance -= risk_amt; total_l += risk_amt; losses += 1
                    equity_curve.append(balance)
                    logging.info(f"{'BUY' if buy else 'SELL'} trade at {bt_data.index[i]}: Result={res}, Balance={balance:.2f}")
                    break
    return balance, equity_curve, wins, losses, total_p, total_l

# 4. APP TABS
tab1, tab2, tab3 = st.tabs(["🎯 Live Analysis", "📊 12-Month Backtest", "📓 Journal & Roadmap"])

# --- TAB 1: LIVE ANALYSIS ---
with tab1:

    # Show charts and confluence for all 3 currency pairs
    min_candles = 50
    pairs = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"]
    cols = st.columns(3)
    for idx, pair in enumerate(pairs):
        with cols[idx]:
            st.subheader(f"{pair.replace('=X','')}")
            # Multi-timeframe trend analysis (price above 200 EMA)
            mtf_trends = {}
            for tf in ["15m", "1h", "4h"]:
                try:
                    tf_data = yf.download(pair, period="60d", interval=tf)
                    if tf_data.empty or len(tf_data) < 200:
                        mtf_trends[tf] = False
                    else:
                        tf_data['200ema'] = ta.ema(tf_data['Close'], length=200)
                        close_val = tf_data['Close'].iloc[-1]
                        ema_val = tf_data['200ema'].iloc[-1]
                        mtf_trends[tf] = bool(close_val is not None and ema_val is not None and float(close_val) > float(ema_val))
                except Exception as e:
                    mtf_trends[tf] = False
            st.write(f"Multi-timeframe trend (price > 200 EMA): {mtf_trends}")

            # Main timeframe data for other filters
            data = get_data(pair, "60d")
            adx_ok = False
            volume_ok = False
            rsi_ok = False
            us10y_ok = False
            if data is not None and len(data) >= min_candles:
                # --- ADX momentum filter (> 25) ---
                try:
                    adx = ta.adx(data['high'], data['low'], data['close'], length=14)
                    adx_val = adx['ADX_14'].iloc[-1] if 'ADX_14' in adx.columns else adx.iloc[-1,0]
                    adx_ok = adx_val > 25
                except Exception:
                    adx_ok = False

                # --- Volume surge filter (> 20%) ---
                try:
                    if 'volume' in data.columns:
                        avg_vol = data['volume'].rolling(window=20).mean()
                        volume_ok = data['volume'].iloc[-1] > avg_vol.iloc[-1] * 1.2
                except Exception:
                    volume_ok = False

                # --- RSI pullback filter (< 50) ---
                try:
                    data['RSI'] = ta.rsi(data['close'], length=14)
                    rsi_val = data['RSI'].iloc[-1]
                    rsi_ok = rsi_val < 50
                except Exception:
                    rsi_ok = False

                # --- US 10Y Yield correlation filter (> 0.3) ---
                try:
                    us10y = yf.download('^TNX', period='60d', interval="1d")
                    if not us10y.empty and len(us10y) == len(data):
                        corr = data['close'].tail(20).corr(us10y['Close'].tail(20))
                        us10y_ok = corr > 0.3
                except Exception:
                    us10y_ok = False

                # --- Confluence logic: require 3 of 5 categories ---
                mtf_bool = sum([bool(x) for x in mtf_trends.values()]) >= 2
                categories = [mtf_bool, adx_ok, volume_ok, rsi_ok, us10y_ok]
                met_count = sum(categories)
                clean_cats = [bool(x) for x in categories]
                st.write(f"Confluence met: {met_count}/5", clean_cats)

                # Only trigger trade signal if 3 or more categories are met
                if met_count >= 3:
                    advice = "NO SIGNAL"
                    last_row = data.iloc[-1]
                    buy_signal, sell_signal, trend_up = analyze_signal(last_row)
                    if buy_signal:
                        advice = "BUY"
                    elif sell_signal:
                        advice = "SELL"
                    # Log to diary if signal
                    today = datetime.date.today().isoformat()
                    if advice in ["BUY", "SELL"]:
                        st.success(f"Trade suggestion: {advice}")
                        # Only log once per day per pair per signal
                        diary_entry = {
                            'date': today,
                            'pair': pair.replace('=X',''),
                            'signal': advice
                        }
                        if diary_entry not in st.session_state['trade_diary']:
                            st.session_state['trade_diary'].append(diary_entry)
                        if push_enabled and push_api_key:
                            send_push(f"{pair.replace('=X','')} Trade Alert", f"Advice: {advice}", api_key=push_api_key)
                    else:
                        st.info("No trade: No valid signal.")
                else:
                    st.info("No trade: Not enough confluence (need at least 3/5 categories met).")

                # Use strict signal confirmation (same as backtest)
                advice = "NO SIGNAL"
                last_row = data.iloc[-1]
                buy_signal, sell_signal, trend_up = analyze_signal(last_row)
                if buy_signal:
                    advice = "BUY"
                elif sell_signal:
                    advice = "SELL"
                # Log to diary if signal
                today = datetime.date.today().isoformat()
                if advice in ["BUY", "SELL"]:
                    st.success(f"Trade suggestion: {advice}")
                    # Only log once per day per pair per signal
                    diary_entry = {
                        'date': today,
                        'pair': pair.replace('=X',''),
                        'signal': advice
                    }
                    if diary_entry not in st.session_state['trade_diary']:
                        st.session_state['trade_diary'].append(diary_entry)
                    if push_enabled and push_api_key:
                        send_push(f"{pair.replace('=X','')} Trade Alert", f"Advice: {advice}", api_key=push_api_key)
                else:
                    st.info("No trade: No valid signal.")

# --- Trade Signal Diary Display (Today) ---
with st.sidebar.expander("📔 Trade Signal Diary (Today)", expanded=False):
    today = datetime.date.today().isoformat()
    diary_today = [d for d in st.session_state['trade_diary'] if d['date'] == today]
    if diary_today:
        for entry in diary_today:
            st.write(f"{entry['date']} | {entry['pair']}: {entry['signal']}")
    else:
        st.write("No trade signals today.")

 # --- Day Report (Last 7 Days) ---
with st.sidebar.expander("📅 Day Report (Last 7 Days)", expanded=False):
    
    # --- Manual Trade Journal Entry ---
    with st.sidebar.expander("✍️ Add Trade to Journal", expanded=False):
        if 'manual_trades' not in st.session_state:
            st.session_state['manual_trades'] = []
        with st.form("manual_trade_form"):
            trade_date = st.date_input("Date", value=datetime.date.today(), key="trade_date")
            trade_pair = st.selectbox("Pair", ["EURUSD", "GBPUSD", "AUDUSD", "Other"], key="trade_pair")
            trade_direction = st.selectbox("Direction", ["BUY", "SELL"], key="trade_direction")
            trade_entry = st.number_input("Entry Price", value=0.0, format="%.5f", key="trade_entry")
            trade_exit = st.number_input("Exit Price", value=0.0, format="%.5f", key="trade_exit")
            trade_result = st.selectbox("Result", ["WIN", "LOSS", "BE"], key="trade_result")
            trade_notes = st.text_area("Notes", key="trade_notes")
            submitted = st.form_submit_button("Add Trade")
            if submitted:
                st.session_state['manual_trades'].append({
                    "date": str(trade_date),
                    "pair": trade_pair,
                    "direction": trade_direction,
                    "entry": trade_entry,
                    "exit": trade_exit,
                    "result": trade_result,
                    "notes": trade_notes
                })
                st.success("Trade added to journal!")

        # Show manual trade journal as a table
        if st.session_state['manual_trades']:
            st.markdown("**Manual Trade Journal (This Session):**")
            import pandas as pd
            df_trades = pd.DataFrame(reversed(st.session_state['manual_trades']))
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.write("No manual trades logged yet.")
    today = datetime.date.today()
    last_7_days = [(today - datetime.timedelta(days=i)).isoformat() for i in range(7)]
    diary_week = [d for d in st.session_state['trade_diary'] if d['date'] in last_7_days]
    if diary_week:
        grouped = {}
        for entry in diary_week:
            grouped.setdefault(entry['date'], []).append(entry)
        for date in sorted(grouped.keys(), reverse=True):
            st.markdown(f"**{date}**")
            for e in grouped[date]:
                st.write(f"{e['pair']}: {e['signal']}")
    else:
        st.write("No trade signals in the last 7 days.")

    c1, c2, c3 = st.columns(3)
    if advice == "BUY":
        c1.success("### ✅ SIGNAL: BUY")
        if entry_price > 0 and stop_loss_pips > 0 and take_profit_pips > 0:
            c1.info(f"SL: {sl_buy:.5f} | TP: {tp_buy:.5f} | RR: {rr_ratio:.2f}")
    elif advice == "SELL":
        c1.error("### ✅ SIGNAL: SELL")
        if entry_price > 0 and stop_loss_pips > 0 and take_profit_pips > 0:
            c1.info(f"SL: {sl_sell:.5f} | TP: {tp_sell:.5f} | RR: {rr_ratio:.2f}")
    else:
        c1.warning("### ❌ STATUS: NO SIGNAL")

    c2.metric("Current Price", round(last_row['close'], 5))
    c3.metric("Trend (SMA)", "UP" if trend_up else "DOWN")

    # Clean data for plotting
    plot_data = data[['open', 'high', 'low', 'close']].dropna()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['open'], high=plot_data['high'], low=plot_data['low'], close=plot_data['close'], name="Price"))

    # Auto-zoom y-axis to price range
    price_min = plot_data['low'].min()
    price_max = plot_data['high'].max()
    y_margin = (price_max - price_min) * 0.05 if price_max > price_min else 0.001
    fig.update_yaxes(range=[price_min - y_margin, price_max + y_margin])
    fig.add_trace(go.Scatter(x=data.index, y=data['ema_fast'], name="EMA Fast", line=dict(color='#00d1ff')))
    fig.add_trace(go.Scatter(x=data.index, y=data['ema_slow'], name="EMA Slow", line=dict(color='#ff9900')))
    fig.add_trace(go.Scatter(x=data.index, y=data['sma_trend'], name="SMA Trend", line=dict(color='white', dash='dash')))

    # --- Add Buy/Sell Signal Markers ---
    buy_signals = []
    sell_signals = []
    for i, row in data.iterrows():
        buy, sell, _ = analyze_signal(row)
        if buy:
            buy_signals.append((i, row['low']))
        if sell:
            sell_signals.append((i, row['high']))
    if buy_signals:
        buy_x, buy_y = zip(*buy_signals)
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', marker=dict(color='lime', size=12, symbol='triangle-up'), name='Buy Signal'))
    if sell_signals:
        sell_x, sell_y = zip(*sell_signals)
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'), name='Sell Signal'))

    if 'rsi' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], name="RSI", line=dict(color='green')))
    if 'macd' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['macd'], name="MACD", line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name="MACD Signal", line=dict(color='orange')))
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=True,
        height=900,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# Error handling for chart display
if data is not None and len(data) < min_candles:
    st.warning(f"Not enough data to display candlestick chart (need at least {min_candles} candles, got {len(data)}). Try a lower timeframe or longer period.")
elif data is None:
    st.error("Failed to load market data. Please check your internet connection or try again later.")

# --- TAB 2: BACKTEST (WIN RATE & COMPONENT) ---
with tab2:
    pairs = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"]
    cols = st.columns(3)
    for idx, pair in enumerate(pairs):
        with cols[idx]:
            st.markdown(f"### <span style='color:#0072C6;font-size:1.3em;'>Testing: <b>{pair.replace('=X','')}</b></span>", unsafe_allow_html=True)
            bt_data = get_data(pair, "1y")
            if bt_data is not None:
                balance, equity_curve, wins, losses, total_p, total_l = run_backtest(
                    bt_data, start_bal, risk_pct, rr_ratio, 2.0
                )
                wr = (wins / (wins+losses) * 100) if (wins+losses) > 0 else 0
                pf = (total_p / total_l) if total_l > 0 else 0
                st.header(f"📈 {pair.replace('=X','')} Backtest (12M)")
                st.metric("Win Rate (%)", f"{wr:.1f}")
                st.metric("Profit Factor", f"{pf:.2f}")
                st.metric("Net Balance", f"${balance:.2f}")
            else:
                st.error("Failed to load market data. Please check your internet connection or try again later.")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Final Balance", f"${balance:.2f}")
        m2.metric("Win Rate", f"{wr:.1f}%")
        m3.metric("Profit Factor", f"{pf:.2f}")
        m4.metric("ROI", f"{((balance-1000)/1000)*100:.1f}%")
        m5.metric("Winning Trades", f"{wins}")
        m6.metric("Losing Trades", f"{losses}")

        st.line_chart(equity_curve)

        st.write(f"### 🎯 Goal Progress: {(balance/2000)*100:.1f}%")
        st.progress(min(balance/2000, 1.0))

        st.subheader("📄 Detailed Backtest Report")
        st.markdown(f"""
        - **Total Winning Trades:** {wins}
        - **Total Losing Trades:** {losses}
        - **Win Rate:** {wr:.1f}%
        - **Profit Factor:** {pf:.2f}
        - **Final Balance:** ${balance:.2f}
        """)

        st.info("🧠 Learning/Improvement: In the future, this app will be able to analyze your trade history and suggest parameter improvements or strategy tweaks based on past performance. For now, you can use the win/loss stats to manually adjust your risk, reward, and indicator settings for better results.")

        # --- Parameter Optimizer ---
        st.subheader("🔍 Parameter Optimizer: Find Settings for 45%/55% Win Rate")
        if st.button("Run Optimizer (Grid Search)", key=f"run_optimizer_btn_{pair}"):
            best_params = []
            for ema_fast in [5, 9, 12]:
                for ema_slow in [21, 50, 100]:
                    for sma_len in [100, 200]:
                        for atr_mult in [1.5, 2.0, 2.5]:
                            for rsi_thresh in [45, 50, 55]:
                                # Use a fresh copy of bt_data for each test
                                test_data = bt_data.copy()
                                st.write(f"Testing: EMA Fast={ema_fast}, EMA Slow={ema_slow}, SMA={sma_len}, ATR Mult={atr_mult}, RSI={rsi_thresh}")
                                test_data['ema_fast'] = test_data['close'].rolling(window=ema_fast).mean()
                                test_data['ema_slow'] = test_data['close'].rolling(window=ema_slow).mean()
                                test_data['sma_trend'] = test_data['close'].rolling(window=sma_len).mean()
                                test_data['atr'] = ta.atr(test_data['high'], test_data['low'], test_data['close'], length=14)
                                test_data['rsi'] = ta.rsi(test_data['close'], length=14)
                                # Run backtest
                                b, eq, w, l, tp, tl = run_backtest(test_data, start_bal, risk_pct, rr_ratio, atr_mult)
                                win_rate = (w / (w+l) * 100) if (w+l) > 0 else 0
                                if (w+l) == 0:
                                    st.warning(f"No trades found for EMA Fast={ema_fast}, EMA Slow={ema_slow}, SMA={sma_len}, ATR Mult={atr_mult}, RSI={rsi_thresh}")
                                if win_rate >= 45 and (w+l) > 0:
                                    best_params.append({
                                        'ema_fast': ema_fast,
                                        'ema_slow': ema_slow,
                                        'sma_len': sma_len,
                                        'atr_mult': atr_mult,
                                        'rsi_thresh': rsi_thresh,
                                        'win_rate': win_rate,
                                        'wins': w,
                                        'losses': l,
                                        'final_balance': b
                                    })
            if best_params:
                st.success(f"Found {len(best_params)} parameter sets with >=45% win rate:")
                df_best = pd.DataFrame(best_params)
                st.table(df_best)
                # Suggest the best parameter set (highest win rate, then highest final balance)
                best_row = df_best.sort_values(['win_rate', 'final_balance'], ascending=False).iloc[0]
                st.info(f"Suggested Strategy: EMA Fast={best_row['ema_fast']}, EMA Slow={best_row['ema_slow']}, SMA={best_row['sma_len']}, ATR Mult={best_row['atr_mult']}, RSI={best_row['rsi_thresh']} (Win Rate: {best_row['win_rate']:.1f}%, Final Balance: ${best_row['final_balance']:.2f})")
                if st.button("Apply Suggested Strategy"):
                    st.session_state['ema_fast_len'] = int(best_row['ema_fast'])
                    st.session_state['ema_slow_len'] = int(best_row['ema_slow'])
                    st.session_state['sma_len'] = int(best_row['sma_len'])
                    st.session_state['atr_mult'] = float(best_row['atr_mult'])
                    st.session_state['rsi_thresh'] = int(best_row['rsi_thresh'])
                    st.success("Suggested strategy applied! Please reload the app or adjust the sidebar sliders to see the effect.")
                # Only one button, but with unique key
                # if st.button("Apply Suggested Strategy", key=f"apply_strategy_{pair}"):
                #     st.session_state['ema_fast_len'] = int(best_row['ema_fast'])
                #     st.session_state['ema_slow_len'] = int(best_row['ema_slow'])
                #     st.session_state['sma_len'] = int(best_row['sma_len'])
                #     st.session_state['atr_mult'] = float(best_row['atr_mult'])
                #     st.session_state['rsi_thresh'] = int(best_row['rsi_thresh'])
                #     st.success("Suggested strategy applied! Please reload the app or adjust the sidebar sliders to see the effect.")
            else:
                st.warning("No parameter sets found with >=45% win rate. Try expanding the ranges or adjusting your strategy.")
    else:
        st.error("Failed to load backtest data. Please check your internet connection or try again later.")

# --- TAB 3: ROADMAP & JOURNAL ---
with tab3:
    st.header("📈 12-Month Target Roadmap")
    roadmap = [{"Month": f"Month {m}", "Target": f"${1000 * (1.06**m):.2f}"} for m in range(1, 13)]
    st.table(pd.DataFrame(roadmap))
