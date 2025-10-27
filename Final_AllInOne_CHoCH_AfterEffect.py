# Final_AllInOne_CHoCH_AfterEffect.py
# pip install yfinance pandas mplfinance matplotlib requests

import time
import os
import traceback
from datetime import datetime, timezone
import requests
import pandas as pd
import yfinance as yf
import mplfinance as mpf

# ---------------- USER CONFIG ----------------
TELEGRAM_TOKEN = "8195495301:AAH8A55Ih7p5dKtF7gkQrW1OUOiaYqhuGAc"   # <<-- put your token
CHAT_ID = "5749976506"                 # <<-- put your chat id

SYMBOLS = ["BTC-USD", "MGC=F"]       # BTC and Gold
TIMEFRAMES = ["15m", "30m", "1h"]    # TFs to monitor

LOOKBACK_DAYS = "14d"
CANDLES_TO_SHOW = 80

MAJOR_WINDOW = 60
MINOR_WINDOW = 20

RETEST_TOLERANCE = 0.003   # 0.3% retest tolerance
STRONG_FACTOR = 1.25       # after-effect candle must be this times stronger than failed candle

STOP_LOSS_PCT = 0.01       # default 1% stop loss (set to 0.02 for 2% if you prefer)
LOOP_SLEEP = 300           # 5 minutes between cycles (300 seconds)
TEST_MODE = False          # True => one-shot demo signal then stop

ENABLE_EMA_FILTER = False  # optional trend filter (False by default)
EMA_FAST = 50
EMA_SLOW = 200
# ----------------------------------------------

# runtime state
last_alerts = {}     # key -> pd.Timestamp used to avoid duplicate alerts per candle
failed_patterns = {} # recorded structure breaks that may fail later: key -> dict

# ----------------- utilities -------------------
def now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def log(msg):
    # ASCII-only console logging
    print(f"[{now_utc_str()}] {msg}")

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        log("Telegram message sent")
    except Exception as e:
        log(f"Telegram send error: {e}")

def send_telegram_photo(path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": f}, timeout=30)
        log(f"Telegram photo sent: {os.path.basename(path)}")
    except Exception as e:
        log(f"Telegram photo error: {e}")

def safe_float(x, default=None):
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return default

# ----------------- data fetch ------------------
def get_ohlcv(symbol, timeframe):
    try:
        df = yf.download(symbol, interval=timeframe, period=LOOKBACK_DAYS, progress=False, threads=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df[['Open','High','Low','Close','Volume']].dropna()
        if ENABLE_EMA_FILTER and len(df) >= EMA_SLOW:
            df['EMA_FAST'] = df['Close'].ewm(span=EMA_FAST, adjust=False).mean()
            df['EMA_SLOW'] = df['Close'].ewm(span=EMA_SLOW, adjust=False).mean()
        return df
    except Exception as e:
        log(f"Data fetch error for {symbol} {timeframe}: {e}")
        return None

# --------------- candle rules -----------------
def is_bullish_engulfing(df, idx):
    try:
        if idx < 1 or idx >= len(df):
            return False
        prev = df.iloc[idx-1]; cur = df.iloc[idx]
        return (cur['Close'] > cur['Open']) and (prev['Open'] > prev['Close']) and (cur['Close'] > prev['Open']) and (cur['Open'] < prev['Close'])
    except Exception:
        return False

def is_bearish_engulfing(df, idx):
    try:
        if idx < 1 or idx >= len(df):
            return False
        prev = df.iloc[idx-1]; cur = df.iloc[idx]
        return (cur['Close'] < cur['Open']) and (prev['Open'] < prev['Close']) and (cur['Close'] < prev['Open']) and (cur['Open'] > prev['Close'])
    except Exception:
        return False

def is_hammer_row(row):
    try:
        o = safe_float(row['Open']); h = safe_float(row['High']); l = safe_float(row['Low']); c = safe_float(row['Close'])
        if None in (o,h,l,c):
            return False
        body = abs(c - o)
        total = (h - l) if (h - l) != 0 else 1.0
        if body == 0:
            return False
        high_wick = h - max(c, o)
        low_wick = min(c, o) - l
        return (low_wick >= 2 * body) and (high_wick <= 0.5 * body) and ((body/total) <= 0.35)
    except Exception:
        return False

# ---------------- chart generator ----------------
def make_chart(df, symbol, entry, sl, tag):
    try:
        window = df.tail(CANDLES_TO_SHOW)
        if window.empty:
            return None
        safe_symbol = symbol.replace('/','_').replace('=','_')
        fn = f"{safe_symbol}_{tag}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.png"
        mpf.plot(window, type='candle', style='yahoo',
                 title=f"{symbol} {tag}",
                 volume=True,
                 savefig=dict(fname=fn, dpi=150, bbox_inches='tight'),
                 hlines=dict(hlines=[sl, entry], colors=['red','green'], linewidths=1.2, linestyle=['--','-']))
        return fn
    except Exception as e:
        log(f"Chart creation error: {e}")
        return None

# ------------- CHoCH detect & record -------------
def detect_choc_and_record(df, symbol, tf):
    # safe guard
    try:
        n = len(df)
        if n < 6:
            return
        base = f"{symbol}_{tf}"

        df['major_low'] = df['Low'].rolling(window=min(MAJOR_WINDOW, n)).min()
        df['major_high'] = df['High'].rolling(window=min(MAJOR_WINDOW, n)).max()
        df['minor_low'] = df['Low'].rolling(window=min(MINOR_WINDOW, n)).min()
        df['minor_high'] = df['High'].rolling(window=min(MINOR_WINDOW, n)).max()

        idx = n - 1
        cur = df.iloc[idx]

        def prev_safe(series, i):
            if i >= 0:
                return safe_float(series.iloc[i])
            return None

        maj_low_prev = prev_safe(df['major_low'], idx-1)
        if (maj_low_prev is not None) and (safe_float(cur['Close']) is not None) and (safe_float(cur['Close']) < maj_low_prev):
            key = base + "_MAJOR_SELL"
            if key not in failed_patterns:
                failed_patterns[key] = {
                    'type': 'MAJOR_SELL',
                    'level': maj_low_prev,
                    'time': df.index[idx],
                    'body': abs(safe_float(cur['Close']) - safe_float(cur['Open'])),
                    'volume': safe_float(cur['Volume']),
                    'failed': False
                }
                log(f"{symbol} {tf} -> recorded CHoCH: {key}")

        min_low_prev = prev_safe(df['minor_low'], idx-1)
        if (min_low_prev is not None) and (safe_float(cur['Close']) is not None) and (safe_float(cur['Close']) < min_low_prev):
            key = base + "_MINOR_SELL"
            if key not in failed_patterns:
                failed_patterns[key] = {
                    'type': 'MINOR_SELL',
                    'level': min_low_prev,
                    'time': df.index[idx],
                    'body': abs(safe_float(cur['Close']) - safe_float(cur['Open'])),
                    'volume': safe_float(cur['Volume']),
                    'failed': False
                }
                log(f"{symbol} {tf} -> recorded CHoCH: {key}")

        maj_high_prev = prev_safe(df['major_high'], idx-1)
        if (maj_high_prev is not None) and (safe_float(cur['Close']) is not None) and (safe_float(cur['Close']) > maj_high_prev):
            key = base + "_MAJOR_BUY"
            if key not in failed_patterns:
                failed_patterns[key] = {
                    'type': 'MAJOR_BUY',
                    'level': maj_high_prev,
                    'time': df.index[idx],
                    'body': abs(safe_float(cur['Close']) - safe_float(cur['Open'])),
                    'volume': safe_float(cur['Volume']),
                    'failed': False
                }
                log(f"{symbol} {tf} -> recorded CHoCH: {key}")

        min_high_prev = prev_safe(df['minor_high'], idx-1)
        if (min_high_prev is not None) and (safe_float(cur['Close']) is not None) and (safe_float(cur['Close']) > min_high_prev):
            key = base + "_MINOR_BUY"
            if key not in failed_patterns:
                failed_patterns[key] = {
                    'type': 'MINOR_BUY',
                    'level': min_high_prev,
                    'time': df.index[idx],
                    'body': abs(safe_float(cur['Close']) - safe_float(cur['Open'])),
                    'volume': safe_float(cur['Volume']),
                    'failed': False
                }
                log(f"{symbol} {tf} -> recorded CHoCH: {key}")
    except Exception as e:
        log(f"detect_choc_and_record error: {e}\n{traceback.format_exc()}")

# ------------- mark failed when recovered -------------
def mark_failed_when_recovered(df, symbol, tf):
    try:
        latest_close = safe_float(df['Close'].iloc[-1])
        if latest_close is None:
            return
        base = f"{symbol}_{tf}"
        for suffix in ["_MAJOR_SELL","_MINOR_SELL","_MAJOR_BUY","_MINOR_BUY"]:
            key = base + suffix
            pat = failed_patterns.get(key)
            if not pat or pat.get('failed', False):
                continue
            level = safe_float(pat['level'])
            typ = pat['type']
            if level is None:
                continue
            if typ.endswith("SELL"):
                if latest_close > level * (1 + RETEST_TOLERANCE * 2):
                    pat['failed'] = True
                    pat['failed_time'] = df.index[-1]
                    log(f"{symbol} {tf} -> marked FAILED: {key}")
            else:
                if latest_close < level * (1 - RETEST_TOLERANCE * 2):
                    pat['failed'] = True
                    pat['failed_time'] = df.index[-1]
                    log(f"{symbol} {tf} -> marked FAILED: {key}")
    except Exception as e:
        log(f"mark_failed_when_recovered error: {e}\n{traceback.format_exc()}")

# ------------- retest + after-effect confirmation -------------
def check_retests_and_aftereffect(df, symbol, tf):
    try:
        idx = len(df) - 1
        latest_close = safe_float(df['Close'].iloc[idx])
        if latest_close is None:
            return
        latest_row = df.iloc[idx]
        base = f"{symbol}_{tf}"

        for suffix in ["_MINOR_SELL","_MAJOR_SELL","_MINOR_BUY","_MAJOR_BUY"]:
            key = base + suffix
            pat = failed_patterns.get(key)
            if not pat or not pat.get('failed', False):
                continue
            typ = pat['type']
            level = safe_float(pat['level'])
            if level is None:
                continue

            # SELL-side failed -> want BUY on retest
            if typ.endswith("SELL"):
                if abs(latest_close - level) / level <= RETEST_TOLERANCE:
                    confirmed = False
                    conf_desc = ""
                    if is_bullish_engulfing(df, idx):
                        confirmed = True; conf_desc = "Bullish Engulfing"
                    elif is_hammer_row(latest_row):
                        cur_body = abs(safe_float(latest_row['Close']) - safe_float(latest_row['Open'])) or 0
                        cur_vol = safe_float(latest_row['Volume']) or 0
                        if (pat.get('body') is not None and pat.get('volume') is not None) and (cur_body >= pat['body'] * STRONG_FACTOR) and (cur_vol >= pat['volume'] * STRONG_FACTOR):
                            confirmed = True; conf_desc = "Strong Hammer (After-Effect)"

                    if confirmed and ENABLE_EMA_FILTER:
                        ema_fast = safe_float(df['EMA_FAST'].iloc[-1]); ema_slow = safe_float(df['EMA_SLOW'].iloc[-1])
                        if not (ema_fast and ema_slow and (latest_close > ema_fast and ema_fast > ema_slow)):
                            confirmed = False; conf_desc += " (EMA blocked)"

                    if confirmed:
                        entry = latest_close
                        sl = entry * (1 - STOP_LOSS_PCT)
                        caption = (f"🟢 AFTER-EFFECT / RETEST BUY\n{symbol} ({tf})\nPattern: {typ}\nConfirm: {conf_desc}\n"
                                   f"Level: {level:.6f}\nEntry: {entry:.6f}\nStopLoss: {sl:.6f}\nTime: {now_utc_str()}")
                        if last_alerts.get(key) != df.index[idx]:
                            chart = make_chart(df, symbol, entry, sl, "AFTER_RETEST_BUY")
                            send_telegram(caption)
                            if chart:
                                send_telegram_photo(chart, caption)
                                try: os.remove(chart)
                                except: pass
                            last_alerts[key] = df.index[idx]
                            failed_patterns.pop(key, None)

            else:
                # BUY-side failed -> want SELL on retest
                if abs(latest_close - level) / level <= RETEST_TOLERANCE:
                    confirmed = False
                    conf_desc = ""
                    if is_bearish_engulfing(df, idx):
                        confirmed = True; conf_desc = "Bearish Engulfing"
                    else:
                        cur_body = abs(safe_float(latest_row['Close']) - safe_float(latest_row['Open'])) or 0
                        cur_vol = safe_float(latest_row['Volume']) or 0
                        if (pat.get('body') is not None and pat.get('volume') is not None) and (cur_body >= pat['body'] * STRONG_FACTOR) and (cur_vol >= pat['volume'] * STRONG_FACTOR):
                            confirmed = True; conf_desc = "Strong Bearish (After-Effect)"

                    if confirmed and ENABLE_EMA_FILTER:
                        ema_fast = safe_float(df['EMA_FAST'].iloc[-1]); ema_slow = safe_float(df['EMA_SLOW'].iloc[-1])
                        if not (ema_fast and ema_slow and (latest_close < ema_fast and ema_fast < ema_slow)):
                            confirmed = False; conf_desc += " (EMA blocked)"

                    if confirmed:
                        entry = latest_close
                        sl = entry * (1 + STOP_LOSS_PCT)
                        caption = (f"🔴 AFTER-EFFECT / RETEST SELL\n{symbol} ({tf})\nPattern: {typ}\nConfirm: {conf_desc}\n"
                                   f"Level: {level:.6f}\nEntry: {entry:.6f}\nStopLoss: {sl:.6f}\nTime: {now_utc_str()}")
                        if last_alerts.get(key) != df.index[idx]:
                            chart = make_chart(df, symbol, entry, sl, "AFTER_RETEST_SELL")
                            send_telegram(caption)
                            if chart:
                                send_telegram_photo(chart, caption)
                                try: os.remove(chart)
                                except: pass
                            last_alerts[key] = df.index[idx]
                            failed_patterns.pop(key, None)
    except Exception as e:
        log(f"check_retests_and_aftereffect error: {e}\n{traceback.format_exc()}")

# ---------------- run cycle -------------------
def run_cycle():
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            df = get_ohlcv(symbol, tf)
            if df is None or len(df) < 6:
                log(f"{symbol} {tf} -> insufficient data or fetch error")
                continue
            log(f"Checking {symbol} {tf} (candles: {len(df)})")
            detect_choc_and_record(df, symbol, tf)
            mark_failed_when_recovered(df, symbol, tf)
            check_retests_and_aftereffect(df, symbol, tf)
            # TEST_MODE one-shot demo
            if TEST_MODE:
                latest = df.iloc[-1]
                entry = safe_float(latest['Close'])
                if entry:
                    sl = entry * (1 - STOP_LOSS_PCT)
                    txt = f"TEST SIGNAL {symbol} ({tf}) - Entry:{entry:.6f} SL:{sl:.6f}"
                    chart = make_chart(df, symbol, entry, sl, "TEST")
                    send_telegram(txt)
                    if chart:
                        send_telegram_photo(chart, txt)
                        try: os.remove(chart)
                        except: pass
                return

# ---------------- main -------------------
def main():
    try:
        send_telegram(f"✅ Swing Bot started (CHoCH+AfterEffect) {now_utc_str()}")
    except Exception:
        log("Couldn't send startup Telegram message (check token/chat).")
    log("Bot started: monitoring symbols and TFs.")
    while True:
        try:
            run_cycle()
        except Exception as e:
            log(f"Runtime error in main loop: {e}\n{traceback.format_exc()}")
        time.sleep(LOOP_SLEEP)

if __name__ == "__main__":
    main()
