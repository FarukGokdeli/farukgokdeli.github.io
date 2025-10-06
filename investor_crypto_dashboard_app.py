# investor_crypto_dashboard_app.py
# Gereksinimler (örnek):
# pip install streamlit requests pandas numpy matplotlib plotly fpdf urllib3 tensorflow

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fpdf import FPDF
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Opsiyonel: TensorFlow (LSTM için)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    tf = None

# Opsiyonel: Binance client (kullanıcı anahtar verirse)
try:
    from binance.client import Client as BinanceClient
except Exception:
    BinanceClient = None

# ---------------------------
# Streamlit başlangıç
# ---------------------------
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Investor Crypto Dashboard — AI Enhanced")


# ---------------------------
# Requests session (retry)
# ---------------------------
def create_session(retries=3, backoff=0.3, status_forcelist=(429, 500, 502, 503, 504)):
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(['GET', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


session = create_session()


# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------
def human_format_number(n):
    try:
        n = float(n)
    except Exception:
        return str(n)
    magnitude = 0
    units = ['', 'K', 'M', 'B', 'T']
    while abs(n) >= 1000 and magnitude < len(units) - 1:
        magnitude += 1
        n /= 1000.0
    if magnitude == 0:
        return f"{n:,.2f}"
    return f"{n:.2f}{units[magnitude]}"


def format_price(p):
    try:
        return f"${float(p):,.4f}"
    except Exception:
        return str(p)


# ---------------------------
# CoinGecko helpers
# ---------------------------
@st.cache_data(ttl=300)
def fetch_top_100_coins(vs_currency='usd'):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": vs_currency, "order": "market_cap_desc", "per_page": 100, "page": 1,
              "sparkline": False, "price_change_percentage": "24h,7d"}
    try:
        r = session.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype(str).str.upper()

    def col_or_nan(df, col):
        return df[col] if col in df.columns else pd.Series([np.nan] * len(df), index=df.index)

    possible_24h = ['price_change_percentage_24h', 'price_change_percentage_24h_in_currency', 'change_24h']
    possible_7d = ['price_change_percentage_7d_in_currency', 'price_change_percentage_7d', 'change_7d']
    change24 = next((c for c in possible_24h if c in df.columns), None)
    change7 = next((c for c in possible_7d if c in df.columns), None)
    df_small = pd.DataFrame({
        'market_cap_rank': col_or_nan(df, 'market_cap_rank'),
        'id': col_or_nan(df, 'id'),
        'symbol': col_or_nan(df, 'symbol'),
        'name': col_or_nan(df, 'name'),
        'price': col_or_nan(df, 'current_price') if 'current_price' in df.columns else col_or_nan(df, 'price'),
        'market_cap': col_or_nan(df, 'market_cap'),
        'volume': col_or_nan(df, 'total_volume') if 'total_volume' in df.columns else col_or_nan(df, 'volume'),
        'change_24h_pct': col_or_nan(df, change24) if change24 else pd.Series([np.nan] * len(df), index=df.index),
        'change_7d': col_or_nan(df, change7) if change7 else pd.Series([np.nan] * len(df), index=df.index)
    })
    for c in ['price', 'market_cap', 'volume', 'change_24h_pct', 'change_7d']:
        if c in df_small.columns:
            df_small[c] = pd.to_numeric(df_small[c], errors='coerce')
    return df_small


@st.cache_data(ttl=3600)
def fetch_historical_prices_coingecko(coin_id, days="max"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    try:
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if 'prices' not in j:
            return pd.DataFrame()
        prices = [p[1] for p in j['prices']]
        dates = [pd.to_datetime(p[0], unit='ms') for p in j['prices']]
        return pd.DataFrame({"date": dates, "price": prices})
    except Exception:
        return pd.DataFrame()


# ---------------------------
# Exchange klines (Gate.io / MEXC)
# ---------------------------
def get_gateio_klines(symbol, interval="1h", limit=500):
    url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    params = {"currency_pair": symbol, "interval": interval, "limit": limit}
    try:
        r = session.get(url, params=params, timeout=12);
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["t", "v", "c", "h", "l", "o", "last_close", "buy_base_volume"])
        df["t"] = pd.to_numeric(df["t"]).apply(lambda x: pd.to_datetime(x, unit='s'))
        df = df.sort_values("t").reset_index(drop=True)
        df.rename(columns={"t": "open_time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
                  inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df[["open_time", "open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


def get_mexc_klines(symbol="BTCUSDT", interval="1h", limit=500):
    url = "https://api.mexc.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = session.get(url, params=params, timeout=12);
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "n",
            "taker_base", "taker_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return pd.DataFrame()


# ---------------------------
# Binance klines
# ---------------------------
def get_binance_client(api_key, api_secret):
    if BinanceClient is None:
        return None
    try:
        client = BinanceClient(api_key, api_secret)
        return client
    except Exception:
        return None


def get_binance_klines(client, symbol="BTCUSDT", interval="1h", limit=500):
    if client is None:
        return pd.DataFrame()
    try:
        data = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return pd.DataFrame()


# ---------------------------
# Teknik indikatörler & metrikler
# ---------------------------
def calculate_rsi_series(prices, period=14):
    s = pd.Series(prices).astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd_series(prices, fast=12, slow=26, signal=9):
    s = pd.Series(prices).astype(float)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def bollinger_bands(prices, window=20, num_std=2):
    s = pd.Series(prices).astype(float)
    ma = s.rolling(window).mean()
    std = s.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def max_drawdown(equity_curve):
    eq = np.array(equity_curve, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    return float(abs(drawdown.min()))


def sharpe_ratio(equity_curve, freq_per_year=252):
    eq = np.array(equity_curve, dtype=float)
    if len(eq) < 2:
        return np.nan
    ret = np.diff(eq) / (eq[:-1] + 1e-9)
    if np.nanstd(ret) == 0:
        return np.nan
    return float(np.nanmean(ret) / np.nanstd(ret) * math.sqrt(freq_per_year))


def sortino_ratio(equity_curve, freq_per_year=252, target=0.0):
    eq = np.array(equity_curve, dtype=float)
    if len(eq) < 2:
        return np.nan
    ret = np.diff(eq) / (eq[:-1] + 1e-9)
    downside = ret[ret < target]
    if len(downside) == 0:
        return np.nan
    return float(np.nanmean(ret) / np.nanstd(downside) * math.sqrt(freq_per_year))


# ---------------------------
# Scoring helpers
# ---------------------------
def normalize_scores(raw_scores, scale_max=1000):
    arr = np.array(raw_scores, dtype=float)
    if arr.size == 0:
        return []
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if math.isclose(mx, mn) or np.isnan(mx) or np.isnan(mn):
        return [float(scale_max / 2)] * len(arr)
    norm = (arr - mn) / (mx - mn)
    return (norm * scale_max).round(2)


# ---------------------------
# Symbol analysis
# ---------------------------
def analyze_single_symbol_worker(symbol, interval="1h", limit=500, binance_client=None, rsi_period=14):
    try:
        base = symbol.upper().replace("_USDT", "").replace("USDT", "")
        df = pd.DataFrame()
        if binance_client is not None:
            try:
                df = get_binance_klines(binance_client, symbol=base + "USDT", interval=interval, limit=limit)
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            df = get_gateio_klines(base + "_USDT", interval=interval, limit=limit)
        if df.empty:
            df = get_mexc_klines(base + "USDT", interval='60m' if interval in ('1h', '60m') else interval, limit=limit)
        if df.empty:
            return {"Coin": base, "Symbol": base, "Price": np.nan, "RSI": np.nan, "MACD": np.nan, "Score_raw": 0.0}
        if 'close' in df.columns:
            closes = pd.to_numeric(df['close'], errors='coerce').dropna().reset_index(drop=True)
        else:
            closes = pd.to_numeric(df.iloc[:, 4], errors='coerce').dropna().reset_index(drop=True)
        if closes.empty:
            return {"Coin": base, "Symbol": base, "Price": np.nan, "RSI": np.nan, "MACD": np.nan, "Score_raw": 0.0}
        price = float(closes.iloc[-1])
        rsi_val = float(calculate_rsi_series(closes, period=rsi_period).iloc[-1])
        macd_line, macd_signal, _ = compute_macd_series(closes)
        macd_val = float(macd_line.iloc[-1]) if not macd_line.empty else np.nan
        macd_sig = float(macd_signal.iloc[-1]) if not macd_signal.empty else np.nan
        ma, bb_upper, bb_lower = bollinger_bands(closes, window=20)
        # raw score: kombinasyon -> normalize sonrası 0..1000
        raw = 0.0
        # RSI etkisi (düşük RSI -> daha cazip)
        if not np.isnan(rsi_val):
            raw += max(0.0, (70.0 - rsi_val) * 2.5)
        # MACD etkisi
        if not np.isnan(macd_val) and not np.isnan(macd_sig) and macd_val > macd_sig:
            raw += 25.0
        # BB etkisi: fiyat alt banda yakınsa + küçük bonus
        try:
            rel_bb = (price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-9)
            raw += max(0.0, (1.0 - rel_bb) * 20.0)
        except Exception:
            pass
        return {"Coin": base, "Symbol": base, "Price": price, "RSI": round(rsi_val, 2),
                "MACD": round(macd_val, 6) if not np.isnan(macd_val) else np.nan, "Score_raw": float(raw)}
    except Exception:
        return {"Coin": symbol, "Symbol": symbol, "Price": np.nan, "RSI": np.nan, "MACD": np.nan, "Score_raw": 0.0}


def analyze_symbols_parallel(symbols, interval="1h", limit=500, binance_client=None, max_workers=8, rsi_period=14):
    if not symbols:
        return pd.DataFrame()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_single_symbol_worker, s, interval, limit, binance_client, rsi_period): s for s in
                   symbols}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception:
                continue
    df = pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame()
    df['Score_raw'] = pd.to_numeric(df['Score_raw'], errors='coerce').fillna(0.0)
    df['Score'] = normalize_scores(df['Score_raw'].values, scale_max=1000)
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    return df


# ---------------------------
# Multi-indicator signal
# ---------------------------
def multi_indicator_signal(prices, rsi_period=14, macd_fast=12, macd_slow=26, macd_sig=9, bb_window=20):
    prices = pd.Series(prices).astype(float)
    n = len(prices)
    if n < 20:
        return {"signal": "NÖTR", "reason": "Yetersiz veri (en az 20 kayıt gerekli)", "rsi": np.nan, "macd": np.nan,
                "macd_signal": np.nan, "bb_lower": np.nan, "bb_upper": np.nan}
    rsi = calculate_rsi_series(prices, period=rsi_period).iloc[-1]
    macd_line, macd_signal, _ = compute_macd_series(prices, fast=macd_fast, slow=macd_slow, signal=macd_sig)
    macd_v = macd_line.iloc[-1]
    macd_s = macd_signal.iloc[-1]
    ma, bb_upper, bb_lower = bollinger_bands(prices, window=bb_window)
    price = float(prices.iloc[-1])
    bb_l = bb_lower.iloc[-1]
    bb_u = bb_upper.iloc[-1]
    reasons = []
    buy_votes = 0
    sell_votes = 0
    # RSI condition
    if not np.isnan(rsi):
        if rsi < 40:  # daha esnek: 40
            buy_votes += 1
            reasons.append("RSI düşük")
        elif rsi > 60:
            sell_votes += 1
            reasons.append("RSI yüksek")
    # MACD condition
    try:
        if not (np.isnan(macd_v) or np.isnan(macd_s)):
            if macd_v > macd_s:
                buy_votes += 1
                reasons.append("MACD pozitif")
            else:
                sell_votes += 1
                reasons.append("MACD negatif")
    except Exception:
        pass
    # BB condition (alt/üst band yakınlığı)
    try:
        band_range = bb_u - bb_l + 1e-9
        pct_from_lower = (price - bb_l) / band_range
        pct_from_upper = (bb_u - price) / band_range
        if pct_from_lower <= 0.30:  # alt banda yakın
            buy_votes += 1
            reasons.append("Fiyat alt banda yakın")
        if pct_from_upper <= 0.30:  # üst banda yakın
            sell_votes += 1
            reasons.append("Fiyat üst banda yakın")
    except Exception:
        pass
    # karar: iki veya daha fazla oy -> AL / SAT, eşit ise NÖTR
    if buy_votes >= 2 and buy_votes > sell_votes:
        sig = "AL"
    elif sell_votes >= 2 and sell_votes > buy_votes:
        sig = "SAT"
    else:
        sig = "NÖTR"
    return {"signal": sig, "reason": '; '.join(reasons), "rsi": round(rsi, 2) if not np.isnan(rsi) else np.nan,
            "macd": round(macd_v, 6) if not np.isnan(macd_v) else np.nan,
            "macd_signal": round(macd_s, 6) if not np.isnan(macd_s) else np.nan,
            "bb_lower": round(bb_l, 6) if not np.isnan(bb_l) else np.nan,
            "bb_upper": round(bb_u, 6) if not np.isnan(bb_u) else np.nan}


# ---------------------------
# Backtesting multi-indicator
# ---------------------------
def backtest_strategy_multi(prices, dates=None, initial_cash=1000.0, rsi_period=14, buy_threshold=40, sell_threshold=60,
                            use_multi_indicator=True, trade_fee_pct=0.0):
    prices = pd.Series(prices).astype(float).reset_index(drop=True)
    if prices.empty:
        return [], {}, [], [], [], pd.Series(dtype=float)
    rsi_series = calculate_rsi_series(prices, period=rsi_period)
    macd_line, macd_signal, _ = compute_macd_series(prices)
    ma20, bb_upper, bb_lower = bollinger_bands(prices, window=20)
    cash = initial_cash
    pos = 0.0
    trades = []
    buy_points = []
    sell_points = []
    equity = [initial_cash]
    for i in range(len(prices)):
        p = float(prices.iloc[i])
        r = float(rsi_series.iloc[i]) if not np.isnan(rsi_series.iloc[i]) else np.nan
        macd_v = float(macd_line.iloc[i]) if not np.isnan(macd_line.iloc[i]) else np.nan
        macd_s = float(macd_signal.iloc[i]) if not np.isnan(macd_signal.iloc[i]) else np.nan
        bb_l = float(bb_lower.iloc[i]) if not np.isnan(bb_lower.iloc[i]) else np.nan
        bb_u = float(bb_upper.iloc[i]) if not np.isnan(bb_upper.iloc[i]) else np.nan

        buy_cond = False
        sell_cond = False

        if use_multi_indicator:
            # 2/3 voting rule — daha esnek: iki şart uyarsa al/sat
            votes_buy = 0
            votes_sell = 0
            # RSI
            if not np.isnan(r):
                if r < buy_threshold: votes_buy += 1
                if r > sell_threshold: votes_sell += 1
            # MACD
            if not (np.isnan(macd_v) or np.isnan(macd_s)):
                if macd_v > macd_s:
                    votes_buy += 1
                else:
                    votes_sell += 1
            # BB
            try:
                band_range = bb_u - bb_l + 1e-9
                pct_from_lower = (p - bb_l) / band_range
                pct_from_upper = (bb_u - p) / band_range
                if pct_from_lower <= 0.30: votes_buy += 1
                if pct_from_upper <= 0.30: votes_sell += 1
            except Exception:
                pass
            if votes_buy >= 2 and votes_buy > votes_sell:
                buy_cond = True
            if votes_sell >= 2 and votes_sell > votes_buy:
                sell_cond = True
        else:
            # simple RSI rule
            if not np.isnan(r):
                if r < buy_threshold: buy_cond = True
                if r > sell_threshold: sell_cond = True

        # execute buy
        if buy_cond and cash > 0:
            qty = (cash * (1 - trade_fee_pct)) / p if trade_fee_pct > 0 else cash / p
            pos = qty
            trades.append(('BUY', i, p))
            buy_points.append((i, p))
            cash = 0.0
        # execute sell
        elif sell_cond and pos > 0:
            proceeds = pos * p
            proceeds = proceeds * (1 - trade_fee_pct) if trade_fee_pct > 0 else proceeds
            trades.append(('SELL', i, p))
            sell_points.append((i, p))
            cash = proceeds
            pos = 0.0

        equity.append(cash + pos * p)

    final_value = cash + pos * float(prices.iloc[-1])
    trade_count = len(trades)
    buys = [t for t in trades if t[0] == 'BUY']
    sells = [t for t in trades if t[0] == 'SELL']
    paired = min(len(buys), len(sells))
    wins = 0
    avg_trade_returns = []
    for j in range(paired):
        b = buys[j][2];
        s = sells[j][2]
        ret = (s - b) / (b + 1e-9)
        avg_trade_returns.append(ret)
        if s > b:
            wins += 1
    win_rate = (wins / paired) if paired > 0 else np.nan
    roi = (final_value - initial_cash) / (initial_cash + 1e-9) * 100.0
    md = max_drawdown(equity)
    sr = sharpe_ratio(equity)
    sortino = sortino_ratio(equity)
    metrics = {
        "final_value": float(final_value),
        "ROI_pct": float(roi),
        "trade_count": int(trade_count),
        "paired_trades": int(paired),
        "win_rate": float(win_rate) if not np.isnan(win_rate) else np.nan,
        "avg_trade_return": float(np.nanmean(avg_trade_returns)) if avg_trade_returns else np.nan,
        "max_drawdown_pct": float(md * 100.0),
        "sharpe": float(sr) if not np.isnan(sr) else np.nan,
        "sortino": float(sortino) if not np.isnan(sortino) else np.nan,
        "max_equity": float(np.nanmax(equity)),
        "min_equity": float(np.nanmin(equity)),
    }
    return trades, metrics, equity, buy_points, sell_points, rsi_series


# ---------------------------
# LSTM utilities
# ---------------------------
# ---------------------------
# ADVANCED LSTM + BiLSTM + Attention + MC-Dropout (Production-ready, crypto-tuned)
# ---------------------------
from tensorflow.keras.layers import Layer, Input, Bidirectional, LSTM as KerasLSTM, Dense as KerasDense, \
    Dropout as KerasDropout, GaussianNoise
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# -- Attention layer (simple, works with (batch, timesteps, features) -> attention over timesteps)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_u', shape=(input_shape[-1],),
                                 initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs: (batch, timesteps, features)
        v = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)  # (batch, timesteps, features)
        vu = tf.tensordot(v, self.u, axes=1)  # (batch, timesteps)
        alphas = tf.nn.softmax(vu, axis=1)  # (batch, timesteps)
        alphas_exp = tf.expand_dims(alphas, -1)  # (batch, timesteps, 1)
        output = tf.reduce_sum(inputs * alphas_exp, axis=1)  # (batch, features)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


# ---------------------------
# Scaling helpers (min-max per feature, robust to constant columns)
# ---------------------------
def scale_features_matrix(arr):
    """
    arr: numpy array (n_samples, n_features)
    returns: scaled, mn, mx  (scaled in [0,1] per feature)
    """
    arr = np.array(arr, dtype=float)
    mn = np.nanmin(arr, axis=0)
    mx = np.nanmax(arr, axis=0)
    denom = mx - mn
    denom_fix = denom.copy()
    denom_fix[denom_fix == 0] = 1.0
    scaled = (arr - mn) / denom_fix
    # clip to [0,1] in case of numerical issues
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled, mn, mx


def inverse_scale_single_feature(scaled_val, mn, mx, feature_index=0):
    # mn/mx may be arrays; invert for single feature index (default: price = 0)
    if isinstance(mn, np.ndarray):
        mn0 = mn[feature_index]
        mx0 = mx[feature_index]
    else:
        mn0 = mn;
        mx0 = mx
    return float(scaled_val * (mx0 - mn0) + mn0)


# ---------------------------
# Feature engineering for crypto (multivariate)
# ---------------------------
def build_feature_matrix(prices, rsi_period=14):
    """
    prices: 1D array-like of close prices (float)
    returns: features (n, f) in order:
      [price, RSI, MACD, EMA20, EMA50, volume_norm (if available -> else 0), HL_range]
    volume not always available here; set zeros if absent.
    """
    import pandas as pd
    s = pd.Series(prices).astype(float)
    n = len(s)
    RSI = calculate_rsi_series(s.values, period=rsi_period).fillna(50).values
    macd_line, macd_signal, macd_hist = compute_macd_series(s.values)
    macd_line = np.nan_to_num(macd_line, nan=0.0)
    ema20 = s.ewm(span=20, adjust=False).mean().values
    ema50 = s.ewm(span=50, adjust=False).mean().values
    # approximate HL_range as small volatility proxy: we'll compute rolling high-low from price series (since we don't have high/low here)
    # Use short-window std * price as proxy
    roll_std = s.rolling(window=14, min_periods=1).std().fillna(0).values
    hl_proxy = roll_std  # proxy for volatility
    # volume placeholder (0)
    vol = np.zeros(n)
    features = np.vstack([s.values, RSI, macd_line, ema20, ema50, vol, hl_proxy]).T  # shape (n,7)
    return features


# ---------------------------
# Prepare LSTM data (multivariate) - returns X,y,scaled_features,mn,mx
# ---------------------------
def prepare_advanced_lstm_data(prices, lookback=60, rsi_period=14, add_indicators=True):
    if len(prices) <= lookback:
        return None, None, None, None, None
    if add_indicators:
        feats = build_feature_matrix(prices, rsi_period=rsi_period)
    else:
        feats = np.array(prices).reshape(-1, 1)
    scaled, mn, mx = scale_features_matrix(feats)
    X = []
    y = []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, :])  # (lookback, n_features)
        y.append(scaled[i, 0])  # predict scaled price (feature 0)
    X = np.array(X)
    y = np.array(y)
    return X, y, scaled, mn, mx


# ---------------------------
# Advanced model builder: BiLSTM + Attention + Dense, with regularization

# --- Compatibility wrappers for older function names (keeps backward compatibility) ---
def prepare_lstm_data(prices, lookback=60, add_indicators=True, rsi_period=14):
    """
    Backward-compatible wrapper for older calls expecting:
      X, y, mn, mx, scaled_features
    Internally calls prepare_advanced_lstm_data which returns:
      X, y, scaled, mn, mx
    We reorder to the older expected signature.
    """
    res = prepare_advanced_lstm_data(prices, lookback=lookback, rsi_period=rsi_period, add_indicators=add_indicators)
    if res is None:
        return None, None, None, None, None
    X, y, scaled, mn, mx = res
    # return in old order: X, y, mn, mx, scaled_features
    return X, y, mn, mx, scaled


def create_lstm_model(input_shape, units1=64, units2=32, dropout=0.2):
    """
    Backward-compatible wrapper: old code calls create_lstm_model((timesteps, features), ...)
    We map this to create_advanced_model for improved architecture.
    """
    try:
        timesteps = int(input_shape[0])
        n_features = int(input_shape[1])
    except Exception:
        # if input_shape passed differently, try to be flexible
        if isinstance(input_shape, tuple) and len(input_shape) >= 2:
            timesteps = int(input_shape[0]);
            n_features = int(input_shape[1])
        else:
            raise ValueError("create_lstm_model wrapper: input_shape must be (timesteps, features)")
    # Map provided units to more advanced defaults, but keep parameters
    u1 = max(64, units1)
    u2 = max(32, units2)
    return create_advanced_model(timesteps, n_features, units1=u1, units2=u2, dropout_rate=dropout)


def iterative_predict(model, last_seq_scaled, days, mn, mx):
    """
    Backward-compatible wrapper: returns just the predicted price list (no std).
    Calls iterative_predict_mc and returns only means.
    """
    means, stds = iterative_predict_mc(model, last_seq_scaled, days, mn, mx, mc_runs=30)
    return means


# --- End compatibility wrappers ---
# ---------------------------
def create_advanced_model(timesteps, n_features, units1=128, units2=64, dropout_rate=0.3, l2_reg=1e-4,
                          gaussian_noise=1e-3):
    """
    returns a compiled Keras Model
    """
    inp = Input(shape=(timesteps, n_features))
    x = GaussianNoise(gaussian_noise)(inp)
    # First bidirectional LSTM
    x = Bidirectional(KerasLSTM(units1, return_sequences=True,
                                kernel_regularizer=regularizers.l2(l2_reg)))(x)
    x = KerasDropout(dropout_rate)(x)
    # Second (narrower)
    x = Bidirectional(KerasLSTM(units2, return_sequences=True,
                                kernel_regularizer=regularizers.l2(l2_reg)))(x)
    x = KerasDropout(dropout_rate)(x)
    # Attention pooling across timesteps
    att = AttentionLayer()(x)  # (batch, features)
    # Dense head
    d = KerasDense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(att)
    d = KerasDropout(dropout_rate * 0.5)(d)
    out = KerasDense(1)(d)  # regression -> scaled price
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='huber',
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model


# ---------------------------
# MC Dropout prediction (to estimate predictive mean & uncertainty)
# ---------------------------
def mc_dropout_predict(model, x_input, mc_runs=30):
    """
    x_input: np.array shape (1, timesteps, features)
    returns: mean_preds (list), std_preds (list) for horizon days depending on how you call iterative MC
    For iterative multi-step forecasting we call the model repeatedly while enabling training=True to keep dropout active.
    """
    preds = []
    for i in range(mc_runs):
        p = model(x_input, training=True).numpy()[0, 0]  # returns scaled prediction
        preds.append(p)
    preds = np.array(preds)
    return preds.mean(), preds.std()


# ---------------------------
# Iterative multi-step prediction with MC uncertainty
# ---------------------------
def iterative_predict_mc(model, last_seq_scaled, days, mn, mx, mc_runs=30):
    """
    last_seq_scaled: (lookback, n_features)
    returns:
      preds_mean (list of real-world prices), preds_std (list of std in price units)
    """
    seq = np.array(last_seq_scaled, dtype=float).copy()
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)
    n_timesteps, n_features = seq.shape
    preds_mean = []
    preds_std = []
    for step in range(days):
        x_inp = seq.reshape(1, n_timesteps, n_features)
        mean_s, std_s = mc_dropout_predict(model, x_inp, mc_runs=mc_runs)
        # inverse scale to price
        price_mean = inverse_scale_single_feature(mean_s, mn, mx, feature_index=0)
        price_std = std_s * (mx[0] - mn[0])  # approximate std in real units
        preds_mean.append(price_mean)
        preds_std.append(price_std)
        # create new feature row: carry-forward non-price features, put scaled mean_s into price column
        last_row = seq[-1, :].copy()
        new_row = last_row.copy()
        new_row[0] = mean_s  # scaled
        seq = np.vstack([seq[1:], new_row])
    return preds_mean, preds_std


# ---------------------------
# Training helper (with strong anti-overfitting callbacks)
# ---------------------------
def train_advanced_model(X, y, model=None, epochs=200, batch_size=64, patience=12, model_filepath=None):
    if model is None:
        timesteps, n_features = X.shape[1], X.shape[2]
        model = create_advanced_model(timesteps, n_features)
    # callbacks
    cbs = []
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(3, patience // 4), min_lr=1e-7, verbose=1)
    cbs.extend([es, rlrop])
    if model_filepath:
        ck = ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True, verbose=1)
        cbs.append(ck)
    # fit (no shuffle -> time series)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.12, shuffle=False, callbacks=cbs,
                        verbose=1)
    return model, history


# ---------------------------
# PDF generator (basit)
# ---------------------------
def generate_pdf_bytes(summary, top_table):
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        pdf.set_font('DejaVu', '', 14)
    except Exception:
        pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Crypto Dashboard', ln=True)
    try:
        pdf.set_font('DejaVu', '', 10)
    except Exception:
        pdf.set_font('Arial', '', 10)
    pdf.ln(4)
    for k, v in summary.items():
        pdf.multi_cell(0, 6, f"{k}: {v}")
    pdf.ln(6)
    pdf.cell(0, 8, 'Top coins snapshot (first 20)', ln=True)
    if top_table is None or top_table.empty:
        pdf.cell(0, 8, 'No top-table data', ln=True)
    else:
        top = top_table.head(20).copy()
        for _, r in top.iterrows():
            rank = int(r['market_cap_rank']) if not pd.isna(r.get('market_cap_rank', np.nan)) else ''
            sym = r.get('symbol', '')
            nm = (r.get('name', '') or '')[:20]
            price = format_price(r['price']) if not pd.isna(r.get('price', np.nan)) else '-'
            mcap = human_format_number(r['market_cap']) if not pd.isna(r.get('market_cap', np.nan)) else '-'
            chg = f"{r['change_24h_pct']:+.2f}%" if not pd.isna(r.get('change_24h_pct', np.nan)) else ''
            line = f"{rank:>3} {sym:<6} {nm:<20} {price:>12} {mcap:>10} {chg:>8}"
            pdf.multi_cell(0, 6, line)
    return pdf.output(dest='S').encode('latin-1')


# ---------------------------
# Sidebar & settings
# ---------------------------
st.sidebar.markdown("### Ayarlar & API Keys")
cmc_api_key_input = st.sidebar.text_input("CoinMarketCap API Key (opsiyonel)", type="password")
binance_api_key = st.sidebar.text_input("Binance API Key (opsiyonel)", type="password")
binance_api_secret = st.sidebar.text_input("Binance API Secret (opsiyonel)", type="password")
API_DELAY = st.sidebar.number_input("API çağrıları arası bekleme (saniye)", value=0.12, min_value=0.0, max_value=5.0,
                                    step=0.01)
MAX_WORKERS = st.sidebar.number_input("Paralel işçi sayısı", value=6, min_value=1, max_value=32)
st.sidebar.markdown("---")
st.sidebar.markdown("**Backtest Defaults**")
DEFAULT_RSI_PERIOD = st.sidebar.number_input("RSI Periyodu", min_value=5, max_value=50, value=14)
DEFAULT_BUY = st.sidebar.number_input("Buy threshold (RSI)", min_value=1, max_value=49, value=40)
DEFAULT_SELL = st.sidebar.number_input("Sell threshold (RSI)", min_value=51, max_value=99, value=60)
DEFAULT_INITIAL_CASH = st.sidebar.number_input("Başlangıç Bakiyesi (USD)", value=1000.0)
st.sidebar.markdown("---")
st.sidebar.markdown("**LSTM**")
LSTM_LOOKBACK = st.sidebar.number_input("LSTM lookback (days)", min_value=5, max_value=365, value=60)
LSTM_EPOCHS = st.sidebar.number_input("LSTM Epochs", min_value=100, max_value=300, value=100)
LSTM_BATCH = st.sidebar.number_input("LSTM Batch size", min_value=32, max_value=64, value=32)

menu = st.sidebar.selectbox('Menü',
                            ['Ana Sayfa', 'Coin Detay', 'Backtesting', 'Top100 Analiz', 'Model (LSTM)', 'PDF Rapor'])

# Binance client prepare
binance_client = None
if binance_api_key and binance_api_secret and BinanceClient is not None:
    binance_client = get_binance_client(binance_api_key, binance_api_secret)

# Load top100
top100_df = fetch_top_100_coins()

# ---------------------------
# Pages
# ---------------------------
if menu == 'Ana Sayfa':
    st.header("📊 Piyasa Genel Bakış (Top 100)")
    if top100_df is None or top100_df.empty:
        st.warning("Top100 verisi yok veya alınamadı.")
    else:
        df = top100_df.copy()
        df['symbol'] = df.get('symbol', pd.Series([''] * len(df))).astype(str).str.upper()
        display_cols = ['market_cap_rank', 'symbol', 'name', 'price', 'market_cap', 'volume', 'change_24h_pct',
                        'change_7d']
        display_cols = [c for c in display_cols if c in df.columns]
        col_cfg = {}
        if 'price' in display_cols: col_cfg['price'] = st.column_config.NumberColumn("Fiyat", format="$%.4f")
        if 'market_cap' in display_cols: col_cfg['market_cap'] = st.column_config.NumberColumn("Piyasa Değeri",
                                                                                               format="$%.0f")
        if 'volume' in display_cols: col_cfg['volume'] = st.column_config.NumberColumn("24s Hacim", format="$%.0f")
        if 'change_24h_pct' in display_cols: col_cfg['change_24h_pct'] = st.column_config.NumberColumn("24s % Değişim",
                                                                                                       format="%.2f%%")
        st.dataframe(df[display_cols], use_container_width=True, column_config=col_cfg)

elif menu == 'Coin Detay':
    st.header("🔎 Coin Detay & Teknik Yorum")
    # coin list: top symbols + common coingecko ids
    cand = list(top100_df['symbol'].tolist()) if (
                top100_df is not None and not top100_df.empty and 'symbol' in top100_df.columns) else []
    extras = ['bitcoin', 'ethereum', 'solana', 'ripple', 'dogecoin', 'cardano']
    options = list(dict.fromkeys(cand + extras))
    coin_sel = st.selectbox("Coin seç (symbol veya coin id)", options, index=0)
    timeframe = st.selectbox("Zaman dilimi", ['1h', '4h', '1d'], index=2)
    limit = st.slider("Kline limiti", 100, 1000, 500)

    with st.spinner("Veri çekiliyor..."):
        df_kl = pd.DataFrame()
        # deneme sırası: Binance -> Gate -> MEXC -> CoinGecko daily
        if binance_client:
            df_kl = get_binance_klines(binance_client,
                                       symbol=(coin_sel if coin_sel.endswith("USDT") else coin_sel + "USDT"),
                                       interval=timeframe, limit=limit)
        if df_kl.empty:
            df_kl = get_gateio_klines(
                (coin_sel.replace("USDT", "") + "_USDT") if coin_sel.endswith("USDT") else coin_sel + "_USDT",
                interval=timeframe, limit=limit)
        if df_kl.empty:
            df_kl = get_mexc_klines(coin_sel if coin_sel.endswith("USDT") else coin_sel + "USDT",
                                    interval='60m' if timeframe in ['1h', '60m'] else timeframe, limit=limit)
        if df_kl is None or df_kl.empty:
            # fallback CoinGecko günlük
            hist = fetch_historical_prices_coingecko(coin_sel if coin_sel in extras else coin_sel, days="max")
            if not hist.empty:
                df_plot = hist.copy()
                df_plot['open_time'] = df_plot['date']
                df_plot['close'] = df_plot['price']
            else:
                df_plot = pd.DataFrame()
        else:
            df_plot = df_kl.copy().reset_index(drop=True)
            if 'open_time' not in df_plot.columns:
                df_plot['open_time'] = pd.date_range(end=datetime.utcnow(), periods=len(df_plot))
            if 'close' not in df_plot.columns:
                df_plot['close'] = pd.to_numeric(df_plot.iloc[:, 4], errors='coerce')

    if df_plot is None or df_plot.empty:
        st.error("Veri alınamadı.")
    else:
        df_plot = df_plot.reset_index(drop=True)
        closes = pd.to_numeric(df_plot['close'], errors='coerce').dropna().reset_index(drop=True)
        times = pd.to_datetime(df_plot['open_time']).reset_index(drop=True)
        ind_df = pd.DataFrame({'open_time': times, 'close': closes})
        ind_df['SMA20'] = ind_df['close'].rolling(20).mean()
        ind_df['EMA50'] = ind_df['close'].ewm(span=50, adjust=False).mean()
        ind_df['EMA200'] = ind_df['close'].ewm(span=200, adjust=False).mean()
        ma, bb_upper, bb_lower = bollinger_bands(ind_df['close'], window=20)
        ind_df['BB_UPPER'] = bb_upper;
        ind_df['BB_LOWER'] = bb_lower
        macd_line, macd_signal, _ = compute_macd_series(ind_df['close'])
        ind_df['MACD'] = macd_line;
        ind_df['MACD_Signal'] = macd_signal
        ind_df['RSI'] = calculate_rsi_series(ind_df['close'], period=DEFAULT_RSI_PERIOD)

        # Plot price + overlays
        fig = px.line(ind_df, x='open_time', y='close', title=f"{coin_sel} Fiyat")
        if 'BB_UPPER' in ind_df.columns: fig.add_scatter(x=ind_df['open_time'], y=ind_df['BB_UPPER'], name='BB Üst',
                                                         line=dict(dash='dash'))
        if 'BB_LOWER' in ind_df.columns: fig.add_scatter(x=ind_df['open_time'], y=ind_df['BB_LOWER'], name='BB Alt',
                                                         line=dict(dash='dash'))
        if 'EMA50' in ind_df.columns: fig.add_scatter(x=ind_df['open_time'], y=ind_df['EMA50'], name='EMA50')
        if 'EMA200' in ind_df.columns: fig.add_scatter(x=ind_df['open_time'], y=ind_df['EMA200'], name='EMA200')
        st.plotly_chart(fig, use_container_width=True)

        # RSI plot
        if 'RSI' in ind_df.columns:
            fig2 = px.line(ind_df, x='open_time', y='RSI', title='RSI')
            try:
                fig2.add_hline(y=70, line_dash='dash');
                fig2.add_hline(y=30, line_dash='dash')
            except Exception:
                pass
            st.plotly_chart(fig2, use_container_width=True)

        # Otomatik Türkçe yorum
        sig = multi_indicator_signal(ind_df['close'].values, rsi_period=DEFAULT_RSI_PERIOD)
        st.subheader("Teknik Yorum")
        st.markdown(f"**Sinyal:** {sig['signal']}")
        if sig['reason']:
            st.markdown(f"**Sebep(ler):** {sig['reason']}")
        st.markdown(f"RSI: {sig['rsi']}, MACD: {sig['macd']}, BB Alt: {sig['bb_lower']}, BB Üst: {sig['bb_upper']}")

        # LSTM kısa/orta vadeli tahmin opsiyonları (7 & 30 gün)
        if tf is not None:
            st.markdown("---")
            st.subheader("LSTM Tahmini (7 & 30 gün)")
            if st.button("LSTM ile 7 & 30 günlük tahmin yap"):
                with st.spinner("Model hazırlanıyor ve tahmin yapılıyor..."):
                    prices_series = ind_df['close'].dropna().values
                    if len(prices_series) < (LSTM_LOOKBACK + 10):
                        st.warning("Yeterli veri yok (lookback + ekstra veri gerekli).")
                    else:
                        # Prepare data (çoklu feature)
                        X, y, mn, mx, scaled_features = prepare_lstm_data(prices_series, lookback=LSTM_LOOKBACK)
                        if X is None:
                            st.error("LSTM için veri hazırlanamadı.")
                        else:
                            model = create_lstm_model((X.shape[1], X.shape[2]), units1=128, units2=64, dropout=0.3)
                            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            # eğitim
                            model.fit(X, y, epochs=min(500, int(LSTM_EPOCHS)), batch_size=int(LSTM_BATCH), verbose=0,
                                      validation_split=0.1, shuffle=False, callbacks=[es])
                            # iterative predict
                            last_seq_scaled = scaled_features[-LSTM_LOOKBACK:, :]
                            preds7 = iterative_predict(model, last_seq_scaled, 7, mn, mx)
                            preds30 = iterative_predict(model, last_seq_scaled, 30, mn, mx)
                            st.write("7-günlük tahmin (son):", preds7[-1])
                            st.write("30-günlük tahmin (son):", preds30[-1])
                            # plot
                            recent_dates = pd.to_datetime(ind_df['open_time'].iloc[-LSTM_LOOKBACK:])
                            future7 = [recent_dates.iloc[-1] + pd.Timedelta(days=i + 1) for i in range(7)]
                            plt_fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(recent_dates, prices_series[-LSTM_LOOKBACK:], label='history')
                            ax.plot(future7, preds7, label='pred7', marker='o')
                            ax.legend();
                            ax.set_title('LSTM Tahmin (7 gün)');
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                            plt.xticks(rotation=45)
                            st.pyplot(plt_fig)
                            # 30-günlük
                            future30 = [recent_dates.iloc[-1] + pd.Timedelta(days=i + 1) for i in range(30)]
                            plt_fig2, ax2 = plt.subplots(figsize=(10, 4))
                            ax2.plot(recent_dates, prices_series[-LSTM_LOOKBACK:], label='history')
                            ax2.plot(future30, preds30, label='pred30', marker='o')
                            ax2.legend();
                            ax2.set_title('LSTM Tahmin (30 gün)');
                            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                            plt.xticks(rotation=45)
                            st.pyplot(plt_fig2)
        else:
            st.info("TensorFlow yüklü değil. LSTM tahmin için TensorFlow gereklidir.")

elif menu == 'Backtesting':
    st.header("📈 Backtesting")
    cg_candidates = {"bitcoin": "bitcoin", "ethereum": "ethereum", "solana": "solana", "ripple": "ripple",
                     "dogecoin": "dogecoin", "cardano": "cardano", "litecoin": "litecoin"}
    coin_bt = st.selectbox("Simülasyon Coin ", list(cg_candidates.keys()), index=0)
    days_choice = st.selectbox("Lookback tipi (preset)", ["max (tüm veri)", "365", "180", "90"], index=1)
    days_arg = "max" if days_choice == "max (tüm veri)" else int(days_choice)

    rsi_period = st.number_input("RSI periyodu", min_value=5, max_value=50, value=int(DEFAULT_RSI_PERIOD))
    buy_thresh = st.number_input("Buy threshold (RSI)", min_value=1, max_value=49, value=int(DEFAULT_BUY))
    sell_thresh = st.number_input("Sell threshold (RSI)", min_value=51, max_value=99, value=int(DEFAULT_SELL))
    initial_cash = st.number_input("Başlangıç bakiyesi (USD)", value=float(DEFAULT_INITIAL_CASH))
    trade_fee_pct = st.number_input("Trade fee (ör. 0.001 = 0.1%)", min_value=0.0, max_value=0.05, value=0.0,
                                    step=0.0005)
    use_multi = st.checkbox("Multi-indicator rule (RSI + MACD + BB)", value=True)

    hist_df = fetch_historical_prices_coingecko(cg_candidates[coin_bt], days=days_arg)
    if hist_df is None or hist_df.empty:
        st.warning("Seçilen coin için tarihsel veri bulunamadı.")
    else:
        total_days = len(hist_df)
        # slider max depends on preset: if preset != max, slider max = min(preset, total_days)
        if days_choice != "max (tüm veri)":
            preset = int(days_choice)
            slider_max = min(preset, total_days)
        else:
            slider_max = total_days
        lookback_days = st.slider("Lookback (gün) — 0 = tüm veri", min_value=0, max_value=slider_max,
                                  value=min(365, slider_max))
        lb = total_days if lookback_days == 0 else lookback_days

        if st.button("Simülasyonu Çalıştır"):
            prices = hist_df['price'].iloc[-lb:].values
            dates = hist_df['date'].iloc[-lb:].values
            trades, metrics, equity_curve, buy_points, sell_points, rsi_series = backtest_strategy_multi(
                prices, dates=dates, initial_cash=initial_cash, rsi_period=rsi_period,
                buy_threshold=buy_thresh, sell_threshold=sell_thresh, use_multi_indicator=use_multi,
                trade_fee_pct=trade_fee_pct
            )
            if metrics.get("trade_count", 0) == 0:
                st.warning("Bu aralıkta hiç işlem sinyali oluşmadı. Parametreleri veya lookback'i değiştirin.")
            st.subheader("Simülasyon Özeti")
            st.write(
                f"Başlangıç: ${initial_cash:.2f} — Bitiş: ${metrics.get('final_value', 0):.2f} — ROI: {metrics.get('ROI_pct', 0):.2f}%")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam İşlem", metrics.get("trade_count", 0))
            c2.metric("Paired Trades", metrics.get("paired_trades", 0))
            c3.metric("Win rate (naive)",
                      f"{metrics.get('win_rate', 'N/A') if not pd.isna(metrics.get('win_rate', np.nan)) else 'N/A'}")
            c4.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
            st.write(
                f"Sharpe: {metrics.get('sharpe', 'N/A') if not pd.isna(metrics.get('sharpe', np.nan)) else 'N/A'}, Sortino: {metrics.get('sortino', 'N/A') if not pd.isna(metrics.get('sortino', np.nan)) else 'N/A'}")

            # Price + trades & RSI
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            x = pd.to_datetime(dates)
            ax1.plot(x, prices, label='Price')
            if buy_points:
                bx = [x[p[0]] for p in buy_points];
                by = [p[1] for p in buy_points]
                ax1.scatter(bx, by, marker='^', color='green', s=80, zorder=5, label='Buy')
                for (idx, price) in buy_points:
                    ax1.annotate(f"AL\n{price:.4f}\n{pd.to_datetime(dates[idx]).date()}", (x[idx], price),
                                 textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
            if sell_points:
                sx = [x[p[0]] for p in sell_points];
                sy = [p[1] for p in sell_points]
                ax1.scatter(sx, sy, marker='v', color='red', s=80, zorder=5, label='Sell')
                for (idx, price) in sell_points:
                    ax1.annotate(f"SAT\n{price:.4f}\n{pd.to_datetime(dates[idx]).date()}", (x[idx], price),
                                 textcoords="offset points", xytext=(0, -30), ha="center", fontsize=8)
            ax1.set_title(f"{coin_bt.upper()} Price & Trades")
            ax1.legend(loc='upper left')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'));
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            rsi_to_plot = rsi_series.iloc[-lb:].values if len(rsi_series) >= lb else rsi_series.values
            ax2.plot(x, rsi_to_plot, label='RSI');
            ax2.axhline(y=buy_thresh, color='green', linestyle='--');
            ax2.axhline(y=sell_thresh, color='red', linestyle='--')
            ax2.set_ylim(0, 100);
            ax2.legend(loc='upper left');
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)

            # Equity curve
            fig2 = px.line(y=equity_curve, title="Equity Curve (portfolio)")
            st.plotly_chart(fig2, use_container_width=True)

            # Trades table
            if trades:
                trades_df = pd.DataFrame(trades, columns=["Type", "Index", "Price"])
                trades_df['Date'] = trades_df['Index'].apply(
                    lambda i: pd.to_datetime(dates[i]).date() if (i >= 0 and i < len(dates)) else '')
                st.subheader("İşlem Özeti (ilk 200)")
                st.dataframe(trades_df.head(200))
            else:
                st.info("İşlem yok.")

elif menu == 'Top100 Analiz':
    st.header("🔢 Top100")
    if st.button("Top100'ü Paralel Analiz Et"):
        with st.spinner("Analiz yapılıyor..."):
            top_symbols = top100_df['symbol'].tolist() if (
                        top100_df is not None and not top100_df.empty and 'symbol' in top100_df.columns) else []
            normalized = []
            for s in top_symbols:
                s2 = s.upper().replace("USDT", "").replace("_USDT", "")
                normalized.append(s2)
            df_an = analyze_symbols_parallel(normalized, interval="1h", limit=500, binance_client=binance_client,
                                             max_workers=int(MAX_WORKERS), rsi_period=int(DEFAULT_RSI_PERIOD))
            if df_an is None or df_an.empty:
                st.error("Analiz sonucu boş.")
            else:
                df_show = df_an.copy()
                col_cfg = {}
                if 'Price' in df_show.columns: col_cfg['Price'] = st.column_config.NumberColumn("Fiyat (USD)",
                                                                                                format="$%.4f")
                if 'Score' in df_show.columns:
                    col_cfg['Score'] = st.column_config.NumberColumn(
                        "Puan (0–1000)",
                        format="%.0f",
                        min_value=0,
                        max_value=1000
                    )
                st.dataframe(df_show[['Coin', 'Symbol', 'Price', 'RSI', 'MACD', 'Score']].head(200),
                             use_container_width=True, column_config=col_cfg)
                st.session_state['last_top100_analysis'] = df_an

elif menu == 'Model (LSTM)':
    st.header("🤖 LSTM Model")
    st.markdown("LSTM eğitimi CPU'da yavaş olabilir; Colab Pro GPU tavsiye edilir.")
    model_coin = st.selectbox("Model eğitimi için coin", ['bitcoin', 'ethereum', 'solana', 'ripple', 'dogecoin'])
    days_arg = st.selectbox("Kaç günlük veri (days)", ["max", 365, 730], index=1)
    train_button = st.button("LSTM Eğit & Tahmin (7 ve 30 gün)")
    if train_button:
        if tf is None:
            st.error("TensorFlow kurulmamış. `pip install tensorflow` ile yükleyin.")
        else:
            hist = fetch_historical_prices_coingecko(model_coin, days=days_arg)
            if hist is None or hist.empty:
                st.error("Tarihsel veri alınamadı.")
            else:
                prices = hist['price'].values
                if len(prices) < (LSTM_LOOKBACK + 10):
                    st.error(f"Yetersiz veri: gereken >= {LSTM_LOOKBACK + 10}, mevcut {len(prices)}")
                else:
                    X, y, mn, mx, scaled_features = prepare_lstm_data(prices, lookback=LSTM_LOOKBACK)
                    if X is None:
                        st.error("Veri hazırlanamadı.")
                    else:
                        model = create_lstm_model((X.shape[1], X.shape[2]), units1=128, units2=64, dropout=0.3)
                        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                        with st.spinner("Model eğitiliyor..."):
                            model.fit(X, y, epochs=int(LSTM_EPOCHS), batch_size=int(LSTM_BATCH), verbose=0,
                                      validation_split=0.1, shuffle=False, callbacks=[es])
                        st.success("Eğitim tamamlandı.")
                        last_seq = scaled_features[-LSTM_LOOKBACK:, :]
                        p7 = iterative_predict(model, last_seq, 7, mn, mx)
                        p30 = iterative_predict(model, last_seq, 30, mn, mx)
                        st.write("7-gün sonrası tahmin (son):", p7[-1])
                        st.write("30-gün sonrası tahmin (son):", p30[-1])
                        recent_dates = pd.to_datetime(hist['date'].iloc[-LSTM_LOOKBACK:])
                        future7 = [recent_dates.iloc[-1] + pd.Timedelta(days=i + 1) for i in range(7)]
                        future30 = [recent_dates.iloc[-1] + pd.Timedelta(days=i + 1) for i in range(30)]
                        plt_fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(recent_dates, prices[-LSTM_LOOKBACK:], label='history')
                        ax.plot(future7, p7, label='pred7', marker='o');
                        ax.legend();
                        ax.set_title('LSTM 7 gün')
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'));
                        plt.xticks(rotation=45);
                        st.pyplot(plt_fig)
                        plt_fig2, ax2 = plt.subplots(figsize=(10, 4));
                        ax2.plot(recent_dates, prices[-LSTM_LOOKBACK:], label='history');
                        ax2.plot(future30, p30, label='pred30', marker='o');
                        ax2.legend();
                        ax2.set_title('LSTM 30 gün');
                        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'));
                        plt.xticks(rotation=45);
                        st.pyplot(plt_fig2)

elif menu == 'PDF Rapor':
    st.header("📄 PDF Rapor")
    if st.button("PDF Hazırla"):
        with st.spinner("Rapor hazırlanıyor..."):
            try:
                summary = {}
                if top100_df is not None and not top100_df.empty:
                    try:
                        summary['Total Market Cap (Top100)'] = human_format_number(top100_df['market_cap'].sum())
                    except Exception:
                        summary['Total Market Cap (Top100)'] = str(top100_df['market_cap'].sum())
                    if 'change_24h_pct' in top100_df.columns:
                        top_sorted = top100_df.sort_values('change_24h_pct', ascending=False)
                        bottom_sorted = top100_df.sort_values('change_24h_pct', ascending=True)
                        if not top_sorted.empty: summary['Top Gainer (24h)'] = top_sorted.iloc[0]['symbol']
                        if not bottom_sorted.empty: summary['Top Loser (24h)'] = bottom_sorted.iloc[0]['symbol']
                else:
                    summary['note'] = 'No data'
                pdf_bytes = generate_pdf_bytes(summary, top100_df)
                st.success("PDF hazır.")
                st.download_button("PDF İndir", data=pdf_bytes, file_name="crypto_summary.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF oluşturulamadı: {e}")

# footer
st.markdown('---')
st.caption(
    'Not: Bu uygulama yatırım tavsiyesi değildir. Veriler üçüncü taraf APIlerden gelir ve gecikme/hata olabilir.')
