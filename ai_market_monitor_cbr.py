import os
import time
import json
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import io
import re
import logging
import socket
import requests.packages.urllib3.util.connection as urllib3_cn
from scipy.signal import argrelextrema
import warnings

# ---------------------------------------------------------
# ★環境設定
# ---------------------------------------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

def allowed_gai_family():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = allowed_gai_family

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")

# 設定
LOG_FILE = "ai_trade_memory_aggressive_v11.csv" 
MODEL_NAME = 'models/gemini-2.0-flash'

# V11 Parameters
ADX_THRESHOLD = 25.0
CHOP_THRESHOLD_TREND = 38.2
CHOP_THRESHOLD_RANGE = 61.8
VWAP_WINDOW = 20
ATR_MULTIPLIER = 2.5 
DEFAULT_ATR_SL_MULT = 2.0  
DEFAULT_ATR_TP_MULT = 4.0 

# 銘柄リスト (Core + Growth)
LIST_CORE = [
    "8035.T", "6857.T", "6146.T", "6920.T", "6758.T", "6702.T", "6501.T", "6503.T", "7751.T", "4063.T", "6981.T", "6723.T",
    "7203.T", "7267.T", "6902.T", "6301.T", "6367.T", "7011.T", "7013.T", 
    "8306.T", "8316.T", "8411.T", "8766.T", "8058.T", "8001.T", "8031.T", "8002.T", "9984.T",
    "9432.T", "9983.T", "4568.T", "4543.T", "4661.T", "7974.T", "6506.T"
]
LIST_GROWTH = [
    "5253.T", "5032.T", "9166.T", "4385.T", "4478.T", "4483.T", "3993.T", "4180.T", "3687.T", "6027.T",
    "5595.T", "9348.T", "7012.T", "6203.T", 
    "6254.T", "6315.T", "6526.T", "6228.T", "6963.T", "3436.T", "7735.T", "6890.T",
    "2768.T", "7342.T", "2413.T", "2222.T", "7532.T", "3092.T",
    "9101.T", "9104.T", "9107.T", "1605.T", "5713.T", "5401.T", "5411.T"
]
WATCH_LIST = sorted(list(set(LIST_CORE + LIST_GROWTH)))

plt.rcParams['font.family'] = 'sans-serif'
genai.configure(api_key=GOOGLE_API_KEY, transport="rest")

# ==========================================
# 1. データ取得 (テクニカル & ファンダメンタル)
# ==========================================
def download_data_safe(ticker, period="1y", interval="1d"): 
    try:
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if len(df) < 100: return None
        return df
    except: return None

# ★追加機能: ファンダメンタルズ情報の取得
def get_fundamental_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # 必要な情報のみ抽出 (取得できない場合は None や 0 を設定)
        fundamentals = {
            "MarketCap(Trill)": info.get('marketCap', 0) / 1000000000000 if info.get('marketCap') else 0, # 兆円単位
            "PER": info.get('trailingPE'),          # 株価収益率
            "PBR": info.get('priceToBook'),         # 株価純資産倍率
            "ROE": info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0, # %
            "RevenueGrowth": info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0, # 売上成長率(%)
            "Sector": info.get('sector', 'Unknown'),
            "Industry": info.get('industry', 'Unknown')
        }
        return fundamentals
    except Exception as e:
        return None

def calculate_technical_indicators_v11(df):
    try:
        df = df.copy()
        close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']
        
        df['SMA25'] = close.rolling(25).mean()
        
        tr1 = high - low; tr2 = abs(high - close.shift(1)); tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()
        plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
        tr_smooth = tr.rolling(14).mean().replace(0, np.nan)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / tr_smooth)
        denom = (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = (abs(plus_di - minus_di) / denom) * 100
        df['ADX'] = df['ADX'].rolling(14).mean()

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
        df['BB_Width'] = ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20 * 100
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()

        tp = (high + low + close) / 3
        df['VP'] = tp * vol
        df['VWAP'] = df['VP'].rolling(window=VWAP_WINDOW).sum() / vol.rolling(window=VWAP_WINDOW).sum().replace(0, np.nan)
        df['VWAP_Dev'] = ((close - df['VWAP']) / df['VWAP']) * 100
        
        high_n = high.rolling(14).max(); low_n = low.rolling(14).min()
        atr_sum = tr.rolling(14).sum()
        range_n = (high_n - low_n).replace(0, np.nan)
        df['CHOP'] = (100 * np.log10(atr_sum / range_n) / np.log10(14)).fillna(50)
        
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        df['Cloud_Top'] = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)

        return df.dropna()
    except Exception: return None

def detect_divergence(series_price, series_osc, order=5):
    try:
        if len(series_price) < order * 2: return "None"
        min_idx = argrelextrema(series_price.values, np.less_equal, order=order)[0]
        if len(min_idx) >= 2:
            curr, prev = min_idx[-1], min_idx[-2]
            if series_price.iloc[curr] < series_price.iloc[prev] and series_osc.iloc[curr] > series_osc.iloc[prev]:
                return "Bullish"
        return "None"
    except: return "None"

def get_latest_metrics(df):
    try:
        curr = df.iloc[-1]
        price = float(curr['Close'])
        idx = len(df) - 1
        
        vwap_dev = float(curr.get('VWAP_Dev', 0.0))
        chop = float(curr.get('CHOP', 50.0))
        
        div_window = 60
        start_idx = max(0, idx - div_window)
        rsi_div = detect_divergence(df['Close'].iloc[start_idx:], df['RSI'].iloc[start_idx:])
        
        adx = float(curr.get('ADX', 20.0))
        if adx > ADX_THRESHOLD and chop < CHOP_THRESHOLD_TREND: regime = "Strong Trend"
        elif adx < ADX_THRESHOLD and chop > CHOP_THRESHOLD_RANGE: regime = "Range/Chop"
        elif chop > CHOP_THRESHOLD_RANGE: regime = "Volatile Transition"
        else: regime = "Weak Trend"

        recent_high = df['High'].iloc[-60:].max()
        dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
        ma_deviation = ((price / float(curr['SMA25'])) - 1) * 100
        vol_ratio = float(curr['Volume']) / float(curr['Vol_MA20']) if float(curr['Vol_MA20']) > 0 else 0
        
        macd_hist = float(curr['MACD_Hist'])
        prev_hist = float(df['MACD_Hist'].iloc[-2])
        price_vs_cloud = "Above" if price > float(curr['Cloud_Top']) else "Below"

        return {
            'price': price,
            'dist_to_res': dist_to_res,
            'ma_deviation': ma_deviation,
            'adx': adx,
            'prev_adx': float(df['ADX'].iloc[-2]),
            'vol_ratio': vol_ratio,
            'atr_value': float(curr['ATR']),
            'macd_hist': macd_hist,
            'macd_trend': "Expanding" if abs(macd_hist) > abs(prev_hist) else "Shrinking",
            'price_vs_cloud': price_vs_cloud,
            'rsi': float(curr['RSI']),
            'regime': regime,
            'vwap_dev': vwap_dev,
            'choppiness': chop,
            'rsi_divergence': rsi_div,
            'date': str(curr.name.date())
        }
    except Exception: return None

# ==========================================
# 2. メモリシステム
# ==========================================
class CaseBasedMemory:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        self.feature_cols = ['adx', 'ma_deviation', 'vol_ratio', 'vwap_dev', 'choppiness', 'rsi']
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df.columns = [c.strip() for c in self.df.columns]
            if 'result' in self.df.columns:
                valid_df = self.df[self.df['result'].isin(['WIN', 'LOSS'])].copy()
                if len(valid_df) > 5:
                    for col in self.feature_cols:
                        if col not in valid_df.columns: valid_df[col] = 0
                    features = valid_df[self.feature_cols].fillna(0)
                    self.features_normalized = self.scaler.fit_transform(features)
                    self.valid_df_for_knn = valid_df 
                    self.knn = NearestNeighbors(n_neighbors=min(15, len(valid_df)), metric='euclidean')
                    self.knn.fit(self.features_normalized)
        except Exception: pass

    def get_similar_cases_text(self, current_metrics):
        if self.knn is None: return "（データ不足）"
        vec = [current_metrics.get(col, 0) for col in self.feature_cols]
        input_df = pd.DataFrame([vec], columns=self.feature_cols)
        dists, indices = self.knn.kneighbors(self.scaler.transform(input_df))
        
        text = f"【類似局面】"
        win_c = 0; loss_c = 0
        for idx in indices[0]:
            row = self.valid_df_for_knn.iloc[idx]
            res = str(row.get('result', ''))
            if res == 'WIN': win_c += 1
            if res == 'LOSS': loss_c += 1
        rate = win_c / (win_c + loss_c) * 100 if (win_c + loss_c) > 0 else 0
        text += f"勝率{rate:.0f}% (勝{win_c}/負{loss_c})"
        return text

# ==========================================
# 3. AI判断 (ファンダメンタルズ対応)
# ==========================================
def create_chart_image(df, name):
    data = df.tail(80).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    ax1.plot(data.index, data['Close'], color='black', label='Close')
    ax1.plot(data.index, sma20 + 2*std20, color='green', alpha=0.5, linestyle='--', label='+2σ')
    ax1.plot(data.index, sma20 - 2*std20, color='green', alpha=0.5, linestyle='--', label='-2σ')
    if 'VWAP' in data.columns: ax1.plot(data.index, data['VWAP'], color='orange', alpha=0.7, linestyle='--')
    if 'Cloud_Top' in data.columns:
        ax1.plot(data.index, data['Cloud_Top'], color='blue', alpha=0.2)
        ax1.fill_between(data.index, data['Cloud_Top'], data['Close'].min(), color='blue', alpha=0.05)
    ax1.set_title(f"{name} V11")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def ai_decision_maker_v11(model, chart_bytes, metrics, cbr_text, ticker, fundamentals):
    rsi_context = f"{metrics['rsi']:.1f}"
    if metrics['rsi'] > 70: rsi_context += " (Overbought)"
    elif metrics['rsi'] < 30: rsi_context += " (Oversold)"
    
    chop_context = f"{metrics['choppiness']:.1f}"
    if metrics['choppiness'] < 38.2: chop_context += " (Trending)"
    elif metrics['choppiness'] > 61.8: chop_context += " (Choppy/Range)"
    
    # ファンダメンタルズ情報のテキスト化
    fund_text = "N/A"
    if fundamentals:
        fund_text = f"""
- Sector: {fundamentals['Sector']}
- Market Cap: {fundamentals['MarketCap(Trill)']:.2f} 兆円
- PER: {fundamentals['PER'] if fundamentals['PER'] else 'N/A'}
- PBR: {fundamentals['PBR'] if fundamentals['PBR'] else 'N/A'}
- ROE: {fundamentals['ROE']:.2f}%
- Revenue Growth: {fundamentals['RevenueGrowth']:.2f}%
"""

    prompt = f"""
### Role
あなたはヘッジファンドのシニア・クオンツです。テクニカルとファンダメンタルズの両面から分析します。
リスク管理を最優先します。
### Input Data
銘柄: {ticker} (現在値: {metrics['price']:.0f}円)

[Fundamental Data]
{fund_text}

[Market Regime]
- Status: **{metrics['regime']}**
- ADX: {metrics['adx']:.1f}
- Choppiness Index: {chop_context}

[Advanced Indicators]
- VWAP Deviation: {metrics['vwap_dev']:.2f}%
- RSI(14): {rsi_context}
- RSI Divergence: **{metrics['rsi_divergence']}**
- Cloud Position: {metrics['price_vs_cloud']}

{cbr_text}

### Output Requirement (JSON ONLY)
{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "sl_multiplier": {DEFAULT_ATR_SL_MULT},
  "tp_multiplier": {DEFAULT_ATR_TP_MULT},
  "reason": "ファンダメンタルズ(割安性・成長性)とテクニカルを組み合わせた理由(50文字以内)"
}}
"""
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}])
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: return {"action": "HOLD", "confidence": 0}

def main():
    print(f"=== 📡 V11 Market Monitor (Fundamentals Enhanced) ===")
    print(f"Scanning {len(WATCH_LIST)} tickers...")
    
    memory = CaseBasedMemory(LOG_FILE)
    try: model = genai.GenerativeModel(MODEL_NAME)
    except: print("Model Error"); return

    # 市場全体の環境認識
    nikkei = download_data_safe("^N225")
    market_mode = "NEUTRAL"
    if nikkei is not None:
        nikkei = calculate_technical_indicators_v11(nikkei)
        if nikkei is not None:
            n_metrics = get_latest_metrics(nikkei)
            if n_metrics:
                adx = n_metrics['adx']
                if adx >= 25: market_mode = "HOMERUN (Trend)"
                else: market_mode = "GUERRILLA (Range)"
    print(f"🌍 Market Mode: {market_mode}")

    print("\n🔍 Scanning Candidates...")
    candidates = []
    
    for i, ticker in enumerate(WATCH_LIST):
        print(f"\rProgress: {i+1}/{len(WATCH_LIST)} ({ticker})", end="")
        df = download_data_safe(ticker)
        if df is None: continue
        
        df = calculate_technical_indicators_v11(df)
        if df is None: continue
        
        metrics = get_latest_metrics(df)
        if metrics is None: continue
        
        # 1. テクニカルフィルター (足切り)
        if metrics['vol_ratio'] < 0.5: continue
        if metrics['price_vs_cloud'] == "Below" and metrics['rsi_divergence'] != "Bullish": continue

        # 2. 通過したらファンダメンタルズ取得
        fundamentals = get_fundamental_info(ticker)
        
        # 3. AI分析 (Fundamentals込み)
        cbr_text = memory.get_similar_cases_text(metrics)
        chart_bytes = create_chart_image(df, ticker)
        decision = ai_decision_maker_v11(model, chart_bytes, metrics, cbr_text, ticker, fundamentals)
        
        if decision.get('action') == "BUY" and decision.get('confidence', 0) >= 70:
            atr = metrics['atr_value']
            sl_mult = float(decision.get('sl_multiplier', 2.0))
            sl_price = metrics['price'] - (atr * sl_mult)
            
            per_str = str(fundamentals['PER']) if fundamentals and fundamentals['PER'] else "-"
            pbr_str = str(fundamentals['PBR']) if fundamentals and fundamentals['PBR'] else "-"

            candidates.append({
                'Ticker': ticker,
                'Price': metrics['price'],
                'Conf': decision['confidence'],
                'Reason': decision['reason'],
                'SL': sl_price,
                'PER': per_str,
                'PBR': pbr_str,
                'Regime': metrics['regime'],
                'Date': metrics['date']
            })
            print(f" -> FOUND! {ticker} (PER:{per_str})")

    print("\n\n" + "="*60)
    print(f"🚀 明日の推奨銘柄リスト ({len(candidates)}銘柄)")
    print("="*60)
    
    if candidates:
        df_res = pd.DataFrame(candidates)
        # Markdownで見やすく表示
        print(df_res[['Ticker', 'Price', 'Conf', 'SL', 'PER', 'PBR', 'Regime', 'Reason']].to_markdown(index=False))
        
        today = datetime.datetime.now().strftime('%Y%m%d')
        filename = f"market_scan_v11_fund_{today}.csv"
        df_res.to_csv(filename, index=False)
        print(f"\n📄 Saved to {filename}")
    else:
        print("該当なし。明日は様子見推奨です。")

if __name__ == "__main__":
    main()