import os
import io  # 画像処理に必須
import time
import json
import random
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from google.generativeai.types import HarmCategory, HarmBlockThreshold
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

TRAINING_ROUNDS = 5000 # 学習回数
TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15
TRADE_BUDGET = 1000000 

# V11 Parameters
ADX_THRESHOLD = 25.0
CHOP_THRESHOLD_TREND = 38.2
CHOP_THRESHOLD_RANGE = 61.8
ATR_MULTIPLIER = 2.5
VWAP_WINDOW = 20

DEFAULT_ATR_SL_MULT = 2.0  
DEFAULT_ATR_TP_MULT = 4.0  

# 銘柄リスト
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
TRAINING_LIST = sorted(list(set(LIST_CORE + LIST_GROWTH)))

plt.rcParams['font.family'] = 'sans-serif'
genai.configure(api_key=GOOGLE_API_KEY, transport="rest")

# ==========================================
# 1. データ取得 & 指標計算
# ==========================================
def download_data_safe(ticker, period="2y", interval="1d", retries=3): 
    wait = 1
    for attempt in range(retries):
        try:
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                time.sleep(wait); wait *= 2; continue
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
            if len(df) < 200: return None
            return df
        except: time.sleep(wait); wait *= 2
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
        df['RSI'] = df['RSI'].fillna(50)

        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
        sma20_safe = sma20.replace(0, np.nan)
        df['BB_Width'] = ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20_safe * 100
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()

        tp = (high + low + close) / 3
        df['VP'] = tp * vol
        cumulative_vp = df['VP'].rolling(window=VWAP_WINDOW).sum()
        cumulative_vol = vol.rolling(window=VWAP_WINDOW).sum().replace(0, np.nan)
        df['VWAP'] = cumulative_vp / cumulative_vol
        df['VWAP_Dev'] = np.where(df['VWAP'].notna(), ((close - df['VWAP']) / df['VWAP']) * 100, 0)
        
        high_n = high.rolling(14).max(); low_n = low.rolling(14).min()
        atr_sum = tr.rolling(14).sum()
        range_n = (high_n - low_n).replace(0, np.nan)
        log_range = np.log10(range_n.replace(0, np.nan))
        log_atr = np.log10(atr_sum.replace(0, np.nan))
        df['CHOP'] = (100 * (log_atr - log_range) / np.log10(14)).fillna(50)
        
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

def calculate_metrics_v11(df, idx):
    try:
        if idx < 60 or idx >= len(df): return None
        curr = df.iloc[idx]; price = float(curr['Close'])
        
        vwap_dev = float(curr.get('VWAP_Dev', 0.0))
        chop = float(curr.get('CHOP', 50.0))
        
        div_window = 60
        start_idx = max(0, idx - div_window)
        rsi_div = detect_divergence(df['Close'].iloc[start_idx:idx+1], df['RSI'].iloc[start_idx:idx+1])
        
        adx = float(curr.get('ADX', 20.0))
        if adx > ADX_THRESHOLD and chop < CHOP_THRESHOLD_TREND: regime = "Strong Trend"
        elif adx < ADX_THRESHOLD and chop > CHOP_THRESHOLD_RANGE: regime = "Range/Chop"
        elif chop > CHOP_THRESHOLD_RANGE: regime = "Volatile Transition"
        else: regime = "Weak Trend"

        recent_high = df['High'].iloc[idx-60:idx].max()
        dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
        ma_deviation = ((price / float(curr['SMA25'])) - 1) * 100
        vol_ma = float(curr.get('Vol_MA20', 1.0))
        vol_ratio = float(curr['Volume']) / vol_ma if vol_ma > 0 else 0
        
        macd_hist = float(curr.get('MACD_Hist', 0.0))
        prev_hist = float(df['MACD_Hist'].iloc[idx-1]) if idx > 0 else 0.0
        price_vs_cloud = "Above" if price > float(curr.get('Cloud_Top', price)) else "Below"

        return {
            'price': price,
            'dist_to_res': dist_to_res,
            'ma_deviation': ma_deviation,
            'adx': adx,
            'prev_adx': float(df['ADX'].iloc[idx-1]) if idx > 0 else 0.0,
            'vol_ratio': vol_ratio,
            'atr_value': float(curr.get('ATR', price*0.01)),
            'macd_hist': macd_hist,
            'macd_trend': "Expanding" if abs(macd_hist) > abs(prev_hist) else "Shrinking",
            'price_vs_cloud': price_vs_cloud,
            'rsi': float(curr.get('RSI', 50.0)),
            'regime': regime,
            'vwap_dev': vwap_dev,
            'choppiness': chop,
            'rsi_divergence': rsi_div,
            'expansion_rate': 0.0 
        }
    except Exception: return None

def check_iron_rules_v11(metrics):
    if metrics['vol_ratio'] < 0.5: return "Volume Too Low"
    if metrics['price_vs_cloud'] == "Below" and metrics['rsi_divergence'] != "Bullish":
        return "Below Cloud (No Divergence)"
    return None

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
        
        self.csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "target_price", 
            "Actual_High", "Target_Diff", "Target_Reach",
            "Price", "adx", "ma_deviation", "vol_ratio", 
            "vwap_dev", "choppiness", "rsi", "rsi_divergence",
            "regime", "profit_rate"
        ]
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
                    global CBR_NEIGHBORS_COUNT
                    self.knn = NearestNeighbors(n_neighbors=min(CBR_NEIGHBORS_COUNT, len(valid_df)), metric='euclidean')
                    self.knn.fit(self.features_normalized)
        except Exception: pass

    def search_similar_cases(self, current_metrics):
        if self.knn is None: return "（データ不足）"
        vec = [current_metrics.get(col, 0) for col in self.feature_cols]
        input_df = pd.DataFrame([vec], columns=self.feature_cols)
        dists, indices = self.knn.kneighbors(self.scaler.transform(input_df))
        
        text = f"【類似局面(過去)】\n"
        win_c = 0; loss_c = 0
        for idx in indices[0]:
            row = self.valid_df_for_knn.iloc[idx]
            res = str(row.get('result', ''))
            if res == 'WIN': win_c += 1
            if res == 'LOSS': loss_c += 1
        rate = win_c / (win_c + loss_c) * 100 if (win_c + loss_c) > 0 else 0
        text += f"-> 勝率: {rate:.0f}% (勝{win_c}/負{loss_c})\n"
        return text

    def save_experience(self, data_dict):
        new_df = pd.DataFrame([data_dict])
        for col in self.csv_columns:
            if col not in new_df.columns: new_df[col] = None
        save_cols = [c for c in self.csv_columns if c in new_df.columns]
        new_df = new_df[save_cols]
        try:
            if not os.path.exists(self.csv_path):
                new_df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
            else:
                new_df.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            self.load_and_train() 
        except Exception as e: print(f"❌ 保存エラー: {e}")

# ==========================================
# 3. AIエージェント
# ==========================================
def create_chart_image(df, name):
    data = df.tail(80).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    ax1.plot(data.index, data['Close'], color='black', label='Close')
    ax1.plot(data.index, sma20 + 2*std20, color='green', alpha=0.5, linestyle='--', label='+2σ')
    ax1.plot(data.index, sma20 - 2*std20, color='green', alpha=0.5, linestyle='--', label='-2σ')
    
    if 'VWAP' in data.columns:
        ax1.plot(data.index, data['VWAP'], color='orange', alpha=0.7, linestyle='--', label='VWAP')
    if 'Cloud_Top' in data.columns:
        ax1.plot(data.index, data['Cloud_Top'], color='blue', alpha=0.2, label='Cloud Top')
        ax1.fill_between(data.index, data['Cloud_Top'], data['Close'].min(), color='blue', alpha=0.05)

    ax1.set_title(f"{name} V11 Chart")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
    
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def ai_decision_maker_v11(model, chart_bytes, metrics, cbr_text, ticker):
    rsi_context = f"{metrics['rsi']:.1f}"
    if metrics['rsi'] > 70: rsi_context += " (Overbought)"
    elif metrics['rsi'] < 30: rsi_context += " (Oversold)"
    
    chop_context = f"{metrics['choppiness']:.1f}"
    if metrics['choppiness'] < 38.2: chop_context += " (Trending)"
    elif metrics['choppiness'] > 61.8: chop_context += " (Choppy/Range)"
    
    prompt = f"""
### Role
あなたはヘッジファンドのシニア・クオンツです。リスク管理を最優先します。

### Input Data
銘柄: {ticker} (現在値: {metrics['price']:.0f}円)

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

### Task
1. 局面分析: トレンドかレンジか？
2. シグナル統合: 反転または継続のサインは？
3. リスク評価: 勝算はあるか？

### Output Requirement (JSON ONLY)
{{
  "thought_process": "...",
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "sl_multiplier": {DEFAULT_ATR_SL_MULT},
  "tp_multiplier": {DEFAULT_ATR_TP_MULT},
  "reason": "理由(50文字以内)"
}}
"""
    safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}], safety_settings=safety)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except Exception:
        return {"action": "HOLD", "reason": "AI Error", "confidence": 0}

# ==========================================
# 4. メイン実行
# ==========================================
def main():
    start_time = time.time()
    print(f"=== AI強化合宿 [AGGRESSIVE V11] (Fixed) ===")
    
    memory_system = CaseBasedMemory(LOG_FILE) 
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e: print(f"Model Init Error: {e}"); return

    print("データ取得中...", end="")
    processed_data = {}
    for i, t in enumerate(TRAINING_LIST):
        if i % 10 == 0: print(f"  - {i}/{len(TRAINING_LIST)}")
        df = download_data_safe(t)
        if df is None: continue
        df = calculate_technical_indicators_v11(df)
        if df is not None:
            processed_data[t] = df

    if not processed_data: print("データ不足"); return
    print(f"完了 ({int(time.time() - start_time)}秒)")

    win_count = 0; loss_count = 0
    print(f"\n🥊 トレーニング開始 ({TRAINING_ROUNDS}ラウンド)\n")
    
    for i in range(1, TRAINING_ROUNDS + 1):
        ticker = random.choice(list(processed_data.keys()))
        df = processed_data[ticker]
        
        max_idx = len(df) - 65
        if max_idx < 100: continue
        target_idx = random.randint(100, max_idx) 
        
        metrics = calculate_metrics_v11(df, target_idx)
        if metrics is None: continue

        iron_rule = check_iron_rules_v11(metrics)
        if iron_rule: continue

        cbr_text = memory_system.search_similar_cases(metrics)
        past_df = df.iloc[:target_idx+1]
        chart_bytes = create_chart_image(past_df, ticker)
        
        decision = ai_decision_maker_v11(model_instance, chart_bytes, metrics, cbr_text, ticker)
        
        action = decision.get('action', decision.get('decision', 'HOLD'))
        conf = decision.get('confidence', 0)

        if action == "HOLD": continue

        entry_price = float(metrics['price'])
        atr = metrics['atr_value']
        
        sl_mult = float(decision.get('sl_multiplier', DEFAULT_ATR_SL_MULT))
        tp_mult = float(decision.get('tp_multiplier', DEFAULT_ATR_TP_MULT))
        current_stop_loss = entry_price - (atr * sl_mult)
        target_price = entry_price + (atr * tp_mult)
        
        print(f"Round {i:03}: {ticker} -> BUY (Regime: {metrics['regime']}) 🔴")
        
        shares = int(TRADE_BUDGET // entry_price)
        if shares < 1: shares = 1
        
        future_prices = df.iloc[target_idx+1 : target_idx+61]
        if future_prices.empty: continue

        result = "DRAW"
        final_exit_price = entry_price
        max_price = entry_price
        is_loss = False
        is_win = False
        
        for _, row in future_prices.iterrows():
            high = row['High']; low = row['Low']
            
            if low <= current_stop_loss:
                is_loss = True
                final_exit_price = current_stop_loss
                break
            
            if high > max_price:
                max_price = high
                new_stop = max_price - (atr * sl_mult) # トレーリング
                if new_stop > current_stop_loss:
                    current_stop_loss = new_stop
            
            if high >= target_price:
                is_win = True
                final_exit_price = target_price
                break

        if not is_loss and not is_win:
            final_exit_price = future_prices['Close'].iloc[-1]

        profit_loss = (final_exit_price - entry_price) * shares
        profit_rate = ((final_exit_price - entry_price) / entry_price) * 100
        
        if profit_loss > 0: result = "WIN"; win_count += 1
        elif profit_loss < 0: result = "LOSS"; loss_count += 1

        print(f"   結果: {result} (PL: {profit_loss:+.0f}円 / {profit_rate:+.2f}%)")

        save_data = {
            'Date': df.index[target_idx].strftime('%Y-%m-%d'), 
            'Ticker': ticker, 'Timeframe': TIMEFRAME, 
            'Action': action, 'result': result, 
            'Reason': decision.get('reason', 'None'),
            'Confidence': conf, 
            'stop_loss_price': current_stop_loss, 
            'target_price': target_price, 
            'Actual_High': future_prices['High'].max(), 'Target_Diff': 0, 'Target_Reach': 0,
            'Price': metrics['price'], 
            'adx': metrics['adx'], 
            'ma_deviation': metrics['ma_deviation'], 
            'vol_ratio': metrics['vol_ratio'], 
            'vwap_dev': metrics['vwap_dev'],
            'choppiness': metrics['choppiness'],
            'rsi': metrics['rsi'],
            'rsi_divergence': metrics['rsi_divergence'],
            'regime': metrics['regime'],
            'profit_rate': profit_rate 
        }
        memory_system.save_experience(save_data)
        time.sleep(1)

    elapsed_time = time.time() - start_time
    print(f"\n=== 合宿終了 ({str(datetime.timedelta(seconds=int(elapsed_time)))}) ===")
    print(f"戦績 (BUY): {win_count}勝 {loss_count}敗")

if __name__ == "__main__":
    main()