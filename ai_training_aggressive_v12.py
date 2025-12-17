import os
import io
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

# ★ V12 設定
LOG_FILE = "ai_trade_memory_aggressive_v12.csv"
MODEL_NAME = 'models/gemini-2.0-flash'

TRAINING_ROUNDS = 2000
TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15
TRADE_BUDGET = 1000000 

# ★ V12 ロジックパラメータ (分析結果に基づく最適化)
# トレンドの「初動」を捉えるための狭いADXレンジ
ADX_MIN = 20.0
ADX_MAX = 40.0 # 40以上は過熱感ありとして避ける
ROC_MAX = 15.0 # 短期急騰は避ける

# リスク管理
ATR_MULTIPLIER = 2.5
VWAP_WINDOW = 20

# 銘柄リスト (V9拡張版を継承)
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
# 1. データ取得
# ==========================================
def download_data_safe(ticker, period="5y", interval="1d", retries=3): 
    wait = 1
    for attempt in range(retries):
        try:
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            print(f"   Downloading {ticker}...", end="")
            
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            
            if df.empty:
                print(" -> Empty")
                time.sleep(wait); wait *= 2
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
            
            if len(df) < 200:
                print(f" -> Too short ({len(df)})")
                return None
            
            print(f" -> OK ({len(df)})")
            return df
        except Exception as e:
            print(f" -> Error: {e}")
            time.sleep(wait); wait *= 2
    return None

def calculate_technical_indicators_v12(df):
    """V12仕様: ROC, MFIを追加"""
    try:
        df = df.copy()
        close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']
        
        # SMA
        df['SMA25'] = close.rolling(25).mean()
        
        # ATR (14)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # ADX (14)
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

        # RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)

        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        # ROC (Rate of Change) - 10日間の変化率
        df['ROC'] = close.pct_change(10) * 100

        # MFI (Money Flow Index) - 14日
        typical_price = (high + low + close) / 3
        money_flow = typical_price * vol
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow.replace(0, np.nan)
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        df['MFI'] = df['MFI'].fillna(50)

        # VWAP
        df['VP'] = typical_price * vol
        cumulative_vp = df['VP'].rolling(window=VWAP_WINDOW).sum()
        cumulative_vol = vol.rolling(window=VWAP_WINDOW).sum().replace(0, np.nan)
        df['VWAP'] = cumulative_vp / cumulative_vol
        df['VWAP_Dev'] = np.where(df['VWAP'].notna(), ((close - df['VWAP']) / df['VWAP']) * 100, 0)
        
        # Cloud
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        df['Cloud_Top'] = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)

        return df.dropna()
    except Exception as e:
        # print(f"Calc Error: {e}")
        return None

def calculate_metrics_v12(df, idx):
    try:
        if idx < 60 or idx >= len(df): return None
        curr = df.iloc[idx]
        price = float(curr['Close'])
        
        # V12 Metrics
        adx = float(curr.get('ADX', 20.0))
        roc = float(curr.get('ROC', 0.0))
        mfi = float(curr.get('MFI', 50.0))
        
        # 局面判定 (V12 Optimized)
        # ADX 20-40 を「初動～成熟期」として狙う
        if ADX_MIN <= adx <= ADX_MAX:
            regime = "Trend Start/Growth"
        elif adx > ADX_MAX:
            regime = "Overheated Trend"
        else:
            regime = "Range/Weak"

        recent_high = df['High'].iloc[idx-60:idx].max()
        dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
        ma_deviation = ((price / float(curr['SMA25'])) - 1) * 100
        
        macd_hist = float(curr.get('MACD_Hist', 0.0))
        prev_hist = float(df['MACD_Hist'].iloc[idx-1]) if idx > 0 else 0.0
        
        cloud_top = float(curr.get('Cloud_Top', price))
        price_vs_cloud = "Above" if price > cloud_top else "Below"

        return {
            'price': price,
            'dist_to_res': dist_to_res,
            'ma_deviation': ma_deviation,
            'adx': adx,
            'roc': roc, # V12
            'mfi': mfi, # V12
            'atr_value': float(curr.get('ATR', price*0.01)),
            'macd_hist': macd_hist,
            'macd_trend': "Expanding" if abs(macd_hist) > abs(prev_hist) else "Shrinking",
            'price_vs_cloud': price_vs_cloud,
            'rsi': float(curr.get('RSI', 50.0)),
            'regime': regime,
            'vwap_dev': float(curr.get('VWAP_Dev', 0.0))
        }
    except Exception: return None

def check_iron_rules_v12(metrics):
    # V12の改善点: 高値掴み防止
    if metrics['roc'] > ROC_MAX: return f"ROC Too High ({metrics['roc']:.1f}%)"
    if metrics['adx'] > 50: return "ADX Overheat (>50)"
    if metrics['price_vs_cloud'] == "Below": return "Below Cloud"
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
        # V12特徴量
        self.feature_cols = ['adx', 'roc', 'mfi', 'vwap_dev', 'rsi']
        self.csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "target_price", 
            "Actual_High", "Price", 
            "adx", "roc", "mfi", "vwap_dev", "rsi", # V12
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
        try:
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
        except: return "（検索エラー）"

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
        except Exception: pass

# ==========================================
# 3. AIエージェント
# ==========================================
def create_chart_image(df, name):
    try:
        data = df.tail(80).copy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        ax1.plot(data.index, data['Close'], color='black', label='Close')
        if 'SMA25' in data.columns:
            ax1.plot(data.index, data['SMA25'], color='green', alpha=0.5, label='SMA25')
        if 'VWAP' in data.columns:
            ax1.plot(data.index, data['VWAP'], color='orange', alpha=0.7, linestyle='--', label='VWAP')
        if 'Cloud_Top' in data.columns:
            ax1.fill_between(data.index, data['Cloud_Top'], data['Close'].min(), color='blue', alpha=0.05)

        ax1.set_title(f"{name} V12 Early Trend Hunter")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
        
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception: return None

def ai_decision_maker_v12(model, chart_bytes, metrics, cbr_text, ticker):
    # V12 Prompt: Early Trend Hunter
    prompt = f"""
### Role
あなたは「トレンド初動ハンター」です。成熟したトレンド（高値掴み）を避け、これから伸びる「初動」のみを狙います。

### Input Data
銘柄: {ticker} (現在値: {metrics['price']:.0f}円)

[Early Trend Indicators]
- **ADX**: {metrics['adx']:.1f} (理想: 20-35)
- **ROC(10)**: {metrics['roc']:.1f}% (高すぎると危険)
- **MFI**: {metrics['mfi']:.1f} (資金流入)
- **Regime**: **{metrics['regime']}**

[Confirmation]
- VWAP Deviation: {metrics['vwap_dev']:.2f}%
- RSI(14): {metrics['rsi']:.1f}
- Cloud Position: {metrics['price_vs_cloud']}

{cbr_text}

### Task
1. **初動判定**: ADXは上昇傾向にあり、かつ過熱しすぎていないか？
2. **押し目確認**: ROCが高すぎないか？ VWAP付近での反発か？
3. **リスク評価**: 過去の勝率は？

### Output Requirement (JSON ONLY)
{{
  "thought_process": "...",
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "sl_multiplier": 2.5,
  "tp_multiplier": 5.0,
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
# 4. メイン実行 (トレーニングモード)
# ==========================================
def main():
    start_time = time.time()
    print(f"=== AI強化合宿 [AGGRESSIVE V12] (Early Trend Hunter) ===")
    
    memory_system = CaseBasedMemory(LOG_FILE) 
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e: print(f"Model Init Error: {e}"); return

    print("データ取得中...", end="")
    processed_data = {}
    
    for i, t in enumerate(TRAINING_LIST):
        df = download_data_safe(t, period="5y") 
        if df is None: continue
        
        df = calculate_technical_indicators_v12(df)
        if df is not None:
            processed_data[t] = df
            print(".", end="", flush=True)
            
    print(f"\nデータ取得完了 ({len(processed_data)}銘柄)")

    if not processed_data:
        print("有効なデータがありません。処理を終了します。")
        return

    win_count = 0; loss_count = 0
    
    print(f"\n🥊 トレーニング開始 ({TRAINING_ROUNDS}ラウンド)\n")
    
    for i in range(1, TRAINING_ROUNDS + 1):
        ticker = random.choice(list(processed_data.keys()))
        df = processed_data[ticker]
        
        max_idx = len(df) - 65
        if max_idx < 100: continue
        
        target_idx = random.randint(100, max_idx) 
        metrics = calculate_metrics_v12(df, target_idx)
        
        if metrics is None: continue

        iron_rule = check_iron_rules_v12(metrics)
        if iron_rule: continue

        cbr_text = memory_system.search_similar_cases(metrics)
        past_df = df.iloc[:target_idx+1]
        chart_bytes = create_chart_image(past_df, ticker)
        
        decision = ai_decision_maker_v12(model_instance, chart_bytes, metrics, cbr_text, ticker)
        
        action = decision.get('action', decision.get('decision', 'HOLD'))
        conf = decision.get('confidence', 0)

        if action == "HOLD": continue

        entry_price = float(metrics['price'])
        atr = metrics['atr_value']
        
        # V12: 固定倍率ではなくAI推奨倍率を採用
        sl_mult = float(decision.get('sl_multiplier', ATR_MULTIPLIER))
        current_stop_loss = entry_price - (atr * sl_mult)
        
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
        
        ai_target = decision.get('target_price', 0) # V12はTPもAIに委ねる

        for _, row in future_prices.iterrows():
            high = row['High']; low = row['Low']; close = row['Close']
            
            if low <= current_stop_loss:
                is_loss = True
                final_exit_price = current_stop_loss
                break
            
            # トレーリングストップ
            if high > max_price:
                max_price = high
                new_stop = max_price - (atr * sl_mult)
                if new_stop > current_stop_loss:
                    current_stop_loss = new_stop
            
            # AIターゲットがあれば利確
            if ai_target > 0 and high >= ai_target:
                is_win = True
                final_exit_price = ai_target
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
            'target_price': ai_target, 
            'Actual_High': future_prices['High'].max(), 
            'Price': metrics['price'], 
            'adx': metrics['adx'], 
            'roc': metrics['roc'], # V12
            'mfi': metrics['mfi'], # V12
            'vwap_dev': metrics['vwap_dev'], 
            'rsi': metrics['rsi'],
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