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

# ==========================================
# ★設定エリア
# ==========================================
GOOGLE_API_KEY = "AIzaSyDsOgmFFantDOzD6scaNNVal1hDg9TGsNE".strip() 
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = 'models/gemini-2.5-pro' 
LOG_FILE = "ai_trade_memory_risk_managed.csv" 

TRAINING_ROUNDS = 50
TIMEFRAME = "1d" 
CBR_NEIGHBORS_COUNT = 11

# ★低ボラティリティ足切りライン (これ未満は取引しない)
MIN_VOLATILITY = 1.0 

TRAINING_LIST = [
    "8035.T", "6146.T", "6920.T", "6857.T", "6723.T", "7735.T", "6526.T",
    "6758.T", "6861.T", "6501.T", "6503.T", "6981.T", "6954.T", "7741.T", 
    "6902.T", "6367.T", "6594.T", "7751.T", "7203.T", "7267.T", "7270.T", 
    "8306.T", "8316.T", "8411.T", "8766.T", "8725.T", "8591.T", "8604.T",
    "8058.T", "8031.T", "8001.T", "8002.T", "8015.T", "2768.T",
    "7011.T", "7012.T", "7013.T", "6301.T", "5401.T", "9101.T", "9104.T", "9107.T",
    "9432.T", "9433.T", "9984.T", "9434.T", "4661.T", "6098.T",
    "7974.T", "9684.T", "9697.T", "7832.T", "9983.T", "3382.T", "8267.T", 
    "9843.T", "3092.T", "4385.T", "7532.T", "4568.T", "4519.T", "4503.T", 
    "4502.T", "4063.T", "4901.T", "4452.T", "2914.T", "8801.T", "8802.T", 
    "1925.T", "1801.T", "9501.T", "9503.T", "1605.T", "5020.T",
    "9020.T", "9202.T", "2802.T"
]

plt.rcParams['font.family'] = 'sans-serif' 

# ==========================================
# 1. データ取得 & テクニカル計算
# ==========================================
def download_data_safe(ticker, period="3y", interval="1d", retries=3):
    wait = 2
    for _ in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: raise ValueError("Empty Data")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except:
            time.sleep(wait); wait *= 2
    return None

def calculate_metrics_enhanced(df, idx):
    curr = df.iloc[idx]
    price = float(curr['Close'])
    
    sma25 = float(curr['SMA25'])
    sma25_dev = ((price / sma25) - 1) * 100
    
    prev_sma25 = float(df['SMA25'].iloc[idx-5])
    slope = (sma25 - prev_sma25) / 5
    trend_momentum = (slope / price) * 1000
    
    macd = float(curr['MACD'])
    signal = float(curr['Signal'])
    macd_power = ((macd - signal) / price) * 10000
    
    atr = float(curr['ATR'])
    entry_volatility = (atr / price) * 100
    
    return {
        'sma25_dev': sma25_dev,
        'trend_momentum': trend_momentum,
        'macd_power': macd_power,
        'entry_volatility': entry_volatility, 
        'price': price,
        'atr_value': atr 
    }

def calculate_technical_indicators(df):
    df = df.copy()
    df['SMA25'] = df['Close'].rolling(25).mean()
    df['SMA75'] = df['Close'].rolling(75).mean()
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR計算
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    return df.dropna()

# ==========================================
# 2. CBRメモリシステム (カラム名統一 & 堅牢化)
# ==========================================
class CaseBasedMemory:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        self.feature_cols = ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility']
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # カラム名を統一 (CSV内が小文字でも大文字始まりに変換)
            rename_map = {
                'date': 'Date', 'ticker': 'Ticker', 'action': 'Action', 
                'reason': 'Reason', 'timeframe': 'Timeframe', 
                'stop_loss_price': 'stop_loss_price', 
                'stop_loss_reason': 'stop_loss_reason',
                'result': 'result', 'profit_loss': 'profit_loss'
            }
            # 小文字にしてからマッピングを適用
            self.df.columns = [rename_map.get(col.lower(), col) for col in self.df.columns]
            
            if len(self.df) < 5: return

            for col in self.feature_cols:
                 if col not in self.df.columns: self.df[col] = 0.0
            
            features = self.df[self.feature_cols].fillna(0)
            self.features_normalized = self.scaler.fit_transform(features)
            
            global CBR_NEIGHBORS_COUNT
            self.knn = NearestNeighbors(n_neighbors=min(CBR_NEIGHBORS_COUNT, len(self.df)), metric='euclidean')
            self.knn.fit(self.features_normalized)
            print(f"Memory Loaded: {len(self.df)} records.")
        except Exception as e:
            print(f"Memory Load Error: {e}")

    def search_similar_cases(self, current_metrics):
        if self.knn is None or len(self.df) < 5: return "（データ不足）"

        input_df = pd.DataFrame([current_metrics], columns=self.feature_cols)
        scaled_vec = self.scaler.transform(input_df)
        distances, indices = self.knn.kneighbors(scaled_vec)
        
        text = f"【類似過去事例 ({len(indices[0])}件)】\n"
        text += "※注意: 過去に「逆張り」で失敗している場合は、今回は「順張り」を検討せよ。\n"
        text += "--------------------------------------------------\n"
        
        for idx in indices[0]:
            row = self.df.iloc[idx]
            icon = "WIN ⭕" if row['result'] == 'WIN' else "LOSS ❌" if row['result'] == 'LOSS' else "➖"
            act = row.get('Action', '?')
            text += f"● {row['Date']} {row['Ticker']} [{act}] -> {icon}\n"
            text += f"   乖離:{row['sma25_dev']:.1f}%, 勢い:{row['trend_momentum']:.1f}\n"
            
            sl_price = row.get('stop_loss_price', 0)
            try: sl_price = float(sl_price)
            except: sl_price = 0.0
                
            if sl_price > 0:
                text += f"   SL:{sl_price:.0f} (理由: {str(row.get('stop_loss_reason', ''))[:20]}...)\n"
            else:
                text += f"   理由: {str(row.get('Reason', ''))[:40]}...\n"
        
        text += "--------------------------------------------------\n"
        return text

    def save_experience(self, data_dict):
        new_row = pd.DataFrame([data_dict])
        if not os.path.exists(self.csv_path):
            new_row.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
        else:
            new_row.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        self.load_and_train()

# ==========================================
# 3. AIスパーリング (順張り特化型)
# ==========================================
def create_chart_image(df, ticker_name):
    data = df.tail(100).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(data.index, data['Close'], label='Close', color='black', linewidth=1.2)
    ax1.plot(data.index, data['SMA25'], label='SMA25', color='orange', alpha=0.8)
    ax1.set_title(f"{ticker_name} Trend Chart")
    ax1.legend(loc='upper left'); ax1.grid(True)
    ax2.plot(data.index, data['MACD'], label='MACD', color='red', linewidth=1.0)
    ax2.bar(data.index, data['MACD']-data['Signal'], color='gray', alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def ai_decision_maker(model, chart_bytes, metrics, similar_cases_text):
    # トレンド方向の判定
    trend_dir = "上昇" if metrics['trend_momentum'] > 0 else "下降"
    mech_sl_long = metrics['price'] - (metrics['atr_value'] * 2.0)

    # ★超厳格フィルタ：ボラティリティ2.0%以上は即HOLD
    vol_limit_msg = ""
    if metrics['entry_volatility'] >= 2.0:
        vol_limit_msg = "⚠️【禁止事項】現在のボラティリティ(2.0%以上)はリスク許容範囲外です。絶対にBUYしてはいけません。必ずHOLDを選択してください。"

    prompt = f"""
あなたは「超低リスク・買い専門のAIトレーダー」です。
勝率100%を目指すため、リスクの高い局面は全て見送ります。

=== 分析対象データ ===
現在値: {metrics['price']:.0f} 円
トレンド方向: {trend_dir} (勢い: {metrics['trend_momentum']:.2f})
SMA25乖離率: {metrics['sma25_dev']:.2f}%
ボラティリティ: {metrics['entry_volatility']:.2f}%

{similar_cases_text}

=== 鉄の掟 (売買基準) ===

{vol_limit_msg}

1. **【BUY (新規買い)】の絶対条件**
   - **ボラティリティが 2.0% 未満であること。** (これを超えていたら即却下)
   - SMA25が上向きで、上昇トレンド中であること。
   - 価格がSMA25付近（押し目）にあること。

2. **【HOLD (様子見)】**
   - ボラティリティが 2.0% 以上の場合。
   - 下降トレンド、またはトレンドが不明確な場合。
   - SELL（空売り）は禁止。すべてHOLDで対応せよ。

=== 出力 (JSONのみ) ===
{{
  "action": "BUY", "HOLD" のいずれか,
  "confidence": 0-100,
  "stop_loss_price": 数値 (HOLDなら0),
  "stop_loss_reason": "直近安値◯◯円割れ... (30文字以内)",
  "reason": "ボラティリティ1.5%と低く、SMA25の押し目... (100文字以内)"
}}
"""
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}])
        text_clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text_clean)
    except Exception as e:
        return {"action": "HOLD", "reason": f"Error: {e}", "confidence": 0, "stop_loss_price": 0}

# ==========================================
# 4. メイン実行
# ==========================================
def main():
    print(f"=== AI強化合宿（順張り特化 & ボラティリティフィルタ） ===")
    
    memory_system = CaseBasedMemory(LOG_FILE)
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except: return

    processed_data = {}
    print(f"Downloading data...")
    for t in TRAINING_LIST:
        df = download_data_safe(t, interval=TIMEFRAME)
        if df is None or len(df) < 100: continue
        df = calculate_technical_indicators(df)
        processed_data[t] = df

    win_count = 0
    loss_count = 0
    
    print(f"\n🥊 トレーニング開始 ({TRAINING_ROUNDS}ラウンド)\n")
    
    for i in range(1, TRAINING_ROUNDS + 1):
        if not processed_data: break
        ticker = random.choice(list(processed_data.keys()))
        df = processed_data[ticker]
        
        if len(df) < 110: continue 
        target_idx = random.randint(100, len(df) - 6)
        current_date_str = df.index[target_idx].strftime('%Y-%m-%d')
        
        metrics = calculate_metrics_enhanced(df, target_idx)
        
        # ★ボラティリティフィルタ（足切り）
        if metrics['entry_volatility'] < MIN_VOLATILITY:
            print(f"Round {i:03}: {ticker} Skip (低ボラ: {metrics['entry_volatility']:.1f}%)")
            continue

        similar_cases_text = memory_system.search_similar_cases(metrics)
        past_df = df.iloc[:target_idx+1]
        chart_bytes = create_chart_image(past_df, ticker)
        
        print(f"Round {i:03}: {ticker} ({current_date_str}) Vol:{metrics['entry_volatility']:.1f}% ... ", end="", flush=True)
        
        decision = ai_decision_maker(model_instance, chart_bytes, metrics, similar_cases_text)
        action = decision.get('action', 'HOLD')
        
        if action == "HOLD":
            print("✋ HOLD")
        else:
            curr_price = float(metrics['price'])
            future_price = float(df.iloc[target_idx + 5]['Close'])
            sl_price_raw = decision.get('stop_loss_price', 0)
            try: sl_price = float(sl_price_raw)
            except: sl_price = 0.0

            # 結果判定（SL考慮）
            period_low = df.iloc[target_idx+1 : target_idx+6]['Low'].min()
            period_high = df.iloc[target_idx+1 : target_idx+6]['High'].max()
            result = "DRAW"
            
            if action == "BUY":
                if sl_price > 0 and period_low <= sl_price:
                    result = "LOSS"; future_price = sl_price
                elif future_price > curr_price * 1.02: result = "WIN"
                elif future_price < curr_price * 0.98: result = "LOSS"
            elif action == "SELL":
                if sl_price > 0 and period_high >= sl_price:
                    result = "LOSS"; future_price = sl_price
                elif future_price < curr_price * 0.98: result = "WIN"
                elif future_price > curr_price * 1.02: result = "LOSS"
            
            icon = "⭕" if result == "WIN" else "❌" if result == "LOSS" else "➖"
            print(f"{action} -> {icon} ({result}) | SL:{sl_price:.0f}")
            
            if result == "WIN": win_count += 1
            if result == "LOSS": loss_count += 1
            
            save_data = {
                'Date': current_date_str, 'Ticker': ticker, 'Timeframe': TIMEFRAME, 
                'Action': action, 'result': result, 
                'Reason': decision.get('reason', 'None'),
                'Confidence': decision.get('confidence', 0),
                'stop_loss_price': sl_price, 
                'stop_loss_reason': decision.get('stop_loss_reason', 'None'), 
                'Price': metrics['price'],
                'sma25_dev': metrics['sma25_dev'], 
                'trend_momentum': metrics['trend_momentum'],
                'macd_power': metrics['macd_power'],
                'entry_volatility': metrics['entry_volatility'],
                'profit_loss': future_price - curr_price
            }
            memory_system.save_experience(save_data)

        time.sleep(2)

    print(f"\n=== 合宿終了 ===")
    print(f"戦績: {win_count}勝 {loss_count}敗")

if __name__ == "__main__":
    main()