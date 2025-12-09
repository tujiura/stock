import yfinance as yf
import pandas as pd
import google.generativeai as genai
import json
import time
import datetime
import urllib.parse
import feedparser
import os
import io
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ==========================================
# ★設定エリア (Ver.2 ハイボラ対応版)
# ==========================================
GOOGLE_API_KEY = "ここにAPIキーを貼り付け".strip() 
LOG_FILE = "ai_trade_memory_high_vol.csv"  # ★ファイル名を変更（混同防止）
MODEL_NAME = 'models/gemini-2.5-pro'

TIMEFRAME = "1d" 
CBR_NEIGHBORS_COUNT = 11
TRAINING_ROUNDS = 100 # 学習回数

# 監視リスト（同じものでOK）
WATCH_LIST = [
    "8035.T", "6146.T", "6920.T", "6857.T", "6723.T", "7735.T", "6526.T",
    "6758.T", "6861.T", "6501.T", "6503.T", "6981.T", "6954.T", "7741.T", 
    "6902.T", "6367.T", "6594.T", "7751.T", "7203.T", "7267.T", "7270.T", 
    "8306.T", "8316.T", "8411.T", "8766.T", "8725.T", "8591.T", "8604.T",
    "8058.T", "8031.T", "8001.T", "8002.T", "8015.T", "2768.T",
    "7011.T", "7012.T", "7013.T", "6301.T", "5401.T", "9101.T", "9104.T", 
    "9432.T", "9433.T", "9984.T", "9434.T", "4661.T", "6098.T", "7974.T", 
    "9684.T", "9697.T", "7832.T", "9983.T", "3382.T", "8267.T", "9843.T", 
    "3092.T", "4385.T", "7532.T", "4568.T", "4519.T", "4503.T", "4502.T", 
    "4063.T", "4901.T", "4452.T", "2914.T", "8801.T", "8802.T", "1925.T", 
    "1801.T", "9501.T", "9503.T", "1605.T", "5020.T", "9020.T", "9202.T", "2802.T"
]

plt.rcParams['font.family'] = 'sans-serif' 

# ==========================================
# 1. 高度な指標計算 (RSI, Volume, Bollinger)
# ==========================================
def download_data_safe(ticker, period="1y", interval="1d", retries=3):
    wait = 2
    for _ in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: raise ValueError("Empty")
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            return df
        except: time.sleep(wait); wait *= 2
    return None

def calculate_metrics_pro(df, idx):
    if len(df) < 30: return None
    curr = df.iloc[idx]
    price = float(curr['Close'])
    
    # 1. 基本指標
    sma25 = df['Close'].rolling(25).mean().iloc[idx]
    sma25_dev = ((price / sma25) - 1) * 100
    prev_sma25 = df['Close'].rolling(25).mean().iloc[idx-5]
    trend_momentum = ((sma25 - prev_sma25) / 5 / price) * 1000
    
    # 2. MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_power = ((macd_line.iloc[idx] - signal_line.iloc[idx]) / price) * 10000
    
    # 3. ボラティリティ (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[idx]
    volatility = (atr / price) * 100
    
    # 4. ★新指標: RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[idx]
    
    # 5. ★新指標: 出来高倍率 (Volume Ratio)
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[idx]
    volume_ratio = (curr['Volume'] / vol_ma5) if vol_ma5 > 0 else 1.0
    
    # 6. ★新指標: ボリンジャーバンド位置
    std20 = df['Close'].rolling(20).std().iloc[idx]
    sma20 = df['Close'].rolling(20).mean().iloc[idx]
    upper_2sigma = sma20 + (2 * std20)
    # 現在価格がバンドのどこにいるか (0=中心, 1.0=2σ, -1.0=-2σ)
    bb_position = (price - sma20) / (2 * std20)

    return {
        'price': price,
        'sma25_dev': sma25_dev,
        'trend_momentum': trend_momentum,
        'macd_power': macd_power,
        'entry_volatility': volatility,
        'rsi': rsi,
        'volume_ratio': volume_ratio,
        'bb_position': bb_position,
        'atr_value': atr
    }

# ==========================================
# 2. メモリシステム (新指標に対応)
# ==========================================
class CaseBasedMemoryPro:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        # ★特徴量を増やして精度アップ
        self.feature_cols = ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility', 'rsi', 'volume_ratio', 'bb_position']
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path)
            if len(self.df) < 5: return
            
            # 数値列の確保
            for col in self.feature_cols:
                if col not in self.df.columns: self.df[col] = 0.0
            
            features = self.df[self.feature_cols].fillna(0)
            self.features_normalized = self.scaler.fit_transform(features)
            
            self.knn = NearestNeighbors(n_neighbors=min(CBR_NEIGHBORS_COUNT, len(self.df)), metric='euclidean')
            self.knn.fit(self.features_normalized)
            print(f"Memory Loaded: {len(self.df)} records.")
        except: pass

    def search_similar_cases(self, metrics):
        if self.knn is None or len(self.df) < 5: return "（データ不足）"
        
        # 辞書から必要な特徴量だけ抽出してDF化
        input_data = {k: [metrics[k]] for k in self.feature_cols}
        input_df = pd.DataFrame(input_data)
        
        scaled_vec = self.scaler.transform(input_df)
        distances, indices = self.knn.kneighbors(scaled_vec)
        
        text = f"【類似局面 ({len(indices[0])}件)】\n"
        for idx in indices[0]:
            row = self.df.iloc[idx]
            res = str(row.get('result', ''))
            icon = "WIN ⭕" if res=='WIN' else "LOSS ❌" if res=='LOSS' else "➖"
            text += f"● {row['Ticker']} ({row.get('Action','?')}) -> {icon}\n"
            text += f"   RSI:{row.get('rsi',0):.0f}, Vol倍率:{row.get('volume_ratio',0):.1f}倍, BB位置:{row.get('bb_position',0):.1f}σ\n"
        return text

    def save_experience(self, data_dict):
        # 辞書をDataFrame化
        new_df = pd.DataFrame([data_dict])
        
        if not os.path.exists(self.csv_path):
            new_df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
        else:
            # 既存ファイルのヘッダーに合わせて追記
            existing_cols = pd.read_csv(self.csv_path, nrows=0).columns.tolist()
            # 足りない列を埋める
            for col in existing_cols:
                if col not in new_df.columns: new_df[col] = None
            # 列順を揃える
            new_df = new_df[existing_cols]
            new_df.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        self.load_and_train()

# ==========================================
# 3. AI分析 (ハイボラティリティ対応プロンプト)
# ==========================================
def create_chart_image(df, ticker_name):
    # (既存と同じなので省略可だが、念のため記載)
    data = df.tail(100).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(data.index, data['Close'], label='Close', color='black')
    sma25 = data['Close'].rolling(25).mean()
    
    # ボリンジャーバンド表示
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    ax1.plot(data.index, sma20 + 2*std20, color='green', alpha=0.3, label='+2σ')
    ax1.plot(data.index, sma20 - 2*std20, color='green', alpha=0.3, label='-2σ')
    
    ax1.set_title(f"{ticker_name} Pro Analysis")
    ax1.legend(); ax1.grid(True)
    
    # RSI表示
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    ax2.plot(data.index, rsi, label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='blue', linestyle='--')
    ax2.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def ai_decision_maker_pro(model, chart_bytes, metrics, similar_cases_text):
    # RSIと出来高による相場判定
    market_condition = "通常"
    if metrics['entry_volatility'] > 2.5: market_condition = "荒れ相場(高ボラ)"
    
    prompt = f"""
あなたは「あらゆる相場環境に適応するプロのAIトレーダー」です。
今回は新指標（RSI, 出来高, ボリンジャーバンド）を駆使し、**「高ボラティリティ局面」でも利益を出せる判断** をしてください。

=== 分析データ ===
市場環境: {market_condition}
1. 基本指標
   - SMA25乖離: {metrics['sma25_dev']:.2f}%
   - トレンド勢い: {metrics['trend_momentum']:.2f}
2. 高度指標 (New!)
   - **RSI(14)**: {metrics['rsi']:.1f} (30以下は売られすぎ、70以上は買われすぎ)
   - **出来高倍率**: {metrics['volume_ratio']:.2f}倍 (1.5倍以上は強い資金流入)
   - **BB位置**: {metrics['bb_position']:.2f}σ (2.0を超えるとバンドウォークまたは行き過ぎ)
   - ボラティリティ: {metrics['entry_volatility']:.2f}%

{similar_cases_text}

=== 戦略ガイドライン ===

A. **【順張り (BUY)】のチャンス**
   - **バンドウォーク**: BB位置が +1.0σ 〜 +2.0σ で、出来高が増加(1.2倍以上)している時。
   - **押し目**: 上昇トレンド中で、RSIが 40〜50 まで下がった時。
   - **高ボラ対策**: ボラティリティが高くても、**「出来高急増 + 陽線」** ならトレンド発生とみなしてBUY可。

B. **【逆張り (逆張りBUY/SELL)】のチャンス**
   - **セリングクライマックス**: ボラティリティが高く急落し、**RSIが 25以下** になったら「自律反発狙いのBUY」可。
   - **過熱感**: **RSIが 80以上** かつ BB位置が +2.5σ を超えたら「調整狙いのSELL」可（ただし慎重に）。

C. **【HOLD (様子見)】**
   - 出来高が伴わない急騰・急落（ダマシの可能性大）。
   - RSIが 30〜70 の中間で、方向感がない時。

=== 出力 (JSONのみ) ===
{{
  "action": "BUY", "SELL", "HOLD" のいずれか,
  "confidence": 0-100,
  "stop_loss_price": 数値 (必須),
  "stop_loss_reason": "RSIが30を割り込み... (30文字以内)",
  "reason": "高ボラティリティだが出来高が2倍に急増しており、本物のトレンド初動と判断... (100文字以内)"
}}
"""
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}])
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except: return {"action": "HOLD", "reason": "Error", "confidence": 0, "stop_loss_price": 0}

# ==========================================
# 4. メイン実行 (トレーニングモード)
# ==========================================
def main():
    print(f"=== AI強化合宿 (Ver.2 ハイボラ対応版) ===")
    print(f"保存先: {LOG_FILE}")
    
    # ⚠️APIキーを設定してください
    if GOOGLE_API_KEY == "ここにAPIキーを貼り付け":
        print("Error: API Key not set."); exit()
    
    genai.configure(api_key=GOOGLE_API_KEY)
    try: model = genai.GenerativeModel(MODEL_NAME)
    except: return

    memory = CaseBasedMemoryPro(LOG_FILE)
    
    # データ取得
    data_cache = {}
    print("Downloading data...")
    for t in WATCH_LIST:
        df = download_data_safe(t)
        if df is not None and len(df) > 50:
            data_cache[t] = df
    
    win_count = 0
    loss_count = 0
    
    print(f"\n🥊 Start Training ({TRAINING_ROUNDS} rounds)\n")
    
    for i in range(1, TRAINING_ROUNDS + 1):
        if not data_cache: break
        ticker = random.choice(list(data_cache.keys()))
        df = data_cache[ticker]
        
        # ランダムな過去の日付を選ぶ
        target_idx = random.randint(40, len(df) - 6)
        curr_date = df.index[target_idx].strftime('%Y-%m-%d')
        
        metrics = calculate_metrics_pro(df, target_idx)
        if metrics is None: continue
        
        # 類似検索
        cbr_text = memory.search_similar_cases(metrics)
        # チャート作成
        chart_bytes = create_chart_image(df.iloc[:target_idx+1], ticker)
        
        # AI判断
        print(f"Round {i:03}: {ticker} ({curr_date}) Vol:{metrics['entry_volatility']:.1f}% RSI:{metrics['rsi']:.0f} ... ", end="", flush=True)
        
        decision = ai_decision_maker_pro(model, chart_bytes, metrics, cbr_text)
        action = decision.get('action', 'HOLD')
        
        if action == "HOLD":
            print("✋ HOLD")
        else:
            # 勝敗判定
            curr_price = metrics['price']
            future_price = float(df.iloc[target_idx + 5]['Close'])
            sl_price = float(decision.get('stop_loss_price', 0))
            
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
            print(f"{action} -> {icon} ({result})")
            
            if result == "WIN": win_count += 1
            if result == "LOSS": loss_count += 1
            
            # データ保存 (辞書キーをカラム名と一致させる)
            save_data = {
                'Date': curr_date, 'Ticker': ticker, 'Timeframe': TIMEFRAME, 
                'Action': action, 'result': result, 
                'Reason': decision.get('reason', '-'), 
                'Confidence': decision.get('confidence', 0),
                'stop_loss_price': sl_price,
                'stop_loss_reason': decision.get('stop_loss_reason', '-'),
                'Price': curr_price,
                'profit_loss': future_price - curr_price,
                # --- ここから新指標 ---
                'sma25_dev': metrics['sma25_dev'], 
                'trend_momentum': metrics['trend_momentum'],
                'macd_power': metrics['macd_power'],
                'entry_volatility': metrics['entry_volatility'],
                'rsi': metrics['rsi'],
                'volume_ratio': metrics['volume_ratio'],
                'bb_position': metrics['bb_position']
            }
            memory.save_experience(save_data)
            
        time.sleep(1.5)

    print(f"\n=== 合宿終了 ===")
    print(f"戦績: {win_count}勝 {loss_count}敗")

if __name__ == "__main__":
    main()