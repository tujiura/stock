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

# ---------------------------------------------------------
# ★環境設定
# ---------------------------------------------------------
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

# ★ファイル名をV7_VOLに変更
LOG_FILE = "ai_trade_memory_aggressive_v7.csv" 
MODEL_NAME = 'models/gemini-2.0-flash'

TRAINING_ROUNDS = 20000
TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15
TRADE_BUDGET = 100000  # 1トレードあたりの投資金額 (10万円)

# ★ V7 (Sniper Strategy) パラメータ
ATR_STOP_MULTIPLIER = 1.8      
TRAILING_TRIGGER = 0.10        
TRAILING_MULTIPLIER = 2.0      

MA_DEV_DANGER_LOW = 10.0     
MA_DEV_DANGER_HIGH = 15.0    

# トレーニングリスト
TRAINING_LIST = [
    "6254.T", "8035.T", "2768.T", "6305.T", "6146.T",
    "6920.T", "6857.T", "7735.T", "6723.T", "6963.T", "3436.T", "6526.T", "6315.T",
    "6758.T", "6861.T", "6981.T", "6594.T", "6954.T", "6506.T", "6702.T", "6752.T", "7751.T", "6501.T", "6503.T",
    "7203.T", "7267.T", "7269.T", "7270.T", "7201.T", "7259.T", "6902.T",
    "7011.T", "7013.T", "7012.T", "6301.T", "6367.T", "7003.T",
    "8058.T", "8001.T", "8031.T", "8002.T", "8053.T", "7459.T",
    "8306.T", "8316.T", "8411.T", "8766.T", "8725.T", "8591.T", "8604.T", "8698.T",
    "9984.T", "9432.T", "9433.T", "9434.T", "6098.T", "2413.T", "4661.T", "4385.T", "4751.T", "9613.T",
    "9983.T", "3382.T", "8267.T", "2802.T", "2914.T", "4911.T", "4543.T", "4503.T", "4568.T",
    "7974.T", "9697.T", "9766.T", "5253.T", 
    "9101.T", "9104.T", "9107.T", "5401.T", "5411.T", "1605.T", "5713.T", "5020.T", "4063.T", "4901.T"
]

plt.rcParams['font.family'] = 'sans-serif'
genai.configure(api_key=GOOGLE_API_KEY, transport="rest")

# ==========================================
# 1. データ取得 & テクニカル計算 (V7強化版)
# ==========================================
def download_data_safe(ticker, period="5y", interval="1d", retries=3): 
    wait = 2
    for attempt in range(retries):
        try:
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            if len(df) < 100: return None
            return df
        except:
            time.sleep(wait); wait *= 2
    return None

def calculate_technical_indicators(df):
    df = df.copy()
    close = df['Close']; high = df['High']; low = df['Low']
    
    # 基本指標
    df['SMA25'] = close.rolling(25).mean()
    
    # 1. DMI/ADX
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    tr_smooth = tr.rolling(14).mean()
    df['PlusDI'] = 100 * (plus_dm.rolling(14).mean() / tr_smooth)
    df['MinusDI'] = 100 * (minus_dm.rolling(14).mean() / tr_smooth)
    dx = (abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])) * 100
    df['ADX'] = dx.rolling(14).mean()
    df['ATR'] = tr.rolling(14).mean()

    # 2. Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Width'] = ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20 * 100
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    
    # 3. RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    df['RSI9'] = 100 - (100 / (1 + gain/loss))

    # 4. MACD (12, 26, 9)
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # 5. 一目均衡表 (雲のみ)
    high9 = high.rolling(9).max(); low9 = low.rolling(9).min()
    tenkan = (high9 + low9) / 2
    high26 = high.rolling(26).max(); low26 = low.rolling(26).min()
    kijun = (high26 + low26) / 2
    
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    high52 = high.rolling(52).max(); low52 = low.rolling(52).min()
    senkou_b = ((high52 + low52) / 2).shift(26)
    
    df['Cloud_Top'] = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)

    return df.dropna()

def calculate_metrics_for_training(df, idx):
    curr = df.iloc[idx]
    price = float(curr['Close'])
    
    past_60 = df.iloc[idx-60:idx]
    recent_high = past_60['High'].max()
    dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
    
    bb_width = float(curr['BB_Width'])
    prev_width = float(df['BB_Width'].iloc[idx-5]) if df['BB_Width'].iloc[idx-5] > 0 else 0.1
    expansion_rate = bb_width / prev_width
    
    # ★出来高推移の生成 (5日分)
    # Vol_Ratio = Volume / Vol_MA20
    vol_history = []
    for i in range(4, -1, -1):
        if idx-i >= 0:
            row = df.iloc[idx-i]
            vr = float(row['Volume']) / float(row['Vol_MA20']) if float(row['Vol_MA20']) > 0 else 0
            vol_history.append(f"{vr:.1f}")
    vol_history_str = "->".join(vol_history)
    vol_ratio = float(curr['Volume']) / float(curr['Vol_MA20']) if float(curr['Vol_MA20']) > 0 else 0
    
    macd = float(curr['MACD'])
    macd_hist = float(curr['MACD_Hist'])
    prev_hist = float(df['MACD_Hist'].iloc[idx-1])
    
    cloud_top = float(curr['Cloud_Top']) if not pd.isna(curr['Cloud_Top']) else 0
    price_vs_cloud = "Above" if price > cloud_top else "Below"
    
    open_p = float(curr['Open']); close_p = float(curr['Close']); high_p = float(curr['High']); low_p = float(curr['Low'])
    body_top = max(open_p, close_p)
    upper_shadow = high_p - body_top
    total_range = high_p - low_p
    shadow_ratio = upper_shadow / total_range if total_range > 0 else 0
    candle_shape = "Good" if shadow_ratio < 0.3 else "Bad (Long Upper Shadow)"

    return {
        'price': price,
        'dist_to_res': dist_to_res,
        'ma_deviation': ((price / float(curr['SMA25'])) - 1) * 100,
        'adx': float(curr['ADX']),
        'prev_adx': float(df['ADX'].iloc[idx-1]),
        'plus_di': float(curr['PlusDI']),
        'minus_di': float(curr['MinusDI']),
        'vol_ratio': vol_ratio,
        'vol_history': vol_history_str, # ★追加
        'expansion_rate': expansion_rate,
        'atr_value': float(curr['ATR']),
        'macd_val': macd,
        'macd_hist': macd_hist,
        'macd_trend': "Expanding" if abs(macd_hist) > abs(prev_hist) else "Shrinking",
        'price_vs_cloud': price_vs_cloud,
        'candle_shape': candle_shape,
        'rsi_9': float(curr['RSI9']),
        'rs_rating': 0
    }

def check_iron_rules(metrics):
    if metrics['adx'] < 20: return "ADX<20"
    if metrics['vol_ratio'] < 0.8: return "Vol<0.8"
    
    ma_dev = metrics['ma_deviation']
    if MA_DEV_DANGER_LOW <= ma_dev <= MA_DEV_DANGER_HIGH: 
        return f"DangerZone({ma_dev:.1f}%)"
    if metrics['adx'] > 55: return "ADX Overheat"
    
    if metrics['price_vs_cloud'] == "Below": return "Below Ichimoku Cloud"
    
    return None

# ==========================================
# 2. CBRメモリシステム
# ==========================================
class CaseBasedMemory:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        self.feature_cols = ['adx', 'prev_adx', 'ma_deviation', 'vol_ratio', 'expansion_rate', 'dist_to_res']
        
        self.csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "target_price", 
            "Actual_High", "Target_Diff", "Target_Reach",
            "Price", "adx", "prev_adx", "ma_deviation", "rs_rating", 
            "vol_ratio", "expansion_rate", "dist_to_res", 
            "days_to_earnings", "margin_ratio", "profit_rate"
        ]
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path)
            for col in self.csv_columns:
                if col not in self.df.columns: self.df[col] = 0.0
        except Exception: return

        try:
            self.df.columns = [c.strip() for c in self.df.columns]
            valid_df = self.df[self.df['result'].isin(['WIN', 'LOSS'])].copy()
            if len(valid_df) > 5:
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
        # 不要なカラム（内部計算用）を除外して保存
        save_cols = [c for c in self.csv_columns if c in new_df.columns]
        new_df = new_df[save_cols]
        
        try:
            if not os.path.exists(self.csv_path):
                new_df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
            else:
                new_df.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            self.load_and_train() 
        except Exception as e:
            print(f"❌ 保存エラー: {e}")

# ==========================================
# 3. AIスパーリング
# ==========================================
def create_chart_image(df, name):
    data = df.tail(80).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    ax1.plot(data.index, data['Close'], color='black', label='Close')
    ax1.plot(data.index, sma20 + 2*std20, color='green', alpha=0.5, linestyle='--', label='+2σ')
    ax1.plot(data.index, sma20 - 2*std20, color='green', alpha=0.5, linestyle='--', label='-2σ')
    
    if 'Cloud_Top' in data.columns:
        ax1.plot(data.index, data['Cloud_Top'], color='blue', alpha=0.2, label='Cloud Top')
        ax1.fill_between(data.index, data['Cloud_Top'], data['Close'].min(), color='blue', alpha=0.05)

    ax1.set_title(f"{name} V7 Sniper Chart")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
    
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def ai_decision_maker(model, chart_bytes, metrics, cbr_text, ticker):
    if metrics['adx'] < 20: return {"action": "HOLD", "reason": "ADX<20"}
    
    # ★V7 改良プロンプト (Volume History追加)
    prompt = f"""
# ROLE
あなたは世界最高峰の「クオンツ・テクニカルアナリスト」であり、高精度のトレンドフォロー戦略を実行するAIエージェントです。
あなたの使命は、提供されたデータに基づき、感情を排して数学的かつ論理的にトレード判断を下すことです。
特に「ダマシ（False Breakout）」を回避し、優位性（Edge）のあるエントリーポイントのみを厳選します。

# CONTEXT & OBJECTIVE
ユーザーは、以下の銘柄について「買い（BUY）」か「見送り（HOLD）」かの二択の判断を求めています。
曖昧な状況や、リスクリワード比が悪い局面では、資産を守るために迷わず「HOLD」を選択する必要があります。

# INPUT DATA
## 対象銘柄
* Ticker: {ticker}
* Current Price: {metrics['price']:.0f} JPY

## 1. Technical Indicators (Quantitative)
* **Trend Strength ($ADX$):** {metrics['adx']:.1f} (Threshold: $\ge 25$)
* **Directional Movement:** $+DI$ = {metrics['plus_di']:.1f} vs $-DI$ = {metrics['minus_di']:.1f}
* **Volatility Expansion:** {metrics['expansion_rate']:.2f}x (Squeeze $\\to$ Expansion is ideal)
* **Volume Ratio:** {metrics['vol_ratio']:.2f}x (Trend: {metrics['vol_history']})
* **MACD:** Histogram = {metrics['macd_hist']:.2f} ({metrics['macd_trend']})
* **Ichimoku Cloud:** Price is **{metrics['price_vs_cloud']}** the Cloud.
* **Resistance Distance:** {metrics['dist_to_res']:.1f}%

## 2. Qualitative Factors (Price Action & Environment)
* **Candle Shape:** {metrics['candle_shape']}

## 3. External Factors
{cbr_text}

# STRICT EVALUATION RULES (AND/OR LOGIC)
判断を下す際は、以下の論理ゲートを厳守してください。

1.  **Trend Filter (Must Pass):**
    * 価格が「一目均衡表の雲」の上にあること（$Price > Cloud$）。これが満たされない場合、**即座にHOLD**とする。
    * トレンドの強さ（$ADX$）が基準を満たしている、または勢いが増していること。

2.  **Momentum Trigger:**
    * MACDヒストグラムが拡大傾向、かつプラス圏にあることが望ましい。
    * 出来高（Volume）が増加傾向にあり、値動きを裏付けていること。

3.  **Risk Check:**
    * 長い上ヒゲ（Selling Pressure）が出現していないか？
    * 決算発表が直近（3日以内など）に迫っていないか？
    * 抵抗線（Resistance）が極端に近くないか？

# OUTPUT FORMAT
回答は**JSON形式のみ**で出力してください。Markdownのコードブロックや説明文は一切不要です。

{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100 (Integer),
  "stop_loss": {metrics['price'] * 0.95:.0f},  // 例: 現在価格から計算、またはロジックで算出
  "target_price": {metrics['price'] * 1.10:.0f}, // 例: リスクリワード1:2などを想定
  "reason": "判断の決定的な理由を簡潔に記述 (Max 60 chars)"
}}

# FINAL INSTRUCTION
情け容赦ないプロの視点で分析し、JSONデータのみを返してください。
"""
    safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}], safety_settings=safety)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: text = match.group(0)
        return json.loads(text)
    except Exception:
        return {"action": "HOLD", "reason": "AI Error", "confidence": 0}

# ==========================================
# 4. メイン実行 (トレーニングモード)
# ==========================================
def main():
    start_time = time.time()
    print(f"=== AI強化合宿 [AGGRESSIVE V7] (Sniper Precision + Vol History) ===")
    
    memory_system = CaseBasedMemory(LOG_FILE) 
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e: print(f"Model Init Error: {e}"); return

    processed_data = {}
    print(f"データをダウンロード中...")
    for i, t in enumerate(TRAINING_LIST):
        if i % 10 == 0: print(f"  - {i}/{len(TRAINING_LIST)}")
        df = download_data_safe(t, interval=TIMEFRAME)
        if df is None: continue
        df = calculate_technical_indicators(df)
        processed_data[t] = df

    if not processed_data: print("データ不足"); return
    print(f"データ取得完了 ({int(time.time() - start_time)}秒)")

    win_count = 0; loss_count = 0
    
    print(f"\n🥊 トレーニング開始 ({TRAINING_ROUNDS}ラウンド)\n")
    
    for i in range(1, TRAINING_ROUNDS + 1):
        ticker = random.choice(list(processed_data.keys()))
        df = processed_data[ticker]
        if len(df) < 110: continue 
        
        target_idx = random.randint(100, len(df) - 65) 
        metrics = calculate_metrics_for_training(df, target_idx)
        
        iron_rule = check_iron_rules(metrics)
        if iron_rule: continue

        cbr_text = memory_system.search_similar_cases(metrics)
        past_df = df.iloc[:target_idx+1]
        chart_bytes = create_chart_image(past_df, ticker)
        
        decision = ai_decision_maker(model_instance, chart_bytes, metrics, cbr_text, ticker)
        action = decision.get('action', 'HOLD')
        conf = decision.get('confidence', 0)

        if action == "HOLD": continue

        print(f"Round {i:03}: {ticker} -> BUY 🔴 (自信:{conf}%)")

        entry_price = float(metrics['price'])
        atr = metrics['atr_value']
        
        ai_stop = decision.get('stop_loss', 0)
        ai_target = decision.get('target_price', 0)
        try: ai_stop = int(ai_stop); ai_target = int(ai_target)
        except: ai_stop = 0; ai_target = 0
        
        current_stop_loss = ai_stop if ai_stop > 0 else entry_price - (atr * ATR_STOP_MULTIPLIER)
        
        shares = int(TRADE_BUDGET // entry_price)
        if shares < 1: shares = 1
        
        # --- シミュレーション (ホームラン狙い) ---
        future_prices = df.iloc[target_idx+1 : target_idx+61]
        result = "DRAW"
        final_exit_price = entry_price
        max_price = entry_price
        is_loss = False
        
        actual_high = future_prices['High'].max()
        
        for _, row in future_prices.iterrows():
            high = row['High']; low = row['Low']; close = row['Close']
            
            if low <= current_stop_loss:
                is_loss = True
                final_exit_price = current_stop_loss
                break
            
            if high > max_price:
                max_price = high
            
            profit_pct_high = (max_price - entry_price) / entry_price
            
            if profit_pct_high > TRAILING_TRIGGER:
                trail_dist = atr * TRAILING_MULTIPLIER
                new_stop = max_price - trail_dist
                if profit_pct_high > 0.15:
                     new_stop = max(new_stop, entry_price * 1.005)
                if new_stop > current_stop_loss:
                    current_stop_loss = new_stop

        if not is_loss:
            final_exit_price = future_prices['Close'].iloc[-1]

        profit_loss = (final_exit_price - entry_price) * shares
        profit_rate = ((final_exit_price - entry_price) / entry_price) * 100
        
        if profit_loss > 0: result = "WIN"; win_count += 1
        elif profit_loss < 0: result = "LOSS"; loss_count += 1

        print(f"   結果: {result} (PL: {profit_loss:+.0f}円 / {profit_rate:+.2f}%) Tgt:{ai_target} ActHigh:{actual_high:.0f}")

        target_diff = actual_high - ai_target if ai_target > 0 else 0
        target_reach = 0
        if ai_target > 0:
            upside = ai_target - entry_price
            act_up = actual_high - entry_price
            if upside > 0: target_reach = (act_up / upside) * 100

        save_data = {
            'Date': df.index[target_idx].strftime('%Y-%m-%d'), 
            'Ticker': ticker, 'Timeframe': TIMEFRAME, 
            'Action': action, 'result': result, 
            'Reason': decision.get('reason', 'None'),
            'Confidence': conf, 
            'stop_loss_price': current_stop_loss, 
            'target_price': ai_target, 
            'Actual_High': actual_high, 'Target_Diff': target_diff, 'Target_Reach': target_reach,
            'Price': metrics['price'], 
            'adx': metrics['adx'], 
            'prev_adx': metrics['prev_adx'],
            'ma_deviation': metrics['ma_deviation'], 
            'rs_rating': 0, 
            'vol_ratio': metrics['vol_ratio'], 
            'expansion_rate': metrics['expansion_rate'],
            'dist_to_res': metrics['dist_to_res'], 
            'days_to_earnings': 999, 
            'margin_ratio': 1.0, 
            'profit_rate': profit_rate 
        }
        memory_system.save_experience(save_data)
        time.sleep(1)

    elapsed_time = time.time() - start_time
    print(f"\n=== 合宿終了 ({str(datetime.timedelta(seconds=int(elapsed_time)))}) ===")
    print(f"戦績 (BUY): {win_count}勝 {loss_count}敗")

if __name__ == "__main__":
    main()