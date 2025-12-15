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

# ★ファイル名をV5_OPTに変更 (最適化版学習データ)
LOG_FILE = "ai_trade_memory_aggressive_v6.csv" 
MODEL_NAME = 'models/gemini-2.0-flash'

TRAINING_ROUNDS = 1500  # トレーニング回数
TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15
TRADE_BUDGET = 1000000 

# ★ V5改 (Optimization) ロジックパラメータ
ATR_STOP_MULTIPLIER = 2.5      # 初期損切り幅
PARTIAL_PROFIT_TARGET = 0.035  # 分割利確ライン (+5%)
PARTIAL_EXIT_RATIO = 0.5       # 分割利確割合 (50%)
TRAILING_WIDE_MULTIPLIER = 2.5 # 分割後の追従幅 (広げる)

# 鉄の掟用
MA_DEV_DANGER_LOW = 10.0     
MA_DEV_DANGER_HIGH = 15.0    

# トレーニング対象リスト
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
# 1. データ取得 & テクニカル計算
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
    df['SMA25'] = df['Close'].rolling(25).mean()
    
    high = df['High']; low = df['Low']; close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    tr_smooth = tr.rolling(14).mean()
    plus_dm_smooth = plus_dm.rolling(14).mean()
    minus_dm_smooth = minus_dm.rolling(14).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()
    df['PlusDI'] = plus_di
    df['MinusDI'] = minus_di

    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Width'] = ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20 * 100
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['ATR'] = tr.rolling(14).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    rs = gain / loss
    df['RSI9'] = 100 - (100 / (1 + rs))

    return df.dropna()

def calculate_metrics_for_training(df, idx):
    curr = df.iloc[idx]
    price = float(curr['Close'])
    
    past_60 = df.iloc[idx-60:idx]
    recent_high = past_60['High'].max()
    dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
    
    adx = float(curr['ADX'])
    prev_adx = float(df['ADX'].iloc[idx-1])
    
    sma25 = float(curr['SMA25'])
    ma_deviation = ((price / sma25) - 1) * 100
    
    bb_width = float(curr['BB_Width'])
    prev_width = float(df['BB_Width'].iloc[idx-5]) if df['BB_Width'].iloc[idx-5] > 0 else 0.1
    expansion_rate = bb_width / prev_width
    
    vol_ma20 = float(curr['Vol_MA20'])
    vol_ratio = float(curr['Volume']) / vol_ma20 if vol_ma20 > 0 else 0
    rsi_9 = float(curr['RSI9'])
    
    return {
        'price': price,
        'resistance_price': recent_high,
        'dist_to_res': dist_to_res,
        'ma_deviation': ma_deviation,
        'adx': adx,
        'prev_adx': prev_adx,
        'plus_di': float(curr['PlusDI']),
        'minus_di': float(curr['MinusDI']),
        'rs_rating': 0.0, 
        'vol_ratio': vol_ratio,
        'expansion_rate': expansion_rate,
        'atr_value': float(curr['ATR']),
        'rsi_9': rsi_9
    }

def check_iron_rules(metrics):
    if metrics['adx'] < 20: return "ADX<20"
    if metrics['vol_ratio'] < 0.8: return "Vol<0.8"
    
    ma_dev = metrics['ma_deviation']
    if MA_DEV_DANGER_LOW <= ma_dev <= MA_DEV_DANGER_HIGH: 
        return f"DangerZone({ma_dev:.1f}%)"
    
    if metrics['adx'] > 55: return "ADX Overheat"
    
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
        
        # ★保存カラム (Actual_High, Target_Diff, Profit_Rateなど)
        self.csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "target_price", 
            "Actual_High", "Target_Diff", "Target_Reach",
            "Price", "adx", "prev_adx", "ma_deviation", "rs_rating", 
            "vol_ratio", "expansion_rate", "dist_to_res", "days_to_earnings", 
            "margin_ratio", "profit_rate"
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
        new_df = new_df[self.csv_columns]
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
    ax1.set_title(f"{name} Training Chart")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def ai_decision_maker(model, chart_bytes, metrics, cbr_text, ticker):
    if metrics['adx'] < 20: return {"action": "HOLD", "reason": "ADX<20"}
    
    prompt = f"""
### ROLE
あなたは「高ボラティリティ・トレンドフォロー特化型AI」です。
小さな利益は無視し、発生し始めた「大きなトレンド（急騰）」や「ブレイクアウト」のみを捕捉します。

### INPUT DATA
銘柄: {ticker} (現在価格: {metrics['price']:.0f}円)

[テクニカル指標]
1. Trend Strength (ADX): {metrics['adx']:.1f} (閾値: 25以上, 前日: {metrics['prev_adx']:.1f})
2. Direction (+DI/-DI): +DI({metrics['plus_di']:.1f}) vs -DI({metrics['minus_di']:.1f})
3. Volatility (BB Exp): {metrics['expansion_rate']:.2f}倍
4. Volume Flow: {metrics['vol_ratio']:.2f}倍
5. MA Deviation: {metrics['ma_deviation']:.2f}% (過熱感チェック)

[重要コンテキスト]
- **抵抗線位置**: {metrics['resistance_price']:.0f}円 (現在価格との差: {metrics['dist_to_res']:.1f}%)
{cbr_text}

### EVALUATION LOGIC
1. **ブレイクアウト判定**:
   - 抵抗線(resistance_price)を価格が上回っている、または抵抗線での攻防を制しつつあるか？
   - 抵抗線を明確に超えていれば "BUY" の確度アップ。
   
2. **過熱感チェック**:
   - MA乖離率(ma_deviation)が +30% を超えている場合は "HOLD" (高値掴み警戒)。

### OUTPUT REQUIREMENT (JSON ONLY)
{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "stop_loss": "推奨する損切り価格（整数）",
  "target_price": "推奨する利確目標価格（整数）",
  "reason": "判断理由(50文字以内)"
}}
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
# 4. メイン実行 (トレーニングモード: V5改シミュレーション)
# ==========================================
def main():
    start_time = time.time()
    print(f"=== AI強化合宿 [AGGRESSIVE V5 OPT] (Split Exit Training) ===")
    print(f"Strategy: Half Profit @ +{PARTIAL_PROFIT_TARGET*100}% / Trail ATRx{TRAILING_WIDE_MULTIPLIER}")
    
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
        
        # 鉄の掟チェック
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
        
        # AI損切り・ターゲット
        ai_stop = decision.get('stop_loss', 0)
        ai_target = decision.get('target_price', 0)
        try: ai_stop = int(ai_stop); ai_target = int(ai_target)
        except: ai_stop = 0; ai_target = 0
        
        current_stop_loss = ai_stop if ai_stop > 0 else entry_price - (atr * ATR_STOP_MULTIPLIER)
        
        # シミュレーション用設定
        initial_shares = int(TRADE_BUDGET // entry_price)
        if initial_shares < 1: initial_shares = 1
        
        shares = initial_shares
        realized_profit = 0.0
        partial_exit_done = False
        
        future_prices = df.iloc[target_idx+1 : target_idx+61]
        result = "DRAW"
        final_exit_price = entry_price
        max_price = entry_price
        is_loss = False
        
        actual_high = future_prices['High'].max()
        
        # --- 未来データ走査 ---
        for _, row in future_prices.iterrows():
            high = row['High']; low = row['Low']; open_p = row['Open']
            
            # 1. 損切り判定
            if low <= current_stop_loss:
                exec_price = current_stop_loss
                if open_p < current_stop_loss: exec_price = open_p
                
                loss_amount = (exec_price - entry_price) * shares
                realized_profit += loss_amount
                is_loss = True
                break
            
            # 2. 分割利確判定
            target_price_partial = entry_price * (1 + PARTIAL_PROFIT_TARGET)
            
            if not partial_exit_done and high >= target_price_partial:
                exec_price = target_price_partial
                if open_p > target_price_partial: exec_price = open_p
                
                exit_shares = int(shares * PARTIAL_EXIT_RATIO)
                if exit_shares > 0:
                    profit_amount = (exec_price - entry_price) * exit_shares
                    realized_profit += profit_amount
                    shares -= exit_shares
                    partial_exit_done = True

            # 3. トレーリングストップ
            if high > max_price:
                max_price = high
            
            current_multiplier = TRAILING_WIDE_MULTIPLIER if partial_exit_done else ATR_STOP_MULTIPLIER
            trail_dist = atr * current_multiplier
            new_stop = max_price - trail_dist
            
            profit_pct_max = (max_price - entry_price) / entry_price
            if partial_exit_done or profit_pct_max > 0.03:
                 new_stop = max(new_stop, entry_price * 1.002)
            
            if new_stop > current_stop_loss:
                current_stop_loss = new_stop

        # 期間終了後の強制決済
        if not is_loss and shares > 0:
            final_exit_price = future_prices['Close'].iloc[-1]
            profit_amount = (final_exit_price - entry_price) * shares
            realized_profit += profit_amount

        # 結果判定
        if realized_profit > 0: result = "WIN"; win_count += 1
        elif realized_profit < 0: result = "LOSS"; loss_count += 1
        
        initial_invest = entry_price * initial_shares
        profit_rate = (realized_profit / initial_invest) * 100

        print(f"   結果: {result} (PL: {realized_profit:+.0f}円 / {profit_rate:+.2f}%) Tgt:{ai_target}")

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