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
import subprocess 
import logging

# ==========================================
# ★追加: GitHub自動同期機能
# ==========================================
def auto_git_push(commit_message="Auto update trade memory"):
    """
    学習結果(CSV)を自動でGitHubにプッシュする
    """
    try:
        print("\n☁️ GitHubへ同期中...")
        subprocess.run(["git", "add", "ai_trade_memory_risk_managed.csv"], check=True)
        try:
            subprocess.run(["git", "commit", "-m", commit_message], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            print("   (変更がないためコミットはスキップされました)")
            
        print("   最新データを取得中...")
        try:
            subprocess.run(["git", "pull", "--rebase"], check=True)
        except subprocess.CalledProcessError:
            print("⚠️ Pull中に競合が発生しました。手動で解決してください。")
            return

        subprocess.run(["git", "push"], check=True)
        print("✅ 同期完了！クラウドダッシュボードが更新されました。")
        
    except Exception as e:
        print(f"⚠️ GitHub同期エラー: {e}")

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# ==========================================
# ★設定エリア
# ==========================================
GOOGLE_API_KEY = os.getenv("TRAINING_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY (TRAINING_API_KEY) が設定されていません。")

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = 'models/gemini-2.0-flash' 
LOG_FILE = "ai_trade_memory_risk_managed.csv" 

TRAINING_ROUNDS = 15000 
TIMEFRAME = "1d" 
CBR_NEIGHBORS_COUNT = 15 
MIN_VOLATILITY = 1.0 
TRADE_BUDGET = 1000000 # 1トレードあたりの予算

# 練習用銘柄リスト
TRAINING_LIST = [
    "8035.T", "6146.T", "6857.T", "6723.T", "7735.T", "6526.T", "6758.T", "6861.T", "6501.T",
    "6594.T", "7751.T", "6702.T", "6752.T", "6981.T", "6954.T", "6920.T",
    "7203.T", "7267.T", "7269.T", "7270.T", "7011.T", "6301.T", "6367.T", "6098.T",
    "8306.T", "8316.T", "8411.T", "8766.T", "8591.T", "8604.T",
    "8058.T", "8031.T", "8001.T", "8002.T", "8053.T",
    "9432.T", "9433.T", "9984.T", "4661.T", "9613.T", "2413.T", "4751.T", "4385.T",
    "9983.T", "3382.T", "8267.T", "2802.T", "2914.T", "4911.T",
    "9101.T", "9104.T", "9107.T", "9020.T", "9021.T", "9201.T",
    "5401.T", "1605.T", "5713.T", "4063.T", "4901.T"
]
plt.rcParams['font.family'] = 'sans-serif' 

# ==========================================
# 1. データ取得 & テクニカル計算
# ==========================================
def download_data_safe(ticker, period="10y", interval="1d", retries=5):
    wait = 2
    for attempt in range(retries):
        try:
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            if len(df) < 50: return None
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
                wait *= 2
            else:
                return None
    return None

def calculate_technical_indicators(df):
    df = df.copy()
    df['SMA25'] = df['Close'].rolling(25).mean()
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['VolumeMA5'] = df['Volume'].rolling(5).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    rs = gain / loss
    df['RSI9'] = 100 - (100 / (1 + rs))
    return df.dropna()

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
    std = df['Close'].iloc[idx-19:idx+1].std()
    bb_width = (4 * std) / df['Close'].iloc[idx-19:idx+1].mean() * 100
    vol_ma5 = float(curr['VolumeMA5'])
    volume_ratio = float(curr['Volume']) / vol_ma5 if vol_ma5 > 0 else 1.0
    
    return {
        'sma25_dev': sma25_dev,
        'trend_momentum': trend_momentum,
        'macd_power': macd_power,
        'entry_volatility': entry_volatility, 
        'price': price,
        'atr_value': atr,
        'bb_width': bb_width,
        'volume_ratio': volume_ratio,
        'rsi_9': float(curr['RSI9'])
    }

# ==========================================
# 2. CBRメモリシステム
# ==========================================
class CaseBasedMemory:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        self.feature_cols = ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility', 'rsi_9']
        
        self.csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "stop_loss_reason", "Price", 
            "sma25_dev", "trend_momentum", "macd_power", "entry_volatility", 
            "rsi_9", "profit_loss", "profit_rate"
        ]
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # --- スキーマ更新 & 過去データの再計算 ---
            cols_added = False
            for col in self.csv_columns:
                if col not in self.df.columns:
                    self.df[col] = 0.0 if col != 'Reason' else ''
                    cols_added = True
            
            if 'profit_rate' in self.df.columns and 'profit_loss' in self.df.columns and 'Price' in self.df.columns:
                mask = (self.df['profit_rate'] == 0) & (self.df['profit_loss'] != 0) & (self.df['Price'] != 0)
                if mask.any():
                    print(f"🔄 過去データの利益率を自動計算して補完します ({mask.sum()}件)...")
                    self.df.loc[mask, 'profit_rate'] = (self.df.loc[mask, 'profit_loss'] / self.df.loc[mask, 'Price']) * 100
                    cols_added = True

            if cols_added:
                self.df = self.df[self.csv_columns] 
                self.df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
                print("✅ CSVファイルを最新形式にアップデートしました。")

        except Exception as e:
            print(f"⚠️ CSV読込エラー: {e}")
            try:
                self.df = pd.read_csv(self.csv_path, on_bad_lines='skip')
                for col in self.csv_columns:
                    if col not in self.df.columns: self.df[col] = 0.0
                self.df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
            except Exception: return

        try:
            rename_map = {
                'date': 'Date', 'ticker': 'Ticker', 'action': 'Action', 
                'result': 'result', 'profit_loss': 'profit_loss',
                'rsi_9': 'rsi_9', 'rsi': 'rsi_9'
            }
            self.df.columns = [rename_map.get(col.lower(), col) for col in self.df.columns]
            
            valid_df = self.df[self.df['result'].isin(['WIN', 'LOSS'])].copy()
            if len(valid_df) < 5: return

            for col in self.feature_cols:
                 if col not in valid_df.columns: valid_df[col] = 0.0
            
            features = valid_df[self.feature_cols].fillna(0)
            self.features_normalized = self.scaler.fit_transform(features)
            
            self.valid_df_for_knn = valid_df 
            global CBR_NEIGHBORS_COUNT
            self.knn = NearestNeighbors(n_neighbors=min(CBR_NEIGHBORS_COUNT, len(valid_df)), metric='euclidean')
            self.knn.fit(self.features_normalized)
            print(f"Memory Loaded: {len(valid_df)} valid records.")
        except Exception as e:
            print(f"Memory Init Error: {e}")

    def search_similar_cases(self, current_metrics):
        if self.knn is None: return "（学習データ不足）"
        metrics_vec = [current_metrics.get(col, 0) for col in self.feature_cols]
        # DataFrame化して警告回避
        input_df = pd.DataFrame([metrics_vec], columns=self.feature_cols)
        scaled_vec = self.scaler.transform(input_df)
        distances, indices = self.knn.kneighbors(scaled_vec)
        
        text = f"【過去の類似局面 ({len(indices[0])}件)】\n"
        win_c = 0; loss_c = 0
        for idx in indices[0]:
            row = self.valid_df_for_knn.iloc[idx]
            res = str(row.get('result', ''))
            if res == 'WIN': win_c += 1
            if res == 'LOSS': loss_c += 1
            icon = "⭕" if res == 'WIN' else "❌"
            text += f"- {row.get('Date')} {row.get('Ticker')}: {icon} (MOM:{row.get('trend_momentum',0):.1f})\n"
        text += f"-> 類似データ傾向: 勝ち{win_c} / 負け{loss_c}\n"
        return text

    def save_experience(self, data_dict):
        new_df = pd.DataFrame([data_dict])
        for col in self.csv_columns:
            if col not in new_df.columns: new_df[col] = None
        new_df = new_df[self.csv_columns]

        max_retries = 5
        for i in range(max_retries):
            try:
                if not os.path.exists(self.csv_path):
                    new_df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
                else:
                    new_df.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                self.load_and_train() 
                return
            except PermissionError: time.sleep(2)
            except Exception as e:
                print(f"❌ 保存エラー: {e}"); break

# ==========================================
# 3. AIスパーリング
# ==========================================
def create_chart_image(df, ticker_name):
    data = df.tail(75).copy() 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(data.index, data['Close'], label='Close', color='black', linewidth=1.2)
    ax1.plot(data.index, data['SMA25'], label='SMA25', color='orange', alpha=0.8, linestyle='--')
    ax1.set_title(f"{ticker_name} Trend Chart")
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
    ax2.plot(data.index, data['MACD'], label='MACD', color='red', linewidth=1.0)
    ax2.bar(data.index, data['MACD']-data['Signal'], color='gray', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def ai_decision_maker(model, chart_bytes, metrics, similar_cases_text, ticker):
    # --- 🛡️ 鉄の掟フィルター (2.3%採用) ---
    if metrics['entry_volatility'] > 2.3:
        return {"action": "HOLD", "confidence": 0, "reason": f"【鉄の掟】ボラティリティ過大 ({metrics['entry_volatility']:.2f}%)", "stop_loss_price": 0}
    if metrics['trend_momentum'] < 0:
        return {"action": "HOLD", "confidence": 0, "reason": "【鉄の掟】下降トレンド中 (Momentum < 0)", "stop_loss_price": 0}
    if metrics['sma25_dev'] < 0:
        return {"action": "HOLD", "confidence": 0, "reason": "【鉄の掟】SMA25割れ (戻り待ち)", "stop_loss_price": 0}

    prompt = f"""
### CONTEXT
対象: {ticker}
【テクニカル指標】
- トレンド勢い: {metrics['trend_momentum']:.2f} (プラス必須)
- SMA25乖離率: {metrics['sma25_dev']:.2f}% (プラス圏)
- ボラティリティ: {metrics['entry_volatility']:.2f}% (2.3%以下推奨)
- RSI(9): {metrics['rsi_9']:.1f}
- ATR: {metrics['atr_value']:.1f}

【ニュース】
{similar_cases_text}
### TASK
「資産防衛型AI」として「買い (BUY)」か「様子見 (HOLD)」のみ判定せよ。空売り不可。
売却判断はATRトレーリングストップに一任するため、ここでは「エントリーの優位性」のみを審査する。

### RULES (統計的優位性に基づく鉄の掟)

**1. エントリー禁止 (即時HOLD対象):**
   以下のいずれか1つでも該当する場合は、絶対にBUYしてはならない。
   - **ボラティリティ >= 2.6%**: (勝率34%以下) 相場が荒れており危険。
   - **SMA25乖離率 <= 0.5%**: (勝率32%以下) トレンドが出ていない、または逆張り。SMA25は四捨五入ではなく
   - **MACDパワー <= 0**: (勝率低) 下落圧力が残っている。

**2. BUY (新規買い) の条件:**
   *前提: 上記の禁止条件を全てクリアしていること。*
   
   - **[ゴールデン・ゾーン]:**
     - SMA25乖離率が **+0.5% 〜 +4.7%** の範囲にある。
     - MACDパワーがプラスで推移している。
     - RSIが 40〜65 の範囲（過熱感がない）。

### SCORING (自信度の採点 - 厳格化)**
   データ分析の結果、**自信過剰(85点以上)は負けフラグ**であることが判明している。
   - **80-85 (推奨):** [ゴールデン・ゾーン] に完全に合致し、ボラティリティが2.0%未満の場合。
   - **60-79 (慎重):** 条件は満たすが、ボラティリティが2.0%〜2.6%の場合。
   - **0 (論外):** 禁止条件に1つでも該当する場合。自信度を0にせよ。

### OUTPUT FORMAT (JSON ONLY)
{{
  "action": "BUY", "HOLD", "SELL",
  "confidence": 0-100,
  "stop_loss_price": 0.0,
  "stop_loss_reason": "理由",
  "target_price": 0.0,
  "reason": "理由(100文字以内)"
}}
"""
    safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}], safety_settings=safety_settings)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: text = match.group(0)
        return json.loads(text)
    except Exception as e:
        return {"action": "HOLD", "reason": f"Error: {e}", "confidence": 0, "stop_loss_price": 0}

# ==========================================
# 4. メイン実行 (★攻撃的利益追求ロジック版)
# ==========================================
def main():
    print(f"=== AI強化合宿 (Aggressive Profit Pursuit ver) ===")
    
    memory_system = CaseBasedMemory(LOG_FILE) 
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e: print(f"Model Init Error: {e}"); return

    processed_data = {}
    print(f"Downloading data...")
    for t in TRAINING_LIST:
        df = download_data_safe(t, interval=TIMEFRAME)
        if df is None or len(df) < 150: continue
        df = calculate_technical_indicators(df)
        processed_data[t] = df

    if not processed_data: print("データ不足のため終了します。"); return

    win_count = 0; loss_count = 0
    total_profit_loss = 0.0 
    total_invested_count = 0 
    
    print(f"\n🥊 トレーニング開始 ({TRAINING_ROUNDS}ラウンド)\n")
    
    for i in range(1, TRAINING_ROUNDS + 1):
        ticker = random.choice(list(processed_data.keys()))
        df = processed_data[ticker]
        if len(df) < 110: continue 
        target_idx = random.randint(100, len(df) - 10) 
        current_date_str = df.index[target_idx].strftime('%Y-%m-%d')
        metrics = calculate_metrics_enhanced(df, target_idx)
        
        cbr_text = memory_system.search_similar_cases(metrics)
        past_df = df.iloc[:target_idx+1]
        chart_bytes = create_chart_image(past_df, ticker)
        
        decision = ai_decision_maker(model_instance, chart_bytes, metrics, cbr_text, ticker)
        action = decision.get('action', 'HOLD')
        if action == "HOLD" and "鉄の掟" in decision.get('reason', ''): continue

        conf = decision.get('confidence', 0)
        if action == "BUY" :
            print(f"Round {i:03}: {ticker} ({current_date_str}) -> BUY 🔴 (自信:{conf}%)")

        result = "DRAW"; profit_loss = 0.0; profit_rate = 0.0
        
        if action == "BUY":
            entry_price = float(metrics['price'])
            shares = int(TRADE_BUDGET // entry_price)
            if shares < 1: shares = 1
            invested_capital = shares * entry_price

            current_atr = float(metrics['atr_value']) 
            
            # 初期SL: ATR x 2.0
            current_stop_loss = entry_price - (current_atr * 1.5)
            max_price = entry_price 

            # 未来データ (最大60営業日まで追跡)
            future_prices = df['Close'].iloc[target_idx+1 : target_idx+61]
            future_lows = df['Low'].iloc[target_idx+1 : target_idx+61]
            future_highs = df['High'].iloc[target_idx+1 : target_idx+61]
            
            is_win = False; is_loss = False
            final_exit_price = entry_price

            if len(future_prices) > 0:
                for j in range(len(future_prices)):
                    p_low = future_lows.iloc[j]; p_high = future_highs.iloc[j]
                    
                    if p_low <= current_stop_loss:
                        is_loss = True
                        final_exit_price = current_stop_loss 
                        break
                    
                    # ★攻撃的トレーリングロジック★
                    if p_high > max_price:
                        max_price = p_high
                        current_profit_pct = (max_price - entry_price) / entry_price
                        
                        # ★ここを変更: 利益を伸ばす設定
                        if current_profit_pct > 0.10:   # +10%超え: ようやく激狭追従
                            trail_width = current_atr * 0.5 
                        elif current_profit_pct > 0.05: # +5%超え: まだ余裕を持たせる
                            trail_width = current_atr * 1.0 
                        elif current_profit_pct > 0.03: # +3%超え: 標準追従
                            trail_width = current_atr * 1.5
                        else:
                            trail_width = 999999 # 追従しない
                            
                        new_stop_loss = max_price - trail_width
                        
                        # 建値ガード (+2.5%で発動は維持)
                        if current_profit_pct > 0.025:
                             break_even_price = entry_price * 1.001
                             new_stop_loss = max(new_stop_loss, break_even_price)

                        if new_stop_loss > current_stop_loss:
                            current_stop_loss = new_stop_loss
                
                if not is_loss:
                    final_exit_price = future_prices.iloc[-1] 

                profit_loss = (final_exit_price - entry_price) * shares
                profit_rate = ((final_exit_price - entry_price) / entry_price) * 100
                
                if profit_loss > 0: result = "WIN"; win_count += 1
                elif profit_loss < 0: result = "LOSS"; loss_count += 1
                
                total_profit_loss += profit_loss
                total_invested_count += 1

            else: result = "Unknown"

            icon = "🏆" if result == "WIN" else "💀" if result == "LOSS" else "➖"
            print(f"   結果: {icon} {result} (PL: {profit_loss:+.0f}円 / {profit_rate:+.2f}%) 株数:{shares} > {decision.get('reason')}")
            
            save_data = {
                'Date': current_date_str, 'Ticker': ticker, 'Timeframe': TIMEFRAME, 
                'Action': action, 'result': result, 
                'Reason': decision.get('reason', 'None'),
                'Confidence': conf,
                'stop_loss_price': current_stop_loss, 'stop_loss_reason': "Aggressive_Trailing", 
                'Price': metrics['price'], 'sma25_dev': metrics['sma25_dev'], 
                'trend_momentum': metrics['trend_momentum'], 'macd_power': metrics['macd_power'],
                'entry_volatility': metrics['entry_volatility'], 'rsi_9': metrics['rsi_9'], 
                'profit_loss': profit_loss, 'profit_rate': profit_rate # ★必須
            }
            memory_system.save_experience(save_data)
        time.sleep(1)

    print(f"\n=== 合宿終了 ===")
    print(f"戦績 (BUY): {win_count}勝 {loss_count}敗")
    print(f"合計損益: {total_profit_loss:+.0f}円")
    
    total_invested_sum = total_invested_count * TRADE_BUDGET
    if total_invested_sum > 0:
        total_roi = (total_profit_loss / total_invested_sum) * 100
        print(f"総収益率 (ROI): {total_roi:+.2f}%")

if __name__ == "__main__":
    main()
    auto_git_push(commit_message="Training Camp: Aggressive Profit Logic")