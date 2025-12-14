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

# ==========================================
# ★設定エリア: バックテスト設定
# ==========================================
START_DATE = "2024-06-01"  # 検証開始日
INITIAL_CAPITAL = 100000 # 初期資金
RISK_PER_TRADE = 0.02      # 1トレードあたりの許容リスク (2%)
MAX_POSITIONS = 10         # 最大保有銘柄数

# ファイル設定
LOG_FILE = "ai_trade_memory_risk_managed.csv" 
TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15

# 環境設定
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError: pass

GOOGLE_API_KEY = os.getenv("TRAINING_API_KEY")
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = 'models/gemini-2.0-flash'

# 監視リスト (検証用)
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
# 1. 共通関数群
# ==========================================
def download_data_safe(ticker, period="3y", interval="1d", retries=3):
    wait = 1
    for attempt in range(retries):
        try:
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            return df
        except:
            time.sleep(wait); wait *= 2
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

def calculate_metrics_at_date(df, date_idx):
    curr = df.iloc[date_idx]
    price = float(curr['Close'])
    sma25 = float(curr['SMA25'])
    sma25_dev = ((price / sma25) - 1) * 100
    
    prev_sma25 = float(df['SMA25'].iloc[date_idx-5])
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
        'atr_value': atr,
        'rsi_9': float(curr['RSI9'])
    }

def create_chart_image_at_date(df, idx, ticker_name):
    start_idx = max(0, idx - 75)
    data = df.iloc[start_idx : idx + 1].copy()
    if len(data) < 20: return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(data.index, data['Close'], label='Close', color='black', linewidth=1.2)
    ax1.plot(data.index, data['SMA25'], label='SMA25', color='orange', alpha=0.8, linestyle='--')
    ax1.set_title(f"{ticker_name} Analysis")
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
    ax2.plot(data.index, data['MACD'], label='MACD', color='red', linewidth=1.0)
    ax2.bar(data.index, data['MACD']-data['Signal'], color='gray', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# ==========================================
# 2. エージェント定義 (Analyst & Commander)
# ==========================================

# 🕵️‍♂️ 市場分析官 (Market Analyst)
class MarketAnalystAI:
    def __init__(self, model):
        self.model = model

    def analyze(self, ticker, metrics, chart_bytes, similar_cases):
        prompt = f"""
あなたはプロの「株式市場分析官」です。
チャートとデータに基づき、相場環境を客観的に分析してください。

### 分析対象
銘柄: {ticker} | 現在値: {metrics['price']:.0f}円

### テクニカル指標
- Momentum: {metrics['trend_momentum']:.2f} (プラスなら上昇基調)
- SMA25 Dev: {metrics['sma25_dev']:.2f}%
- Volatility: {metrics['entry_volatility']:.2f}% (2.3%以下が望ましい)
- RSI(9): {metrics['rsi_9']:.1f}
- ATR: {metrics['atr_value']:.1f}

### 過去の類似パターン
{similar_cases}

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
        try:
            response = self.model.generate_content(
                [prompt, {'mime_type': 'image/png', 'data': chart_bytes}],
                safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
            )
            return response.text
        except: return "分析エラー"

# 👮‍♂️ 運用指令官 (Strategy Commander)
class StrategyCommanderAI:
    def __init__(self, model):
        self.model = model

    def make_decision(self, ticker, metrics, analyst_report, cash):
        # 資金管理計算 (ヒントとして提示)
        risk_amount = cash * RISK_PER_TRADE
        risk_per_share = metrics['atr_value'] * 2.0
        max_shares = int(risk_amount // risk_per_share) if risk_per_share > 0 else 0
        
        prompt = f"""
あなたは冷徹な「運用指令官」です。
分析官の報告と資金状況に基づき、売買命令を下してください。

### 状況
- 銘柄: {ticker}
- 現在値: {metrics['price']:.0f}円
- 手元資金: {cash:,.0f}円
- 最大購入可能株数（リスク管理上）: {max_shares}株

### 分析官の報告
{analyst_report}

### 鉄の掟（厳守）
1. ボラティリティが2.3%超えならHOLD。
2. 下降トレンド中の逆張り禁止。
3. 少しでも不安要素があればHOLD。

### 任務
JSON形式で指令を出力してください。
{{
  "action": "BUY" or "HOLD",
  "shares": (自信度に応じて最大株数以下で調整),
  "stop_loss": (現在値 - ATR*2.0 を基準に設定),
  "reason": (理由を100文字以内で)
}}
"""
        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match: return json.loads(match.group(0))
        except: return {"action": "HOLD", "reason": "System Error"}
        return {"action": "HOLD", "reason": "No response"}

# ==========================================
# 3. システムコンポーネント
# ==========================================
class MemorySystem:
    def __init__(self, csv_path):
        self.df = pd.DataFrame()
        if os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path)
                self.valid_df = self.df[self.df['result'].isin(['WIN', 'LOSS'])].copy()
                if len(self.valid_df) > 5:
                    cols = ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility', 'rsi_9']
                    features = self.valid_df[cols].fillna(0)
                    self.scaler = StandardScaler()
                    self.knn = NearestNeighbors(n_neighbors=10).fit(self.scaler.fit_transform(features))
            except: pass

    def get_similar_cases_text(self, metrics):
        if not hasattr(self, 'knn'): return "データ不足"
        vec = [metrics[c] for c in ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility', 'rsi_9']]
        dists, indices = self.knn.kneighbors(self.scaler.transform([vec]))
        
        wins = 0
        for idx in indices[0]:
            if self.valid_df.iloc[idx]['result'] == 'WIN': wins += 1
        win_rate = wins / len(indices[0]) * 100
        return f"過去の類似局面勝率: {win_rate:.1f}%"

# ==========================================
# 4. バックテスト実行メイン
# ==========================================
def main():
    print(f"=== 📅 AI Dual-Agent バックテスト ({START_DATE} 〜) ===")
    print(f"初期資金: {INITIAL_CAPITAL:,.0f}円 | リスク: {RISK_PER_TRADE*100}%")
    
    # 準備
    memory = MemorySystem(LOG_FILE)
    model = genai.GenerativeModel(MODEL_NAME)
    analyst = MarketAnalystAI(model)
    commander = StrategyCommanderAI(model)

    # 1. データ取得
    print("データ取得中...", end="")
    tickers_data = {}
    for t in TRAINING_LIST:
        df = download_data_safe(t, period="2y")
        if df is not None:
            df = calculate_technical_indicators(df)
            tickers_data[t] = df
    print(f"完了 ({len(tickers_data)}銘柄)")

    # 2. カレンダー生成
    all_dates = sorted(list(set([d for t in tickers_data for d in tickers_data[t].index])))
    start_dt = pd.to_datetime(START_DATE).tz_localize(None)
    sim_dates = [d for d in all_dates if d.tz_localize(None) >= start_dt]
    
    # ポートフォリオ初期化
    cash = INITIAL_CAPITAL
    portfolio = {} 
    trade_history = []
    equity_curve = []

    print(f"\n🎬 シミュレーション開始 ({len(sim_dates)}営業日)...")

    for current_date in sim_dates:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n[{date_str}]", end="", flush=True)
        
        # --- A. ポートフォリオ管理 (決済判定) ---
        closed_tickers = []
        for ticker, pos in portfolio.items():
            df = tickers_data[ticker]
            if current_date not in df.index: continue
            
            day_data = df.loc[current_date]
            day_low = float(day_data['Low'])
            day_high = float(day_data['High'])
            
            # 損切り判定
            if day_low <= pos['sl_price']:
                exec_price = pos['sl_price']
                if float(day_data['Open']) < pos['sl_price']: exec_price = float(day_data['Open'])
                
                proceeds = exec_price * pos['shares']
                cash += proceeds
                profit = proceeds - (pos['buy_price'] * pos['shares'])
                profit_rate = (exec_price - pos['buy_price']) / pos['buy_price'] * 100
                
                icon = "🏆" if profit > 0 else "💀"
                print(f"\n   {icon} 決済 {ticker}: {profit:+,.0f}円 ({profit_rate:+.2f}%)")
                trade_history.append({'Result': 'WIN' if profit>0 else 'LOSS', 'PL': profit})
                closed_tickers.append(ticker)
                continue

            # トレーリング更新 (可変式ラチェット)
            if day_high > pos['max_price']:
                pos['max_price'] = day_high
                profit_pct = (pos['max_price'] - pos['buy_price']) / pos['buy_price']
                
                # ラチェット幅調整
                width = pos['atr'] * 2.0 # 基本
                if profit_pct > 0.05: # +5%超え (鬼利確)
                    width = pos['atr'] * 0.5
                elif profit_pct > 0.03: # +3%超え (利益確保)
                    width = pos['atr'] * 1.5 
                else:
                    # +3%未満は追従せず、初期SLのまま耐える (ノイズ対策)
                    width = 999999 # 実質追従なし

                new_sl = pos['max_price'] - width

                # 建値ガード (発動ラインを 1.5% -> 2.5% に引き上げ)
                if profit_pct > 0.025: 
                    new_sl = max(new_sl, pos['buy_price'] * 1.001)

                # ただし、初期SLより下がることはない
                if new_sl > pos['sl_price']:
                    pos['sl_price'] = new_sl

                new_sl = pos['max_price'] - width
                
                # 建値ガード (+1.5%で発動)
                if profit_pct > 0.015:
                    new_sl = max(new_sl, pos['buy_price'] * 1.001)
                
                if new_sl > pos['sl_price']:
                    pos['sl_price'] = new_sl

        for t in closed_tickers: del portfolio[t]

        # --- B. 新規エントリー探索 ---
        # 資金と枠に余裕がある時だけ
        if len(portfolio) < MAX_POSITIONS and cash > 500000:
            candidates = [t for t in tickers_data.keys() if t not in portfolio]
            import random
            random.shuffle(candidates)
            
            check_count = 0
            for ticker in candidates:
                if check_count >= 3: break # 1日3銘柄まで (API制限対策)
                
                df = tickers_data[ticker]
                if current_date not in df.index: continue
                idx = df.index.get_loc(current_date)
                if idx < 50: continue
                
                metrics = calculate_metrics_at_date(df, idx)
                
                # 鉄の掟フィルター (コードレベル)
                if metrics['entry_volatility'] > 2.3 or metrics['trend_momentum'] < 0 or metrics['sma25_dev'] < 0:
                    continue
                
                # ここまで来たらAI出動
                check_count += 1
                
                chart_bytes = create_chart_image_at_date(df, idx, ticker)
                if not chart_bytes: continue
                
                similar_text = memory.get_similar_cases_text(metrics)
                
                # 1. 分析官
                report = analyst.analyze(ticker, metrics, chart_bytes, similar_text)
                
                # 2. 指令官
                decision = commander.make_decision(ticker, metrics, report, cash)
                
                if decision.get('action') == "BUY":
                    shares = decision.get('shares', 0)
                    if shares > 0:
                        cost = shares * metrics['price']
                        if cost <= cash:
                            cash -= cost
                            atr_val = metrics['atr_value']
                            initial_sl = decision.get('stop_loss', metrics['price'] - atr_val * 2.0)
                            
                            portfolio[ticker] = {
                                'buy_price': metrics['price'], 'shares': shares,
                                'sl_price': initial_sl, 'max_price': metrics['price'], 'atr': atr_val
                            }
                            print(f"\n   🔴 新規 {ticker}: {shares}株 (SL:{initial_sl:.0f})")
                            print(f"      理由: {decision.get('reason')}")
                
                time.sleep(2)

        # --- C. 資産集計 ---
        current_equity = cash
        for t, pos in portfolio.items():
            if current_date in tickers_data[t].index:
                price = float(tickers_data[t].loc[current_date]['Close'])
                current_equity += price * pos['shares']
        equity_curve.append(current_equity)

    # --- 結果表示 ---
    print("\n" + "="*50)
    print(f"🏁 バックテスト終了")
    print(f"最終資産: {equity_curve[-1]:,.0f}円 (初期: {INITIAL_CAPITAL:,}円)")
    
    profit = equity_curve[-1] - INITIAL_CAPITAL
    roi = (profit / INITIAL_CAPITAL) * 100
    print(f"合計損益: {profit:+,.0f}円 ({roi:+.2f}%)")
    
    wins = len([t for t in trade_history if t['Result']=='WIN'])
    losses = len([t for t in trade_history if t['Result']=='LOSS'])
    total = wins + losses
    if total > 0:
        print(f"勝敗: {wins}勝 {losses}敗 (勝率: {wins/total*100:.1f}%)")

if __name__ == "__main__":
    main()