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
# ★設定エリア: V7 (Sniper & Home Run) バックテスト
# ==========================================
START_DATE = "2023-01-01"
END_DATE   = "2025-11-30"

INITIAL_CAPITAL = 100000 
RISK_PER_TRADE = 0.40      
MAX_POSITIONS = 10         
MAX_INVEST_RATIO = 0.4    

# ★ V7 ロジックパラメータ
ATR_STOP_MULTIPLIER = 1.8      # 初期損切り幅 (ATR x 1.8)
TRAILING_TRIGGER = 0.10        # トレーリング開始ライン (+10%までは耐える)
TRAILING_MULTIPLIER = 2.0      # トレーリング追従幅 (ATR x 2.0)

# 鉄の掟
MA_DEV_DANGER_LOW = 10.0     
MA_DEV_DANGER_HIGH = 15.0    

# 保存ファイル名
LOG_FILE = "ai_trade_memory_aggressive_v7.csv" 
HISTORY_CSV = "backtest_history_v7.csv" 

TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15
MODEL_NAME = 'models/gemini-2.0-flash'

# 監視銘柄リスト
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

# ---------------------------------------------------------
# 環境設定
# ---------------------------------------------------------
def allowed_gai_family():
    import socket
    return socket.AF_INET
import requests.packages.urllib3.util.connection as urllib3_cn
urllib3_cn.allowed_gai_family = allowed_gai_family

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 1. データ取得 & テクニカル計算 (V7仕様)
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

    # ★ V7追加: MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # ★ V7追加: 一目均衡表 (雲)
    high9 = high.rolling(9).max(); low9 = low.rolling(9).min()
    tenkan = (high9 + low9) / 2
    high26 = high.rolling(26).max(); low26 = low.rolling(26).min()
    kijun = (high26 + low26) / 2
    
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    high52 = high.rolling(52).max(); low52 = low.rolling(52).min()
    senkou_b = ((high52 + low52) / 2).shift(26)
    
    df['Cloud_Top'] = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)

    return df.dropna()

def calculate_metrics_at_date(df, idx):
    curr = df.iloc[idx]
    price = float(curr['Close'])
    
    recent_high = df['High'].iloc[idx-60:idx].max()
    dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
    
    adx = float(curr['ADX'])
    prev_adx = float(df['ADX'].iloc[idx-1])
    
    sma25 = float(curr['SMA25'])
    ma_deviation = ((price / sma25) - 1) * 100
    
    bb_width = float(curr['BB_Width'])
    prev_width = float(df['BB_Width'].iloc[idx-5]) if df['BB_Width'].iloc[idx-5] > 0 else 0.1
    expansion_rate = bb_width / prev_width
    
    vol_ratio = float(curr['Volume']) / float(curr['Vol_MA20']) if float(curr['Vol_MA20']) > 0 else 0
    
    # 出来高推移 (5日)
    vol_history = []
    for i in range(4, -1, -1):
        if idx-i >= 0:
            row = df.iloc[idx-i]
            vr = float(row['Volume']) / float(row['Vol_MA20']) if float(row['Vol_MA20']) > 0 else 0
            vol_history.append(f"{vr:.1f}")
    vol_history_str = "->".join(vol_history)

    # V7指標
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
        'ma_deviation': ma_deviation,
        'adx': adx,
        'prev_adx': prev_adx,
        'plus_di': float(curr['PlusDI']),
        'minus_di': float(curr['MinusDI']),
        'vol_ratio': vol_ratio,
        'vol_history': vol_history_str,
        'expansion_rate': expansion_rate,
        'atr_value': float(curr['ATR']),
        'macd_val': macd,
        'macd_hist': macd_hist,
        'macd_trend': "Expanding" if abs(macd_hist) > abs(prev_hist) else "Shrinking",
        'price_vs_cloud': price_vs_cloud,
        'candle_shape': candle_shape,
        'rsi_9': float(curr['RSI9'])
    }

# ==========================================
# 2. 鉄の掟 & 補助関数 (V7)
# ==========================================
def check_iron_rules(metrics):
    if metrics['adx'] < 20: return "ADX<20"
    if metrics['vol_ratio'] < 0.8: return "Vol<0.8"
    
    ma_dev = metrics['ma_deviation']
    if MA_DEV_DANGER_LOW <= ma_dev <= MA_DEV_DANGER_HIGH: 
        return f"DangerZone({ma_dev:.1f}%)"
    if metrics['adx'] > 55: return "ADX Overheat"
    
    # ★V7追加: 雲の下でのロングは禁止
    if metrics['price_vs_cloud'] == "Below": return "Below Ichimoku Cloud"
    
    return None

def create_chart_image_at_date(df, idx, ticker):
    try:
        data = df.iloc[idx-60:idx+1].copy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        sma20 = data['Close'].rolling(20).mean()
        std20 = data['Close'].rolling(20).std()
        ax1.plot(data.index, data['Close'], color='black', label='Close')
        ax1.plot(data.index, sma20 + 2*std20, color='green', alpha=0.5, linestyle='--', label='+2σ')
        ax1.plot(data.index, sma20 - 2*std20, color='green', alpha=0.5, linestyle='--', label='-2σ')
        
        # 雲
        if 'Cloud_Top' in data.columns:
            ax1.plot(data.index, data['Cloud_Top'], color='blue', alpha=0.2, label='Cloud Top')
            ax1.fill_between(data.index, data['Cloud_Top'], data['Close'].min(), color='blue', alpha=0.05)

        ax1.set_title(f"{ticker} V7 Chart")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"Chart Error: {e}")
        return None

# ==========================================
# 3. CBRメモリシステム
# ==========================================
class MemorySystem:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        self.feature_cols = ['adx', 'prev_adx', 'ma_deviation', 'vol_ratio', 'expansion_rate', 'dist_to_res']
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception: return

        try:
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

    def get_similar_cases_text(self, current_metrics):
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

# ==========================================
# 4. AI エージェント (V7 Sniper)
# ==========================================
def run_analyst(model, ticker, metrics, chart_bytes, cbr_text):
    prompt = f"""
### ROLE
あなたは「高精度スナイパー・トレンドフォローAI」です。
ダマシ(False Breakout)を極限まで回避し、本物のトレンド初動のみを狙撃します。

### INPUT DATA
銘柄: {ticker} (現在価格: {metrics['price']:.0f}円)

[基本指標]
1. Trend (ADX): {metrics['adx']:.1f} (閾値25以上)
2. Direction: +DI({metrics['plus_di']:.1f}) vs -DI({metrics['minus_di']:.1f})
3. Volatility: {metrics['expansion_rate']:.2f}倍 (スクイーズからの拡大が良い)
4. Volume: {metrics['vol_ratio']:.2f}倍
   - 推移: {metrics['vol_history']}

[★ダマシ回避・精密検査]
1. **MACD**: Hist={metrics['macd_hist']:.2f} ({metrics['macd_trend']})
   - ヒストグラムがプラス圏で拡大中なら強い。マイナスなら警戒。
2. **Ichimoku Cloud**: Price is {metrics['price_vs_cloud']} the Cloud.
   - 雲の下(Below)での買いは自殺行為のため禁止。
3. **Candle Shape**: {metrics['candle_shape']}
   - 長い上ヒゲ(Bad)は売り圧力の証明。大陽線(Good)が理想。
4. **Resistance**: 距離 {metrics['dist_to_res']:.1f}%

{cbr_text}

### EVALUATION LOGIC
- **BUY条件**:
  1. 抵抗線を明確に超えている、または直前でMACD等のモメンタムが強い。
  2. 価格が「雲」の上にあること (必須)。
  3. ローソク足に長い上ヒゲがないこと。
  4. 出来高が伴っていること。

- **HOLD条件**:
  - 上記のいずれかに懸念がある場合。特に「上ヒゲ」や「雲の下」は即HOLD。

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
        return response.text
    except Exception:
        return "分析エラー"

def run_commander_batch(model, candidates_data, current_cash, current_portfolio_text):
    candidates_text = ""
    max_invest_amount = current_cash * MAX_INVEST_RATIO 
    
    for c in candidates_data:
        risk_per_share = c['metrics']['atr_value'] * 2.0
        risk_based_shares = int((current_cash * RISK_PER_TRADE) // risk_per_share) if risk_per_share > 0 else 0
        cap_based_shares = int(max_invest_amount // c['metrics']['price'])
        final_max_shares = min(risk_based_shares, cap_based_shares)
        if final_max_shares < 1: final_max_shares = 1 

        candidates_text += f"""
--- 候補: {c['ticker']} ---
現在値: {c['metrics']['price']:.0f}円
推奨最大株数: {final_max_shares}株
【分析官報告】
{c['report']}
-------------------------
"""

    prompt = f"""
あなたは冷徹な運用指令官（ファンドマネージャー）です。
分析官から上がってきた有望銘柄のレポートと、現在の資金・ポートフォリオ状況を総合的に判断し、ベストな買い注文を決定してください。

### 現在の状況
- 手元資金: {current_cash:,.0f}円
- 保有銘柄: {current_portfolio_text}
- 全体方針: 資産防衛最優先。自信のある銘柄に絞る。

### 候補銘柄レポート
{candidates_text}

### 鉄の掟（厳守）
1. ボラティリティが高い銘柄、下降トレンドの銘柄は除外。
2. アナリスト報告に不安要素がある場合は見送り。
3. 資金内で買える範囲に収めること。

### あなたの任務
**JSON形式**で、実際にエントリーする銘柄と数量のリストを出力してください。
見送る銘柄はリストに含めなくて良いです。

出力フォーマット:
{{
  "orders": [
    {{
      "ticker": "銘柄コード",
      "action": "BUY",
      "shares": 購入株数 (整数),
      "stop_loss": 損切り価格 (数値のみ),
      "reason": "選定理由を50文字以内で"
    }}
  ]
}}
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: return {"orders": []}
    return {"orders": []}

# ==========================================
# 5. メイン実行
# ==========================================
def main():
    print(f"=== 🧪 酸性試験 (V7: Sniper & Home Run) ({START_DATE} ~ {END_DATE}) ===")
    print(f"初期資金: {INITIAL_CAPITAL:,.0f}円 | 損切: ATRx{ATR_STOP_MULTIPLIER}")
    print(f"利確設定: +{TRAILING_TRIGGER*100}%超えまで我慢 -> 以降ATRx{TRAILING_MULTIPLIER}追従")

    memory = MemorySystem(LOG_FILE)
    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"Model Init Error: {e}")
        return

    print("データ取得中...", end="")
    tickers_data = {}
    for t in TRAINING_LIST:
        df = download_data_safe(t)
        if df is not None:
            df = calculate_technical_indicators(df)
            tickers_data[t] = df
    print(f"完了 ({len(tickers_data)}銘柄)")

    all_dates = sorted(list(set([d for t in tickers_data for d in tickers_data[t].index])))
    start_dt = pd.to_datetime(START_DATE).tz_localize(None)
    end_dt = pd.to_datetime(END_DATE).tz_localize(None)

    sim_dates = [d for d in all_dates if start_dt <= d.tz_localize(None) <= end_dt]

    if not sim_dates:
        print("期間内のデータがありません。")
        return

    cash = INITIAL_CAPITAL
    portfolio = {} 
    trade_history = []
    equity_curve = []
    daily_history = []

    print(f"\n🎬 シミュレーション開始 ({len(sim_dates)}営業日)...")

    for current_date in sim_dates:
        date_str = current_date.strftime('%Y-%m-%d')

        # --- A. ポートフォリオ管理 ---
        closed_tickers = []
        for ticker, pos in portfolio.items():
            df = tickers_data[ticker]
            if current_date not in df.index: continue

            day_data = df.loc[current_date]
            day_low = float(day_data['Low'])
            day_high = float(day_data['High'])
            day_open = float(day_data['Open'])
            
            # --- 1. 損切り判定 (Stop Loss) ---
            current_sl = float(pos['sl_price'])
            if day_low <= current_sl:
                exec_price = current_sl
                if day_open < current_sl: exec_price = day_open # ギャップダウン

                proceeds = exec_price * pos['shares']
                cash += proceeds
                profit = proceeds - (pos['buy_price'] * pos['shares'])
                profit_rate = (exec_price - pos['buy_price']) / pos['buy_price'] * 100

                icon = "🏆" if profit > 0 else "💀"
                print(f"\n[{date_str}] {icon} 決済 {ticker}: {profit:+,.0f}円 ({profit_rate:+.2f}%)")
                trade_history.append({'Result': 'WIN' if profit>0 else 'LOSS', 'PL': profit})
                closed_tickers.append(ticker)
                continue

            # --- 2. トレーリングストップ更新 (Home Run Logic) ---
            if day_high > pos['max_price']:
                pos['max_price'] = day_high
            
            # 現在の含み益率（最高値ベース）
            profit_pct_high = (pos['max_price'] - pos['buy_price']) / pos['buy_price']
            
            # ★ +10% を超えるまではストップを動かさない（初期損切りで耐える）
            if profit_pct_high > TRAILING_TRIGGER:
                # +10%超えたら、ATR x 2.0 で追従開始
                trail_dist = pos['atr'] * TRAILING_MULTIPLIER
                new_sl = pos['max_price'] - trail_dist
                
                # 建値保証 (+15%以上伸びたら)
                if profit_pct_high > 0.15:
                     new_sl = max(new_sl, pos['buy_price'] * 1.005) # 手数料+α確保
                
                if new_sl > pos['sl_price']:
                    pos['sl_price'] = new_sl

        for t in closed_tickers: del portfolio[t]

        # --- B. バッチ新規エントリー ---
        if len(portfolio) < MAX_POSITIONS and cash > 10000:
            candidates_data = []

            for ticker in tickers_data.keys():
                if ticker in portfolio: continue
                df = tickers_data[ticker]
                if current_date not in df.index: continue
                idx = df.index.get_loc(current_date)
                if idx < 60: continue

                metrics = calculate_metrics_at_date(df, idx)

                # 鉄の掟チェック (V7)
                iron_rule_check = check_iron_rules(metrics)
                if iron_rule_check: continue 

                if len(candidates_data) >= 5: break

                chart_bytes = create_chart_image_at_date(df, idx, ticker)
                if not chart_bytes: continue
                similar_text = memory.get_similar_cases_text(metrics)
                report = run_analyst(model, ticker, metrics, chart_bytes, similar_text)

                candidates_data.append({'ticker': ticker, 'metrics': metrics, 'report': report})
                time.sleep(1) 

            if candidates_data:
                current_portfolio_text = ", ".join([t for t in portfolio.keys()]) or "なし"
                decision_data = run_commander_batch(model, candidates_data, cash, current_portfolio_text)

                for order in decision_data.get('orders', []):
                    tic = order.get('ticker')
                    
                    try:
                        raw_shares = order.get('shares', 0)
                        if isinstance(raw_shares, str): raw_shares = float(raw_shares.replace(',', ''))
                        shares = int(raw_shares)
                    except: shares = 0

                    if shares > 0:
                        target = next((c for c in candidates_data if c['ticker'] == tic), None)
                        if target:
                            metrics = target['metrics']
                            cost = shares * metrics['price']
                            
                            if cost <= cash:
                                cash -= cost
                                atr_val = metrics['atr_value']
                                
                                try:
                                    raw_sl = order.get('stop_loss')
                                    if isinstance(raw_sl, str): raw_sl = float(raw_sl.replace(',', ''))
                                    
                                    if raw_sl and float(raw_sl) > 0:
                                        initial_sl = float(raw_sl)
                                    else:
                                        initial_sl = metrics['price'] - atr_val * ATR_STOP_MULTIPLIER
                                except:
                                    initial_sl = metrics['price'] - atr_val * ATR_STOP_MULTIPLIER
                                
                                portfolio[tic] = {
                                    'buy_price': metrics['price'], 'shares': shares,
                                    'sl_price': initial_sl, 'max_price': metrics['price'], 'atr': atr_val
                                }
                                print(f"\n[{date_str}] 🔴 新規 {tic}: {shares}株 (約{cost:,.0f}円)")

        # --- C. 資産集計 ---
        current_equity = cash
        holdings_val = 0
        holdings_detail = []

        for t, pos in portfolio.items():
            if current_date in tickers_data[t].index:
                price = float(tickers_data[t].loc[current_date]['Close'])
                val = price * pos['shares']
                current_equity += val
                holdings_val += val
                holdings_detail.append(f"{t}:{pos['shares']}")

        print(f"\r[{date_str}] 資産: {current_equity:,.0f}円 (現金: {cash:,.0f}円 / 株: {holdings_val:,.0f}円)", end="")
        equity_curve.append(current_equity)

        daily_history.append({
            "Date": date_str,
            "Total_Equity": int(current_equity),
            "Cash": int(cash),
            "Holdings_Value": int(holdings_val),
            "Positions_Count": len(portfolio),
            "Holdings_Detail": ", ".join(holdings_detail)
        })

    # --- 終了処理 ---
    print("\n" + "="*50)
    print(f"🏁 バックテスト終了")

    if daily_history:
        df_history = pd.DataFrame(daily_history)
        df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
        print(f"📄 詳細履歴を '{HISTORY_CSV}' に保存しました。")

    final_equity = equity_curve[-1] if equity_curve else INITIAL_CAPITAL
    profit = final_equity - INITIAL_CAPITAL
    roi = (profit / INITIAL_CAPITAL) * 100
    print(f"最終資産: {final_equity:,.0f}円")
    print(f"合計損益: {profit:+,.0f}円 ({roi:+.2f}%)")

    wins = len([t for t in trade_history if t['Result']=='WIN'])
    losses = len([t for t in trade_history if t['Result']=='LOSS'])
    total = wins + losses
    if total > 0:
        print(f"勝敗: {wins}勝 {losses}敗 (勝率: {wins/total*100:.1f}%)")

if __name__ == "__main__":
    main()