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
# ★設定エリア: V9 (Expanded List)
# ==========================================
START_DATE = "2023-01-01"
END_DATE   = "2025-11-30"

INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.40      
MAX_POSITIONS = 12         # 銘柄増えたので最大保有数も少し増やす
MAX_INVEST_RATIO = 0.4    # 分散投資のため1銘柄の上限を40%に

# ★ V9 ロジックパラメータ
MARKET_ADX_THRESHOLD = 25.0    

# [A] ゲリラモード 
GUERRILLA_TARGET = 0.06        
GUERRILLA_STOP = 1.5           

# [B] ホームランモード
HOMERUN_STOP_INIT = 1.8        
HOMERUN_TRAIL_TRIGGER = 0.10   
HOMERUN_TRAIL_WIDTH = 2.0      

# 保存ファイル名
LOG_FILE = "ai_trade_memory_aggressive_v9_exp.csv" 
HISTORY_CSV = "backtest_history_v9_exp.csv" 

TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15
MODEL_NAME = 'models/gemini-2.0-flash'

# === 銘柄リスト (拡張版) ===

# 1. 主力・大型株リスト (トレンド相場で輝く: CORE)
LIST_CORE = [
    # 半導体・ハイテク
    "8035.T", "6857.T", "6146.T", "6920.T", "6758.T", "6702.T", "6501.T", "6503.T", "7751.T", "4063.T", "6981.T", "6723.T",
    # 自動車・機械
    "7203.T", "7267.T", "6902.T", "6301.T", "6367.T", "7011.T", "7013.T", 
    # 金融・商社
    "8306.T", "8316.T", "8411.T", "8766.T", "8058.T", "8001.T", "8031.T", "8002.T", "9984.T",
    # 内需・通信・その他
    "9432.T", "9983.T", "4568.T", "4543.T", "4661.T", "7974.T", "6506.T"
]

# 2. 中小型・材料株・高ボラリスト (レンジ相場で輝く: GROWTH)
LIST_GROWTH = [
    # AI・SaaS・ネット
    "5253.T", "5032.T", "9166.T", "4385.T", "4478.T", "4483.T", "3993.T", "4180.T", "3687.T", "6027.T",
    # 宇宙・防衛・深海
    "5595.T", "9348.T", "7012.T", "6203.T", "186A", # 186Aはアストロスケール(対応していれば)
    # 半導体中小型
    "6254.T", "6315.T", "6526.T", "6228.T", "6963.T", "3436.T", "7735.T", "6890.T",
    # エンタメ・消費
    "2768.T", "7342.T", "2413.T", "2222.T", "7532.T", "3092.T",
    # 海運・資源・市況
    "9101.T", "9104.T", "9107.T", "1605.T", "5713.T", "5401.T", "5411.T"
]

# 重複除去 & ソート
LIST_CORE = sorted(list(set(LIST_CORE)))
LIST_GROWTH = sorted(list(set(LIST_GROWTH)))

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
    close = df['Close']; high = df['High']; low = df['Low']
    
    df['SMA25'] = close.rolling(25).mean()
    
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

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Width'] = ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20 * 100
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    df['RSI9'] = 100 - (100 / (1 + gain/loss))

    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

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
    
    vol_history = []
    for i in range(4, -1, -1):
        if idx-i >= 0:
            row = df.iloc[idx-i]
            vr = float(row['Volume']) / float(row['Vol_MA20']) if float(row['Vol_MA20']) > 0 else 0
            vol_history.append(f"{vr:.1f}")
    vol_history_str = "->".join(vol_history)

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
    candle_shape = "Good" if shadow_ratio < 0.3 else "Bad"

    return {
        'price': price,
        'resistance_price': recent_high,
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
# 2. 鉄の掟 & 補助関数
# ==========================================
def check_iron_rules(metrics):
    if metrics['adx'] < 20: return "ADX<20"
    if metrics['vol_ratio'] < 0.8: return "Vol<0.8"
    ma_dev = metrics['ma_deviation']
    if 10.0 <= ma_dev <= 15.0: return f"DangerZone({ma_dev:.1f}%)"
    if metrics['adx'] > 55: return "ADX Overheat"
    if metrics['price_vs_cloud'] == "Below": return "Below Cloud"
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
        
        if 'Cloud_Top' in data.columns:
            ax1.plot(data.index, data['Cloud_Top'], color='blue', alpha=0.2, label='Cloud Top')
            ax1.fill_between(data.index, data['Cloud_Top'], data['Close'].min(), color='blue', alpha=0.05)

        ax1.set_title(f"{ticker} Chart")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e: return None

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
# 4. AI エージェント (V9 Variable Prompt)
# ==========================================
def run_analyst(model, ticker, metrics, chart_bytes, cbr_text, strategy_mode):
    
    if strategy_mode == 'HOMERUN':
        role_text = "あなたは「強気トレンドフォロワー」です。強いトレンドに乗って利益を最大化します。"
        strategy_desc = "現在は「戦時(トレンド相場)」です。押し目より高値ブレイクを優先し、小さな過熱感は無視して大きく狙ってください。"
        eval_focus = "1. MACD拡大中か？ 2. 雲の上か？ 3. 新高値更新の勢いがあるか？"
    else:
        role_text = "あなたは「逆張りスナイパー」です。レンジ相場での反発や押し目を狙います。"
        strategy_desc = "現在は「平時(レンジ相場)」です。ブレイクアウトはダマシの可能性が高いです。RSIの売られすぎやバンド下限からの反発を狙ってください。"
        eval_focus = "1. RSIは低位か？ 2. 移動平均線でのサポートはあるか？ 3. 下ヒゲなどの反発サインはあるか？"

    prompt = f"""
### ROLE
{role_text}

### INPUT DATA
銘柄: {ticker} (現在価格: {metrics['price']:.0f}円)
モード: {strategy_mode}

[テクニカル指標]
- ADX: {metrics['adx']:.1f}
- RSI(9): {metrics['rsi_9']:.1f}
- MACD Hist: {metrics['macd_hist']:.2f} ({metrics['macd_trend']})
- Cloud: {metrics['price_vs_cloud']}

### STRATEGY
{strategy_desc}

### EVALUATION FOCUS
{eval_focus}

{cbr_text}

### OUTPUT REQUIREMENT (JSON ONLY)
{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "stop_loss": "推奨損切り価格",
  "target_price": "推奨利確価格",
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
--- 候補: {c['ticker']} (Mode: {c['mode']}) ---
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
    print(f"=== 🧪 酸性試験 (V9: Market Switching & Expanded List) ({START_DATE} ~ {END_DATE}) ===")
    print(f"Logic: ADX<{MARKET_ADX_THRESHOLD} => Guerrilla (Target:Growth), ADX>={MARKET_ADX_THRESHOLD} => Homerun (Target:Core)")

    memory = MemorySystem(LOG_FILE)
    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"Model Init Error: {e}")
        return

    print("データ取得中...", end="")
    # 1. 指数データの取得
    nikkei = download_data_safe("^N225")
    if nikkei is not None:
        nikkei = calculate_technical_indicators(nikkei)
        print("日経平均データ取得完了")
    else:
        print("日経平均データ取得失敗。デフォルトでCOREリストを使用します。")

    # 2. 個別銘柄データの取得 (全銘柄)
    tickers_data = {}
    all_tickers = sorted(list(set(LIST_CORE + LIST_GROWTH))) 
    
    for t in all_tickers:
        df = download_data_safe(t)
        if df is not None:
            df = calculate_technical_indicators(df)
            tickers_data[t] = df
    print(f"\n完了 ({len(tickers_data)}銘柄)")

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

        # --- A. 環境認識 & リスト選択 ---
        market_adx = 0
        if nikkei is not None and current_date in nikkei.index:
            market_adx = nikkei.loc[current_date]['ADX']
        
        # ★ここがスイッチングの肝
        if market_adx >= MARKET_ADX_THRESHOLD:
            todays_mode = 'HOMERUN'
            target_list = LIST_CORE
            mode_icon = "🔥" # 戦時
        else:
            todays_mode = 'GUERRILLA'
            target_list = LIST_GROWTH
            mode_icon = "☁️" # 平時

        # --- B. ポートフォリオ管理 ---
        closed_tickers = []
        for ticker, pos in portfolio.items():
            df = tickers_data[ticker]
            if current_date not in df.index: continue

            day_data = df.loc[current_date]
            day_low = float(day_data['Low'])
            day_high = float(day_data['High'])
            day_open = float(day_data['Open'])
            
            pos_mode = pos.get('mode', 'HOMERUN') 
            
            # 1. 損切り判定
            current_sl = float(pos['sl_price'])
            if day_low <= current_sl:
                exec_price = current_sl
                if day_open < current_sl: exec_price = day_open
                proceeds = exec_price * pos['shares']
                cash += proceeds
                profit = proceeds - (pos['buy_price'] * pos['shares'])
                profit_rate = (exec_price - pos['buy_price']) / pos['buy_price'] * 100
                print(f"\n[{date_str}] 💀 損切({pos_mode}) {ticker}: {profit:+,.0f}円 ({profit_rate:+.2f}%)")
                trade_history.append({'Result': 'WIN' if profit>0 else 'LOSS', 'PL': profit})
                closed_tickers.append(ticker)
                continue

            # 2. 利確判定 (ゲリラのみ)
            if pos_mode == 'GUERRILLA':
                target_price = pos['buy_price'] * (1 + GUERRILLA_TARGET)
                if day_high >= target_price:
                    exec_price = target_price
                    if day_open > target_price: exec_price = day_open
                    proceeds = exec_price * pos['shares']
                    cash += proceeds
                    profit = proceeds - (pos['buy_price'] * pos['shares'])
                    profit_rate = (exec_price - pos['buy_price']) / pos['buy_price'] * 100
                    print(f"\n[{date_str}] 💰 利確({pos_mode}) {ticker}: {profit:+,.0f}円 ({profit_rate:+.2f}%)")
                    trade_history.append({'Result': 'WIN', 'PL': profit})
                    closed_tickers.append(ticker)
                    continue

            # 3. トレーリングストップ (ホームランのみ)
            if pos_mode == 'HOMERUN':
                if day_high > pos['max_price']: pos['max_price'] = day_high
                profit_pct_high = (pos['max_price'] - pos['buy_price']) / pos['buy_price']
                if profit_pct_high > HOMERUN_TRAIL_TRIGGER:
                    trail_dist = pos['atr'] * HOMERUN_TRAIL_WIDTH
                    new_sl = pos['max_price'] - trail_dist
                    if profit_pct_high > 0.15: new_sl = max(new_sl, pos['buy_price'] * 1.005)
                    if new_sl > pos['sl_price']: pos['sl_price'] = new_sl

        for t in closed_tickers: del portfolio[t]

        # --- C. 新規エントリー (対象リストのみスキャン) ---
        if len(portfolio) < MAX_POSITIONS and cash > 10000:
            candidates_data = []
            
            # その日のモードに合ったリストだけを見る
            for ticker in target_list:
                if ticker in portfolio: continue
                if ticker not in tickers_data: continue # データ取得失敗時
                
                df = tickers_data[ticker]
                if current_date not in df.index: continue
                idx = df.index.get_loc(current_date)
                if idx < 60: continue

                metrics = calculate_metrics_at_date(df, idx)
                iron_rule_check = check_iron_rules(metrics)
                if iron_rule_check: continue 

                if len(candidates_data) >= 5: break

                chart_bytes = create_chart_image_at_date(df, idx, ticker)
                if not chart_bytes: continue
                similar_text = memory.get_similar_cases_text(metrics)
                
                # モードを渡してAIに判断させる
                report = run_analyst(model, ticker, metrics, chart_bytes, similar_text, todays_mode)

                candidates_data.append({'ticker': ticker, 'metrics': metrics, 'report': report, 'mode': todays_mode})
                time.sleep(1) 

            if candidates_data:
                current_portfolio_text = ", ".join([t for t in portfolio.keys()]) or "なし"
                decision_data = run_commander_batch(model, candidates_data, cash, current_portfolio_text)

                for order in decision_data.get('orders', []):
                    tic = order.get('ticker')
                    try:
                        shares = int(order.get('shares', 0))
                    except: shares = 0

                    if shares > 0:
                        target = next((c for c in candidates_data if c['ticker'] == tic), None)
                        if target:
                            metrics = target['metrics']
                            cost = shares * metrics['price']
                            if cost <= cash:
                                cash -= cost
                                atr_val = metrics['atr_value']
                                
                                # モードに応じた初期損切り
                                if todays_mode == 'GUERRILLA':
                                    stop_mult = GUERRILLA_STOP # 1.5
                                else:
                                    stop_mult = HOMERUN_STOP_INIT # 1.8
                                
                                initial_sl = metrics['price'] - atr_val * stop_mult
                                
                                portfolio[tic] = {
                                    'buy_price': metrics['price'], 'shares': shares,
                                    'sl_price': initial_sl, 'max_price': metrics['price'], 'atr': atr_val,
                                    'mode': todays_mode
                                }
                                print(f"\n[{date_str}] {mode_icon} 新規({todays_mode}) {tic}: {shares}株")

        # --- D. 資産集計 ---
        current_equity = cash
        holdings_val = 0
        holdings_detail = []
        for t, pos in portfolio.items():
            if current_date in tickers_data[t].index:
                price = float(tickers_data[t].loc[current_date]['Close'])
                val = price * pos['shares']
                current_equity += val
                holdings_val += val
                holdings_detail.append(f"{t}({pos['mode'][0]})")

        print(f"\r[{date_str}] {mode_icon}資産:{current_equity:,.0f} (H:{len(portfolio)})", end="")
        equity_curve.append(current_equity)

        daily_history.append({
            "Date": date_str,
            "Total_Equity": int(current_equity),
            "Cash": int(cash),
            "Holdings_Value": int(holdings_val),
            "Positions_Count": len(portfolio),
            "Holdings_Detail": ", ".join(holdings_detail),
            "Market_Mode": todays_mode
        })

    # --- 終了処理 ---
    print("\n" + "="*50)
    if daily_history:
        df_history = pd.DataFrame(daily_history)
        df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
        print(f"📄 履歴保存: {HISTORY_CSV}")

    final_equity = equity_curve[-1] if equity_curve else INITIAL_CAPITAL
    profit = final_equity - INITIAL_CAPITAL
    print(f"最終資産: {final_equity:,.0f}円 ({profit:+,.0f}円)")

if __name__ == "__main__":
    main()