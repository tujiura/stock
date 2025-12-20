import os
import io
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
import re
import logging
import socket
import requests.packages.urllib3.util.connection as urllib3_cn
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
USE_LLM = False
if GOOGLE_API_KEY:
    USE_LLM = True
    print("✅ Gemini APIモードで実行します (AI司令官起動)")
    genai.configure(api_key=GOOGLE_API_KEY, transport="rest")
    MODEL_NAME = 'models/gemini-2.0-flash'
else:
    print("⚠️ GOOGLE_API_KEYが見つかりません。ルールベースモードで実行します。")

# 設定
HISTORY_FILE = "backtest_history_v14_progress.csv"
LOG_FILE = "ai_trade_memory_v14_refined.csv"

# バックテスト期間
START_DATE = "2015-01-01"
END_DATE = "2018-12-31"

# 資金管理
INITIAL_CAPITAL = 100000
MAX_POSITIONS = 5
RISK_PER_TRADE = 0.90

# V14 Parameters
ADX_MIN = 20.0
ADX_MAX = 40.0
ROC_MAX = 15.0
ATR_MULTIPLIER = 2.5
VWAP_WINDOW = 20
WAIT_DAYS = 3

# 銘柄リスト
WATCH_LIST = [
    "8035.T", "6857.T", "6146.T", "6920.T", "6758.T", "6702.T", "6501.T", "6503.T", "7751.T", 
    "4063.T", "6981.T", "6723.T", "7203.T", "7267.T", "6902.T", "6301.T", "6367.T", "7011.T", 
    "7013.T", "8306.T", "8316.T", "8411.T", "8766.T", "8058.T", "8001.T", "8031.T", "8002.T", 
    "9984.T", "9432.T", "9983.T", "4568.T", "4543.T", "4661.T", "7974.T", "6506.T", "5253.T", 
    "5032.T", "9166.T", "4385.T", "4478.T", "4483.T", "3993.T", "4180.T", "3687.T", "6027.T",
    "5595.T", "9348.T", "7012.T", "6203.T", "6254.T", "6315.T", "6526.T", "6228.T", "6963.T", 
    "3436.T", "7735.T", "6890.T", "2768.T", "7342.T", "2413.T", "2222.T", "7532.T", "3092.T",
    "9101.T", "9104.T", "9107.T", "1605.T", "5713.T", "5401.T", "5411.T"
]
WATCH_LIST = sorted(list(set(WATCH_LIST)))

# ==========================================
# データ処理
# ==========================================
def download_data_safe(ticker, start, end): 
    try:
        s_date = pd.to_datetime(start) - pd.Timedelta(days=365)
        e_date = pd.to_datetime(end) + pd.Timedelta(days=10)
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        
        df = yf.download(ticker, start=s_date, end=e_date, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if len(df) < 200: return None
        return df
    except: return None

def calculate_market_filter(market_df):
    try:
        df = market_df.copy()
        close = df['Close']
        df['SMA25'] = close.rolling(25).mean()
        df['SMA200'] = close.rolling(200).mean()
        df['SMA200_Slope'] = df['SMA200'].diff(25) # V14
        
        conditions = [
            (close > df['SMA200']),
            (close <= df['SMA200']) & (close > df['SMA25']),
            (close <= df['SMA200']) & (close <= df['SMA25'])
        ]
        choices = ['Bullish', 'Recovery', 'Bearish']
        df['Market_Regime'] = np.select(conditions, choices, default='Unknown')
        
        high = df['High']; low = df['Low']
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()
        tr_smooth = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr_smooth)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df['Market_ADX'] = dx.rolling(14).mean()

        df['Days_Since_Change'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['Market_Regime'] != df.iloc[i-1]['Market_Regime']:
                df.iat[i, df.columns.get_loc('Days_Since_Change')] = 0
            else:
                df.iat[i, df.columns.get_loc('Days_Since_Change')] = df.iloc[i-1]['Days_Since_Change'] + 1
        return df
    except: return None

def calculate_technical_indicators_v12(df):
    try:
        df = df.copy()
        close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']
        
        df['SMA25'] = close.rolling(25).mean()
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()
        tr_smooth = tr.rolling(14).mean().replace(0, np.nan)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / tr_smooth)
        df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df['ADX'] = df['ADX'].rolling(14).mean()
        
        df['ROC'] = close.pct_change(10) * 100
        
        tp = (high + low + close) / 3
        df['VP'] = tp * vol
        cumulative_vp = df['VP'].rolling(window=VWAP_WINDOW).sum()
        cumulative_vol = vol.rolling(window=VWAP_WINDOW).sum().replace(0, np.nan)
        df['VWAP'] = cumulative_vp / cumulative_vol
        df['VWAP_Dev'] = np.where(df['VWAP'].notna(), ((close - df['VWAP']) / df['VWAP']) * 100, 0)
        
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        df['Cloud_Top'] = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        return df
    except Exception: return None

def calculate_metrics_v14(df, idx, market_regime_val):
    try:
        if idx < 60 or idx >= len(df): return None
        curr = df.iloc[idx]
        price = float(curr['Close'])
        
        adx = float(curr.get('ADX', 20.0))
        roc = float(curr.get('ROC', 0.0))
        
        if ADX_MIN <= adx <= ADX_MAX: regime = "Trend Start/Growth"
        elif adx > ADX_MAX: regime = "Overheated Trend"
        else: regime = "Range/Weak"

        recent_high = df['High'].iloc[idx-60:idx].max()
        dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
        
        cloud_top = float(curr.get('Cloud_Top', price))
        price_vs_cloud = "Above" if price > cloud_top else "Below"

        return {
            'date': df.index[idx].strftime('%Y-%m-%d'),
            'price': price,
            'dist_to_res': dist_to_res,
            'adx': adx,
            'roc': roc,
            'atr_value': float(curr.get('ATR', price*0.01)),
            'price_vs_cloud': price_vs_cloud,
            'regime': regime,
            'market_regime': market_regime_val
        }
    except Exception: return None

def check_iron_rules_v14(metrics):
    if metrics['market_regime'] == 'Bearish': return "Market Bearish"
    if metrics['roc'] > ROC_MAX: return f"ROC Too High ({metrics['roc']:.1f}%)"
    if metrics['adx'] > 50: return "ADX Overheat (>50)"
    if metrics['price_vs_cloud'] == "Below": return "Below Cloud"
    return None

class MemorySystem:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        self.feature_cols = ['adx', 'roc', 'mfi', 'vwap_dev', 'rsi']
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
                    n_neighbors = min(15, len(valid_df))
                    self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
                    self.knn.fit(self.features_normalized)
        except Exception: pass

    def get_similar_cases_text(self, current_metrics):
        if self.knn is None: return "（データ不足）"
        try:
            vec = [current_metrics.get(col, 0) for col in self.feature_cols]
            input_df = pd.DataFrame([vec], columns=self.feature_cols)
            dists, indices = self.knn.kneighbors(self.scaler.transform(input_df))
            
            win_c = 0; loss_c = 0
            for idx in indices[0]:
                row = self.valid_df_for_knn.iloc[idx]
                res = str(row.get('result', ''))
                if res == 'WIN': win_c += 1
                if res == 'LOSS': loss_c += 1
            rate = win_c / (win_c + loss_c) * 100 if (win_c + loss_c) > 0 else 0
            return f"【類似局面(過去)】-> 勝率: {rate:.0f}% (勝{win_c}/負{loss_c})\n"
        except: return "（検索エラー）"

def create_chart_image(df, name):
    try:
        data = df.tail(80).copy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.plot(data.index, data['Close'], color='black', label='Close')
        if 'VWAP' in data.columns:
            ax1.plot(data.index, data['VWAP'], color='orange', alpha=0.7, linestyle='--', label='VWAP')
        if 'Cloud_Top' in data.columns:
            ax1.fill_between(data.index, data['Cloud_Top'], data['Close'].min(), color='blue', alpha=0.05)
        ax1.set_title(f"{name} Chart")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception: return None

def ai_market_commander(model, chart_bytes, metrics):
    if model is None:
        if metrics['market_regime'] == 'Bullish' and metrics['sma200_slope'] < 0:
            if metrics['days_since_regime_change'] <= WAIT_DAYS:
                return {"directive": "WAIT", "reason": "Rule: 200MA Down Trend (Wait)"}
        return {"directive": "GO", "reason": "Rule: Safe"}

    prompt = f"""
### Role
あなたは「AI運用司令官 (Market Commander)」です。

### Market Data (Nikkei 225)
- Regime: {metrics['market_regime']}
- 200MA Slope (25-day diff): {metrics['sma200_slope']:.2f} (正=上昇トレンド/負=下落トレンド)
- Market ADX: {metrics['market_adx']:.1f}
- Days since Change: {metrics['days_since_regime_change']} days

### Strategy Doctrine (V14)
1. **騙し回避**: 200MA Slopeが「負」でRegimeがBullish転換した場合、{WAIT_DAYS}日間は **"WAIT"**。
2. **順張り**: 200MA Slopeが「正」なら **"GO"**。
3. **乱高下警戒**: Market ADXが低い場合は慎重に。

### Output Requirement (JSON ONLY)
{{
  "thought_process": "...",
  "directive": "GO" or "WAIT" or "DEFEND",
  "reason": "短い指示理由"
}}
"""
    safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}], safety_settings=safety)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: return {"directive": "WAIT", "reason": "Commander Error"}

def ai_decision_maker_v14(model, chart_bytes, metrics, cbr_text, ticker, commander_instruction):
    if model is None:
        if commander_instruction['directive'] != 'GO':
            return {"action": "HOLD", "reason": f"Commander: {commander_instruction['directive']}"}
        return {"action": "BUY", "confidence": 80, "sl_multiplier": 2.5, "reason": "Rule Based"}

    prompt = f"""
### Role
あなたは「トレンド初動ハンター」です。司令官の指示に従ってください。

### Commander's Order
**Directive**: {commander_instruction['directive']}
**Reason**: {commander_instruction['reason']}
(※ WAIT/DEFENDならエントリー禁止)

### Input Data
銘柄: {ticker} (Price: {metrics['price']:.0f})
[Market Data]
- ADX: {metrics['adx']:.1f}
- ROC: {metrics['roc']:.1f}%

{cbr_text}

### Output Requirement (JSON ONLY)
{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "sl_multiplier": 2.5,
  "reason": "理由"
}}
"""
    safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}], safety_settings=safety)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: return {"action": "HOLD", "reason": "Error", "confidence": 0}

class Portfolio:
    def __init__(self, initial_capital):
        self.cash = initial_capital
        self.positions = {} 
        self.history = []
        self.total_equity = initial_capital

    def update_equity(self, date, current_prices):
        holdings_val = 0
        detail_str = []
        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, pos['entry_price'])
            val = pos['shares'] * price
            holdings_val += val
            detail_str.append(f"{ticker}:{val:.0f}")
            
        self.total_equity = self.cash + holdings_val
        self.history.append({
            'Date': date,
            'Total_Equity': self.total_equity,
            'Cash': self.cash,
            'Holdings_Value': holdings_val,
            'Positions_Count': len(self.positions),
            'Holdings_Detail': ";".join(detail_str)
        })
        # ★ ログ修正: 行頭に戻って上書き (前の文字を消すパディングを追加)
        print(f"\r[{date.date()}] 資産: {self.total_equity:,.0f}円 (Pos: {len(self.positions)})" + " "*30)

    def check_exit(self, date, ticker, row):
        if ticker not in self.positions: return
        pos = self.positions[ticker]
        low = row['Low']; high = row['High']; atr = row['ATR']
        
        if low <= pos['sl_price']:
            exit_price = pos['sl_price']
            amount = exit_price * pos['shares']
            self.cash += amount
            pnl = amount - (pos['entry_price'] * pos['shares'])
            print(f"\n[{date.date()}] {ticker} SELL (SL) PnL: {pnl:+.0f}") # 改行を入れる
            del self.positions[ticker]
            return

        new_sl = high - (atr * ATR_MULTIPLIER)
        if new_sl > pos['sl_price']:
            pos['sl_price'] = new_sl

    def enter_position(self, date, ticker, row, decision):
        if len(self.positions) >= MAX_POSITIONS: return
        if self.cash < (self.total_equity * 0.1): return 
        
        price = row['Close']
        budget = self.total_equity * RISK_PER_TRADE
        shares = int(budget // price)
        
        if shares > 0:
            cost = shares * price
            if self.cash >= cost:
                self.cash -= cost
                atr = row['ATR']
                sl_mult = float(decision.get('sl_multiplier', ATR_MULTIPLIER))
                sl_price = price - (atr * sl_mult)
                
                self.positions[ticker] = {
                    'shares': shares,
                    'entry_price': price,
                    'sl_price': sl_price,
                    'entry_date': date
                }
                print(f"\n[{date.date()}] {ticker} BUY  @ {price:.0f} (Cmd: {decision.get('reason','')})") # 改行を入れる

def main():
    print(f"=== V14 Backtest (Opt 1&3 + Progress) ===")
    print(f"期間: {START_DATE} 〜 {END_DATE}")
    
    print("データ取得中...")
    nikkei = download_data_safe("^N225", START_DATE, END_DATE)
    if nikkei is None: return
    nikkei = calculate_market_filter(nikkei)
    
    market_data = {}
    for t in WATCH_LIST:
        df = download_data_safe(t, START_DATE, END_DATE)
        if df is not None:
            df = calculate_technical_indicators_v12(df)
            if df is not None:
                market_data[t] = df
                print(".", end="", flush=True)
    print(f"\nデータ取得完了: {len(market_data)}銘柄")
    
    portfolio = Portfolio(INITIAL_CAPITAL)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    
    memory = MemorySystem(LOG_FILE)
    model = None
    if USE_LLM:
        try: model = genai.GenerativeModel(MODEL_NAME)
        except: pass

    prev_regime = "Unknown"
    regime_start_date = pd.Timestamp.min

    for date in dates:
        # 1. 司令官の判断
        commander_instruction = {"directive": "WAIT", "reason": "No Data"}
        try:
            if date in nikkei.index:
                m_row = nikkei.loc[date]
                m_idx = nikkei.index.get_loc(date)
            else:
                idx = nikkei.index.get_indexer([date], method='pad')[0]
                if idx == -1: continue
                m_row = nikkei.iloc[idx]
                m_idx = idx
            
            market_regime = m_row['Market_Regime']
            sma200_slope = m_row.get('SMA200_Slope', 0.0)
            market_adx = m_row.get('Market_ADX', 0.0)
            days_since = int(m_row.get('Days_Since_Change', 0))
            
            market_metrics = {
                'market_regime': market_regime,
                'sma200_slope': sma200_slope,
                'market_adx': market_adx,
                'days_since_regime_change': days_since
            }
            
            nk_past = nikkei.iloc[:m_idx+1]
            nk_chart = create_chart_image(nk_past, "^N225")
            commander_instruction = ai_market_commander(model, nk_chart, market_metrics)
            
        except Exception: pass

        # 2. 個別銘柄の処理
        current_prices = {}
        total_stocks = len(market_data)

        # ★ 修正: enumerateで進捗表示
        for i, (ticker, df) in enumerate(market_data.items(), 1):
            # 進捗ログ (上書き)
            print(f"\r[{date.date()}] 🔍 Check: {ticker} ({i}/{total_stocks}) {' '*5}", end="", flush=True)

            if date not in df.index: continue
            row = df.loc[date]
            current_prices[ticker] = row['Close']
            
            portfolio.check_exit(date, ticker, row)
            
            # 高速化1
            if len(portfolio.positions) >= MAX_POSITIONS: continue
            # 高速化3
            if commander_instruction['directive'] != 'GO': continue

            if ticker not in portfolio.positions:
                idx = df.index.get_loc(date)
                if idx < 60: continue

                metrics = calculate_metrics_v14(df, idx, market_regime)
                if metrics is None: continue
                
                iron_rule = check_iron_rules_v14(metrics)
                if iron_rule: continue
                
                cbr_text = memory.get_similar_cases_text(metrics)
                past_df = df.iloc[:idx+1]
                chart_bytes = create_chart_image(past_df, ticker)
                
                decision = ai_decision_maker_v14(model, chart_bytes, metrics, cbr_text, ticker, commander_instruction)
                
                if decision.get('action') == 'BUY':
                     portfolio.enter_position(date, ticker, row, decision)
                
                if USE_LLM: time.sleep(0.5)

        portfolio.update_equity(date, current_prices)

    df_res = pd.DataFrame(portfolio.history)
    df_res.to_csv(HISTORY_FILE, index=False)
    
    print(f"\n=== バックテスト終了 ===")
    print(f"最終資産: {portfolio.total_equity:,.0f}円")
    
if __name__ == "__main__":
    main()