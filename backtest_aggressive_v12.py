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
from scipy.signal import argrelextrema
import warnings

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

# 設定
LOG_FILE = "ai_trade_memory_aggressive_v12.csv" 
HISTORY_CSV = "backtest_history_v12.csv"
MODEL_NAME = 'models/gemini-2.0-flash'

START_DATE = "2020-01-01"
END_DATE   = "2022-12-30"

INITIAL_CAPITAL = 100000 
RISK_PER_TRADE = 0.40      
MAX_POSITIONS = 5         
MAX_INVEST_RATIO = 1    

# V12 Parameters
ADX_MIN = 20.0
ADX_MAX = 40.0
ROC_MAX = 15.0
ATR_MULTIPLIER = 2.5
VWAP_WINDOW = 20

# 銘柄リスト
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
def download_data_safe(ticker, period="7y", interval="1d", retries=3): 
    wait = 1
    for attempt in range(retries):
        try:
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                time.sleep(wait); wait *= 2
                continue
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
            if len(df) < 200: return None
            return df
        except Exception:
            time.sleep(wait); wait *= 2
    return None

def calculate_market_filter(market_df):
    try:
        df = market_df.copy()
        close = df['Close']
        df['SMA25'] = close.rolling(25).mean()
        df['SMA200'] = close.rolling(200).mean()
        
        conditions = [
            (close > df['SMA200']),
            (close <= df['SMA200']) & (close > df['SMA25']),
            (close <= df['SMA200']) & (close <= df['SMA25'])
        ]
        choices = ['Bullish', 'Recovery', 'Bearish']
        df['Market_Regime'] = np.select(conditions, choices, default='Unknown')
        return df['Market_Regime']
    except: return None

def calculate_technical_indicators_v12(df):
    try:
        df = df.copy()
        close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']
        df['SMA25'] = close.rolling(25).mean()
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
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
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        df['ROC'] = close.pct_change(10) * 100
        tp = (high + low + close) / 3
        df['VP'] = tp * vol
        cumulative_vp = df['VP'].rolling(window=VWAP_WINDOW).sum()
        cumulative_vol = vol.rolling(window=VWAP_WINDOW).sum().replace(0, np.nan)
        df['VWAP'] = cumulative_vp / cumulative_vol
        df['VWAP_Dev'] = np.where(df['VWAP'].notna(), ((close - df['VWAP']) / df['VWAP']) * 100, 0)
        money_flow = tp * vol
        positive_flow = money_flow.where(tp > tp.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(tp < tp.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow.replace(0, np.nan)
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        df['MFI'] = df['MFI'].fillna(50)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        df['Cloud_Top'] = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        return df.dropna()
    except Exception: return None

def calculate_metrics_v12(df, idx, market_regime_series=None):
    try:
        if idx < 60 or idx >= len(df): return None
        curr = df.iloc[idx]
        price = float(curr['Close'])
        
        market_regime = "Unknown"
        if market_regime_series is not None:
            target_date = df.index[idx]
            try:
                if target_date in market_regime_series.index:
                    market_regime = market_regime_series.loc[target_date]
                else:
                    prev_loc = market_regime_series.index.get_indexer([target_date], method='pad')[0]
                    if prev_loc != -1: market_regime = market_regime_series.iloc[prev_loc]
            except: pass

        adx = float(curr.get('ADX', 20.0))
        roc = float(curr.get('ROC', 0.0))
        mfi = float(curr.get('MFI', 50.0))
        
        if ADX_MIN <= adx <= ADX_MAX: regime = "Trend Start/Growth"
        elif adx > ADX_MAX: regime = "Overheated Trend"
        else: regime = "Range/Weak"

        recent_high = df['High'].iloc[idx-60:idx].max()
        dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
        ma_deviation = ((price / float(curr['SMA25'])) - 1) * 100
        
        macd_hist = float(curr.get('MACD_Hist', 0.0))
        prev_hist = float(df['MACD_Hist'].iloc[idx-1]) if idx > 0 else 0.0
        
        cloud_top = float(curr.get('Cloud_Top', price))
        price_vs_cloud = "Above" if price > cloud_top else "Below"

        return {
            'date': df.index[idx].strftime('%Y-%m-%d'),
            'price': price,
            'dist_to_res': dist_to_res,
            'ma_deviation': ma_deviation,
            'adx': adx,
            'roc': roc,
            'mfi': mfi,
            'atr_value': float(curr.get('ATR', price*0.01)),
            'macd_hist': macd_hist,
            'macd_trend': "Expanding" if abs(macd_hist) > abs(prev_hist) else "Shrinking",
            'price_vs_cloud': price_vs_cloud,
            'rsi': float(curr.get('RSI', 50.0)),
            'regime': regime,
            'vwap_dev': float(curr.get('VWAP_Dev', 0.0)),
            'market_regime': market_regime
        }
    except Exception: return None

def check_iron_rules_v12(metrics):
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
                    global CBR_NEIGHBORS_COUNT
                    self.knn = NearestNeighbors(n_neighbors=min(CBR_NEIGHBORS_COUNT, len(valid_df)), metric='euclidean')
                    self.knn.fit(self.features_normalized)
        except Exception: pass

    def get_similar_cases_text(self, current_metrics):
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

def create_chart_image(df, name):
    try:
        data = df.tail(80).copy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.plot(data.index, data['Close'], color='black', label='Close')
        if 'VWAP' in data.columns:
            ax1.plot(data.index, data['VWAP'], color='orange', alpha=0.7, linestyle='--', label='VWAP')
        if 'Cloud_Top' in data.columns:
            ax1.fill_between(data.index, data['Cloud_Top'], data['Close'].min(), color='blue', alpha=0.05)
        ax1.set_title(f"{name} V12 Chart")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception: return None

def ai_decision_maker_v12(model, chart_bytes, metrics, cbr_text, ticker):
    prompt = f"""
### Role
あなたは「トレンド初動ハンター」です。

### Input Data
銘柄: {ticker} (現在値: {metrics['price']:.0f}円)

[Market Data]
- ADX: {metrics['adx']:.1f}
- ROC(10): {metrics['roc']:.1f}%
- Market Regime: {metrics['market_regime']}

{cbr_text}

### Output Requirement (JSON ONLY)
{{
  "thought_process": "...",
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "sl_multiplier": 2.5,
  "tp_multiplier": 5.0,
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
--- 候補: {c['ticker']} (Regime: {c['metrics']['regime']}) ---
現在値: {c['metrics']['price']:.0f}円
推奨最大株数: {final_max_shares}株
【分析官報告】
{c['report']}
-------------------------
"""

    prompt = f"""
あなたは運用指令官です。

### 現在の状況
- 手元資金: {current_cash:,.0f}円
- 保有銘柄: {current_portfolio_text}

### 候補銘柄
{candidates_text}

### 任務
JSON形式で注文を出力してください。

出力フォーマット:
{{
  "orders": [
    {{
      "ticker": "銘柄コード",
      "action": "BUY",
      "shares": 購入株数 (整数),
      "stop_loss": 損切り価格 (数値のみ),
      "reason": "理由"
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

def main():
    print(f"=== 🧪 酸性試験 (V12: Early Trend Hunter + Market Filter) ({START_DATE} ~ {END_DATE}) ===")
    
    memory = MemorySystem(LOG_FILE)
    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"Model Init Error: {e}")
        return

    print("データ取得中...", end="")
    
    # 1. 市場データ取得
    nikkei = download_data_safe("^N225")
    market_regime_series = None
    if nikkei is not None:
        market_regime_series = calculate_market_filter(nikkei)
        print("Market data OK.")

    tickers_data = {}
    for i, t in enumerate(TRAINING_LIST):
        df = download_data_safe(t)
        if df is not None:
            df = calculate_technical_indicators_v12(df)
            if df is not None:
                tickers_data[t] = df
                print(".", end="", flush=True)
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

        # --- A. ポートフォリオ管理 ---
        closed_tickers = []
        for ticker, pos in portfolio.items():
            df = tickers_data[ticker]
            if current_date not in df.index: continue

            day_data = df.loc[current_date]
            day_low = float(day_data['Low'])
            day_high = float(day_data['High'])
            
            # 損切り
            current_sl = float(pos['sl_price'])
            if day_low <= current_sl:
                exec_price = current_sl
                proceeds = exec_price * pos['shares']
                cash += proceeds
                profit = proceeds - (pos['buy_price'] * pos['shares'])
                profit_rate = (exec_price - pos['buy_price']) / pos['buy_price'] * 100
                print(f"\n[{date_str}] 💀 損切 {ticker}: {profit:+,.0f}円 ({profit_rate:+.2f}%)")
                trade_history.append({'Result': 'LOSS', 'PL': profit})
                closed_tickers.append(ticker)
                continue

            # トレーリング
            if day_high > pos['max_price']:
                pos['max_price'] = day_high
                new_sl = pos['max_price'] - (pos['atr'] * ATR_MULTIPLIER)
                if new_sl > current_sl:
                    pos['sl_price'] = new_sl
            
            # 利確
            if pos.get('tp_price', 0) > 0 and day_high >= pos['tp_price']:
                exec_price = pos['tp_price']
                proceeds = exec_price * pos['shares']
                cash += proceeds
                profit = proceeds - (pos['buy_price'] * pos['shares'])
                profit_rate = (exec_price - pos['buy_price']) / pos['buy_price'] * 100
                print(f"\n[{date_str}] 💰 利確 {ticker}: {profit:+,.0f}円 ({profit_rate:+.2f}%)")
                trade_history.append({'Result': 'WIN', 'PL': profit})
                closed_tickers.append(ticker)
                continue

        for t in closed_tickers: del portfolio[t]

        # --- B. 新規エントリー ---
        if len(portfolio) < MAX_POSITIONS and cash > 10000:
            candidates_data = []
            
            for ticker in tickers_data.keys():
                if ticker in portfolio: continue
                df = tickers_data[ticker]
                if current_date not in df.index: continue
                idx = df.index.get_loc(current_date)
                if idx < 60: continue

                metrics = calculate_metrics_v12(df, idx, market_regime_series)
                if metrics is None: continue

                iron_rule_check = check_iron_rules_v12(metrics)
                if iron_rule_check: continue 

                if len(candidates_data) >= 5: break

                chart_bytes = create_chart_image(df, ticker)
                if not chart_bytes: continue
                similar_text = memory.get_similar_cases_text(metrics)
                
                decision = ai_decision_maker_v12(model, chart_bytes, metrics, similar_text, ticker)
                
                report = f"Action: {decision.get('action')}, Conf: {decision.get('confidence')}%, Reason: {decision.get('reason')}"
                decision['metrics'] = metrics 
                
                candidates_data.append({'ticker': ticker, 'metrics': metrics, 'report': report, 'ai_decision': decision})
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
                            ai_dec = target['ai_decision']
                            cost = shares * metrics['price']
                            
                            if cost <= cash:
                                cash -= cost
                                atr_val = metrics['atr_value']
                                
                                sl_mult = float(ai_dec.get('sl_multiplier', ATR_MULTIPLIER))
                                tp_mult = float(ai_dec.get('tp_multiplier', 5.0))
                                
                                initial_sl = metrics['price'] - (atr_val * sl_mult)
                                initial_tp = metrics['price'] + (atr_val * tp_mult)
                                
                                portfolio[tic] = {
                                    'buy_price': metrics['price'], 'shares': shares,
                                    'sl_price': initial_sl, 'tp_price': initial_tp,
                                    'max_price': metrics['price'], 'atr': atr_val
                                }
                                print(f"\n[{date_str}] 🔴 新規 {tic}: {shares}株 (SL:{initial_sl:.0f})")

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
                holdings_detail.append(f"{t}")

        print(f"\r[{date_str}] 資産:{current_equity:,.0f} (H:{len(portfolio)})", end="")
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
    if daily_history:
        df_history = pd.DataFrame(daily_history)
        df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
        print(f"📄 履歴保存: {HISTORY_CSV}")

    final_equity = equity_curve[-1] if equity_curve else INITIAL_CAPITAL
    profit = final_equity - INITIAL_CAPITAL
    print(f"最終資産: {final_equity:,.0f}円 ({profit:+,.0f}円)")

if __name__ == "__main__":
    main()