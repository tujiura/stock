import os
import io
import time
import json
import random
import datetime
import logging # ★追加: これが抜けていました
import pandas as pd
import numpy as np
import yfinance as yf

# GUIエラー回避設定
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import google.generativeai as genai
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import re
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
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")

# 設定
LOG_FILE = "ai_trade_memory_v14_fixed.csv"
MODEL_NAME = 'models/gemini-2.0-flash'

TRAINING_ROUNDS = 1000 
TIMEFRAME = "1d"
TRADE_BUDGET = 1000000 

# V14 Parameters
ADX_MIN = 20.0
ADX_MAX = 40.0
ROC_MAX = 15.0
ATR_MULTIPLIER = 2.5
VWAP_WINDOW = 20
WAIT_DAYS = 3

# 銘柄リスト
LIST_CORE = [
    "8035.T", "6857.T", "6146.T", "6920.T", "6758.T", "6702.T", "6501.T", "6503.T", "7751.T", 
    "4063.T", "6981.T", "6723.T", "7203.T", "7267.T", "6902.T", "6301.T", "6367.T", "7011.T", 
    "7013.T", "8306.T", "8316.T", "8411.T", "8766.T", "8058.T", "8001.T", "8031.T", "8002.T", 
    "9984.T", "9432.T", "9983.T", "4568.T", "4543.T", "4661.T", "7974.T", "6506.T", "5253.T", 
    "5032.T", "9166.T", "4385.T", "4478.T", "4483.T", "3993.T", "4180.T", "3687.T", "6027.T",
    "5595.T", "9348.T", "7012.T", "6203.T", "6254.T", "6315.T", "6526.T", "6228.T", "6963.T", 
    "3436.T", "7735.T", "6890.T", "2768.T", "7342.T", "2413.T", "2222.T", "7532.T", "3092.T",
    "9101.T", "9104.T", "9107.T", "1605.T", "5713.T", "5401.T", "5411.T"
]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY, transport="rest")

# ==========================================
# データ処理関数
# ==========================================
def download_data_safe(ticker, period="11y", interval="1d"): 
    try:
        # ロギング設定 (ここで logging が必要になります)
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if len(df) < 200: return None
        return df
    except Exception as e:
        # デバッグ用にエラーを表示したほうが原因がわかります
        # print(f"Download Error ({ticker}): {e}")
        return None

def calculate_market_filter_v14(market_df):
    try:
        df = market_df.copy()
        close = df['Close']
        df['SMA25'] = close.rolling(25).mean()
        df['SMA200'] = close.rolling(200).mean()
        
        # V14: 25日比較の傾き
        df['SMA200_Slope'] = df['SMA200'].diff(25)
        
        conditions = [
            (close > df['SMA200']),
            (close <= df['SMA200']) & (close > df['SMA25']),
            (close <= df['SMA200']) & (close <= df['SMA25'])
        ]
        choices = ['Bullish', 'Recovery', 'Bearish']
        df['Market_Regime'] = np.select(conditions, choices, default='Unknown')
        
        # 待機ルール
        df['Trade_Allowed'] = True
        wait_days = 0
        for i in range(1, len(df)):
            curr = df.index[i]; prev = df.index[i-1]
            if df.loc[curr, 'Market_Regime'] == 'Bullish' and df.loc[prev, 'Market_Regime'] != 'Bullish':
                if df.loc[curr, 'SMA200_Slope'] < 0:
                    wait_days = WAIT_DAYS
                else:
                    wait_days = 0
            
            if wait_days > 0:
                df.at[curr, 'Trade_Allowed'] = False
                wait_days -= 1
            else:
                df.at[curr, 'Trade_Allowed'] = True

        return df
    except: return None

def calculate_indicators_extended(df):
    try:
        df = df.copy()
        close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']
        
        df['SMA25'] = close.rolling(25).mean()
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()
        tr_smooth = tr.rolling(14).mean().replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr_smooth)
        df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df['ADX'] = df['ADX'].rolling(14).mean()
        
        df['ROC'] = close.pct_change(10) * 100
        
        # VWAP
        tp = (high + low + close) / 3
        df['VP'] = tp * vol
        cumulative_vp = df['VP'].rolling(window=VWAP_WINDOW).sum()
        cumulative_vol = vol.rolling(window=VWAP_WINDOW).sum().replace(0, np.nan)
        df['VWAP'] = cumulative_vp / cumulative_vol
        
        # VWAP Distance (%)
        df['VWAP_Dist'] = ((close - df['VWAP']) / df['VWAP']) * 100
        
        # Volume Change (%)
        df['Vol_Change'] = vol.pct_change() * 100
        
        # Cloud
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        df['Cloud_Top'] = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        
        return df
    except: return None

def calculate_full_metrics(df, idx, market_df):
    try:
        if idx < 60 or idx >= len(df): return None
        curr = df.iloc[idx]
        price = float(curr['Close'])
        date = df.index[idx]
        
        # 市場データ
        m_regime = "Unknown"
        sma_slope = 0.0
        trade_allowed = True
        
        if market_df is not None:
            if date in market_df.index:
                m_row = market_df.loc[date]
                m_regime = m_row['Market_Regime']
                sma_slope = float(m_row.get('SMA200_Slope', 0))
                trade_allowed = bool(m_row.get('Trade_Allowed', True))
            else:
                # 日付が見つからない場合は直近を探す
                try:
                    loc = market_df.index.get_indexer([date], method='pad')[0]
                    if loc != -1:
                        m_row = market_df.iloc[loc]
                        m_regime = m_row['Market_Regime']
                        sma_slope = float(m_row.get('SMA200_Slope', 0))
                        trade_allowed = bool(m_row.get('Trade_Allowed', True))
                except: pass
        
        if not trade_allowed: return None

        # スコア計算 (簡易版)
        score = 0
        if m_regime == 'Bullish': score = 100 if sma_slope > 0 else 70
        elif m_regime == 'Recovery': score = 50
        else: score = 0
        
        # サブレジーム (Slopeの状態)
        sub_regime = "Uptrend" if sma_slope > 0 else "Downtrend"

        return {
            'Date': date.strftime('%Y-%m-%d'),
            'Ticker': "", # 後で埋める
            'Timeframe': TIMEFRAME,
            'Price': price,
            'Vol': float(curr['Volume']),
            'ATR': float(curr.get('ATR', price*0.01)),
            'ADX': float(curr.get('ADX', 0)),
            'ROC_10': float(curr.get('ROC', 0)),
            'Market_Regime_Score': score,
            'VWAP_Distance_Percent': float(curr.get('VWAP_Dist', 0)),
            'Volume_Change_Percent': float(curr.get('Vol_Change', 0)),
            'Market_Regime': m_regime,
            'Sub_Regime': sub_regime,
            'Price_vs_Cloud': "Above" if price > curr.get('Cloud_Top', 0) else "Below"
        }
    except: return None

def check_iron_rules(metrics):
    if metrics['Market_Regime'] == 'Bearish': return "Market Bearish"
    if metrics['ROC_10'] > ROC_MAX: return "ROC Too High"
    if metrics['ADX'] > 50: return "ADX Overheat"
    if metrics['Price_vs_Cloud'] == "Below": return "Below Cloud"
    return None

class MemorySystem:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.headers = [
            "Date","Ticker","Timeframe","Action","Result","Reasoning","Confidence",
            "Entry_Price","Entry_Signal_Volume","Exit_Price","Exit_Signal_Price",
            "ADX","ROC_10","Market_Regime_Score","VWAP_Distance_Percent",
            "Volume_Change_Percent","Market_Regime","Sub_Regime","profit_rate"
        ]
        self.load_and_train()

    def load_and_train(self):
        self.df = pd.DataFrame(columns=self.headers)
        if os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path)
            except: pass
        
        # KNN用 (ROC, ADX, VWAP_Dist)
        self.feature_cols = ['ADX', 'ROC_10', 'VWAP_Distance_Percent']
        self.knn = None
        
        if len(self.df) > 5:
            # 必要なカラムがあるか確認して埋める
            for c in self.feature_cols:
                if c not in self.df.columns: self.df[c] = 0
            
            valid_df = self.df[self.df['Result'].isin(['WIN', 'LOSS'])].copy()
            if len(valid_df) > 5:
                features = valid_df[self.feature_cols].fillna(0)
                self.scaler = StandardScaler()
                features_norm = self.scaler.fit_transform(features)
                n = min(15, len(valid_df))
                self.knn = NearestNeighbors(n_neighbors=n).fit(features_norm)
                self.valid_df_knn = valid_df

    def save_experience(self, data_dict):
        # 指定順序で並べ替え
        row = {k: data_dict.get(k, "") for k in self.headers}
        df_row = pd.DataFrame([row])
        
        if not os.path.exists(self.csv_path):
            df_row.to_csv(self.csv_path, index=False)
        else:
            df_row.to_csv(self.csv_path, mode='a', header=False, index=False)

    def get_similar_cases_text(self, metrics):
        if self.knn is None: return "（データ不足）"
        try:
            vec = [[metrics.get(c, 0) for c in self.feature_cols]]
            vec_norm = self.scaler.transform(vec)
            dists, indices = self.knn.kneighbors(vec_norm)
            
            win = 0; loss = 0
            for idx in indices[0]:
                res = self.valid_df_knn.iloc[idx]['Result']
                if res == 'WIN': win += 1
                if res == 'LOSS': loss += 1
            rate = win / (win+loss) * 100 if (win+loss)>0 else 0
            return f"【過去類似局面】勝率: {rate:.0f}% (勝{win}/負{loss})"
        except: return "（検索エラー）"

def create_chart_image(df, name):
    try:
        data = df.tail(60).copy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.plot(data.index, data['Close'], color='black', label='Close')
        if 'VWAP' in data.columns:
            ax1.plot(data.index, data['VWAP'], color='orange', linestyle='--')
        ax1.set_title(f"{name}")
        ax2.bar(data.index, data['Volume'], color='gray')
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except: return None

def ai_decision_maker(model, chart, metrics, cbr):
    prompt = f"""
### Role
あなたはプロのトレーダーです。

### Market Data
Ticker: {metrics['Ticker']}
Price: {metrics['Price']:.0f}
Regime: {metrics['Market_Regime']} ({metrics['Sub_Regime']})
ADX: {metrics['ADX']:.1f}
ROC: {metrics['ROC_10']:.1f}%
VWAP Dist: {metrics['VWAP_Distance_Percent']:.1f}%

{cbr}

### Output (JSON)
{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "reason": "短い理由",
  "sl_multiplier": 2.5
}}
"""
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart}])
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(re.search(r'\{.*\}', text, re.DOTALL).group(0))
    except: return {"action": "HOLD", "confidence": 0}

def main():
    print("=== AI Training V14 Fixed Format ===")
    
    # 1. メモリシステム起動確認
    print("1. メモリシステム起動中...", end="")
    try:
        memory = MemorySystem(LOG_FILE)
        print("OK")
    except Exception as e:
        print(f"\n❌ メモリシステムエラー: {e}")
        return

    # 2. AIモデルの初期化確認
    print("2. Geminiモデル初期化中...", end="")
    try: 
        model = genai.GenerativeModel(MODEL_NAME)
        print("OK")
    except Exception as e: 
        print(f"\n❌ モデル初期化エラー: {e}")
        return

    # 3. データ取得の確認
    print("3. 市場データ取得中(^N225)...")
    nikkei = download_data_safe("^N225")
    if nikkei is None: 
        print("❌ 日経平均データの取得に失敗しました。")
        print("👉 yfinanceのバージョン、または通信環境を確認してください。")
        return
    else:
        print(f"   -> 取得成功: {len(nikkei)}件")

    # 4. 指標計算の確認
    print("4. 市場フィルター計算中...", end="")
    market_df = calculate_market_filter_v14(nikkei)
    if market_df is None:
        print("\n❌ 市場フィルターの計算に失敗しました。")
        return
    print("OK")

    count = 0
    print(f"5. トレーニングループ開始 (目標: {TRAINING_ROUNDS}回)")
    
    while count < TRAINING_ROUNDS:
        try:
            ticker = random.choice(LIST_CORE)
            df = download_data_safe(ticker)
            if df is None: continue
            
            df = calculate_indicators_extended(df)
            if df is None or len(df) < 100: continue
            
            idx = random.randint(100, len(df)-25)
            metrics = calculate_full_metrics(df, idx, market_df)
            
            if metrics is None: continue 
            
            metrics['Ticker'] = ticker
            if check_iron_rules(metrics): continue

            # AI判定
            cbr = memory.get_similar_cases_text(metrics)
            chart = create_chart_image(df.iloc[:idx+1], ticker)
            
            decision = ai_decision_maker(model, chart, metrics, cbr)
            
            if decision.get('action') != 'BUY': continue
            
            # シミュレーション実行
            count += 1
            entry_price = metrics['Price']
            atr = metrics['ATR']
            sl_multiplier = float(decision.get('sl_multiplier', 2.5))
            sl = entry_price - (atr * sl_multiplier)
            
            future = df.iloc[idx+1:idx+21]
            exit_price = entry_price
            result = "LOSS"
            
            for i in range(len(future)):
                row = future.iloc[i]
                if row['Low'] <= sl:
                    exit_price = sl
                    result = "LOSS"
                    break
                if row['High'] >= entry_price + (atr * 3.0):
                    exit_price = entry_price + (atr * 3.0)
                    result = "WIN"
                    break
                if i == len(future)-1:
                    exit_price = row['Close']
                    result = "WIN" if exit_price > entry_price else "LOSS"

            # 損益計算
            shares = int((TRADE_BUDGET * 0.2) / entry_price)
            if shares == 0: shares = 1
            pl_amount = (exit_price - entry_price) * shares
            pl_rate = ((exit_price - entry_price) / entry_price) * 100
            
            # ログ表示
            log_color = "🔴" if result == "WIN" else "🔵"
            print(f"{log_color} [{result}] {ticker} : {pl_amount:+,.0f}円 ({pl_rate:+.2f}%) | Reason: {decision.get('reason','-')}")

            # データ保存
            save_data = {
                "Date": metrics['Date'],
                "Ticker": ticker,
                "Timeframe": TIMEFRAME,
                "Action": "BUY",
                "Result": result,
                "Reasoning": decision.get('reason', 'None'),
                "Confidence": decision.get('confidence', 0),
                "Entry_Price": entry_price,
                "Entry_Signal_Volume": metrics['Vol'],
                "Exit_Price": exit_price,
                "Exit_Signal_Price": exit_price,
                "ADX": metrics['ADX'],
                "ROC_10": metrics['ROC_10'],
                "Market_Regime_Score": metrics['Market_Regime_Score'],
                "VWAP_Distance_Percent": metrics['VWAP_Distance_Percent'],
                "Volume_Change_Percent": metrics['Volume_Change_Percent'],
                "Market_Regime": metrics['Market_Regime'],
                "Sub_Regime": metrics['Sub_Regime'],
                "profit_rate": pl_rate
            }
            memory.save_experience(save_data)
            time.sleep(1)

        except KeyboardInterrupt:
            print("\nユーザーによる中断")
            break
        except Exception as e:
            # print(f"ループ内エラー（スキップします）: {e}")
            continue

    print("\nトレーニング完了")

if __name__ == "__main__":
    main()