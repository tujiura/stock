import os
import io
import time
import json
import random
import datetime
import logging
import pandas as pd
import numpy as np
import yfinance as yf

# GUIエラー回避
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
LOG_FILE = "ai_trade_memory_v15_aggressive.csv" # V15用ファイル
MODEL_NAME = 'models/gemini-2.0-flash'

TRAINING_ROUNDS = 8000 
TIMEFRAME = "1d"
TRADE_BUDGET = 1000000 

# ★ V15 Aggressive Parameters (リミッター解除)
ADX_MIN = 15.0  # 基準を下げる（少しのトレンドでも反応）
ADX_MAX = 75.0  # ★上限大幅引き上げ（過熱相場も許容）
ROC_MAX = 100.0 # ★実質上限なし（急騰銘柄に飛び乗る）
ATR_MULTIPLIER = 2.0 # 損切りは浅めに（回転率重視）
VWAP_WINDOW = 20

# 銘柄リスト (Core)
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
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if len(df) < 200: return None
        return df
    except: return None

def calculate_market_filter_v15(market_df):
    try:
        df = market_df.copy()
        close = df['Close']
        df['SMA25'] = close.rolling(25).mean()
        df['SMA200'] = close.rolling(200).mean()
        df['SMA200_Slope'] = df['SMA200'].diff(25)
        
        conditions = [
            (close > df['SMA200']),
            (close <= df['SMA200']) & (close > df['SMA25']),
            (close <= df['SMA200']) & (close <= df['SMA25'])
        ]
        choices = ['Bullish', 'Recovery', 'Bearish']
        df['Market_Regime'] = np.select(conditions, choices, default='Unknown')
        
        # ★ V15: 待機ルール完全撤廃
        # どんな相場環境でも、個別銘柄に勢いがあればGOサインを出すため、
        # 市場全体の「禁止フラグ」は立てない。
        df['Trade_Allowed'] = True 

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
        
        tp = (high + low + close) / 3
        df['VP'] = tp * vol
        cumulative_vp = df['VP'].rolling(window=VWAP_WINDOW).sum()
        cumulative_vol = vol.rolling(window=VWAP_WINDOW).sum().replace(0, np.nan)
        df['VWAP'] = cumulative_vp / cumulative_vol
        df['VWAP_Dist'] = ((close - df['VWAP']) / df['VWAP']) * 100
        df['Vol_Change'] = vol.pct_change() * 100
        
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
        
        m_regime = "Unknown"
        sma_slope = 0.0
        
        if market_df is not None:
            if date in market_df.index:
                m_row = market_df.loc[date]
                m_regime = m_row['Market_Regime']
                sma_slope = float(m_row.get('SMA200_Slope', 0))
            else:
                try:
                    loc = market_df.index.get_indexer([date], method='pad')[0]
                    if loc != -1:
                        m_row = market_df.iloc[loc]
                        m_regime = m_row['Market_Regime']
                        sma_slope = float(m_row.get('SMA200_Slope', 0))
                except: pass
        
        score = 0
        if m_regime == 'Bullish': score = 100 
        elif m_regime == 'Recovery': score = 80 # Recoveryも高評価
        elif m_regime == 'Bearish': score = 20

        sub_regime = "Uptrend" if sma_slope > 0 else "Downtrend"

        return {
            'Date': date.strftime('%Y-%m-%d'),
            'Ticker': "",
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

def check_iron_rules_v15(metrics):
    # ★ V15: 鉄の掟を大幅緩和
    # Market Bearishでも、個別株が爆上げ(ROC>10)なら許可する
    if metrics['Market_Regime'] == 'Bearish' and metrics['ROC_10'] < 10:
        return "Market Bearish (Low ROC)"
    
    # ROC上限撤廃: 勢いがあるならOK
    # ADX上限撤廃: トレンド強すぎてもOK
    
    # 唯一のNG: 雲の下で、かつ勢いがない場合
    if metrics['Price_vs_Cloud'] == "Below" and metrics['ROC_10'] < 5: 
        return "Below Cloud (Weak)"
        
    return None

class MemorySystem:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        
        # ★ V15でAI検索に使用する特徴量（これを先に定義！）
        self.feature_cols = ['adx', 'roc', 'vwap_dev'] 
        
        # 保存用ヘッダー定義（CSVの並び順と一致させる）
        self.headers = [
            "Date","Ticker","Timeframe","Action","Result","Reasoning","Confidence",
            "Entry_Price","Entry_Signal_Volume","Exit_Price","Exit_Signal_Price",
            "ADX","ROC_10","Market_Regime_Score","VWAP_Distance_Percent",
            "Volume_Change_Percent","Market_Regime","Sub_Regime","profit_rate"
        ]
        
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return

        try:
            self.df = pd.read_csv(self.csv_path)
            self.df.columns = [c.strip() for c in self.df.columns]
            
            # カラム名の読み替え（CSVの大文字 → プログラムの小文字）
            col_map = {
                'Result': 'result', 
                'Action': 'action',
                'ADX': 'adx', 
                'ROC_10': 'roc', 
                'VWAP_Distance_Percent': 'vwap_dev', 
                'Volume_Change_Percent': 'vol_change'
            }
            self.df.rename(columns=col_map, inplace=True)

            if 'result' in self.df.columns:
                valid_df = self.df[self.df['result'].isin(['WIN', 'LOSS'])].copy()
                if len(valid_df) > 5:
                    # 必要な特徴量カラムがなければ0で埋める
                    for col in self.feature_cols:
                        if col not in valid_df.columns: valid_df[col] = 0
                    
                    features = valid_df[self.feature_cols].fillna(0)
                    self.features_normalized = self.scaler.fit_transform(features)
                    self.valid_df_for_knn = valid_df 
                    
                    n = min(15, len(valid_df))
                    self.knn = NearestNeighbors(n_neighbors=n, metric='euclidean')
                    self.knn.fit(self.features_normalized)
                    print(f"✅ 記憶ロード完了: {len(valid_df)}件のトレード経験")
        except Exception as e:
            print(f"❌ 記憶ロードエラー: {e}")

    def get_similar_cases_text(self, current_metrics):
        if self.knn is None: return "（データ不足）"
        try:
            # 入力データをDataFrame化して警告を回避
            vec = [current_metrics.get(col, 0) for col in self.feature_cols]
            input_df = pd.DataFrame([vec], columns=self.feature_cols)
            
            # 検索実行
            vec_norm = self.scaler.transform(input_df)
            dists, indices = self.knn.kneighbors(vec_norm)
            
            win_c = 0; loss_c = 0
            for idx in indices[0]:
                row = self.valid_df_for_knn.iloc[idx]
                res = str(row.get('result', ''))
                if res == 'WIN': win_c += 1
                if res == 'LOSS': loss_c += 1
            
            rate = win_c / (win_c + loss_c) * 100 if (win_c + loss_c) > 0 else 0
            return f"勝率: {rate:.0f}% (勝{win_c}/負{loss_c})"
        except: return "（検索エラー）"

    # ★復活させた保存機能
    def save_experience(self, data_dict):
        try:
            # 必要なカラムだけ抽出して並べ替え
            row = {k: data_dict.get(k, "") for k in self.headers}
            df_row = pd.DataFrame([row])
            
            # ファイルが存在しない場合は新規作成（ヘッダー付き）
            if not os.path.exists(self.csv_path):
                df_row.to_csv(self.csv_path, index=False, encoding='utf-8')
            else:
                # 存在する場合は追記（ヘッダーなし）
                df_row.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8')
        except Exception as e:
            print(f"❌ 保存エラー: {e}")
            
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

def ai_decision_maker_aggressive(model, chart, metrics, cbr):
    # ★ V15 Prompt: 超強気人格
    prompt = f"""
### Role
あなたは「V15 バーサーカー・トレーダー」です。
臆病な守りは捨て、リスクを取って爆発的な利益(Home Run)を狙います。

### Market Data
Ticker: {metrics['Ticker']}
Price: {metrics['Price']:.0f}
Regime: {metrics['Market_Regime']} (Slope: {metrics['Sub_Regime']})
ADX: {metrics['ADX']:.1f} (Trend Strength)
ROC: {metrics['ROC_10']:.1f}% (Momentum)

{cbr}

### Doctrine (Aggressive)
1. **Momentum is King**: ROCが高いなら、高値掴みを恐れずエントリーせよ。
2. **Ignore Overheat**: ADXが高くても、それはトレンドが最強である証拠だ。順張りせよ。
3. **Catch the Falling Knife**: もしMarketがBearishでも、ROCが+10%を超えているなら「リバウンド」の大チャンスだ。狙え。
4. **Speed**: 迷ったら買え。機会損失(FOMO)こそが最大のリスクだ。

### Output (JSON)
{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "reason": "攻撃的な理由",
  "sl_multiplier": 2.0 
}}
"""
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart}])
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(re.search(r'\{.*\}', text, re.DOTALL).group(0))
    except: return {"action": "HOLD", "confidence": 0}

def main():
    print("=== AI Training V15 Aggressive (Berserker Mode) ===")
    
    # 1. メモリシステム
    print("1. メモリシステム起動中...", end="")
    try:
        memory = MemorySystem(LOG_FILE)
        print("OK")
    except Exception as e:
        print(f"\n❌ エラー: {e}"); return

    # 2. AIモデル
    print("2. Geminiモデル初期化中...", end="")
    try: 
        model = genai.GenerativeModel(MODEL_NAME)
        print("OK")
    except Exception as e: 
        print(f"\n❌ エラー: {e}"); return

    # 3. データ取得
    print("3. 市場データ取得中(^N225)...")
    nikkei = download_data_safe("^N225")
    if nikkei is None: 
        print("❌ 取得失敗: yfinanceを更新してください"); return
    
    market_df = calculate_market_filter_v15(nikkei)
    if market_df is None: print("❌ 計算失敗"); return

    count = 0
    print(f"5. トレーニング開始 (目標: {TRAINING_ROUNDS}回)")
    
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
            
            # ★ V15: 鉄の掟チェック (かなり緩い)
            if check_iron_rules_v15(metrics): continue

            # AI判定
            cbr = memory.get_similar_cases_text(metrics)
            chart = create_chart_image(df.iloc[:idx+1], ticker)
            
            decision = ai_decision_maker_aggressive(model, chart, metrics, cbr)
            
            # 攻撃的なので、Confidenceが低くてもBUYする可能性があるが、一応Actionで判断
            if decision.get('action') != 'BUY': continue
            
            # シミュレーション
            count += 1
            entry_price = metrics['Price']
            atr = metrics['ATR']
            
            # 損切りは浅く(回転率重視)
            sl_mult = float(decision.get('sl_multiplier', 2.0))
            sl = entry_price - (atr * sl_mult)
            
            future = df.iloc[idx+1:idx+21]
            exit_price = entry_price
            result = "LOSS"
            
            for i in range(len(future)):
                row = future.iloc[i]
                if row['Low'] <= sl:
                    exit_price = sl
                    result = "LOSS"
                    break
                # 利確目標は強気に (4ATR)
                if row['High'] >= entry_price + (atr * 4.0):
                    exit_price = entry_price + (atr * 4.0)
                    result = "WIN"
                    break
                if i == len(future)-1:
                    exit_price = row['Close']
                    result = "WIN" if exit_price > entry_price else "LOSS"

            shares = int((TRADE_BUDGET * 0.2) / entry_price)
            if shares == 0: shares = 1
            pl_amount = (exit_price - entry_price) * shares
            pl_rate = ((exit_price - entry_price) / entry_price) * 100
            
            log_color = "🔴" if result == "WIN" else "🔵"
            print(f"{log_color} [{result}] {ticker} : {pl_amount:+,.0f}円 ({pl_rate:+.2f}%) | Reason: {decision.get('reason','-')}")

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
            print("\nユーザー中断"); break
        except Exception:
            continue

    print("\nトレーニング完了")

if __name__ == "__main__":
    main()