import os
import io
import time
import json
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
# GUI画面を出さずに画像だけ作るモード('Agg')に設定
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import re
import logging
import socket
import requests
import requests.packages.urllib3.util.connection as urllib3_cn
import warnings

# ---------------------------------------------------------
# ★環境設定
# ---------------------------------------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

def allowed_gai_family():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = allowed_gai_family

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")

# 設定
LOG_FILE = "ai_trade_memory_v15_aggressive.csv" # V15の記憶ファイル
MODEL_NAME = 'models/gemini-3-pro-preview'  # V15モデル

# V15 Aggressive Parameters
ADX_MIN = 15.0  
ADX_MAX = 75.0  
ROC_MAX = 100.0 
ATR_MULTIPLIER = 2.0 
VWAP_WINDOW = 20

# 監視リスト
WATCH_LIST = [
    "8035.T", "6857.T", "6146.T", "6920.T", "6758.T", "6702.T", "6501.T", "6503.T", "7751.T", "4063.T", "6981.T", "6723.T",
    "7203.T", "7267.T", "6902.T", "6301.T", "6367.T", "7011.T", "7013.T", 
    "8306.T", "8316.T", "8411.T", "8766.T", "8058.T", "8001.T", "8031.T", "8002.T", "9984.T",
    "9432.T", "9983.T", "4568.T", "4543.T", "4661.T", "7974.T", "6506.T",
    "5253.T", "5032.T", "9166.T", "4385.T", "4478.T", "4483.T", "3993.T", "4180.T", "3687.T", "6027.T",
    "5595.T", "9348.T", "7012.T", "6203.T", 
    "6254.T", "6315.T", "6526.T", "6228.T", "6963.T", "3436.T", "7735.T", "6890.T",
    "2768.T", "7342.T", "2413.T", "2222.T", "7532.T", "3092.T",
    "9101.T", "9104.T", "9107.T", "1605.T", "5713.T", "5401.T", "5411.T"
]
WATCH_LIST = sorted(list(set(WATCH_LIST)))

plt.rcParams['font.family'] = 'sans-serif'
genai.configure(api_key=GOOGLE_API_KEY, transport="rest")

# ==========================================
# 0. Discord 通知機能 (V15仕様)
# ==========================================
def send_discord_notify(message, filename=None):
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ Discord Webhook URLが設定されていません。通知をスキップします。")
        return
    try:
        # 日時タグ (自動変換)
        ts = int(time.time())
        time_tag = f"<t:{ts}:f>" # 例: 2024年1月1日 12:00
        
        # メッセージ本文 (V15仕様)
        content_body = f"🚀 **AI市場監視レポート V15 (Berserker) {time_tag}**\n\n{message}"
        if len(content_body) > 1900:
            content_body = content_body[:1900] + "\n...(詳細は添付ファイルを参照)"

        payload = {"content": content_body}
        files = {}
        
        if filename:
            files["file"] = (filename, message.encode('utf-8'))

        response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files if filename else None)
        
        if response.status_code in [200, 204]:
            print("✅ Discord通知送信成功")
        else:
            print(f"⚠️ Discord送信エラー: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"⚠️ Discord送信例外: {e}")

# ==========================================
# 1. データ取得 & テクニカル計算 (V15)
# ==========================================
def download_data_safe(ticker, period="1y", interval="1d", retries=3): 
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
            if len(df) < 100: return None
            return df
        except:
            time.sleep(wait); wait *= 2
    return None

def get_fundamentals(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            'PER': info.get('trailingPE', None),
            'PBR': info.get('priceToBook', None)
        }
    except:
        return {'PER': None, 'PBR': None}

def calculate_technical_v15(df):
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
        df['VWAP_Dev'] = np.where(df['VWAP'].notna(), ((close - df['VWAP']) / df['VWAP']) * 100, 0)
        
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        df['Cloud_Top'] = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        
        return df.dropna()
    except Exception: return None

def calculate_metrics_v15(df, idx):
    try:
        if idx < 60 or idx >= len(df): return None
        curr = df.iloc[idx]
        price = float(curr['Close'])
        
        adx = float(curr.get('ADX', 20.0))
        roc = float(curr.get('ROC', 0.0))
        
        # V15 Regime
        if ADX_MIN <= adx <= ADX_MAX: regime = "Trend"
        elif adx > ADX_MAX: regime = "Super Trend" # V15 Overheat
        else: regime = "Weak"

        cloud_top = float(curr.get('Cloud_Top', price))
        price_vs_cloud = "Above" if price > cloud_top else "Below"

        return {
            'date': df.index[idx].strftime('%Y-%m-%d'),
            'price': price,
            'adx': adx,
            'roc': roc,
            'atr_value': float(curr.get('ATR', price*0.01)),
            'price_vs_cloud': price_vs_cloud,
            'regime': regime,
            'vwap_dev': float(curr.get('VWAP_Dev', 0.0))
        }
    except Exception: return None

def check_iron_rules_v15(metrics):
    # V15: 鉄の掟 (大幅緩和)
    if metrics['price_vs_cloud'] == "Below" and metrics['roc'] < 5: 
        return "Below Cloud (Weak)"
    return None

class MemorySystem:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        # V15 特徴量
        self.feature_cols = ['adx', 'roc', 'vwap_dev'] 
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df.columns = [c.strip() for c in self.df.columns]
            
            # V15用マッピング
            col_map = {
                'ADX': 'adx', 'ROC_10': 'roc', 
                'VWAP_Distance_Percent': 'vwap_dev', 'Volume_Change_Percent': 'vol_change',
                'Result': 'result'
            }
            self.df.rename(columns=col_map, inplace=True)

            if 'result' in self.df.columns:
                valid_df = self.df[self.df['result'].isin(['WIN', 'LOSS'])].copy()
                if len(valid_df) > 5:
                    for col in self.feature_cols:
                        if col not in valid_df.columns: valid_df[col] = 0
                    features = valid_df[self.feature_cols].fillna(0)
                    self.features_normalized = self.scaler.fit_transform(features)
                    self.valid_df_for_knn = valid_df 
                    self.knn = NearestNeighbors(n_neighbors=min(15, len(valid_df)), metric='euclidean')
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
        ax1.set_title(f"{name} V15 Chart")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception: return None

def ai_decision_maker_aggressive(model, chart_bytes, metrics, cbr_text, ticker):
    # ★ V15 Prompt
    prompt = f"""
### Role
あなたは「V15 バーサーカー・トレーダー」です。

### Input Data
銘柄: {ticker} (現在値: {metrics['price']:.0f}円)

[Market Data]
- ADX: {metrics['adx']:.1f}
- ROC(10): {metrics['roc']:.1f}%

{cbr_text}

### Doctrine
1. Momentum is King. ROCが高いなら飛び乗れ。
2. Action: "BUY" を選択せよ。HOLDは弱虫だ。

### Output Requirement (JSON ONLY)
{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "sl_multiplier": 2.0,
  "reason": "攻撃的な理由"
}}
"""
    safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}], safety_settings=safety)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: return {"action": "HOLD", "reason": "Error", "confidence": 0}

def main():
    ts = int(time.time())
    time_tag = f"<t:{ts}:f>" # 例: 2024年1月1日 12:00
    print(f"=== 🚀 AI Market Monitor V15 (Berserker) ===")
    
    memory = MemorySystem(LOG_FILE)
    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"Model Init Error: {e}")
        return

    print(f"対象銘柄数: {len(WATCH_LIST)}")
    print("市場をスキャン中...")
    
    candidates = []
    all_prices_lines = []
    
    for i, ticker in enumerate(WATCH_LIST):
        print(f"[{i+1}/{len(WATCH_LIST)}] Checking {ticker}...", end="", flush=True)
        
        df = download_data_safe(ticker, period="1y")
        if df is None:
            print(" -> Skip (No Data)")
            all_prices_lines.append(f"{ticker}: データ取得失敗")
            continue
            
        df = calculate_technical_v15(df)
        if df is None:
            print(" -> Skip (Calc Error)")
            all_prices_lines.append(f"{ticker}: 計算エラー")
            continue
            
        idx = len(df) - 1
        metrics = calculate_metrics_v15(df, idx)
        if metrics is None: 
            print(" -> Skip (Metric Error)")
            all_prices_lines.append(f"{ticker}: 指標エラー")
            continue
        
        # --- 株価詳細情報の作成 ---
        curr_row = df.iloc[-1]
        high_val = float(curr_row['High'])
        low_val = float(curr_row['Low'])
        atr_val = metrics['atr_value']
        adx_val = metrics['adx']
        
        # ADXトレンド判定
        prev_adx = df['ADX'].iloc[-2] if len(df) > 1 else adx_val
        adx_trend = "➚" if adx_val > prev_adx else "➘"
        
        # フォーマット (維持)
        price_info = f"{ticker} 現在値: {metrics['price']:.0f} 高値: {high_val:.0f} / 安値: {low_val:.0f} ATR: {atr_val:.0f} (ADX: {adx_val:.0f}{adx_trend})"
        all_prices_lines.append(price_info)
        # ------------------------

        iron_rule = check_iron_rules_v15(metrics)
        if iron_rule:
            print(f" -> Skip ({iron_rule})")
            continue
            
        print(" -> Analyzing...", end="", flush=True)
        chart_bytes = create_chart_image(df, ticker)
        cbr_text = memory.get_similar_cases_text(metrics)
        
        decision = ai_decision_maker_aggressive(model, chart_bytes, metrics, cbr_text, ticker)
        
        action = decision.get('action', 'HOLD')
        conf = decision.get('confidence', 0)
        
        # V15の基準 (攻撃的)
        if action == "BUY" and conf >= 60: # 閾値を70->60に緩和
            atr = metrics['atr_value']
            sl_mult = float(decision.get('sl_multiplier', ATR_MULTIPLIER))
            sl_price = metrics['price'] - (atr * sl_mult)
            fund = get_fundamentals(ticker)
            per_str = f"{fund['PER']:.1f}" if fund['PER'] else "-"
            
            candidates.append({
                'Ticker': ticker,
                'Price': metrics['price'],
                'Conf': conf,
                'Reason': decision['reason'],
                'SL': sl_price,
                'PER': per_str,
                'Regime': metrics['regime']
            })
            print(f" -> FOUND! {ticker} (Conf:{conf}%)")
        else:
            print(f" -> Pass ({action}, {conf}%)")
            
        time.sleep(1) 

    print("\n" + "="*60)
    print(f"🚀 本日の推奨銘柄リスト ({len(candidates)}銘柄)")
    print("="*60)
    
    discord_message = ""
    
    if candidates:
        df_res = pd.DataFrame(candidates)
        print(df_res[['Ticker', 'Price', 'Conf', 'SL', 'Regime', 'Reason']].to_string(index=False))
        
        # Discordメッセージフォーマット (V15仕様)
        discord_message = f"**【AI推奨銘柄 V15 (Berserker)】**\n\n" + time_tag + "\n\n"
        for c in candidates:
            discord_message += f"**{c['Ticker']}** (現在値: {c['Price']:.0f}円)\n"
            discord_message += f"📊 {c['Regime']} | 🔥 自信度: {c['Conf']}%\n"
            discord_message += f"🛡️ SL目安: {c['SL']:.0f}円 | 📝 {c['Reason']}\n"
            discord_message += "-"*20 + "\n"
    else:
        print("本日の推奨銘柄はありませんでした。")
        discord_message = "本日の推奨銘柄はありませんでした。\n"

    # 全銘柄リストの追加 (そのまま維持)
    discord_message += "\n**【全監視銘柄 詳細データ】**\n"
    discord_message += "(Code | Close | High/Low | ATR | Memo)\n"
    discord_message += "\n".join(all_prices_lines)

    # 送信
    send_discord_notify(discord_message, filename="Market_Monitor_Full_V15.txt")

if __name__ == "__main__":
    main()