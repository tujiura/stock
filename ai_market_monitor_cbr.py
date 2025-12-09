import yfinance as yf
import pandas as pd
import google.generativeai as genai
import json
import time
import datetime
import urllib.parse
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import socket
import requests.packages.urllib3.util.connection as urllib3_cn

# ---------------------------------------------------------
# ★環境設定 & おまじない (Windows/GitHub Actions対応)
# ---------------------------------------------------------
# 1. Windowsでの文字化け防止
sys.stdout.reconfigure(encoding='utf-8')

# 2. 通信エラー防止 (IPv6無効化)
def allowed_gai_family():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = allowed_gai_family

# 3. ローカル環境用 (.env読み込み)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==========================================
# ★設定エリア
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

LOG_FILE = "ai_trade_memory_risk_managed.csv"
MODEL_NAME = 'models/gemini-2.5-pro' # モデル名は適宜変更してください
TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 11

# 監視リスト
WATCH_LIST = [
    "8035.T", "6146.T", "6920.T", "6857.T", "6723.T", "7735.T", "6526.T",
    "6758.T", "6861.T", "6501.T", "6503.T", "6981.T", "6954.T", "7741.T", 
    "6902.T", "6367.T", "6594.T", "7751.T",
    "7203.T", "7267.T", "7270.T", "7201.T", "7269.T",
    "8306.T", "8316.T", "8411.T", "8766.T", "8725.T", "8591.T", "8604.T",
    "8058.T", "8031.T", "8001.T", "8002.T", "8015.T", "2768.T",
    "7011.T", "7012.T", "7013.T", "6301.T", "5401.T", "9101.T", "9104.T", "9107.T",
    "9432.T", "9433.T", "9984.T", "9434.T", "4661.T", "6098.T",
    "7974.T", "9684.T", "9697.T", "7832.T",
    "9983.T", "3382.T", "8267.T", "9843.T", "3092.T", "4385.T", "7532.T",
    "4568.T", "4519.T", "4503.T", "4502.T", "4063.T", "4901.T", "4452.T", "2914.T",
    "8801.T", "8802.T", "1925.T", "1801.T",
    "9501.T", "9503.T", "1605.T", "5020.T",
    "9020.T", "9202.T", "2802.T"
]

plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 1. 通知機能 (Discord対応・堅牢版)
# ==========================================
def send_discord_notify(message):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("DISCORD_WEBHOOK_URLが設定されていません")
        return

    # Discord用データ作成
    data = {
        "content": message,
        "username": "AI投資アドバイザー",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/4228/4228956.png"
    }

    # リトライ設定
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        response = session.post(webhook_url, json=data, timeout=20)
        response.raise_for_status()
        print("✅ Discord通知送信成功")
    except Exception as e:
        print(f"⚠️ Discord送信エラー: {e}")

# ==========================================
# 2. データ取得 & テクニカル計算
# ==========================================
def download_data_safe(ticker, period="6mo", interval="1d", retries=3):
    wait = 2
    for _ in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: raise ValueError("Empty Data")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except:
            time.sleep(wait); wait *= 2
    return None

def calculate_metrics_enhanced(df):
    if len(df) < 15: return None 
    
    curr = df.iloc[-1]
    price = float(curr['Close'])
    
    sma25 = float(curr['SMA25'])
    sma25_dev = ((price / sma25) - 1) * 100
    
    if len(df) < 6: return None
    prev_sma25 = float(df['SMA25'].iloc[-6]) 
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
        'atr_value': atr
    }

# ==========================================
# 3. CBRメモリシステム
# ==========================================
class CaseBasedMemory:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        self.feature_cols = ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility']
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path)
            rename_map = {
                'date': 'Date', 'ticker': 'Ticker', 'action': 'Action', 
                'reason': 'Reason', 'timeframe': 'Timeframe',
                'result': 'result', 'profit_loss': 'profit_loss',
                'stop_loss_price': 'stop_loss_price', 
                'stop_loss_reason': 'stop_loss_reason' 
            }
            self.df.columns = [col.lower() for col in self.df.columns]
            self.df.rename(columns=rename_map, inplace=True)
            
            if len(self.df) < 5: return

            features = self.df[self.feature_cols].fillna(0)
            self.features_normalized = self.scaler.fit_transform(features)
            
            self.knn = NearestNeighbors(n_neighbors=min(CBR_NEIGHBORS_COUNT, len(self.df)), metric='euclidean')
            self.knn.fit(self.features_normalized)
            print(f"Memory System: Loaded {len(self.df)} cases.")
        except Exception as e:
            print(f"Memory Load Error: {e}")

    def search_similar_cases(self, current_metrics):
        if self.knn is None or len(self.df) < 5:
            return "（データ不足のため参照なし）"

        input_df = pd.DataFrame([current_metrics], columns=self.feature_cols)
        scaled_vec = self.scaler.transform(input_df) 
        distances, indices = self.knn.kneighbors(scaled_vec)
        
        text = f"【システム検索: 類似過去事例（{len(indices[0])}件）】\n"
        for idx in indices[0]:
            row = self.df.iloc[idx]
            res = str(row.get('result', ''))
            icon = "WIN ⭕" if res=='WIN' else "LOSS ❌" if res=='LOSS' else "➖"
            text += f"● {row['Date']} {row['Ticker']} -> {icon}\n"
        return text

# ==========================================
# 4. 外部データ & AI分析ロジック
# ==========================================
def get_macro_data():
    return "【マクロ環境】\n- 日経平均: レンジ相場\n- ドル円: 150円付近で推移"

def get_latest_news(keyword):
    q = urllib.parse.quote(f"{keyword} 株価 決算")
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja")
        return "\n".join([f"・{e.title}" for e in feed.entries[:2]]) if feed.entries else "なし"
    except: return "エラー"

def create_chart_image(df, name):
    data = df.tail(100).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(data.index, data['Close'], color='black', label='Close')
    ax1.plot(data.index, data['SMA25'], color='orange', label='SMA25')
    ax1.set_title(f"{name} ({TIMEFRAME}) Analysis")
    ax1.grid(True)
    ax2.plot(data.index, data['MACD'], color='red')
    ax2.bar(data.index, data['MACD']-data['Signal'], color='gray', alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
    return {"mime_type": "image/png", "data": buf.getvalue()}

def analyze_vision_agent(model_instance, chart, metrics, cbr_text, macro, news, name):
    mech_sl_long = metrics['price'] - (metrics['atr_value'] * 2.0)
    trend_dir = "上昇" if metrics['trend_momentum'] > 0 else "下降"

    vol_limit_msg = ""
    if metrics['entry_volatility'] >= 2.0:
        vol_limit_msg = "⚠️【禁止事項】現在のボラティリティ(2.0%以上)はリスク許容範囲外です。絶対にBUYしてはいけません。必ずHOLDを選択してください。"

    prompt = f"""
あなたは「超低リスク・買い専門のヘッジファンドCIO」です。
勝率100%を目指すため、リスクの高い局面は全て見送ります。

=== 入力情報 ===
銘柄: {name}
1. テクニカル:
   - トレンド: {trend_dir} (勢い: {metrics['trend_momentum']:.2f})
   - SMA25乖離: {metrics['sma25_dev']:.2f}%
   - ボラティリティ: {metrics['entry_volatility']:.2f}%
   - 統計的SL目安: {mech_sl_long:.0f} 円付近
2. ニュース: {news}

{cbr_text}

=== 鉄の掟 (売買基準) ===
{vol_limit_msg}

1. **【BUY (新規買い)】の絶対条件**
   - **ボラティリティが 2.0% 未満であること。** (これを超えていたら即却下)
   - SMA25が上向きで、明確な上昇トレンド中であること。
   - 価格がSMA25付近（押し目）にあること。
   - 必ず損切り(stop_loss_price)を設定すること。

2. **【HOLD (様子見・利益確定)】**
   - ボラティリティが 2.0% 以上の場合。
   - 下降トレンド、またはトレンドが不明確な場合。
   - **SELL（空売り）は禁止。** すべてHOLDで対応せよ。

=== 出力 (JSONのみ) ===
{{
  "action": "BUY", "HOLD" のいずれか,
  "confidence": 0-100,
  "stop_loss_price": 数値 (HOLDなら0),
  "stop_loss_reason": "直近安値◯◯円割れ... (30文字以内)",
  "reason": "ボラティリティ1.5%と低く、SMA25の押し目... (100文字以内)"
}}
"""
    try:
        response = model_instance.generate_content([prompt, chart])
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except: return {"action": "HOLD", "confidence": 0, "reason": "API Error", "stop_loss_price": 0}

# ==========================================
# 5. メイン実行 (実戦監視)
# ==========================================
if __name__ == "__main__":
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"=== AI市場監視システム ({today}) ===")
    
    try:
        model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"Error: {e}"); exit()

    cbr = CaseBasedMemory(LOG_FILE)
    macro = get_macro_data()
    
    report_message = f"**📊 AI市場監視レポート ({today})**\n"
    buy_list = []
    hold_list = []

    for i, tic in enumerate(WATCH_LIST, 1):
        name = tic 
        print(f"[{i}/{len(WATCH_LIST)}] {name}... ", end="", flush=True)
        
        df = download_data_safe(tic, interval=TIMEFRAME)
        if df is None or len(df) < 100:
            print("Skip")
            continue
            
        df['SMA25'] = df['Close'].rolling(25).mean()
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        df = df.dropna()
        metrics = calculate_metrics_enhanced(df)
        if metrics is None: 
            print("Skip")
            continue
        
        cbr_text = cbr.search_similar_cases(metrics)
        chart = create_chart_image(df, name)
        news = get_latest_news(name)
        
        res = analyze_vision_agent(model_instance, chart, metrics, cbr_text, macro, news, name)
        
        action = res.get('action', 'HOLD')
        conf = res.get('confidence', 0)
        sl_price_raw = res.get('stop_loss_price', 0)
        try: sl_price = float(sl_price_raw)
        except: sl_price = 0.0
        
        item = {
            "Date": today, "Ticker": tic, "Timeframe": TIMEFRAME, 
            "Action": action, "Confidence": conf,
            "stop_loss_price": sl_price, 
            "stop_loss_reason": res.get('stop_loss_reason', '-'),
            "Reason": res.get('reason', 'None'), 
            "Price": metrics['price'],
            "sma25_dev": metrics['sma25_dev'], 
            "trend_momentum": metrics['trend_momentum'],
            "macd_power": metrics['macd_power'],
            "entry_volatility": metrics['entry_volatility'],
            "profit_loss": 0
        }
        
        df_new = pd.DataFrame([item])
        if not os.path.exists(LOG_FILE):
            df_new.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
            
        action_icon = "BUY 🔴" if action == "BUY" else "HOLD ⚪"
        print(f"{action_icon}")

        if action == "BUY":
            sl_str = f"(SL: {sl_price:.0f}円)" if sl_price > 0 else ""
            msg = f"🔴 **{tic}** : {metrics['price']:.0f}円 {sl_str}\n> 理由: {res.get('reason')}"
            buy_list.append(msg)
        elif action == "HOLD":
            hold_list.append(f"⚪ {tic}")
        
        time.sleep(2)

    # 通知作成
    if buy_list:
        report_message += "\n🚀 **新規BUY銘柄**\n" + "\n\n".join(buy_list)
    else:
        report_message += "\n💤 本日は「BUY」銘柄はありませんでした。"

    if hold_list:
        report_message += f"\n\n☕ **HOLD銘柄 ({len(hold_list)}件)**\n"
        report_message += ", ".join(hold_list)

    # Discord送信
    send_discord_notify(report_message)