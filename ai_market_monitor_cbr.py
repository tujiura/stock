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
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import re
import logging

# ---------------------------------------------------------
# ★環境設定
# ---------------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')

def allowed_gai_family():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = allowed_gai_family

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# ==========================================
# ★設定エリア
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")

webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
genai.configure(api_key=GOOGLE_API_KEY)

# ファイル設定
LOG_FILE = "ai_trade_memory_risk_managed.csv" # 学習用メモリ
REAL_TRADE_LOG_FILE = "real_trade_record.csv" # 実戦用ログ
MODEL_NAME = 'models/gemini-2.0-flash' 

TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15

# 監視リスト (ユーザー指定の厳選80銘柄)
WATCH_LIST = [
    # --- 🏆 エース級 ---
    "6146.T", "8035.T", "9983.T", "7741.T", "6857.T", "7012.T", "6367.T", "7832.T",
    "1801.T", "9766.T", "2801.T", "4063.T", "4543.T", "4911.T", "4507.T",
    # --- 🛡️ 安定・ディフェンシブ ---
    "9432.T", "9433.T", "9434.T", "4503.T", "4502.T", "2502.T", "2503.T", "2802.T",
    "4901.T", "1925.T", "1928.T", "1802.T", "1803.T", "1812.T", "9020.T", "9021.T",
    "9532.T", "9735.T", "9613.T",
    # --- 💰 金融・銀行 ---
    "8306.T", "8316.T", "8411.T", "8308.T", "8309.T", "8331.T", "8354.T", "8766.T",
    "8725.T", "8591.T", "8593.T", "8604.T", "8473.T", "8630.T", "8697.T",
    # --- 🏢 商社・卸売 ---
    "8058.T", "8031.T", "8001.T", "8002.T", "8015.T", "2768.T", "8053.T", "7459.T",
    "8088.T", "9962.T", "3092.T", "3382.T",
    # --- 🏭 重厚長大・自動車 ---
    "7011.T", "7013.T", "6301.T", "7203.T", "7267.T", "7269.T", "7270.T", "7201.T",
    "5401.T", "5411.T", "5713.T", "1605.T", "5020.T",
    # --- 📦 その他・機械 ---
    "6501.T", "6503.T", "6305.T", "6326.T", "6383.T", "6471.T", "6473.T", "7751.T"
]

plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 1. データ取得関数群
# ==========================================
def download_data_safe(ticker, period="6mo", interval="1d", retries=3):
    wait = 2
    for attempt in range(retries):
        try:
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            if len(df) < 50: return None
            return df
        except:
            time.sleep(wait); wait *= 2
    return None

def get_macro_data():
    """主要指数を取得"""
    tickers = {"^N225": "日経平均", "JPY=X": "ドル円", "^GSPC": "米S&P500", "^TNX": "米10年債", "^VIX": "VIX"}
    report = "【🌎 マクロ環境】\n"
    try:
        data = yf.download(list(tickers.keys()), period="5d", progress=False)
        if isinstance(data.columns, pd.MultiIndex): df_close = data['Close']
        else: df_close = data['Close'] if 'Close' in data else data

        for symbol, name in tickers.items():
            try:
                series = df_close[symbol].dropna()
                if len(series) < 2: continue
                current = float(series.iloc[-1])
                change = (current - float(series.iloc[-2])) / float(series.iloc[-2]) * 100
                icon = "↗️" if change > 0 else "↘️"
                report += f"- {name}: {current:.2f} ({icon} {change:+.2f}%)\n"
            except: pass
    except: return "【マクロ環境】取得エラー"
    return report.strip()

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return "データなし"
        name = info.get("longName", ticker)
        per = info.get("trailingPE", "-")
        pbr = info.get("priceToBook", "-")
        roe = info.get("returnOnEquity", "-")
        per_str = f"{per:.1f}倍" if isinstance(per, (int, float)) else "-"
        pbr_str = f"{pbr:.2f}倍" if isinstance(pbr, (int, float)) else "-"
        roe_str = f"{roe*100:.1f}%" if isinstance(roe, (int, float)) else "-"
        return f"- {name}\n- PER:{per_str}, PBR:{pbr_str}, ROE:{roe_str}"
    except: return "取得エラー"

def get_latest_news(ticker):
    try:
        q = urllib.parse.quote(f"{ticker} 株価 ニュース")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        feed = feedparser.parse(url)
        if not feed.entries: return "特になし"
        return "\n".join([f"・{e.title}" for e in feed.entries[:2]])
    except: return "取得エラー"

def get_earnings_date(ticker):
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar and 'Earnings Date' in calendar:
            earnings_date = calendar['Earnings Date'][0]
            if isinstance(earnings_date, (datetime.date, datetime.datetime)):
                return earnings_date.strftime('%Y-%m-%d')
    except: pass
    return "-"

def get_weekly_trend(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1wk", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 26: return "不明"
        sma13 = df['Close'].rolling(13).mean().iloc[-1]
        sma26 = df['Close'].rolling(26).mean().iloc[-1]
        price = float(df['Close'].iloc[-1])
        if price > sma13 > sma26: return "上昇 📈 (強)"
        elif price > sma13: return "上昇 ↗️ (短)"
        elif price < sma13 < sma26: return "下降 📉 (弱)"
        else: return "レンジ ➡️"
    except: return "不明"

# ==========================================
# 2. 指標計算
# ==========================================
def calculate_metrics_enhanced(df):
    if len(df) < 26: return None
    curr = df.iloc[-1]
    price = float(curr['Close'])
    
    sma25 = float(curr['SMA25'])
    sma25_dev = ((price / sma25) - 1) * 100
    
    prev_sma25 = float(df['SMA25'].iloc[-6])
    slope = (sma25 - prev_sma25) / 5
    trend_momentum = (slope / price) * 1000
    
    macd = float(curr['MACD'])
    signal = float(curr['Signal'])
    macd_power = ((macd - signal) / price) * 10000
    
    atr = float(curr['ATR'])
    entry_volatility = (atr / price) * 100
    
    # BB幅
    std = df['Close'].rolling(20).std().iloc[-1]
    bb_width = (4 * std) / df['Close'].rolling(20).mean().iloc[-1] * 100
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    rs = gain / loss
    rsi_9 = 100 - (100 / (1 + rs)).iloc[-1]

    return {
        'sma25_dev': sma25_dev,
        'trend_momentum': trend_momentum,
        'macd_power': macd_power,
        'entry_volatility': entry_volatility,
        'price': price,
        'atr_value': atr,
        'bb_width': bb_width,
        'rsi_9': rsi_9
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
        self.feature_cols = ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility', 'rsi_9']
        
        self.csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "stop_loss_reason", "Price", 
            "sma25_dev", "trend_momentum", "macd_power", "entry_volatility", "rsi_9", 
            "profit_loss", "profit_rate" 
        ]
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # --- スキーマ自動更新 ---
            missing_cols = [col for col in self.csv_columns if col not in self.df.columns]
            if missing_cols:
                for col in missing_cols: self.df[col] = 0.0
                self.df = self.df[self.csv_columns]
                self.df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"⚠️ CSV読込エラー: {e}")
            try:
                self.df = pd.read_csv(self.csv_path, on_bad_lines='skip')
                missing_cols = [col for col in self.csv_columns if col not in self.df.columns]
                for col in missing_cols: self.df[col] = 0.0
                self.df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
            except Exception: return

        try:
            rename_map = {'date': 'Date', 'ticker': 'Ticker', 'action': 'Action', 'result': 'result'}
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
        if self.knn is None: return "（データ不足）"
        metrics_vec = [current_metrics.get(col, 0) for col in self.feature_cols]
        input_df = pd.DataFrame([metrics_vec], columns=self.feature_cols)
        scaled_vec = self.scaler.transform(input_df)
        distances, indices = self.knn.kneighbors(scaled_vec)
        
        text = f"【類似過去事例】\n"
        win_c = 0; loss_c = 0
        for idx in indices[0]:
            row = self.valid_df_for_knn.iloc[idx]
            res = str(row.get('result', ''))
            if res == 'WIN': win_c += 1
            if res == 'LOSS': loss_c += 1
            icon = "⭕" if res=='WIN' else "❌"
            text += f"- {row.get('Date')} {row.get('Ticker')}: {icon} (MOM:{row.get('trend_momentum',0):.1f})\n"
        text += f"-> 傾向: 勝ち{win_c} / 負け{loss_c}\n"
        return text

# ==========================================
# 4. AI判定
# ==========================================
def create_chart_image(df, name):
    data = df.tail(100).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(data.index, data['Close'], color='black', label='Close')
    ax1.plot(data.index, data['SMA25'], color='orange', label='SMA25')
    ax1.set_title(f"{name} Analysis")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(data.index, data['MACD'], color='red')
    ax2.bar(data.index, data['MACD']-data['Signal'], color='gray', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
    return {"mime_type": "image/png", "data": buf.getvalue()}

def ai_decision_maker(model, chart_bytes, metrics, cbr_text, macro, news, fundamentals, weekly, ticker):
    # 鉄の掟フィルター (最適化: 2.3%)
    if metrics['entry_volatility'] > 2.3:
        return {"action": "HOLD", "confidence": 0, "reason": f"【鉄の掟】ボラティリティ過大 ({metrics['entry_volatility']:.2f}%)"}
    if metrics['trend_momentum'] < 0:
        return {"action": "HOLD", "confidence": 0, "reason": "【鉄の掟】下降トレンド中 (Momentum < 0)"}
    if metrics['sma25_dev'] < 0:
        return {"action": "HOLD", "confidence": 0, "reason": "【鉄の掟】SMA25割れ (戻り待ち)"}

    prompt = f"""
### CONTEXT
対象: {ticker}
【テクニカル指標】
- トレンド勢い: {metrics['trend_momentum']:.2f} (プラス必須)
- SMA25乖離率: {metrics['sma25_dev']:.2f}% (プラス圏)
- ボラティリティ: {metrics['entry_volatility']:.2f}% (2.3%以下推奨)
- RSI(9): {metrics['rsi_9']:.1f}
- ATR: {metrics['atr_value']:.1f}
- 週足トレンド: {weekly}

{macro}

{fundamentals}

【ニュース】
{news}

{cbr_text}

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
    safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    try:
        response = model.generate_content([prompt, chart_bytes], safety_settings=safety)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: text = match.group(0)
        return json.loads(text)
    except Exception as e:
        return {"action": "HOLD", "reason": f"AI Error: {e}", "confidence": 0}

def send_discord_notify(message, filename=None):
    if not webhook_url: return
    try:
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        payload = {"content": f"📊 **AI市場監視レポート ({now_str})**\n{message[:1500]}"}
        
        files = {}
        if filename:
            files["file"] = (f"Report_{now_str.replace(':','-')}.txt", message.encode('utf-8'))
            
        requests.post(webhook_url, data=payload, files=files if filename else None)
        print("✅ Discord通知送信")
    except Exception as e:
        print(f"⚠️ Discord送信エラー: {e}")

# ==========================================
# 5. メイン実行
# ==========================================
if __name__ == "__main__":
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"=== AI市場監視システム ({today}) ===")
    
    WATCH_LIST = sorted(list(set(WATCH_LIST)))
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e: print(f"Error: {e}"); exit()

    memory = CaseBasedMemory(LOG_FILE)
    macro = get_macro_data()
    print(macro)
    
    report_message = f"**📊 AI市場監視レポート ({today})**\n\n{macro}\n"
    buy_list = []
    all_stock_prices = [] # 全銘柄価格リスト
    
    SAVE_TARGETS = [
        {"path": LOG_FILE, "name": "学習メモリ"},
        {"path": REAL_TRADE_LOG_FILE, "name": "実戦ログ"}
    ]

    current_hour_jst = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).hour
    is_closing_time = (current_hour_jst >= 15)
    print(f"🕒 現在 {current_hour_jst}時: {'記録モード' if is_closing_time else '監視モード'}")

    for i, tic in enumerate(WATCH_LIST, 1):
        print(f"[{i}/{len(WATCH_LIST)}] {tic}... ", end="", flush=True)
        
        df = download_data_safe(tic, interval=TIMEFRAME)
        if df is None: print("Skip"); continue
        
        df['SMA25'] = df['Close'].rolling(25).mean()
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        
        df = df.dropna()
        metrics = calculate_metrics_enhanced(df)
        if metrics is None: print("Skip"); continue
        
        # --- 株価リスト用データ作成 (フィルター前に実施) ---
        current_price = metrics['price']
        try:
            prev_close = df['Close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            price_str = f"• {tic}: {current_price:,.0f}円 ({change:+.0f} / {change_pct:+.2f}%)"
        except:
            price_str = f"• {tic}: {current_price:,.0f}円"
        all_stock_prices.append(price_str)

        # 3. 鉄の掟フィルター (最適化: 2.3%)
        if metrics['trend_momentum'] < 0 or metrics['sma25_dev'] < 0 or metrics['entry_volatility'] > 2.3:
             print("⏹️ Filtered"); continue

        # 付加情報取得
        earnings_date = get_earnings_date(tic)
        cbr_text = memory.search_similar_cases(metrics)
        chart = create_chart_image(df, tic)
        news = get_latest_news(tic)
        fund = get_fundamentals(tic)
        weekly = get_weekly_trend(tic)
        
        res = ai_decision_maker(model_instance, chart, metrics, cbr_text, macro, news, fund, weekly, tic)
        
        action = res.get('action', 'HOLD')
        conf = res.get('confidence', 0)
        
        # ATRトレーリングストップ計算 (BUYの場合)
        stop_loss_price = 0
        if action == "BUY":
            atr_stop = metrics['atr_value'] * 2.0
            stop_loss_price = metrics['price'] - atr_stop
        
        # CSVデータ作成 (★profit_rate 0.0で初期化)
        item = {
            "Date": today, "Ticker": tic, "Timeframe": TIMEFRAME, 
            "Action": action, "result": "", "Reason": res.get('reason', 'None'), 
            "Confidence": conf, "stop_loss_price": stop_loss_price, "stop_loss_reason": "ATR_Trailing_Stop",
            "Price": metrics['price'], "sma25_dev": metrics['sma25_dev'], 
            "trend_momentum": metrics['trend_momentum'], "macd_power": metrics['macd_power'],
            "entry_volatility": metrics['entry_volatility'], "rsi_9": metrics['rsi_9'], 
            "profit_loss": 0,
            "profit_rate": 0.0 
        }
        
        # 保存処理 (15時以降のみ)
        if is_closing_time:
            df_new = pd.DataFrame([item])
            # カラム順序を揃える
            for col in memory.csv_columns:
                if col not in df_new.columns: df_new[col] = None
            df_new = df_new[memory.csv_columns]
            
            for target in SAVE_TARGETS:
                try:
                    path = target["path"]
                    if os.path.exists(path):
                        df_new.to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')
                    else:
                        df_new.to_csv(path, index=False, encoding='utf-8-sig')
                except: pass
            print(f"📝 {action} ({conf}%)")
        else:
            print(f"👀 {action} ({conf}%)")

        if action == "BUY" and conf >= 70:
            earnings_warning = f"\n⚠️ **決算注意**: {earnings_date}" if earnings_date != "-" else ""
            msg = (
                f"🔴 **BUY {tic}**: {metrics['price']:.0f}円\n"
                f"🛡️ **推奨損切り**: **{stop_loss_price:.0f}円** (ATR x2.0)\n"
                f"💡 **勝利の戦術**: \n"
                f"   ・含み益+3.0%までは逆指値を動かさない(忍耐)\n"
                f"   ・+2.5%超で建値ガード発動\n"
                f"   ・+3%超でATR追従開始、+5%超で鬼利確\n"
                f"{earnings_warning}\n"
                f"> 理由: {res.get('reason')}"
            )
            buy_list.append(msg)
        time.sleep(2)

    if buy_list:
        report_message += "\n\n🚀 **新規BUY推奨**\n" + "\n\n".join(buy_list)
    else:
        report_message += "\n\n💤 推奨なし"

    # 全銘柄リストを追加
    if all_stock_prices:
        report_message += "\n\n" + "="*30 + "\n📉 **監視銘柄 株価一覧**\n" + "="*30 + "\n"
        report_message += "\n".join(all_stock_prices)

    send_discord_notify(report_message, filename="FullReport")