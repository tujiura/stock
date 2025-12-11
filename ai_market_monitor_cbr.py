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
    # exit() 

webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
genai.configure(api_key=GOOGLE_API_KEY)

# ファイル設定
LOG_FILE = "ai_trade_memory_risk_managed.csv" # 学習用メモリ（AIの脳）
REAL_TRADE_LOG_FILE = "real_trade_record.csv" # 実戦用ログ（あなたの記録）

MODEL_NAME = 'models/gemini-3-pro-preview' # 最新モデル推奨 (または gemini-2.0-pro-exp)
TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 11


# 監視リスト (スナイパー仕様・厳選80銘柄)
WATCH_LIST = [
    # --- 🏆 エース級 (高収益・相性良) ---
    "6146.T", "8035.T", "9983.T", "7741.T", "6857.T", "7012.T", "6367.T", "7832.T",
    "1801.T", "9766.T", "2801.T", "4063.T", "4543.T", "4911.T", "4507.T",

    # --- 🛡️ 安定・ディフェンシブ (低ボラ・堅実) ---
    "9432.T", "9433.T", "9434.T", "4503.T", "4502.T", "2502.T", "2503.T", "2802.T",
    "4901.T", "1925.T", "1928.T", "1802.T", "1803.T", "1812.T", "9020.T", "9021.T",
    "9532.T", "9735.T", "9613.T",

    # --- 💰 金融・銀行 (金利メリット・トレンド良) ---
    "8306.T", "8316.T", "8411.T", "8308.T", "8309.T", "8331.T", "8354.T", "8766.T",
    "8725.T", "8591.T", "8593.T", "8604.T", "8473.T", "8630.T", "8697.T",

    # --- 🏢 商社・卸売 (割安・高配当) ---
    "8058.T", "8031.T", "8001.T", "8002.T", "8015.T", "2768.T", "8053.T", "7459.T",
    "8088.T", "9962.T", "3092.T", "3382.T",

    # --- 🏭 重厚長大・自動車 (円安恩恵) ---
    "7011.T", "7013.T", "6301.T", "7203.T", "7267.T", "7269.T", "7270.T", "7201.T",
    "5401.T", "5411.T", "5713.T", "1605.T", "5020.T",

    # --- 📦 その他・機械 (選抜) ---
    "6501.T", "6503.T", "6305.T", "6326.T", "6383.T", "6471.T", "6473.T", "7751.T"
]

plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 1. データ取得 & テクニカル計算
# ==========================================
def download_data_safe(ticker, period="6mo", interval="1d", retries=3):
    wait = 2
    for _ in range(retries):
        try:
            # yfinanceのログ抑制
            import logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: raise ValueError("Empty Data")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except:
            time.sleep(wait); wait *= 2
    return None

def get_macro_data():
    """主要指数をリアルタイム取得して整形"""
    tickers = {
        "^N225": "日経平均", "JPY=X": "ドル円", "^GSPC": "米S&P500", 
        "^TNX": "米10年債利回り", "^VIX": "VIX(恐怖指数)"
    }
    report = "【🌎 マクロ環境】\n"
    try:
        data = yf.download(list(tickers.keys()), period="5d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close']
        else:
            df_close = data['Close'] if 'Close' in data else data

        for symbol, name in tickers.items():
            try:
                series = df_close[symbol].dropna()
                if len(series) < 2: continue
                current = float(series.iloc[-1])
                prev = float(series.iloc[-2])
                change = (current - prev) / prev * 100
                icon = "↗️" if change > 0 else "↘️"
                val_str = f"{current:.2f}"
                report += f"- {name}: {val_str} ({icon} {change:+.2f}%)\n"
            except: pass
    except:
        return "【マクロ環境】取得エラー"
    return report.strip()

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
        except:
            return "【ファンダメンタルズ】取得不可"
            
        if not info: return "【ファンダメンタルズ】データなし"

        name = info.get("longName", ticker)
        sector = info.get("sector", "-")
        per = info.get("trailingPE", "-")
        pbr = info.get("priceToBook", "-")
        roe = info.get("returnOnEquity", "-")
        
        per_str = f"{per:.1f}倍" if isinstance(per, (int, float)) else "-"
        pbr_str = f"{pbr:.2f}倍" if isinstance(pbr, (int, float)) else "-"
        roe_str = f"{roe*100:.1f}%" if isinstance(roe, (int, float)) else "-"

        return f"【ファンダメンタルズ】\n- {name} ({sector})\n- PER: {per_str}, PBR: {pbr_str}, ROE: {roe_str}"
    except:
        return "【ファンダメンタルズ】取得エラー"

def get_weekly_trend(ticker):
    """週足トレンド判定"""
    try:
        df = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 26: return "不明"
        
        sma13 = df['Close'].rolling(13).mean().iloc[-1]
        sma26 = df['Close'].rolling(26).mean().iloc[-1]
        price = float(df['Close'].iloc[-1])
        
        if price > sma13 > sma26: return "上昇 📈 (強)"
        elif price > sma13: return "上昇 ↗️ (短)"
        elif price < sma13 < sma26: return "下降 📉 (弱)"
        else: return "レンジ ➡️"
    except: return "取得エラー"

def get_latest_news(ticker):
    # 簡易版ニュース取得 (Google News RSS)
    try:
        q = urllib.parse.quote(f"{ticker} 株価 ニュース")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        feed = feedparser.parse(url)
        if not feed.entries: return "特になし"
        return "\n".join([f"・{e.title}" for e in feed.entries[:2]])
    except: return "取得エラー"

def get_earnings_date(ticker):
    """決算発表日を取得する（取得できない場合は'-'）"""
    try:
        stock = yf.Ticker(ticker)
        # 次回の決算日を取得
        calendar = stock.calendar
        if calendar and 'Earnings Date' in calendar:
            # 複数の日付候補がある場合は最初の日付を取得
            earnings_date = calendar['Earnings Date'][0]
            if isinstance(earnings_date, (datetime.date, datetime.datetime)):
                return earnings_date.strftime('%Y-%m-%d')
        # 代替手段: earnings_datesプロパティ
        dates = stock.earnings_dates
        if dates is not None and not dates.empty:
            # 未来の日付を探す
            future_dates = dates[dates.index > datetime.datetime.now()]
            if not future_dates.empty:
                return future_dates.index[-1].strftime('%Y-%m-%d')
    except:
        pass
    return "-"

def calculate_metrics_enhanced(df):
    if len(df) < 25: return None 
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

    # BB幅
    std = df['Close'].rolling(20).std().iloc[-1]
    bb_width = (4 * std) / df['Close'].rolling(20).mean().iloc[-1] * 100

    # 出来高倍率
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    volume_ratio = float(curr['Volume']) / vol_ma5 if vol_ma5 > 0 else 1.0

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
        'volume_ratio': volume_ratio,
        'rsi_9': rsi_9
    }

# ==========================================
# 2. CBRメモリシステム
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
            self.df = pd.read_csv(self.csv_path, on_bad_lines='skip')
            rename_map = {
                'date': 'Date', 'ticker': 'Ticker', 'action': 'Action', 
                'result': 'result', 'reason': 'Reason', 
                'confidence': 'Confidence',
                'stop_loss_price': 'stop_loss_price', 
                'stop_loss_reason': 'stop_loss_reason' 
            }
            self.df.columns = [rename_map.get(col.lower(), col) for col in self.df.columns]
            
            if len(self.df) < 5: return

            for col in self.feature_cols:
                if col not in self.df.columns: self.df[col] = 0.0

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
        
        text = f"【システム検索: 類似過去事例】\n"
        for idx in indices[0]:
            row = self.df.iloc[idx]
            res = str(row.get('result', ''))
            icon = "WIN ⭕" if res=='WIN' else "LOSS ❌" if res=='LOSS' else "➖"
            date = str(row.get('Date', ''))
            ticker = str(row.get('Ticker', ''))
            text += f"● {date} {ticker} -> {icon}\n"
        return text

# ==========================================
# 3. AI分析
# ==========================================
def create_chart_image(df, name):
    data = df.tail(100).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(data.index, data['Close'], color='black', label='Close')
    ax1.plot(data.index, data['SMA25'], color='orange', label='SMA25')
    ax1.set_title(f"{name} Analysis")
    ax1.grid(True)
    ax2.plot(data.index, data['MACD'], color='red')
    ax2.bar(data.index, data['MACD']-data['Signal'], color='gray', alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
    return {"mime_type": "image/png", "data": buf.getvalue()}

def analyze_vision_agent(model_instance, chart, metrics, cbr_text, macro, news, fundamentals, weekly_trend, name):
    trend_dir = "上昇" if metrics['trend_momentum'] > 0 else "下降"
    
    vol_msg = ""
    if metrics['entry_volatility'] >= 3.0:
        vol_msg = "⚠️ 現在ボラティリティが極めて高い(3.0%以上)です。急落リスクがあるため、新規BUYは慎重に判断してください。"

    prompt = f"""
### CONTEXT (入力データ)
対象銘柄: {name}
1. マクロ環境 (地合い):
   {macro if 'macro' in locals() else 'なし'}
   ※ VIX指数や指数全体のトレンドに注目せよ。

2. テクニカル指標:
   - 週足トレンド: {weekly_trend if 'weekly_trend' in locals() else '不明'}
   - 日足トレンド: {trend_dir} (勢い: {metrics['trend_momentum']:.2f})
   - SMA25乖離率: {metrics['sma25_dev']:.2f}% (プラス＝SMAより上)
   - ボラティリティ: {metrics['entry_volatility']:.2f}%
   - BB幅(スクイーズ度): {metrics['bb_width']:.2f}%
   - 出来高倍率: {metrics['volume_ratio']:.2f}倍
   - RSI(9): {metrics['rsi_9']:.1f}

3. ファンダメンタルズ:
   {fundamentals if 'fundamentals' in locals() else 'なし'}

{cbr_text}

### TASK (タスク)
あなたは百戦錬磨のファンドマネージャーです。
提供されたデータに基づき、**「確率的優位性」**が最も高いアクション（BUY, HOLD, SELL）を選択してください。

### CONSTRAINTS & RULES (厳格な売買ルール)

**1. ボラティリティ判定 (リスク管理):**
   - **< 2.0%**: [安全圏] 理想的なエントリー環境。
   - **2.0% 〜 2.99%**: [警戒圏] 「強い上昇モメンタム」かつ「出来高倍率 > 1.0」の場合のみ、リスク許容のうえBUY可。
   - **>= 3.0%**: [危険域] **新規BUYは絶対禁止**。保有ポジションは縮小・撤退を推奨。

**2. BUY (新規買い) の条件 - 以下の [パターンA] か [パターンB] に合致する場合のみ:**
   *前提: 価格がSMA25の上にあり、ボラティリティが3.0%未満であること。*
   
   - **[パターンA: ブレイクアウト] (攻め)**
     - BB幅が狭い状態(<15%)から拡大傾向にある。
     - **出来高倍率が 1.2倍以上** に急増している（資金流入）。
     - RSIは 50〜70 の範囲（勢いがある）。
     
   - **[パターンB: 押し目買い] (守り)**
     - 上昇トレンドが継続中（週足・日足ともに上向き）。
     - 一時的な調整で、RSIが **40〜55** まで低下している。
     - SMA25付近で下げ止まりの兆候がある。

**3. SELL (利益確定・損切り) の条件:**
   - **トレンド崩壊 (損切り):** 価格がSMA25を明確に下回った（終値ベース）。
   - **クライマックス (利確):** 短期間で急騰し、RSIが **85以上** に達した、またはSMA25乖離率が **+10%以上** に開いた（過熱）。
   - **パニック (撤退):** ボラティリティが **3.0%以上** に急拡大し、相場がコントロール不能になった。

**4. HOLD (様子見) の条件:**
   - 明確な「サイン」が出ていない中間領域。
   - 地合い（マクロ）が暴落中で、個別銘柄の買いが危険な場合。
   - 迷う場合は常にHOLD（ノーポジションは最強のポジション）。

### SELF-CORRECTION (自己検証)
- 出力する前に確認せよ: 「ボラティリティが高いのにBUYしていないか？」「トレンドが下向なのにBUYしていないか？」
- ルール違反がある場合は、アクションを "HOLD" に修正すること。

=== 出力 (JSONのみ) ===
{{
  "action": "BUY" | "HOLD" | "SELL",
  "confidence": <int 0-100>,
  "stop_loss_price": <float> (HOLD/SELLの場合は0),
  "stop_loss_reason": "理由(30文字以内)",
  "target_price": <float> (BUYの場合の利確目標。HOLD/SELLなら0),
  "reason": "理由(100文字以内)"
}}
"""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        response = model_instance.generate_content(
            [prompt, chart],
            safety_settings=safety_settings
        )
        
        if not response.parts:
            finish_reason = "Unknown"
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
            
            print(f"⚠️ AI Blocked: Reason={finish_reason}")
            return {"action": "HOLD", "confidence": 0, "reason": "AI生成がブロックされました", "stop_loss_price": 0}

        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except Exception as e:
        print(f"\n⚠️ AI ERROR: {e}") 
        return {"action": "HOLD", "confidence": 0, "reason": f"API Error: {e}", "stop_loss_price": 0}
    
def send_discord_notify(message):
    if not webhook_url: return
    
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    filename = f"AI_Report_{today_str}.txt"
    
    try:
        files = {
            "file": (filename, message.encode('utf-8'))
        }
        payload = {
            "content": f"📊 **本日のAI市場監視レポート ({today_str})**\n詳細は添付ファイルを確認してください。"
        }
        requests.post(webhook_url, data=payload, files=files)
        print("✅ Discord通知送信 (ファイル添付)")
    except Exception as e:
        print(f"⚠️ Discord送信エラー: {e}")

# ==========================================
# 5. メイン実行 (実戦監視・全株価記録版)
# ==========================================
if __name__ == "__main__":
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"=== AI市場監視システム ({today}) ===")
    
    WATCH_LIST = sorted(list(set(WATCH_LIST)))
    
    try:
        model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"Error: {e}"); exit()

    cbr = CaseBasedMemory(LOG_FILE)
    macro = get_macro_data()
    print(macro)
    
    report_message = f"**📊 AI市場監視レポート ({today})**\n\n{macro}\n"
    buy_list = []
    all_stock_prices = []
    
    SAVE_TARGETS = [
        {"path": LOG_FILE, "name": "学習メモリ"},
        {"path": REAL_TRADE_LOG_FILE, "name": "実戦ログ"}
    ]

    now_utc = datetime.datetime.utcnow()
    now_jst = now_utc + datetime.timedelta(hours=9)
    current_hour_jst = now_jst.hour
    
    is_closing_time = (current_hour_jst >= 15)
    
    if is_closing_time:
        print(f"🕒 現在 {current_hour_jst}時: 市場終了後のため「記録モード」で実行します。")
    else:
        print(f"🕒 現在 {current_hour_jst}時: 市場稼働中のため「監視モード（記録なし）」で実行します。")

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
        
        earnings_date = get_earnings_date(tic)
        
        current_price = metrics['price']
        try:
            prev_close = df.iloc[-2]['Close']
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            earnings_mark = ""
            if earnings_date != "-":
                e_date = datetime.datetime.strptime(earnings_date, '%Y-%m-%d')
                days_to_earnings = (e_date - datetime.datetime.now()).days
                if 0 <= days_to_earnings <= 14:
                    earnings_mark = f" ⚠️決算:{earnings_date}"
                else:
                    earnings_mark = f" (決算:{earnings_date})"
            
            price_str = f"• {name:<8}: {current_price:7,.0f}円 ({change:+5,.0f} / {change_pct:+5.2f}%){earnings_mark}"
        except:
            price_str = f"• {name:<8}: {current_price:7,.0f}円"
        
        all_stock_prices.append(price_str)
        
        cbr_text = cbr.search_similar_cases(metrics)
        chart = create_chart_image(df, name)
        news = get_latest_news(name)
        fundamentals = get_fundamentals(name)
        weekly_trend = get_weekly_trend(name)
        
        res = analyze_vision_agent(model_instance, chart, metrics, cbr_text, macro, news, fundamentals, weekly_trend, name)
              
        action = res.get('action', 'HOLD')
        conf = res.get('confidence', 0)
        sl_price_raw = res.get('stop_loss_price', 0)
        tp_price_raw = res.get('target_price', 0)
        
        try: sl_price = float(sl_price_raw)
        except: sl_price = 0.0
        try: tp_price = float(tp_price_raw)
        except: tp_price = 0.0
        
        item = {
            "Date": today, "Ticker": tic, "Timeframe": TIMEFRAME, 
            "Action": action, "result": "", 
            "Reason": res.get('reason', 'None'), 
            "Confidence": conf,
            "stop_loss_price": sl_price, 
            "stop_loss_reason": res.get('stop_loss_reason', '-'),
            "Price": metrics['price'],
            "sma25_dev": metrics['sma25_dev'], 
            "trend_momentum": metrics['trend_momentum'],
            "macd_power": metrics['macd_power'],
            "entry_volatility": metrics['entry_volatility'],
            "profit_loss": 0
        }
        
        csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "stop_loss_reason", "Price", 
            "sma25_dev", "trend_momentum", "macd_power", "entry_volatility", "profit_loss"
        ]
        
        df_new = pd.DataFrame([item])
        for col in csv_columns:
            if col not in df_new.columns: df_new[col] = None
        df_new = df_new[csv_columns]

        if is_closing_time:
            for target in SAVE_TARGETS:
                path = target["path"]
                try:
                    if os.path.exists(path):
                        try:
                            df_exist = pd.read_csv(path, on_bad_lines='skip')
                            is_duplicate = ((df_exist['Date'] == today) & (df_exist['Ticker'] == tic)).any()
                            if not is_duplicate:
                                df_new.to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')
                                print(f"📝", end=" ")
                        except:
                            df_new.to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')
                    else:
                        df_new.to_csv(path, index=False, encoding='utf-8-sig')
                        print(f"🆕", end=" ")
                except Exception as e:
                    print(f"x", end=" ")

        action_icon = "🔴" if action == "BUY" else "🔵" if action == "SELL" else "🟡"
        sl_str = f"(SL: {sl_price:.0f})" if action == "BUY" and sl_price > 0 else ""
        print(f"-> {action_icon} {conf}% {sl_str}")

        if action == "BUY":
            sl_str = f"{sl_price:.0f}" if sl_price > 0 else "-"
            tp_str = f"{tp_price:.0f}" if tp_price > 0 else "-"
            
            earnings_warning = ""
            if earnings_date != "-" and "⚠️" in price_str:
                 earnings_warning = f"\n⚠️ **注意**: 決算発表({earnings_date})が近いです。持ち越しリスクを考慮してください。"

            msg = (
                f"🔴 **BUY {name}**: {metrics['price']:.0f}円\n"
                f"🎯 **利確目標 (TP)**: {tp_str}円\n"
                f"🛡️ **鉄の掟**: 購入と同時に **{sl_str}円** に「逆指値(損切り)」を入れてください。\n"
                f"{earnings_warning}\n"
                f"> 理由: {res.get('reason')}"
            )
            buy_list.append(msg)
            
        elif action == "SELL":
            msg = f"🔵 **SELL (決済) {name}**: {metrics['price']:.0f}円\n> 理由: {res.get('reason')}"
            buy_list.append(msg)
            
        time.sleep(5)

    if buy_list:
        report_message += "\n\n🚀 **新規BUY/SELL推奨**\n" + "\n\n".join(buy_list)
    else:
        report_message += "\n\n💤 本日は「BUY/SELL」推奨銘柄はありませんでした。"
    
    if not is_closing_time:
        report_message += "\n\n(※市場稼働中のため、CSVへの記録はスキップされました)"

    if all_stock_prices:
        report_message += "\n\n" + "="*30 + "\n📉 **全監視銘柄 株価一覧**\n" + "="*30 + "\n"
        report_message += "\n".join(all_stock_prices)

    send_discord_notify(report_message)

    try:
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        file_path = os.path.join(report_dir, "latest_report.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_message)
    except: pass
