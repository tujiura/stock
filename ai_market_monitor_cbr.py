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

genai.configure(api_key=GOOGLE_API_KEY, transport="rest")

# ★ファイル設定 (攻撃型V7)
LOG_FILE = "ai_trade_memory_aggressive_v7.csv" 
REAL_TRADE_LOG_FILE = "real_trade_record_aggressive.csv" 
MODEL_NAME = 'models/gemini-3-pro-preview'

TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15

# ★監視リスト (精鋭銘柄)
WATCH_LIST = [
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
            if len(df) < 60: return None
            return df
        except:
            time.sleep(wait); wait *= 2
    return None

def get_macro_data():
    tickers = {"^N225": "日経平均", "JPY=X": "ドル円", "^GSPC": "米S&P500"}
    report = "【🌎 マクロ環境】\n"
    try:
        data = yf.download(list(tickers.keys()), period="5d", progress=False, auto_adjust=True)
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

def get_latest_news(ticker):
    try:
        q = urllib.parse.quote(f"{ticker} 株価 材料")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        feed = feedparser.parse(url)
        return feed.entries[0].title if feed.entries else "特になし"
    except: return "取得エラー"

def get_weekly_trend(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1wk", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 26: return "不明"
        price = float(df['Close'].iloc[-1])
        sma13 = df['Close'].rolling(13).mean().iloc[-1]
        sma26 = df['Close'].rolling(26).mean().iloc[-1]
        if price > sma13 > sma26: return "上昇(強) 📈"
        elif price > sma13: return "上昇 ↗️"
        elif price < sma13 < sma26: return "下降 📉"
        else: return "レンジ ➡️"
    except: return "不明"

def get_fundamental_data(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        sector = info.get('sector', 'Unknown')
        margin_ratio = info.get('shortRatio', 0.0) 
        if margin_ratio is None: margin_ratio = 1.0
        
        days_to_earnings = 999
        try:
            cal = t.calendar
            if cal and 'Earnings Date' in cal:
                earnings_date = cal['Earnings Date'][0]
                edate = earnings_date.date()
                today = datetime.datetime.now().date()
                days_to_earnings = (edate - today).days
        except: pass

        return {
            "sector": sector,
            "margin_ratio": margin_ratio,
            "days_to_earnings": days_to_earnings
        }
    except:
        return {"sector": "-", "margin_ratio": 0, "days_to_earnings": 999}

# ==========================================
# 2. 指標計算 (攻撃型V7仕様: MACD/雲)
# ==========================================
def calculate_metrics_aggressive(df, market_df=None):
    if len(df) < 60: return None
    curr = df.iloc[-1]
    price = float(curr['Close'])
    
    # 1. DMI/ADX
    high = df['High']; low = df['Low']; close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    tr_smooth = tr.rolling(14).mean()
    plus_dm_smooth = plus_dm.rolling(14).mean()
    minus_dm_smooth = minus_dm.rolling(14).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_series = dx.rolling(14).mean()
    
    adx = adx_series.iloc[-1]
    prev_adx = adx_series.iloc[-2] if len(adx_series) > 1 else adx

    # 2. MA乖離率
    sma25 = df['Close'].rolling(25).mean().iloc[-1]
    ma_deviation = ((price / sma25) - 1) * 100 

    # 3. 抵抗線
    recent_high = df['High'].tail(60).max()
    dist_to_res = 0
    if recent_high > 0:
        dist_to_res = ((price - recent_high) / recent_high) * 100

    # 4. Relative Strength
    rs_rating = 0
    if market_df is not None and len(market_df) > 25:
        try:
            stock_perf = (price / df['Close'].iloc[-21]) - 1
            market_perf = (market_df['Close'].iloc[-1] / market_df['Close'].iloc[-21]) - 1
            rs_rating = (stock_perf - market_perf) * 100 
        except: pass

    # 5. ボリンジャーバンド & 出来高
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    bb_width = ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20 * 100
    prev_width = bb_width.iloc[-6] if bb_width.iloc[-6] > 0 else 0.1
    expansion_rate = bb_width.iloc[-1] / prev_width

    vol_ma20 = df['Volume'].rolling(20).mean()
    current_vol = float(curr['Volume'])
    vol_ratio = current_vol / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 0
    trading_value_oku = (price * current_vol) / 100000000 

    # 出来高履歴
    vol_history = []
    for i in range(4, -1, -1):
        if len(df)-1-i >= 0:
            row = df.iloc[len(df)-1-i]
            vr = float(row['Volume']) / float(vol_ma20.iloc[len(df)-1-i]) if float(vol_ma20.iloc[len(df)-1-i]) > 0 else 0
            vol_history.append(f"{vr:.1f}")
    vol_history_str = "->".join(vol_history)

    # 6. MACD (V7)
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    prev_hist = macd_hist.iloc[-2]
    
    # 7. 一目均衡表 (雲) (V7)
    high9 = high.rolling(9).max(); low9 = low.rolling(9).min()
    tenkan = (high9 + low9) / 2
    high26 = high.rolling(26).max(); low26 = low.rolling(26).min()
    kijun = (high26 + low26) / 2
    
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    high52 = high.rolling(52).max(); low52 = low.rolling(52).min()
    senkou_b = ((high52 + low52) / 2).shift(26)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1).iloc[-1]
    
    price_vs_cloud = "Above" if price > cloud_top else "Below"

    # ローソク足
    open_p = float(curr['Open']); close_p = float(curr['Close']); high_p = float(curr['High']); low_p = float(curr['Low'])
    body_top = max(open_p, close_p)
    upper_shadow = high_p - body_top
    total_range = high_p - low_p
    shadow_ratio = upper_shadow / total_range if total_range > 0 else 0
    candle_shape = "Good" if shadow_ratio < 0.3 else "Bad (Long Upper Shadow)"

    atr = tr.rolling(14).mean().iloc[-1]

    return {
        'price': price,
        'resistance_price': recent_high,
        'dist_to_res': dist_to_res,
        'ma_deviation': ma_deviation,
        'adx': adx,
        'prev_adx': prev_adx,
        'plus_di': plus_di.iloc[-1],
        'minus_di': minus_di.iloc[-1],
        'rs_rating': rs_rating,
        'trading_value': trading_value_oku,
        'vol_ratio': vol_ratio,
        'vol_history': vol_history_str,
        'expansion_rate': expansion_rate,
        'atr_value': atr,
        'macd_hist': macd_hist.iloc[-1],
        'macd_trend': "Expanding" if abs(macd_hist.iloc[-1]) > abs(prev_hist) else "Shrinking",
        'price_vs_cloud': price_vs_cloud,
        'candle_shape': candle_shape
    }

def check_breakout_rules(metrics):
    """攻撃型V7フィルタリング"""
    
    # 1. 流動性
    if metrics['trading_value'] < 5.0:
        return {"action": "HOLD", "reason": f"【対象外】流動性不足 ({metrics['trading_value']:.1f}億円)"}

    # 2. トレンド強度 (ADX)
    if metrics['adx'] < 20:
        return {"action": "HOLD", "reason": f"【対象外】トレンドレス (ADX {metrics['adx']:.1f})"}
    
    # 3. 過熱感 (ADX)
    if metrics['adx'] > 55:
        return {"action": "HOLD", "reason": f"【対象外】トレンド過熱 (ADX {metrics['adx']:.1f})"}

    # 4. 魔の乖離ゾーン
    ma_dev = metrics['ma_deviation']
    if 10.0 <= ma_dev <= 15.0:
        return {"action": "HOLD", "reason": f"【対象外】魔の乖離ゾーン ({ma_dev:.1f}%) 調整警戒"}

    # 5. ★V7追加: 雲の下
    if metrics['price_vs_cloud'] == "Below":
        return {"action": "HOLD", "reason": "【対象外】雲の下 (トレンド弱)"}

    return None

# ==========================================
# 3. CBRメモリシステム (V7対応)
# ==========================================
class CaseBasedMemory:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        self.feature_cols = ['adx', 'prev_adx', 'ma_deviation', 'rs_rating', 'vol_ratio', 'expansion_rate', 'dist_to_res']
        
        # 保存カラム
        self.csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "target_price", 
            "Actual_High", "Target_Diff", "Target_Reach",
            "Price", "adx", "prev_adx", "ma_deviation", "rs_rating", 
            "vol_ratio", "expansion_rate", "dist_to_res", 
            "days_to_earnings", "margin_ratio", "profit_rate"
        ]
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path)
            for col in self.csv_columns:
                if col not in self.df.columns: self.df[col] = 0.0
        except Exception: return

        try:
            self.df.columns = [c.strip() for c in self.df.columns]
            valid_df = self.df[self.df['result'].isin(['WIN', 'LOSS'])].copy()
            if len(valid_df) > 5:
                features = valid_df[self.feature_cols].fillna(0)
                self.features_normalized = self.scaler.fit_transform(features)
                self.valid_df_for_knn = valid_df 
                global CBR_NEIGHBORS_COUNT
                self.knn = NearestNeighbors(n_neighbors=min(CBR_NEIGHBORS_COUNT, len(valid_df)), metric='euclidean')
                self.knn.fit(self.features_normalized)
        except Exception as e:
            print(f"Memory Init Error: {e}")

    def search_similar_cases(self, current_metrics):
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
# 4. AI判定
# ==========================================
def create_chart_image(df, name):
    data = df.tail(80).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    ax1.plot(data.index, data['Close'], color='black', label='Close')
    ax1.plot(data.index, sma20 + 2*std20, color='green', alpha=0.5, linestyle='--', label='+2σ')
    ax1.plot(data.index, sma20 - 2*std20, color='green', alpha=0.5, linestyle='--', label='-2σ')
    
    # 雲 (簡易計算で表示)
    # 実運用では正確な雲を描画すべきだが、画像用としては直近の傾向が見えればOK
    
    ax1.set_title(f"{name} V7 Sniper")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
    return {"mime_type": "image/png", "data": buf.getvalue()}

def ai_decision_maker(model, chart_bytes, metrics, cbr_text, macro, news, weekly, ticker, fund_data):
    
    sector_trend_desc = f"{fund_data['sector']} (RS: {metrics['rs_rating']:.1f})"
    
    # ★V7 スナイパー型プロンプト (Vol History入り)
    prompt = f"""
# ROLE
あなたは世界最高峰の「クオンツ・テクニカルアナリスト」であり、高精度のトレンドフォロー戦略を実行するAIエージェントです。
あなたの使命は、提供されたデータに基づき、感情を排して数学的かつ論理的にトレード判断を下すことです。
特に「ダマシ（False Breakout）」を回避し、優位性（Edge）のあるエントリーポイントのみを厳選します。

# CONTEXT & OBJECTIVE
ユーザーは、以下の銘柄について「買い（BUY）」か「見送り（HOLD）」かの二択の判断を求めています。
曖昧な状況や、リスクリワード比が悪い局面では、資産を守るために迷わず「HOLD」を選択する必要があります。

# INPUT DATA
## 対象銘柄
* Ticker: {ticker}
* Current Price: {metrics['price']:.0f} JPY

## 1. Technical Indicators (Quantitative)
* **Trend Strength ($ADX$):** {metrics['adx']:.1f} (Threshold: $\ge 25$)
* **Directional Movement:** $+DI$ = {metrics['plus_di']:.1f} vs $-DI$ = {metrics['minus_di']:.1f}
* **Volatility Expansion:** {metrics['expansion_rate']:.2f}x (Squeeze $\\to$ Expansion is ideal)
* **Volume Ratio:** {metrics['vol_ratio']:.2f}x (Trend: {metrics['vol_history']})
* **MACD:** Histogram = {metrics['macd_hist']:.2f} ({metrics['macd_trend']})
* **Ichimoku Cloud:** Price is **{metrics['price_vs_cloud']}** the Cloud.
* **Resistance Distance:** {metrics['dist_to_res']:.1f}%

## 2. Qualitative Factors (Price Action & Environment)
* **Candle Shape:** {metrics['candle_shape']}
* **Sector Trend:** {sector_trend_desc}
* **Weekly Trend:** {weekly}
* **Earnings Date:** {fund_data['days_to_earnings']} days later

## 3. External Factors
{macro}
{news}
{cbr_text}

# STRICT EVALUATION RULES (AND/OR LOGIC)
判断を下す際は、以下の論理ゲートを厳守してください。

1.  **Trend Filter (Must Pass):**
    * 価格が「一目均衡表の雲」の上にあること（$Price > Cloud$）。これが満たされない場合、**即座にHOLD**とする。
    * トレンドの強さ（$ADX$）が基準を満たしている、または勢いが増していること。

2.  **Momentum Trigger:**
    * MACDヒストグラムが拡大傾向、かつプラス圏にあることが望ましい。
    * 出来高（Volume）が増加傾向にあり、値動きを裏付けていること。

3.  **Risk Check:**
    * 長い上ヒゲ（Selling Pressure）が出現していないか？
    * 決算発表が直近（3日以内など）に迫っていないか？
    * 抵抗線（Resistance）が極端に近くないか？

# OUTPUT FORMAT
回答は**JSON形式のみ**で出力してください。Markdownのコードブロックや説明文は一切不要です。

{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100 (Integer),
  "stop_loss": {metrics['price'] * 0.95:.0f},  // 例: 現在価格から計算、またはロジックで算出
  "target_price": {metrics['price'] * 1.10:.0f}, // 例: リスクリワード1:2などを想定
  "reason": "判断の決定的な理由を簡潔に記述 (Max 60 chars)"
}}

# FINAL INSTRUCTION
情け容赦ないプロの視点で分析し、JSONデータのみを返してください。
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
        payload = {"content": f"🚀 **AI攻撃型トレードレポート V7 ({now_str})**\n{message[:1500]}"}
        files = {}
        if filename:
            files["file"] = (f"Aggressive_{now_str.replace(':','-')}.txt", message.encode('utf-8'))
        requests.post(webhook_url, data=payload, files=files if filename else None)
        print("✅ Discord通知送信")
    except Exception as e:
        print(f"⚠️ Discord送信エラー: {e}")

# ==========================================
# 5. メイン実行
# ==========================================
if __name__ == "__main__":
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"=== AI市場監視システム [AGGRESSIVE MODE V7] (Sniper) ===")
    
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e: print(f"Error: {e}"); exit()

    memory = CaseBasedMemory(LOG_FILE)
    macro = get_macro_data()
    print(macro)

    print("市場データ(日経平均)を取得中...")
    market_df = download_data_safe("^N225")
    
    report_message = f"**🚀 AI攻撃型トレードレポート V7 ({today})**\n\n{macro}\n"
    buy_list = []
    
    SAVE_TARGETS = [
        {"path": LOG_FILE, "name": "学習メモリ"},
        {"path": REAL_TRADE_LOG_FILE, "name": "実戦ログ"}
    ]

    current_hour_jst = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).hour
    is_closing_time = (current_hour_jst >= 15)
    
    WATCH_LIST = sorted(list(set(WATCH_LIST)))

    for i, tic in enumerate(WATCH_LIST, 1):
        print(f"[{i}/{len(WATCH_LIST)}] {tic}... ", end="", flush=True)
        
        df = download_data_safe(tic)
        if df is None: print("Skip(NoData)"); continue
        
        # 指標計算 (V7)
        metrics = calculate_metrics_aggressive(df, market_df)
        if metrics is None: print("Skip(Calc)"); continue
        
        # 鉄の掟 (V7)
        iron_res = check_breakout_rules(metrics)
        if iron_res:
            print(f"⏹️ {iron_res['reason']}")
            continue

        fund_data = get_fundamental_data(tic)
        cbr_text = memory.search_similar_cases(metrics)
        chart = create_chart_image(df, tic)
        news = get_latest_news(tic)
        weekly = get_weekly_trend(tic)
        
        # AI判定
        res = ai_decision_maker(model_instance, chart, metrics, cbr_text, macro, news, weekly, tic, fund_data)
        
        action = res.get('action', 'HOLD')
        conf = res.get('confidence', 0)
        
        ai_stop = res.get('stop_loss', 0)
        ai_target = res.get('target_price', 0)
        try: ai_stop = int(ai_stop); ai_target = int(ai_target)
        except: ai_stop = 0; ai_target = 0
        
        # 損切り: ATR x 1.8 (ホームラン狙いの初期設定)
        stop_loss_price = ai_stop if ai_stop > 0 else (metrics['price'] - metrics['atr_value'] * 1.8)
        
        # CSVデータ作成 (V7仕様)
        item = {
            "Date": today, "Ticker": tic, "Timeframe": TIMEFRAME, 
            "Action": action, "result": "", "Reason": res.get('reason', 'None'), 
            "Confidence": conf, 
            "stop_loss_price": stop_loss_price, 
            "target_price": ai_target, 
            "Price": metrics['price'], 
            "adx": metrics['adx'], 
            "prev_adx": metrics['prev_adx'],
            "ma_deviation": metrics['ma_deviation'], 
            "rs_rating": metrics['rs_rating'], 
            "vol_ratio": metrics['vol_ratio'], 
            "expansion_rate": metrics['expansion_rate'],
            "dist_to_res": metrics['dist_to_res'], 
            "days_to_earnings": fund_data['days_to_earnings'], 
            "margin_ratio": fund_data['margin_ratio'],
            "profit_rate": 0.0 
        }
        
        if is_closing_time:
            df_new = pd.DataFrame([item])
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
            msg = (
                f"🔥 **BUY ALERT {tic}**: {metrics['price']:.0f}円\n"
                f"📊 **ADX**: {metrics['adx']:.1f} | **RS**: {metrics['rs_rating']:.1f}\n"
                f"🛡️ **損切り**: {stop_loss_price:.0f}円\n"
                f"🎯 **目標**: {ai_target}円\n"
                f"☁️ **雲判定**: {metrics['price_vs_cloud']}\n"
                f"> 理由: {res.get('reason')}"
            )
            buy_list.append(msg)
        time.sleep(2)

    if buy_list:
        report_message += "\n\n🔥 **強気シグナル (V7 Sniper)**\n" + "\n\n".join(buy_list)
    else:
        report_message += "\n\n💤 チャンスなし (フィルタ作動中)"

    send_discord_notify(report_message)