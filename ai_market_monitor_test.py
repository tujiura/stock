import yfinance as yf
import pandas as pd
import google.generativeai as genai
import json
import time
import datetime
import urllib.parse
import feedparser
import requests
import os
import io
import sys 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import re
import logging

# ---------------------------------------------------------
# ★環境設定
# ---------------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError: pass

# ==========================================
# ★設定エリア
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")

webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
genai.configure(api_key=GOOGLE_API_KEY)

# ファイル設定
LOG_FILE = "ai_trade_memory_risk_managed.csv" 
REAL_TRADE_LOG_FILE = "real_trade_record.csv" 
MODEL_NAME = 'models/gemini-2.0-flash' 

TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15

# ★資金設定
INITIAL_CAPITAL = 100000 # 運用資金 (10万円)
RISK_PER_TRADE = 0.02      # リスク許容率 (2%)
MAX_POSITIONS = 10        # 最大保有銘柄数

# 監視リスト
WATCH_LIST = [
    "6146.T", "8035.T", "9983.T", "7741.T", "6857.T", "7012.T", "6367.T", "7832.T",
    "1801.T", "9766.T", "2801.T", "4063.T", "4543.T", "4911.T", "4507.T",
    "9432.T", "9433.T", "9434.T", "4503.T", "4502.T", "2502.T", "2503.T", "2802.T",
    "4901.T", "1925.T", "1928.T", "1802.T", "1803.T", "1812.T", "9020.T", "9021.T",
    "9532.T", "9735.T", "9613.T",
    "8306.T", "8316.T", "8411.T", "8308.T", "8309.T", "8331.T", "8354.T", "8766.T",
    "8725.T", "8591.T", "8593.T", "8604.T", "8473.T", "8630.T", "8697.T",
    "8058.T", "8031.T", "8001.T", "8002.T", "8015.T", "2768.T", "8053.T", "7459.T",
    "8088.T", "9962.T", "3092.T", "3382.T",
    "7011.T", "7013.T", "6301.T", "7203.T", "7267.T", "7269.T", "7270.T", "7201.T",
    "5401.T", "5411.T", "5713.T", "1605.T", "5020.T",
    "6501.T", "6503.T", "6305.T", "6326.T", "6383.T", "6471.T", "6473.T", "7751.T"
]
WATCH_LIST = sorted(list(set(WATCH_LIST)))

plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 1. データ取得・計算関数
# ==========================================
def download_data_safe(ticker, period="1y", interval="1d", retries=3):
    for _ in range(retries):
        try:
            import logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            if len(df) < 30: return None
            return df
        except: time.sleep(2)
    return None

def get_macro_data():
    tickers = {"^N225": "日経平均", "JPY=X": "ドル円", "^GSPC": "米S&P500"}
    report = "【🌎 マクロ環境】\n"
    try:
        data = yf.download(list(tickers.keys()), period="5d", progress=False)
        if isinstance(data.columns, pd.MultiIndex): df_close = data['Close']
        else: df_close = data['Close'] if 'Close' in data else data
        for symbol, name in tickers.items():
            try:
                series = df_close[symbol].dropna()
                if len(series) < 2: continue
                cur = series.iloc[-1]
                chg = (cur - series.iloc[-2]) / series.iloc[-2] * 100
                icon = "↗️" if chg > 0 else "↘️"
                report += f"- {name}: {cur:.2f} ({icon} {chg:+.2f}%)\n"
            except: pass
    except: return "取得エラー"
    return report.strip()

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return f"PER:{info.get('trailingPE','-')} PBR:{info.get('priceToBook','-')} ROE:{info.get('returnOnEquity','-')}"
    except: return "データなし"

def get_latest_news(ticker):
    try:
        q = urllib.parse.quote(f"{ticker} 株価 ニュース")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        return feedparser.parse(url).entries[0].title if feedparser.parse(url).entries else "なし"
    except: return "エラー"

def get_earnings_date(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        if cal and 'Earnings Date' in cal:
            return cal['Earnings Date'][0].strftime('%Y-%m-%d')
    except: pass
    return "-"

def get_weekly_trend(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1wk", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 26: return "不明"
        price = df['Close'].iloc[-1]
        sma13 = df['Close'].rolling(13).mean().iloc[-1]
        sma26 = df['Close'].rolling(26).mean().iloc[-1]
        if price > sma13 > sma26: return "上昇 📈"
        elif price > sma13: return "上昇 ↗️"
        elif price < sma13 < sma26: return "下降 📉"
        return "レンジ ➡️"
    except: return "不明"

def calculate_metrics_enhanced(df):
    if len(df) < 26: return None
    curr = df.iloc[-1]
    price = float(curr['Close'])
    sma25 = float(curr['SMA25'])
    sma25_dev = ((price / sma25) - 1) * 100
    prev_sma25 = float(df['SMA25'].iloc[-6])
    trend_momentum = (sma25 - prev_sma25) / 5 / price * 1000
    macd = float(curr['MACD'])
    signal = float(curr['Signal'])
    macd_power = ((macd - signal) / price) * 10000
    atr = float(curr['ATR'])
    entry_volatility = (atr / price) * 100
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    rs = gain / loss
    rsi_9 = 100 - (100 / (1 + rs)).iloc[-1]
    
    return {
        'sma25_dev': sma25_dev, 'trend_momentum': trend_momentum,
        'macd_power': macd_power, 'entry_volatility': entry_volatility,
        'price': price, 'atr_value': atr, 'rsi_9': rsi_9
    }

def create_chart_image(df, name):
    data = df.tail(75).copy()
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

# ==========================================
# 2. ポートフォリオ管理クラス (ステートフル)
# ==========================================
class PortfolioManager:
    def __init__(self, log_file):
        self.log_file = log_file
        self.holdings = {} # {ticker: {entry_price, shares, sl_price, max_price, atr}}
        self.closed_profit = 0.0
        self.load_portfolio()

    def load_portfolio(self):
        """CSVから現在の保有状況を復元"""
        if not os.path.exists(self.log_file): return
        try:
            df = pd.read_csv(self.log_file, on_bad_lines='skip')
            df.columns = [c.strip().lower() for c in df.columns]
            
            if 'profit_loss' in df.columns:
                self.closed_profit = pd.to_numeric(df['profit_loss'], errors='coerce').fillna(0).sum()

            if 'result' in df.columns:
                # 決済されていないBUYレコードを抽出
                holding_df = df[ (df['action'] == 'BUY') & (df['result'].isna() | (df['result'] == '')) ]
                
                for _, row in holding_df.iterrows():
                    ticker = row['ticker']
                    try:
                        entry_price = float(row['price'])
                        # CSVに記録されたSLがあれば使う、なければ ATR x 2.0 (Volatilityから逆算)
                        vol = float(row['entry_volatility']) if pd.notna(row['entry_volatility']) else 2.0
                        atr = entry_price * (vol / 100)
                        
                        sl_price = float(row['stop_loss_price']) if pd.notna(row['stop_loss_price']) else entry_price - (atr * 2.0)
                        
                        # MaxPriceは記録がないため現在値などから推定（暫定的にEntryPrice）
                        # ※本格運用ではMaxPriceもCSVに記録すべきだが、ここでは簡易復元
                        
                        self.holdings[ticker] = {
                            'entry_price': entry_price,
                            'sl_price': sl_price,
                            'max_price': entry_price, 
                            'atr': atr,
                            'index': _ 
                        }
                    except: pass
        except Exception as e:
            print(f"Portfolio Load Error: {e}")

    def update_holding(self, ticker, current_price, current_high):
        """保有株のトレーリング更新ロジック (最強版: 3%遅延・2.5%建値ガード)"""
        if ticker not in self.holdings: return None
        
        pos = self.holdings[ticker]
        
        # 最高値更新
        if current_high > pos['max_price']:
            pos['max_price'] = current_high
        
        # 含み益率
        profit_pct = (pos['max_price'] - pos['entry_price']) / pos['entry_price']
        
        # --- 可変式トレーリングロジック ---
        # +3.0%までは初期SLで耐える (ノイズ対策)
        width = 999999 
        if profit_pct > 0.05: width = pos['atr'] * 0.5  # 鬼利確
        elif profit_pct > 0.03: width = pos['atr'] * 1.5 # 追従開始
        
        new_sl_trail = pos['max_price'] - width
        
        # 建値ガード (+2.5%で発動)
        new_sl_guard = 0
        if profit_pct > 0.025:
            new_sl_guard = pos['entry_price'] * 1.001 
            
        new_sl = max(new_sl_trail, new_sl_guard)
        
        if new_sl > pos['sl_price']:
            pos['sl_price'] = new_sl
            return True 
            
        return False 

    def get_current_cash(self):
        return INITIAL_CAPITAL + self.closed_profit

# ==========================================
# 3. エージェント定義
# ==========================================
def run_analyst(model, ticker, metrics, chart_bytes, similar_text, news, fundamentals, weekly):
    prompt = f"""
あなたは株式市場分析官です。データを客観的に分析してください。
対象: {ticker} | 現在値: {metrics['price']:.0f}円 | 週足: {weekly}
指標: Momentum {metrics['trend_momentum']:.2f}, SMA乖離 {metrics['sma25_dev']:.2f}%, Vol {metrics['entry_volatility']:.2f}%, RSI {metrics['rsi_9']:.1f}
類似局面: {similar_text}
{news} {fundamentals}
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
    try:
        response = model.generate_content(
            [prompt, {'mime_type': 'image/png', 'data': chart_bytes}],
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
        return response.text
    except: return "分析エラー"

def run_commander_batch(model, candidates_data, current_cash, current_portfolio_text):
    # 候補リストのテキスト化
    candidates_text = ""
    for c in candidates_data:
        risk_per_share = c['metrics']['atr_value'] * 2.0
        max_shares = int((current_cash * RISK_PER_TRADE) // risk_per_share) if risk_per_share > 0 else 0
        
        candidates_text += f"""
--- 候補: {c['ticker']} ---
現在値: {c['metrics']['price']:.0f}円
推奨最大株数: {max_shares}株
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
- 全体方針: 資産防衛最優先。無理に全額投資する必要はない。自信のある銘柄に絞る。

### 候補銘柄レポート
{candidates_text}

### 鉄の掟（厳守）
1. ボラティリティが高い銘柄、下降トレンドの銘柄は除外せよ。
2. アナリスト報告に不安要素がある場合は見送れ。
3. 資金内で買える範囲に収めること。
4. 既に保有している銘柄と似たような銘柄ばかり買わないこと（分散効果）。

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
      "stop_loss": 損切り価格 (現在値 - ATR*2.0を目安),
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

class MemorySystem:
    def __init__(self, csv_path):
        self.df = pd.DataFrame()
        if os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path)
                self.valid = self.df[self.df['result'].isin(['WIN', 'LOSS'])].copy()
                if len(self.valid) > 5:
                    cols = ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility', 'rsi_9']
                    self.features = self.valid[cols].fillna(0)
                    self.scaler = StandardScaler()
                    self.knn = NearestNeighbors(n_neighbors=10).fit(self.scaler.fit_transform(self.features))
            except: pass
    def get_similar(self, metrics):
        if not hasattr(self, 'knn'): return "データ不足"
        vec = [metrics[c] for c in ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility', 'rsi_9']]
        dists, indices = self.knn.kneighbors(self.scaler.transform(pd.DataFrame([vec], columns=['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility', 'rsi_9'])))
        wins = sum([1 for i in indices[0] if self.valid.iloc[i]['result'] == 'WIN'])
        return f"類似局面勝率: {wins/len(indices[0])*100:.1f}%"

def send_discord(msg, filename=None):
    if not webhook_url: return
    files = {"file": (f"Report.txt", msg.encode('utf-8'))} if filename else None
    requests.post(webhook_url, data={"content": msg[:1500]}, files=files)

# ==========================================
# 4. メイン実行
# ==========================================
if __name__ == "__main__":
    print(f"=== 🤖 AI市場監視システム (Analyst & Commander Batch) ===")
    
    # 初期化
    memory = MemorySystem(LOG_FILE)
    model = genai.GenerativeModel(MODEL_NAME)
    pm = PortfolioManager(REAL_TRADE_LOG_FILE)
    
    macro = get_macro_data()
    print(macro)
    
    current_cash = pm.get_current_cash()
    print(f"💰 有効資金: {current_cash:,.0f}円")
    print(f"📂 保有銘柄数: {len(pm.holdings)}")
    
    report_msg = f"**📊 AI市場監視レポート**\n資金: {current_cash:,.0f}円\n保有: {len(pm.holdings)}銘柄\n\n{macro}\n"
    alert_list = []
    
    is_closing_time = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).hour >= 15
    print(f"🕒 モード: {'記録(引け後)' if is_closing_time else '監視(ザラ場)'}")

    # --- Phase 1: 保有株の管理 (決済監視) ---
    if pm.holdings:
        report_msg += "\n🛑 **保有株 状況確認**\n"
        print("\n--- 保有株チェック ---")
        
        for ticker, pos in pm.holdings.items():
            df = download_data_safe(ticker)
            if df is None: continue
            
            curr_price = df['Close'].iloc[-1]
            curr_high = df['High'].iloc[-1]
            
            # トレーリング更新
            updated = pm.update_holding(ticker, curr_price, curr_high)
            sl_price = pos['sl_price']
            
            # 状態判定
            status_icon = "✅"
            warning_text = ""
            
            # 撤退ライン割れ判定
            if curr_price < sl_price:
                status_icon = "⚠️"
                warning_text = f"\n🚨 **警告**: 現在値({curr_price:.0f})が撤退ライン({sl_price:.0f})を下回っています。\n➡️ **終値で確定したら翌朝成行で決済してください。**"
            
            print(f"{ticker}: {curr_price:.0f}円 (撤退: {sl_price:.0f}円) {status_icon}")
            
            msg = f"{status_icon} **{ticker}**: {curr_price:,.0f}円\n   🛡️ 撤退ライン: **{sl_price:,.0f}円**\n   (取得: {pos['entry_price']:,.0f}円){warning_text}"
            alert_list.append(msg)
            
            # CSVのSL価格を更新 (簡易的)
            if updated and is_closing_time:
                # 本来はCSVを書き換えるが、ここではログ出力のみ（次回の起動時に再計算されるため）
                pass

    # --- Phase 2: バッチ新規エントリー探索 ---
    new_buy_list = []
    
    # 資金に余裕がある場合のみスキャン開始
    if len(pm.holdings) < MAX_POSITIONS and current_cash > 100000:
        print("\n--- 新規チャンス探索 (Analyst Scan) ---")
        
        candidates = [] # 有望株リスト
        
        for i, tic in enumerate(WATCH_LIST):
            if tic in pm.holdings: continue 
            print(f"\r[{i+1}/{len(WATCH_LIST)}] {tic}...", end="")
            
            df = download_data_safe(tic)
            if df is None: continue
            
            # テクニカル計算
            df['SMA25'] = df['Close'].rolling(25).mean()
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Signal'] = df['MACD'].ewm(span=9).mean()
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
            df = df.dropna()
            
            metrics = calculate_metrics_enhanced(df)
            if metrics is None: continue
            
            # 鉄の掟 (2.3%)
            if metrics['entry_volatility'] > 2.3 or metrics['trend_momentum'] < 0 or metrics['sma25_dev'] < 0:
                continue
            
            # 有望株 -> 分析官レポート
            # API負荷軽減のため、1日最大5〜10銘柄程度に制限すると良い
            if len(candidates) >= 10: break 
            
            chart_bytes = create_chart_image(df, tic)
            similar_text = memory.get_similar(metrics)
            news = get_latest_news(tic)
            fund = get_fundamentals(tic)
            weekly = get_weekly_trend(tic)
            
            report = run_analyst(model, tic, metrics, chart_bytes, similar_text, news, fund, weekly)
            
            candidates.append({
                'ticker': tic,
                'metrics': metrics,
                'report': report
            })
            time.sleep(1)

        # 指令官による一括判断
        if candidates:
            print(f"\n👮 指令官が候補{len(candidates)}件を審査中...")
            
            # 保有株テキスト
            current_portfolio_text = "なし"
            if pm.holdings:
                current_portfolio_text = ", ".join([t for t in pm.holdings.keys()])
            
            decision_data = run_commander_batch(model, candidates, current_cash, current_portfolio_text)
            
            orders = decision_data.get('orders', [])
            for order in orders:
                tic = order.get('ticker')
                shares = order.get('shares', 0)
                
                if shares > 0:
                    target = next((c for c in candidates if c['ticker'] == tic), None)
                    if target:
                        metrics = target['metrics']
                        cost = shares * metrics['price']
                        
                        if cost <= current_cash:
                            current_cash -= cost # 仮想的に減らす
                            atr_val = metrics['atr_value']
                            initial_sl = order.get('stop_loss', metrics['price'] - atr_val * 2.0)
                            
                            print(f" -> 🔴 BUY確定: {tic} {shares}株")
                            
                            # CSV記録 (Closing時のみ)
                            if is_closing_time:
                                item = {
                                    "Date": datetime.datetime.now().strftime('%Y-%m-%d'), 
                                    "Ticker": tic, "Timeframe": TIMEFRAME, 
                                    "Action": "BUY", "result": "", "Reason": order.get('reason'), 
                                    "Confidence": 80, "stop_loss_price": initial_sl, "stop_loss_reason": "AI_Commander",
                                    "Price": metrics['price'], "sma25_dev": metrics['sma25_dev'], 
                                    "trend_momentum": metrics['trend_momentum'], "macd_power": metrics['macd_power'],
                                    "entry_volatility": metrics['entry_volatility'], "rsi_9": metrics['rsi_9'],
                                    "profit_loss": 0, "profit_rate": 0.0 
                                }
                                df_new = pd.DataFrame([item])
                                try:
                                    df_new.to_csv(REAL_TRADE_LOG_FILE, mode='a', header=not os.path.exists(REAL_TRADE_LOG_FILE), index=False, encoding='utf-8-sig')
                                except: pass

                            # 通知
                            earnings = get_earnings_date(tic)
                            warn = f"\n⚠️ 決算: {earnings}" if earnings != "-" else ""
                            msg = (
                                f"🔴 **BUY {tic}**: {metrics['price']:,.0f}円\n"
                                f"💰 **指令**: {shares}株 (約{cost:,.0f}円)\n"
                                f"🛡️ **初期撤退**: **{initial_sl:,.0f}円** (終値判断)\n"
                                f"👮 **理由**: {order.get('reason')}\n"
                                f"💡 **戦術**: +3%まで我慢、+2.5%で建値ガード\n"
                                f"{warn}"
                            )
                            new_buy_list.append(msg)

    # --- 通知 ---
    if alert_list:
        report_msg += "\n".join(alert_list) + "\n"
    
    if new_buy_list:
        report_msg += "\n🚀 **新規BUY指令**\n" + "\n\n".join(new_buy_list)
    elif not alert_list:
        report_msg += "\n💤 特筆すべき動きはありません。"

    send_discord(report_msg, filename="FullReport")
    print("\n✅ 完了")