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
    load_dotenv(override=True)
except ImportError:
    pass

# ==========================================
# ★設定エリア
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")
    exit()
webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
genai.configure(api_key=GOOGLE_API_KEY)

LOG_FILE = "ai_trade_memory_risk_managed.csv"
REAL_TRADE_LOG_FILE = "real_trade_record.csv"
MODEL_NAME = 'models/gemini-3-pro-preview' # モデル名は適宜変更してください
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
    # --- テキストファイル保存処理 ---
    try:
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        file_path = os.path.join(report_dir, "latest_report.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(message)
        print(f"✅ メッセージをファイルに保存しました: {file_path}")
    except Exception as e:
        print(f"⚠️ ファイル保存エラー: {e}")
    # -------------------------------

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("DISCORD_WEBHOOK_URLが設定されていません")
        return

    # リトライ設定
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # ★ここが修正ポイント: メッセージを1900文字ごとに分割して送信
    # Discordの制限は2000文字ですが、余裕を持って1900文字で切ります
    chunk_size = 1900
    total_len = len(message)
    
    print(f"メッセージサイズ: {total_len}文字 -> 分割送信します")

    try:
        for i in range(0, total_len, chunk_size):
            chunk = message[i : i + chunk_size]
            
            # 分割した切れ端を送信
            data = {
                "content": chunk,
                "username": "AI投資アドバイザー",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4228/4228956.png"
            }
            
            response = session.post(webhook_url, json=data, timeout=20)
            response.raise_for_status()
            
            # 順番が前後しないように少し待つ
            time.sleep(1)

        print("✅ Discord通知送信成功 (分割完了)")

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

# ==========================================
# ★追加: ファンダメンタルズ情報取得関数
# ==========================================
# ==========================================
# ★修正版: ファンダメンタルズ情報取得関数 (エラー回避版)
# ==========================================
def get_fundamentals(ticker):
    try:
        # yfinanceのTickerオブジェクト
        stock = yf.Ticker(ticker)
        
        # .info は非常に重くエラーになりやすいため、
        # タイムアウト対策として fast_info を優先利用する手もありますが、
        # ここではエラーハンドリングを強化して info を取得します。
        try:
            info = stock.info
        except:
            # 取得失敗時は空のデータを返して処理を止めない
            return "【ファンダメンタルズ】データ取得不可（テクニカルのみで判断します）"

        # 必須データがない場合はスキップ
        if not info:
            return "【ファンダメンタルズ】データなし"

        data = {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "不明"),
            "market_cap": info.get("marketCap", 0),
            "per": info.get("trailingPE", 0),
            "pbr": info.get("priceToBook", 0),
            "roe": info.get("returnOnEquity", 0),
            "dividend_yield": info.get("dividendYield", 0)
        }
        
        # データ整形
        m_cap = data['market_cap']
        if m_cap > 1000000000000:
            cap_str = f"{m_cap/1000000000000:.1f}兆円"
        elif m_cap > 100000000:
            cap_str = f"{m_cap/100000000:.0f}億円"
        else:
            cap_str = "-"

        roe_str = f"{data['roe']*100:.1f}%" if data['roe'] else "-"
        div_str = f"{data['dividend_yield']*100:.2f}%" if data['dividend_yield'] else "-"
        per_str = f"{data['per']:.1f}倍" if data['per'] else "-"
        pbr_str = f"{data['pbr']:.2f}倍" if data['pbr'] else "-"

        text = f"""
【ファンダメンタルズ】
- 企業名: {data['name']} (セクター: {data['sector']})
- 時価総額: {cap_str}
- 割安性: PER {per_str}, PBR {pbr_str}
- 収益・還元: ROE {roe_str}, 配当利回り {div_str}
"""
        return text.strip()

    except Exception:
        # どんなエラーが起きても絶対にプログラムを止めない
        return "【ファンダメンタルズ】取得エラー（無視して続行）"

# ==========================================
# ★追加: 週足トレンド判定関数 (マルチタイムフレーム)
# ==========================================
def get_weekly_trend(ticker):
    """
    週足データを取得し、中期トレンドを判定する
    """
    try:
        # 過去2年分の週足を取得
        df = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        if len(df) < 26: return "不明 (データ不足)"

        # SMA13 (約3ヶ月) と SMA26 (約半年) を計算
        sma13 = df['Close'].rolling(13).mean().iloc[-1]
        sma26 = df['Close'].rolling(26).mean().iloc[-1]
        current_price = float(df['Close'].iloc[-1])
        
        # 判定ロジック
        if current_price > sma13 and sma13 > sma26:
            return "上昇 📈 (最強)"
        elif current_price > sma13:
            return "上昇 ↗️ (短期強気)"
        elif current_price < sma13 and sma13 < sma26:
            return "下降 📉 (最弱)"
        elif current_price < sma13:
            return "下降 ↘️ (調整/下落)"
        else:
            return "レンジ ➡️"
    except:
        return "取得エラー"
    
def calculate_metrics_enhanced(df):
    if len(df) < 25: return None 
    
    curr = df.iloc[-1]
    price = float(curr['Close'])
    
    # --- 基本トレンド ---
    sma25 = float(curr['SMA25'])
    sma25_dev = ((price / sma25) - 1) * 100
    
    if len(df) < 6: return None
    prev_sma25 = float(df['SMA25'].iloc[-6]) 
    slope = (sma25 - prev_sma25) / 5
    trend_momentum = (slope / price) * 1000 
    
    macd = float(curr['MACD'])
    signal = float(curr['Signal'])
    macd_power = ((macd - signal) / price) * 10000 

    # --- リスク指標 ---
    atr = float(curr['ATR'])
    entry_volatility = (atr / price) * 100

    # --- ★追加: 精度向上用指標 ---
    # 1. Bollinger Bandwidth (スクイーズ判定)
    std = df['Close'].rolling(20).std().iloc[-1]
    # バンド幅 = (4 * 標準偏差) / 単純移動平均
    # 値が小さいほどエネルギーが溜まっている
    bb_width = (4 * std) / df['Close'].rolling(20).mean().iloc[-1] * 100

    # 2. Volume Ratio (出来高急増判定)
    # 直近5日の平均出来高に対する、今日の出来高の倍率
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    if vol_ma5 > 0:
        volume_ratio = float(curr['Volume']) / vol_ma5
    else:
        volume_ratio = 1.0

    # 3. RSI (9日) - 押し目判定用
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
        # 新指標
        'bb_width': bb_width,
        'volume_ratio': volume_ratio,
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
# ==========================================
# ★強化版: マクロ経済データ取得
# ==========================================
def get_macro_data():
    """
    主要指数(日経平均, ドル円, S&P500, 米金利, VIX)をリアルタイム取得し、
    AIが環境認識するためのテキストを生成する。
    """
    tickers = {
        "^N225": "日経平均",
        "JPY=X": "ドル円",
        "^GSPC": "米S&P500",
        "^TNX": "米10年債利回り",
        "^VIX": "VIX(恐怖指数)"
    }
    
    report = "【🌎 マクロ環境・地合い】\n"
    
    try:
        # 一括ダウンロードで高速化 (直近5日分)
        data = yf.download(list(tickers.keys()), period="5d", progress=False)
        
        # マルチインデックス対応 (yfinanceのバージョン差異対策)
        if isinstance(data.columns, pd.MultiIndex):
            # Close列だけ抽出して簡素化
            df_close = data['Close']
        else:
            df_close = data['Close'] if 'Close' in data else data

        for symbol, name in tickers.items():
            try:
                # 個別の列データを取得
                series = df_close[symbol].dropna()
                if len(series) < 2:
                    report += f"- {name}: データ不足\n"
                    continue

                current = float(series.iloc[-1])
                prev = float(series.iloc[-2])
                
                # 変化率計算
                change = current - prev
                pct_change = (change / prev) * 100
                
                # アイコンと評価
                trend_icon = "↗️" if change > 0 else "↘️"
                
                # 金利やVIXは「単位」が違うので表示調整
                if symbol == "^TNX":
                    val_str = f"{current:.3f}%"
                elif symbol == "JPY=X":
                    val_str = f"{current:.2f}円"
                else:
                    val_str = f"{current:,.0f}"

                report += f"- {name}: {val_str} ({trend_icon} {pct_change:+.2f}%)\n"

            except Exception:
                report += f"- {name}: 取得エラー\n"

        # 簡易的な環境判定コメントを追加
        vix_val = float(df_close["^VIX"].iloc[-1]) if "^VIX" in df_close else 0
        if vix_val > 30:
            report += "⚠️ **警告**: VIX指数が30を超えており、市場はパニック状態です。エントリーは極めて慎重に。\n"
        elif vix_val > 20:
            report += "⚠️ **注意**: ボラティリティが高まっています。急落に警戒してください。\n"
        else:
            report += "✅ 市場心理は比較的落ち着いています。\n"

    except Exception as e:
        print(f"Macro Data Error: {e}")
        return "【マクロ環境】データ取得失敗（テクニカルのみで判断します）"

    return report.strip()

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

# fundamentals を追加して 8個 にする
def analyze_vision_agent(model_instance, chart, metrics, cbr_text, macro, news, fundamentals, weekly_trend, name):
    """
    【AI判断】高精度・スナイパー版
    堅牢さを維持しつつ、「スクイーズ」や「出来高」を見て勝率の高い局面を狙う
    """
    mech_sl_long = metrics['price'] - (metrics['atr_value'] * 2.0)
    trend_dir = "上昇" if metrics['trend_momentum'] > 0 else "下降"

    # ボラティリティ警告
    vol_msg = ""
    if metrics['entry_volatility'] >= 2.0:
        vol_msg = "⚠️ 現在ボラティリティが高すぎます(2.0%以上)。新規BUYは禁止。SELL(逃げ)を検討してください。"

    prompt = f"""
あなたは「百発百中のスナイパー・トレーダー」です。
「負けないこと」は当然として、**「確実に勝てる局面」だけ** を選び抜いてください。

=== 入力情報 ===
銘柄: {name}
0. マクロ環境（市場全体の地合い）:
   {macro}

1. テクニカル分析:
   - トレンド: {trend_dir} (勢い: {metrics['trend_momentum']:.2f})
   - SMA25乖離: {metrics['sma25_dev']:.2f}% (プラスならSMAより上)
   - ボラティリティ: {metrics['entry_volatility']:.2f}% (2.0%未満が理想)
    - **週足(中期): {weekly_trend}  
   **【重要指標】**
   - **BB幅(スクイーズ度)**: {metrics['bb_width']:.2f}% (10%未満はエネルギー充填中)
   - **出来高倍率**: {metrics['volume_ratio']:.2f}倍 (1.0超えは資金流入)
   - **RSI(9)**: {metrics['rsi_9']:.1f} (40-60は押し目買いの好機)

2. ファンダメンタルズ:
   {fundamentals}

3. ニュース: {news}

{cbr_text}

=== 判断基準 ===

{vol_msg}

**【BUY 🔴: 新規買い（高勝率チャンス）】**
以下をすべて満たす「黄金パターン」を念頭においてエントリーせよ。
1. **安全性:** ボラティリティが 2.0% 未満であること。
2. **トレンド:** SMA25が上向きで、価格がSMA25の上にあること。
3. **エッジ (以下のいずれかがあること):**
   - **スクイーズからの初動:** BB幅が狭く、かつ出来高が増加傾向にある。
   - **押し目:** 上昇トレンド中で、RSIが 40〜50 まで調整している。

**【HOLD 🟡: 様子見・保有継続】**
- トレンドは悪くないが、爆発の予兆（出来高急増など）がない場合。
- 迷う場合はすべてHOLDを選択せよ。

**【SELL 🔵: 決済・逃げ（手仕舞いの合図）】**
- トレンド崩壊、またはボラティリティの急拡大。
- ※空売りではなく「保有ポジションの決済（逃げ）」の合図として判断すること。
- リスク回避を最優先し、負けないことを最重要視せよ。

=== 出力 (JSONのみ) ===
{{
  "action": "BUY", "HOLD", "SELL" のいずれか,
  "confidence": 0-100,
  "stop_loss_price": 数値,
  "stop_loss_reason": "直近安値かつATR2倍ライン... (30文字以内)",
  "reason": "ボラティリティ1.2%と低く、BB幅が収縮した状態で出来高が1.5倍に急増。爆発の初動と判断... (100文字以内)"
}}
"""
    try:
        response = model_instance.generate_content([prompt, chart])
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        # ★エラー内容をコンソールに赤字で表示する
        print(f"\n⚠️ AI ERROR: {e}") 
        return {"action": "HOLD", "confidence": 0, "reason": f"API Error: {e}", "stop_loss_price": 0}
# ==========================================
# 5. メイン実行 (実戦監視)
# ==========================================
# ==========================================
# 5. メイン実行 (実戦監視)
# ==========================================
if __name__ == "__main__":
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"=== AI市場監視システム ({today}) ===")
    
    # ★追加: 監視リストの重複排除とソート
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
    
    # 保存先ファイルの定義
    SAVE_TARGETS = [
        {"path": LOG_FILE, "name": "学習メモリ"},
        {"path": REAL_TRADE_LOG_FILE, "name": "実戦ログ"}
    ]

    for i, tic in enumerate(WATCH_LIST, 1):
        name = tic 
        print(f"[{i}/{len(WATCH_LIST)}] {name}... ", end="", flush=True)
        
        # --- (中略) データ取得やAI分析のコードはそのまま ---
        
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
        fundamentals = get_fundamentals(name)
        weekly_trend = get_weekly_trend(name) # ★追加済みなら
        
        # AI分析実行
        res = analyze_vision_agent(model_instance, chart, metrics, cbr_text, macro, news, fundamentals, weekly_trend, name)
              
        action = res.get('action', 'HOLD')
        conf = res.get('confidence', 0)
        sl_price_raw = res.get('stop_loss_price', 0)
        try: sl_price = float(sl_price_raw)
        except: sl_price = 0.0
        
        # --- 保存用データの作成 ---
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
        # カラム順序の強制
        for col in csv_columns:
            if col not in df_new.columns: df_new[col] = None
        df_new = df_new[csv_columns]

        # --- ★修正: 重複チェック付き保存処理 ---
        for target in SAVE_TARGETS:
            path = target["path"]
            name_label = target["name"]
            
            try:
                # 既存ファイルの読み込みと重複チェック
                if os.path.exists(path):
                    try:
                        df_exist = pd.read_csv(path, on_bad_lines='skip')
                        # 「同じ日付」かつ「同じ銘柄」のデータが既にあるか確認
                        is_duplicate = ((df_exist['Date'] == today) & (df_exist['Ticker'] == tic)).any()
                        
                        if is_duplicate:
                            # 重複があれば保存しない（コンソールに小さく表示）
                            # print(f"({name_label}: 済)", end=" ")
                            pass 
                        else:
                            df_new.to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')
                            print(f"📝 {name_label}保存", end=" ")
                    except:
                        # 読み込みエラー等の場合は強制追記（安全策）
                        df_new.to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')
                else:
                    # 新規作成
                    df_new.to_csv(path, index=False, encoding='utf-8-sig')
                    print(f"🆕 {name_label}作成", end=" ")
                    
            except PermissionError:
                print(f"❌{name_label}ロック中", end=" ")
            except Exception as e:
                print(f"❌{name_label}エラー:{e}", end=" ")

        # --- コンソール表示 ---
        action_icon = "🔴" if action == "BUY" else "🔵" if action == "SELL" else "🟡"
        sl_str = f"(SL: {sl_price:.0f})" if action == "BUY" and sl_price > 0 else ""
        print(f"-> {action_icon} {conf}% {sl_str}")

        if action == "BUY":
            sl_str = f"(SL: {sl_price:.0f}円)" if sl_price > 0 else ""
            msg = f"🔴 **BUY {name}**: {metrics['price']:.0f}円 {sl_str}\n> 理由: {res.get('reason')}"
            buy_list.append(msg)
            
        elif action == "SELL":
            msg = f"🔵 **SELL (決済) {name}**: {metrics['price']:.0f}円\n> 理由: {res.get('reason')}"
            buy_list.append(msg)
            
        time.sleep(2)

    # 通知作成
    if buy_list:
        report_message += "\n🚀 **新規BUY/SELL銘柄**\n" + "\n\n".join(buy_list)
    else:
        report_message += "\n💤 本日は「BUY/SELL」銘柄はありませんでした。"

    # Discord送信
    send_discord_notify(report_message)

    # レポート保存
    try:
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        file_path = os.path.join(report_dir, "latest_report.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_message)
        print(f"✅ レポート保存: {file_path}")
    except Exception as e:
        print(f"⚠️ ファイル保存エラー: {e}")