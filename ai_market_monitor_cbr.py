import yfinance as yf
import pandas as pd
import google.generativeai as genai
import json
import time
import datetime
import urllib.parse
import feedparser
import requests # 追加: LINE送信用ライブラリ
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def send_line_notify(message):
            line_notify_token = os.environ.get("LINE_TOKEN") # GitHubの設定から読み込む
            if not line_notify_token:
                print("LINE_TOKENが設定されていません")
                return

            line_notify_api = "https://notify-api.line.me/api/notify"
            headers = {"Authorization": f"Bearer {line_notify_token}"}
            data = {"message": f"\n{message}"}
            try:
                requests.post(line_notify_api, headers=headers, data=data)
            except Exception as e:
                print(f"LINE送信エラー: {e}")


# メイン実行部分の修正
if __name__ == "__main__":
    # ==========================================
    # ★設定エリア
    # ==========================================
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY").strip() 

    # ログファイル名 (リスク管理対応版)
    LOG_FILE = "ai_trade_memory_risk_managed.csv"
    MODEL_NAME = 'models/gemini-3-pro-preview' # 必要に応じて pro 等に変更

    # タイムフレーム設定 (明記)
    TIMEFRAME = "1d" 

    # CBR (過去事例学習) 設定 
    # 💡【修正点】参照する過去の類似事例の数 (最大値) を10に増加
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
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    plt.rcParams['font.family'] = 'sans-serif'
# ★通知用のメッセージを貯める変数
    report_message = f"【AI市場監視 ({today})】\n"
    buy_list = []
    hold_list = []

    for i, tic in enumerate(WATCH_LIST, 1):    
        # ==========================================
        # 1. 共通ロジック (ATR計算 & データ取得)
        # ==========================================
        def download_data_safe(ticker, period="6mo", interval="1d", retries=3):
            """【共通】interval引数を追加し、タイムフレームを固定"""
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
            """
            【共通】特徴量計算 (ATR/Volatility追加)
            """
            if len(df) < 15: return None 
            
            curr = df.iloc[-1]
            price = float(curr['Close'])
            
            # --- 既存指標 ---
            sma25 = float(curr['SMA25'])
            sma25_dev = ((price / sma25) - 1) * 100
            
            # -6は5営業日前のSMA25を指す
            if len(df) < 6: return None
            prev_sma25 = float(df['SMA25'].iloc[-6]) 
            slope = (sma25 - prev_sma25) / 5
            trend_momentum = (slope / price) * 1000 
            
            macd = float(curr['MACD'])
            signal = float(curr['Signal'])
            macd_power = ((macd - signal) / price) * 10000 

            # --- ★リスク指標 (ATR & Volatility) ---
            atr = float(curr['ATR'])
            # 株価に対する変動率（％）
            entry_volatility = (atr / price) * 100

            return {
                'sma25_dev': sma25_dev,
                'trend_momentum': trend_momentum,
                'macd_power': macd_power,
                'entry_volatility': entry_volatility, # 追加
                'price': price,
                'atr_value': atr # 生のATR値（SL計算用）
            }

        # ==========================================
        # 2. CBRメモリシステム (リスク指標対応 & 堅牢化)
        # ==========================================
        class CaseBasedMemory:
            def __init__(self, csv_path):
                self.csv_path = csv_path
                self.scaler = StandardScaler()
                self.knn = None
                self.df = pd.DataFrame()
                # ★検索特徴量に 'entry_volatility' を追加
                self.feature_cols = ['sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility']
                self.load_and_train()

            def load_and_train(self):
                if not os.path.exists(self.csv_path): return
                try:
                    self.df = pd.read_csv(self.csv_path)
                    
                    # 過去データとの互換性のため、カラム名を標準化
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

                    # 欠損値埋めと正規化
                    features = self.df[self.feature_cols].fillna(0)
                    self.features_normalized = self.scaler.fit_transform(features)
                    
                    # 💡【修正点】参照する事例の数を CBR_NEIGHBORS_COUNT に変更
                    self.knn = NearestNeighbors(n_neighbors=min(CBR_NEIGHBORS_COUNT, len(self.df)), metric='euclidean')
                    self.knn.fit(self.features_normalized)
                    print(f"Memory System: Loaded {len(self.df)} cases. (Neighbors: {min(CBR_NEIGHBORS_COUNT, len(self.df))})")
                except Exception as e:
                    print(f"Memory Load Error: {e}")

            def search_similar_cases(self, current_metrics):
                if self.knn is None or len(self.df) < 5:
                    return "（データ不足のため参照なし）"

                input_df = pd.DataFrame([current_metrics], columns=self.feature_cols)
                scaled_vec = self.scaler.transform(input_df) 
                distances, indices = self.knn.kneighbors(scaled_vec)
                
                text = f"【システム検索: 現在({TIMEFRAME})と波形・ボラティリティが酷似する過去事例（{len(indices[0])}件参照）】\n"
                text += "--------------------------------------------------\n"
                for idx in indices[0]:
                    row = self.df.iloc[idx]
                    res = str(row.get('result', ''))
                    
                    icon = "WIN ⭕" if res=='WIN' else "LOSS ❌" if res=='LOSS' else "➖"
                    
                    text += f"● {row['Date']} {row['Ticker']} -> {icon}\n"
                    text += f"   乖離:{row['sma25_dev']:.1f}%, 勢い:{row['trend_momentum']:.1f}, ボラ:{row['entry_volatility']:.1f}%\n"
                    
                    sl_price_raw = row.get('stop_loss_price', 0)
                    try:
                        sl_price = float(sl_price_raw)
                    except (ValueError, TypeError):
                        sl_price = 0.0

                    if sl_price > 0.0:
                        text += f"   SL設定: {sl_price:.0f} (理由: {str(row.get('stop_loss_reason',''))[:20]}...)\n"
                    else:
                        text += f"   理由: {str(row.get('Reason',''))[:40]}...\n"

                text += "--------------------------------------------------\n"
                return text

        # ==========================================
        # 3. 外部データ & AI分析ロジック (リスク管理強化)
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
            """
            【AI判断】本番用プロンプト（買い専門・防御特化版）を適用
            """
            # 統計的なSL目安 (ATR x 2.0)
            mech_sl_long = metrics['price'] - (metrics['atr_value'] * 2.0)
            
            # トレンド方向の判定
            trend_dir = "上昇" if metrics['trend_momentum'] > 0 else "下降"

            # ボラティリティ警告（2.0%以上は即HOLD）
            vol_limit_msg = ""
            if metrics['entry_volatility'] >= 2.0:
                vol_limit_msg = "⚠️【禁止事項】現在のボラティリティ(2.0%以上)はリスク許容範囲外です。絶対にBUYしてはいけません。必ずHOLDを選択してください。"

            prompt = f"""
        あなたは「超低リスク・買い専門のヘッジファンドCIO」です。
        勝率100%を目指すため、リスクの高い局面は全て見送ります。

        === 入力情報 ===
        銘柄: {name} (Timeframe: {TIMEFRAME})
        1. マクロ環境: {macro}
        2. テクニカル:
        - 価格: {metrics['price']:.0f}
        - トレンド方向: {trend_dir} (勢い: {metrics['trend_momentum']:.2f})
        - SMA25乖離: {metrics['sma25_dev']:.2f}%
        - ボラティリティ(ATR比): {metrics['entry_volatility']:.2f}%
        - 統計的SL目安(ATR x2): {mech_sl_long:.0f} 円付近
        3. 最新ニュース: {news}

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
        - すでに保有している場合、「HOLD」は**利益確定（売り）のサイン**となる。
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
            except: return {"action": "HOLD", "confidence": 0, "reason": "API Error", "stop_loss_price": 0}# ==========================================

        # 4. メイン実行 (実戦監視)
        # ==========================================
        if __name__ == "__main__":
            print(f"=== AI市場監視システム (Risk Managed: {TIMEFRAME}) ===")
            
            # ⚠️APIキーを設定してください
            if GOOGLE_API_KEY == "ここにAPIキーを貼り付け":
                print("Error: Please set your GOOGLE_API_KEY.")
                exit()
                
            genai.configure(api_key=GOOGLE_API_KEY)
            try:
                model_instance = genai.GenerativeModel(MODEL_NAME)
            except Exception as e:
                print(f"Error: {e}"); exit()

            # CBR初期化
            cbr = CaseBasedMemory(LOG_FILE)
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            macro = get_macro_data()
            print(macro)
            
            for i, tic in enumerate(WATCH_LIST, 1):
                name = tic 
                print(f"[{i}/{len(WATCH_LIST)}] {name}... ", end="", flush=True)
                
                # 1. データ取得
                df = download_data_safe(tic, interval=TIMEFRAME)
                if df is None or len(df) < 100:
                    print("Skip (Data Error)")
                    continue
                    
                # 2. 指標計算
                df['SMA25'] = df['Close'].rolling(25).mean()
                df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                df['Signal'] = df['MACD'].ewm(span=9).mean()
                
                # ★ATR計算
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift())
                low_close = np.abs(df['Low'] - df['Close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['ATR'] = tr.rolling(14).mean()
                
                df = df.dropna()
                metrics = calculate_metrics_enhanced(df)
                if metrics is None: 
                    print("Skip (Metrics Error)")
                    continue
                
                # 3. CBR & AI判断
                cbr_text = cbr.search_similar_cases(metrics)
                chart = create_chart_image(df, name)
                news = get_latest_news(name)
                
                res = analyze_vision_agent(model_instance, chart, metrics, cbr_text, macro, news, name)
                
                # 4. 結果表示 & 保存
                action = res.get('action', 'HOLD')
                conf = res.get('confidence', 0)
                # SL価格の安全な取得
                sl_price_raw = res.get('stop_loss_price', 0)
                try:
                    sl_price = float(sl_price_raw)
                except (ValueError, TypeError):
                    sl_price = 0.0 # 数値変換失敗時は0とする
                
                item = {
                    "Date": today, "Ticker": tic, "Timeframe": TIMEFRAME, 
                    "Action": action, "Confidence": conf,
                    "stop_loss_price": sl_price, # SL保存
                    "stop_loss_reason": res.get('stop_loss_reason', '-'),
                    "Reason": res.get('reason', 'None'), 
                    "Price": metrics['price'],
                    "sma25_dev": metrics['sma25_dev'], 
                    "trend_momentum": metrics['trend_momentum'],
                    "macd_power": metrics['macd_power'], # 指標を追加
                    "entry_volatility": metrics['entry_volatility'], # ボラティリティ保存
                    "result": "", "profit_loss": ""
                }
                
                # CSV保存
                df_new = pd.DataFrame([item])
                if not os.path.exists(LOG_FILE):
                    df_new.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
                else:
                    df_new.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
                    
                action_icon = "BUY 🔴" if action == "BUY" else "SELL 🔵" if action == "SELL" else "HOLD ⚪"
                sl_str = f"(SL: {sl_price:.0f})" if action == "BUY" and sl_price > 0 else ""
                print(f"{action_icon} {conf}% {sl_str}")

                if action == "BUY":
                    sl_info = f"(SL: {sl_price:.0f})" if sl_price > 0 else ""
                    msg = f"🔴 {tic}: {metrics['price']:.0f}円 {sl_info}\n理由: {res.get('reason')[:40]}..."
                    buy_list.append(msg)
                elif action == "HOLD":
                    # HOLDは1行でシンプルに
                    hold_list.append(f"⚪ {tic}")
                
                # 全銘柄のHOLD理由などは長くなるので通知しない（または簡潔にする）
                
                time.sleep(3) # 待機
            # 最終通知
            # ループ終了後、最後にまとめてLINE送信
            if buy_list:
                report_message += "\n🚀【新規BUY銘柄】\n" + "\n\n".join(buy_list)
            else:
                report_message += "\n本日は「BUY」銘柄はありませんでした。"

            # 2. HOLD銘柄の報告（件数が多いのでリストで羅列）
            if hold_list:
                report_message += f"\n\n☕【HOLD銘柄 ({len(hold_list)}件)】\n"
                report_message += ", ".join(hold_list) # カンマ区切りで表示

            # 送信実行
            send_line_notify(report_message)
                

            