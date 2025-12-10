import os
import io
import time
import json
import random
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import subprocess # これが必要です

# ==========================================
# ★追加: GitHub自動同期機能
# ==========================================
def auto_git_push(commit_message="Auto update trade memory"):
    """
    学習結果(CSV)を自動でGitHubにプッシュする
    """
    try:
        print("\n☁️ GitHubへ同期中...")
        
        # 1. 変更をステージング
        subprocess.run(["git", "add", "ai_trade_memory_risk_managed.csv"], check=True)
        
        # 2. コミット (変更がない場合はエラーになるのでtryで囲む)
        try:
            subprocess.run(["git", "commit", "-m", commit_message], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            print("   (変更がないためコミットはスキップされました)")
            return

        # 3. プッシュ
        subprocess.run(["git", "push"], check=True)
        print("✅ 同期完了！クラウドダッシュボードが更新されました。")
        
    except Exception as e:
        print(f"⚠️ GitHub同期エラー: {e}")
        print("   (Gitがインストールされているか、リポジトリ内か確認してください)")


try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# ==========================================
# ★設定エリア
# ==========================================
GOOGLE_API_KEY = os.getenv("TRAINING_API_KEY").strip()
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")
    # exit() # 環境によってはコメントアウト

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = 'models/gemini-2.0-flash' # コストパフォーマンスの良いモデル推奨
LOG_FILE = "ai_trade_memory_risk_managed.csv" 

TRAINING_ROUNDS = 100 # 1回の実行で行う回数
TIMEFRAME = "1d" 
CBR_NEIGHBORS_COUNT = 11
MIN_VOLATILITY = 1.0 

# 練習用銘柄リスト
TRAINING_LIST = [
    # --- 電気機器・半導体・精密 ---
    "8035.T", "6146.T", "6920.T", "6857.T", "6723.T", "7735.T", "6526.T", "6758.T", "6861.T", "6501.T",
    "6503.T", "6981.T", "6954.T", "7741.T", "6902.T", "6367.T", "6594.T", "7751.T", "6701.T", "6702.T",
    "6752.T", "6762.T", "6479.T", "7733.T", "6645.T", "6971.T", "6976.T", "6988.T", "7203.T", "7267.T",
    
    # --- 自動車・輸送用機器 ---
    "7201.T", "7269.T", "7270.T", "7202.T", "7211.T", "7259.T", "7261.T", "7272.T", "7011.T", "7012.T",
    "7013.T",
    
    # --- 銀行・証券・保険・その他金融 ---
    "8306.T", "8316.T", "8411.T", "8766.T", "8725.T", "8591.T", "8604.T", "8308.T", "8309.T", "8331.T",
    "8354.T", "8473.T", "8601.T", "8630.T", "8697.T", "8750.T", "8795.T", "8570.T", "8593.T",
    
    # --- 商社・卸売 ---
    "8058.T", "8031.T", "8001.T", "8002.T", "8015.T", "2768.T", "8053.T", "2760.T", "7459.T", "8088.T",
    "9962.T", # 9810.T (日鉄物産) は上場廃止のため削除済み
    
    # --- 医薬品 ---
    "4568.T", "4519.T", "4503.T", "4502.T", "4507.T", "4523.T", "4151.T", "4506.T", "4528.T", "4543.T",
    "4578.T",
    
    # --- 化学・素材 ---
    "4063.T", "4901.T", "4452.T", "4911.T", "3402.T", "3407.T", "4005.T", "4183.T", "4188.T", "4204.T",
    "4021.T", "4091.T", "4114.T", "4185.T", "4202.T", "4208.T", "4403.T", "4631.T", "4912.T",
    
    # --- 機械 ---
    "6301.T", "6367.T", "6326.T", "6098.T", "6113.T", "6178.T", "6302.T", "6305.T", "6383.T", "6471.T",
    "6472.T", "6473.T", "7004.T",
    
    # --- 情報通信・サービス ---
    "9432.T", "9433.T", "9984.T", "9434.T", "4661.T", "6098.T", "9684.T", "9697.T", "4385.T", "2413.T",
    "3659.T", "3938.T", "4307.T", "4684.T", "4689.T", "4704.T", "4751.T", "4755.T", "6701.T", "9435.T",
    "9602.T", "9613.T", "9735.T", "9766.T",
    
    # --- 小売 ---
    "9983.T", "3382.T", "8267.T", "9843.T", "3092.T", "7532.T", "2651.T", "2670.T", "3086.T", "3099.T",
    "3391.T", "7453.T", "8233.T", "8252.T", "9989.T",
    
    # --- 建設・不動産 ---
    "1925.T", "1801.T", "8801.T", "8802.T", "1802.T", "1803.T", "1812.T", "1928.T", "3231.T", "3289.T",
    "8830.T", "8804.T",
    
    # --- 鉄鋼・非鉄・鉱業 ---
    "5401.T", "5411.T", "1605.T", "5713.T", "5711.T", "5714.T", "5406.T", "5423.T", "5486.T",
    
    # --- 電力・ガス・石油 ---
    "9501.T", "9503.T", "5020.T", "9502.T", "9531.T", "9532.T", "5019.T", "9504.T", "9506.T", "9508.T",
    
    # --- 運輸（陸・海・空） ---
    "9101.T", "9104.T", "9107.T", "9020.T", "9021.T", "9022.T", "9201.T", "9202.T", "9064.T", "9143.T",
    "9147.T",
    
    # --- 食品・その他 ---
    "2802.T", "2914.T", "2502.T", "2503.T", "2801.T", "2269.T", "2282.T", "4911.T", "7832.T", "7974.T"
]
plt.rcParams['font.family'] = 'sans-serif' 

# ==========================================
# 1. データ取得 & テクニカル計算
# ==========================================
def download_data_safe(ticker, period="2y", interval="1d", retries=3):
    wait = 1
    for _ in range(retries):
        try:
            # yfinanceの共有エラーを無効化して取得
            import logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            
            if df.empty: 
                # データが空ならリトライせずにNoneを返す(上場廃止などの可能性が高いため)
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # データ数が少なすぎる場合もスキップ
            if len(df) < 50:
                return None
                
            return df
            
        except Exception as e:
            # その他のエラーは少し待ってリトライ
            time.sleep(wait)
            wait += 1
            
    return None

def calculate_technical_indicators(df):
    df = df.copy()
    df['SMA25'] = df['Close'].rolling(25).mean()
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR計算
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # 出来高移動平均
    df['VolumeMA5'] = df['Volume'].rolling(5).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    rs = gain / loss
    df['RSI9'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

def calculate_metrics_enhanced(df, idx):
    curr = df.iloc[idx]
    price = float(curr['Close'])
    
    sma25 = float(curr['SMA25'])
    sma25_dev = ((price / sma25) - 1) * 100
    
    # トレンドの勢い
    prev_sma25 = float(df['SMA25'].iloc[idx-5])
    slope = (sma25 - prev_sma25) / 5
    trend_momentum = (slope / price) * 1000
    
    macd = float(curr['MACD'])
    signal = float(curr['Signal'])
    macd_power = ((macd - signal) / price) * 10000
    
    atr = float(curr['ATR'])
    entry_volatility = (atr / price) * 100
    
    # BB幅 (スクイーズ判定)
    std = df['Close'].iloc[idx-19:idx+1].std()
    bb_width = (4 * std) / df['Close'].iloc[idx-19:idx+1].mean() * 100
    
    # 出来高倍率
    vol_ma5 = float(curr['VolumeMA5'])
    volume_ratio = float(curr['Volume']) / vol_ma5 if vol_ma5 > 0 else 1.0
    
    return {
        'sma25_dev': sma25_dev,
        'trend_momentum': trend_momentum,
        'macd_power': macd_power,
        'entry_volatility': entry_volatility, 
        'price': price,
        'atr_value': atr,
        'bb_width': bb_width,
        'volume_ratio': volume_ratio,
        'rsi_9': float(curr['RSI9'])
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
            self.df = pd.read_csv(self.csv_path)
            
            # カラム名を統一
            rename_map = {
                'date': 'Date', 'ticker': 'Ticker', 'action': 'Action', 
                'reason': 'Reason', 'timeframe': 'Timeframe', 
                'stop_loss_price': 'stop_loss_price', 
                'stop_loss_reason': 'stop_loss_reason',
                'result': 'result', 'profit_loss': 'profit_loss',
                'confidence': 'Confidence'
            }
            self.df.columns = [rename_map.get(col.lower(), col) for col in self.df.columns]
            
            if len(self.df) < 5: return

            # 特徴量の準備
            for col in self.feature_cols:
                 if col not in self.df.columns: self.df[col] = 0.0
            
            features = self.df[self.feature_cols].fillna(0)
            self.features_normalized = self.scaler.fit_transform(features)
            
            global CBR_NEIGHBORS_COUNT
            self.knn = NearestNeighbors(n_neighbors=min(CBR_NEIGHBORS_COUNT, len(self.df)), metric='euclidean')
            self.knn.fit(self.features_normalized)
            print(f"Memory Loaded: {len(self.df)} records.")
        except Exception as e:
            print(f"Memory Load Error: {e}")

    def search_similar_cases(self, current_metrics):
        if self.knn is None or len(self.df) < 5: return "（データ不足）"

        input_df = pd.DataFrame([current_metrics], columns=self.feature_cols)
        scaled_vec = self.scaler.transform(input_df)
        distances, indices = self.knn.kneighbors(scaled_vec)
        
        text = f"【類似過去事例 ({len(indices[0])}件)】\n"
        for idx in indices[0]:
            row = self.df.iloc[idx]
            res = str(row.get('result', ''))
            icon = "WIN ⭕" if res == 'WIN' else "LOSS ❌" if res == 'LOSS' else "➖"
            text += f"● {row.get('Date','?')} {row.get('Ticker','?')} -> {icon}\n"
        return text

    def save_experience(self, data_dict):
        # 保存するカラムの順序を強制（CSV破損防止）
        csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "stop_loss_reason", "Price", 
            "sma25_dev", "trend_momentum", "macd_power", "entry_volatility", "profit_loss"
        ]
        
        new_df = pd.DataFrame([data_dict])
        
        # カラム不足があれば補完し、順序を整える
        for col in csv_columns:
            if col not in new_df.columns: new_df[col] = None
        new_df = new_df[csv_columns]

        # ★強化ポイント: Excelが開いていてもリトライする処理
        max_retries = 5
        for i in range(max_retries):
            try:
                if not os.path.exists(self.csv_path):
                    new_df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
                else:
                    # 追記モード
                    new_df.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                
                # 成功したらループを抜ける
                print(f"   💾 記録しました") 
                self.load_and_train() # メモリ再読み込み
                return
            
            except PermissionError:
                if i < max_retries - 1:
                    print(f"⚠️ CSVがExcel等で開かれています。閉じてください... ({i+1}/{max_retries}回 再試行中)")
                    time.sleep(3)
                else:
                    print("❌ 書き込み失敗: CSVファイルを閉じてから再実行してください。")
            except Exception as e:
                print(f"❌ 保存エラー: {e}")
                break

# ==========================================
# 3. AIスパーリング (スナイパー版)
# ==========================================
def create_chart_image(df, ticker_name):
    data = df.tail(100).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(data.index, data['Close'], label='Close', color='black', linewidth=1.2)
    ax1.plot(data.index, data['SMA25'], label='SMA25', color='orange', alpha=0.8)
    ax1.set_title(f"{ticker_name} Trend Chart")
    ax1.legend(loc='upper left'); ax1.grid(True)
    ax2.plot(data.index, data['MACD'], label='MACD', color='red', linewidth=1.0)
    ax2.bar(data.index, data['MACD']-data['Signal'], color='gray', alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def ai_decision_maker(model, chart_bytes, metrics, similar_cases_text, ticker):
    # トレンド方向の判定
    trend_dir = "上昇" if metrics['trend_momentum'] > 0 else "下降"
    
    # ボラティリティ警告（基準を 2.0% -> 3.0% に緩和）
    vol_msg = ""
    if metrics['entry_volatility'] >= 3.0:
        vol_msg = "⚠️ 現在ボラティリティが極めて高い(3.0%以上)です。急落リスクがあるため、新規BUYは慎重に判断してください。"

    # プロンプト (KERNELフレームワーク適用・日本語版)
    prompt = f"""
### CONTEXT (入力データ)
対象銘柄: {ticker}


2. テクニカル指標:
   - 日足トレンド: {trend_dir} (勢い: {metrics['trend_momentum']:.2f})
   - SMA25乖離率: {metrics['sma25_dev']:.2f}% (プラス＝SMAより上)
   - ボラティリティ: {metrics['entry_volatility']:.2f}%
   - BB幅(スクイーズ度): {metrics['bb_width']:.2f}%
   - 出来高倍率: {metrics['volume_ratio']:.2f}倍
   - RSI(9): {metrics['rsi_9']:.1f}

{similar_cases_text}

### TASK (タスク)
あなたは百戦錬磨のファンドマネージャーです。
提供されたデータに基づき、**「確率的優位性」**が最も高いアクション（BUY, HOLD, SELL）を選択してください。

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

### FORMAT (出力形式: JSONのみ)
Markdown記法や説明文を含めず、以下のJSONオブジェクトのみを出力すること。
{{
  "action": "BUY" または "HOLD" または "SELL",
  "confidence": 0〜100の整数,
  "stop_loss_price": 数値 (HOLD/SELLの場合は0),
  "stop_loss_reason": "理由(30文字以内)",
  "reason": "理由(100文字以内)"
}}
"""
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}])
        text_clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text_clean)
    except Exception as e:
        # エラー時はHOLDを返す（安全策）
        return {"action": "HOLD", "reason": f"Error: {e}", "confidence": 0, "stop_loss_price": 0}

# ==========================================
# 4. メイン実行
# ==========================================
def main():
    print(f"=== AI強化合宿（スナイパー仕様） ===")
    
    memory_system = CaseBasedMemory(LOG_FILE)
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e: 
        print(f"Model Init Error: {e}")
        return

    processed_data = {}
    print(f"Downloading data...")
    for t in TRAINING_LIST:
        df = download_data_safe(t, interval=TIMEFRAME)
        if df is None or len(df) < 150: continue
        df = calculate_technical_indicators(df)
        processed_data[t] = df

    if not processed_data:
        print("データが取得できませんでした。")
        return

    win_count = 0
    loss_count = 0
    draw_count = 0
    
    print(f"\n🥊 トレーニング開始 ({TRAINING_ROUNDS}ラウンド)\n")
    
    for i in range(1, TRAINING_ROUNDS + 1):
        ticker = random.choice(list(processed_data.keys()))
        df = processed_data[ticker]
        
        # ランダムな日付を選ぶ（過去データから）
        if len(df) < 110: continue 
        target_idx = random.randint(100, len(df) - 10) # 未来の判定用に余裕を持たせる
        current_date_str = df.index[target_idx].strftime('%Y-%m-%d')
        
        metrics = calculate_metrics_enhanced(df, target_idx)
        
        # ボラティリティフィルタ（練習なので少し緩くても良いが、今回は本番同様に表示）
        # ※ ここではスキップせず、AIにHOLDと判断させる練習とする
        
        similar_cases_text = memory_system.search_similar_cases(metrics)
        past_df = df.iloc[:target_idx+1]
        chart_bytes = create_chart_image(past_df, ticker)
        
        # AI判断
        decision = ai_decision_maker(model_instance, chart_bytes, metrics, similar_cases_text, ticker)
        action = decision.get('action', 'HOLD')
        conf = decision.get('confidence', 0)
        sl_price_raw = decision.get('stop_loss_price', 0)
        try: sl_price = float(sl_price_raw)
        except: sl_price = 0.0

        # アイコン表示の統一
        if action == "BUY":
            action_display = "BUY 🔴"
        elif action == "SELL":
            action_display = "SELL 🔵"
        else:
            action_display = "HOLD 🟡"

        print(f"Round {i:03}: {ticker} ({current_date_str}) -> {action_display} (自信:{conf}%)")

        # 結果判定（BUYの場合のみ勝敗をつける）
        result = "DRAW"
        profit_loss = 0.0
        
        if action == "BUY":
            curr_price = float(metrics['price'])
            # 未来5日間のデータをチェック
            future_prices = df['Close'].iloc[target_idx+1 : target_idx+6]
            future_lows = df['Low'].iloc[target_idx+1 : target_idx+6]
            
            if len(future_prices) > 0:
                # 簡易判定: 5日以内に2%上昇で勝ち、損切り価格割れor2%下落で負け
                target_profit = curr_price * 1.02
                target_loss = sl_price if sl_price > 0 else curr_price * 0.98
                
                is_win = False
                is_loss = False
                
                for j in range(len(future_prices)):
                    p = future_prices.iloc[j]
                    l = future_lows.iloc[j]
                    
                    if l <= target_loss:
                        is_loss = True
                        profit_loss = l - curr_price
                        break
                    if p >= target_profit:
                        is_win = True
                        profit_loss = p - curr_price
                        break
                
                if is_win: result = "WIN"
                elif is_loss: result = "LOSS"
                else: 
                    # 5日後までの変化
                    final_p = future_prices.iloc[-1]
                    profit_loss = final_p - curr_price
                    result = "WIN" if profit_loss > 0 else "LOSS"

        elif action == "SELL":
             # 練習モードでのSELLは「逃げの判断が正しかったか」を見る
             # ここでは簡易的に「その後下がったら正解(WIN)」とする
             curr_price = float(metrics['price'])
             future_prices = df['Close'].iloc[target_idx+1 : target_idx+6]
             if len(future_prices) > 0:
                 if future_prices.iloc[-1] < curr_price:
                     result = "WIN" # 下がって正解
                     profit_loss = curr_price - future_prices.iloc[-1] # 仮想的な利益
                 else:
                     result = "LOSS" # 上がってしまった（逃げる必要なかった）
                     profit_loss = curr_price - future_prices.iloc[-1]

        # 結果表示
        if action != "HOLD":
            icon = "🏆" if result == "WIN" else "💀" if result == "LOSS" else "➖"
            print(f"   結果: {icon} {result} (PL: {profit_loss:.1f})")
            print(f"   理由: {decision.get('reason')}")
            
            if result == "WIN": win_count += 1
            if result == "LOSS": loss_count += 1
            if result == "DRAW": draw_count += 1
            
            # 結果を学習データとして保存
            save_data = {
                'Date': current_date_str, 'Ticker': ticker, 'Timeframe': TIMEFRAME, 
                'Action': action, 'result': result, 
                'Reason': decision.get('reason', 'None'),
                'Confidence': conf,
                'stop_loss_price': sl_price, 
                'stop_loss_reason': decision.get('stop_loss_reason', 'None'), 
                'Price': metrics['price'],
                'sma25_dev': metrics['sma25_dev'], 
                'trend_momentum': metrics['trend_momentum'],
                'macd_power': metrics['macd_power'],
                'entry_volatility': metrics['entry_volatility'],
                'profit_loss': profit_loss
            }
            memory_system.save_experience(save_data)
        else:
            print(f"   (様子見: {decision.get('reason')})")

        print("-" * 50)
        time.sleep(2) # APIレート制限対策

    print(f"\n=== 合宿終了 ===")
    print(f"戦績 (BUY/SELL): {win_count}勝 {loss_count}敗 {draw_count}分")

if __name__ == "__main__":
    main()
    auto_git_push(commit_message="Training Camp Result Update")