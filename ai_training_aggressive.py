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
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import re
import subprocess 
import logging
import socket
import requests.packages.urllib3.util.connection as urllib3_cn

# ---------------------------------------------------------
# ★環境設定
# ---------------------------------------------------------
# IPv4強制
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
GOOGLE_API_KEY = os.getenv("TRAINING_API_KEY") # または 
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")

# ★攻撃型V2用のファイルを使用
LOG_FILE = "ai_trade_memory_aggressive.csv" 
MODEL_NAME = 'models/gemini-2.0-flash'

TRAINING_ROUNDS = 2000 # トレーニング回数
TIMEFRAME = "1d"
CBR_NEIGHBORS_COUNT = 15
TRADE_BUDGET = 1000000 # 1トレードあたりの予算

# ★高ボラティリティ・トレンド銘柄リスト（半導体、グロース、主力大型）
# ★監視・トレーニング対象リスト (全100銘柄)
TRAINING_LIST = [
    # --- 1. 半導体・ハイテク (最重要・高ボラ) ---
    "6920.T", # レーザーテック (売買代金トップ常連)
    "8035.T", # 東京エレクトロン
    "6146.T", # ディスコ
    "6857.T", # アドバンテスト
    "7735.T", # スクリーン
    "6723.T", # ルネサス
    "6963.T", # ローム
    "3436.T", # SUMCO
    "6526.T", # ソシオネクスト
    "6315.T", # TOWA
    "6254.T", # 野村マイクロ

    # --- 2. 電気機器・電子部品 (世界景気連動) ---
    "6758.T", # ソニーG
    "6861.T", # キーエンス (値がさ株の王)
    "6981.T", # 村田製作所
    "6594.T", # ニデック (旧日本電産)
    "6954.T", # ファナック (ロボット)
    "6506.T", # 安川電機
    "6702.T", # 富士通
    "6752.T", # パナソニック
    "7751.T", # キヤノン
    "6501.T", # 日立製作所
    "6503.T", # 三菱電機

    # --- 3. 自動車・輸送用機器 (円安メリット) ---
    "7203.T", # トヨタ
    "7267.T", # ホンダ
    "7269.T", # スズキ
    "7270.T", # SUBARU (為替感応度高い)
    "7201.T", # 日産自動車
    "7259.T", # アイシン
    "6902.T", # デンソー

    # --- 4. 機械・重工・防衛 (地政学・インフラ) ---
    "7011.T", # 三菱重工 (防衛筆頭)
    "7013.T", # IHI
    "7012.T", # 川崎重工
    "6301.T", # コマツ (建機・中国関連)
    "6305.T", # 日立建機
    "6367.T", # ダイキン (空調世界一)
    "7003.T", # 三井E&S (造船・クレーン)

    # --- 5. 商社・卸売 (バフェット銘柄・高配当) ---
    "8058.T", # 三菱商事
    "8001.T", # 伊藤忠
    "8031.T", # 三井物産
    "8002.T", # 丸紅
    "8053.T", # 住友商事
    "2768.T", # 双日
    "7459.T", # メディパル (医薬品卸)

    # --- 6. 金融・銀行・保険 (金利テーマ) ---
    "8306.T", # 三菱UFJ
    "8316.T", # 三井住友FG
    "8411.T", # みずほFG
    "8766.T", # 東京海上
    "8725.T", # MS&AD
    "8591.T", # オリックス
    "8604.T", # 野村HD
    "8698.T", # マネックスG (暗号資産連動)

    # --- 7. 通信・サービス・AI (内需・グロース) ---
    "9984.T", # ソフトバンクG (AI投資会社)
    "9432.T", # NTT (ディフェンシブだが流動性高い)
    "9433.T", # KDDI
    "9434.T", # ソフトバンク
    "6098.T", # リクルート
    "2413.T", # エムスリー (グロース代表)
    "4661.T", # オリエンタルランド
    "4385.T", # メルカリ
    "4751.T", # サイバーエージェント
    "9613.T", # NTTデータ

    # --- 8. 小売・食品・消費 (インバウンド) ---
    "9983.T", # ファーストリテイリング (日経寄与度1位)
    "3382.T", # セブン＆アイ
    "8267.T", # イオン
    "2802.T", # 味の素
    "2914.T", # JT
    "4911.T", # 資生堂
    "4543.T", # テルモ
    "4503.T", # アステラス製薬
    "4568.T", # 第一三共 (がん治療薬で急伸)

    # --- 9. ゲーム・エンタメ (ヒット作で急騰) ---
    "7974.T", # 任天堂
    "9697.T", # カプコン
    "9766.T", # コナミG
    "5253.T", # カバー (ホロライブ)
    "9166.T", # GENDA

    # --- 10. 海運・鉄鋼・資源 (市況関連) ---
    "9101.T", # 日本郵船
    "9104.T", # 商船三井
    "9107.T", # 川崎汽船 (高ボラティリティ)
    "5401.T", # 日本製鉄
    "5411.T", # JFE
    "1605.T", # INPEX (原油)
    "5713.T", # 住友金属鉱山 (金・銅)
    "5020.T", # ENEOS
    "4063.T", # 信越化学
    "4901.T"  # 富士フイルム
]

plt.rcParams['font.family'] = 'sans-serif'
genai.configure(api_key=GOOGLE_API_KEY, transport="rest")

# ==========================================
# 1. データ取得 & テクニカル計算
# ==========================================
def download_data_safe(ticker, period="5y", interval="1d", retries=3): # 期間を長めに
    wait = 2
    for attempt in range(retries):
        try:
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            if len(df) < 100: return None
            return df
        except:
            time.sleep(wait); wait *= 2
    return None

def calculate_technical_indicators(df):
    df = df.copy()
    
    # 基本指標
    df['SMA25'] = df['Close'].rolling(25).mean()
    
    # DMI / ADX
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
    df['ADX'] = dx.rolling(14).mean()
    df['PlusDI'] = plus_di
    df['MinusDI'] = minus_di

    # ボリンジャーバンド & 出来高
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Width'] = ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20 * 100
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    
    # ATR
    df['ATR'] = tr.rolling(14).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    rs = gain / loss
    df['RSI9'] = 100 - (100 / (1 + rs))

    return df.dropna()

def calculate_metrics_for_training(df, idx):
    curr = df.iloc[idx]
    price = float(curr['Close'])
    
    # 抵抗線（過去60日高値）
    past_60 = df.iloc[idx-60:idx]
    recent_high = past_60['High'].max()
    dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
    
    # ADXトレンド
    adx = float(curr['ADX'])
    prev_adx = float(df['ADX'].iloc[idx-1])
    
    # MA乖離
    sma25 = float(curr['SMA25'])
    ma_deviation = ((price / sma25) - 1) * 100
    
    # BB拡大率
    bb_width = float(curr['BB_Width'])
    prev_width = float(df['BB_Width'].iloc[idx-5]) if df['BB_Width'].iloc[idx-5] > 0 else 0.1
    expansion_rate = bb_width / prev_width
    
    # 出来高倍率
    vol_ma20 = float(curr['Vol_MA20'])
    vol_ratio = float(curr['Volume']) / vol_ma20 if vol_ma20 > 0 else 0
    
    # RS (簡易版: 市場データとの比較は省略し、自身のモメンタムで代用)
    # 本番では市場との比較を行うが、トレーニングでは「強い動き」かどうかを見る
    rsi_9 = float(curr['RSI9'])
    
    return {
        'price': price,
        'resistance_price': recent_high,
        'dist_to_res': dist_to_res,
        'ma_deviation': ma_deviation,
        'adx': adx,
        'prev_adx': prev_adx,
        'plus_di': float(curr['PlusDI']),
        'minus_di': float(curr['MinusDI']),
        'rs_rating': 0.0, # トレーニングでは省略
        'vol_ratio': vol_ratio,
        'expansion_rate': expansion_rate,
        'atr_value': float(curr['ATR']),
        'rsi_9': rsi_9
    }

# ==========================================
# 2. CBRメモリシステム (攻撃型V2対応)
# ==========================================
class CaseBasedMemory:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.knn = None
        self.df = pd.DataFrame()
        # ★特徴量を攻撃型に合わせて変更
        self.feature_cols = ['adx', 'prev_adx', 'ma_deviation', 'vol_ratio', 'expansion_rate', 'dist_to_res']
        
        # ★保存カラム (V2仕様)
        self.csv_columns = [
            "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
            "Confidence", "stop_loss_price", "target_price", 
            "Price", 
            "adx", "prev_adx", "ma_deviation", "rs_rating", 
            "vol_ratio", "expansion_rate", 
            "dist_to_res", "days_to_earnings", "margin_ratio", 
            "profit_rate"
        ]
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path)
            # カラム補完
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

    def save_experience(self, data_dict):
        new_df = pd.DataFrame([data_dict])
        # カラム順序を揃える
        for col in self.csv_columns:
            if col not in new_df.columns: new_df[col] = None
        new_df = new_df[self.csv_columns]

        try:
            if not os.path.exists(self.csv_path):
                new_df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
            else:
                new_df.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            self.load_and_train() 
        except Exception as e:
            print(f"❌ 保存エラー: {e}")

# ==========================================
# 3. AIスパーリング
# ==========================================
def create_chart_image(df, name):
    data = df.tail(80).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    ax1.plot(data.index, data['Close'], color='black', label='Close')
    ax1.plot(data.index, sma20 + 2*std20, color='green', alpha=0.5, linestyle='--', label='+2σ')
    ax1.plot(data.index, sma20 - 2*std20, color='green', alpha=0.5, linestyle='--', label='-2σ')
    ax1.set_title(f"{name} Training Chart")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def ai_decision_maker(model, chart_bytes, metrics, cbr_text, ticker):
    # --- フィルタリング ---
    if metrics['adx'] < 20:
         return {"action": "HOLD", "reason": "【鉄の掟】トレンドレス (ADX<20)"}
    if metrics['vol_ratio'] < 0.8:
         return {"action": "HOLD", "reason": "【鉄の掟】出来高不足"}

    # ★攻撃型プロンプト (トレーニング用)
    prompt = f"""
### ROLE
あなたは「高ボラティリティ・トレンドフォロー特化型AI」です。
小さな利益は無視し、発生し始めた「大きなトレンド（急騰）」や「ブレイクアウト」のみを捕捉します。

### INPUT DATA
銘柄: {ticker} (現在価格: {metrics['price']:.0f}円)

[テクニカル指標]
1. Trend Strength (ADX): {metrics['adx']:.1f} (閾値: 25以上, 前日: {metrics['prev_adx']:.1f})
2. Direction (+DI/-DI): +DI({metrics['plus_di']:.1f}) vs -DI({metrics['minus_di']:.1f})
3. Volatility (BB Exp): {metrics['expansion_rate']:.2f}倍
4. Volume Flow: {metrics['vol_ratio']:.2f}倍
5. MA Deviation: {metrics['ma_deviation']:.2f}% (過熱感チェック)

[重要コンテキスト]
- **抵抗線位置**: {metrics['resistance_price']:.0f}円 (現在価格との差: {metrics['dist_to_res']:.1f}%)
{cbr_text}

### EVALUATION LOGIC
1. **ブレイクアウト判定**:
   - 抵抗線(resistance_price)を価格が上回っている、または抵抗線での攻防を制しつつあるか？
   - 抵抗線の直前(差が0〜1%程度)で止まっている場合は "HOLD" (反落リスク)。
   - 抵抗線を超えていれば "BUY" の確度アップ。
   
2. **過熱感チェック**:
   - MA乖離率(ma_deviation)が +30% を超えている場合は "HOLD" (高値掴み警戒)。

### OUTPUT REQUIREMENT (JSON ONLY)
{{
  "action": "BUY" or "HOLD",
  "confidence": 0-100,
  "stop_loss": "推奨する損切り価格（整数）",
  "target_price": "推奨する利確目標価格（整数）",
  "reason": "判断理由(50文字以内)"
}}
"""
    safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    try:
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': chart_bytes}], safety_settings=safety)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: text = match.group(0)
        return json.loads(text)
    except Exception as e:
        return {"action": "HOLD", "reason": f"AI Error: {e}", "confidence": 0}

# ==========================================
# 4. メイン実行 (トレーニングモード)
# ==========================================
def main():
    start_time = time.time()
    print(f"=== AI強化合宿 [AGGRESSIVE MODE] ===")
    
    memory_system = CaseBasedMemory(LOG_FILE) 
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e: print(f"Model Init Error: {e}"); return

    processed_data = {}
    print(f"データをダウンロード中...")
    for t in TRAINING_LIST:
        df = download_data_safe(t,period="10y", interval=TIMEFRAME)
        if df is None: continue
        df = calculate_technical_indicators(df)
        processed_data[t] = df

    if not processed_data: print("データ不足のため終了します。"); return

    win_count = 0; loss_count = 0
    total_profit_loss = 0.0 
    
    print(f"\n🥊 トレーニング開始 ({TRAINING_ROUNDS}ラウンド)\n")
    
    for i in range(1, TRAINING_ROUNDS + 1):
        ticker = random.choice(list(processed_data.keys()))
        df = processed_data[ticker]
        if len(df) < 110: continue 
        
        # 過去のランダムな時点を選択
        target_idx = random.randint(100, len(df) - 65) # 未来データ確保のため-65
        current_date_str = df.index[target_idx].strftime('%Y-%m-%d')
        
        metrics = calculate_metrics_for_training(df, target_idx)
        
        cbr_text = memory_system.search_similar_cases(metrics)
        past_df = df.iloc[:target_idx+1]
        chart_bytes = create_chart_image(past_df, ticker)
        
        decision = ai_decision_maker(model_instance, chart_bytes, metrics, cbr_text, ticker)
        action = decision.get('action', 'HOLD')
        conf = decision.get('confidence', 0)

        # 鉄の掟やAIの判断でHOLDならスキップ
        if action == "HOLD": 
            # HOLDでも稀にログ出力（生存確認用）
            if i % 20 == 0: print(f"Round {i:03}: {ticker} -> HOLD ({decision.get('reason')})")
            continue

        print(f"Round {i:03}: {ticker} ({current_date_str}) -> BUY 🔴 (自信:{conf}%)")

        # --- シミュレーション ---
        entry_price = float(metrics['price'])
        atr = metrics['atr_value']
        
        # AI提案のSLがあれば採用、なければ ATR x 2.5
        ai_stop = decision.get('stop_loss', 0)
        try: ai_stop = int(ai_stop)
        except: ai_stop = 0
        current_stop_loss = ai_stop if ai_stop > 0 else entry_price - (atr * 2.0)
        
        shares = int(TRADE_BUDGET // entry_price)
        if shares < 1: shares = 1
        
        # 未来60日のデータを取得
        future_prices = df.iloc[target_idx+1 : target_idx+61]
        
        result = "DRAW"; profit_loss = 0.0; final_exit_price = entry_price
        max_price = entry_price
        
        is_loss = False
        
        # 日ごとの値動きを追跡
        for _, row in future_prices.iterrows():
            high = row['High']; low = row['Low']; close = row['Close']
            
            # 1. 損切り判定
            if low <= current_stop_loss:
                is_loss = True
                final_exit_price = current_stop_loss
                break
            
            # 2. トレーリングストップ判定（攻撃型）
            if high > max_price:
                max_price = high
                profit_pct = (max_price - entry_price) / entry_price
                
                # 利益が乗ってきたらストップを引き上げる
                trail_dist = atr * 2.5 # 標準
                if profit_pct > 0.10: trail_dist = atr * 1.0 # +10%超えでタイトに
                elif profit_pct > 0.20: trail_dist = atr * 0.5 # +20%超えで超タイトに
                
                new_stop = max_price - trail_dist
                # 建値決済保証 (+3%乗ったら建値以上にSLを置く)
                if profit_pct > 0.03:
                    new_stop = max(new_stop, entry_price * 1.005)
                
                if new_stop > current_stop_loss:
                    current_stop_loss = new_stop

        if not is_loss:
            final_exit_price = future_prices['Close'].iloc[-1]

        profit_loss = (final_exit_price - entry_price) * shares
        profit_rate = ((final_exit_price - entry_price) / entry_price) * 100
        
        if profit_loss > 0: result = "WIN"; win_count += 1
        elif profit_loss < 0: result = "LOSS"; loss_count += 1

        print(f"   結果: {result} (PL: {profit_loss:+.0f}円 / {profit_rate:+.2f}%) > {decision.get('reason')}")

        # --- 結果保存 ---
        save_data = {
            'Date': current_date_str, 'Ticker': ticker, 'Timeframe': TIMEFRAME, 
            'Action': action, 'result': result, 
            'Reason': decision.get('reason', 'None'),
            'Confidence': conf, 
            'stop_loss_price': current_stop_loss, 
            'target_price': decision.get('target_price', 0), 
            'Price': metrics['price'], 
            'adx': metrics['adx'], 
            'prev_adx': metrics['prev_adx'],
            'ma_deviation': metrics['ma_deviation'], 
            'rs_rating': 0, # トレーニングでは省略
            'vol_ratio': metrics['vol_ratio'], 
            'expansion_rate': metrics['expansion_rate'],
            'dist_to_res': metrics['dist_to_res'],       
            'days_to_earnings': 999, # トレーニングでは省略
            'margin_ratio': 1.0,     # トレーニングでは省略
            'profit_rate': profit_rate 
        }
        memory_system.save_experience(save_data)
        time.sleep(1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
    print(f"\n=== 合宿終了 ===")
    print(f"戦績 (BUY): {win_count}勝 {loss_count}敗")
    print(f"合計損益: {total_profit_loss:+.0f}円")
    print(f"合宿時間: {elapsed_str}")

if __name__ == "__main__":
    main()