import yfinance as yf
import pandas as pd
import google.generativeai as genai
import json
import time
import datetime
import os
import random

# ==========================================
# ★設定エリア
# ==========================================
GOOGLE_API_KEY = "AIzaSyCHCZ4PznaZsaqmS0YLHPwPqXohr0tTyvw" 
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-flash')

# 先生役（学習データ）になってもらう銘柄
# トレンドが明確で学習効果が高い銘柄を選定
TRAINING_TICKER = "6501.T" # 日立製作所
START_DATE = "2024-01-01"
LOG_FILE = "ai_learning_log.csv"

# ==========================================
# 1. 記憶の読み書き機能
# ==========================================
def load_memory():
    """CSVから過去の教訓を読み出す"""
    if not os.path.exists(LOG_FILE): return ""
    
    try:
        df = pd.read_csv(LOG_FILE)
        wins = df[df['Result'] == 'WIN'].tail(5)  # 直近の勝ち5件
        losses = df[df['Result'] == 'LOSS'].tail(5) # 直近の負け5件
        
        text = "【あなたの過去の経験（学習データ）】\n"
        if not wins.empty:
            text += "■ 成功パターン（これを繰り返せ）:\n"
            for _, r in wins.iterrows():
                text += f"- {r['Date']}: {r['Action']}して勝利。理由: {r['Reason']}\n"
        if not losses.empty:
            text += "■ 失敗パターン（これを避けろ）:\n"
            for _, r in losses.iterrows():
                text += f"- {r['Date']}: {r['Action']}して敗北。理由: {r['Reason']}\n"
        return text
    except:
        return ""

def save_experience(date, action, price, reason, result, exit_price):
    """トレード結果を即座に記憶する"""
    data = {
        "Date": date,
        "Ticker": TRAINING_TICKER,
        "Name": "学習用データ",
        "Action": action,
        "Confidence": 90, # 学習用なので高めに仮定
        "Price": price,
        "Reason": reason,
        "Result": result,
        "ExitPrice": exit_price
    }
    df_new = pd.DataFrame([data])
    
    if not os.path.exists(LOG_FILE):
        df_new.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
    else:
        df_new.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

# ==========================================
# 2. テクニカル計算
# ==========================================
def get_technical_data(df, target_date):
    # target_date時点のデータを切り出す
    curr = df[df.index <= target_date].iloc[-1]
    
    price = int(curr['Close'])
    sma25 = curr['Close'] # 簡易計算（本来はRollingが必要だがdfには計算済みが入る想定）
    
    # トレンド判定などの文字列作成
    return f"株価:{price}円, SMA25:{int(curr['SMA25'])}円, MACD:{curr['MACD']:.2f}"

# ==========================================
# 3. AI判断（過去の記憶を参照しながら）
# ==========================================
def ai_practice(tech_text, memory_text):
    prompt = f"""
    あなたはトレーニング中のAIトレーダーです。
    過去の学習データ（成功/失敗）を参考にして、今回の判断を行ってください。

    {memory_text}

    【今回の状況】
    {tech_text}

    【指示】
    - 過去に似た失敗があれば、同じ轍を踏まないでください。
    - 過去に似た成功があれば、積極的に真似してください。

    【出力】JSONのみ: {{"action": "BUY" or "SELL", "reason": "短い理由"}}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        return {"action": "HOLD", "reason": "Error"}

# ==========================================
# メイン：高速育成ループ
# ==========================================
print("=== AI高速育成プログラム（スパルタ教育モード） ===")
print("過去の相場を走り抜けながら、経験値を蓄積します...\n")

# データ準備
df = yf.download(TRAINING_TICKER, start="2023-06-01", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 指標計算
df['SMA25'] = df['Close'].rolling(25).mean()
exp12 = df['Close'].ewm(span=12).mean()
exp26 = df['Close'].ewm(span=26).mean()
df['MACD'] = exp12 - exp26

# テスト期間の日付リスト（1週間おきにトレードさせる）
dates = df[df.index >= START_DATE].index[::5] 

win_count = 0
loss_count = 0

# もし古いログがあれば削除してリセット（新しい脳みそにする）
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
    print("※過去の記憶をリセットしました。\n")

for i in range(len(dates) - 5): # 判定用に未来5日分が必要なので -5
    current_date = dates[i]
    future_date = dates[i+1] # 1週間後の価格を見る
    
    # 1. その時点の記憶を呼び出す
    memory = load_memory()
    
    # 2. その時点のテクニカル情報
    tech_info = get_technical_data(df, current_date)
    
    # 3. AIに判断させる
    decision = ai_practice(tech_info, memory)
    action = decision.get('action', 'HOLD')
    reason = decision.get('reason', 'None')
    
    if action == "HOLD":
        print(f"📅 {current_date.date()}: スルー")
        continue

    # 4. 即座に答え合わせ（カンニング）
    current_price = int(df.loc[current_date]['Close'])
    future_price = int(df.loc[future_date]['Close'])
    
    result = "DRAW"
    if action == "BUY":
        if future_price > current_price: result = "WIN"
        else: result = "LOSS"
    elif action == "SELL":
        if future_price < current_price: result = "WIN"
        else: result = "LOSS"
    
    # 5. 結果を記憶（CSV）に書き込む
    save_experience(current_date.date(), action, current_price, reason, result, future_price)
    
    # ログ表示
    icon = "⭕" if result == "WIN" else "❌"
    print(f"📅 {current_date.date()} {action} -> {future_price}円 ({icon})")
    print(f"   理由: {reason}")
    
    if result == "WIN": win_count += 1
    else: loss_count += 1
    
    time.sleep(2) # API制限回避

print("\n" + "="*40)
print("【学習完了】")
print(f"戦績: {win_count}勝 {loss_count}敗")
print(f"生成された経験値データ: {LOG_FILE}")
print("このファイルを読み込ませることで、次回の本番トレードが賢くなります。")
print("="*40)