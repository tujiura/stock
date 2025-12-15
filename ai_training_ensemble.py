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
from concurrent.futures import ThreadPoolExecutor
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import re
import logging
import socket
import requests.packages.urllib3.util.connection as urllib3_cn

# ---------------------------------------------------------
# ★環境設定
# ---------------------------------------------------------
def allowed_gai_family():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = allowed_gai_family

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("エラー: GOOGLE_API_KEY が設定されていません。")

# ログファイル名
LOG_FILE = "ai_trade_memory_ensemble_full.csv"
MODEL_NAME = 'models/gemini-2.0-flash'

TRAINING_ROUNDS = 100
TIMEFRAME = "1d"
TRADE_BUDGET = 1000000 

TRAINING_LIST = [
    "6254.T", "8035.T", "6146.T", "6920.T", "6857.T", "7735.T", "6723.T", 
    "6758.T", "6861.T", "6594.T", "6954.T", "7751.T", "6501.T",
    "7203.T", "7267.T", "7011.T", "7013.T", "6301.T", "8058.T", "8001.T", 
    "8306.T", "8316.T", "9984.T", "9432.T", "6098.T", "9983.T",
    "9101.T", "9104.T", "9107.T", "5401.T", "1605.T", "4063.T"
]

plt.rcParams['font.family'] = 'sans-serif'
genai.configure(api_key=GOOGLE_API_KEY, transport="rest")

# ==========================================
# 1. データ取得 & 指標計算
# ==========================================
def download_data_safe(ticker, period="5y", interval="1d", retries=3): 
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
    df['SMA25'] = df['Close'].rolling(25).mean()
    close = df['Close']
    
    high = df['High']; low = df['Low']
    tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    tr_s = tr.rolling(14).mean()
    p_dm_s = plus_dm.rolling(14).mean()
    m_dm_s = minus_dm.rolling(14).mean()

    df['PlusDI'] = 100 * (p_dm_s / tr_s)
    df['MinusDI'] = 100 * (m_dm_s / tr_s)
    dx = (abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])) * 100
    df['ADX'] = dx.rolling(14).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Width'] = ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20 * 100
    df['BB_Min60'] = df['BB_Width'].rolling(60).min()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['ATR'] = tr.rolling(14).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    rs = gain / loss
    df['RSI9'] = 100 - (100 / (1 + rs))

    return df.dropna()

def get_metrics_at_index(df, idx):
    curr = df.iloc[idx]
    price = float(curr['Close'])
    recent_high = df['High'].iloc[idx-60:idx].max()
    dist_to_res = ((price - recent_high) / recent_high) * 100 if recent_high > 0 else 0
    bb_width = float(curr['BB_Width'])
    bb_min_60 = float(curr['BB_Min60'])
    historical_volatility_rank = bb_width / bb_min_60 if bb_min_60 > 0 else 1.0
    prev_width = float(df['BB_Width'].iloc[idx-5]) if df['BB_Width'].iloc[idx-5] > 0 else 0.1
    expansion_rate = bb_width / prev_width

    return {
        'price': price,
        'adx': float(curr['ADX']),
        'plus_di': float(curr['PlusDI']),
        'minus_di': float(curr['MinusDI']),
        'vol_ratio': float(curr['Volume']) / float(curr['Vol_MA20']) if curr['Vol_MA20'] > 0 else 0,
        'ma_deviation': ((price / float(curr['SMA25'])) - 1) * 100,
        'rsi_9': float(curr['RSI9']),
        'expansion_rate': expansion_rate,
        'historical_volatility_rank': historical_volatility_rank,
        'dist_to_res': dist_to_res,
        'atr_value': float(curr['ATR'])
    }

# ==========================================
# 2. AIエージェント定義 & 並列実行
# ==========================================
def call_gemini_json(model, prompt, image_bytes=None):
    safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    try:
        inputs = [prompt]
        if image_bytes:
            inputs.append({'mime_type': 'image/png', 'data': image_bytes})
        response = model.generate_content(inputs, safety_settings=safety)
        text = response.text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
        return {"action": "HOLD", "confidence": 0, "reason": "JSON Parse Error"}
    except Exception as e:
        return {"action": "HOLD", "confidence": 0, "reason": f"API Error: {e}"}

def run_ensemble_analysis(model, metrics, chart_bytes, ticker):
    # 疑似ファンダメンタル生成
    base_rs = 50

    """
    if metrics['adx'] > 25 and metrics['plus_di'] > metrics['minus_di']:
        base_rs = random.randint(60, 99)
    else:
        base_rs = random.randint(10, 60)
    """
    """ 
    macro_states = ["上昇トレンド", "レンジ相場", "下落トレンド", "不安定(VIX高)"]
    """
    current_macro = ["none"]
    """
    fund_data = {
        'days_to_earnings': random.choice([2, 5, 15, 30, 60]), 
        'margin_ratio': random.uniform(0.5, 12.0),
        'rs_rating': base_rs,
        'macro': current_macro,
        'news': "特になし"
    }
    """
    fund_data = {
        'days_to_earnings': 60, 
        'margin_ratio': 5.0,
        'rs_rating': base_rs,
        'macro': current_macro,
        'news': "特になし"
    }
    
    if fund_data['margin_ratio'] > 10.0:
        fund_data['news'] = "信用買い残が高水準で上値重い"

    # 🤖 AI A: トレンドフォロー (順張り特化)
    prompt_a = f"""
    ### ROLE
    あなたは「順張りトレンドフォロー特化型AI」です。
    「頭と尻尾はくれてやれ」。底値拾いはせず、明確な上昇トレンドが発生している中間部分だけを狙います。
    レンジ相場や下降トレンドは徹底的に無視（HOLD）します。

    ### INPUT DATA
    1. ADX (トレンド強度): {metrics['adx']:.1f} (基準: 25以上でトレンド発生)
    2. DI方向性: +DI({metrics['plus_di']:.1f}) vs -DI({metrics['minus_di']:.1f})
    3. Vol Ratio (出来高変化): {metrics['vol_ratio']:.2f}倍 (1.0以上で活況)
    4. MA Deviation (移動平均乖離): {metrics['ma_deviation']:.2f}%

    ### STRATEGY (A: 順張り)
    以下の論理で判断せよ。

    1. **必須条件 (BUYの最低ライン)**:
        - +DI が -DI より大きいこと。
        - MA Deviation が プラスであること（価格が移動平均線より上）。

    2. **評価基準**:
        - **ADX**: 25未満は「トレンドなし」としてHOLD推奨。30以上で強い確信を持つ。
        - **Volume**: 出来高の裏付け（Vol Ratio > 1.0）があればconfidenceを加点。
        - **過熱感**: MA乖離が大きすぎる(例えば20%以上)場合は「高値掴みリスク」としてconfidenceを少し下げる。
    {{ "action": "BUY"/"HOLD", "confidence": 0-100, "reason": "30字以内" }}
    """

    # 🤖 AI B: ボラティリティ・スクイーズ (ブレイクアウト特化)
    prompt_b = f"""
    ### ROLE
    あなたは「ボラティリティ・スクイーズ特化型AI」です。
    価格変動が極端に小さくなった「静寂」からの「爆発（初動）」のみを狙います。
    すでに大きく動いてしまった後の「飛び乗り」は厳禁です。

    ### INPUT DATA
    1. BB Expansion (バンド拡大率): {metrics['expansion_rate']:.2f}倍 (1.0=変化なし, 1.2以上=拡大開始)
    2. Squeeze Rank (エネルギー蓄積): {metrics['historical_volatility_rank']:.2f} (0.0=拡散中, 1.0=極度の煮詰まり)
    3. 抵抗線距離: {metrics['dist_to_res']:.1f}% (0に近いほどブレイク直前)

    ### STRATEGY (B: ブレイクアウト)
    以下の論理で判断せよ。

    1. **セットアップ (準備)**:
        - Squeeze Rankが高い(0.8以上など)場合、エネルギーが十分に溜まっていると評価する。

    2. **トリガー (発火)**:
        - BB Expansionが「拡大し始めた」瞬間(例: 1.1〜1.3倍付近)を狙う。
        - 逆に、Expansionがすでに大きすぎる(1.5倍以上など)場合は「手遅れ」としてHOLD。

    3. **位置取り**:
        - 抵抗線距離が近い(数%以内)場合、レジスタンスブレイクの期待値を高く見積もる。

    ### OUTPUT (JSON)
    {{ "action": "BUY"/"HOLD", "confidence": 0-100, "reason": "30字以内" }}
    """

    prompt_c = f"""
    ### ROLE
    あなたは「リスク・フィルタリング特化型AI」です。
    あなたの仕事は「買い推奨」することではなく、致命的な欠陥がある場合のみ「拒否権(VETO)」を行使することです。
    致命的なリスクが見当たらない場合は、静かに「承認(ALLOW)」してください。

    ### INPUT DATA
    銘柄: {ticker}
    1. RS Rating (0-99の相対強度): {fund_data['rs_rating']}
    2. 決算発表: {fund_data['days_to_earnings']}日後
    3. 信用倍率: {fund_data['margin_ratio']:.2f}倍
    4. ニュース: {fund_data['news']}
    5. マクロ環境: {fund_data['macro']}

    ### STRATEGY (C: 消去法)
    以下の「即レッドカード条件」に該当するか確認せよ。該当しなければすべて ALLOW とする。

    1. **レッドカード条件 (VETO)**:
        - 決算発表まで「3日以内」である。
        - ニュースに「不祥事」「粉飾」「倒産懸念」「特設注意市場」などの明確な破滅的ワードが含まれる。
        - 信用倍率が「15倍」を超えている（極端な需給悪化）。

    2. **イエローカード判断**:
        - RS Ratingが低い、またはマクロが悪いだけでは VETO しない（それは他のAIの判断に委ねる）。
        - ただし、リスク要因が複数重なる場合のみ VETO を検討する。
        
    ### OUTPUT (JSON)
    {{ "action": "VETO"/"ALLOW", "confidence": 0-100, "risk_factor": 0.0-1.0, "reason": "30字以内" }}
    """

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_a = executor.submit(call_gemini_json, model, prompt_a, chart_bytes)
        future_b = executor.submit(call_gemini_json, model, prompt_b, chart_bytes)
        future_c = executor.submit(call_gemini_json, model, prompt_c, None)
        res_a, res_b, res_c = future_a.result(), future_b.result(), future_c.result()

    decision = integrate_decisions(res_a, res_b, res_c)
    return decision, fund_data, res_a, res_b, res_c

def integrate_decisions(res_a, res_b, res_c):
    # ★ ここを修正: 安全に値を取り出すヘルパー関数
    def get_conf(res):
        if not isinstance(res, dict): return 0
        return res.get('confidence', res.get('Confidence', 0))

    risk_factor = res_c.get('risk_factor', 0.0)
    
    # 1. AI C (Risk) の拒否権
    if res_c.get('action') == "VETO" or risk_factor >= 0.8:
        return {
            "final_action": "HOLD",
            "final_confidence": 0,
            "reason": f"⛔ [Risk:{risk_factor:.1f}] {res_c.get('reason')}"
        }

    # 2. コンセンサス確認
    is_a_buy = res_a.get('action') == "BUY"
    is_b_buy = res_b.get('action') == "BUY"

    if is_a_buy and is_b_buy:
        # ヘルパー関数を使って安全に計算
        conf_a = get_conf(res_a)
        conf_b = get_conf(res_b)
        
        base_conf = (conf_a * 0.4) + (conf_b * 0.6)
        penalty_multiplier = 1.0 - (risk_factor * 0.5)
        final_conf = int(base_conf * penalty_multiplier)
        
        return {
            "final_action": "BUY",
            "final_confidence": final_conf,
            "reason": f"🚀 [合意] 勢い&凝縮"
        }
    else:
        blocker = "Trend" if not is_a_buy else "Energy"
        return {
            "final_action": "HOLD",
            "final_confidence": 0,
            "reason": f"⏸ [不一致] {blocker}不足"
        }

# ==========================================
# 3. メイン実行
# ==========================================
def main():
    start_time = time.time()
    print(f"=== AIアンサンブル学習 [Robust Fix Mode] ===")
    
    try: model_instance = genai.GenerativeModel(MODEL_NAME)
    except Exception as e: print(f"Model Init Error: {e}"); return

    processed_data = {}
    print(f"データをダウンロード中...")
    for i, t in enumerate(TRAINING_LIST):
        if i % 10 == 0: print(f"  - {i}/{len(TRAINING_LIST)}")
        df = download_data_safe(t)
        if df is None: continue
        df = calculate_technical_indicators(df)
        processed_data[t] = df

    if not processed_data: return
    print(f"準備完了 ({int(time.time() - start_time)}秒)")
    
    csv_columns = [
        "Date", "Ticker", "Action", "result", "Confidence", "Reason", "profit_rate",
        "adx", "plus_di", "minus_di", "vol_ratio", "ma_deviation", "rsi_9",
        "expansion_rate", "historical_volatility_rank", "dist_to_res", "atr_value",
        "days_to_earnings", "margin_ratio", "rs_rating", "macro",
        "AI_A_Action", "AI_A_Conf", "AI_B_Action", "AI_B_Conf", "AI_C_Action", "AI_C_Conf", "AI_C_Risk", "AI_C_Reason"
    ]
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=csv_columns).to_csv(LOG_FILE, index=False)

    win_count = 0; loss_count = 0; total_pl = 0
    
    print(f"\n🥊 アンサンブル・トレーニング開始 ({TRAINING_ROUNDS}ラウンド)\n")
    
    for i in range(1, TRAINING_ROUNDS + 1):
        ticker = random.choice(list(processed_data.keys()))
        df = processed_data[ticker]
        if len(df) < 110: continue 
        
        target_idx = random.randint(100, len(df) - 65) 
        metrics = get_metrics_at_index(df, target_idx)
        
        data = df.tail(80).copy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.plot(data.index, data['Close'], color='black')
        ax1.plot(data.index, data['SMA25'], color='orange', linestyle='--')
        ax2.bar(data.index, data['Volume'], color='gray')
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close(fig); buf.seek(0)
        chart_bytes = buf.getvalue()
        
        decision, fund_data, res_a, res_b, res_c = run_ensemble_analysis(model_instance, metrics, chart_bytes, ticker)
        
        action = decision['final_action']
        conf = decision['final_confidence']
        reason = decision['reason']

        # ★ ログ表示ロジック
        should_print = False
        if action == "BUY":
            should_print = True
        elif "Risk" in reason: 
            should_print = True
        elif res_a.get('action') == "BUY" and res_b.get('action') == "BUY": 
            should_print = True
        
        if not should_print: continue

        print(f"Round {i:03}: {ticker} -> {action} ({reason})")
        print(f"   🤖 Trend(A): {res_a.get('action')} ({res_a.get('confidence')}%)")
        print(f"   🤖 Squeez(B): {res_b.get('action')} ({res_b.get('confidence')}%)")
        print(f"   🤖 Risk (C): {res_c.get('action')} (Risk:{res_c.get('risk_factor',0):.1f}) > {res_c.get('reason')}")

        if action == "HOLD":
            print("-" * 30)
            continue

        # --- シミュレーション ---
        entry_price = metrics['price']
        atr = metrics['atr_value']
        current_stop = entry_price - (atr * 1.8)
        shares = int(TRADE_BUDGET // entry_price)
        if shares < 1: shares = 1
        
        future = df.iloc[target_idx+1 : target_idx+61]
        result = "DRAW"; pl = 0; exit_price = entry_price
        max_price = entry_price
        is_loss = False
        
        for _, row in future.iterrows():
            if row['Low'] <= current_stop:
                is_loss = True; exit_price = current_stop; break
            if row['High'] > max_price:
                max_price = row['High']
                pct = (max_price - entry_price) / entry_price
                trail = atr * 1.8
                if pct > 0.05: trail = atr * 1.0
                new_stop = max_price - trail
                if pct > 0.02: new_stop = max(new_stop, entry_price * 1.002)
                if new_stop > current_stop: current_stop = new_stop

        if not is_loss: exit_price = future['Close'].iloc[-1]
        
        pl = (exit_price - entry_price) * shares
        rate = (exit_price - entry_price) / entry_price * 100
        
        if pl > 0: result = "WIN"; win_count += 1
        elif pl < 0: result = "LOSS"; loss_count += 1
        total_pl += pl
        
        print(f"   🏁 結果: {result} ({rate:+.2f}%)")
        print("-" * 30)
        
        # 保存
        log_data = {
            "Date": df.index[target_idx].strftime('%Y-%m-%d'), "Ticker": ticker, "Action": action, 
            "result": result, "Confidence": conf, "Reason": reason, "profit_rate": rate,
            "adx": metrics['adx'], "plus_di": metrics['plus_di'], "minus_di": metrics['minus_di'], 
            "vol_ratio": metrics['vol_ratio'], "ma_deviation": metrics['ma_deviation'], "rsi_9": metrics['rsi_9'],
            "expansion_rate": metrics['expansion_rate'], "historical_volatility_rank": metrics['historical_volatility_rank'], 
            "dist_to_res": metrics['dist_to_res'], "atr_value": metrics['atr_value'],
            "days_to_earnings": fund_data['days_to_earnings'], "margin_ratio": fund_data['margin_ratio'], 
            "rs_rating": fund_data['rs_rating'], "macro": fund_data['macro'],
            "AI_A_Action": res_a.get('action'), "AI_A_Conf": res_a.get('confidence'),
            "AI_B_Action": res_b.get('action'), "AI_B_Conf": res_b.get('confidence'),
            "AI_C_Action": res_c.get('action'), "AI_C_Conf": res_c.get('confidence'), 
            "AI_C_Risk": res_c.get('risk_factor', 0), "AI_C_Reason": res_c.get('reason'),
        }
        
        save_df = pd.DataFrame([log_data])
        save_df = save_df[csv_columns]
        save_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
        time.sleep(1)

    print(f"\n=== 終了 ===")
    print(f"戦績: {win_count}勝 {loss_count}敗")
    print(f"損益: {total_pl:+.0f}円")

if __name__ == "__main__":
    main()