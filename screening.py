import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# --- 設定 ---
# 調査対象（今回は業種を散らして配置）
tickers = [
    '9984.T', '7203.T', '6758.T', '8306.T', '9101.T', 
    '6857.T', '6146.T', '4502.T', '7974.T', '9432.T',
    '7011.T', '5401.T' 
]

def calculate_rsi(series, period=14):
    """RSI（相対力指数）を計算する関数"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

print(f"--- 分析開始: {datetime.date.today()} ---")
results = []

for ticker in tickers:
    try:
        # データ取得（半年分）
        df = yf.download(ticker, period='6mo', progress=False)
        
        # マルチインデックス対策
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 30: continue

        # --- 1. テクニカル指標の計算 ---
        
        # A. 移動平均 (SMA)
        df['SMA25'] = df['Close'].rolling(window=25).mean()
        
        # B. ボリンジャーバンド (±2σ)
        std = df['Close'].rolling(window=25).std()
        df['Upper'] = df['SMA25'] + (2 * std)
        df['Lower'] = df['SMA25'] - (2 * std)
        
        # C. MACD (12, 26, 9)
        # EMA（指数平滑移動平均）を使うのが正式
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # D. RSI (14日)
        df['RSI'] = calculate_rsi(df['Close'], period=14)

        # 直近データの取得
        today = df.iloc[-1]
        prev = df.iloc[-2]

        # --- 2. ロジック判定（シグナル探し） ---
        
        signal_type = "なし"
        
        # 【戦略1：順張りトレンドフォロー】
        # 条件: MACDがゴールデンクロス かつ RSIが50以上（勢いがある）
        macd_cross = (prev['MACD'] < prev['Signal']) and (today['MACD'] > today['Signal'])
        trend_ok = today['RSI'] > 50
        
        if macd_cross and trend_ok:
            signal_type = "★上昇トレンド入り (MACD GC)"

        # 【戦略2：逆張りリバウンド狙い】
        # 条件: RSIが30以下（売られすぎ） かつ 価格がボリンジャー-2σ以下
        oversold = today['RSI'] < 30
        band_touch = today['Close'] <= today['Lower']
        
        if oversold or band_touch:
             # 両方満たせば最強、片方でも警戒
            if oversold and band_touch:
                signal_type = "★激アツ反発狙い (売られすぎ+底値)"
            elif oversold:
                signal_type = "反発監視 (RSI低)"

        # シグナルが出ている場合のみリストに追加
        if signal_type != "なし":
            results.append({
                '銘柄': ticker,
                'シグナル': signal_type,
                '現在値': int(today['Close']),
                'RSI': round(today['RSI'], 1),
                'MACD': round(today['MACD'], 2)
            })
            print(f"ヒット: {ticker} -> {signal_type}")

    except Exception as e:
        print(f"Error {ticker}: {e}")

# --- 結果表示 ---
print("\n" + "="*50)
if results:
    df_res = pd.DataFrame(results)
    # 見やすいようにカラムを並べ替え
    print(df_res[['銘柄', 'シグナル', '現在値', 'RSI', 'MACD']])
else:
    print("現在、条件に合致する銘柄はありません。")
print("="*50)