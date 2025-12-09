import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. データ取得と前処理 ---
ticker = '7203.T'  # トヨタ自動車
df = yf.download(ticker, period='3y', progress=False)

# マルチインデックス対策
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# --- 2. 指標計算 (MACD) ---
# 短期EMA(12), 長期EMA(26)
exp12 = df['Close'].ewm(span=12, adjust=False).mean()
exp26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp12 - exp26
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# --- 3. シミュレーション処理 ---
# 変数の初期化
initial_capital = 1_000_000  # 元手100万円
cash = initial_capital
position = 0                 # 保有株数
trade_log = []               # 取引履歴
equity_curve = []            # 資産推移

# ループ処理（1日ずつ進める）
for i in range(1, len(df)-1):
    today = df.iloc[i]
    prev = df.iloc[i-1]
    
    # 次の日の始値で売買すると仮定（シグナルは当日の終値で確定するため）
    next_open = df.iloc[i+1]['Open']
    date = df.index[i+1]

    # 現在の資産総額（現金 + 株の評価額）
    current_equity = cash + (position * today['Close'])
    equity_curve.append(current_equity)

    # --- 買いシグナル (ゴールデンクロス) ---
    # MACDがSignalを下から上に抜けた & ノートレード状態(position==0)なら買う
    if (prev['MACD'] < prev['Signal']) and (today['MACD'] > today['Signal']) and (position == 0):
        # 買えるだけ買う（100株単位で計算するのは複雑になるので、今回は1株単位で全力買いする簡易計算）
        buy_shares = int(cash // next_open)
        cost = buy_shares * next_open
        
        if buy_shares > 0:
            cash -= cost
            position += buy_shares
            trade_log.append({
                'Type': 'BUY',
                'Date': date,
                'Price': next_open,
                'Shares': buy_shares,
                'Total': -cost
            })

    # --- 売りシグナル (デッドクロス) ---
    # MACDがSignalを上から下に抜けた & 株を持っている(position>0)なら売る
    elif (prev['MACD'] > prev['Signal']) and (today['MACD'] < today['Signal']) and (position > 0):
        revenue = position * next_open
        cash += revenue
        trade_log.append({
            'Type': 'SELL',
            'Date': date,
            'Price': next_open,
            'Shares': position,
            'Total': revenue
        })
        position = 0 # 全売却

# --- 4. 結果集計 ---
# 最終日時点の資産
final_equity = cash + (position * df.iloc[-1]['Close'])
total_return = (final_equity - initial_capital) / initial_capital * 100

print(f"=== バックテスト結果: {ticker} (3年間) ===")
print(f"元手: {initial_capital:,} 円")
print(f"最終: {int(final_equity):,} 円")
print(f"収益率: {total_return:.2f} %")

# 勝率の計算（売りトレードのみ抽出）
sells = [t for t in trade_log if t['Type'] == 'SELL']
wins = 0
for i in range(len(sells)):
    # 売り価格 > 直前の買い価格 なら勝ち
    buy_trade = trade_log[trade_log.index(sells[i]) - 1]
    if sells[i]['Price'] > buy_trade['Price']:
        wins += 1

win_rate = (wins / len(sells)) * 100 if len(sells) > 0 else 0
print(f"トレード回数: {len(sells)} 回")
print(f"勝率: {win_rate:.2f} %")

# --- 5. グラフ化 ---
plt.figure(figsize=(12, 6))
# 株価
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Close'], label='Price', color='black', alpha=0.5)
# 売買ポイントをプロット
buys = pd.DataFrame([t for t in trade_log if t['Type'] == 'BUY'])
sells_df = pd.DataFrame([t for t in trade_log if t['Type'] == 'SELL'])

if not buys.empty:
    plt.scatter(buys['Date'], buys['Price'], marker='^', color='red', s=100, label='Buy')
if not sells_df.empty:
    plt.scatter(sells_df['Date'], sells_df['Price'], marker='v', color='blue', s=100, label='Sell')
plt.legend()
plt.title('Stock Price & Trade Points')

# 資産曲線のプロット
plt.subplot(2, 1, 2)
# 長さを合わせるために調整
plt.plot(df.index[1:len(equity_curve)+1], equity_curve, label='Equity', color='green')
plt.axhline(y=initial_capital, color='red', linestyle='--', label='Start Capital')
plt.legend()
plt.title('Equity Curve (Asset Growth)')

plt.tight_layout()
plt.show()
