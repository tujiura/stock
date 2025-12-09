import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt  # グラフを描画するためにインポート

# 1. データを取得
df = yf.download('9432.T', period='1y')

# 【ここが修正ポイント！】
# カラムが ('Close', '9432.T') のような2段組みになっている場合、
# 1段目の 'Close' だけを残して単純な形にします。
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. 移動平均(SMA)の計算
# これで df['Close'] が確実に「1列の数値」として扱われます
df['SMA25'] = df['Close'].rolling(window=25).mean()

# 3. ボリンジャーバンドの計算
sigma = df['Close'].rolling(window=25).std()
df['Upper'] = df['SMA25'] + (2 * sigma)
df['Lower'] = df['SMA25'] - (2 * sigma)

# 確認のため、最後の5行を表示
print(df[['Close', 'SMA25', 'Upper', 'Lower']].tail())

# 4. グラフを表示（せっかくなので可視化しましょう）
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Close', color='black', alpha=0.6)
plt.plot(df.index, df['SMA25'], label='SMA25', color='orange')
plt.plot(df.index, df['Upper'], label='Upper (2sigma)', color='green', linestyle='--')
plt.plot(df.index, df['Lower'], label='Lower (2sigma)', color='green', linestyle='--')

# ±2σの間を薄い色で塗る
plt.fill_between(df.index, df['Upper'], df['Lower'], color='green', alpha=0.1)

plt.title('NTT (9432.T) Bollinger Bands')
plt.legend()
plt.grid()
plt.show()