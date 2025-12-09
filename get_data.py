import yfinance as yf
import pandas as pd
import datetime
import os

# ==========================================
# ★設定エリア
# ==========================================
# 取得したい銘柄リスト (適宜変更してください)
TARGET_TICKERS = [
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

OUTPUT_FILE = "stock_price_list.csv" # 保存ファイル名

# ==========================================
# メイン処理
# ==========================================
def get_stock_list():
    print(f"データ取得中... ({len(TARGET_TICKERS)}銘柄)")
    
    stock_data = []
    
    # 過去5日分のデータを一括取得（前日比計算のため）
    # group_by='ticker' で銘柄ごとにデータをまとめる
    try:
        data = yf.download(TARGET_TICKERS, period="5d", group_by='ticker', progress=True)
    except Exception as e:
        print(f"ダウンロードエラー: {e}")
        return

    print("\n集計中...")

    for ticker in TARGET_TICKERS:
        try:
            # 該当銘柄のデータを抽出
            if len(TARGET_TICKERS) == 1:
                df = data # 1銘柄の時は構造が違うため
            else:
                df = data[ticker]

            # データが空、または足りない場合はスキップ
            if df.empty or len(df) < 2:
                print(f"⚠️ データ不足: {ticker}")
                continue

            # 最新のデータ（今日）と1つ前のデータ（昨日）を取得
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            price = float(latest['Close'])
            prev_close = float(prev['Close'])
            
            # 変動幅と騰落率
            change = price - prev_close
            pct_change = (change / prev_close) * 100
            
            # ボラティリティ (High - Low) / Close
            high = float(latest['High'])
            low = float(latest['Low'])
            volatility = ((high - low) / price) * 100

            # 通貨・指数の場合の桁数調整
            if "JPY=X" in ticker or "^" in ticker:
                price_fmt = f"{price:,.2f}"
                change_fmt = f"{change:+.2f}"
            else:
                price_fmt = f"{price:,.0f}"
                change_fmt = f"{change:+.0f}"

            # データをリストに追加
            stock_data.append({
                "コード": ticker,
                "現在値": price_fmt,
                "前日比": change_fmt,
                "騰落率": f"{pct_change:+.2f}%",
                "ボラティリティ": f"{volatility:.1f}%",
                "日付": df.index[-1].strftime('%Y-%m-%d')
            })

        except Exception as e:
            print(f"エラー ({ticker}): {e}")

    # DataFrameに変換
    df_result = pd.DataFrame(stock_data)
    
    # コンソールに見やすく表示
    print("\n" + "="*60)
    print(f"📈 株価リスト ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("="*60)
    
    # データを表示 (インデックスなし)
    print(df_result.to_string(index=False))
    print("="*60)

    # CSVに保存
    df_result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ ファイルに保存しました: {OUTPUT_FILE}")

if __name__ == "__main__":
    get_stock_list()