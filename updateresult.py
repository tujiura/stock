import pandas as pd
import yfinance as yf
import datetime
import os
import time

# 設定
LOG_FILE = "ai_trade_memory_risk_managed.csv"
HOLD_PERIOD = 5  # 何日後に決済するか（5営業日）

def update_past_results():
    print("=== 過去のトレード答え合わせ（成績更新） ===")
    
    if not os.path.exists(LOG_FILE):
        print("ファイルが見つかりません。")
        return

    try:
        # CSV読み込み
        df = pd.read_csv(LOG_FILE)
        
        # 更新対象: Resultが空欄、かつActionがBUY/SELLの行
        target_rows = df[
            (df['result'].isna() | (df['result'] == '')) & 
            (df['Action'].isin(['BUY', 'SELL']))
        ]
        
        if len(target_rows) == 0:
            print("更新が必要な未確定データはありません。")
            return

        print(f"未確定データ: {len(target_rows)}件")
        
        updated_count = 0
        
        for index, row in target_rows.iterrows():
            ticker = row['Ticker']
            entry_date_str = row['Date']
            action = row['Action']
            entry_price = float(row['Price'])
            
            # 損切り価格（あれば取得）
            try: sl_price = float(row['stop_loss_price'])
            except: sl_price = 0.0

            # 日付解析
            try:
                entry_date = pd.to_datetime(entry_date_str).date()
            except:
                continue

            # 経過日数が足りているか確認
            days_passed = (datetime.date.today() - entry_date).days
            if days_passed < 3: # 土日含め最低3日は待つ
                continue

            print(f"チェック中: {entry_date} {ticker} ({action}) ...", end="")

            # 株価データ取得（エントリー日から今日まで）
            try:
                # 少し広めに取る
                hist = yf.download(ticker, start=entry_date, progress=False, auto_adjust=True)
                
                # マルチインデックス対応
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel(1)
                
                if len(hist) <= HOLD_PERIOD:
                    print(" データ不足（まだ期間経過待ち）")
                    continue
                
                # 判定ロジック
                # 期間中の安値・高値を取得（損切り判定用）
                period_data = hist.iloc[1:HOLD_PERIOD+1] # エントリー翌日からN日後まで
                
                low_price = period_data['Low'].min()
                high_price = period_data['High'].max()
                exit_price = float(period_data.iloc[-1]['Close']) # N日後の終値
                
                result = "DRAW"
                final_price = exit_price
                
                # --- BUYの場合 ---
                if action == "BUY":
                    # 損切りにかかったか？
                    if sl_price > 0 and low_price <= sl_price:
                        result = "LOSS"
                        final_price = sl_price # 損切り価格で決済
                    # 利確か？
                    elif exit_price > entry_price * 1.02:
                        result = "WIN"
                    elif exit_price < entry_price * 0.98:
                        result = "LOSS"
                    else:
                        result = "DRAW"
                        
                    profit = final_price - entry_price

                # --- SELLの場合 ---
                elif action == "SELL":
                    # 損切りにかかったか？
                    if sl_price > 0 and high_price >= sl_price:
                        result = "LOSS"
                        final_price = sl_price
                    # 利確か？
                    elif exit_price < entry_price * 0.98:
                        result = "WIN"
                    elif exit_price > entry_price * 1.02:
                        result = "LOSS"
                    else:
                        result = "DRAW"
                        
                    profit = entry_price - final_price # 売りは下がれば利益

                # データフレーム更新
                df.at[index, 'result'] = result
                df.at[index, 'profit_loss'] = profit
                
                print(f" -> {result} (損益: {profit:.0f})")
                updated_count += 1
                
                time.sleep(1) # API制限考慮

            except Exception as e:
                print(f" エラー: {e}")
                continue

        # 保存
        if updated_count > 0:
            df.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
            print(f"✅ {updated_count}件の成績を更新しました！")
        else:
            print("今回は更新データなし")

    except Exception as e:
        print(f"全体エラー: {e}")

if __name__ == "__main__":
    update_past_results()