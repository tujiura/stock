import pandas as pd
import yfinance as yf
import datetime
import os
import time

# 設定
LOG_FILE = "ai_trade_memory_risk_managed.csv"
HOLD_PERIOD = 5  # 何日後に決済するか（5営業日）

def update_past_results():
    print("=== 過去のトレード答え合わせ（成績更新ツール） ===")
    
    if not os.path.exists(LOG_FILE):
        print(f"エラー: {LOG_FILE} が見つかりません。")
        return

    try:
        # CSV読み込み
        df = pd.read_csv(LOG_FILE)
        
        # カラム名の正規化（小文字に統一してから、扱いやすい名前に戻す）
        # これにより、新旧どちらのフォーマットでも動くようにする
        rename_map = {
            'date': 'Date', 'ticker': 'Ticker', 'action': 'Action', 
            'price': 'Price', 'result': 'result', 'profit_loss': 'profit_loss',
            'stop_loss_price': 'stop_loss_price'
        }
        df.columns = [rename_map.get(col.lower(), col) for col in df.columns]
        
        # 更新対象: Resultが空欄、かつActionがBUYの行
        # ※現在のロジックでは「BUY」のみが新規エントリーのため
        target_rows = df[
            (df['result'].isna() | (df['result'] == '') | (df['result'] == 'nan')) & 
            (df['Action'] == 'BUY')
        ]
        
        if len(target_rows) == 0:
            print("✅ 更新が必要な未確定データはありません。")
            return

        print(f"未確定データ: {len(target_rows)}件")
        
        updated_count = 0
        
        for index, row in target_rows.iterrows():
            ticker = row['Ticker']
            entry_date_str = str(row['Date'])
            entry_price = float(row['Price'])
            
            # 損切り価格（あれば取得）
            try: sl_price = float(row['stop_loss_price'])
            except: sl_price = 0.0

            # 日付解析 (YYYY-MM-DD)
            try:
                entry_date = pd.to_datetime(entry_date_str).date()
            except:
                print(f"スキップ: 日付形式エラー ({entry_date_str})")
                continue

            # 経過日数が足りているか確認
            days_passed = (datetime.date.today() - entry_date).days
            # 土日含め7日（約5営業日）経過していないと判定できない
            if days_passed < 7:
                continue

            print(f"チェック中: {entry_date} {ticker} ...", end="")

            # 株価データ取得（エントリー日から今日まで）
            try:
                # yfinanceでデータ取得
                # startはエントリー日、endは今日
                hist = yf.download(ticker, start=entry_date, progress=False, auto_adjust=True)
                
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel(1)
                
                # データが少なすぎる場合はスキップ
                if len(hist) < 2:
                    print(" データ不足")
                    continue
                
                # 判定期間のデータを抽出（エントリー翌日〜5営業日後まで）
                # ※ iloc[0]はエントリー当日なので除外
                period_data = hist.iloc[1:HOLD_PERIOD+1]
                
                if period_data.empty:
                    print(" 期間データなし")
                    continue

                # 期間中の最安値・最高値・最終価格
                low_price = period_data['Low'].min()
                high_price = period_data['High'].max()
                exit_price = float(period_data.iloc[-1]['Close'])
                
                result = "DRAW"
                final_price = exit_price
                
                # --- 勝敗判定ロジック ---
                # 1. 損切りにかかったか？ (SL設定がある場合)
                if sl_price > 0 and low_price <= sl_price:
                    result = "LOSS"
                    final_price = sl_price # 損切り価格で決済とみなす
                
                # 2. 利確か？ (2%以上上昇)
                elif exit_price > entry_price * 1.02:
                    result = "WIN"
                
                # 3. 負けか？ (2%以上下落)
                elif exit_price < entry_price * 0.98:
                    result = "LOSS"
                
                # 4. それ以外は引き分け (DRAW)
                else:
                    result = "DRAW"
                        
                profit = final_price - entry_price

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
            print(f"\n✅ {updated_count}件の成績を更新し、AIの記憶を強化しました！")
        else:
            print("\n今回は更新可能なデータ（期間経過済み）はありませんでした。")

    except Exception as e:
        print(f"全体エラー: {e}")

if __name__ == "__main__":
    update_past_results()