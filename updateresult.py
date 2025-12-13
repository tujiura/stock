import pandas as pd
import yfinance as yf
import datetime
import os
import time
import sys

# ---------------------------------------------------------
# ★設定エリア
# ---------------------------------------------------------
# 更新対象のファイルリスト
TARGET_FILES = [
    "ai_trade_memory_risk_managed.csv", # 学習用（AIの脳）
    "real_trade_record.csv"             # 実戦用（あなたの記録）
]

# 判定期間設定
JUDGE_PERIOD_DAYS = 30     # エントリーから最大何日まで見るか

# CSVの列順序定義 (★rsi_9 を追加)
CSV_COLUMNS = [
    "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
    "Confidence", "stop_loss_price", "stop_loss_reason", "Price", 
    "sma25_dev", "trend_momentum", "macd_power", "entry_volatility", 
    "rsi_9", "profit_loss", "profit_rate" # <--- ここに追加
]

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------
def get_stock_data(ticker, start_date):
    """
    指定日以降の株価データを取得
    """
    try:
        import logging
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        
        # エントリー当日を含めて取得
        fetch_start = start_date - datetime.timedelta(days=5)
        
        df = yf.download(ticker, start=fetch_start, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty: return None
        return df
    except Exception as e:
        print(f"   ⚠️ データ取得エラー {ticker}: {e}")
        return None

def update_single_file(file_path):
    """
    1つのCSVファイルを読み込み、最新ロジック（可変式トレーリング）で結果を判定・更新する
    """
    if not os.path.exists(file_path):
        print(f"⏩ ファイルなし（スキップ）: {file_path}")
        return

    print(f"\n📂 ファイル読み込み中: {file_path}")
    
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        rename_map = {
            'date': 'Date', 'ticker': 'Ticker', 'action': 'Action', 
            'result': 'result', 'price': 'Price', 'stop_loss_price': 'stop_loss_price',
            'profit_loss': 'profit_loss', 'profit_rate': 'profit_rate',
            'entry_volatility': 'entry_volatility',
            'rsi_9': 'rsi_9', 'rsi': 'rsi_9' # 表記ゆれ対応
        }
        df.columns = [rename_map.get(col.lower(), col) for col in df.columns]
        
        # 足りない列があれば追加
        for col in CSV_COLUMNS:
            if col not in df.columns: df[col] = None
            
    except Exception as e:
        print(f"❌ 読み込みエラー: {e}")
        return

    updated_count = 0
    now = datetime.datetime.now()
    
    # 行ごとの処理
    for index, row in df.iterrows():
        # 既に結果が出ている、またはHOLDの場合はスキップ
        if pd.notna(row['result']) and str(row['result']).strip() != "":
            continue
        if row['Action'] == 'HOLD':
            continue
            
        ticker = row['Ticker']
        entry_date_str = row['Date']
        action = row['Action']
        
        try:
            entry_price = float(row['Price']) if pd.notna(row['Price']) else 0
            initial_sl_price = float(row['stop_loss_price']) if pd.notna(row['stop_loss_price']) else 0
            volatility = float(row['entry_volatility']) if pd.notna(row['entry_volatility']) else 1.5
            atr_value = entry_price * (volatility / 100)
        except ValueError:
            continue
        
        if entry_price == 0: continue

        try:
            entry_date = pd.to_datetime(entry_date_str)
            stock_data = get_stock_data(ticker, entry_date)
        except:
            continue

        if stock_data is None or len(stock_data) < 2: continue

        period_data = stock_data[stock_data.index > entry_date].copy()
        
        if len(period_data) == 0: continue

        # --- 🏆 最新判定ロジック: 可変式ATRトレーリングストップ ---
        
        current_sl = initial_sl_price
        if current_sl == 0: 
            current_sl = entry_price - (atr_value * 2.0)
            
        max_price = entry_price
        result = ""
        exit_price = 0.0
        is_settled = False

        for date, day_data in period_data.iterrows():
            day_low = float(day_data['Low'])
            day_high = float(day_data['High'])
            day_close = float(day_data['Close'])

            if action == "BUY":
                # 1. 損切り判定
                if day_low <= current_sl:
                    result = "LOSS"
                    exit_price = current_sl 
                    is_settled = True
                    if exit_price > entry_price:
                        result = "WIN"
                    print(f"   💀 {ticker}: 決済 ({date.strftime('%m/%d')}) SL接触 {exit_price:.0f}")
                    break

                # 2. トレーリング更新
                if day_high > max_price:
                    max_price = day_high
                    current_profit_pct = (max_price - entry_price) / entry_price
                    
                    if current_profit_pct > 0.05:
                        trail_width = atr_value * 0.5
                    elif current_profit_pct > 0.03:
                        trail_width = atr_value * 1.0
                    else:
                        trail_width = atr_value * 2.0
                        
                    new_sl = max_price - trail_width
                    
                    if current_profit_pct > 0.015:
                        break_even_price = entry_price * 1.001
                        new_sl = max(new_sl, break_even_price)

                    if new_sl > current_sl:
                        current_sl = new_sl

            elif action == "SELL":
                pass 

        # 期限切れ判定
        if not is_settled:
            limit_date = entry_date + datetime.timedelta(days=JUDGE_PERIOD_DAYS)
            if now > limit_date:
                exit_price = day_close # 最終値
                is_settled = True
                if exit_price > entry_price:
                    result = "WIN"
                    print(f"   ⏰ {ticker}: 期限切れ WIN (終値 {exit_price:.0f})")
                else:
                    result = "LOSS"
                    print(f"   ⏰ {ticker}: 期限切れ LOSS (終値 {exit_price:.0f})")

        # 結果書き込み
        if is_settled:
            profit_loss = exit_price - entry_price
            if action == "SELL": profit_loss = entry_price - exit_price
            
            profit_rate = 0.0
            if entry_price != 0:
                profit_rate = (profit_loss / entry_price) * 100
            
            df.at[index, 'result'] = result
            df.at[index, 'profit_loss'] = profit_loss
            df.at[index, 'profit_rate'] = profit_rate
            df.at[index, 'stop_loss_price'] = current_sl
            
            updated_count += 1

    # 保存処理
    if updated_count > 0:
        print(f"   💾 {updated_count} 件のデータを更新して保存します...")
        
        # 列順序を整えて保存（ここでRSI列も保存される）
        final_cols = [c for c in CSV_COLUMNS if c in df.columns]
        df = df[final_cols]
        
        for i in range(5):
            try:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print("   ✅ 保存成功")
                
                try:
                    import subprocess
                    subprocess.run(["git", "add", file_path], check=True, capture_output=True)
                    subprocess.run(["git", "commit", "-m", f"Auto update results (RSI added): {file_path}"], check=True, capture_output=True)
                    print("   ☁️ Gitコミット完了")
                except: pass
                break
            except PermissionError:
                print(f"   ⚠️ ファイルが開かれています。閉じてください ({i+1}/5)")
                time.sleep(3)
            except Exception as e:
                print(f"   ❌ 保存エラー: {e}")
                break
    else:
        print("   (更新対象なし)")

if __name__ == "__main__":
    print("=== 📈 トレード結果 自動更新システム (RSI対応版) ===")
    
    do_push = False
    for file_name in TARGET_FILES:
        update_single_file(file_name)
        do_push = True

    if do_push:
        try:
            import subprocess
            print("\n☁️ GitHubへ同期中...")
            subprocess.run(["git", "push"], check=True)
            print("✅ 同期完了")
        except:
            print("⚠️ Git Pushスキップ")
            
    print("\nすべての処理が完了しました。")
    time.sleep(3)