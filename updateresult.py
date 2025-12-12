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

# 判定設定
PROFIT_TARGET_PCT = 0.05  # 5%利益で早期利確（WIN）
JUDGE_PERIOD_DAYS = 15     # 判定期間（これを超えたら強制決済）

# CSVの列順序定義（profit_rateを追加）
CSV_COLUMNS = [
    "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
    "Confidence", "stop_loss_price", "stop_loss_reason", "Price", 
    "sma25_dev", "trend_momentum", "macd_power", "entry_volatility", 
    "profit_loss", "profit_rate"  # <--- ★追加
]

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------
def get_stock_data(ticker, start_date):
    """
    指定日以降の株価データを取得
    """
    try:
        # yfinanceのログを抑制
        import logging
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        
        # 開始日から今日までのデータを取得
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty: return None
        return df
    except Exception as e:
        print(f"   ⚠️ データ取得エラー {ticker}: {e}")
        return None

def update_single_file(file_path):
    """
    1つのCSVファイルを読み込み、結果を更新して保存する
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
            'profit_loss': 'profit_loss', 'profit_rate': 'profit_rate'
        }
        df.columns = [rename_map.get(col.lower(), col) for col in df.columns]
        
        # 足りない列があれば追加（profit_rate含む）
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
        
        # 数値変換とエラーハンドリング
        try:
            entry_price = float(row['Price']) if pd.notna(row['Price']) else 0
            sl_price = float(row['stop_loss_price']) if pd.notna(row['stop_loss_price']) else 0
        except ValueError:
            continue
        
        if entry_price == 0: continue

        try:
            entry_date = pd.to_datetime(entry_date_str)
            stock_data = get_stock_data(ticker, entry_date_str)
        except:
            continue

        if stock_data is None or len(stock_data) < 2: continue

        # --- 判定ロジック ---
        
        # 1. 判定期限日を計算
        limit_date = entry_date + datetime.timedelta(days=JUDGE_PERIOD_DAYS)
        
        # 2. エントリー翌日から期限日までのデータを抽出
        period_data = stock_data[(stock_data.index >= entry_date) & (stock_data.index <= limit_date)]
        
        if len(period_data) == 0: continue

        # 期間内の高値・安値
        period_low = float(period_data['Low'].min())
        period_high = float(period_data['High'].max())
        
        # 期間最終日の終値（タイムアップ時の決済価格）
        final_close = float(period_data.iloc[-1]['Close'])
        
        result = ""
        profit_loss = 0.0
        profit_rate = 0.0 # ★利益率
        is_settled = False

        # A. 期間内のSL/TPチェック（優先）
        if action == "BUY":
            # 損切り (SL)
            if sl_price > 0 and period_low <= sl_price:
                result = "LOSS"
                profit_loss = sl_price - entry_price
                print(f"   💀 {ticker}: 期間内損切り (安値 {period_low:.0f} <= SL {sl_price:.0f})")
                is_settled = True
            # 利確 (TP)
            elif period_high >= entry_price * (1 + PROFIT_TARGET_PCT):
                result = "WIN"
                profit_loss = (entry_price * (1 + PROFIT_TARGET_PCT)) - entry_price
                print(f"   🏆 {ticker}: 期間内利確 (目標到達)")
                is_settled = True

        elif action == "SELL":
            # 損切り (SL: 空売りなので高値で損切り)
            if sl_price > 0 and period_high >= sl_price:
                result = "LOSS"
                profit_loss = entry_price - sl_price
                print(f"   💀 {ticker}: 期間内損切り (高値 {period_high:.0f} >= SL {sl_price:.0f})")
                is_settled = True
            # 利確 (TP: 空売りなので安値で利確)
            elif period_low <= entry_price * (1 - PROFIT_TARGET_PCT):
                result = "WIN"
                profit_loss = entry_price - (entry_price * (1 - PROFIT_TARGET_PCT))
                print(f"   🏆 {ticker}: 期間内利確 (目標到達)")
                is_settled = True

        # B. 期間終了による強制判定 (Time Stop)
        if not is_settled:
            if now > limit_date:
                # 最終日の終値で決済したとみなす
                if action == "BUY":
                    profit_loss = final_close - entry_price
                elif action == "SELL":
                    profit_loss = entry_price - final_close
                
                if profit_loss > 0:
                    result = "WIN"
                    print(f"   ⏰ {ticker}: 期限切れ WIN (終値 {final_close:.0f})")
                else:
                    result = "LOSS"
                    print(f"   ⏰ {ticker}: 期限切れ LOSS (終値 {final_close:.0f})")
            else:
                # まだ期間内で、かつSL/TPにもかかっていない -> 結果保留
                pass

        # 結果が出た場合のみ更新
        if result != "":
            # ★利益率の計算
            if entry_price != 0:
                profit_rate = (profit_loss / entry_price) * 100
            
            df.at[index, 'result'] = result
            df.at[index, 'profit_loss'] = profit_loss
            df.at[index, 'profit_rate'] = profit_rate # ★記録
            updated_count += 1

    # 保存処理
    if updated_count > 0:
        print(f"   💾 {updated_count} 件のデータを更新して保存します...")
        df = df[CSV_COLUMNS] # 列順序を整える
        for i in range(5):
            try:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print("   ✅ 保存成功")
                
                try:
                    import subprocess
                    subprocess.run(["git", "add", file_path], check=True, capture_output=True)
                    subprocess.run(["git", "commit", "-m", f"Auto update results: {file_path}"], check=True, capture_output=True)
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

# ---------------------------------------------------------
# メイン実行
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== 📈 トレード結果 自動更新システム (利益率対応版) ===")
    
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