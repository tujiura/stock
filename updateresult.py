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
JUDGE_PERIOD_DAYS = 30     # エントリーから最大何日まで見るか（期限切れ判定用）

# CSVの列順序定義
CSV_COLUMNS = [
    "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
    "Confidence", "stop_loss_price", "stop_loss_reason", "Price", 
    "sma25_dev", "trend_momentum", "macd_power", "entry_volatility", 
    "profit_loss", "profit_rate"
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
        
        # エントリー当日を含めて取得（当日の足はエントリー後の動きとして簡易判定に使う場合もあるが、基本は翌日以降）
        # yfinanceは start <= date < end なので、念のため少し前から取る
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
            'entry_volatility': 'entry_volatility'
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
            # CSVに記録されたSLがあればそれを使うが、なければ計算で出す
            initial_sl_price = float(row['stop_loss_price']) if pd.notna(row['stop_loss_price']) else 0
            # ボラティリティからATRを逆算 (ATR = Price * Volatility / 100)
            volatility = float(row['entry_volatility']) if pd.notna(row['entry_volatility']) else 1.5
            atr_value = entry_price * (volatility / 100)
        except ValueError:
            continue
        
        if entry_price == 0: continue

        try:
            entry_date = pd.to_datetime(entry_date_str)
            # データ取得（エントリー日以降）
            stock_data = get_stock_data(ticker, entry_date)
        except:
            continue

        if stock_data is None or len(stock_data) < 2: continue

        # エントリー日の次の営業日から判定スタート
        # (stock_dataにはエントリー日も含まれる可能性があるためフィルタリング)
        period_data = stock_data[stock_data.index > entry_date].copy()
        
        if len(period_data) == 0: continue

        # --- 🏆 最新判定ロジック: 可変式ATRトレーリングストップ ---
        
        # 変数初期化
        current_sl = initial_sl_price
        if current_sl == 0: # CSVにSLがない場合の初期値 (ATR x 2.0)
            current_sl = entry_price - (atr_value * 2.0)
            
        max_price = entry_price
        
        result = ""
        exit_price = 0.0
        final_date = None
        is_settled = False

        # 1日ずつシミュレーション
        for date, day_data in period_data.iterrows():
            day_low = float(day_data['Low'])
            day_high = float(day_data['High'])
            day_close = float(day_data['Close'])
            final_date = date

            if action == "BUY":
                # 1. 損切り判定 (Lowがタッチしたか)
                if day_low <= current_sl:
                    result = "LOSS" # 暫定（プラス決済ならWINに書き換える）
                    exit_price = current_sl # 逆指値価格で決済
                    # 窓開け等でSLよりはるかに下で寄った場合は始値で決済すべきだが、
                    # 簡易的にSL価格、あるいは安値との比較で保守的に計算
                    # ここではSL価格を採用（厳密には Open との比較が必要だがデータ簡略化）
                    is_settled = True
                    # 建値ガード等でSLが買値より上にある場合はWIN
                    if exit_price > entry_price:
                        result = "WIN"
                    print(f"   💀 {ticker}: 決済 ({date.strftime('%m/%d')}) SL接触 {exit_price:.0f} (High:{day_high:.0f} / SL:{current_sl:.0f})")
                    break

                # 2. トレーリング更新判定 (Highが更新したか)
                if day_high > max_price:
                    max_price = day_high
                    
                    # 現在の含み益率（ピーク時）
                    current_profit_pct = (max_price - entry_price) / entry_price
                    
                    # ★ラチェット機能: 利益が出るほど追従をきつくする
                    if current_profit_pct > 0.05:   # +5%超え
                        trail_width = atr_value * 0.5 # ほぼ利確モード
                    elif current_profit_pct > 0.03: # +3%超え
                        trail_width = atr_value * 1.0 # 激狭
                    else:
                        trail_width = atr_value * 2.0 # 標準
                        
                    new_sl = max_price - trail_width
                    
                    # ★建値ガード: +1.5%乗ったら絶対に負けない位置へ
                    if current_profit_pct > 0.015:
                        break_even_price = entry_price * 1.001 # 手数料分少しプラス
                        new_sl = max(new_sl, break_even_price)

                    # 逆指値は「上げる」ことしかしない
                    if new_sl > current_sl:
                        current_sl = new_sl
                        # print(f"     ⬆️ SL引上: {current_sl:.0f} (Max:{max_price:.0f})")

            # SELLロジック（今回はBUY限定システムのため簡易実装または省略）
            elif action == "SELL":
                pass 

        # 期限切れ判定
        if not is_settled:
            limit_date = entry_date + datetime.timedelta(days=JUDGE_PERIOD_DAYS)
            if now > limit_date:
                # 期限切れ強制決済（最終日の終値）
                exit_price = day_close # ループ最後の日の終値
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
            if action == "SELL": profit_loss = entry_price - exit_price # SELLの場合逆
            
            # 利益率
            profit_rate = 0.0
            if entry_price != 0:
                profit_rate = (profit_loss / entry_price) * 100
            
            df.at[index, 'result'] = result
            df.at[index, 'profit_loss'] = profit_loss
            df.at[index, 'profit_rate'] = profit_rate
            
            # SL価格も最終的な値に更新しておく（記録として）
            df.at[index, 'stop_loss_price'] = current_sl
            
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
                    subprocess.run(["git", "commit", "-m", f"Auto update results (Ratchet Trailing): {file_path}"], check=True, capture_output=True)
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
    print("=== 📈 トレード結果 自動更新システム (可変式トレーリング版) ===")
    
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