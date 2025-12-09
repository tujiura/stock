import pandas as pd
import csv
import os
import shutil

# 設定
FILE_NAME = "ai_trade_memory_risk_managed.csv"
BACKUP_NAME = "ai_trade_memory_risk_managed.bak"
EXPECTED_COLUMNS = 15  # 正しい列数

def fix_csv():
    print(f"🔧 CSV修復ツール: {FILE_NAME}")

    if not os.path.exists(FILE_NAME):
        print("❌ ファイルが見つかりません。")
        return

    # バックアップを作成
    try:
        shutil.copy(FILE_NAME, BACKUP_NAME)
        print(f"📦 バックアップ作成完了: {BACKUP_NAME}")
    except Exception as e:
        print(f"⚠️ バックアップ作成失敗: {e}")

    # 正常な行だけを抽出
    valid_rows = []
    error_count = 0
    
    try:
        with open(FILE_NAME, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if header:
                # ヘッダーの列数を確認（強制修正）
                if len(header) != EXPECTED_COLUMNS:
                    print(f"⚠️ ヘッダー列数が不正({len(header)})です。標準ヘッダーに置換します。")
                    header = [
                        "Date", "Ticker", "Timeframe", "Action", "result", "Reason", 
                        "Confidence", "stop_loss_price", "stop_loss_reason", "Price", 
                        "sma25_dev", "trend_momentum", "macd_power", "entry_volatility", "profit_loss"
                    ]
                valid_rows.append(header)

            for i, row in enumerate(reader):
                if len(row) == EXPECTED_COLUMNS:
                    valid_rows.append(row)
                else:
                    error_count += 1
                    # 最初の数件だけエラー内容を表示
                    if error_count <= 3:
                        print(f"   ❌ 削除対象 (行 {i+2}): 列数 {len(row)} -> {row[:3]}...")

        # 上書き保存
        with open(FILE_NAME, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(valid_rows)

        print(f"\n✅ 修復完了！")
        print(f"   - 正常なデータ: {len(valid_rows) - 1} 件")
        print(f"   - 削除した破損データ: {error_count} 件")
        print("   これでトレーニングを再開できます。")

    except Exception as e:
        print(f"❌ 修復中にエラーが発生しました: {e}")

if __name__ == "__main__":
    fix_csv()