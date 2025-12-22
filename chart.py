import os
import pandas as pd

# あなたが設定したつもりのファイル名
target_file = "ai_trade_memory_v15_aggressive.csv"

print(f"=== ファイル診断: {target_file} ===")
print(f"📂 現在の作業フォルダ: {os.getcwd()}")

# 1. ファイルの存在確認
if os.path.exists(target_file):
    print("✅ ファイルは存在します。")
    
    # 2. 読み込みテスト (Excelロックなどを検知)
    try:
        df = pd.read_csv(target_file)
        print(f"✅ 読み込み成功！ データ件数: {len(df)}行")
        print("📝 カラム一覧:", df.columns.tolist())
        
        # 3. データの中身チェック
        wins = len(df[df['Result'] == 'WIN'])
        losses = len(df[df['Result'] == 'LOSS'])
        print(f"📊 勝敗データ: WIN={wins}, LOSS={losses}")
        
        if wins + losses < 5:
            print("⚠️ 注意: データは読めましたが、勝敗データが少なすぎます（5件未満）。")
        else:
            print("🎉 診断結果: データは完璧です。AIコード側の問題の可能性が高いです。")
            
    except PermissionError:
        print("\n❌ 【原因特定】ファイルが開かれています！")
        print("👉 Excelなどでこのファイルを開いていませんか？ 閉じてから再実行してください。")
    except Exception as e:
        print(f"\n❌ 読み込みエラー: {e}")
else:
    print("\n❌ 【原因特定】ファイルが見つかりません！")
    print("👉 フォルダの中に以下のファイルがあるか確認してください:")
    files = [f for f in os.listdir() if f.endswith(".csv")]
    for f in files:
        print(f"   - {f}")
    
    print("\n※ 'agressive' (gが1つ) と 'aggressive' (gが2つ) の違いに注意してください。")