import pandas as pd

# 重複を削除したいファイル
TARGET_FILE = "ai_trade_memory_risk_managed.csv" 

try:
    # 読み込み
    df = pd.read_csv(TARGET_FILE)
    print(f"元の行数: {len(df)}")
    
    # 重複削除 (すべての列が同じ行を削除)
    df_clean = df.drop_duplicates()
    print(f"削除後の行数: {len(df_clean)}")
    
    # 上書き保存
    df_clean.to_csv(TARGET_FILE, index=False, encoding='utf-8-sig')
    print("✅ 重複削除完了！")
    
except Exception as e:
    print(f"エラー: {e}")