import pandas as pd
import os

# ==========================================
# 設定
# ==========================================
OLD_FILE = "ai_trade_memory_cbr.csv"          # 元のファイル
NEW_FILE = "ai_trade_memory_risk_managed.csv" # 新しいシステム用ファイル

def migrate_data():
    if not os.path.exists(OLD_FILE):
        print(f"エラー: 元ファイル {OLD_FILE} が見つかりません。")
        return

    print(f"🔄 データを読み込んでいます: {OLD_FILE}")
    df = pd.read_csv(OLD_FILE)

    # 1. カラム名のマッピング（新システムのTitle Caseに合わせる）
    # 元: date, ticker, action, result, reason, sma25_dev, trend_momentum, macd_power, profit_loss
    rename_map = {
        'date': 'Date',
        'ticker': 'Ticker',
        'action': 'Action',
        'reason': 'Reason',
        # sma25_dev, trend_momentum, macd_power, result, profit_loss はそのまま小文字でOK
    }
    df = df.rename(columns=rename_map)

    # 2. 新しい必須カラムの追加と初期値設定
    print("🛠️  新しいデータ形式に変換中...")

    # Timeframe: 過去データはすべて日足とみなす
    if 'Timeframe' not in df.columns:
        df['Timeframe'] = '1d'

    # entry_volatility: 過去データには無いので「0」とする
    # (注意: 0だと「ボラティリティ極小」として検索されるが、データが無い以上やむを得ない)
    if 'entry_volatility' not in df.columns:
        df['entry_volatility'] = 0.0

    # stop_loss関連: 過去データには無いのでプレースホルダーを入れる
    if 'stop_loss_price' not in df.columns:
        df['stop_loss_price'] = 0.0
    
    if 'stop_loss_reason' not in df.columns:
        df['stop_loss_reason'] = 'Legacy Data'

    # Confidence: 過去データに無い場合は0とする
    if 'Confidence' not in df.columns:
        df['Confidence'] = 0

    # Price: スナップショット価格が無い場合は0とする（計算には影響しないが表示用）
    if 'Price' not in df.columns:
        df['Price'] = 0.0

    # 3. カラムの並び順を整える（可読性のため）
    target_order = [
        'Date', 'Ticker', 'Timeframe', 
        'Action', 'Confidence', 
        'stop_loss_price', 'stop_loss_reason', 
        'Reason', 
        'Price', 'sma25_dev', 'trend_momentum', 'macd_power', 'entry_volatility',
        'result', 'profit_loss'
    ]
    
    # 存在しないカラムがあれば自動追加して並べ替え
    for col in target_order:
        if col not in df.columns:
            df[col] = 0 if 'price' in col.lower() or 'confidence' in col.lower() else ""
            
    df = df[target_order]

    # 4. 新しいファイルとして保存
    df.to_csv(NEW_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ 変換完了! 保存しました: {NEW_FILE}")
    print(f"📊 データ件数: {len(df)} 件")
    print("👉 今後はこの新しいCSVファイルを使ってトレーニング/監視を行ってください。")

if __name__ == "__main__":
    migrate_data()