import pandas as pd
import os
import shutil
import datetime

FILE_NAME = "ai_trade_memory_risk_managed.csv"

def sanitize_csv():
    print(f"=== CSV緊急浄化プログラム開始 ===")
    
    if not os.path.exists(FILE_NAME):
        print("❌ ファイルが見つかりません。")
        return

    # 1. バックアップ作成
    backup_name = f"{FILE_NAME}.sanitized_bak_{datetime.datetime.now().strftime('%H%M%S')}"
    shutil.copy(FILE_NAME, backup_name)
    print(f"📦 バックアップ作成: {backup_name}")

    # 2. データを読み込む（全列を文字列として）
    df = pd.read_csv(FILE_NAME, dtype=str, header=None)
    
    # ヘッダー行を特定（1行目と仮定）
    header = [
        'Date', 'Ticker', 'Timeframe', 'Action', 'result', 'Reason', 'Confidence', 
        'stop_loss_price', 'stop_loss_reason', 'Price', 'sma25_dev', 'trend_momentum', 
        'macd_power', 'entry_volatility', 'profit_loss'
    ]
    
    # 数値であるべきカラムのインデックス（0始まり）
    # Confidence(6), SL_Price(7), Price(9), SMA(10), Mom(11), MACD(12), Vol(13), PL(14)
    numeric_indices = [6, 7, 9, 10, 11, 12, 13, 14]
    
    sanitized_rows = []
    fixed_count = 0

    # 3. 全行スキャン & 修復
    for i, row in df.iterrows():
        vals = row.tolist()
        
        # ヘッダー行はスキップして、正しいヘッダーをあとでつける
        if vals[0] == 'Date':
            continue
            
        # 列数が足りない場合は空文字で埋める
        if len(vals) < 15:
            vals += [''] * (15 - len(vals))
        vals = vals[:15] # 多い場合は切る

        # --- 数値カラムの浄化 ---
        for idx in numeric_indices:
            val = str(vals[idx])
            
            # 数値変換テスト
            try:
                # すでに数値ならOK
                float(val)
            except ValueError:
                # ❌ エラー！ 数値列に文字が入っている
                if len(val) > 0 and val != 'nan':
                    print(f"⚠️ 行{i} 列{idx}({header[idx]}) の不正データを検知: '{val}'")
                    
                    # 救済措置: もしこれが「ATR...」などの理由テキストなら、stop_loss_reason(8)に移動
                    if idx != 8 and len(val) > 3: # 8はSL理由なので除外
                        current_reason = str(vals[8])
                        if current_reason == 'nan' or current_reason == '':
                            vals[8] = val # 移動
                        else:
                            vals[8] += f" / {val}" # 追記
                    
                    fixed_count += 1
                
                # 強制的に 0 (または安全な値) に書き換え
                vals[idx] = "0"

        sanitized_rows.append(vals)

    # 4. 保存
    df_clean = pd.DataFrame(sanitized_rows, columns=header)
    df_clean.to_csv(FILE_NAME, index=False, encoding='utf-8-sig')
    
    print(f"✅ 完了: {len(df_clean)}行 を処理しました。")
    print(f"🔧 修復箇所: {fixed_count} 個")
    print("これで 'could not convert string to float' エラーは出なくなります。")

if __name__ == "__main__":
    sanitize_csv()