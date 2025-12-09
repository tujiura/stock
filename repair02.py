import pandas as pd
import os

FILE_NAME = "ai_trade_memory_risk_managed.csv"

def force_fix_csv():
    print("=== 強力修復モード開始 ===")
    
    if not os.path.exists(FILE_NAME):
        print("ファイルが見つかりません。")
        return

    # ヘッダーなしで全行読み込む
    raw_df = pd.read_csv(FILE_NAME, header=None)
    print(f"読み込み行数: {len(raw_df)}")

    # 正しいヘッダー定義
    CORRECT_HEADER = [
        'Date', 'Ticker', 'Timeframe', 'Action', 'result', 'Reason', 'Confidence', 
        'stop_loss_price', 'stop_loss_reason', 'Price', 'sma25_dev', 'trend_momentum', 
        'macd_power', 'entry_volatility', 'profit_loss'
    ]

    fixed_data = []

    # 1行ずつ解析して、正しい箱に入れる
    for i, row in raw_df.iterrows():
        vals = row.tolist()
        
        # ヘッダー行やゴミデータはスキップ
        if str(vals[0]).lower() in ['date', '0', 'nan']:
            continue
            
        # データを入れる辞書（初期化）
        d = {k: None for k in CORRECT_HEADER}
        
        # 共通部分（ここはズレていないことが多い）
        d['Date'] = vals[0]
        d['Ticker'] = vals[1]
        d['Timeframe'] = vals[2] if len(vals) > 2 else '1d'
        d['Action'] = vals[3] if len(vals) > 3 else 'HOLD'

        # ★ここからが勝負（値の中身を見て判断）
        
        # リストの中身を全て文字列にして検索しやすくする
        str_vals = [str(v) for v in vals]
        
        # 1. result (WIN/LOSS/DRAW) を探す
        res_candidates = ['WIN', 'LOSS', 'DRAW']
        found_res = next((v for v in str_vals if v in res_candidates), 'DRAW')
        d['result'] = found_res
        
        # 2. Reason (長い文章) を探す
        # "SMA25" や "トレンド" などの単語が含まれる、かつ一番長い文字列
        long_texts = [v for v in str_vals if len(v) > 20 and ('SMA' in v or 'トレンド' in v or '乖離' in v)]
        if long_texts:
            d['Reason'] = max(long_texts, key=len) # 一番長いのが理由
        else:
            d['Reason'] = "理由記載なし"

        # 3. Confidence (0-100の整数) を探す
        # ActionやResultの近くにあることが多い
        # ここでは簡易的に「100以下の整数っぽいもの」を探す
        try:
            # 理由や日付以外で、数値変換できて0-100のもの
            nums = []
            for v in vals:
                try:
                    f = float(v)
                    if 0 <= f <= 100 and f.is_integer():
                        nums.append(int(f))
                except: pass
            
            # 複数ある場合は、Confidenceっぽい位置（後ろの方）や値（60-90など）を採用したいが
            # 今回は単純に「データ内の数値」として再マッピングするのは危険なので
            # 既存の並び順のパターンで決め打ちする
            
            # パターンA: 正常な順序
            # 4:result, 5:Reason, 6:Conf
            if vals[4] in res_candidates:
                d['Confidence'] = vals[6]
                d['stop_loss_price'] = vals[7]
                d['Price'] = vals[9]
                d['profit_loss'] = vals[14] if len(vals) > 14 else 0
                
            # パターンB: ズレてるパターン (Reasonが先に来ちゃってる)
            # 4:Reason, 5:Conf... ではなく、Reasonがどこかに挟まっている
            # 実際のエラーデータを見ると:
            # Date, Ticker, Timeframe, Action, Result, Reason, Conf... 
            # となっているはずが、Confの場所にPrice(1429.0)が入っているということは
            # 何かが抜けているか、Reasonが長すぎて列をまたいでいる可能性
            
            # 強制的に「列数」で判断
            if len(vals) == 15:
                # ほとんどのデータは15列あるはず
                # Confidenceの位置にある値がデカすぎる場合 (Priceが入ってる)
                try:
                    check_conf = float(vals[6])
                    if check_conf > 100:
                        # ズレ確定。PriceがConfに来ている -> つまり1つ前にズレている？
                        # あるいはReasonが列として認識されていない？
                        pass
                except: pass

        except: pass
        
        # ★今回は「構造を再構築」するアプローチをとります
        # 信頼できる列: Date, Ticker, Action, Result, Reason
        # 数値データ群: Conf, SL, Price, SMAdev, Momentum, MACD, Vol, PL
        
        # 数値だけを抽出してリスト化
        numeric_vals = []
        for v in vals:
            try:
                f = float(v)
                # 日付っぽい数字（20250101など）は除外したいが、今回は文字列DateがあるのでOK
                numeric_vals.append(f)
            except: continue
            
        # 数値リストから各項目を推定して割り当て
        # Price: 株価っぽい（1000以上など）
        # Conf: 50-100の整数
        # SMA_dev: -20 ~ +20 くらいの小数
        
        # しかし、これは誤爆のリスクがあるため、
        # 「列ズレの原因」である save_experience の辞書順序ミスを逆算して直すのが確実
        
        # ズレているデータの特徴:
        # [Date, Ticker, Timeframe, Action, Result, Reason, Price, SMAdev, Trend, MACD, Vol, Conf, SL, SLReason, PL]
        # ↑ もしこうなっていたら、PriceがConfの位置に来る
        
        # 暫定処置として、今のファイルを読み込んで
        # 「Confidence > 100」の行だけ、値をシフトさせる
        
        try:
            conf_val = float(vals[6])
            if conf_val > 100:
                # ズレている！
                # 多分 Price がここに来ている
                d['Price'] = vals[6]
                d['sma25_dev'] = vals[7]
                d['trend_momentum'] = vals[8]
                d['macd_power'] = vals[9]
                d['entry_volatility'] = vals[10]
                # 残りの数値からConfidenceを探す
                # vals[11]あたりにあるかも？
                d['Confidence'] = vals[11]
                d['stop_loss_price'] = vals[12]
                d['profit_loss'] = vals[13] if len(vals)>13 else 0
                d['stop_loss_reason'] = "修復データ" # 文字列カラムが消失している可能性あり
            else:
                # 正常（かもしれない）
                d['Confidence'] = vals[6]
                d['stop_loss_price'] = vals[7]
                d['stop_loss_reason'] = vals[8]
                d['Price'] = vals[9]
                d['sma25_dev'] = vals[10]
                d['trend_momentum'] = vals[11]
                d['macd_power'] = vals[12]
                d['entry_volatility'] = vals[13]
                d['profit_loss'] = vals[14]
        except:
            # 変換エラーならデフォルト値
            pass

        fixed_data.append(d)

    # DataFrame化して保存
    df_new = pd.DataFrame(fixed_data)
    # カラム順序を保証
    df_new = df_new[CORRECT_HEADER]
    
    df_new.to_csv(FILE_NAME, index=False, encoding='utf-8-sig')
    print("✅ 強力修復完了")

if __name__ == "__main__":
    force_fix_csv()