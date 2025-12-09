import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
import pandas as pd
import socket
import requests.packages.urllib3.util.connection as urllib3_cn

# ---------------------------------------------------------
# ★【重要】GitHub Actionsでの通信エラーを防ぐおまじない
# IPv6を無効化し、強制的にIPv4を使用させます
# ---------------------------------------------------------
def allowed_gai_family():
    return socket.AF_INET

urllib3_cn.allowed_gai_family = allowed_gai_family
# ---------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    print("=== 🧪 GitHub Actions 動作確認テスト (IPv4強制版) ===")
    
    # 1. 環境変数のチェック
    print("\n🔍 [1] 環境変数の確認")
    google_key = os.getenv("GOOGLE_API_KEY")
    line_token = os.getenv("LINE_TOKEN")
    
    if google_key:
        print(f"✅ GOOGLE_API_KEY: OK (文字数: {len(google_key)})")
    else:
        print("❌ GOOGLE_API_KEY: 設定されていません")
        
    if line_token:
        print(f"✅ LINE_TOKEN: OK (文字数: {len(line_token)})")
    else:
        print("❌ LINE_TOKEN: 設定されていません")

    # 2. 外部通信＆データ取得テスト
    print("\n📡 [2] 株価データ取得テスト (yfinance)")
    ticker = "7203.T"
    try:
        print(f"銘柄 {ticker} にアクセス中...")
        df = yf.download(ticker, period="1d", interval="1d", progress=False)
        
        if not df.empty:
            print("✅ データ取得成功！")
            price = float(df['Close'].iloc[-1])
            print(f"最新株価: {price:.0f}円")
        else:
            print("⚠️ データは空でした")
            
    except Exception as e:
        print(f"❌ データ取得エラー: {e}")

    # 3. LINE通知テスト
    print("\n📱 [3] LINE通知テスト")
    if line_token:
        url = "https://notify-api.line.me/api/notify"
        headers = {"Authorization": f"Bearer {line_token}"}
        msg = "\nこれはGitHub Actionsからのテスト通知です。\nIPv4強制モードで成功しました！🚀"
        
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))

        try:
            print("LINEサーバーに送信中...")
            res = session.post(url, headers=headers, data={"message": msg}, timeout=20)
            
            if res.status_code == 200:
                print("✅ LINE送信成功！スマホを確認してください。")
            else:
                print(f"❌ 送信失敗 (Status: {res.status_code}): {res.text}")
                
        except Exception as e:
            print(f"❌ 通信エラー: {e}")
    else:
        print("⚠️ トークンがないためスキップ")

    print("\n=== ✨ テスト終了 ===")

if __name__ == "__main__":
    main()