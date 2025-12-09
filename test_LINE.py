import os
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket
import requests.packages.urllib3.util.connection as urllib3_cn

# ---------------------------------------------------------
# ★【Windows対策】文字コードをUTF-8に強制する
# ---------------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------
# ★【通信対策】IPv6を無効化し、強制的にIPv4を使用させます
# ---------------------------------------------------------
def allowed_gai_family():
    return socket.AF_INET

urllib3_cn.allowed_gai_family = allowed_gai_family



# ローカル環境用（GitHub Actionsでは無視されます）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass



def main():
    print("=== 🧪 Discord 通知テスト (最強版) ===")
    
    # 1. 環境変数のチェック

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if webhook_url:
        print(f"✅ Webhook URL: 設定済み (文字数: {len(webhook_url)})")
    else:
        print("❌ Webhook URL: 設定されていません！GitHub Secretsを確認してください。")
        return

    # 2. 通知送信テスト
    print("\n📨 Discordにメッセージを送信中...")
    
    # 送信データ（JSON形式）
    payload = {
        "content": "✅ **テスト成功！**\nGitHub Actionsからの通知テストです。\nこのメッセージが見えていれば、設定は完璧です！🚀",
        "username": "AI投資アドバイザー(テスト)",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/4228/4228956.png"
    }

    # リトライ設定（通信を頑丈にする）
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        # Discordは json=payload で送るのがポイント
        res = session.post(webhook_url, json=payload, timeout=10)
        
        if 200 <= res.status_code < 300:
            print("✅ 送信成功！Discordの画面を確認してください。")
        else:
            print(f"❌ 送信失敗 (Status: {res.status_code})")
            print(f"エラー内容: {res.text}")
            
    except Exception as e:
        print(f"❌ 通信エラー発生: {e}")

    print("\n=== ✨ テスト終了 ===")

if __name__ == "__main__":
    main()