import os
from dotenv import load_dotenv

load_dotenv(override=True) # .envファイルを読み込む

key = os.getenv("GOOGLE_API_KEY")

if key:
    print("✅ キーが見つかりました")
    print(f"キーの先頭: {key[:30]}...")
    print(f"キーの末尾: ...{key[-5:]}")
    print("↑ これが新しく作ったキーと一致していますか？")
else:
    print("❌ キーが読み込めていません。.envファイルの場所や名前を確認してください。")