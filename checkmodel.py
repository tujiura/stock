import google.generativeai as genai
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
# ここにAPIキーを入れる
GOOGLE_API_KEY = os.getenv("TRAINING_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

print("--- 使用可能なモデル一覧 ---")
try:
    for m in genai.list_models():
        # "generateContent"（文章生成）に対応しているモデルだけ表示
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
            
except Exception as e:
    print(f"エラー: {e}")