import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# ★設定: ページ構成
# ==========================================
st.set_page_config(
    page_title="AI Sniper Control Room",
    page_icon="🎯",
    layout="wide",
)

# タイトル
st.title("🎯 AI Sniper Control Room")
st.markdown("### 資産防衛型AI 自動売買・戦績分析ダッシュボード")

# ==========================================
# 1. データ読み込み
# ==========================================
DATA_FILE = "real_trade_record.csv" # 実戦データのみ

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    
    try:
        # エラー行をスキップして読み込む堅牢な設定
        df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
        
        # 日付変換
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date', ascending=False)
        
        # 数値変換 (profit_loss)
        if 'profit_loss' in df.columns:
            df['profit_loss'] = pd.to_numeric(df['profit_loss'], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None

df_raw = load_data()

if df_raw is None or len(df_raw) == 0:
    st.warning("データファイルが見つかりません、またはデータが空です。システムを稼働させてデータを蓄積してください。")
    st.stop()

# ==========================================
# 2. サイドバー (フィルタリング機能)
# ==========================================
st.sidebar.header("🔍 検索フィルタ")

# キャッシュクリアボタン
if st.sidebar.button("🔄 データを更新"):
    st.cache_data.clear()
    st.rerun()
    
# 銘柄フィルタ
tickers = ["ALL"] + list(df_raw['Ticker'].unique())
selected_ticker = st.sidebar.selectbox("銘柄を選択", tickers)

# 期間フィルタ
if 'Date' in df_raw.columns:
    min_date = df_raw['Date'].min()
    max_date = df_raw['Date'].max()
    # 日付が取得できない場合の安全策
    if pd.isna(min_date):
        import datetime
        min_date = datetime.date.today()
        max_date = datetime.date.today()
    
    start_date, end_date = st.sidebar.date_input(
        "期間を選択",
        [min_date, max_date]
    )

# データの絞り込み
df = df_raw.copy()
if selected_ticker != "ALL":
    df = df[df['Ticker'] == selected_ticker]

# 日付フィルタの適用
if 'Date' in df.columns:
    df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

# 戦績データのみ抽出 (WIN/LOSSがついているもの)
df_results = df[df['result'].isin(['WIN', 'LOSS', 'DRAW'])]

# ==========================================
# 3. KPI (重要指標) の表示
# ==========================================
st.markdown("---")

# 計算
total_trades = len(df_results)
wins = len(df_results[df_results['result'] == 'WIN'])
losses = len(df_results[df_results['result'] == 'LOSS'])
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
total_profit = df_results['profit_loss'].sum()

# 表示レイアウト
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("総トレード数", f"{total_trades} 回", delta=f"勝: {wins} / 負: {losses}")

with col2:
    st.metric("勝率 (Win Rate)", f"{win_rate:.1f} %", 
              delta_color="normal" if win_rate >= 50 else "inverse")

with col3:
    color = "normal" if total_profit >= 0 else "inverse"
    st.metric("累積損益 (Total P/L)", f"{total_profit:,.0f} 円", delta=total_profit, delta_color=color)

with col4:
    # 直近のアクション
    last_action = df.iloc[0]['Action'] if len(df) > 0 else "-"
    last_ticker = df.iloc[0]['Ticker'] if len(df) > 0 else "-"
    st.metric("最新シグナル", f"{last_action}", f"{last_ticker}")

# ==========================================
# 4. グラフ分析エリア
# ==========================================
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 資産推移 (累積損益)")
    if len(df_results) > 0:
        # 日付順に並べ替え
        df_chart = df_results.sort_values('Date', ascending=True).copy()
        df_chart['Cumulative PL'] = df_chart['profit_loss'].cumsum()
        
        fig_equity = px.line(df_chart, x='Date', y='Cumulative PL', markers=True,
                             title="損益カーブ (右肩上がりが理想)")
        fig_equity.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("戦績データがまだありません。")

with col_right:
    st.subheader("📊 勝敗比率")
    if len(df_results) > 0:
        fig_pie = px.pie(df_results, names='result', 
                         color='result',
                         color_discrete_map={'WIN':'#00cc96', 'LOSS':'#EF553B', 'DRAW':'gray'},
                         title="WIN / LOSS / DRAW")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("データ不足")

# ==========================================
# 5. 銘柄別パフォーマンス
# ==========================================
st.subheader("🏆 銘柄別 損益ランキング")
if len(df_results) > 0:
    ticker_perf = df_results.groupby('Ticker')['profit_loss'].sum().reset_index()
    ticker_perf = ticker_perf.sort_values('profit_loss', ascending=False)
    
    fig_bar = px.bar(ticker_perf, x='Ticker', y='profit_loss',
                     color='profit_loss',
                     color_continuous_scale=['red', 'gray', 'green'],
                     title="AIが得意な銘柄 vs 苦手な銘柄")
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# 6. 生データ (フィルタリング結果)
# ==========================================
st.subheader("📝 取引履歴 (Raw Data)")
st.dataframe(df, use_container_width=True)

# フッター
st.markdown("---")
st.caption("AI Market Monitor System - Sniper Edition v2.0")