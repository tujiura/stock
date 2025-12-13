import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

# ==========================================
# ★設定: ページ構成
# ==========================================
st.set_page_config(
    page_title="AI Sniper Control Room",
    page_icon="🎯",
    layout="wide",
)

# タイトル
st.title("🎯 AI Sniper Control Room v2.0")
st.markdown("### 資産防衛型AI 自動売買・戦績分析ダッシュボード")

# ==========================================
# 1. データ読み込み
# ==========================================
DATA_FILE = "ai_trade_memory_risk_managed.csv" 

@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    
    try:
        # ファイル読み込み (エラー行スキップ)
        df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
        
        # カラム名の正規化 (空白削除・小文字対応)
        df.columns = [c.strip() for c in df.columns]
        rename_map = {
            'ticker': 'Ticker', 'date': 'Date', 'action': 'Action', 
            'result': 'result', 'profit_loss': 'profit_loss',
            'profit_rate': 'profit_rate', 'reason': 'Reason',
            'stop_loss_reason': 'stop_loss_reason'
        }
        df.columns = [rename_map.get(col.lower(), col) for col in df.columns]
        
        # 日付変換
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        
        # 数値変換
        if 'profit_loss' in df.columns:
            df['profit_loss'] = pd.to_numeric(df['profit_loss'], errors='coerce').fillna(0)
        else:
            df['profit_loss'] = 0.0
            
        if 'profit_rate' not in df.columns:
            df['profit_rate'] = 0.0
        else:
            df['profit_rate'] = pd.to_numeric(df['profit_rate'], errors='coerce').fillna(0)

        # 累積損益
        df['Equity'] = df['profit_loss'].cumsum()
        
        return df
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return pd.DataFrame()

df = load_data()

# 更新ボタン
if st.button('🔄 データを最新に更新'):
    st.cache_data.clear()
    st.rerun()

if df.empty:
    st.warning("データファイルが見つかりません。トレーニングを実行してください。")
    st.stop()

# ==========================================
# 2. KPI メトリクス (上部表示)
# ==========================================
# BUYエントリーかつ結果が出ているものだけ抽出
df_results = df[(df['Action'] == 'BUY') & (df['result'].isin(['WIN', 'LOSS']))].copy()

if len(df_results) > 0:
    total_trades = len(df_results)
    wins = len(df_results[df_results['result'] == 'WIN'])
    losses = len(df_results[df_results['result'] == 'LOSS'])
    win_rate = (wins / total_trades) * 100
    
    total_pl = df_results['profit_loss'].sum()
    avg_pl = df_results['profit_loss'].mean()
    
    # 利益率ベースの計算
    avg_return = df_results['profit_rate'].mean()
    avg_win_rate = df_results[df_results['result'] == 'WIN']['profit_rate'].mean()
    avg_loss_rate = df_results[df_results['result'] == 'LOSS']['profit_rate'].mean()
    
    # プロフィットファクター (総利益 / 総損失の絶対値)
    gross_profit = df_results[df_results['profit_loss'] > 0]['profit_loss'].sum()
    gross_loss = abs(df_results[df_results['profit_loss'] < 0]['profit_loss'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # カラム表示
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 合計損益", f"{total_pl:,.0f}円", delta=f"{total_pl:,.0f}円")
    col2.metric("📊 勝率", f"{win_rate:.1f}%", f"{wins}勝 {losses}敗")
    col3.metric("📈 プロフィットファクター", f"{pf:.2f}")
    col4.metric("🟢 平均勝ちトレード", f"+{avg_win_rate:.2f}%")
    col5.metric("🔴 平均負けトレード", f"{avg_loss_rate:.2f}%")
else:
    st.info("まだ決済されたトレードデータがありません。")

st.divider()

# ==========================================
# 3. 資産推移 & 利益率分布 (メインチャート)
# ==========================================
col_main, col_sub = st.columns([2, 1])

with col_main:
    st.subheader("📈 資産曲線 (Equity Curve)")
    if len(df) > 0:
        fig_equity = px.line(df, x='Date', y='Equity', markers=True, 
                             title="損益の積み上げ推移",
                             labels={'Equity': '累積損益(円)', 'Date': '日付'})
        # ゼロライン
        fig_equity.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_equity, use_container_width=True)

with col_sub:
    st.subheader("📊 利益率の分布")
    if len(df_results) > 0:
        # ヒストグラム
        fig_hist = px.histogram(df_results, x="profit_rate", nbins=20,
                                color="result",
                                color_discrete_map={'WIN':'#00cc96', 'LOSS':'#EF553B'},
                                title="1トレードあたりの利益率(%)",
                                labels={'profit_rate': '利益率(%)'})
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# 4. 詳細分析 (円グラフ等)
# ==========================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🛑 決済理由の内訳 (Exit Reason)")
    if 'stop_loss_reason' in df_results.columns:
        # 空白やNaNを 'Unknown' に置換
        df_results['stop_loss_reason'] = df_results['stop_loss_reason'].fillna('Unknown')
        reason_counts = df_results['stop_loss_reason'].value_counts().reset_index()
        reason_counts.columns = ['reason', 'count']
        
        fig_pie = px.pie(reason_counts, names='reason', values='count',
                         title="どのような理由で決済されたか",
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("決済理由データがありません。")

with col_right:
    st.subheader("🏆 銘柄別 パフォーマンス")
    if len(df_results) > 0:
        ticker_perf = df_results.groupby('Ticker')['profit_loss'].sum().reset_index()
        ticker_perf = ticker_perf.sort_values('profit_loss', ascending=False).head(10) # Top 10
        
        fig_bar = px.bar(ticker_perf, x='Ticker', y='profit_loss',
                         color='profit_loss',
                         color_continuous_scale=['red', 'gray', 'green'],
                         title="損益貢献度ランキング (Top 10)")
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# 5. 直近のトレード履歴 (データテーブル)
# ==========================================
st.subheader("📝 直近のトレード履歴")
if len(df) > 0:
    # 表示するカラムを絞る
    display_cols = ['Date', 'Ticker', 'Action', 'result', 'Price', 'profit_loss', 'profit_rate', 'Reason', 'stop_loss_reason']
    # 存在しないカラムは除外
    display_cols = [c for c in display_cols if c in df.columns]
    
    # 最新順に並べ替え
    df_display = df.sort_values('Date', ascending=False)
    
    # 色付け用のスタイル関数
    def highlight_result(val):
        color = 'red' if val == 'LOSS' else 'green' if val == 'WIN' else 'black'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df_display[display_cols].style.map(highlight_result, subset=['result'])
        .format({'profit_loss': '{:+.0f}', 'profit_rate': '{:+.2f}%', 'Price': '{:.0f}'}),
        use_container_width=True
    )