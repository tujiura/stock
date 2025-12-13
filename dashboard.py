import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime

# ==========================================
# ★設定: ページ構成
# ==========================================
st.set_page_config(
    page_title="AI Sniper Control Room",
    page_icon="🎯",
    layout="wide",
)

# タイトル
st.title("🎯 AI Sniper Control Room v2.1")
st.markdown("### 資産防衛型AI 自動売買・戦績分析ダッシュボード")

# ==========================================
# 1. データ読み込み
# ==========================================
DATA_FILE = "ai_trade_memory_risk_managed.csv" # 実戦データのみ

# ---------------------------------------------------------
# データの読み込み (修正版: 列ズレ自動補正付き)
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    
    try:
        # ファイル読み込み
        df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
        
        # 1. カラム名の空白削除 & 小文字化
        df.columns = [c.strip() for c in df.columns]
        
        # 2. カラム名の正規化
        rename_map = {
            'ticker': 'Ticker', 'date': 'Date', 'timeframe': 'Timeframe',
            'action': 'Action', 'result': 'result', 'price': 'Price',
            'reason': 'Reason', 'confidence': 'Confidence',
            'profit_loss': 'profit_loss', 'profit_rate': 'profit_rate',
            'entry_volatility': 'entry_volatility', 'rsi_9': 'RSI', 'rsi': 'RSI',
            'stop_loss_reason': 'stop_loss_reason'
        }
        new_cols = []
        for col in df.columns:
            new_cols.append(rename_map.get(col.lower(), col))
        df.columns = new_cols
        
        # 3. 日付変換
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date', ascending=False) # 最新順

        # 4. 数値変換
        numeric_cols = ['profit_loss', 'profit_rate', 'Price', 'RSI', 'entry_volatility', 'Confidence']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        # ★重要: データ列ズレの補正ロジック
        # 利益率(profit_rate)の平均が「50」を超えている場合、それは「円（利益額）」である可能性が高い
        # (通常、1トレードの利益率は数%〜10%程度のため)
        if df['profit_rate'].abs().mean() > 50:
            # カラムがずれていると判断し、補正を行う
            # profit_rate(実は金額) -> profit_loss
            # profit_loss(実はRSI等) -> RSI (簡易的対応)
            
            # 1. profit_rate列の値を、真のprofit_lossとして採用
            real_profit_loss = df['profit_rate']
            
            # 2. 真のprofit_rateを再計算 (投資額100万円前提の簡易計算: 利益額 / 10000)
            # ※厳密には (利益額 / 投資額 * 100) だが、投資額=100万固定なら ÷10000 で%になる
            real_profit_rate = real_profit_loss / 10000.0
            
            # データフレーム更新
            df['profit_loss'] = real_profit_loss
            df['profit_rate'] = real_profit_rate
            
        # 累積損益 (時系列順に直してから計算)
        df_sorted = df.sort_values('Date')
        df['Equity'] = df_sorted['profit_loss'].cumsum()

        return df
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return pd.DataFrame()

# データをロード
df = load_data()

if df.empty:
    st.warning("データファイルが見つかりません。")
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
    
    # 平均利益率の計算
    avg_win_rate = df_results[df_results['result'] == 'WIN']['profit_rate'].mean()
    avg_loss_rate = df_results[df_results['result'] == 'LOSS']['profit_rate'].mean()
    
    # プロフィットファクター
    gross_profit = df_results[df_results['profit_loss'] > 0]['profit_loss'].sum()
    gross_loss = abs(df_results[df_results['profit_loss'] < 0]['profit_loss'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 合計損益", f"{total_pl:,.0f}円", delta_color="normal")
    col2.metric("📊 勝率", f"{win_rate:.1f}%", f"{wins}勝 {losses}敗")
    col3.metric("📈 PF (期待値)", f"{pf:.2f}")
    
    # 色付きフォーマットで表示
    col4.metric("🟢 平均勝ちトレード", f"+{avg_win_rate:.2f}%")
    col5.metric("🔴 平均負けトレード", f"{avg_loss_rate:.2f}%")
else:
    st.info("まだ決済されたトレードデータがありません。")

st.divider()

# ==========================================
# 3. 資産推移 & 利益率分布
# ==========================================
col_main, col_sub = st.columns([2, 1])

with col_main:
    st.subheader("📈 資産曲線 (Equity Curve)")
    if len(df) > 0:
        # 日付順にソートしてプロット
        df_chart = df.sort_values('Date')
        fig_equity = px.line(df_chart, x='Date', y='Equity', markers=True, 
                             title="損益の積み上げ推移",
                             labels={'Equity': '累積損益(円)', 'Date': '日付'})
        fig_equity.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_equity, use_container_width=True)

with col_sub:
    st.subheader("📊 利益率の分布")
    if len(df_results) > 0:
        fig_hist = px.histogram(df_results, x="profit_rate", nbins=20,
                                color="result",
                                color_discrete_map={'WIN':'#00cc96', 'LOSS':'#EF553B'},
                                title="1トレードあたりの利益率(%)",
                                labels={'profit_rate': '利益率(%)'})
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# 4. 詳細分析
# ==========================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🛑 決済理由の内訳")
    if 'stop_loss_reason' in df_results.columns:
        df_results['stop_loss_reason'] = df_results['stop_loss_reason'].fillna('Unknown')
        reason_counts = df_results['stop_loss_reason'].value_counts().reset_index()
        reason_counts.columns = ['reason', 'count']
        fig_pie = px.pie(reason_counts, names='reason', values='count', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("🏆 銘柄別 損益TOP10")
    if len(df_results) > 0:
        ticker_perf = df_results.groupby('Ticker')['profit_loss'].sum().reset_index()
        ticker_perf = ticker_perf.sort_values('profit_loss', ascending=False).head(10)
        fig_bar = px.bar(ticker_perf, x='Ticker', y='profit_loss',
                         color='profit_loss', color_continuous_scale=['red', 'gray', 'green'])
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# 5. 直近のトレード履歴
# ==========================================
st.subheader("📝 直近のトレード履歴")
if len(df) > 0:
    display_cols = ['Date', 'Ticker', 'Action', 'result', 'Price', 'profit_loss', 'profit_rate', 'RSI', 'Reason', 'stop_loss_reason']
    display_cols = [c for c in display_cols if c in df.columns]
    
    # 色付け関数
    def highlight_result(val):
        if val == 'WIN': return 'color: green; font-weight: bold'
        elif val == 'LOSS': return 'color: red; font-weight: bold'
        return ''

    st.dataframe(
        df[display_cols].style.map(highlight_result, subset=['result'])
        .format({'profit_loss': '{:+.0f}', 'profit_rate': '{:+.2f}%', 'Price': '{:.0f}', 'RSI': '{:.1f}'}),
        use_container_width=True
    )