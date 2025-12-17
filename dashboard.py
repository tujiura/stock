import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# ★設定エリア
# ==========================================
FILES = {
    "🎯 攻撃型 V11 (最終形)": "ai_trade_memory_aggressive_v11.csv",
    "🚀 攻撃型 V7 (ホームラン狙い)": "ai_trade_memory_aggressive_v7.csv",
    "🔥 攻撃型 V5 (分割決済)": "ai_trade_memory_aggressive_v5.csv",
    "🛡️ 資産防衛型": "ai_trade_memory_risk_managed.csv",
}

PAGE_TITLE = "📊 AI Trade Analysis Dashboard V8"
LAYOUT = "wide"

# ==========================================
# 1. データ読み込み & 前処理
# ==========================================
def load_data(csv_file):
    if not os.path.exists(csv_file):
        return None, f"ファイルが見つかりません: {csv_file}"
    
    try:
        df = pd.read_csv(csv_file)
        df.columns = [c.strip() for c in df.columns] # 空白除去
        
        if 'Date' not in df.columns:
            return None, "CSVに 'Date' 列がありません。"

        df['Date'] = pd.to_datetime(df['Date'])
        
        # 数値変換 (V7の新カラム含む)
        numeric_cols = [
            'profit_rate', 'Confidence', 'Price', 
            'adx', 'ma_deviation', 'vol_ratio', 'dist_to_res',
            'Actual_High', 'target_price', 'Target_Reach',
            'macd_hist' # V7追加
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 損益額シミュレーション (1トレード100万円換算)
        if 'profit_loss' not in df.columns:
            df['profit_loss'] = df['profit_rate'] * 10000 
            
        df = df.sort_values('Date')
        df['cumulative_profit'] = df['profit_loss'].cumsum()
        
        return df, None
    except Exception as e:
        return None, f"読み込みエラー: {e}"

# ==========================================
# 2. メインアプリ
# ==========================================
def main():
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)
    st.title(PAGE_TITLE)
    
    # --- サイドバー ---
    st.sidebar.header("⚙️ 設定")
    selected_mode = st.sidebar.radio("分析ファイル選択", list(FILES.keys()), index=0)
    target_file = FILES[selected_mode]
    
    st.sidebar.markdown("---")
    
    df, error_msg = load_data(target_file)
    if df is None:
        st.error(f"⚠️ {error_msg}")
        st.info("※ V7を選択している場合、先にトレーニングスクリプト(v7)を実行してCSVを作成してください。")
        # フォールバック: 存在するファイルを探す提案
        existing = [f for f in FILES.values() if os.path.exists(f)]
        if existing:
            st.warning(f"現在利用可能なファイル: {', '.join(existing)}")
        return

    if df.empty:
        st.warning("データが空です。")
        return

    # 銘柄フィルター
    all_tickers = ["All"] + sorted(list(df['Ticker'].unique()))
    selected_ticker = st.sidebar.selectbox("銘柄フィルター", all_tickers)
    if selected_ticker != "All":
        df = df[df['Ticker'] == selected_ticker]

    # --- KPI エリア ---
    df_finished = df[df['result'].isin(['WIN', 'LOSS'])]
    total_trades = len(df_finished)
    wins = df_finished[df_finished['result'] == 'WIN']
    losses = df_finished[df_finished['result'] == 'LOSS']
    
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_profit = df_finished['profit_loss'].sum()
    pf = wins['profit_loss'].sum() / abs(losses['profit_loss'].sum()) if not losses.empty else float('inf')

    # ★修正箇所: 変数名を統一しました
    target_reach_kpi = "-" 
    if 'Target_Reach' in df.columns:
        reached = df_finished[df_finished['Target_Reach'] >= 100]
        rate = len(reached) / total_trades * 100 if total_trades > 0 else 0
        target_reach_kpi = f"{rate:.1f}%"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("総トレード", f"{total_trades}")
    k2.metric("勝率", f"{win_rate:.1f}%")
    k3.metric("合計損益 (100万)", f"{total_profit:,.0f}円", delta_color="normal")
    k4.metric("PF", f"{pf:.2f}")
    k5.metric("目標到達率", target_reach_kpi)

    st.markdown("---")

    # --- グラフエリア 1: 資産推移 ---
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📈 資産推移")
        fig = px.line(df_finished, x='Date', y='cumulative_profit', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("📊 勝ち負け分布")
        fig_hist = px.histogram(df_finished, x='profit_rate', color='result', nbins=30, title="利益率分布")
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- グラフエリア 2: 要因分析 (V7対応) ---
    st.subheader("🔬 要因分析")
    t1, t2, t3 = st.columns(3)
    
    with t1:
        if 'adx' in df.columns:
            fig = px.scatter(df_finished, x='adx', y='profit_rate', color='result', title="ADX vs 利益")
            st.plotly_chart(fig, use_container_width=True)
    
    with t2:
        # V7新指標: MACDヒストグラム
        if 'macd_hist' in df.columns:
            fig = px.scatter(df_finished, x='macd_hist', y='profit_rate', color='result', title="MACD Hist vs 利益")
            fig.add_vline(x=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
        elif 'ma_deviation' in df.columns:
            fig = px.scatter(df_finished, x='ma_deviation', y='profit_rate', color='result', title="MA乖離 vs 利益")
            st.plotly_chart(fig, use_container_width=True)

    with t3:
        # V7新指標: 雲との位置関係
        if 'price_vs_cloud' in df.columns:
            cloud_stats = df_finished.groupby('price_vs_cloud')['result'].apply(lambda x: (x=='WIN').mean()*100).reset_index()
            fig = px.bar(cloud_stats, x='price_vs_cloud', y='result', title="雲(Cloud)と勝率", labels={'result':'勝率%'})
            st.plotly_chart(fig, use_container_width=True)
        elif 'vol_ratio' in df.columns:
            df['vol_bin'] = pd.cut(df['vol_ratio'], bins=[0,0.8,1.2,2.0,10])
            vol_stats = df.groupby('vol_bin')['result'].apply(lambda x: (x=='WIN').mean()*100).reset_index()
            fig = px.bar(vol_stats, x=vol_stats['vol_bin'].astype(str), y='result', title="出来高倍率と勝率")
            st.plotly_chart(fig, use_container_width=True)

    # --- データ詳細 ---
    st.subheader("📝 トレード詳細")
    
    cols = ['Date', 'Ticker', 'Action', 'result', 'profit_rate', 'Reason', 'Actual_High', 'target_price']
    # 存在するカラムのみ表示
    show_cols = [c for c in cols if c in df.columns]
    
    st.dataframe(
        df[show_cols].sort_values('Date', ascending=False)
        .style.applymap(lambda x: 'color: red' if x=='LOSS' else 'color: green' if x=='WIN' else '', subset=['result'])
        .format({'profit_rate': '{:.2f}%'}),
        use_container_width=True
    )

if __name__ == "__main__":
    main()