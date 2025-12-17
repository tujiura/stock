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

PAGE_TITLE = "📊 AI Trade Analysis Dashboard V11"
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
        
        # 数値変換 (エラー回避)
        num_cols = ['profit_rate', 'adx', 'ma_deviation', 'vol_ratio', 'rsi', 'vwap_dev', 'choppiness']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        return df, None
    except Exception as e:
        return None, f"読み込みエラー: {e}"

# ==========================================
# 2. メインアプリ
# ==========================================
st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)
st.title(PAGE_TITLE)

# サイドバー: ファイル選択
selected_file_label = st.sidebar.selectbox("分析対象ファイル", list(FILES.keys()))
csv_path = FILES[selected_file_label]

df, error = load_data(csv_path)

if error:
    st.error(error)
else:
    # フィルタリング (完了したトレードのみ)
    df_finished = df[df['result'].isin(['WIN', 'LOSS', 'HOMERUN'])].copy()
    
    # KPI計算
    total_trades = len(df_finished)
    win_trades = len(df_finished[df_finished['result'].isin(['WIN', 'HOMERUN'])])
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_profit = df_finished[df_finished['profit_rate'] > 0]['profit_rate'].mean()
    avg_loss = df_finished[df_finished['profit_rate'] < 0]['profit_rate'].mean()
    profit_factor = abs(df_finished[df_finished['profit_rate'] > 0]['profit_rate'].sum() / 
                        df_finished[df_finished['profit_rate'] < 0]['profit_rate'].sum()) if len(df_finished[df_finished['profit_rate'] < 0]) > 0 else 0

    # --- KPI表示 ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("総トレード数", total_trades)
    c2.metric("勝率", f"{win_rate:.1f}%")
    c3.metric("平均利益", f"+{avg_profit:.2f}%" if pd.notnull(avg_profit) else "-")
    c4.metric("平均損失", f"{avg_loss:.2f}%" if pd.notnull(avg_loss) else "-")
    c5.metric("プロフィットファクター", f"{profit_factor:.2f}")

    st.markdown("---")

    # --- V11 特有の分析 (タブ切り替え) ---
    t1, t2, t3, t4, t5 = st.tabs(["📈 損益分布", "📊 テクニカル分析", "🌪️ ボラティリティ & レジーム", "🧠 AI思考", "📝 詳細データ"])

    with t1:
        # 損益ヒストグラム
        fig = px.histogram(df_finished, x='profit_rate', color='result', nbins=50, 
                           title="損益率分布", color_discrete_map={'WIN':'blue', 'LOSS':'red', 'HOMERUN':'gold'})
        st.plotly_chart(fig, use_container_width=True)
        
        # 累積損益 (単利ベースの簡易シミュレーション)
        df_finished = df_finished.sort_values('Date')
        df_finished['cumulative_profit'] = df_finished['profit_rate'].cumsum()
        fig2 = px.line(df_finished, x='Date', y='cumulative_profit', title="累積損益率の推移")
        st.plotly_chart(fig2, use_container_width=True)

    with t2:
        c_left, c_right = st.columns(2)
        
        with c_left:
            # ADX vs 損益
            fig = px.scatter(df_finished, x='adx', y='profit_rate', color='result', title="ADX vs 利益率")
            st.plotly_chart(fig, use_container_width=True)
            
            # MA乖離率 vs 損益
            fig = px.scatter(df_finished, x='ma_deviation', y='profit_rate', color='result', title="MA乖離 vs 利益率")
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            # RSI vs 損益 (V11)
            if 'rsi' in df.columns:
                fig = px.scatter(df_finished, x='rsi', y='profit_rate', color='result', title="RSI vs 利益率")
                st.plotly_chart(fig, use_container_width=True)
            
            # VWAP乖離 vs 損益 (V11)
            if 'vwap_dev' in df.columns:
                fig = px.scatter(df_finished, x='vwap_dev', y='profit_rate', color='result', title="VWAP乖離 vs 利益率")
                st.plotly_chart(fig, use_container_width=True)

    with t3:
        # V11 レジーム分析
        if 'regime' in df.columns:
            st.subheader("🌍 市場局面 (Regime) 別パフォーマンス")
            regime_stats = df_finished.groupby('regime').agg(
                Trades=('result', 'count'),
                WinRate=('result', lambda x: (x.isin(['WIN', 'HOMERUN'])).mean() * 100),
                AvgProfit=('profit_rate', 'mean')
            ).reset_index()
            
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                fig = px.bar(regime_stats, x='regime', y='WinRate', title="局面別 勝率", color='WinRate')
                st.plotly_chart(fig, use_container_width=True)
            with c_r2:
                fig = px.bar(regime_stats, x='regime', y='AvgProfit', title="局面別 平均利益率", color='AvgProfit')
                st.plotly_chart(fig, use_container_width=True)

        if 'choppiness' in df.columns:
            st.subheader("🌊 Choppiness Index (トレンド強度)")
            fig = px.histogram(df_finished, x='choppiness', color='result', nbins=30, title="CHOP指数の分布と勝敗")
            st.plotly_chart(fig, use_container_width=True)

    with t4:
        # V11 CoT分析
        st.subheader("🧠 AIの判断ロジック (Chain of Thought)")
        
        # RSI Divergence
        if 'rsi_divergence' in df.columns:
            div_counts = df_finished['rsi_divergence'].value_counts().reset_index()
            div_counts.columns = ['Divergence Type', 'Count']
            fig = px.pie(div_counts, values='Count', names='Divergence Type', title="RSIダイバージェンス検出比率")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**ダイバージェンス発生時の平均利益:**")
            st.dataframe(df_finished.groupby('rsi_divergence')['profit_rate'].mean())

    with t5:
        # データ詳細
        st.subheader("📝 トレード詳細データ")
        
        # V11用の表示カラム選択
        display_cols = ['Date', 'Ticker', 'Action', 'result', 'profit_rate', 'Reason']
        v11_cols = ['regime', 'rsi', 'vwap_dev', 'choppiness', 'rsi_divergence']
        
        # 存在するカラムだけ追加
        for c in v11_cols:
            if c in df.columns:
                display_cols.append(c)
                
        st.dataframe(df_finished[display_cols].sort_values('Date', ascending=False), use_container_width=True)