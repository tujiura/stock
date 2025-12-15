import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# ★設定エリア
# ==========================================
# ファイルパス定義
FILES = {
    "🛡️ 資産防衛型 (Risk Managed)": "ai_trade_memory_risk_managed.csv",
    "🚀 攻撃型 (Aggressive V2)": "ai_trade_memory_aggressive_v6.csv",
    "最新版": "ai_trade_memory_aggressive_v5.csv"
}

PAGE_TITLE = "📊 AI Trade Analysis Dashboard"
LAYOUT = "wide"

# ==========================================
# 1. データ読み込み & 前処理
# ==========================================
def load_data(csv_file):
    if not os.path.exists(csv_file):
        return None
    
    try:
        df = pd.read_csv(csv_file)
        # 日付変換
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 共通の数値カラム変換
        numeric_cols = ['profit_rate', 'Confidence', 'Price']
        # 攻撃型特有カラム
        agg_cols = ['adx', 'ma_deviation', 'vol_ratio', 'dist_to_res']
        # 防衛型特有カラム
        def_cols = ['trend_momentum', 'entry_volatility', 'sma25_dev']
        
        for col in numeric_cols + agg_cols + def_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 損益額のシミュレーション (1トレード100万円投資と仮定)
        if 'profit_loss' not in df.columns:
            df['profit_loss'] = df['profit_rate'] * 10000  # 1% = 1万円
            
        # 累積損益 (資産推移)
        df = df.sort_values('Date')
        df['cumulative_profit'] = df['profit_loss'].cumsum()
        
        return df
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None

# ==========================================
# 2. メインアプリ
# ==========================================
def main():
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)
    st.title(PAGE_TITLE)
    
    # ------------------------------------------
    # サイドバー: モード切替 & フィルター
    # ------------------------------------------
    st.sidebar.header("⚙️ 設定・フィルター")
    
    # ★モード切替スイッチ
    selected_mode = st.sidebar.radio("分析モード選択", list(FILES.keys()), index=1)
    target_file = FILES[selected_mode]
    
    st.sidebar.markdown("---")
    
    df = load_data(target_file)
    if df is None or df.empty:
        st.warning(f"⚠️ データファイル (`{target_file}`) が見つかりません。\n\n先にトレーニングスクリプトを実行してCSVを生成してください。")
        return

    # 銘柄フィルター
    all_tickers = ["All"] + sorted(list(df['Ticker'].unique()))
    selected_ticker = st.sidebar.selectbox("銘柄フィルター", all_tickers)
    
    if selected_ticker != "All":
        df = df[df['Ticker'] == selected_ticker]

    if df.empty:
        st.warning("該当するデータがありません。")
        return

    # ------------------------------------------
    # KPI エリア (共通)
    # ------------------------------------------
    st.markdown(f"### {selected_mode} の分析結果")
    
    total_trades = len(df)
    wins = df[df['result'] == 'WIN']
    losses = df[df['result'] == 'LOSS']
    
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_profit = df['profit_loss'].sum()
    avg_profit = wins['profit_loss'].mean() if not wins.empty else 0
    avg_loss = losses['profit_loss'].mean() if not losses.empty else 0
    
    # プロフィットファクター (PF)
    gross_profit = wins['profit_loss'].sum()
    gross_loss = abs(losses['profit_loss'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("総トレード数", f"{total_trades}回")
    kpi2.metric("勝率", f"{win_rate:.1f}%", delta_color="normal")
    kpi3.metric("合計損益 (想定)", f"{total_profit:,.0f}円", delta=f"{total_profit:,.0f}円")
    kpi4.metric("プロフィットファクター", f"{pf:.2f}")
    kpi5.metric("リスクリワード比", f"{abs(avg_profit/avg_loss):.2f}" if avg_loss != 0 else "-")

    st.markdown("---")

    # ------------------------------------------
    # グラフエリア 1: 資産推移 & 自信度 (共通)
    # ------------------------------------------
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 資産推移シミュレーション")
        fig_equity = px.line(df, x='Date', y='cumulative_profit', markers=True, 
                             title="1トレード100万円投資時の推移")
        fig_equity.update_layout(xaxis_title="日付", yaxis_title="累積損益 (円)")
        st.plotly_chart(fig_equity, use_container_width=True)

    with c2:
        st.subheader("🤖 自信度分析")
        if 'Confidence' in df.columns and df['Confidence'].sum() > 0:
            df['conf_bin'] = pd.cut(df['Confidence'], bins=range(0, 101, 10), right=False)
            conf_stats = df.groupby('conf_bin')['result'].apply(lambda x: (x == 'WIN').mean() * 100).reset_index()
            conf_counts = df.groupby('conf_bin')['result'].count().reset_index()
            
            fig_conf = go.Figure()
            fig_conf.add_trace(go.Bar(x=conf_stats['conf_bin'].astype(str), y=conf_counts['result'], name="トレード数", opacity=0.3))
            fig_conf.add_trace(go.Scatter(x=conf_stats['conf_bin'].astype(str), y=conf_stats['result'], name="勝率(%)", yaxis="y2", mode='lines+markers'))
            
            fig_conf.update_layout(
                title="自信度と勝率の関係",
                yaxis=dict(title="トレード数"),
                yaxis2=dict(title="勝率(%)", overlaying="y", side="right", range=[0, 100]),
                legend=dict(x=0, y=1.2, orientation="h")
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("自信度データがありません")

    # ------------------------------------------
    # グラフエリア 2: 戦略別 詳細分析 (切り替え)
    # ------------------------------------------
    st.subheader(f"🔬 {selected_mode} 要因分析")
    
    t1, t2, t3 = st.columns(3)

    # ★攻撃型 (Aggressive) の場合
    if "攻撃型" in selected_mode:
        with t1:
            if 'adx' in df.columns:
                fig_adx = px.scatter(df, x='adx', y='profit_rate', color='result',
                                     title="ADX (トレンド強度) vs 利益", hover_data=['Ticker', 'Date'])
                fig_adx.add_vline(x=25, line_dash="dash", line_color="green", annotation_text="ADX>25")
                st.plotly_chart(fig_adx, use_container_width=True)
            else: st.info("ADXデータなし")

        with t2:
            if 'ma_deviation' in df.columns:
                fig_ma = px.scatter(df, x='ma_deviation', y='profit_rate', color='result',
                                    title="MA乖離率 vs 利益", hover_data=['Ticker'])
                fig_ma.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Overheat")
                st.plotly_chart(fig_ma, use_container_width=True)
            else: st.info("乖離率データなし")
            
        with t3:
            if 'vol_ratio' in df.columns:
                df['vol_bin'] = pd.cut(df['vol_ratio'], bins=[0, 0.8, 1.2, 2.0, 5.0, 10.0])
                vol_win_rate = df.groupby('vol_bin')['result'].apply(lambda x: (x == 'WIN').mean() * 100).reset_index()
                fig_vol = px.bar(vol_win_rate, x=vol_win_rate['vol_bin'].astype(str), y='result',
                                 title="出来高倍率ごとの勝率", labels={'result': '勝率(%)'})
                st.plotly_chart(fig_vol, use_container_width=True)
            else: st.info("出来高データなし")

    # ★資産防衛型 (Risk Managed) の場合
    else:
        with t1:
            if 'trend_momentum' in df.columns:
                fig_mom = px.scatter(df, x='trend_momentum', y='profit_rate', color='result',
                                     title="モメンタム vs 利益")
                st.plotly_chart(fig_mom, use_container_width=True)
            else: st.info("モメンタムデータなし")

        with t2:
            if 'entry_volatility' in df.columns:
                fig_vol = px.scatter(df, x='entry_volatility', y='profit_rate', color='result',
                                     title="ボラティリティ vs 利益")
                fig_vol.add_vrect(x0=0, x1=2.3, fillcolor="green", opacity=0.1, annotation_text="Safe Zone")
                st.plotly_chart(fig_vol, use_container_width=True)
            else: st.info("ボラティリティデータなし")

        with t3:
            if 'sma25_dev' in df.columns:
                fig_sma = px.histogram(df, x='sma25_dev', color='result', nbins=20,
                                       title="SMA25乖離率の分布", barmode='overlay')
                st.plotly_chart(fig_sma, use_container_width=True)
            else: st.info("SMAデータなし")

    # ------------------------------------------
    # データ詳細テーブル
    # ------------------------------------------
    st.subheader("📝 トレード履歴")
    
    # 表示カラムの選定
    base_cols = ['Date', 'Ticker', 'Action', 'result', 'Confidence', 'Price', 'profit_rate', 'Reason']
    extra_cols = ['adx', 'ma_deviation', 'vol_ratio'] if "攻撃型" in selected_mode else ['entry_volatility', 'trend_momentum']
    display_cols = [c for c in base_cols + extra_cols if c in df.columns]

    st.dataframe(
        df[display_cols]
        .sort_values('Date', ascending=False)
        .style.applymap(lambda x: 'color: red' if x == 'LOSS' else 'color: green' if x == 'WIN' else '', subset=['result'])
        .format({'profit_rate': '{:.2f}%', 'Price': '{:,.0f}', 'Confidence': '{:.0f}%'}),
        use_container_width=True
    )

if __name__ == "__main__":
    main()