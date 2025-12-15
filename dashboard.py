import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# ★設定エリア
# ==========================================
# ファイルパス定義 (V6を追加)
FILES = {
    "🚀 攻撃型 V6 (最新: 目標到達分析)": "ai_trade_memory_aggressive_v7.csv",
    "🔥 攻撃型 V5 (分割決済)": "ai_trade_memory_aggressive_v5.csv",
    "🛡️ 資産防衛型 (Risk Managed)": "ai_trade_memory_risk_managed.csv",
}

PAGE_TITLE = "📊 AI Trade Analysis Dashboard V6"
LAYOUT = "wide"

# ==========================================
# 1. データ読み込み & 前処理
# ==========================================
def load_data(csv_file):
    if not os.path.exists(csv_file):
        return None, f"ファイルが見つかりません: {csv_file}"
    
    try:
        # ファイル読み込み
        df = pd.read_csv(csv_file)
        
        # カラム名の空白除去 (エラー対策)
        df.columns = [c.strip() for c in df.columns]
        
        # 必須カラム 'Date' の確認
        if 'Date' not in df.columns:
            return None, "CSVに 'Date' 列が見つかりません。"

        # 日付変換
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 数値型変換 (V6の新カラム含む)
        numeric_cols = [
            'profit_rate', 'Confidence', 'Price', 
            'adx', 'ma_deviation', 'vol_ratio', 'dist_to_res',
            'Actual_High', 'target_price', 'Target_Reach', 'Target_Diff'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 損益額のシミュレーション (1トレード100万円投資と仮定)
        if 'profit_loss' not in df.columns:
            df['profit_loss'] = df['profit_rate'] * 10000  # 1% = 1万円
            
        # 累積損益 (資産推移)
        df = df.sort_values('Date')
        df['cumulative_profit'] = df['profit_loss'].cumsum()
        
        return df, None
    except Exception as e:
        return None, f"データ読み込みエラー: {e}"

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
    selected_mode = st.sidebar.radio("分析モード選択", list(FILES.keys()), index=0)
    target_file = FILES[selected_mode]
    
    st.sidebar.markdown("---")
    
    df, error_msg = load_data(target_file)
    if df is None:
        st.error(f"⚠️ {error_msg}")
        st.info("トレーニングスクリプトを実行してCSVを生成してください。")
        return

    if df.empty:
        st.warning("データが空です。")
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
    
    # 完了したトレードのみで集計
    df_finished = df[df['result'].isin(['WIN', 'LOSS'])]
    
    total_trades = len(df_finished)
    wins = df_finished[df_finished['result'] == 'WIN']
    losses = df_finished[df_finished['result'] == 'LOSS']
    
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_profit = df_finished['profit_loss'].sum()
    avg_profit = wins['profit_rate'].mean() if not wins.empty else 0
    avg_loss = losses['profit_rate'].mean() if not losses.empty else 0
    
    # プロフィットファクター (PF)
    gross_profit = wins['profit_loss'].sum()
    gross_loss = abs(losses['profit_loss'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # 目標到達率 (V6専用)
    target_reach_kpi = "-"
    if 'Target_Reach' in df.columns:
        reached = df_finished[df_finished['Target_Reach'] >= 100]
        rate = len(reached) / len(df_finished) * 100 if len(df_finished) > 0 else 0
        target_reach_kpi = f"{rate:.1f}%"

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("総トレード数", f"{total_trades}回")
    kpi2.metric("勝率", f"{win_rate:.1f}%", delta_color="normal")
    kpi3.metric("合計損益 (想定)", f"{total_profit:,.0f}円", delta=f"{total_profit:,.0f}円")
    kpi4.metric("プロフィットファクター", f"{pf:.2f}")
    kpi5.metric("目標価格 到達率", target_reach_kpi)

    st.markdown("---")

    # ------------------------------------------
    # グラフエリア 1: 資産推移 & 目標分析 (V6対応)
    # ------------------------------------------
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 資産推移シミュレーション")
        fig_equity = px.line(df_finished, x='Date', y='cumulative_profit', markers=True, 
                             title="1トレード100万円投資時の推移")
        fig_equity.update_layout(xaxis_title="日付", yaxis_title="累積損益 (円)")
        st.plotly_chart(fig_equity, use_container_width=True)

    with c2:
        if 'Target_Reach' in df.columns:
            st.subheader("🎯 目標到達率の分布")
            # 異常値を除外して表示
            valid_reach = df_finished[(df_finished['Target_Reach'] > -50) & (df_finished['Target_Reach'] < 300)]
            fig_hist = px.histogram(valid_reach, x="Target_Reach", nbins=20, 
                                    title="目標達成率 (%)", color="result")
            fig_hist.add_vline(x=100, line_dash="dash", line_color="green", annotation_text="Target")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.subheader("🤖 自信度分析")
            if 'Confidence' in df.columns:
                df['conf_bin'] = pd.cut(df['Confidence'], bins=range(0, 101, 10), right=False)
                conf_stats = df.groupby('conf_bin')['result'].apply(lambda x: (x == 'WIN').mean() * 100).reset_index()
                fig_conf = px.bar(conf_stats, x=conf_stats['conf_bin'].astype(str), y='result', title="自信度別勝率")
                st.plotly_chart(fig_conf, use_container_width=True)

    # ------------------------------------------
    # グラフエリア 2: 戦略別 詳細分析
    # ------------------------------------------
    st.subheader(f"🔬 {selected_mode} 要因分析")
    
    t1, t2, t3 = st.columns(3)

    # ★攻撃型 (Aggressive) の場合
    if "攻撃型" in selected_mode:
        with t1:
            if 'adx' in df.columns:
                fig_adx = px.scatter(df_finished, x='adx', y='profit_rate', color='result',
                                     title="ADX (トレンド強度) vs 利益", hover_data=['Ticker', 'Date'])
                fig_adx.add_vline(x=25, line_dash="dash", line_color="green", annotation_text="ADX>25")
                st.plotly_chart(fig_adx, use_container_width=True)
            else: st.info("ADXデータなし")

        with t2:
            if 'ma_deviation' in df.columns:
                fig_ma = px.scatter(df_finished, x='ma_deviation', y='profit_rate', color='result',
                                    title="MA乖離率 vs 利益", hover_data=['Ticker'])
                fig_ma.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Overheat")
                st.plotly_chart(fig_ma, use_container_width=True)
            else: st.info("乖離率データなし")
            
        with t3:
            if 'vol_ratio' in df.columns:
                df_finished['vol_bin'] = pd.cut(df_finished['vol_ratio'], bins=[0, 0.8, 1.2, 2.0, 5.0, 10.0])
                vol_win_rate = df_finished.groupby('vol_bin')['result'].apply(lambda x: (x == 'WIN').mean() * 100).reset_index()
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
                st.plotly_chart(fig_vol, use_container_width=True)
            else: st.info("ボラティリティデータなし")

    # ------------------------------------------
    # データ詳細テーブル
    # ------------------------------------------
    st.subheader("📝 トレード履歴詳細")
    
    # 表示カラムの選定 (V6対応)
    base_cols = ['Date', 'Ticker', 'Action', 'result', 'profit_rate', 'Reason']
    v6_cols = ['target_price', 'Actual_High', 'Target_Reach']
    agg_cols = ['adx', 'ma_deviation', 'vol_ratio']
    
    # 存在するカラムだけを選択
    display_cols = [c for c in base_cols + v6_cols + agg_cols if c in df.columns]

    st.dataframe(
        df[display_cols]
        .sort_values('Date', ascending=False)
        .style.applymap(lambda x: 'color: red' if x == 'LOSS' else 'color: green' if x == 'WIN' else '', subset=['result'])
        .format({
            'profit_rate': '{:.2f}%', 
            'Price': '{:,.0f}', 
            'target_price': '{:,.0f}',
            'Actual_High': '{:,.0f}',
            'Target_Reach': '{:.1f}%'
        }),
        use_container_width=True
    )

if __name__ == "__main__":
    main()