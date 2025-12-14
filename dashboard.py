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
st.title("🎯 AI Sniper Control Room v3.0")
st.caption("資産防衛型AI 自動売買・戦績分析ダッシュボード")

# ==========================================
# 1. データ読み込み & クリーニング
# ==========================================
DATA_FILE = "ai_trade_memory_risk_managed.csv" 

@st.cache_data(ttl=60)
def load_and_clean_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    
    try:
        # ファイル読み込み (エラー行は無視)
        df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
        
        # 1. カラム名の正規化 (空白削除・小文字化)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # マッピング辞書 (表記ゆれ対応)
        col_map = {
            'date': 'Date', 'ticker': 'Ticker', 'action': 'Action', 
            'result': 'Result', 'reason': 'Reason', 'confidence': 'Confidence',
            'price': 'Entry_Price', 
            'profit_loss': 'Profit_Loss', 'profit_rate': 'Profit_Rate',
            'rsi_9': 'RSI', 'rsi': 'RSI',
            'stop_loss_reason': 'Exit_Reason'
        }
        # 辞書にあるカラム名のみリネーム
        new_cols = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=new_cols)
        
        # 必須カラムの確認
        if 'Date' not in df.columns or 'Result' not in df.columns:
            return pd.DataFrame()

        # 2. 日付変換
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date')

        # 3. 数値変換 (強制)
        num_cols = ['Profit_Loss', 'Profit_Rate', 'Entry_Price', 'RSI', 'Confidence']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        # 4. 異常値の補正ロジック
        # Profit_Rate が異常に大きい(絶対値が500%超え)場合、それは円(Profit_Loss)が誤って入っている可能性が高い
        # グラフが見づらくなるため、これらを統計から除外するか、補正する
        # ここでは「投資額100万円」と仮定して逆算補正を試みる
        mask_anomaly = df['Profit_Rate'].abs() > 500
        if mask_anomaly.any():
            # 異常値は Profit_Loss / 10000 (100万円投資想定) で再計算
            df.loc[mask_anomaly, 'Profit_Rate'] = df.loc[mask_anomaly, 'Profit_Loss'] / 10000.0

        # 5. 累積損益 (Equity)
        df['Equity'] = df['Profit_Loss'].cumsum()
        
        return df

    except Exception as e:
        st.error(f"データ読み込み中にエラーが発生しました: {e}")
        return pd.DataFrame()

df = load_and_clean_data()

# サイドバー: 更新ボタン
if st.sidebar.button('🔄 データを最新に更新'):
    st.cache_data.clear()
    st.rerun()

# データチェック
if df.empty:
    st.warning("⚠️ 表示できるデータがありません。プログラムを実行してデータを蓄積してください。")
    st.stop()

# ==========================================
# 2. KPI ボード
# ==========================================
# BUYエントリーで、結果が出ているものだけ抽出
df_res = df[(df['Action'] == 'BUY') & (df['Result'].isin(['WIN', 'LOSS']))].copy()

if not df_res.empty:
    # 集計
    total_trades = len(df_res)
    wins = len(df_res[df_res['Result'] == 'WIN'])
    losses = len(df_res[df_res['Result'] == 'LOSS'])
    win_rate = (wins / total_trades) * 100
    total_pl = df_res['Profit_Loss'].sum()
    
    # 平均値
    avg_win_pl = df_res[df_res['Result'] == 'WIN']['Profit_Loss'].mean()
    avg_loss_pl = df_res[df_res['Result'] == 'LOSS']['Profit_Loss'].mean()
    avg_win_rate = df_res[df_res['Result'] == 'WIN']['Profit_Rate'].mean()
    avg_loss_rate = df_res[df_res['Result'] == 'LOSS']['Profit_Rate'].mean()

    # プロフィットファクター
    gross_profit = df_res[df_res['Profit_Loss'] > 0]['Profit_Loss'].sum()
    gross_loss = abs(df_res[df_res['Profit_Loss'] < 0]['Profit_Loss'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # 表示
    st.markdown("### 📊 パフォーマンス概要")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.metric("💰 合計損益", f"{total_pl:,.0f} 円", delta_color="normal")
    c2.metric("🎯 勝率", f"{win_rate:.1f} %", f"{wins}勝 {losses}敗")
    c3.metric("⚖️ PF (期待値)", f"{pf:.2f}")
    c4.metric("📈 平均利益", f"+{avg_win_rate:.2f}%", f"¥{avg_win_pl:,.0f}")
    c5.metric("📉 平均損失", f"{avg_loss_rate:.2f}%", f"¥{avg_loss_pl:,.0f}")

else:
    st.info("まだ決済されたトレードがありません。")

st.markdown("---")

# ==========================================
# 3. メインチャート
# ==========================================
col_main, col_side = st.columns([3, 1])

with col_main:
    st.subheader("📈 資産成長曲線 (Equity Curve)")
    if not df.empty:
        fig_eq = px.line(df, x='Date', y='Equity', markers=True,
                         title="損益の積み上げ推移",
                         labels={'Equity': '累積損益(円)', 'Date': '日付'})
        
        # ゼロライン強調
        fig_eq.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # トレンドラインの色分け (プラスなら緑、マイナスなら赤)
        line_color = '#00cc96' if df['Equity'].iloc[-1] >= 0 else '#EF553B'
        fig_eq.update_traces(line_color=line_color)
        
        st.plotly_chart(fig_eq, use_container_width=True)

with col_side:
    st.subheader("📊 損益分布")
    if not df_res.empty:
        # ヒストグラム
        fig_hist = px.histogram(df_res, x="Profit_Rate", nbins=30,
                                color="Result",
                                color_discrete_map={'WIN':'#00cc96', 'LOSS':'#EF553B'},
                                title="1トレードの利益率分布",
                                labels={'Profit_Rate': '利益率(%)'})
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# 4. 分析エリア (2カラム)
# ==========================================
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("🛑 決済理由 (Exit Analysis)")
    if 'Exit_Reason' in df_res.columns:
        # 欠損値を埋める
        df_res['Exit_Reason'] = df_res['Exit_Reason'].replace('', 'Unknown').fillna('Unknown')
        
        reason_counts = df_res['Exit_Reason'].value_counts().reset_index()
        reason_counts.columns = ['Reason', 'Count']
        
        fig_pie = px.pie(reason_counts, names='Reason', values='Count', hole=0.4,
                         title="決済トリガーの内訳")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.caption("決済理由データがありません")

with c_right:
    st.subheader("🏆 銘柄別 損益 (Best & Worst)")
    if not df_res.empty:
        ticker_pl = df_res.groupby('Ticker')['Profit_Loss'].sum().sort_values(ascending=False)
        # 上位5と下位5を結合して表示
        top5 = ticker_pl.head(5)
        worst5 = ticker_pl.tail(5)
        disp_ticker = pd.concat([top5, worst5]).sort_values(ascending=True) # グラフ用に昇順
        
        fig_bar = px.bar(x=disp_ticker.values, y=disp_ticker.index, orientation='h',
                         title="銘柄別 累計損益",
                         labels={'x': '損益(円)', 'y': '銘柄'},
                         color=disp_ticker.values,
                         color_continuous_scale=['red', 'gray', 'green'])
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# 5. 詳細データテーブル
# ==========================================
st.subheader("📝 トレード履歴一覧")

# 表示用カラムの選定
cols_to_show = ['Date', 'Ticker', 'Result', 'Profit_Loss', 'Profit_Rate', 'Entry_Price', 'RSI', 'Confidence', 'Reason']
available_cols = [c for c in cols_to_show if c in df.columns]

if not df.empty:
    # フィルタリング (サイドバー)
    ticker_filter = st.sidebar.selectbox("銘柄フィルタ", ["ALL"] + list(df['Ticker'].unique()))
    
    df_view = df.copy()
    if ticker_filter != "ALL":
        df_view = df_view[df_view['Ticker'] == ticker_filter]
        
    # 最新順
    df_view = df_view.sort_values('Date', ascending=False)

    # スタイリング関数
    def style_result(val):
        color = 'green' if val == 'WIN' else 'red' if val == 'LOSS' else 'black'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df_view[available_cols].style.map(style_result, subset=['Result'])
        .format({
            'Profit_Loss': '{:+,.0f}', 
            'Profit_Rate': '{:+.2f}%', 
            'Entry_Price': '{:,.0f}',
            'RSI': '{:.1f}',
            'Confidence': '{:.0f}%'
        }),
        use_container_width=True
    )