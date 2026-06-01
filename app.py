# app.py
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import os

# -----------------------------
# Kaggle 인증 (Streamlit Secrets → 환경변수 주입)
# -----------------------------
if "kaggle" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
    os.environ["KAGGLE_KEY"]      = st.secrets["kaggle"]["key"]

# -----------------------------
# 한글 폰트 설정
# -----------------------------
def set_korean_font():
    if platform.system() == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif platform.system() == "Darwin":
        plt.rc("font", family="AppleGothic")
    else:
        candidates = [f for f in fm.findSystemFonts() if any(k in f for k in ["Nanum", "NotoSansCJK", "CJK"])]
        if candidates:
            nanum = [f for f in candidates if "Nanum" in f]
            chosen = nanum[0] if nanum else candidates[0]
            fm.fontManager.addfont(chosen)
            font_name = fm.FontProperties(fname=chosen).get_name()
            plt.rc("font", family=font_name)
    plt.rcParams["axes.unicode_minus"] = False

sns.set()
set_korean_font()

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="세계 지진 위험도 예측",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 세계 지진 위험도 예측 시스템")
st.caption("1990–2023 지진 데이터 기반 K-Means 군집 분석")

# -----------------------------
# 모델 / 스케일러 로드
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model-earthquake.pkl")
    scaler = joblib.load("earthquake-scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -----------------------------
# 데이터 로드 (kagglehub으로 자동 다운로드)
# -----------------------------
@st.cache_data
def load_data():
    csv_filename = "Eartquakes-1990-2023.csv"

    # 로컬에 없으면 kagglehub으로 다운로드
    if not os.path.exists(csv_filename):
        with st.spinner("📥 데이터 다운로드 중... (최초 1회, 약 1분 소요)"):
            import kagglehub
            path = kagglehub.dataset_download(
                "alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023"
            )
            csv_path = os.path.join(path, csv_filename)
    else:
        csv_path = csv_filename

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df_eq = df[df['data_type'] == 'earthquake'].copy()
    df_new = df_eq[df_eq['date'].dt.year >= 2022].copy()
    df_new = df_new[df_new['magnitudo'] >= 0.3].copy()
    df_sample = df_new.sample(500, random_state=42).copy()

    df_sample = df_sample.rename(columns={
        'tsunami': '쓰나미여부',
        'depth': '진원깊이',
        'magnitudo': '규모',
        'longitude': '경도',
        'latitude': '위도',
    })

    X = df_sample[["쓰나미여부", "진원깊이", "규모"]]
    X_scaled = scaler.transform(X)
    df_sample['cluster'] = model.predict(X_scaled)

    return df_sample

df_sample = load_data()

risk_dict  = {0: '낮음', 1: '중간', 2: '높음'}
color_dict = {0: 'blue',  1: 'green', 2: 'red'}
df_sample['위험도'] = df_sample['cluster'].map(risk_dict)

# -----------------------------
# 사이드바 입력
# -----------------------------
st.sidebar.header("📍 위치 입력")
st.sidebar.markdown("예측할 지점의 위도/경도를 입력하세요.")

lat = st.sidebar.number_input("위도 (Latitude)",  min_value=-90.0,  max_value=90.0,  value=35.6,  step=0.1, format="%.4f")
lon = st.sidebar.number_input("경도 (Longitude)", min_value=-180.0, max_value=180.0, value=139.7, step=0.1, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("**탐색 반경**: ±5도 이내 지진 데이터로 위험도 판정")

# -----------------------------
# 예측 로직
# -----------------------------
near_df = df_sample[
    (df_sample['위도'] >= lat - 5) & (df_sample['위도'] <= lat + 5) &
    (df_sample['경도'] >= lon - 5) & (df_sample['경도'] <= lon + 5)
]

# -----------------------------
# 결과 출력
# -----------------------------
st.subheader("🤖 예측 결과")

col1, col2, col3 = st.columns(3)

if len(near_df) == 0:
    st.warning("⚠️ 입력한 위치 주변(±5도)에 분석 데이터가 없습니다. 다른 위치를 입력해보세요.")
    main_cluster = None
else:
    cluster_ratio = near_df['cluster'].value_counts(normalize=True)
    main_cluster  = cluster_ratio.idxmax()
    risk_label    = risk_dict[main_cluster]

    with col1:
        st.metric("주변 지진 데이터 수", f"{len(near_df)}건")
    with col2:
        st.metric("판정 군집", f"군집 {main_cluster}")
    with col3:
        st.metric("위험도", risk_label)

    if main_cluster == 2:
        st.error(f"🔴 위험도 **{risk_label}** — 쓰나미 동반 가능성 있는 고위험 지역입니다.")
    elif main_cluster == 1:
        st.warning(f"🟡 위험도 **{risk_label}** — 심발 지진이 발생하는 중간 위험 지역입니다.")
    else:
        st.success(f"🔵 위험도 **{risk_label}** — 상대적으로 위험도가 낮은 지역입니다.")

    # 군집 비율 바차트
    st.markdown("**주변 군집 비율**")
    ratio_df = cluster_ratio.reset_index()
    ratio_df.columns = ['군집', '비율']
    ratio_df['위험도']  = ratio_df['군집'].map(risk_dict)
    ratio_df['비율(%)'] = (ratio_df['비율'] * 100).round(1)

    fig_bar, ax_bar = plt.subplots(figsize=(5, 2.5))
    bars = ax_bar.barh(
        ratio_df['위험도'],
        ratio_df['비율(%)'],
        color=[color_dict[c] for c in ratio_df['군집']],
        alpha=0.8
    )
    ax_bar.set_xlabel("비율 (%)")
    ax_bar.set_title("주변 지진 군집 비율")
    for bar, val in zip(bars, ratio_df['비율(%)']):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{val}%", va='center')
    st.pyplot(fig_bar)

# -----------------------------
# 시각화 탭
# -----------------------------
st.subheader("📊 데이터 시각화")

tab1, tab2, tab3 = st.tabs(["진원깊이 vs 규모", "쓰나미 vs 규모", "군집 분포 지도"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, color in color_dict.items():
        subset = df_sample[df_sample['cluster'] == c]
        ax.scatter(subset['진원깊이'], subset['규모'], c=color, label=f"군집{c}({risk_dict[c]})", alpha=0.5, s=15)
    ax.set_xlabel("진원깊이 (km)")
    ax.set_ylabel("규모")
    ax.set_title("진원깊이 vs 규모 군집 분포")
    ax.legend()
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, color in color_dict.items():
        subset = df_sample[df_sample['cluster'] == c]
        ax.scatter(subset['쓰나미여부'], subset['규모'], c=color, label=f"군집{c}({risk_dict[c]})", alpha=0.5, s=15)
    ax.set_xlabel("쓰나미 여부 (0/1)")
    ax.set_ylabel("규모")
    ax.set_title("쓰나미 여부 vs 규모 군집 분포")
    ax.legend()
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(12, 6))
    for c, color in color_dict.items():
        subset = df_sample[df_sample['cluster'] == c]
        ax.scatter(subset['경도'], subset['위도'], c=color, label=f"군집{c}({risk_dict[c]})", alpha=0.5, s=15)
    if main_cluster is not None:
        ax.scatter(lon, lat, c='black', s=200, marker='*', zorder=5, label="입력 위치")
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.set_title("세계 지진 군집 분포 (500개 샘플)")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# 군집별 통계
# -----------------------------
st.subheader("📈 군집별 평균 통계")
stats = df_sample.groupby('위험도')[["쓰나미여부","진원깊이","규모"]].mean().round(3)
st.dataframe(stats)

# -----------------------------
# 하단
# -----------------------------
st.markdown("---")
st.caption("Streamlit 기반 세계 지진 군집 분석 시스템 | 데이터: Kaggle 1990–2023")
