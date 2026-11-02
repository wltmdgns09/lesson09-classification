import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/wltmdgns09/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()

# 뇌졸중 여부를 사람이 읽기 쉬운 글자로 바꾼 열 (그래프와 표에서 사용)
df["뇌졸중 여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})
group_order = ["겪지 않음", "겪음"]

st.title("탐색")
st.write("같은 데이터를 여러 방향에서 살펴보는 페이지입니다.")

# 1. 나이와 평균 혈당의 분포 (히스토그램 두 개, 나란히)
st.header("1. 나이와 평균 혈당의 분포")
col1, col2 = st.columns(2)
with col1:
    fig_age = px.histogram(df, x="age", title="나이 분포")
    st.plotly_chart(fig_age)
with col2:
    fig_glucose = px.histogram(df, x="avg_glucose_level", title="평균 혈당 분포")
    st.plotly_chart(fig_glucose)

# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교 (상자그림 두 개 + 평균 표)
st.header("2. 뇌졸중 여부에 따른 비교")
col3, col4 = st.columns(2)
with col3:
    fig_box_age = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        category_orders={"뇌졸중 여부": group_order},
        title="나이",
    )
    st.plotly_chart(fig_box_age)
with col4:
    fig_box_glucose = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        category_orders={"뇌졸중 여부": group_order},
        title="평균 혈당",
    )
    st.plotly_chart(fig_box_glucose)

mean_table = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .round(2)
    .reindex(group_order)
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당 평균"})
)
st.write("두 그룹의 평균값")
st.table(mean_table)

# 3. 고혈압, 심장병 여부별 뇌졸중 비율 (막대그래프)
st.header("3. 고혈압·심장병 여부별 뇌졸중 비율")
col5, col6 = st.columns(2)


def stroke_rate_by(column, label):
    rate = df.groupby(column)["stroke"].mean().mul(100).round(2).reset_index()
    rate[column] = rate[column].map({0: "없음", 1: "있음"})
    rate = rate.rename(columns={column: label, "stroke": "뇌졸중 비율(%)"})
    return rate


with col5:
    rate_hyper = stroke_rate_by("hypertension", "고혈압")
    fig_hyper = px.bar(
        rate_hyper,
        x="고혈압",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="고혈압 여부별 뇌졸중 비율",
    )
    st.plotly_chart(fig_hyper)
with col6:
    rate_heart = stroke_rate_by("heart_disease", "심장병")
    fig_heart = px.bar(
        rate_heart,
        x="심장병",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="심장병 여부별 뇌졸중 비율",
    )
    st.plotly_chart(fig_heart)

# 4. bmi가 비어 있는 사람들의 뇌졸중 비율 vs 전체 뇌졸중 비율
st.header("4. bmi가 비어 있는 사람들")
bmi_missing = df[df["bmi"].isna()]
bmi_table = pd.DataFrame(
    {
        "사람 수": [len(bmi_missing), len(df)],
        "뇌졸중 겪은 사람 수": [int(bmi_missing["stroke"].sum()), int(df["stroke"].sum())],
        "뇌졸중 비율(%)": [
            round(bmi_missing["stroke"].mean() * 100, 2),
            round(df["stroke"].mean() * 100, 2),
        ],
    },
    index=["bmi가 빈 사람들", "전체"],
)
st.table(bmi_table)

# 5. 흡연 상태의 값마다 사람 수
st.header("5. 흡연 상태별 사람 수")
smoking_table = (
    df["smoking_status"]
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.table(smoking_table)
