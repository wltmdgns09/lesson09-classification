import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/wltmdgns09/data/refs/heads/main/stroke.csv"
NO_VALUE = "계산할 수 없음"

LOGIT_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "훈련용에서 많은 쪽으로만 답하는 모델"

# 화면에 보이는 이름 -> 데이터의 열 이름
FEATURES = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease",
}
DEFAULT_FEATURES = ["나이", "평균 혈당", "고혈압", "심장병"]  # 처음엔 체질량지수만 뺀 넷

st.set_page_config(page_title="채점표", layout="wide")
st.title("채점표")
st.caption("테스트용 사람들을 두 모델이 얼마나 맞혔는지 채점합니다. 뇌졸중을 '양성'으로 봅니다.")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()

# ---------- 입력 속성 고르기 ----------
picked = st.multiselect("입력 속성", list(FEATURES.keys()), default=DEFAULT_FEATURES)
if not picked:
    st.warning("입력 속성을 하나 이상 골라 주세요.")
    st.stop()
cols = [FEATURES[name] for name in picked]

# ---------- 나누기: 번호순으로 열 명씩 묶어 앞 세 명은 테스트용, 나머지는 훈련용 ----------
df = df.sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3
train_df = df[~is_test].copy()
test_df = df[is_test].copy()

# 체질량지수의 빈 값은 훈련용의 중앙값으로 채움 (고른 경우에만 해당)
if "bmi" in cols:
    median_bmi = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(median_bmi)
    test_df["bmi"] = test_df["bmi"].fillna(median_bmi)

X_train, y_train = train_df[cols], train_df["stroke"]
X_test, y_test = test_df[cols], test_df["stroke"]

# ---------- 모델 학습 ----------
logit = LogisticRegression(max_iter=1000).fit(X_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
majority = y_train.value_counts().idxmax()

preds = {
    LOGIT_NAME: pd.Series(logit.predict(X_test), index=test_df.index),
    TREE_NAME: pd.Series(tree.predict(X_test), index=test_df.index),
    BASE_NAME: pd.Series(majority, index=test_df.index),
}


# ---------- 네 칸 세기와 지표 ----------
def count_cells(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def fmt(value):
    return f"{value:.3f}"


def four_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) > 0 else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None  # 분모: 뇌졸중이라 예측한 사람 수
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "정확도": fmt(accuracy),
        "재현율": fmt(recall) if recall is not None else NO_VALUE,
        "정밀도": fmt(precision) if precision is not None else NO_VALUE,
        "F1": fmt(f1) if f1 is not None else NO_VALUE,
    }


def confusion_table(tp, fp, fn, tn):
    table = pd.DataFrame(
        [
            [f"TP {tp:,}명", f"FN {fn:,}명", f"{tp + fn:,}명"],
            [f"FP {fp:,}명", f"TN {tn:,}명", f"{fp + tn:,}명"],
            [f"{tp + fp:,}명", f"{fn + tn:,}명", f"{tp + fp + fn + tn:,}명"],
        ],
        index=["실제 뇌졸중", "실제 뇌졸중 아님", "합계"],
        columns=["예측 뇌졸중", "예측 뇌졸중 아님", "합계"],
    )
    table.index.name = "실제 ↓ / 예측 →"
    return table


cells = {name: count_cells(y_test, p) for name, p in preds.items()}

# ---------- 혼동행렬 ----------
st.header("혼동행렬")
st.caption("행은 실제, 열은 예측입니다. 테스트 데이터의 사람 수를 적었습니다.")
left, right = st.columns(2)
for box, name in zip((left, right), (LOGIT_NAME, TREE_NAME)):
    with box:
        st.subheader(name)
        st.table(confusion_table(*cells[name]))

# ---------- 지표 비교표 ----------
st.header("네 가지 점수 비교")
n_test = len(test_df)
n_pos = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 중 뇌졸중인 사람은 {n_pos:,}명입니다.")

metric_rows = {name: four_metrics(*cells[name]) for name in (LOGIT_NAME, TREE_NAME, BASE_NAME)}
metric_table = pd.DataFrame(metric_rows).T
metric_table.index.name = "모델"
st.table(metric_table)
st.caption(
    "재현율 = TP ÷ (TP + FN), 분모는 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP ÷ (TP + FP), 분모는 뇌졸중이라 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율) · "
    "정확도 = (TP + TN) ÷ 전체 사람 수"
)

# ---------- 틀린 사람 목록 ----------
st.header("틀린 사람 목록")
chosen = st.selectbox("어느 모델의 목록을 볼까요?", [LOGIT_NAME, TREE_NAME])
pred = preds[chosen]

MAX_ROWS = 20


def make_list(mask):
    part = test_df[mask].sort_values("age", ascending=False)
    total = len(part)
    shown = pd.DataFrame(
        {
            "번호": part["id"],
            "나이": part["age"],
            "평균 혈당": part["avg_glucose_level"],
            "고혈압 여부": part["hypertension"].map({1: "예", 0: "아니오"}),
            "심장병 여부": part["heart_disease"].map({1: "예", 0: "아니오"}),
        }
    ).head(MAX_ROWS)
    return shown, total


missed, n_missed = make_list((y_test == 1) & (pred == 0))
wrong, n_wrong = make_list((y_test == 0) & (pred == 1))

left, right = st.columns(2)
with left:
    title = f"놓친 환자 (실제 뇌졸중인데 아니라고 예측) · 전체 {n_missed:,}명"
    if n_missed > MAX_ROWS:
        title += f" 중 앞 {MAX_ROWS}명"
    st.subheader(title)
    st.dataframe(missed, hide_index=True, use_container_width=True)
with right:
    title = f"헛짚은 사람 (아닌데 뇌졸중이라 예측) · 전체 {n_wrong:,}명"
    if n_wrong > MAX_ROWS:
        title += f" 중 앞 {MAX_ROWS}명"
    st.subheader(title)
    st.dataframe(wrong, hide_index=True, use_container_width=True)

_, fp_cell, fn_cell, _ = cells[chosen]
same = (n_missed == fn_cell) and (n_wrong == fp_cell)
st.write(
    f"놓친 환자 {n_missed:,}명은 혼동행렬 FN 칸 {fn_cell:,}명과, "
    f"헛짚은 사람 {n_wrong:,}명은 FP 칸 {fp_cell:,}명과 "
    + ("같습니다." if same else "다릅니다.")
)
