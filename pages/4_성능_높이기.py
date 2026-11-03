import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/wltmdgns09/data/refs/heads/main/stroke.csv"
NO_VALUE = "계산할 수 없음"

LOGIT_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BEFORE = "가중치 맞추기 전"
AFTER = "가중치 맞춘 뒤"

# 화면에 보이는 이름 -> 데이터의 열 이름
FEATURES = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease",
}
KOREAN = {col: name for name, col in FEATURES.items()}  # 열 이름 -> 화면에 보이는 이름
DEFAULT_FEATURES = ["나이", "평균 혈당", "고혈압", "심장병"]

COLOR_NO = "#7F8C8D"  # 실제 뇌졸중 아님
COLOR_YES = "#E74C3C"  # 실제 뇌졸중
COLOR_BEFORE = "#2E86DE"  # 가중치 맞추기 전 경계
COLOR_AFTER = "#27AE60"  # 가중치 맞춘 뒤 경계

st.set_page_config(page_title="성능 높이기", layout="wide")
st.title("성능 높이기")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 맨 위: 입력 속성 고르기 ----------
picked = st.multiselect("입력 속성", list(FEATURES.keys()), default=DEFAULT_FEATURES)
if not picked:
    st.warning("입력 속성을 하나 이상 골라 주세요.")
    st.stop()
cols = [FEATURES[name] for name in picked]

st.caption(
    "뇌졸중을 겪은 사람이 아주 드물어서, 모델이 '아니다'만 외치기 쉽습니다. "
    "'가중치를 맞춘다'는 것은 훈련용에서 뇌졸중인 사람과 아닌 사람이 학습에 미치는 힘을 같게 만든다는 뜻입니다. "
    "뇌졸중인 사람 한 명을 틀렸을 때 더 크게 벌점을 줍니다."
)

# ---------- 읽기와 나누기 ('채점표'와 같은 방법) ----------
df = load_data().sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3  # 번호순 열 명 묶음의 앞 세 명은 테스트용
train_df = df[~is_test].copy()
test_df = df[is_test].copy()

# 체질량지수의 빈 값은 훈련용의 중앙값으로 채움
if "bmi" in cols:
    median_bmi = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(median_bmi)
    test_df["bmi"] = test_df["bmi"].fillna(median_bmi)

X_train, y_train = train_df[cols], train_df["stroke"]
X_test, y_test = test_df[cols], test_df["stroke"]
medians = X_train.median()  # 그림에서 축이 아닌 속성을 고정할 값

# ---------- 모델 학습: 가중치 맞추기 전·후 ----------
models = {
    LOGIT_NAME: {
        BEFORE: LogisticRegression(max_iter=1000),
        AFTER: LogisticRegression(max_iter=1000, class_weight="balanced"),
    },
    TREE_NAME: {
        BEFORE: DecisionTreeClassifier(max_depth=3, random_state=42),
        AFTER: DecisionTreeClassifier(max_depth=3, random_state=42, class_weight="balanced"),
    },
}
for versions in models.values():
    for model in versions.values():
        model.fit(X_train, y_train)


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


cells = {}  # cells[모델 이름][전/후] = (tp, fp, fn, tn)
for name, versions in models.items():
    cells[name] = {}
    for version, model in versions.items():
        pred = pd.Series(model.predict(X_test), index=test_df.index)
        cells[name][version] = count_cells(y_test, pred)

# ---------- 전·후 비교표 ----------
st.header("가중치를 맞추기 전과 맞춘 뒤")
n_test = len(test_df)
n_pos = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 중 뇌졸중인 사람은 {n_pos:,}명입니다.")

row_labels = [
    "TP (찾아낸 환자)",
    "FN (놓친 환자)",
    "FP (헛짚은 사람)",
    "TN (아닌 사람을 아니라고 맞힘)",
    "정확도",
    "재현율",
    "정밀도",
    "F1",
]
table_data = {}
for name in (LOGIT_NAME, TREE_NAME):
    for version in (BEFORE, AFTER):
        tp, fp, fn, tn = cells[name][version]
        m = four_metrics(tp, fp, fn, tn)
        table_data[(name, version)] = [
            f"{tp:,}명",
            f"{fn:,}명",
            f"{fp:,}명",
            f"{tn:,}명",
            m["정확도"],
            m["재현율"],
            m["정밀도"],
            m["F1"],
        ]
compare = pd.DataFrame(table_data, index=row_labels)
st.table(compare)

parts = []
for name in (LOGIT_NAME, TREE_NAME):
    tp0, fp0, fn0, _ = cells[name][BEFORE]
    tp1, fp1, fn1, _ = cells[name][AFTER]
    parts.append(
        f"{name}: 찾아낸 환자 {tp0:,}명 → {tp1:,}명, "
        f"놓친 환자 {fn0:,}명 → {fn1:,}명, "
        f"헛짚은 사람 {fp0:,}명 → {fp1:,}명"
    )
st.write(" ｜ ".join(parts))
st.caption(
    "재현율 = TP ÷ (TP + FN), 분모는 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP ÷ (TP + FP), 분모는 뇌졸중이라 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율) · "
    "정확도 = (TP + TN) ÷ 전체 사람 수"
)


# ---------- 그림 준비 ----------
def span(series, pad_ratio=0.05):
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * pad_ratio or 0.5
    return lo - pad, hi + pad


def linspace(lo, hi, n):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def stroke_proba(model, points):
    """points: {열 이름: 값 목록}. 들어 있지 않은 속성은 훈련용 중앙값으로 고정한다."""
    n = len(next(iter(points.values())))
    data = {c: (points[c] if c in points else [float(medians[c])] * n) for c in cols}
    frame = pd.DataFrame(data)[cols]
    return model.predict_proba(frame)[:, 1]


def side_of_half(probs):
    """0.5 경계가 그 범위 안에 있는지 확인한다. 없으면 어느 쪽인지 글로 돌려준다."""
    if probs.max() <= 0.5:
        return "이 범위 안에서는 뇌졸중일 확률이 모두 0.5 이하입니다(모두 '아니다'로 답하는 자리)."
    if probs.min() >= 0.5:
        return "이 범위 안에서는 뇌졸중일 확률이 모두 0.5 이상입니다(모두 '뇌졸중이다'로 답하는 자리)."
    return None


st.header("0.5 경계선 겹쳐 보기")
shown_model_name = st.radio(
    "그림으로 볼 모델", [LOGIT_NAME, TREE_NAME], horizontal=True
)
shown = models[shown_model_name]
st.caption(
    "경계선은 뇌졸중일 확률이 0.5가 되는 곳입니다. "
    "그림의 축으로 고르지 않은 속성은 훈련용의 중앙값으로 고정해서 그립니다. "
    "점은 테스트용 사람입니다."
)

line_colors = {BEFORE: COLOR_BEFORE, AFTER: COLOR_AFTER}

# ---------- 2차원 산점도 ----------
st.subheader("2차원 산점도")
if len(picked) < 2:
    st.info("속성을 둘 이상 고르면 2차원 산점도가 나타납니다.")
else:
    c1, c2 = st.columns(2)
    with c1:
        x_name = st.selectbox("가로축", picked, index=0, key="x2")
    with c2:
        y_options = [p for p in picked if p != x_name]
        y_name = st.selectbox("세로축", y_options, index=0, key="y2")
    x_col, y_col = FEATURES[x_name], FEATURES[y_name]

    x_lo, x_hi = span(X_test[x_col])
    y_lo, y_hi = span(X_test[y_col])
    xs = linspace(x_lo, x_hi, 100)
    ys = linspace(y_lo, y_hi, 100)
    gx = [xv for _ in ys for xv in xs]
    gy = [yv for yv in ys for _ in xs]

    fig2 = go.Figure()
    for label, flag, color in (("실제 뇌졸중 아님", 0, COLOR_NO), ("실제 뇌졸중", 1, COLOR_YES)):
        part = test_df[test_df["stroke"] == flag]
        fig2.add_trace(
            go.Scatter(
                x=part[x_col],
                y=part[y_col],
                mode="markers",
                name=label,
                marker=dict(color=color, size=7 if flag == 1 else 5, opacity=0.8 if flag == 1 else 0.45),
            )
        )

    outside_notes = []
    for version in (BEFORE, AFTER):
        probs = stroke_proba(shown[version], {x_col: gx, y_col: gy})
        note = side_of_half(probs)
        if note is not None:
            outside_notes.append(f"{version}의 0.5 경계선은 이 그림 밖에 있습니다. {note}")
            continue
        z = [probs[i * len(xs):(i + 1) * len(xs)].tolist() for i in range(len(ys))]
        color = line_colors[version]
        fig2.add_trace(
            go.Contour(
                x=xs,
                y=ys,
                z=z,
                name=f"{version} 0.5 경계선",
                showlegend=True,
                showscale=False,
                contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                colorscale=[[0, color], [1, color]],
                line=dict(width=3),
                hoverinfo="skip",
            )
        )

    fig2.update_layout(
        xaxis_title=x_name,
        yaxis_title=y_name,
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig2.update_xaxes(range=[x_lo, x_hi])
    fig2.update_yaxes(range=[y_lo, y_hi])
    st.plotly_chart(fig2, use_container_width=True)
    for note in outside_notes:
        st.write(note)

# ---------- 3차원 산점도 ----------
if len(picked) >= 3:
    st.subheader("3차원 산점도")
    d1, d2, d3 = st.columns(3)
    with d1:
        x3_name = st.selectbox("가로축 (x)", picked, index=0, key="x3")
    with d2:
        y3_options = [p for p in picked if p != x3_name]
        y3_name = st.selectbox("세로축 (y)", y3_options, index=0, key="y3")
    with d3:
        z3_options = [p for p in picked if p not in (x3_name, y3_name)]
        z3_name = st.selectbox("깊이축 (z)", z3_options, index=0, key="z3")
    x3, y3, z3 = FEATURES[x3_name], FEATURES[y3_name], FEATURES[z3_name]

    n = 18
    xs3 = linspace(*span(X_test[x3]), n)
    ys3 = linspace(*span(X_test[y3]), n)
    zs3 = linspace(*span(X_test[z3]), n)
    gx3 = [a for a in xs3 for _ in ys3 for _ in zs3]
    gy3 = [b for _ in xs3 for b in ys3 for _ in zs3]
    gz3 = [c for _ in xs3 for _ in ys3 for c in zs3]

    fig3 = go.Figure()
    for label, flag, color in (("실제 뇌졸중 아님", 0, COLOR_NO), ("실제 뇌졸중", 1, COLOR_YES)):
        part = test_df[test_df["stroke"] == flag]
        fig3.add_trace(
            go.Scatter3d(
                x=part[x3],
                y=part[y3],
                z=part[z3],
                mode="markers",
                name=label,
                marker=dict(color=color, size=4 if flag == 1 else 2, opacity=0.9 if flag == 1 else 0.35),
            )
        )

    outside_notes3 = []
    for version in (BEFORE, AFTER):
        probs = stroke_proba(shown[version], {x3: gx3, y3: gy3, z3: gz3})
        note = side_of_half(probs)
        if note is not None:
            outside_notes3.append(f"{version}의 0.5 평면은 이 그림 밖에 있습니다. {note}")
            continue
        color = line_colors[version]
        fig3.add_trace(
            go.Isosurface(
                x=gx3,
                y=gy3,
                z=gz3,
                value=probs,
                isomin=0.5,
                isomax=0.5,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=0.45,
                name=f"{version} 0.5 평면",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig3.update_layout(
        scene=dict(xaxis_title=x3_name, yaxis_title=y3_name, zaxis_title=z3_name),
        height=680,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("마우스로 끌면 돌려 볼 수 있고, 휠로 확대·축소할 수 있습니다.")
    for note in outside_notes3:
        st.write(note)
