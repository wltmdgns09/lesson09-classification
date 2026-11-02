import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/wltmdgns09/data/refs/heads/main/stroke.csv"

# 열 이름 -> 화면에 보여 줄 우리말 이름
LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
ALL_FEATURES = list(LABELS)
BINARY = ["hypertension", "heart_disease"]  # 0 또는 1 값만 갖는 속성
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LOGI_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

BLUE_FILL = "rgba(76,120,168,0.2)"  # '아님' 칸 (옅은 파랑)
RED_FILL = "rgba(228,87,86,0.2)"  # '뇌졸중' 칸 (옅은 빨강)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def fmt_value(col, value):
    if col in BINARY:
        return f"{'있음' if value >= 0.5 else '없음'}({value:g})"
    return f"{value:.1f}"


def axis_range(col, X):
    if col in BINARY:
        return -0.25, 1.25
    lo, hi = float(X[col].min()), float(X[col].max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def line_points(A, B, C, x0, x1, y0, y1):
    """직선 A*x + B*y + C = 0 이 그림 상자(x0~x1, y0~y1)와 만나는 점들."""
    pts = []
    tol = 1e-9
    if abs(B) > 1e-12:
        for x in (x0, x1):
            y = -(A * x + C) / B
            if y0 - tol <= y <= y1 + tol:
                pts.append((x, min(max(y, y0), y1)))
    if abs(A) > 1e-12:
        for y in (y0, y1):
            x = -(B * y + C) / A
            if x0 - tol <= x <= x1 + tol:
                pts.append((min(max(x, x0), x1), y))
    unique = []
    for p in pts:
        if all(
            abs(p[0] - q[0]) > 1e-9 * (x1 - x0) or abs(p[1] - q[1]) > 1e-9 * (y1 - y0)
            for q in unique
        ):
            unique.append(p)
    return unique


def build_dot(tree, X_train, y_train, features):
    """의사결정트리를 graphviz DOT 문자열로 바꾼다."""
    t = tree.tree_
    # 마디마다 그 자리에 온 훈련용 사람 수 / 그중 뇌졸중인 사람 수를 직접 센다
    path = tree.decision_path(X_train)
    n_all = np.asarray(path.sum(axis=0)).ravel()
    n_pos = np.asarray(path[y_train == 1].sum(axis=0)).ravel()

    lines = [
        "digraph Tree {",
        'graph [rankdir=TB, bgcolor="white"];',
        'node [shape=box, style="rounded,filled", fontname="sans-serif", fontcolor="black", color="gray40"];',
        'edge [fontname="sans-serif", fontcolor="black", color="gray40"];',
    ]
    for node in range(t.node_count):
        rate = n_pos[node] / n_all[node] * 100
        stats = f"훈련용 {n_all[node]}명\\n뇌졸중 {n_pos[node]}명 ({rate:.1f}%)"
        if t.children_left[node] == -1:  # 더 묻지 않고 답을 내는 마디
            answer = int(np.argmax(t.value[node][0]))
            word = "뇌졸중" if answer == 1 else "아님"
            fill = "#f6c9c9" if answer == 1 else "#cfe0f1"
            label = f"답: {word}\\n{stats}"
        else:
            col = features[t.feature[node]]
            if col in BINARY:
                question = f"{LABELS[col]}이 있나요?"
            else:
                question = f"{LABELS[col]} ≤ {t.threshold[node]:.1f}?"
            fill = "#f2f2f2"
            label = f"{question}\\n{stats}"
        lines.append(f'{node} [label="{label}", fillcolor="{fill}"];')

    for node in range(t.node_count):
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            continue
        col = features[t.feature[node]]
        if col in BINARY:
            left_word, right_word = "아니요", "예"  # 왼쪽 = 0(없음), 오른쪽 = 1(있음)
        else:
            left_word, right_word = "예", "아니요"  # 왼쪽 = 기준값 이하
        lines.append(f'{node} -> {left} [label="{left_word}"];')
        lines.append(f'{node} -> {right} [label="{right_word}"];')
    lines.append("}")
    return "\n".join(lines)


st.title("분류 모델")
st.write("나이, 혈당 같은 속성으로 뇌졸중 여부(stroke=1이면 뇌졸중, 0이면 아님)를 맞히는 모델을 만들어 봐요. 뇌졸중을 양성으로 둬요.")

df = load_data().sort_values("id").reset_index(drop=True)

# ---------------------------------------------------------------
# 입력 속성 고르기
# ---------------------------------------------------------------
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: LABELS[c],
)
features = [c for c in ALL_FEATURES if c in chosen]
if len(features) < 2:
    st.info("입력으로 사용할 속성을 두 개 이상 골라 주세요.")
    st.stop()

# ---------------------------------------------------------------
# 테스트용 / 훈련용 나누기: 번호 순으로 열 명씩 묶어 앞 세 명은 테스트용
# ---------------------------------------------------------------
is_test = (np.arange(len(df)) % 10) < 3
train_df = df[~is_test]
test_df = df[is_test]

X_train = train_df[features].reset_index(drop=True).copy()
X_test = test_df[features].reset_index(drop=True).copy()
y_train = train_df["stroke"].to_numpy()
y_test = test_df["stroke"].to_numpy()

split_note = (
    f"훈련용 {len(X_train):,}명 · 테스트용 {len(X_test):,}명 "
    "(번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요.)"
)
# 체질량지수는 고른 경우에만, 훈련용의 중앙값으로 빈 곳을 채운다
if "bmi" in features:
    bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    split_note += f" 체질량지수의 빈 값은 훈련용 중앙값({bmi_median:.1f})으로 채웠어요."

# 크기 맞추기: 훈련용으로만 기준을 정하고 테스트용에는 그 기준을 그대로 적용
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# 모델 만들기
# ---------------------------------------------------------------
logi = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
# 나무는 크기를 맞추지 않은 값으로 학습해서, 질문의 기준값을 원래 단위로 읽을 수 있게 했어요.
# 마지막 마디에 사람이 다섯 명 미만이 되는 나누기는 하지 않아요(min_samples_leaf=5).
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

acc = {
    LOGI_NAME: (
        accuracy_score(y_train, logi.predict(X_train_s)),
        accuracy_score(y_test, logi.predict(X_test_s)),
    ),
    TREE_NAME: (
        accuracy_score(y_train, tree.predict(X_train)),
        accuracy_score(y_test, tree.predict(X_test)),
    ),
    BASE_NAME: (
        accuracy_score(y_train, base.predict(X_train)),
        accuracy_score(y_test, base.predict(X_test)),
    ),
}

# ---------------------------------------------------------------
# 정확도 카드
# ---------------------------------------------------------------
st.header("정확도")
st.caption(split_note)
card_cols = st.columns(3)
for col, (name, (train_acc, test_acc)) in zip(card_cols, acc.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.metric("테스트 데이터 정확도", f"{test_acc * 100:.2f}%")
            st.caption(f"훈련 데이터 {train_acc * 100:.2f}%  |  테스트 데이터 {test_acc * 100:.2f}%")

# ---------------------------------------------------------------
# 산점도: 두 속성을 골라 테스트 데이터를 찍고, 로지스틱 0.5 선과 트리의 칸을 겹쳐 그림
# ---------------------------------------------------------------
st.header("테스트 데이터 산점도")
sel1, sel2 = st.columns(2)
with sel1:
    x_col = st.selectbox("가로축", features, format_func=lambda c: LABELS[c])
with sel2:
    y_options = [c for c in features if c != x_col]
    y_col = st.selectbox("세로축", y_options, format_func=lambda c: LABELS[c])

others = [c for c in features if c not in (x_col, y_col)]
med = X_test.median()  # 두 축이 아닌 속성을 세워 둘 값 (테스트 데이터의 중앙값)

x0, x1 = axis_range(x_col, X_test)
y0, y1 = axis_range(y_col, X_test)

# 의사결정트리가 나눈 칸: 그림 전체를 촘촘한 격자로 나누어 트리의 답을 구한다
gx = np.linspace(x0, x1, 200)
gy = np.linspace(y0, y1, 200)
XX, YY = np.meshgrid(gx, gy)
grid = pd.DataFrame({c: np.full(XX.size, med[c]) for c in features})
grid[x_col] = XX.ravel()
grid[y_col] = YY.ravel()
Z = tree.predict(grid[features]).reshape(XX.shape)

# 로지스틱 회귀가 확률 0.5로 가르는 선: (크기 맞춘 값의 식) 을 원래 단위의 식으로 풀어 쓴다
idx = {c: i for i, c in enumerate(features)}
w = logi.coef_[0]
b = logi.intercept_[0]
a = w / scaler.scale_
c0 = b - np.sum(w * scaler.mean_ / scaler.scale_)
A = a[idx[x_col]]
B = a[idx[y_col]]
C = c0 + sum(a[idx[c]] * med[c] for c in others)
pts = line_points(A, B, C, x0, x1, y0, y1)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=gx,
        y=gy,
        z=Z,
        zmin=0,
        zmax=1,
        colorscale=[[0, BLUE_FILL], [1, RED_FILL]],
        showscale=False,
        showlegend=False,
        hoverinfo="skip",
    )
)

# 0 또는 1만 갖는 속성은 점이 겹치지 않게 보여 주기만 살짝 흩어 찍는다 (계산에는 영향 없음)
rng = np.random.default_rng(0)


def display_values(col):
    v = X_test[col].to_numpy(dtype=float)
    if col in BINARY:
        v = v + rng.uniform(-0.08, 0.08, size=len(v))
    return v


x_vals = display_values(x_col)
y_vals = display_values(y_col)

for value, name, color, size, opacity in [
    (0, "실제: 아님", "#4c78a8", 6, 0.5),
    (1, "실제: 뇌졸중", "#e45756", 9, 0.95),
]:
    m = y_test == value
    fig.add_trace(
        go.Scatter(
            x=x_vals[m],
            y=y_vals[m],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=opacity, line=dict(width=0.6, color="white")),
        )
    )

if len(pts) >= 2:
    fig.add_trace(
        go.Scatter(
            x=[pts[0][0], pts[-1][0]],
            y=[pts[0][1], pts[-1][1]],
            mode="lines",
            name="로지스틱 회귀 0.5 선",
            line=dict(color="black", width=3, dash="dash"),
        )
    )

# 트리의 칸 색을 범례에도 보여 준다
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        name="트리의 답: 아님 칸",
        marker=dict(size=12, symbol="square", color=BLUE_FILL, line=dict(width=1, color="#4c78a8")),
    )
)
if (Z == 1).any():
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="트리의 답: 뇌졸중 칸",
            marker=dict(size=12, symbol="square", color=RED_FILL, line=dict(width=1, color="#e45756")),
        )
    )

fig.update_xaxes(title_text=LABELS[x_col], range=[x0, x1])
fig.update_yaxes(title_text=LABELS[y_col], range=[y0, y1])
if x_col in BINARY:
    fig.update_xaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
if y_col in BINARY:
    fig.update_yaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_layout(height=560, legend_title_text="")
st.plotly_chart(fig)

if x_col in BINARY or y_col in BINARY:
    st.caption("고혈압·심장병은 0 또는 1이라 점이 겹치지 않도록 보여 주기만 살짝 흩어 찍었어요.")

if others:
    parts = ", ".join(f"{LABELS[c]} {fmt_value(c, med[c])}" for c in others)
    text = f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {parts}."
    if "bmi" in others:
        text += " (체질량지수는 빈 값을 훈련용 중앙값으로 채운 테스트 데이터의 중앙값이에요.)"
    st.write(text)
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둔 값은 없어요.")

if len(pts) >= 2:
    st.write(f"검은 점선은 {LOGI_NAME}의 0.5 선, 즉 뇌졸중일 확률이 0.5가 되는 자리예요.")
else:
    st.write(f"{LOGI_NAME}의 0.5 선(뇌졸중일 확률이 0.5가 되는 자리)은 이 그림 범위 밖에 있어서 그려지지 않았어요.")

region_text = f"옅은 파란 칸은 {TREE_NAME}가 '아님'이라고 답하는 자리"
if (Z == 1).any():
    region_text += ", 옅은 붉은 칸은 '뇌졸중'이라고 답하는 자리예요."
else:
    region_text += "예요. 이 그림 범위에는 '뇌졸중'이라고 답하는 칸이 없어요."
st.write(region_text)

# ---------------------------------------------------------------
# 의사결정트리가 던진 질문 (가지 그림)
# ---------------------------------------------------------------
st.header("의사결정트리(질문으로 답하는 모델)의 질문 가지")
st.graphviz_chart(build_dot(tree, X_train, y_train, features))

t = tree.tree_
is_leaf = t.children_left == -1
leaf_answers = np.argmax(t.value[is_leaf][:, 0, :], axis=1)
n_leaves = int(is_leaf.sum())
n_no = int((leaf_answers == 0).sum())
st.write(f"답을 내는 마디는 모두 {n_leaves}칸이고, 그중 {n_no}칸이 '아님'이라고 답해요.")

used = sorted(set(int(i) for i in t.feature[~is_leaf]))
used_names = ", ".join(LABELS[features[i]] for i in used) if used else "하나도 없어요"
st.write(f"고른 속성 {len(features)}개 가운데 이 나무가 실제로 물은 것: {used_names}")
