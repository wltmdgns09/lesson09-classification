import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/wltmdgns09/data/refs/heads/main/stroke.csv"

RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "기록.json"

# 브라우저 탭 제목과 아이콘 (반드시 가장 먼저 호출)
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# ---------- 내 기록 불러오기 / 저장하기 ----------
def load_records():
    try:
        data = json.loads(RECORD_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    return {
        "meanings": data.get("meanings", {}),
        "source": data.get("source", ""),
    }


def save_records(records):
    RECORD_DIR.mkdir(exist_ok=True)
    RECORD_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------- 데이터 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(series):
    """열에 어떤 값들이 들어 있는지 한 줄로 설명한다."""
    values = series.dropna().unique()
    if len(values) <= 10:
        return ", ".join(str(v) for v in sorted(values, key=str))
    if pd.api.types.is_numeric_dtype(series):
        return f"숫자 ({series.min():g} ~ {series.max():g})"
    return f"{len(values)}가지"


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

records = load_records()

if "saved_msg" in st.session_state:
    st.toast(st.session_state.pop("saved_msg"), icon="✅")

# ---------- 화면 맨 위: 제목 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.write(
    "이 데이터는 사람마다 성별, 나이, 건강 상태, 생활 습관 등을 적어 놓은 표예요. "
    "맨 마지막 `stroke` 열은 뇌졸중 여부를 나타내고, 1이면 뇌졸중이 있었던 사람이에요."
)

# ---------- 큰 숫자 카드 4개 ----------
st.markdown(
    """
    <style>
    .card {
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 16px;
        padding: 24px 12px;
        text-align: center;
        background: rgba(128, 128, 128, 0.08);
    }
    .card .num { font-size: 2.8rem; font-weight: 800; line-height: 1.2; }
    .card .label { font-size: 1rem; opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)

total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

cards = [
    ("전체 사람 수", f"{total:,}명"),
    ("열 개수", f"{n_cols}개"),
    ("stroke가 1인 사람 수", f"{n_stroke:,}명"),
    ("stroke가 1인 비율", f"{ratio:.2f}%"),
]
for col, (label, number) in zip(st.columns(4), cards):
    col.markdown(
        f'<div class="card"><div class="num">{number}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )

# ---------- 열 정보 표 ----------
st.subheader("열 정보")
st.caption("‘우리말 뜻’ 칸을 눌러 교재를 보고 직접 적은 뒤, 아래 저장 버튼을 눌러 주세요.")

info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [records["meanings"].get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

with st.form("meaning_form"):
    edited = st.data_editor(
        info,
        hide_index=True,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻", help="교재를 보고 직접 적어 주세요."
            ),
        },
    )
    meaning_saved = st.form_submit_button("💾 우리말 뜻 저장")

if meaning_saved:
    records["meanings"] = {
        row["열 이름"]: (row["우리말 뜻"] or "").strip()
        for _, row in edited.iterrows()
    }
    save_records(records)
    st.session_state["saved_msg"] = "우리말 뜻을 저장했어요."
    st.rerun()

# ---------- 처음 다섯 줄 ----------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), hide_index=True)

# ---------- 데이터 출처 ----------
st.subheader("데이터 출처")
with st.form("source_form"):
    source_text = st.text_area(
        "출처",
        value=records["source"],
        placeholder="교재에 나온 데이터 출처를 여기에 적어 주세요.",
        height=120,
        label_visibility="collapsed",
    )
    source_saved = st.form_submit_button("💾 출처 저장")

if source_saved:
    records["source"] = source_text.strip()
    save_records(records)
    st.session_state["saved_msg"] = "데이터 출처를 저장했어요."
    st.rerun()
