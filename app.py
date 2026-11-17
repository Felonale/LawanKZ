"""Streamlit-приложение пятой недели: классификация фрагментов НПА."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lawankz.rk2_week2 import LABEL_NAMES
from lawankz.rk2_week5 import load_control_examples, load_model_bundle, predict_text


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "results" / "rk2_week4" / "selected_model"
EXAMPLES_PATH = PROJECT_ROOT / "results" / "rk2_week5" / "control_examples.json"


st.set_page_config(
    page_title="LawanKZ — классификация НПА",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background: #f5f7fb; }
    .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
    .hero {
        padding: 1.65rem 1.8rem;
        border-radius: 20px;
        color: white;
        background: linear-gradient(120deg, #172554 0%, #1d4ed8 62%, #0891b2 100%);
        box-shadow: 0 12px 35px rgba(23, 37, 84, .18);
        margin-bottom: 1.4rem;
    }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2.05rem; }
    .hero p { margin: 0; color: #dbeafe; font-size: 1.02rem; }
    .result-card {
        padding: 1.25rem 1.35rem;
        border: 1px solid #bfdbfe;
        border-left: 6px solid #2563eb;
        border-radius: 16px;
        background: white;
        box-shadow: 0 8px 24px rgba(30, 64, 175, .08);
    }
    .result-label { color: #64748b; font-size: .86rem; text-transform: uppercase; letter-spacing: .07em; }
    .result-value { color: #172554; font-size: 1.55rem; font-weight: 750; margin-top: .2rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; padding: .85rem 1rem; border-radius: 14px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.65rem; }
    .prob-row { margin: .72rem 0; }
    .prob-meta { display: flex; justify-content: space-between; gap: 1rem; color: #334155; font-size: .88rem; }
    .prob-track { height: 10px; margin-top: .28rem; border-radius: 999px; background: #dbeafe; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2563eb, #0891b2); }
    div.stButton > button { border-radius: 12px; font-weight: 650; min-height: 2.8rem; }
    [data-testid="stSidebar"] { background: #eef2ff; }
    [data-testid="stToolbar"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_model_bundle(MODEL_DIR)


def show_result(result: dict, expected_label: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Предсказанный нормативный акт</div>
            <div class="result-value">{result['predicted_label_ru']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Уверенность", f"{result['confidence']:.1%}")
    metric_middle.metric("Уровень", result["confidence_level"].capitalize())
    metric_right.metric("Фрагменты", result["chunk_count"])
    if expected_label:
        if result["predicted_label"] == expected_label:
            st.success("Контрольный ответ совпал с известным источником статьи.", icon="✅")
        else:
            st.warning(
                "Предсказание не совпало с источником статьи. Это полезный пример ошибки модели.",
                icon="⚠️",
            )
    st.markdown("#### Вероятности по всем классам")
    probability_frame = pd.DataFrame(result["probabilities"])[
        ["label_ru", "probability"]
    ].rename(columns={"label_ru": "Класс", "probability": "Вероятность"})
    bars = "".join(
        f'<div class="prob-row">'
        f'<div class="prob-meta"><span>{row["Класс"]}</span>'
        f'<strong>{row["Вероятность"]:.1%}</strong></div>'
        f'<div class="prob-track"><div class="prob-fill" '
        f'style="width:{row["Вероятность"] * 100:.2f}%"></div></div>'
        f'</div>'
        for _, row in probability_frame.iterrows()
    )
    st.markdown(f'<div data-testid="probability-bars">{bars}</div>', unsafe_allow_html=True)
    with st.expander("Показать точные значения"):
        table = probability_frame.copy()
        table["Вероятность"] = table["Вероятность"].map(lambda value: f"{value:.2%}")
        st.dataframe(table, hide_index=True, width="stretch")


examples = load_control_examples(EXAMPLES_PATH)
query_sample = st.query_params.get("sample", "1")
try:
    initial_index = min(max(int(query_sample) - 1, 0), max(len(examples) - 1, 0))
except ValueError:
    initial_index = 0

if examples and "analysis_text" not in st.session_state:
    st.session_state.analysis_text = examples[initial_index]["text"]
if "control_choice" not in st.session_state:
    st.session_state.control_choice = initial_index


def update_example_text() -> None:
    if examples:
        st.session_state.analysis_text = examples[st.session_state.control_choice]["text"]


with st.sidebar:
    st.markdown("## ⚖️ LawanKZ")
    st.caption("Учебный NLP-прототип · 5-я неделя")
    st.divider()
    st.markdown("**Модель**")
    st.code("fine-tuned RuBERT", language=None)
    st.markdown("**Поддерживаемые классы**")
    for label in LABEL_NAMES.values():
        st.markdown(f"- {label}")
    st.divider()
    st.caption(
        "Результат носит исследовательский характер и не является юридической консультацией."
    )

st.markdown(
    """
<div class="hero">
    <h1>Анализ текста нормативного акта</h1>
    <p>Вставьте фрагмент закона — модель определит наиболее вероятный исходный документ.</p>
</div>
""",
    unsafe_allow_html=True,
)

input_column, result_column = st.columns([1.08, 0.92], gap="large")

with input_column:
    st.markdown("### 1. Введите текст")
    if examples:
        st.selectbox(
            "Контрольный пример",
            options=range(len(examples)),
            format_func=lambda index: (
                f"Тест {examples[index]['sample_id']} · {examples[index]['expected_label_ru']}"
            ),
            key="control_choice",
            on_change=update_example_text,
            help="Все примеры взяты из отложенной test-выборки и не участвовали в обучении.",
        )
    text = st.text_area(
        "Текст статьи или фрагмента",
        key="analysis_text",
        height=330,
        placeholder="Например: оператор обязан обеспечить защиту персональных данных...",
        label_visibility="collapsed",
    )
    word_count = len(text.split()) if text else 0
    st.caption(f"Слов: {word_count}. Длинный текст автоматически делится на фрагменты.")
    analyze_clicked = st.button(
        "Анализировать текст",
        type="primary",
        width="stretch",
        icon="🔎",
    )

autorun = st.query_params.get("autorun", "0") == "1"
should_run = analyze_clicked or (autorun and not st.session_state.get("autorun_done", False))

with result_column:
    st.markdown("### 2. Результат")
    if should_run:
        try:
            with st.spinner("RuBERT читает текст и сравнивает шесть классов..."):
                tokenizer, model = get_model_bundle()
                result = predict_text(text, tokenizer, model)
            st.session_state.last_result = result
            st.session_state.autorun_done = True
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error), icon="🚫")
    if "last_result" in st.session_state:
        expected = None
        if examples:
            expected = examples[st.session_state.control_choice]["expected_label"]
        show_result(st.session_state.last_result, expected)
    else:
        st.info(
            "После запуска здесь появятся класс, уверенность и диаграмма всех вероятностей.",
            icon="💡",
        )

st.divider()
with st.expander("Как это работает?"):
    st.markdown(
        """
1. Текст делится на фрагменты, чтобы не превысить ограничение BERT.
2. Токенизатор превращает слова в числовые идентификаторы.
3. Fine-tuned RuBERT выдаёт шесть наборов оценок — по одному на каждый класс.
4. Для длинного текста оценки фрагментов усредняются.
5. Softmax превращает оценки в вероятности, сумма которых равна 100%.

Низкая уверенность означает, что несколько документов используют похожую лексику.
"""
    )
