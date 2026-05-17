import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from model_core import (
    INPUT_INFO,
    create_default_hvp2_model,
    mape_percent,
    relative_error_percent,
)


st.set_page_config(
    page_title="Цифровая модель ХВП-2 — адаптация",
    layout="wide"
)

st.title("Цифровая модель ХВП-2 — интерфейс адаптации")
st.write(
    "Загрузка производственных данных → выбор выходного параметра → расчет по модели Approx → "
    "оценка погрешности → адаптация свободного коэффициента K0 → экспорт результатов."
)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def to_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Аккуратное преобразование числовых столбцов.
    Столбцы, которые не удалось преобразовать в числа, остаются без изменений.
    """
    out = df.copy()

    for col in out.columns:
        original = out[col]

        if original.dtype == object:
            prepared = (
                original.astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace("\u00a0", "", regex=False)
            )
        else:
            prepared = original

        converted = pd.to_numeric(prepared, errors="coerce")

        # Если в столбце реально есть числовые значения — используем преобразованный вариант.
        # Если весь столбец стал NaN, оставляем исходный столбец.
        if converted.notna().sum() > 0:
            out[col] = converted
        else:
            out[col] = original

    return out


def guess_column(alias: str, columns: list) -> str:
    """
    Примерный автоподбор столбца по названию.
    Если не угадал — пользователь вручную выберет нужный столбец.
    """
    low_cols = {c: str(c).lower() for c in columns}

    patterns = {
        "X1_Qfeed": ["q", "feed", "frca", "расход"],
        "X2_Pfeed": ["p_feed", "pressure", "давлен", "prsa"],
        "X3_TafterHX": ["temp", "темп", "afterhx", "trca"],
        "X4_TurbUFin": ["turb", "мутн", "ntu"],
        "X5_CondUOO2in": ["cond", "электроп", "ara"],
        "X6_DoseAS": ["dose", "as", "антискал"],
        "X7_DoseMBS": ["mbs"],
        "X8_DoseNaOCl": ["naocl"],
        "X9_DoseNaOH": ["naoh"],
        "X10_DoseHCl": ["hcl"],
        "X11_UFwashpressure": ["wash", "пром", "uf", "давл"],
        "X12_ROHPspeed": ["speed", "скор", "ro"],
    }

    keys = patterns.get(alias, [])
    for col, lc in low_cols.items():
        if all(k in lc for k in keys[:2]) and keys:
            return col

    for col, lc in low_cols.items():
        if any(k in lc for k in keys):
            return col

    return columns[0]


def make_alias_dataframe(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Создает таблицу с едиными именами X1...X12,
    независимо от того, как столбцы названы в исходном Excel.
    """
    model_df = pd.DataFrame(index=df.index)

    for alias, source_col in col_map.items():
        values = df[source_col]
        if values.dtype == object:
            values = (
                values.astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
        model_df[alias] = pd.to_numeric(values, errors="coerce")

    return model_df


def plot_compare(y_true, y_before, y_after, y_label):
    t = np.arange(len(y_true))

    fig = plt.figure(figsize=(11, 5))
    plt.plot(t, y_true, label="Эксперимент")
    plt.plot(t, y_before, label="Модель до адаптации")
    plt.plot(t, y_after, label="Модель после адаптации")
    plt.title(f"Сравнение экспериментальных и расчетных значений: {y_label}")
    plt.xlabel("Номер точки")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    return fig


def plot_error(y_true, y_before, y_after):
    t = np.arange(len(y_true))

    err_before = (y_before - y_true) / np.maximum(np.abs(y_true), 1e-9) * 100.0
    err_after = (y_after - y_true) / np.maximum(np.abs(y_true), 1e-9) * 100.0

    fig = plt.figure(figsize=(11, 5))
    plt.plot(t, err_before, label="До адаптации")
    plt.plot(t, err_after, label="После адаптации")
    plt.axhline(5, linestyle="--")
    plt.axhline(-5, linestyle="--")
    plt.title("Относительная погрешность, %")
    plt.xlabel("Номер точки")
    plt.ylabel("δ, %")
    plt.grid(True)
    plt.legend()
    return fig


def build_excel_report(summary_df, coef_df, result_df) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        coef_df.to_excel(writer, index=False, sheet_name="Coefficients")
        result_df.to_excel(writer, index=False, sheet_name="Results")
    buf.seek(0)
    return buf.read()


def build_pdf_report(summary: dict) -> bytes:
    pdf_path = Path("hvp2_adaptation_report.pdf")
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h - 50, "HVP-2 Digital Model Adaptation Report")

    c.setFont("Helvetica", 11)
    y = h - 85
    lines = [
        f"Output: {summary.get('Y', '')}",
        f"Title: {summary.get('title', '')}",
        f"Unit: {summary.get('unit', '')}",
        f"K0 before: {summary.get('k0_before', 0):.6f}",
        f"K0 after: {summary.get('k0_after', 0):.6f}",
        f"MAPE before: {summary.get('MAPE_before_percent', 0):.4f} %",
        f"MAPE after: {summary.get('MAPE_after_percent', 0):.4f} %",
        f"Threshold: {summary.get('threshold_percent', 0):.2f} %",
        f"Alpha: {summary.get('alpha', 0):.3f}",
        f"Adapted: {summary.get('adapted', False)}",
    ]

    for line in lines:
        c.drawString(40, y, line)
        y -= 18

    c.showPage()
    c.save()

    return pdf_path.read_bytes()


# -------------------- загрузка файла --------------------

uploaded = st.file_uploader(
    "Загрузи данные для расчета и адаптации модели",
    type=["csv", "xlsx"]
)

if uploaded is None:
    st.info("Загрузи CSV или XLSX файл, чтобы продолжить.")
    st.stop()

df_raw = read_uploaded_file(uploaded)
df_raw = to_numeric_dataframe(df_raw)

st.success(
    f"Файл загружен: {uploaded.name}. "
    f"Строк: {len(df_raw)}, столбцов: {len(df_raw.columns)}."
)

with st.expander("Просмотр исходных данных"):
    st.dataframe(df_raw.head(30), use_container_width=True)


# -------------------- модель --------------------

model = create_default_hvp2_model()
model_names = [m.y_name for m in model.models]
model_titles = {m.y_name: f"{m.y_name} — {m.title}" for m in model.models}

st.sidebar.header("Выбор модели")

selected_y = st.sidebar.selectbox(
    "Выходной параметр",
    model_names,
    format_func=lambda name: model_titles[name]
)

selected_model = model.get_model(selected_y)

st.sidebar.write("**Выбранная модель:**")
st.sidebar.write(selected_model.title)
st.sidebar.write(f"Единица измерения: `{selected_model.unit}`")
st.sidebar.write(f"Постоянная времени T: `{selected_model.tau}` с")


# -------------------- сопоставление X --------------------

st.subheader("1. Сопоставление входных параметров")

st.write(
    "Ниже нужно сопоставить унифицированные параметры модели X1...X12 "
    "со столбцами исходного файла. Если автоподбор ошибся, выбери нужный столбец вручную."
)

columns = df_raw.columns.tolist()
col_map = {}

with st.expander("Настроить соответствие столбцов X1...X12", expanded=True):
    for alias, description in INPUT_INFO.items():
        guess = guess_column(alias, columns)
        default_index = columns.index(guess) if guess in columns else 0

        col_map[alias] = st.selectbox(
            f"{alias} — {description}",
            columns,
            index=default_index,
            key=f"map_{alias}"
        )

model_df = make_alias_dataframe(df_raw, col_map)

if model_df.isna().any().any():
    st.warning(
        "В выбранных входных столбцах есть пропуски или нечисловые значения. "
        "Строки с пропусками будут исключены."
    )

valid_mask = ~model_df.isna().any(axis=1)
model_df = model_df.loc[valid_mask].copy()
df_valid = df_raw.loc[valid_mask].copy()

st.write(f"Для расчета доступно строк после очистки: **{len(model_df)}**")


# -------------------- режим работы --------------------

st.subheader("2. Расчет и адаптация")

mode = st.radio(
    "Режим работы",
    [
        "Только расчет выходных параметров",
        "Оценка погрешности и адаптация K0"
    ]
)

alpha = st.slider(
    "Коэффициент адаптации α",
    min_value=0.05,
    max_value=1.00,
    value=0.20,
    step=0.05
)

threshold = st.slider(
    "Допустимая средняя относительная погрешность, %",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)

y_true_col = None

if mode == "Оценка погрешности и адаптация K0":
    y_true_col = st.selectbox(
        "Столбец с экспериментальным/фактическим значением выбранного Y",
        columns
    )

run = st.button("Запустить расчет")

if not run:
    st.info("Проверь сопоставление входов и нажми «Запустить расчет».")
    st.stop()


# -------------------- расчет --------------------

if mode == "Только расчет выходных параметров":
    y_all = model.predict_all(model_df)

    st.subheader("Результаты расчета Y1...Y18")
    st.dataframe(y_all.head(50), use_container_width=True)

    json_bytes = model.to_json_bytes()

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        model_df.to_excel(writer, index=False, sheet_name="Input_X")
        y_all.to_excel(writer, index=False, sheet_name="Calculated_Y")
    excel_buf.seek(0)

    st.download_button(
        "Скачать JSON модели",
        data=json_bytes,
        file_name="hvp2_model_coefficients.json",
        mime="application/json"
    )

    st.download_button(
        "Скачать XLSX с расчетом",
        data=excel_buf.read(),
        file_name="hvp2_calculated_outputs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    y_true_raw = df_valid[y_true_col]
    if y_true_raw.dtype == object:
        y_true_raw = (
            y_true_raw.astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
    y_true = pd.to_numeric(y_true_raw, errors="coerce").to_numpy(dtype=float)

    mask_y = ~np.isnan(y_true)
    y_true = y_true[mask_y]
    model_df_for_y = model_df.loc[mask_y].copy()

# Добавляем фактический Y во внутреннюю таблицу,
# чтобы функция adapt_one могла его найти.
model_df_for_y["_Y_EXP_"] = y_true

y_before = model.predict_one(model_df_for_y, selected_y)

summary_before = {
    "MAPE_before_percent": mape_percent(y_true, y_before),
    "max_error_before_percent": float(np.max(relative_error_percent(y_true, y_before))),
}

adapt_summary = model.adapt_one(
    model_df_for_y,
    selected_y,
    y_true_col="_Y_EXP_",
    alpha=alpha,
    threshold_percent=threshold
)

y_after = model.predict_one(model_df_for_y, selected_y)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAPE до, %", f"{summary_before['MAPE_before_percent']:.3f}")
c2.metric("MAPE после, %", f"{adapt_summary['MAPE_after_percent']:.3f}")
c3.metric("K0 до", f"{adapt_summary['k0_before']:.6f}")
c4.metric("K0 после", f"{adapt_summary['k0_after']:.6f}")

if adapt_summary["adapted"]:
        st.success("Модель была адаптирована, так как погрешность превышала допустимый порог.")
    else:
        st.info("Адаптация не выполнялась: погрешность не превышает заданный порог.")

    st.subheader("Графики")

    fig1 = plot_compare(
        y_true,
        y_before,
        y_after,
        selected_y
    )
    st.pyplot(fig1, clear_figure=True)

    fig2 = plot_error(
        y_true,
        y_before,
        y_after
    )
    st.pyplot(fig2, clear_figure=True)

    st.subheader("Коэффициенты выбранной модели")

    adapted_model = model.get_model(selected_y)

    coef_df = pd.DataFrame({
        "input": adapted_model.x_vars,
        "coef": adapted_model.coef
    })

    k0_row = pd.DataFrame({
        "input": ["K0"],
        "coef": [adapted_model.k0]
    })

    coef_df = pd.concat([coef_df, k0_row], ignore_index=True)

    st.dataframe(coef_df, use_container_width=True)

    result_df = pd.DataFrame({
        "Y_exp": y_true,
        "Y_calc_before": y_before,
        "Y_calc_after": y_after,
        "error_before_%": relative_error_percent(y_true, y_before),
        "error_after_%": relative_error_percent(y_true, y_after),
    })

    st.subheader("Таблица результатов")
    st.dataframe(result_df.head(100), use_container_width=True)

    summary_df = pd.DataFrame([adapt_summary])

    json_bytes = model.to_json_bytes()
    excel_bytes = build_excel_report(summary_df, coef_df, result_df)
    pdf_bytes = build_pdf_report(adapt_summary)

    st.subheader("Экспорт")

    st.download_button(
        "Скачать JSON с обновленными коэффициентами",
        data=json_bytes,
        file_name="hvp2_adapted_model.json",
        mime="application/json"
    )

    st.download_button(
        "Скачать XLSX отчет",
        data=excel_bytes,
        file_name="hvp2_adaptation_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.download_button(
        "Скачать PDF отчет",
        data=pdf_bytes,
        file_name="hvp2_adaptation_report.pdf",
        mime="application/pdf"
    )
