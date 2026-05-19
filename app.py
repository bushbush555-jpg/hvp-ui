import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from model_core import (
    INPUT_INFO,
    INPUT_LIMITS,
    create_default_hvp2_model,
    mean_relative_error_percent,
    relative_error_percent,
    signed_relative_error_percent,
)


st.set_page_config(
    page_title="Цифровая модель ХВП-2 — адаптация",
    layout="wide"
)

st.title("Цифровая модель ХВП-2 — интерфейс адаптации")
st.write(
    "Загрузка производственных данных → выбор выходного параметра → расчет по модели Approx → "
    "оценка относительной погрешности → при необходимости адаптация свободного коэффициента K0 → "
    "экспорт результатов."
)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def to_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Аккуратное преобразование числовых столбцов.

    Если столбец содержит числа в формате с запятой, они преобразуются в float.
    Если столбец полностью текстовый, он остается без изменений.
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

        if converted.notna().sum() > 0:
            out[col] = converted
        else:
            out[col] = original

    return out


def guess_column(alias: str, columns: list) -> str:
    """
    Примерный автоподбор столбца по названию.
    Если автоподбор ошибся, пользователь вручную выбирает нужный столбец.
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
        if keys and all(k in lc for k in keys[:2]):
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
                .str.replace("\u00a0", "", regex=False)
            )

        model_df[alias] = pd.to_numeric(values, errors="coerce")

    return model_df


def validate_input_ranges(model_df: pd.DataFrame, limits: dict):
    """
    Проверка входных параметров X1...X12 на выход за допустимые диапазоны.
    Возвращает маску корректных строк и таблицу нарушений.
    """
    valid_mask = pd.Series(True, index=model_df.index)
    report_rows = []

    for alias, lim in limits.items():
        if alias not in model_df.columns:
            continue

        values = model_df[alias]
        bad = (values < lim["min"]) | (values > lim["max"])

        if bad.any():
            valid_mask = valid_mask & (~bad)

            bad_indices = model_df.index[bad].astype(str).tolist()
            bad_values = values[bad].head(5).tolist()

            report_rows.append({
                "Параметр": alias,
                "Описание": lim["name"],
                "Допустимый минимум": lim["min"],
                "Допустимый максимум": lim["max"],
                "Ед. изм.": lim["unit"],
                "Количество нарушений": int(bad.sum()),
                "Примеры строк": ", ".join(bad_indices[:10]),
                "Примеры значений": ", ".join([str(v) for v in bad_values]),
            })

    return valid_mask, pd.DataFrame(report_rows)


def plot_compare(y_true, y_before, y_after, y_label: str, adapted: bool):
    """
    График сравнения экспериментальных и расчетных значений.

    Если адаптация не выполнялась, отображается только эксперимент и расчетная модель.
    Если адаптация выполнялась, отображается эксперимент, расчет до адаптации и расчет после адаптации.
    """
    t = np.arange(len(y_true))

    fig = plt.figure(figsize=(11, 5))

    plt.plot(
        t,
        y_true,
        label="Экспериментальные значения",
        linestyle="-",
        marker="o",
        markersize=3.5,
        linewidth=1.3
    )

    if adapted:
        plt.plot(
            t,
            y_before,
            label="Расчет до адаптации",
            linestyle="--",
            linewidth=1.7
        )

        plt.plot(
            t,
            y_after,
            label="Расчет после адаптации",
            linestyle="-",
            linewidth=1.8
        )
    else:
        plt.plot(
            t,
            y_before,
            label="Расчетные значения модели",
            linestyle="-",
            linewidth=1.8
        )

    plt.title(f"Сравнение экспериментальных и расчетных значений: {y_label}")
    plt.xlabel("Номер точки")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(frameon=True)

    return fig


def plot_error(y_true, y_before, y_after, threshold: float, adapted: bool):
    """
    График знаковой относительной погрешности.

    Если адаптация не выполнялась, отображается одна линия погрешности расчетной модели.
    Если адаптация выполнялась, отображаются две линии: до и после адаптации.
    """
    t = np.arange(len(y_true))

    err_before = signed_relative_error_percent(y_true, y_before)
    err_after = signed_relative_error_percent(y_true, y_after)

    fig = plt.figure(figsize=(11, 5))

    if adapted:
        plt.plot(
            t,
            err_before,
            label="Погрешность до адаптации",
            linestyle="--",
            linewidth=1.6
        )

        plt.plot(
            t,
            err_after,
            label="Погрешность после адаптации",
            linestyle="-",
            linewidth=1.8
        )
    else:
        plt.plot(
            t,
            err_before,
            label="Относительная погрешность расчетной модели",
            linestyle="-",
            linewidth=1.8
        )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        label=f"+{threshold:.1f} %"
    )

    plt.axhline(
        -threshold,
        linestyle="--",
        linewidth=1.2,
        label=f"-{threshold:.1f} %"
    )

    plt.title("Относительная погрешность, %")
    plt.xlabel("Номер точки")
    plt.ylabel("δ, %")
    plt.grid(True)
    plt.legend(frameon=True)

    return fig


def build_excel_report(summary_df: pd.DataFrame, coef_df: pd.DataFrame, result_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        coef_df.to_excel(writer, index=False, sheet_name="Coefficients")
        result_df.to_excel(writer, index=False, sheet_name="Results")

    buf.seek(0)
    return buf.read()


def build_pdf_report(summary: dict) -> bytes:
    """
    PDF-отчет выполнен латиницей, чтобы в ReportLab не возникало проблем с кириллицей.
    """
    pdf_path = Path("hvp2_adaptation_report.pdf")
    c = canvas.Canvas(str(pdf_path), pagesize=A4)

    _, h = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h - 50, "HVP-2 Digital Model Adaptation Report")

    c.setFont("Helvetica", 11)

    y = h - 85

    lines = [
        f"Output: {summary.get('Y', '')}",
        f"Title: {summary.get('title', '')}",
        f"Unit: {summary.get('unit', '')}",
        f"Tau: {summary.get('tau', 0)} s",
        f"K0 before: {summary.get('k0_before', 0):.6f}",
        f"K0 after: {summary.get('k0_after', 0):.6f}",
        f"Mean relative error before: {summary.get('mean_error_before_percent', 0):.4f} %",
        f"Mean relative error after: {summary.get('mean_error_after_percent', 0):.4f} %",
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

range_valid_mask, range_report = validate_input_ranges(model_df, INPUT_LIMITS)

if not range_report.empty:
    st.warning(
        "Обнаружены значения входных параметров, выходящие за допустимые диапазоны. "
        "Такие строки могут привести к некорректному расчету модели."
    )

    st.dataframe(range_report, use_container_width=True)

    range_action = st.radio(
        "Что сделать со строками вне допустимых диапазонов?",
        [
            "Исключить некорректные строки из расчета",
            "Остановить расчет и исправить исходные данные"
        ]
    )

    if range_action == "Остановить расчет и исправить исходные данные":
        st.stop()

    model_df = model_df.loc[range_valid_mask].copy()
    df_valid = df_valid.loc[range_valid_mask].copy()

if len(model_df) == 0:
    st.error(
        "После очистки данных не осталось строк для расчета. "
        "Проверь выбранные столбцы и диапазоны входных параметров."
    )
    st.stop()

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
            .str.replace("\u00a0", "", regex=False)
        )

    y_true_series = pd.to_numeric(y_true_raw, errors="coerce")

    if y_true_series.isna().any():
        st.warning(
            "В столбце с фактическим значением Y есть пропуски или нечисловые значения. "
            "Такие строки будут исключены из проверки."
        )

    mask_y = ~y_true_series.isna()

    y_true = y_true_series.loc[mask_y].to_numpy(dtype=float)
    model_df_for_y = model_df.loc[mask_y].copy()

    if len(model_df_for_y) == 0:
        st.error(
            "После очистки фактических значений Y не осталось строк для проверки модели."
        )
        st.stop()

    model_df_for_y["_Y_EXP_"] = y_true

    y_before = model.predict_one(model_df_for_y, selected_y)

    mean_error_before = mean_relative_error_percent(y_true, y_before)

    adapt_summary = model.adapt_one(
        model_df_for_y,
        selected_y,
        y_true_col="_Y_EXP_",
        alpha=alpha,
        threshold_percent=threshold
    )

    y_after = model.predict_one(model_df_for_y, selected_y)

    mean_error_after = adapt_summary["mean_error_after_percent"]
    adapted = adapt_summary["adapted"]

    if adapted:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("δср до, %", f"{mean_error_before:.3f}")
        c2.metric("δср после, %", f"{mean_error_after:.3f}")
        c3.metric("K0 до", f"{adapt_summary['k0_before']:.6f}")
        c4.metric("K0 после", f"{adapt_summary['k0_after']:.6f}")

        if mean_error_after <= threshold:
            st.success(
                "Модель была адаптирована. После корректировки средняя относительная "
                "погрешность находится в допустимых пределах."
            )
        else:
            st.warning(
                "Модель была адаптирована, однако средняя относительная погрешность "
                "все еще превышает допустимый порог. Для повышения точности требуется "
                "дальнейшее уточнение коэффициентов модели, а не только корректировка K0."
            )

    else:
        c1, c2, c3 = st.columns(3)

        c1.metric("δср, %", f"{mean_error_before:.3f}")
        c2.metric("Допустимый порог, %", f"{threshold:.3f}")
        c3.metric("K0", f"{adapt_summary['k0_before']:.6f}")

        st.info(
            "Адаптация не выполнялась: средняя относительная погрешность не превышает "
            "заданный допустимый порог."
        )

    st.subheader("Графики")

    y_label = f"{selected_y}, {selected_model.unit}"

    fig1 = plot_compare(
        y_true=y_true,
        y_before=y_before,
        y_after=y_after,
        y_label=y_label,
        adapted=adapted
    )

    st.pyplot(fig1, clear_figure=True)

    fig2 = plot_error(
        y_true=y_true,
        y_before=y_before,
        y_after=y_after,
        threshold=threshold,
        adapted=adapted
    )

    st.pyplot(fig2, clear_figure=True)

    st.subheader("Коэффициенты выбранной модели")

    adapted_model = model.get_model(selected_y)

    coef_df = pd.DataFrame({
        "input": adapted_model.x_vars,
        "coef": adapted_model.coef
    })

    k0_row = pd.DataFrame({
        "input": ["K0 — свободный коэффициент"],
        "coef": [adapted_model.k0]
    })

    coef_df = pd.concat([coef_df, k0_row], ignore_index=True)

    st.dataframe(coef_df, use_container_width=True)

    if adapted:
        result_df = pd.DataFrame({
            "Y_exp": y_true,
            "Y_calc_before": y_before,
            "Y_calc_after": y_after,
            "rel_error_before_%": relative_error_percent(y_true, y_before),
            "rel_error_after_%": relative_error_percent(y_true, y_after),
        })
    else:
        result_df = pd.DataFrame({
            "Y_exp": y_true,
            "Y_calc": y_before,
            "relative_error_%": relative_error_percent(y_true, y_before),
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
