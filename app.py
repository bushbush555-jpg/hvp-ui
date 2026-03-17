import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from model_core import fit_polynomial, export_json_bytes, export_py_bytes


st.set_page_config(page_title="Цифровая модель ХВП-2 — интерфейс адаптации", layout="wide")

st.title("Цифровая модель ХВП-2 — интерфейс адаптации математической модели")
st.write("Загрузка данных → выбор X/Y → адаптация коэффициентов полинома → оценка качества → экспорт модели и отчёта.")

# -------- Upload --------
uploaded = st.file_uploader("Загрузи данные (CSV или XLSX)", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Загрузи файл, чтобы продолжить.")
    st.stop()

if uploaded.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

st.success(f"Файл загружен: {uploaded.name} | строк: {len(df)} | столбцов: {len(df.columns)}")
st.dataframe(df.head(15), use_container_width=True)

cols = df.columns.tolist()

st.sidebar.header("Настройки модели")
degree = st.sidebar.selectbox("Степень полинома", [1, 2], index=1)
y_col = st.sidebar.selectbox("Выход Y", cols)

default_x = []
for c in cols:
    cl = str(c).lower()
    if c == y_col:
        continue
    if "№" in cl or "номер" in cl or "опыт" in cl:
        continue
    if any(k in cl for k in ["q", "p", "t_", "turb", "cond", "dose", "uf", "ro", "speed", "int"]):
        default_x.append(c)
default_x = default_x[:12] if len(default_x) >= 2 else cols[:5]

x_cols = st.sidebar.multiselect("Входы X (выбери 8–12 параметров)", cols, default=default_x)

if len(x_cols) < 2:
    st.warning("Выбери минимум 2 входа X.")
    st.stop()

# -------- Run --------
st.subheader("Адаптация модели")
run = st.button("Запустить адаптацию")

if not run:
    st.info("Выбери X/Y и нажми «Запустить адаптацию».")
    st.stop()

report, y_true, y_pred = fit_polynomial(df, x_cols, y_col, degree=degree)

c1, c2, c3 = st.columns(3)
c1.metric("R²", f"{report['r2']:.4f}")
c2.metric("MAPE, %", f"{report['mape']:.2f}")
c3.metric("Строк данных", f"{report['n_rows']}")

st.subheader("Графики")
t = np.arange(len(y_true))

fig1 = plt.figure()
plt.plot(t, y_true, label="y (измер.)")
plt.plot(t, y_pred, label="ŷ (модель)")
plt.title("Сравнение y и ŷ")
plt.xlabel("Номер точки")
plt.ylabel(y_col)
plt.legend()
plt.grid(True)
st.pyplot(fig1, clear_figure=True)

rel_err = (y_pred - y_true) / np.maximum(np.abs(y_true), 1e-9) * 100.0
fig2 = plt.figure()
plt.plot(t, rel_err)
plt.axhline(5, linestyle="--")
plt.axhline(-5, linestyle="--")
plt.title("Относительная погрешность ε, %")
plt.xlabel("Номер точки")
plt.ylabel("ε, %")
plt.grid(True)
st.pyplot(fig2, clear_figure=True)

st.subheader("Коэффициенты")
coef_df = pd.DataFrame({"term": report["feature_names"], "coef": report["coef"]})
st.dataframe(coef_df, use_container_width=True)

# -------- Exports --------
st.subheader("Экспорт")

json_bytes = export_json_bytes(report)
py_bytes = export_py_bytes(report)

st.download_button("Скачать JSON (коэффициенты модели)", data=json_bytes, file_name="hvp_adapted_model.json", mime="application/json")
st.download_button("Скачать PY (функция predict + коэффициенты)", data=py_bytes, file_name="hvp_model_predict.py", mime="text/x-python")

# Excel export
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    pd.DataFrame([{
        "y_name": report["y_name"],
        "degree": report["degree"],
        "r2": report["r2"],
        "mape_%": report["mape"],
        "n_rows": report["n_rows"],
        "n_x": len(report["x_cols"]),
    }]).to_excel(writer, index=False, sheet_name="Summary")
    coef_df.to_excel(writer, index=False, sheet_name="Coefficients")
    pd.DataFrame({"x_cols": report["x_cols"]}).to_excel(writer, index=False, sheet_name="X_columns")
    df.head(50).to_excel(writer, index=False, sheet_name="Data_head")
excel_buf.seek(0)

st.download_button("Скачать XLSX (отчёт)", data=excel_buf, file_name="hvp_adaptation_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# PDF export (без кириллицы-шрифтов в cloud лучше писать латиницей/коротко)
# Чтобы гарантировать читабельность без настройки шрифтов на сервере — делаем PDF на английском + цифры.
pdf_path = Path("hvp_report.pdf")
c = canvas.Canvas(str(pdf_path), pagesize=A4)
w, h = A4
c.setFont("Helvetica-Bold", 14)
c.drawString(40, h-50, "HVP-2 Model Adaptation Report")
c.setFont("Helvetica", 11)
c.drawString(40, h-80, f"Y: {report['y_name']}")
c.drawString(40, h-100, f"Degree: {report['degree']}")
c.drawString(40, h-120, f"R2: {report['r2']:.4f}")
c.drawString(40, h-140, f"MAPE: {report['mape']:.2f} %")
c.drawString(40, h-160, f"Rows: {report['n_rows']}")
c.drawString(40, h-180, f"X count: {len(report['x_cols'])}")
c.drawString(40, h-205, "X columns:")
yy = h-225
for i, xc in enumerate(report["x_cols"][:18]):
    c.drawString(60, yy - i*14, f"- {xc}")
c.showPage()
c.save()

st.download_button("Скачать PDF (краткий отчёт)", data=pdf_path.read_bytes(), file_name="hvp_report.pdf", mime="application/pdf")
