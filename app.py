# app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import io
from names_db import detect_gender, _RU_NAMES
from generator import generate_groups

st.set_page_config(page_title="РАСХОДИМСЯ ПО ГРУППАМ!", layout="wide")
st.title("🔥 РАСХОДИМСЯ ПО ГРУППАМ!")

DATA_KEY = "residents_data"
WIDGET_KEY = "residents_widget"
COLUMNS = ["Имя", "Пол", "Роль", "🚦 Статус"]

# 🛡 Валидация состояния при старте (чистит битые сессии после деплоя)
if DATA_KEY not in st.session_state or st.session_state[DATA_KEY].columns.tolist() != COLUMNS:
    st.session_state[DATA_KEY] = pd.DataFrame(columns=COLUMNS)

# Хелпер с защитой от KeyError
def get_current_df():
    df = st.session_state.get(WIDGET_KEY, st.session_state.get(DATA_KEY, pd.DataFrame(columns=COLUMNS)))
    if "Имя" not in df.columns:
        return pd.DataFrame(columns=COLUMNS)
    return df

# 2. Массовая вставка
with st.expander("📋 Массовое добавление резидентов", expanded=True):
    bulk_text = st.text_area("Вставьте список имён (каждое с новой строки)", height=80)
    if st.button("➕ Добавить в таблицу", width="stretch"):
        names = [n.strip() for n in bulk_text.splitlines() if n.strip()]
        if names:
            df = get_current_df()
            existing = set(df["Имя"].dropna().str.strip().tolist())
            unique_names = list(dict.fromkeys(names))
            to_add = [n for n in unique_names if n not in existing]
            if to_add:
                genders, statuses = [], []
                for n in to_add:
                    first = n.strip().split()[0].lower().rstrip('.,!?:;')
                    genders.append(detect_gender(n))
                    statuses.append("✅
