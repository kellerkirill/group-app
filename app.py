# app.py
import streamlit as st
import pandas as pd
import io
from names_db import detect_gender
from generator import generate_groups

st.set_page_config(page_title="РАСХОДИМСЯ ПО ГРУППАМ!", layout="wide")
st.title("🔥 РАСХОДИМСЯ ПО ГРУППАМ!")

DATA_KEY = "residents_data"
WIDGET_KEY = "residents_widget"
COLUMNS = ["Имя", "Пол", "Роль", "🚦 Статус"]

# 1. Инициализация
if DATA_KEY not in st.session_state:
    st.session_state[DATA_KEY] = pd.DataFrame(columns=COLUMNS)
else:
    df = st.session_state[DATA_KEY]
    if not df.empty and "Роль" in df.columns and df["Роль"].isin(["regular", "expert", "newbie"]).any():
        df["Роль"] = df["Роль"].replace({"regular": "Обычный", "expert": "ВПИ", "newbie": "Новичок"})
        st.session_state[DATA_KEY] = df
        if WIDGET_KEY in st.session_state: del st.session_state[WIDGET_KEY]
        st.rerun()

def get_current_df():
    return st.session_state.get(WIDGET_KEY, st.session_state.get(DATA_KEY, pd.DataFrame()))

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
                    statuses.append("✅" if first in __import__('names_db')._RU_NAMES else "🔴 Не определён")
                new_rows = pd.DataFrame({"Имя": to_add, "Пол": genders, "Роль": "Обычный", "🚦 Статус": statuses})
                st.session_state[DATA_KEY] = pd.concat([df, new_rows], ignore_index=True)
                if WIDGET_KEY in st.session_state: del st.session_state[WIDGET_KEY]
                st.success(f"✅ Добавлено: {len(to_add)}")
                st.rerun()
            else:
                st.warning("Все имена уже есть в таблице.")

# 3. Авто-определение пола
if st.button("🔍 Авто-определить пол у всех", type="secondary", width="stretch"):
    df = get_current_df().copy()
    if not df.empty:
        df["Пол"] = df["Имя"].astype(str).apply(detect_gender)
        df["🚦 Статус"] = df["Имя"].apply(lambda n: "✅" if n.strip().split()[0].lower().rstrip('.,!?:;') in __import__('names_db')._RU_NAMES else "🔴 Не определён")
        st.session_state[DATA_KEY] = df
        if WIDGET_KEY in st.session_state: del st.session_state[WIDGET_KEY]
        st.rerun()

# 4. Предупреждение
current_df = st.session_state[DATA_KEY]
if "🚦 Статус" in current_df.columns:
    undetected = current_df[current_df["🚦 Статус"].str.contains("🔴", na=False)]
    if not undetected.empty:
        st.warning(f"🔴 Проверьте вручную: {', '.join(undetected['Имя'])}")

# 5. Таблица резидентов
st.subheader("📝 Резиденты")
table_data = st.session_state[DATA_KEY] if WIDGET_KEY not in st.session_state else None

st.data_editor(
    table_data,
    key=WIDGET_KEY,
    column_config={
        "Имя": st.column_config.TextColumn("Имя", required=True),
        "Пол": st.column_config.SelectboxColumn("Пол", options=["M", "F"], required=True, default="M"),
        "Роль": st.column_config.SelectboxColumn("Роль", options=["Обычный", "ВПИ", "Новичок"], required=True, default="Обычный"),
        "🚦 Статус": st.column_config.TextColumn("Статус", width="small", disabled=True),
    },
    hide_index=True,
    width="stretch",
    num_rows="dynamic"
)

# 6. Границы
st.subheader("🌍 Границы")
limits_txt = st.text_area(
    "Укажите, кто не должен быть в одной группе (Имя: Через запятую)",
    placeholder="Олег С: Леша Ч, Иван П\nАня К: Петя О, Маша И",
    height=240
)

# 7. Настройки
with st.expander("⚙️ Настройки генерации"):
    n = st.number_input("Количество групп", min_value=1, value=2)
    strict_r = st.checkbox("Строгий баланс ролей", value=True)
    strict_g = st.checkbox("Строгий баланс полов", value=True)
    seed = st.number_input("Seed (опционально)", value=None, step=1)

# 8. Генерация
st.markdown("---")
run = st.button("🚀 РАСПРЕДЕЛИТЬ!", type="primary", width="stretch")

if run:
    df = st.session_state.get(WIDGET_KEY, st.session_state.get(DATA_KEY, pd.DataFrame()))
    valid_df = df.dropna(subset=["Имя"])
    if valid_df.empty:
        st.error("Таблица пуста. Добавьте резидентов.")
        st.stop()
    names = valid_df["Имя"].str.strip().tolist()
    if len(set(names)) != len(names):
        st.error("В таблице есть дубликаты имён.")
        st.stop()
        
    genders = dict(zip(valid_df["Имя"].str.strip(), valid_df["Пол"]))
    newbies = valid_df[valid_df["Роль"]=="Новичок"]["Имя"].str.strip().tolist()
    experts = valid_df[valid_df["Роль"]=="ВПИ"]["Имя"].str.strip().tolist()
    
    limits = {}
    for line in limits_txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k_clean = k.strip()
            if not k_clean: continue
            limits[k_clean] = [x.strip() for x in v.split(',') if x.strip()]

    try:
        res = generate_groups(n, names, genders, newbies, experts, limits=limits, 
                              seed=int(seed) if seed else None, strict_r=strict_r, strict_g=strict_g)
        st.success(f"✅ Seed: {res.used_seed} | Попыток: {res.attempts}")
        if res.warnings: st.warning("⚠️ " + "; ".join(res.warnings))
        
        for i, g in enumerate(res.groups, 1):
            st.subheader(f"Группа {i} ({len(g)} чел.)")
            st.write(", ".join(g))
            
        if res.groups:
            max_len = max(len(g) for g in res.groups)
            export_data = {}
            for i, g in enumerate(res.groups, 1):
                col_name = f"Группа {i} ({len(g)} чел.)"
                export_data[col_name] = g + [""] * (max_len - len(g))
            df_export = pd.DataFrame(export_data)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name="Группы")
            output.seek(0)
            st.download_button("📥 Скачать результат в Excel", data=output, file_name="groups_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
    except Exception as e:
        st.error(f"❌ {e}")
