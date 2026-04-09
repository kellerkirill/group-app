import streamlit as st
from group_generator import generate_groups

def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.replace(',', '\n').split('\n') if x.strip()]

def parse_dict(text: str) -> dict[str, str]:
    d = {}
    for line in text.replace(',', '\n').split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            d[k.strip()] = v.strip()
    return d

def parse_limits(text: str) -> dict[str, list[str]]:
    limits = {}
    for line in text.replace(',', '\n').split('\n'):
        if ':' in line:
            k, vs = line.split(':', 1)
            limits[k.strip()] = [v.strip() for v in vs.replace(',', ' ').split() if v.strip()]
    return limits

st.title("👥 Генератор групп")
with st.form("gen_form"):
    n = st.number_input("Групп", min_value=1, value=2)
    all_p = st.text_area("Все участники (через запятую/строку)")
    gnd = st.text_area("Полы (Имя:Пол)")
    new = st.text_area("Новички")
    exp = st.text_area("Опытные")
    lim = st.text_area("Ограничения (Имя: Запрещённые)")
    strict_r = st.checkbox("Строгий баланс ролей", value=True)
    strict_g = st.checkbox("Строгий баланс полов", value=True)
    seed = st.number_input("Seed (опц.)", value=None)
    go = st.form_submit_button("Сгенерировать")

if go:
    try:
        people = parse_list(all_p)
        if not people: st.error("Введите участников"); st.stop()
        genders = parse_dict(gnd)
        missing = set(people) - set(genders)
        if missing: st.error(f"Укажите пол для: {missing}"); st.stop()

        res = generate_groups(
            group_number=n, all_people=people, genders=genders,
            newbies=parse_list(new), experts=parse_list(exp),
            limits=parse_limits(lim), seed=int(seed) if seed else None,
            strict_roles=strict_r, strict_genders=strict_g
        )
        st.success(f"✅ Seed: {res.used_seed} | Попыток: {res.attempts}")
        if res.warnings: st.warning("⚠️ " + "; ".join(res.warnings))
        for i, g in enumerate(res.groups, 1):
            st.subheader(f"Группа {i}")
            st.write(g)
        st.json(res.balance_metrics)
    except Exception as e:
        st.error(f"❌ {e}")