import streamlit as st
import pandas as pd
import io
import random
from collections import defaultdict
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================
# ЛОГИКА ГЕНЕРАТОРА
# ========================
@dataclass
class GroupingResult:
    groups: List[List[str]]
    attempts: int
    status: str
    warnings: List[str]
    used_seed: int
    balance_metrics: dict

def verify_groups(groups, limits, roles, genders, strict_r=True, strict_g=True):
    flat = [p for g in groups for p in g]
    if len(flat) != len(set(flat)): return False
    if max(len(g) for g in groups) - min(len(g) for g in groups) > 1: return False
    conflicts = defaultdict(set)
    for a, bs in limits.items():
        for b in bs: conflicts[a].add(b)
    for g in groups:
        gs = set(g)
        for p in g:
            if conflicts[p] & gs: return False
    n = len(groups)
    if strict_r:
        rt = defaultdict(int)
        for r in roles.values(): rt[r] += 1
        tgt = {r: [b+(1 if i<e else 0) for i in range(n)] for r, c in rt.items() for b,e in [divmod(c,n)]}
        for i, g in enumerate(groups):
            gc = defaultdict(int)
            for p in g: gc[roles[p]] += 1
            for r in tgt:
                if gc.get(r,0) != tgt[r][i]: return False
    if strict_g:
        gt = defaultdict(int)
        for g in genders.values(): gt[g] += 1
        tgt = {g: [b+(1 if i<e else 0) for i in range(n)] for g, c in gt.items() for b,e in [divmod(c,n)]}
        for i, g in enumerate(groups):
            gc = defaultdict(int)
            for p in g: gc[genders[p]] += 1
            for g_type in tgt:
                if gc.get(g_type,0) != tgt[g_type][i]: return False
    return True

def _run_attempt(seed, cfg):
    rng = random.Random(seed)
    all_p, roles, genders, conflicts = cfg['all'], cfg['roles'], cfg['genders'], cfg['conflicts']
    size_tgt, role_tgt, gender_tgt = cfg['size_tgt'], cfg['role_tgt'], cfg['gender_tgt']
    strict_r, strict_g, g_num = cfg['strict_r'], cfg['strict_g'], cfg['g_num']
    pool = sorted(all_p, key=lambda p: len(conflicts[p]), reverse=True)
    rng.shuffle(pool)
    groups = [[] for _ in range(g_num)]
    g_rc, g_gc = [defaultdict(int) for _ in range(g_num)], [defaultdict(int) for _ in range(g_num)]
    warnings = []
    for p in pool:
        r, gn = roles[p], genders[p]
        valid = [i for i in range(g_num) if len(groups[i]) < size_tgt[i] 
                 and (not strict_r or g_rc[i][r] < role_tgt[r][i])
                 and (not strict_g or g_gc[i][gn] < gender_tgt[gn][i])
                 and not any(p in conflicts[m] for m in groups[i])]
        if not valid:
            if strict_r or strict_g:
                valid = [i for i in range(g_num) if len(groups[i]) < size_tgt[i] and not any(p in conflicts[m] for m in groups[i])]
                if valid and not warnings: warnings.append("Баланс ролей/полов ослаблен.")
        if not valid: return None
        idx = rng.choice(valid)
        groups[idx].append(p); g_rc[idx][r] += 1; g_gc[idx][gn] += 1

    def score():
        dr = sum(sum(abs(g_rc[i][r]-role_tgt[r][i]) for r in role_tgt) for i in range(g_num))
        dg = sum(sum(abs(g_gc[i][g]-gender_tgt[g][i]) for g in gender_tgt) for i in range(g_num))
        return dr+dg
    best, stag = score(), 0
    for _ in range(150):
        if stag >= 25: break
        g1, g2 = rng.sample(range(g_num), 2)
        if not groups[g1] or not groups[g2]: continue
        p1, p2 = rng.choice(groups[g1]), rng.choice(groups[g2])
        r1, r2, gn1, gn2 = roles[p1], roles[p2], genders[p1], genders[p2]
        if any(p1 in conflicts[m] for m in groups[g2] if m!=p2): continue
        if any(p2 in conflicts[m] for m in groups[g1] if m!=p1): continue
        if strict_r and (g_rc[g2][r1]+1>role_tgt[r1][g2] or g_rc[g1][r2]+1>role_tgt[r2][g1]): continue
        if strict_g and (g_gc[g2][gn1]+1>gender_tgt[gn1][g2] or g_gc[g1][gn2]+1>gender_tgt[gn2][g1]): continue
        groups[g1].remove(p1); groups[g1].append(p2)
        groups[g2].remove(p2); groups[g2].append(p1)
        g_rc[g1][r1]-=1; g_rc[g1][r2]+=1; g_rc[g2][r2]-=1; g_rc[g2][r1]+=1
        g_gc[g1][gn1]-=1; g_gc[g1][gn2]+=1; g_gc[g2][gn2]-=1; g_gc[g2][gn1]+=1
        ns = score()
        if ns < best: best, stag = ns, 0
        elif ns == best:
            if rng.random()<0.1: rng.shuffle(groups)
            else: stag += 1
        else:
            groups[g1].remove(p2); groups[g1].append(p1)
            groups[g2].remove(p1); groups[g2].append(p2)
            g_rc[g1][r1]+=1; g_rc[g1][r2]-=1; g_rc[g2][r2]+=1; g_rc[g2][r1]-=1
            g_gc[g1][gn1]+=1; g_gc[g1][gn2]-=1; g_gc[g2][gn2]+=1; g_gc[g2][gn1]-=1
            stag += 1
    n = len(groups)
    bal = {"size_diff": max(len(g) for g in groups)-min(len(g) for g in groups),
           "role_dev": {r:[g_rc[i][r]-role_tgt[r][i] for i in range(n)] for r in role_tgt},
           "gender_dev": {g:[g_gc[i][g]-gender_tgt[g][i] for i in range(n)] for g in gender_tgt}}
    return GroupingResult(groups, 1, "success", warnings, seed, bal)

def generate_groups(n, all_p, genders, newbies=None, experts=None, roles=None, limits=None, seed=None, strict_r=True, strict_g=True, max_att=500, workers=0):
    used = seed if seed is not None else random.randint(0, 2**32-1)
    all_set = set(all_p)
    if len(all_set)!=len(all_p): raise ValueError("Дубликаты в all_people.")
    if len(all_p)<n: raise ValueError("Людей меньше, чем групп.")
    if set(genders)!=all_set: raise ValueError("genders не покрывает all_people.")
    if roles is None:
        if newbies and experts and set(newbies)&set(experts): raise ValueError("newbies и experts пересекаются.")
        roles = {p:'newbie' for p in (newbies or [])}
        roles.update({p:'expert' for p in (experts or [])})
        for p in all_p: roles.setdefault(p,'regular')
    if set(roles)-all_set: raise ValueError("Неизвестные имена в roles.")
    limits = limits or {}
    conflicts = defaultdict(set)
    for a, bs in limits.items():
        if a not in all_set or not all(b in all_set for b in bs): raise ValueError(f"Имя '{a}' или партнёр отсутствуют в списке резидентов.")
        for b in bs:
            if a!=b: conflicts[a].add(b); conflicts[b].add(a)
    base, ext = divmod(len(all_p), n)
    size_tgt = [base+(1 if i<ext else 0) for i in range(n)]
    max_sz = base+(1 if ext>0 else 0)
    for p, cl in conflicts.items():
        if len(cl)>=max_sz: raise RuntimeError(f"Конфликт '{p}' ({len(cl)}) > макс. размера группы ({max_sz}).")
    rt = defaultdict(int)
    for r in roles.values(): rt[r]+=1
    role_tgt = {r:[b+(1 if i<e else 0) for i in range(n)] for r,c in rt.items() for b,e in [divmod(c,n)]}
    gt = defaultdict(int)
    for g in genders.values(): gt[g]+=1
    gender_tgt = {g:[b+(1 if i<e else 0) for i in range(n)] for g,c in gt.items() for b,e in [divmod(c,n)]}
    cfg = {'all':all_p,'roles':roles,'genders':genders,'conflicts':conflicts,'size_tgt':size_tgt,
           'role_tgt':role_tgt,'gender_tgt':gender_tgt,'strict_r':strict_r,'strict_g':strict_g,'g_num':n}
    seeds = [used+i for i in range(max_att)]
    if workers>1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_attempt, s, cfg): i for i, s in enumerate(seeds)}
            for f in as_completed(futs):
                r = f.result()
                if r: r.attempts=futs[f]+1; return r
    else:
        for i, s in enumerate(seeds):
            r = _run_attempt(s, cfg)
            if r: r.attempts=i+1; return r
    raise RuntimeError(f"Сборка не удалась за {max_att} попыток. Seed: {used}")

# ========================
# БАЗА ИМЁН & ОПРЕДЕЛЕНИЕ ПОЛА
# ========================
_RU_NAMES = {
    'александр':'M','саша':'M','саня':'M','дима':'M','димка':'M','николай':'M','коля':'M',
    'сергей':'M','серега':'M','серж':'M','андрей':'M','андрюша':'M','владимир':'M','вова':'M',
    'влад':'M','иван':'M','ваня':'M','юра':'M','юрий':'M','петр':'M','пётр':'M','петя':'M',
    'миша':'M','максим':'M','макс':'M','артем':'M','артём':'M','илья':'M','кирилл':'M',
    'роман':'M','рома':'M','олег':'M','денис':'M','толя':'M','анатолий':'M','алексей':'M',
    'леша':'M','лёша':'M','паша':'M','витя':'M','игорь':'M','егор':'M','гена':'M','геннадий':'M',
    'борис':'M','боря':'M','стас':'M','антон':'M','федя':'M','вадим':'M','тимур':'M','марк':'M',
    'лев':'M','лёва':'M','арсений':'M','сеня':'M','матвей':'M','митя':'M','даниил':'M','даня':'M',
    'тимофей':'M','ян':'M','ярик':'M','ярослав':'M','филипп':'M','филя':'M','роберт':'M',
    'рудик':'M','семен':'M','семён':'M','савва':'M','степан':'M','стёпа':'M','тарас':'M',
    'тихон':'M','фаддей':'M','харитон':'M','эдик':'M','эдуард':'M','юлий':'M','яков':'M',
    'яша':'M','вася':'M','василий':'M','гриша':'M','григорий':'M','георгий':'M','жора':'M',
    'захар':'M','аркадий':'M','аркаша':'M','веня':'M','вениамин':'M','виктор':'M','виталий':'M',
    'владислав':'M','всеволод':'M','вячеслав':'M','слава':'M','глеб':'M','давид':'M','данил':'M',
    'еремей':'M','ермолай':'M','игнат':'M','игнатий':'M','иосиф':'M','осип':'M','константин':'M',
    'костя':'M','леонид':'M','лёня':'M','марат':'M','мирон':'M','назар':'M','никита':'M',
    'нестор':'M','платон':'M','потап':'M','прохор':'M','радик':'M','руслан':'M','савелий':'M',
    'серёжа':'M','станислав':'M','терентий':'M','трофим':'M','устин':'M','фёдор':'M','федор':'M',
    'александра':'F','алёна':'F','алена':'F','алина':'F','алиса':'F','алла':'F','альбина':'F',
    'амелия':'F','анастасия':'F','настя':'F','анжела':'F','анжелика':'F','анна':'F','анюта':'F',
    'антонина':'F','арина':'F','белла':'F','валентина':'F','валя':'F','валерия':'F','лера':'F',
    'василиса':'F','вера':'F','вероника':'F','вика':'F','виктория':'F','виолетта':'F','влада':'F',
    'галина':'F','галя':'F','дарья':'F','даша':'F','дарина':'F','диана':'F','дина':'F',
    'евгения':'F','женя':'F','евдокия':'F','дуня':'F','екатерина':'F','катя':'F','елена':'F',
    'лена':'F','елизавета':'F','лиза':'F','элли':'F','жанна':'F','злата':'F','зина':'F',
    'зинаида':'F','зоя':'F','инга':'F','инна':'F','ирина':'F','ира':'F','камила':'F','карина':'F',
    'кира':'F','клара':'F','ксения':'F','ксюша':'F','лариса':'F','лара':'F','лида':'F','лидия':'F',
    'лилия':'F','лина':'F','люба':'F','любовь':'F','люда':'F','людмила':'F','люся':'F','майя':'F',
    'маргарита':'F','марина':'F','мария':'F','маша':'F','милана':'F','мила':'F','мира':'F',
    'надежда':'F','надя':'F','наина':'F','наталия':'F','наташа':'F','нелли':'F','ника':'F',
    'нина':'F','оксана':'F','оля':'F','ольга':'F','поля':'F','полина':'F','раиса':'F',
    'роза':'F','света':'F','светлана':'F','сима':'F','симона':'F','соня':'F','софа':'F','софия':'F',
    'софья':'F','стелла':'F','таня':'F','татьяна':'F','тина':'F','уля':'F','ульяна':'F','фаина':'F',
    'эвелина':'F','эльвира':'F','эмма':'F','юлия':'F','юля':'F','юнона':'F','яна':'F','янина':'F',
    'ярослава':'F','ева':'F','таисия':'F','тая':'F','варвара':'F','варя':'F','агата':'F','аделина':'F',
    'аглая':'F','аида':'F','албина':'F','алина':'F','алиса':'F','альбина':'F','амалия':'F','амелия':'F',
    'ангелина':'F','анисья':'F','анна':'F','антигона':'F','антонина':'F','анфиса':'F','арина':'F',
    'ася':'F','афанасия':'F','аэлита':'F','беатриса':'F','белла':'F','берта':'F','божена':'F',
    'бронислава':'F','валентина':'F','валерия':'F','ванесса':'F','варвара':'F','василиса':'F',
    'вера':'F','вероника':'F','вика':'F','виолетта':'F','вита':'F','виталина':'F','влада':'F',
    'власта':'F','волга':'F','всеслава':'F','галина':'F','галя':'F','гелена':'F','генриетта':'F',
    'глафира':'F','глаша':'F','глория':'F','голена':'F','грета':'F','дарина':'F','дарья':'F',
    'даша':'F','джанна':'F','диана':'F','динора':'F','дина':'F','евдокия':'F','евгения':'F',
    'екатерина':'F','елена':'F','елизавета':'F','женя':'F','жульетта':'F','зара':'F','звенислава':'F',
    'земфира':'F','зинаида':'F','злата':'F','зоя':'F','ива':'F','изабелла':'F','изольда':'F',
    'ильмира':'F','ильса':'F','ильяна':'F','инга':'F','инесса':'F','инна':'F','иоланта':'F',
    'ипполита':'F','ираида':'F','ирина':'F','исаура':'F','исидора':'F','иулиания':'F','капитолина':'F',
    'карина':'F','карла':'F','кармен':'F','каролина':'F','кира':'F','кирилла':'F','клара':'F',
    'клавдия':'F','клеопатра':'F','констанция':'F','кристина':'F','ксения':'F','куба':'F','лада':'F',
    'лариса':'F','лера':'F','леона':'F','лиана':'F','либерия':'F','лидия':'F','лиза':'F','лилия':'F',
    'лина':'F','лира':'F','лова':'F','лола':'F','луиза':'F','луна':'F','любава':'F','любовь':'F',
    'людмила':'F','майя':'F','маланья':'F','маргарита':'F','марианна':'F','марина':'F','мария':'F',
    'марта':'F','марфа':'F','марьяна':'F','маша':'F','мелания':'F','милана':'F','милена':'F',
    'милослава':'F','мина':'F','мира':'F','мирослава':'F','муза':'F','надежда':'F','наина':'F',
    'настя':'F','наталия':'F','наталья':'F','нелли':'F','ника':'F','нина':'F','нинель':'F','нона':'F',
    'олимпиада':'F','оля':'F','ольга':'F','пелагея':'F','полина':'F','прасковья':'F','рада':'F',
    'раиса':'F','регина':'F','римма':'F','роза':'F','рославля':'F','руфина':'F','светлана':'F',
    'серафима':'F','сима':'F','симона':'F','сна':'F','снежана':'F','софья':'F','стася':'F','стелла':'F',
    'степанида':'F','сусанна':'F','таисия':'F','тамара':'F','таня':'F','тата':'F','таша':'F',
    'татьяна':'F','тереза':'F','ульяна':'F','уля':'F','фатима':'F','феодора':'F','феофания':'F',
    'флора':'F','харита':'F','христина':'F','цао':'F','цезария':'F','целина':'F','цифрина':'F',
    'челси':'F','шахерезада':'F','элвира':'F','элена':'F','эмилия':'F','эмма':'F','эрика':'F',
    'эсмеральда':'F','эсфирь':'F','юлиана':'F','юлия':'F','юнона':'F','янна':'F','яна':'F',
    'ярослава':'F','янина':'F'
}

def detect_gender(name: str) -> str:
    first = name.strip().split()[0].lower().rstrip('.,!?:;')
    return _RU_NAMES.get(first, 'M')

# ========================
# STREAMLIT UI (BULLETPROOF)
# ========================
st.set_page_config(page_title="РАСХОДИМСЯ ПО ГРУППАМ!", layout="wide")
st.title("🔥 РАСХОДИМСЯ ПО ГРУППАМ!")

DATA_KEY = "residents_data"
WIDGET_KEY = "residents_widget"
COLUMNS = ["Имя", "Пол", "Роль", "🚦 Статус"]

# 1. Инициализация хранилища (никогда не пишем в WIDGET_KEY вручную!)
if DATA_KEY not in st.session_state:
    st.session_state[DATA_KEY] = pd.DataFrame(columns=COLUMNS)
else:
    # Миграция старых ролей (идемпотентная)
    df = st.session_state[DATA_KEY]
    if not df.empty and "Роль" in df.columns and df["Роль"].isin(["regular", "expert", "newbie"]).any():
        df["Роль"] = df["Роль"].replace({"regular": "Обычный", "expert": "ВПИ", "newbie": "Новичок"})
        st.session_state[DATA_KEY] = df
        if WIDGET_KEY in st.session_state: del st.session_state[WIDGET_KEY]
        st.rerun()

# Хелпер для получения актуального DF (из виджета или хранилища)
def get_current_df():
    return st.session_state.get(WIDGET_KEY, st.session_state.get(DATA_KEY, pd.DataFrame()))

# 2. Массовая вставка
with st.expander("📋 Массовое добавление резидентов", expanded=True):
    bulk_text = st.text_area("Вставьте список имён (каждое с новой строки)", height=80)
    if st.button("➕ Добавить в таблицу", use_container_width=False, width="stretch"):
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
                    genders.append(_RU_NAMES.get(first, 'M'))
                    statuses.append("✅" if first in _RU_NAMES else "🔴 Не определён")
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
        df["🚦 Статус"] = df["Имя"].apply(lambda n: "✅" if n.strip().split()[0].lower().rstrip('.,!?:;') in _RU_NAMES else "🔴 Не определён")
        st.session_state[DATA_KEY] = df
        if WIDGET_KEY in st.session_state: del st.session_state[WIDGET_KEY]
        st.rerun()

# 4. Предупреждение
current_df = st.session_state[DATA_KEY]
if "🚦 Статус" in current_df.columns:
    undetected = current_df[current_df["🚦 Статус"].str.contains("🔴", na=False)]
    if not undetected.empty:
        st.warning(f"🔴 Проверьте вручную: {', '.join(undetected['Имя'])}")

# 5. Таблица резидентов (КРИТИЧЕСКИЙ ФИКС)
st.subheader("📝 Резиденты")
# Виджет берёт данные из DATA_KEY только при первом рендере. Далее управляет состоянием сам.
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
    width="stretch",  # ✅ Исправлен деприкейт use_container_width
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
    # Читаем напрямую из состояния виджета (там хранятся все правки пользователя)
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
            
        # 📥 Экспорт в Excel
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
