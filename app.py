import streamlit as st
import pandas as pd
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
        if a not in all_set or not all(b in all_set for b in bs): raise ValueError("Невалидные имена в limits.")
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
# STREAMLIT UI (НОВЫЙ)
# ========================
st.title("👥 Генератор групп")
st.markdown("Добавьте участников в таблицу, укажите пол и роль. Конфликты задаются в поле ниже.")

# Инициализация таблицы
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["Имя", "Пол", "Роль"])

edited_df = st.data_editor(
    st.session_state.df,
    column_config={
        "Имя": st.column_config.TextColumn("Имя", required=True, help="Введите имя"),
        "Пол": st.column_config.SelectboxColumn("Пол", options=["M", "F"], required=True, default="M"),
        "Роль": st.column_config.SelectboxColumn("Роль", options=["regular", "newbie", "expert"], required=True, default="regular"),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="dynamic",
    key="participants_table"
)
st.session_state.df = edited_df  # сохраняем изменения

# Ограничения
limits_txt = st.text_area(
    "⚠️ Ограничения (каждая строка: `Имя: Имена_через_пробел`)",
    placeholder="Иван: Петр Мария\nОлег: Анна",
    height=80
)

with st.expander("⚙️ Настройки"):
    n = st.number_input("Количество групп", min_value=1, value=2)
    strict_r = st.checkbox("Строгий баланс ролей", value=True)
    strict_g = st.checkbox("Строгий баланс полов", value=True)
    seed = st.number_input("Seed (опционально)", value=None, step=1)
    run = st.button("🚀 Сгенерировать", type="primary", use_container_width=True)

if run:
    # Валидация
    valid_df = edited_df.dropna(subset=["Имя"])
    if valid_df.empty:
        st.error("Таблица пуста. Добавьте хотя бы одного участника.")
        st.stop()
    names = valid_df["Имя"].str.strip().tolist()
    if len(set(names)) != len(names):
        st.error("В таблице есть дубликаты имён.")
        st.stop()
        
    genders = dict(zip(valid_df["Имя"].str.strip(), valid_df["Пол"]))
    newbies = valid_df[valid_df["Роль"]=="newbie"]["Имя"].str.strip().tolist()
    experts = valid_df[valid_df["Роль"]=="expert"]["Имя"].str.strip().tolist()
    
    limits = {}
    for line in limits_txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            limits[k.strip()] = [x.strip() for x in v.replace(",", " ").split() if x.strip()]

    try:
        res = generate_groups(
            n, names, genders, newbies, experts,
            limits=limits, seed=int(seed) if seed else None,
            strict_r=strict_r, strict_g=strict_g
        )
        st.success(f"✅ Seed: {res.used_seed} | Попыток: {res.attempts}")
        if res.warnings: st.warning("⚠️ " + "; ".join(res.warnings))
        for i, g in enumerate(res.groups, 1):
            st.subheader(f"Группа {i}")
            st.write(", ".join(g))
        st.json(res.balance_metrics)
    except Exception as e:
        st.error(f"❌ {e}")
