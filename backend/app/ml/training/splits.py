"""Particiones train/validation/test sin fuga de datos: se reparten GRUPOS (persona/llamada), no muestras.

Un grupo entero cae en un único split, de modo que grabaciones de la misma persona (o de la misma llamada)
nunca aparecen en TRAIN y TEST a la vez. Además se intenta mantener la proporción de clases en cada split.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict


def group_stratified_split(samples: list[dict], fractions: dict[str, float], seed: int = 42) -> tuple[dict[str, str], dict]:
    """samples: [{id, label, group}] -> ({id: split}, info).  fractions: {"train":.7,"validation":.15,"test":.15}"""
    tot = sum(fractions.values())
    fr = {k: v / tot for k, v in fractions.items() if v > 0}
    splits = list(fr)
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s.get("group") or f"__sample_{s['id']}"].append(s)
    rng = random.Random(seed)
    classes = sorted({s.get("label") or "__none__" for s in samples})
    total_c = Counter(s.get("label") or "__none__" for s in samples)
    target = {sp: {c: total_c[c] * fr[sp] for c in classes} for sp in splits}
    cur = {sp: Counter() for sp in splits}
    size = {sp: 0 for sp in splits}
    n = len(samples)

    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))               # grupos grandes primero
    assign: dict[str, str] = {}
    n_groups = {sp: 0 for sp in splits}
    for gname, members in items:
        gc = Counter(m.get("label") or "__none__" for m in members)
        best, best_score = None, None
        for sp in splits:
            deficit = sum(min(gc[c], max(0.0, target[sp][c] - cur[sp][c])) for c in classes)
            over = sum(max(0.0, cur[sp][c] + gc[c] - target[sp][c]) for c in classes)
            frac_left = fr[sp] - size[sp] / max(n, 1)
            score = deficit - 0.5 * over + 0.5 * frac_left * len(members)
            if best_score is None or score > best_score + 1e-9:
                best, best_score = sp, score
        for m in members:
            assign[m["id"]] = best
        cur[best].update(gc)
        size[best] += len(members)
        n_groups[best] += 1

    real_groups = [g for g in groups if not g.startswith("__sample_")]
    warnings = []
    leakage_free = True
    if len(real_groups) < len(splits) or any(n_groups[sp] == 0 for sp in splits):
        # No hay suficientes grupos: se cae a partición por muestra (con fuga posible) y se avisa.
        leakage_free = False
        warnings.append("Hay muy pocas personas/llamadas distintas para separar los conjuntos sin fuga de datos; "
                        "se particionó por muestra. Los resultados de TEST pueden ser optimistas.")
        assign = _sample_level(samples, fr, seed)
    else:
        for sp in splits:
            if size[sp] == 0:
                warnings.append(f"El conjunto {sp} quedó vacío.")
    counts = Counter(assign.values())
    return assign, {"leakage_free": leakage_free, "warnings": warnings, "counts": dict(counts),
                    "groups": len(groups), "fractions": fr}


def _sample_level(samples: list[dict], fr: dict[str, float], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        by_label[s.get("label") or "__none__"].append(s["id"])
    out: dict[str, str] = {}
    for ids in by_label.values():
        rng.shuffle(ids)
        n, acc = len(ids), 0
        keys = list(fr)
        for i, sp in enumerate(keys):
            cnt = n - acc if i == len(keys) - 1 else int(round(n * fr[sp]))
            for sid in ids[acc:acc + cnt]:
                out[sid] = sp
            acc += cnt
    return out
