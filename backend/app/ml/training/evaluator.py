"""Evaluación de clasificadores de emoción: accuracy, precision, recall, F1, Macro-F1, Weighted-F1, matriz de confusión."""
from __future__ import annotations

import numpy as np


def metrics_from_predictions(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """`labels` = etiquetas del dataset. Si el modelo predice etiquetas fuera de `labels` se añaden a la matriz
    (cuentan como error) para no ocultar confusiones."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)
    extra = sorted({p for p in y_pred if p not in labels})
    all_labels = list(labels) + extra
    present = [l for l in all_labels if l in set(y_true) or l in set(y_pred)]
    if not y_true:
        return {"n_samples": 0, "accuracy": None, "precision": None, "recall": None, "f1": None, "macro_f1": None,
                "weighted_f1": None, "per_class": {}, "confusion_matrix": [], "labels": all_labels}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=all_labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    supp_labels = [i for i, l in enumerate(all_labels) if s[i] > 0]          # solo clases con muestras reales
    macro = lambda v: float(np.mean([v[i] for i in supp_labels])) if supp_labels else 0.0
    wf = float(np.sum([f[i] * s[i] for i in range(len(all_labels))]) / max(np.sum(s), 1))
    return {
        "n_samples": len(y_true), "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": macro(p), "recall": macro(r), "f1": macro(f), "macro_f1": macro(f), "weighted_f1": wf,
        "per_class": {all_labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                      "support": int(s[i])} for i in range(len(all_labels))},
        "confusion_matrix": cm.tolist(), "labels": all_labels,
    }


def evaluate_loader(model, loader, labels: list[str], device: str, loss_fn=None) -> dict:
    """Recorre un DataLoader (input_values, attention_mask, y) y devuelve métricas + pérdida."""
    import torch
    model.eval()
    ys, ps, losses = [], [], []
    with torch.inference_mode():
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device) if mask is not None else None
            logits = model(x, attention_mask=mask).logits.float() if mask is not None else model(x).logits.float()
            if loss_fn is not None:
                losses.append(float(loss_fn(logits, y.to(device)).item()) * len(y))
            ps += logits.argmax(-1).cpu().tolist()
            ys += y.tolist()
    m = metrics_from_predictions([labels[i] for i in ys], [labels[i] for i in ps], labels)
    m["loss"] = float(sum(losses) / max(len(ys), 1)) if losses else None
    return m


def regression_metrics(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - np.sum(err ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)),
            "pearson": float(np.corrcoef(p, y)[0, 1]) if len(y) > 2 and np.std(p) > 0 and np.std(y) > 0 else None,
            "n_samples": int(len(y))}
