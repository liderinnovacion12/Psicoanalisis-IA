"""Fine-tuning de modelos de emoción de audio (wav2vec2). Bucle propio para controlar stop/continuar/progreso.

Características: AdamW + warmup lineal, gradient accumulation, weight decay, class weights u oversampling,
augmentation solo en TRAIN, early stopping por Macro-F1 de validación, checkpoint por época (reanudable),
parada cooperativa, fp16 en CUDA, congelado del extractor de características.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.errors import StopRequested
from app.core.logging import get_logger
from app.ml.training.dataset import AudioDataset, Sample, compute_class_weights, make_collate
from app.ml.training.evaluator import evaluate_loader

log = get_logger(__name__)


@dataclass
class TrainParams:
    learning_rate: float = 3e-5
    batch_size: int = 8
    epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 50
    gradient_accumulation: int = 2
    early_stopping_patience: int = 3
    freeze_feature_encoder: bool = True
    imbalance_strategy: str = "class_weights"      # none | class_weights | oversample
    augmentation: dict = field(default_factory=dict)
    max_seconds: float = 8.0
    seed: int = 42
    fp16: bool = True
    num_workers: int = 0

    @staticmethod
    def from_dict(d: dict) -> "TrainParams":
        known = TrainParams.__dataclass_fields__
        return TrainParams(**{k: v for k, v in d.items() if k in known})


@dataclass
class Callbacks:
    on_step: Callable[[dict], None] = lambda info: None
    on_epoch_end: Callable[[dict], None] = lambda info: None
    should_stop: Callable[[], bool] = lambda: False


def load_for_training(base_source: str, labels: list[str], cache_dir: str | None = None, local_only: bool = False):
    """Carga (extractor, modelo) preservando la arquitectura del checkpoint base y reinicializando SOLO la capa de
    salida si el conjunto de etiquetas cambia."""
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification
    from app.ml.emotion.architectures import Wav2Vec2ForSpeechClassification, uses_speech_classification_head
    kw = {"cache_dir": cache_dir, "local_files_only": local_only}
    extractor = AutoFeatureExtractor.from_pretrained(base_source, **kw)
    cfg = AutoConfig.from_pretrained(base_source, **kw)
    same = [cfg.id2label[i].lower() for i in range(len(cfg.id2label))] == [l.lower() for l in labels] if cfg.id2label else False
    cfg.num_labels = len(labels)
    cfg.id2label = {i: l for i, l in enumerate(labels)}
    cfg.label2id = {l: i for i, l in enumerate(labels)}
    custom = uses_speech_classification_head(cfg)
    cls = Wav2Vec2ForSpeechClassification if custom else AutoModelForAudioClassification
    if custom:
        cfg.architectures = ["Wav2Vec2ForSpeechClassification"]
        cfg.finetuning_task = "wav2vec2_clf"
        if getattr(cfg, "pooling_mode", None) is None:
            cfg.pooling_mode = "mean"
    model = cls.from_pretrained(base_source, config=cfg, ignore_mismatched_sizes=not same, **kw)
    return extractor, model, same


class EmotionTrainer:
    def __init__(self, base_source: str, labels: list[str], params: TrainParams, out_dir: Path,
                 callbacks: Callbacks | None = None, cache_dir: str | None = None, local_only: bool = False):
        self.base_source, self.labels, self.p = base_source, labels, params
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cb = callbacks or Callbacks()
        self.cache_dir, self.local_only = cache_dir, local_only

    # -----------------------------------------------------------------------------------------------------
    def fit(self, train: list[Sample], val: list[Sample], test: list[Sample] | None = None,
            resume: bool = False) -> dict:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from transformers import get_linear_schedule_with_warmup

        p = self.p
        torch.manual_seed(p.seed)
        np.random.seed(p.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor, model, same_labels = load_for_training(self.base_source, self.labels, self.cache_dir, self.local_only)
        if p.freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        model.to(device)
        collate = make_collate(extractor)

        ds_train = AudioDataset(train, self.labels, p.max_seconds, True, p.augmentation, seed=p.seed)
        ds_val = AudioDataset(val, self.labels, p.max_seconds, False)
        sampler = None
        if p.imbalance_strategy == "oversample":
            cnt = {l: max(1, sum(1 for s in train if s.label == l)) for l in self.labels}
            sampler = WeightedRandomSampler([1.0 / cnt[s.label] for s in train], num_samples=len(train), replacement=True)
        dl_train = DataLoader(ds_train, batch_size=p.batch_size, shuffle=sampler is None, sampler=sampler,
                              collate_fn=collate, num_workers=p.num_workers)
        dl_val = DataLoader(ds_val, batch_size=max(1, p.batch_size), collate_fn=collate, num_workers=p.num_workers)

        weights = None
        if p.imbalance_strategy == "class_weights":
            weights = torch.tensor(compute_class_weights([s.label for s in train], self.labels), dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        eval_loss = torch.nn.CrossEntropyLoss()

        params = [q for q in model.parameters() if q.requires_grad]
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
        opt = torch.optim.AdamW([
            {"params": [q for n, q in named if not any(nd in n for nd in no_decay)], "weight_decay": p.weight_decay},
            {"params": [q for n, q in named if any(nd in n for nd in no_decay)], "weight_decay": 0.0}], lr=p.learning_rate)
        steps_per_epoch = max(1, math.ceil(len(dl_train) / p.gradient_accumulation))
        total_steps = steps_per_epoch * p.epochs
        sched = get_linear_schedule_with_warmup(opt, min(p.warmup_steps, max(total_steps // 5, 0)), total_steps)
        use_amp = p.fp16 and device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        state = {"epoch": 0, "best_f1": -1.0, "bad_epochs": 0, "history": [], "global_step": 0}
        ckpt = self.out_dir / "checkpoint.pt"
        if resume and ckpt.exists():
            c = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(c["model"])
            opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
            state = c["state"]
            log.info("entrenamiento reanudado", extra={"epoch": state["epoch"]})

        t0 = time.time()
        stopped = False
        for epoch in range(state["epoch"], p.epochs):
            model.train()
            run_loss, n_b = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for bi, (x, mask, y) in enumerate(dl_train):
                if self.cb.should_stop():
                    stopped = True
                    break
                x, y = x.to(device), y.to(device)
                mask = mask.to(device) if mask is not None else None
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    logits = model(x, attention_mask=mask).logits if mask is not None else model(x).logits
                loss = loss_fn(logits.float(), y) / p.gradient_accumulation
                scaler.scale(loss).backward()
                run_loss += float(loss.item()) * p.gradient_accumulation
                n_b += 1
                if (bi + 1) % p.gradient_accumulation == 0 or bi + 1 == len(dl_train):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt); scaler.update(); sched.step()
                    opt.zero_grad(set_to_none=True)
                    state["global_step"] += 1
                    self.cb.on_step({"epoch": epoch + 1, "step": state["global_step"], "total_steps": total_steps,
                                     "train_loss": run_loss / max(n_b, 1), "lr": sched.get_last_lr()[0]})
            if stopped:
                self._save_ckpt(model, opt, sched, {**state, "epoch": epoch})     # se repite esta época al continuar
                break
            m = evaluate_loader(model, dl_val, self.labels, device, eval_loss)
            rec = {"epoch": epoch + 1, "train_loss": run_loss / max(n_b, 1), "val_loss": m["loss"],
                   "val_accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                   "precision": m["precision"], "recall": m["recall"], "seconds": round(time.time() - t0, 1)}
            state["history"].append(rec)
            improved = (m["macro_f1"] or 0) > state["best_f1"] + 1e-6
            if improved:
                state["best_f1"], state["bad_epochs"] = m["macro_f1"] or 0, 0
                self._save_model(model, extractor, self.out_dir / "best")
            else:
                state["bad_epochs"] += 1
            state["epoch"] = epoch + 1
            self._save_ckpt(model, opt, sched, state)
            self.cb.on_epoch_end({**rec, "best_macro_f1": state["best_f1"], "epochs": p.epochs, "improved": improved})
            if p.early_stopping_patience and state["bad_epochs"] >= p.early_stopping_patience:
                log.info("early stopping", extra={"epoch": epoch + 1})
                break
        if stopped:
            raise StopRequested()

        # Evaluación final con el MEJOR modelo en TEST y VALIDATION
        best_dir = self.out_dir / "best"
        if best_dir.exists():
            _, model, _ = load_for_training(str(best_dir), self.labels, None, True)
            model.to(device)
        results = {"history": state["history"], "best_macro_f1": state["best_f1"], "device": device,
                   "labels": self.labels, "epochs_run": state["epoch"]}
        results["validation"] = evaluate_loader(model, dl_val, self.labels, device, eval_loss)
        if test:
            dl_test = DataLoader(AudioDataset(test, self.labels, p.max_seconds, False), batch_size=max(1, p.batch_size),
                                 collate_fn=collate)
            results["test"] = evaluate_loader(model, dl_test, self.labels, device, eval_loss)
        else:
            results["test"] = None
        if not best_dir.exists():                    # sin épocas completadas
            self._save_model(model, extractor, best_dir)
        (self.out_dir / "training_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    # -----------------------------------------------------------------------------------------------------
    def _save_model(self, model, extractor, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        extractor.save_pretrained(path)

    def _save_ckpt(self, model, opt, sched, state: dict) -> None:
        import torch
        tmp = self.out_dir / "checkpoint.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "state": state}, tmp)
        tmp.replace(self.out_dir / "checkpoint.pt")
