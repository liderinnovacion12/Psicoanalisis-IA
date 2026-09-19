"""Benchmark del pipeline completo: tiempo por etapa, RTF, RAM, CPU, VRAM/GPU.

    python scripts/benchmark.py --audio ../examples/call/llamada_ejemplo_estereo.wav
    python scripts/benchmark.py --audio llamada.mp3 --repeat 40          # simula una llamada larga concatenando el audio
    python scripts/benchmark.py --audio llamada.wav --data-dir ./bench_data --keep

Real Time Factor (RTF) = tiempo_de_procesamiento / duración_del_audio  (menor es mejor; 0.17 = 10 min para 60 min).
Usa una base SQLite y una carpeta de datos temporales (no toca su instalación) salvo que use --data-dir.
Con --repeat, el audio se concatena N veces (prueba de carga/memoria, NO de calidad de resultados).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        import psutil
        self.p = psutil.Process()
        self.stop_ev = threading.Event()
        self.peak_rss = 0.0
        self.cpu = []
        self.peak_vram = 0.0
        self.gpu_util = []

    def run(self):
        import psutil
        self.p.cpu_percent(None)
        while not self.stop_ev.is_set():
            rss = self.p.memory_info().rss / 1024 ** 2
            for c in self.p.children(recursive=True):
                try:
                    rss += c.memory_info().rss / 1024 ** 2
                except Exception:
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            self.cpu.append(self.p.cpu_percent(None))
            try:
                import torch
                if torch.cuda.is_available():
                    self.peak_vram = max(self.peak_vram, torch.cuda.max_memory_allocated() / 1024 ** 2)
                    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True).stdout.strip()
                    if out:
                        self.gpu_util.append(float(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--repeat", type=int, default=1, help="concatenar el audio N veces (simular llamada larga)")
    ap.add_argument("--data-dir", help="carpeta de datos (por defecto temporal)")
    ap.add_argument("--keep", action="store_true", help="conservar los datos generados")
    ap.add_argument("--whisper", default=None, help="tamaño de modelo whisper (tiny|base|small|medium|large-v3)")
    ap.add_argument("--json", help="guardar el resultado en este archivo JSON")
    args = ap.parse_args()

    tmp = Path(args.data_dir) if args.data_dir else Path(tempfile.mkdtemp(prefix="bench_"))
    tmp.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "model_cache"
    os.environ.update({"DATABASE_URL": f"sqlite:///{tmp / 'bench.db'}", "DATA_DIR": str(tmp), "UPLOAD_DIR": str(tmp / "uploads"),
                       "TEMP_DIR": str(tmp / "tmp"), "CACHE_DIR": str(tmp / "cache"), "MODEL_CACHE_DIR": str(cache),
                       "MODELS_DIR": str(tmp / "models"), "REPORT_DIR": str(tmp / "reports"), "TASK_MODE": "inline",
                       "LOG_JSON": "false", "LOG_LEVEL": "WARNING"})
    from app.core.config import get_config_store, get_settings
    from app.core.device import detect_device
    from app.core.logging import setup_logging
    setup_logging("WARNING", False)
    s = get_settings()
    from app.database.base import SessionLocal, init_db
    from app.ml.audio import ffmpeg
    from app.ml.emotion.factory import ensure_builtin_model
    from app.models import Call, Organization, Role, User
    from app.services.call_service import create_call_from_upload
    from app.workers.pipeline import STAGES, run_stage, work_dir

    init_db()
    db = SessionLocal()
    ensure_builtin_model(db)
    org = Organization(name="bench")
    db.add(org); db.flush()
    user = User(org_id=org.id, email="bench@local", hashed_password="x", role=Role.ADMIN.value)
    db.add(user); db.commit()

    src = Path(args.audio)
    if args.repeat > 1:
        lst = tmp / "concat.txt"
        lst.write_text("".join(f"file '{src.resolve().as_posix()}'\n" for _ in range(args.repeat)), encoding="utf-8")
        cat = tmp / f"concat_{args.repeat}x{src.suffix}"
        subprocess.run([ffmpeg.ffmpeg_path(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(cat)], check=True)
        src = cat
    dur = ffmpeg.probe(src)["duration"]
    print(f"Audio    : {src.name}  ({dur / 60:.1f} min)")
    print(f"Device   : {detect_device(s.device).label}")

    if args.whisper:
        from app.services.config_service import set_section
        set_section(db, org.id, "audio", {"transcription": {"model_size": args.whisper}}, user.id)

    with open(src, "rb") as f:
        call = create_call_from_upload(db, user, src.name, f, allow_training=False, display_name="benchmark")
    sampler = Sampler()
    sampler.start()
    times = {}
    t_all = time.time()
    ok = True
    for st in STAGES:
        t0 = time.time()
        ok = run_stage(call.id, st)
        times[st] = time.time() - t0
        print(f"  {st:<18} {times[st]:8.1f}s  {'OK' if ok else 'FALLÓ'}")
        if not ok:
            break
    total = time.time() - t_all
    sampler.stop_ev.set()
    db.expire_all()
    call = db.get(Call, call.id)
    print("-" * 60)
    print(f"Estado            : {call.status}" + (f"  ({call.error_detail.splitlines()[0]})" if call.error_detail else ""))
    print(f"Audio (duración)  : {dur / 60:.2f} min")
    print(f"Processing time   : {total / 60:.2f} min ({total:.1f}s)")
    print(f"Real Time Factor  : {total / dur:.3f}")
    print(f"RAM pico (proceso): {sampler.peak_rss:.0f} MB")
    print(f"CPU media         : {sum(sampler.cpu) / max(len(sampler.cpu), 1):.0f} %")
    if sampler.gpu_util:
        print(f"GPU util. media   : {sum(sampler.gpu_util) / len(sampler.gpu_util):.0f} %   VRAM pico: {sampler.peak_vram:.0f} MB")
    else:
        print("GPU / VRAM        : n/d (modo CPU)")
    res = {"audio_seconds": dur, "processing_seconds": total, "rtf": total / dur, "stages": times,
           "peak_ram_mb": sampler.peak_rss, "status": call.status, "device": detect_device(s.device).label}
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    if not args.keep and not args.data_dir:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
