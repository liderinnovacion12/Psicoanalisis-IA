"""Diarización por canal (ESCENARIO A): audio estéreo con una persona por canal."""
from __future__ import annotations

from pathlib import Path

from app.ml.diarization.base import DiarizationResult, Diarizer, Turn, mark_overlaps, merge_turns


class ChannelDiarizer(Diarizer):
    name = "channels"

    def diarize(self, work_dir: Path, files: dict[str, str], vad: dict, cfg: dict, progress=None):
        turns: list[Turn] = []
        for ch in (0, 1):
            turns += [Turn(f"SPEAKER_0{ch}", s, e) for s, e in vad.get(f"ch{ch}", [])]
        seg = cfg.get("segmentation", {})
        turns = merge_turns(turns, seg.get("merge_gap", 0.8), seg.get("min_turn", 0.6))
        mark_overlaps(turns)
        dur = max((t.end for t in turns), default=0.0)
        overlap_s = sum(t.duration for t in turns if t.overlap)
        tot_s = sum(t.duration for t in turns) or 1.0
        overlap_ratio = overlap_s / tot_s
        warnings = []
        if not any(t.speaker == "SPEAKER_00" for t in turns) or not any(t.speaker == "SPEAKER_01" for t in turns):
            warnings.append("Uno de los canales no contiene voz detectable.")
        # La separación física por canal es muy fiable; se penaliza el solapamiento (posible diafonía).
        quality = round(max(0.0, min(100.0, 96 - 40 * overlap_ratio)), 1)
        return DiarizationResult(turns, self.name, 2, quality, warnings,
                                 {"overlap_ratio": round(overlap_ratio, 4), "duration": dur})
