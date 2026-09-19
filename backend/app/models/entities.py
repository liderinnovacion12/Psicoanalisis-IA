"""Modelos ORM. Jerarquía: Organization → Call → Speaker → AudioSegment → Transcription/EmotionPrediction
→ SatisfactionScore → CriticalEvent. Todas las tablas de negocio llevan `org_id` (multi-tenant)."""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, JSONType, new_id, utcnow


class Role(str, enum.Enum):
    ADMIN = "ADMIN"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"


class CallStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING_AUDIO = "PROCESSING_AUDIO"
    DIARIZING = "DIARIZING"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING_EMOTIONS = "ANALYZING_EMOTIONS"
    CALCULATING_SATISFACTION = "CALCULATING_SATISFACTION"
    GENERATING_REPORT = "GENERATING_REPORT"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class ModelStatus(str, enum.Enum):
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"


class RunStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def _pk() -> Mapped[str]:
    return mapped_column(String(32), primary_key=True, default=new_id)


def _dt(**kw):
    return mapped_column(DateTime(timezone=True), default=utcnow, **kw)


class Organization(Base):
    __tablename__ = "organizations"
    id: Mapped[str] = _pk()
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = _dt()


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    hashed_password: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(16), default=Role.VIEWER.value)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = _dt()


class Call(Base):
    __tablename__ = "calls"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), index=True)
    uploaded_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    processed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)   # usuario / worker
    filename: Mapped[str] = mapped_column(String(500))
    display_name: Mapped[str | None] = mapped_column(String(300), nullable=True)
    storage_key: Mapped[str | None] = mapped_column(String(500), nullable=True)   # original
    playback_key: Mapped[str | None] = mapped_column(String(500), nullable=True)  # mp3 de reproducción
    work_prefix: Mapped[str | None] = mapped_column(String(500), nullable=True)   # audio normalizado (16 kHz)
    file_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bitrate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    language_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=CallStatus.UPLOADED.value, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)                    # 0-100
    checkpoints: Mapped[dict] = mapped_column(JSONType, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)           # SOLO interno
    audio_quality: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    analysis_quality: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    diarization_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    warnings: Mapped[list] = mapped_column(JSONType, default=list)
    summary: Mapped[dict | None] = mapped_column(JSONType, nullable=True)          # métricas agregadas
    interaction: Mapped[dict | None] = mapped_column(JSONType, nullable=True)      # métricas de interacción
    satisfaction_p1: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)   # desnormalizado (filtros)
    satisfaction_p2: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    model_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    config_snapshot: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    allow_training: Mapped[bool] = mapped_column(Boolean, default=False)
    task_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = _dt(index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    speakers: Mapped[list["Speaker"]] = relationship(back_populates="call", cascade="all, delete-orphan",
                                                     order_by="Speaker.label")


class Speaker(Base):
    __tablename__ = "speakers"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(32))            # SPEAKER_00 / SPEAKER_01
    role: Mapped[str] = mapped_column(String(32), default="other")   # client|agent|user|advisor|other
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    channel: Mapped[int | None] = mapped_column(Integer, nullable=True)
    talk_time: Mapped[float] = mapped_column(Float, default=0.0)
    call: Mapped["Call"] = relationship(back_populates="speakers")
    __table_args__ = (UniqueConstraint("call_id", "label"),)


class AudioSegment(Base):
    __tablename__ = "audio_segments"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    speaker_id: Mapped[str] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)   # channel | pyannote | spectral
    overlap: Mapped[bool] = mapped_column(Boolean, default=False)
    prosody: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    __table_args__ = (Index("ix_seg_call_start", "call_id", "start"),)


class Transcription(Base):
    __tablename__ = "transcriptions"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    speaker_id: Mapped[str] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"), index=True)
    segment_id: Mapped[str | None] = mapped_column(ForeignKey("audio_segments.id", ondelete="SET NULL"),
                                                   nullable=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_redacted: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    words: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    sentiment: Mapped[dict | None] = mapped_column(JSONType, nullable=True)    # {score,label,cues,...}
    pii: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    __table_args__ = (Index("ix_tr_call_start", "call_id", "start"),)


class EmotionPrediction(Base):
    __tablename__ = "emotion_predictions"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    segment_id: Mapped[str | None] = mapped_column(ForeignKey("audio_segments.id", ondelete="CASCADE"),
                                                   nullable=True, index=True)
    speaker_id: Mapped[str] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    duration: Mapped[float] = mapped_column(Float)
    emotion: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSONType)          # probabilidades COMPLETAS
    prosody: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    __table_args__ = (Index("ix_emo_call_speaker_start", "call_id", "speaker_id", "start"),)


class SatisfactionScore(Base):
    __tablename__ = "satisfaction_scores"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    speaker_id: Mapped[str | None] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"),
                                                   nullable=True)     # NULL = interacción global
    scope: Mapped[str] = mapped_column(String(16), default="speaker")   # speaker | interaction
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    trend: Mapped[str] = mapped_column(String(16), default="stable")   # improving|declining|stable
    initial_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict] = mapped_column(JSONType, default=dict)      # timeline, factores, métricas
    engine_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = _dt()


class CriticalEvent(Base):
    __tablename__ = "critical_events"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    speaker_id: Mapped[str | None] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"), nullable=True)
    timestamp: Mapped[float] = mapped_column(Float)
    end_timestamp: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    emotion: Mapped[str | None] = mapped_column(String(64), nullable=True)
    satisfaction: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSONType, nullable=True)


class SatisfactionFeedback(Base):
    """Satisfacción REAL (CSAT / NPS / encuesta) para calibrar la predicción."""
    __tablename__ = "satisfaction_feedback"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    speaker_id: Mapped[str | None] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"), nullable=True)
    predicted_satisfaction: Mapped[float | None] = mapped_column(Float, nullable=True)
    real_satisfaction: Mapped[float | None] = mapped_column(Float, nullable=True)   # 0-100
    csat: Mapped[float | None] = mapped_column(Float, nullable=True)
    nps: Mapped[float | None] = mapped_column(Float, nullable=True)
    survey_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = _dt()


class MLModel(Base):
    """Registro de modelos (Model Registry). kind: emotion | satisfaction."""
    __tablename__ = "models"
    id: Mapped[str] = _pk()
    org_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)  # NULL = global/builtin
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(64), default="v1")
    kind: Mapped[str] = mapped_column(String(32), default="emotion", index=True)
    loader: Mapped[str] = mapped_column(String(32), default="wav2vec")   # wav2vec | finetuned | sklearn
    base_model: Mapped[str | None] = mapped_column(String(300), nullable=True)
    hf_id: Mapped[str | None] = mapped_column(String(300), nullable=True)
    path: Mapped[str | None] = mapped_column(String(600), nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    labels: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=ModelStatus.VALIDATION.value, index=True)
    parameters: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    training_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    summary_metrics: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = _dt()
    metrics: Mapped[list["ModelMetric"]] = relationship(back_populates="model", cascade="all, delete-orphan")


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    id: Mapped[str] = _pk()
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    split: Mapped[str] = mapped_column(String(16), default="test")      # validation | test
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)   # macro
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)      # macro
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)          # macro (alias)
    macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    weighted_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    per_class: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    confusion_matrix: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    labels: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONType, nullable=True)     # métricas de regresión, etc.
    created_at: Mapped[datetime] = _dt()
    model: Mapped["MLModel"] = relationship(back_populates="metrics")


class TrainingDataset(Base):
    __tablename__ = "training_datasets"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    labels: Mapped[list | None] = mapped_column(JSONType, nullable=True)   # categorías (incluye personalizadas)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    frozen: Mapped[bool] = mapped_column(Boolean, default=False)
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    total_duration: Mapped[float] = mapped_column(Float, default=0.0)
    stats: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = _dt()
    samples: Mapped[list["TrainingSample"]] = relationship(back_populates="dataset",
                                                           cascade="all, delete-orphan")


class TrainingSample(Base):
    __tablename__ = "training_samples"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("training_datasets.id", ondelete="CASCADE"), index=True)
    call_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    audio_key: Mapped[str | None] = mapped_column(String(600), nullable=True)   # recorte 16 kHz en storage
    source_audio: Mapped[str | None] = mapped_column(String(600), nullable=True)
    speaker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    speaker_group: Mapped[str | None] = mapped_column(String(128), nullable=True)   # persona (anti-fuga)
    emotion: Mapped[str | None] = mapped_column(String(64), nullable=True)
    satisfaction: Mapped[float | None] = mapped_column(Float, nullable=True)
    start: Mapped[float] = mapped_column(Float, default=0.0)
    end: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    audio_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    split: Mapped[str | None] = mapped_column(String(16), nullable=True)   # train|validation|test
    validation: Mapped[dict | None] = mapped_column(JSONType, nullable=True)  # {ok, issues:[...]}
    meta: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = _dt()
    dataset: Mapped["TrainingDataset"] = relationship(back_populates="samples")


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    kind: Mapped[str] = mapped_column(String(32), default="emotion")
    dataset_id: Mapped[str] = mapped_column(String(32))
    base_model_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(32), nullable=True)   # modelo resultante
    params: Mapped[dict] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(String(16), default=RunStatus.PENDING.value, index=True)
    stop_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=0)
    metrics_history: Mapped[list] = mapped_column(JSONType, default=list)
    best_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(String(600), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    device: Mapped[str | None] = mapped_column(String(64), nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = _dt()


class Setting(Base):
    __tablename__ = "settings"
    id: Mapped[str] = _pk()
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[dict] = mapped_column(JSONType, default=dict)
    updated_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    __table_args__ = (UniqueConstraint("org_id", "key"),)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[str] = _pk()
    org_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    entity: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = _dt(index=True)
