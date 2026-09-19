"""Esquemas Pydantic (entrada y salidas tipadas principales)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Msg(BaseModel):
    message: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class RegisterIn(BaseModel):
    organization: str = Field(min_length=2, max_length=200)
    name: str = Field(default="", max_length=200)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserCreate(BaseModel):
    email: EmailStr
    name: str = ""
    password: str = Field(min_length=8, max_length=128)
    role: str = "VIEWER"


class UserUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    org_id: str
    email: str
    name: str
    role: str
    is_active: bool
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class CallPatch(BaseModel):
    display_name: str | None = None
    allow_training: bool | None = None


class SpeakerPatch(BaseModel):
    role: str | None = None
    display_name: str | None = None


class FeedbackIn(BaseModel):
    speaker: str | None = None            # SPEAKER_00 / SPEAKER_01 / None = llamada completa
    real_satisfaction: float | None = Field(default=None, ge=0, le=100)
    csat: float | None = None
    nps: float | None = None
    survey_result: str | None = None


class DatasetCreate(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    description: str | None = None
    language: str | None = None
    labels: list[str] | None = None


class SampleIn(BaseModel):
    call_id: str | None = None
    speaker: str | None = None
    emotion: str | None = None
    satisfaction: float | None = Field(default=None, ge=0, le=100)
    start: float = 0.0
    end: float | None = None
    language: str | None = None
    speaker_group: str | None = None
    meta: dict[str, Any] | None = None


class SplitIn(BaseModel):
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    seed: int = 42


class TrainingStart(BaseModel):
    dataset_id: str
    kind: str = "emotion"                      # emotion | satisfaction
    base_model_id: str | None = None
    name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    resplit: bool = False


class LabelIn(BaseModel):
    label: str = Field(min_length=1, max_length=64)


class ConfigIn(BaseModel):
    value: dict[str, Any]
