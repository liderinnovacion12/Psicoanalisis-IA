"""Arquitecturas de clasificación de audio necesarias para los checkpoints del registry.

HALLAZGO (verificado al descargar el modelo base): `r-f/wav2vec-english-speech-emotion-recognition` NO usa la
cabeza de `Wav2Vec2ForSequenceClassification` (projector + classifier). Su checkpoint contiene
`classifier.dense` (1024x1024) y `classifier.out_proj` (7x1024) con pooling medio: la clase
`Wav2Vec2ForSpeechClassification` de la receta original (finetuning_task = "wav2vec2_clf").
Cargarlo con `AutoModelForAudioClassification` inicializa la cabeza AL AZAR sin fallar: las probabilidades
parecen válidas pero no significan nada. Por eso se implementa aquí la arquitectura correcta y `HFAudioEmotionModel`
verifica que no falte ninguna clave de la cabeza.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(config, "final_dropout", 0.0))
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = getattr(config, "pooling_mode", "mean")
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.post_init()

    def freeze_feature_encoder(self):
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False

    def _pool(self, hidden, attention_mask):
        if attention_mask is not None:
            fmask = self.wav2vec2._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            m = fmask.unsqueeze(-1).to(hidden.dtype)
            if self.pooling_mode == "sum":
                return (hidden * m).sum(1)
            return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        if self.pooling_mode == "sum":
            return hidden.sum(1)
        if self.pooling_mode == "max":
            return hidden.max(1)[0]
        return hidden.mean(1)

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        hidden = self.wav2vec2(input_values, attention_mask=attention_mask)[0]
        logits = self.classifier(self._pool(hidden, attention_mask))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def uses_speech_classification_head(config) -> bool:
    archs = getattr(config, "architectures", None) or []
    return "Wav2Vec2ForSpeechClassification" in archs or getattr(config, "finetuning_task", None) == "wav2vec2_clf" \
        or getattr(config, "pooling_mode", None) is not None
