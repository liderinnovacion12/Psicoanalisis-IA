"""Errores de dominio. Los mensajes `user_message` son los ÚNICOS que se muestran al usuario;
el detalle técnico se registra internamente (nunca se devuelve por la API)."""
from __future__ import annotations


class AppError(Exception):
    status_code = 400
    code = "app_error"
    user_message = "No fue posible completar la operación."

    def __init__(self, user_message: str | None = None, *, detail: str | None = None,
                 code: str | None = None, status_code: int | None = None):
        super().__init__(detail or user_message or self.user_message)
        if user_message:
            self.user_message = user_message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.detail = detail


class NotFound(AppError):
    status_code = 404
    code = "not_found"
    user_message = "El recurso solicitado no existe."


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"
    user_message = "No tiene permisos para realizar esta acción."


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"
    user_message = "Credenciales inválidas o sesión expirada."


class InvalidAudio(AppError):
    status_code = 422
    code = "invalid_audio"
    user_message = "No fue posible procesar el audio. Verifique que el archivo sea válido."


class Conflict(AppError):
    status_code = 409
    code = "conflict"
    user_message = "La operación no es posible en el estado actual."


class ProcessingError(AppError):
    """Error en una etapa del pipeline (se registra con detalle; el usuario ve un mensaje genérico)."""
    status_code = 500
    code = "processing_error"
    user_message = "No fue posible procesar la llamada. Puede reintentar el análisis."

    def __init__(self, stage: str, detail: str | None = None, user_message: str | None = None):
        super().__init__(user_message, detail=detail)
        self.stage = stage


class StopRequested(Exception):
    """Interrupción cooperativa (entrenamiento detenido por el usuario)."""
