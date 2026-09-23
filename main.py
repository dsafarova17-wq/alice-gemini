import os
import re
from typing import Any, Dict, List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

app = FastAPI()

GEMINI_MODEL = "gemini-3.6-flash"

SYSTEM_INSTRUCTION = (
    "Ты — объективный голосовой справочник в умной колонке. Твоя единственная цель — 100% фактическая достоверность.\n"
    "Правила:\n"
    "1. Отвечай исключительно на основе общепринятых, официально подтвержденных фактов и данных. Никогда не выдумывай и не гадай.\n"
    "2. Полностью исключи личное мнение, субъективные суждения, эмоциональные оценки и оценочные прилагательные.\n"
    "3. Если точной подтвержденной информации по запросу нет или факт невозможно верифицировать, отвечай строго: 'Я не могу это подтвердить'.\n"
    "4. Не пытайся домысливать ответ, если данных недостаточно.\n"
    "5. Формат ответа: емкий разговорный стиль (2–3 полных предложения). Всегда обязательно договаривай мысль до конца и закрывай предложение точкой. Не используй списки, звездочки, решетки и служебные знаки Markdown."
)


def clean_tts_text(text: str) -> str:
    text = re.sub(r"[*#_`~>]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = text.strip()
    if len(text) > 1000:
        last_dot = text[:1000].rfind(".")
        if last_dot != -1:
            text = text[:last_dot + 1]
        else:
            text = text[:1000]
    return text


async def ask_gemini(user_prompt: str, history: List[Dict[str, str]]) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "Ошибка: переменная GEMINI_API_KEY не задана в настройках Render."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

    # Формируем цепочку сообщений: история предыдущих реплик + текущий вопрос
    contents = []
    for item in history[-4:]:
        role = item.get("role", "user")
        text = item.get("text", "")
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})

    contents.append({"role": "user", "parts": [{"text": user_prompt}]})

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 250,
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        },
    }

    try:
        async with httpx.AsyncClient(timeout=2.2) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return clean_tts_text(parts[0]["text"])
                return "Не удалось сформировать текстовый ответ."
            else:
                return f"Google {resp.status_code}: {resp.text}"
    except httpx.TimeoutException:
        return "Запрос занял слишком много времени. Пожалуйста, спросите еще раз."
    except Exception as e:
        return f"Внутренняя ошибка сервиса: {e}"


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok", "service": "alice-gemini"}


@app.post("/webhook")
async def yandex_webhook(request: Request) -> JSONResponse:
    event: Dict[str, Any] = await request.json()
    version = event.get("version", "1.0")
    session = event.get("session", {})
    is_new: bool = session.get("new", False)

    # При новом диалоге сбрасываем историю
    if is_new:
        return JSONResponse(
            content={
                "version": version,
                "session_state": {"history": []},
                "response": {
                    "text": "Фактологический справочник на связи. Задайте вопрос.",
                    "end_session": False,
                },
            }
        )

    command: str = event.get("request", {}).get("command", "").strip()

    if command.lower() in ["выход", "стоп", "хватит", "закончить", "пока", "закройся"]:
        return JSONResponse(
            content={
                "version": version,
                "session_state": {},
                "response": {"text": "Сессия завершена.", "end_session": True},
            }
        )

    if not command:
        return JSONResponse(
            content={
                "version": version,
                "response": {
                    "text": "Запрос не распознан. Повторите, пожалуйста.",
                    "end_session": False,
                },
            }
        )

    # Достаем историю из сессионного хранилища Яндекса
    state = event.get("state", {})
    session_data = state.get("session", {}) if isinstance(state, dict) else {}
    history = session_data.get("history", []) if isinstance(session_data, dict) else []

    gemini_reply = await ask_gemini(command, history)

    # Обновляем историю: сохраняем реплику пользователя и ответ модели
    updated_history = history[-4:] + [
        {"role": "user", "text": command},
        {"role": "model", "text": gemini_reply},
    ]

    return JSONResponse(
        content={
            "version": version,
            "session_state": {"history": updated_history},
            "response": {"text": gemini_reply, "end_session": False},
        }
    )
