import os
import re
from typing import Any, Dict, List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(title="Alice Gemini Voice Skill")

# Модель последнего поколения с минимальной задержкой
GEMINI_MODEL = "gemini-3.6-flash"

SYSTEM_INSTRUCTION = (
    "Ты — объективный голосовой справочник в умной колонке. Твоя единственная цель — 100% фактическая достоверность.\n"
    "Правила:\n"
    "1. Отвечай исключительно на основе общепринятых, проверенных и документально подтвержденных фактов. Никогда не строй догадок и гипотез.\n"
    "2. Полностью исключи личное мнение, субъективные оценки, приветствия в середине диалога и эмоциональные вводные слова (например, 'конечно', 'без проблем', 'с удовольствием').\n"
    "3. Если точной верифицированной информации нет, либо факт является спорным или непроверяемым, отвечай строго формулировкой: 'Я не могу это подтвердить'.\n"
    "4. Не пытайся домысливать ответ, если данных в запросе недостаточно.\n"
    "5. Формат ответа: связный разговорный текст (от 1 до 4 полных законченных предложений). Каждая мысль должна быть обязательно завершена точкой. Категорически запрещено использовать списки, нумерацию, звездочки, решетки, тире в начале строк, таблицы и псевдографику Markdown."
)


def format_voice_tts(raw_text: str) -> str:
    """Очищает текст от артефактов форматирования и защищает от превышения лимитов Яндекса."""
    if not raw_text:
        return "Я не могу это подтвердить."

    # Удаление Markdown-разметки: звездочки, решетки, ссылки, псевдотаблицы
    text = re.sub(r"\[.*?\]\(.*?\)", "", raw_text)
    text = re.sub(r"[*#_`~>|]", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()

    # Защита от жесткого лимита Яндекс Диалогов в 1024 символа
    if len(text) > 950:
        truncated = text[:950]
        last_punct = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if last_punct != -1:
            text = truncated[:last_punct + 1]
        else:
            text = truncated.strip() + "."

    # Гарантия закрытия предложения
    if text and text[-1] not in [".", "!", "?"]:
        text += "."

    return text


async def request_gemini_api(user_prompt: str, history: List[Dict[str, str]]) -> str:
    """Отправляет запрос к Google Generative Language API с контролем контекста и времени."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "Внутренняя ошибка конфигурации: отсутствует ключ доступа API."

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

    # Сборка цепочки контекста (до 4 последних сообщений)
    contents: List[Dict[str, Any]] = []
    for turn in history[-4:]:
        role = turn.get("role", "user")
        text = turn.get("text", "").strip()
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
            "maxOutputTokens": 300,
            # Отключение рассуждений для мгновенного голосового отклика
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }
    }

    # Таймаут 2.1 сек гарантирует, что вебхук успеет вернуть ответ до истечения 3.0 сек лимита Алисы
    try:
        async with httpx.AsyncClient(timeout=2.1) as client:
            response = await client.post(endpoint, json=payload)

            if response.status_code == 200:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates:
                    first_candidate = candidates[0]
                    content_block = first_candidate.get("content", {})
                    parts = content_block.get("parts", [])
                    if parts and "text" in parts[0]:
                        return format_voice_tts(parts[0]["text"])

                    # Проверка блокировки по соображениям безопасности или пустого ответа
                    finish_reason = first_candidate.get("finishReason", "")
                    if finish_reason in ["SAFETY", "RECITATION"]:
                        return "Я не могу дать ответ на этот вопрос в соответствии с политикой безопасности."

                return "Я не могу это подтвердить."
            else:
                return f"Внешний сервис вернул статус {response.status_code}."

    except httpx.TimeoutException:
        return "Обработка запроса заняла слишком много времени. Пожалуйста, сформулируйте вопрос иначе."
    except Exception:
        return "Произошла ошибка при обработке данных."


@app.get("/")
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Эндпоинт для поддержания активности контейнера (предотвращает засыпание Render)."""
    return {"status": "ok", "service": "alice-gemini-skill"}


@app.post("/webhook")
async def yandex_skill_webhook(request: Request) -> JSONResponse:
    event: Dict[str, Any] = await request.json()
    version = event.get("version", "1.0")
    session = event.get("session", {})
    is_new_session: bool = session.get("new", False)

    # Старт новой голосовой сессии
    if is_new_session:
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

    # Обработка пользовательской реплики
    request_data = event.get("request", {})
    command: str = request_data.get("command", "").strip()

    # Команды выхода и завершения сессии
    exit_triggers = [
        "выход", "стоп", "хватит", "закончить", "пока", "закройся",
        "отмена", "отключись", "до свидания", "прекрати"
    ]
    if command.lower() in exit_triggers:
        return JSONResponse(
            content={
                "version": version,
                "session_state": {},
                "response": {
                    "text": "Сессия завершена.",
                    "end_session": True
                },
            }
        )

    # Обработка пустых реплик и случайных пауз
    if not command:
        return JSONResponse(
            content={
                "version": version,
                "response": {
                    "text": "Запрос не распознан. Повторите, пожалуйста, ваш вопрос.",
                    "end_session": False,
                },
            }
        )

    # Извлечение истории диалога из session_state
    state = event.get("state", {})
    session_state = state.get("session", {}) if isinstance(state, dict) else {}
    history = session_state.get("history", []) if isinstance(session_state, dict) else []

    # Запрос к нейросети
    ai_answer = await request_gemini_api(command, history)

    # Сохранение текущего шага в контекст для последующих уточнений
    updated_history = history[-4:] + [
        {"role": "user", "text": command},
        {"role": "model", "text": ai_answer},
    ]

    return JSONResponse(
        content={
            "version": version,
            "session_state": {"history": updated_history},
            "response": {
                "text": ai_answer,
                "tts": ai_answer,
                "end_session": False
            },
        }
    )
