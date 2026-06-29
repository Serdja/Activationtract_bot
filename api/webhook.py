import json
import os
import tempfile
import logging
import urllib.request
import urllib.parse

from http.server import BaseHTTPRequestHandler
from markitdown import MarkItDown

logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
md = MarkItDown()

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".csv", ".json", ".xml", ".zip",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
    ".mp3", ".wav", ".m4a",
    ".txt", ".rtf", ".md",
}

START_TEXT = (
    "Привет! Отправь мне любой документ (PDF, DOCX, PPTX, XLSX, HTML, CSV, изображение и др.), "
    "и я конвертирую его в Markdown-текст.\n\n"
    "Поддерживаемые форматы:\n"
    "📄 PDF, DOCX, PPTX, XLSX\n"
    "🌐 HTML, CSV, JSON, XML\n"
    "🖼 JPG, PNG\n"
    "🎵 MP3, WAV\n"
    "📦 ZIP (обработает файлы внутри)"
)


def tg_request(method, data):
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}/{method}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.error("Telegram API error: %s", e)
        return None


def send_message(chat_id, text):
    tg_request("sendMessage", {"chat_id": chat_id, "text": text})


def send_document(chat_id, file_path, filename, caption=""):
    import requests as req_lib

    with open(file_path, "rb") as f:
        req_lib.post(
            f"{API_URL}/sendDocument",
            data={"chat_id": chat_id, "caption": caption},
            files={"document": (filename, f)},
        )


def get_file_url(file_id):
    result = tg_request("getFile", {"file_id": file_id})
    if result and result.get("ok"):
        file_path = result["result"]["file_path"]
        return f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    return None


def download_file(url, dest_path):
    urllib.request.urlretrieve(url, dest_path)


def process_file(chat_id, file_id, file_name):
    send_message(chat_id, "⏳ Обрабатываю файл...")

    ext = os.path.splitext(file_name)[1].lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        send_message(chat_id, f"Формат {ext} не поддерживается.")
        return

    file_url = get_file_url(file_id)
    if not file_url:
        send_message(chat_id, "Не удалось получить файл от Telegram.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, file_name)
        download_file(file_url, local_path)

        try:
            result = md.convert(local_path)
            text = result.text_content
        except Exception as e:
            logger.error("Conversion error: %s", e)
            send_message(chat_id, f"Ошибка при конвертации: {e}")
            return

        if not text or not text.strip():
            send_message(chat_id, "Файл пуст или не удалось извлечь текст.")
            return

        if len(text) <= 4096:
            send_message(chat_id, text)
        else:
            output_path = os.path.join(tmp_dir, f"{file_name}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            send_document(
                chat_id,
                output_path,
                f"{file_name}.md",
                "Результат слишком большой — отправляю файлом.",
            )


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode())

        try:
            update = json.loads(body)
        except json.JSONDecodeError:
            return

        message = update.get("message")
        if not message:
            return

        chat_id = message["chat"]["id"]

        if "text" in message:
            text = message["text"]
            if text.startswith("/start"):
                send_message(chat_id, START_TEXT)
            return

        if "document" in message:
            doc = message["document"]
            file_name = doc.get("file_name", "file")
            process_file(chat_id, doc["file_id"], file_name)
            return

        if "photo" in message:
            photo = message["photo"][-1]
            process_file(chat_id, photo["file_id"], "photo.jpg")
            return

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "bot": "activationtract_bot"}).encode())
