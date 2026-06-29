import os
import tempfile
import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from markitdown import MarkItDown

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

md = MarkItDown()

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".csv", ".json", ".xml", ".zip",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
    ".mp3", ".wav", ".m4a",
    ".txt", ".rtf", ".md",
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь мне любой документ (PDF, DOCX, PPTX, XLSX, HTML, CSV, изображение и др.), "
        "и я конвертирую его в Markdown-текст.\n\n"
        "Поддерживаемые форматы:\n"
        "📄 PDF, DOCX, PPTX, XLSX\n"
        "🌐 HTML, CSV, JSON, XML\n"
        "🖼 JPG, PNG\n"
        "🎵 MP3, WAV\n"
        "📦 ZIP (обработает файлы внутри)"
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    file_name = document.file_name or "file"
    ext = os.path.splitext(file_name)[1].lower()

    if ext and ext not in SUPPORTED_EXTENSIONS:
        await update.message.reply_text(
            f"Формат {ext} не поддерживается. Отправь PDF, DOCX, PPTX, XLSX, HTML, CSV или изображение."
        )
        return

    await update.message.reply_text("⏳ Обрабатываю файл...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, file_name)
        tg_file = await context.bot.get_file(document.file_id)
        await tg_file.download_to_drive(file_path)

        try:
            result = md.convert(file_path)
            text = result.text_content
        except Exception as e:
            logger.error("Conversion error: %s", e)
            await update.message.reply_text(f"Ошибка при конвертации: {e}")
            return

        if not text or not text.strip():
            await update.message.reply_text("Файл пуст или не удалось извлечь текст.")
            return

        if len(text) <= 4096:
            await update.message.reply_text(text)
        else:
            output_path = os.path.join(tmp_dir, f"{file_name}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            await update.message.reply_document(
                document=open(output_path, "rb"),
                filename=f"{file_name}.md",
                caption="Результат слишком большой для сообщения — отправляю файлом.",
            )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    await update.message.reply_text("⏳ Обрабатываю изображение...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "photo.jpg")
        tg_file = await context.bot.get_file(photo.file_id)
        await tg_file.download_to_drive(file_path)

        try:
            result = md.convert(file_path)
            text = result.text_content
        except Exception as e:
            logger.error("Conversion error: %s", e)
            await update.message.reply_text(f"Ошибка при обработке изображения: {e}")
            return

        if not text or not text.strip():
            await update.message.reply_text("Не удалось извлечь текст из изображения.")
            return

        await update.message.reply_text(text[:4096])


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Установи переменную окружения TELEGRAM_BOT_TOKEN с токеном от @BotFather"
        )

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    if os.environ.get("RENDER"):
        port = int(os.environ.get("PORT", 10000))
        hostname = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
        logger.info("Starting webhook mode on port %s", port)
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path="webhook",
            webhook_url=f"https://{hostname}/webhook",
        )
    else:
        logger.info("Starting polling mode")
        app.run_polling()


if __name__ == "__main__":
    main()
