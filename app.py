"""
AVIVO Hybrid AI Bot — Telegram
Supports: /ask (Mini-RAG) + /image (Vision Captioning)
LLM: Ollama (local)
"""

import logging
import os
import tempfile
from pathlib import Path

# Load .env file if present (for local dev without exporting env vars manually)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv is optional — env vars can be exported directly

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from rag.retriever import answer_query
from vision.captioner import describe_image
from utils.cache import get_cached, set_cache
from utils.history import add_to_history, get_history

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")


# ──────────────────────────────────────────────
# /start
# ──────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to AVIVO AI Bot!*\n\n"
        "I'm a hybrid GenAI assistant. Here's what I can do:\n\n"
        "📚 `/ask <your question>` — Ask me anything from my knowledge base (RAG)\n"
        "🖼️ `/image` — Send an image after this command for AI captioning & tags\n"
        "📜 `/history` — See your last 3 interactions\n"
        "❓ `/help` — Show this help message\n\n"
        "_Powered by Ollama + sentence-transformers + BLIP_",
        parse_mode="Markdown",
    )


# ──────────────────────────────────────────────
# /help
# ──────────────────────────────────────────────
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *AVIVO AI Bot — Help*\n\n"
        "*Commands:*\n"
        "• `/ask <query>` — RAG-based Q&A from knowledge base\n"
        "• `/image` — Upload an image to get a caption + 3 keyword tags\n"
        "• `/history` — View your last 3 interactions\n"
        "• `/summarize` — Summarize your last interaction\n\n"
        "*Example:*\n"
        "`/ask What is the refund policy?`\n\n"
        "For images, send `/image` first, then upload your photo.",
        parse_mode="Markdown",
    )


# ──────────────────────────────────────────────
# /ask <query>  — Mini-RAG
# ──────────────────────────────────────────────
async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text(
            "⚠️ Please provide a question.\nExample: `/ask What is the return policy?`",
            parse_mode="Markdown",
        )
        return

    await update.message.reply_text("🔍 Searching knowledge base...")

    # Check cache first
    cached = get_cached(query)
    if cached:
        response_text = f"⚡ *(cached)*\n\n{cached['answer']}\n\n📄 *Source:* `{cached['source']}`"
        await update.message.reply_text(response_text, parse_mode="Markdown")
        add_to_history(user_id, "ask", query, cached["answer"])
        return

    try:
        result = answer_query(query)
        answer = result["answer"]
        source = result["source"]
        snippet = result["snippet"]

        # Cache it
        set_cache(query, {"answer": answer, "source": source})
        add_to_history(user_id, "ask", query, answer)

        response_text = (
            f"💬 *Answer:*\n{answer}\n\n"
            f"📄 *Source:* `{source}`\n\n"
            f"📎 *Relevant snippet:*\n_{snippet}_"
        )
        await update.message.reply_text(response_text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"RAG error: {e}")
        await update.message.reply_text(
            "❌ Something went wrong while searching. Make sure Ollama is running."
        )


# ──────────────────────────────────────────────
# /image  — set mode flag, then handle photo
# ──────────────────────────────────────────────
async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["awaiting_image"] = True
    await update.message.reply_text(
        "🖼️ Ready! Please send me an image now and I'll describe it."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)

    if not context.user_data.get("awaiting_image", False):
        await update.message.reply_text(
            "💡 Tip: Use `/image` first, then send a photo.", parse_mode="Markdown"
        )
        return

    context.user_data["awaiting_image"] = False
    await update.message.reply_text("🔬 Analyzing image...")

    try:
        # Download the highest-res photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        file_path = os.path.join(tempfile.gettempdir(), f"{photo.file_id}.jpg")
        await file.download_to_drive(file_path)

        result = describe_image(file_path)
        caption = result["caption"]
        tags = result["tags"]

        add_to_history(user_id, "image", "uploaded image", caption)

        response_text = (
            f"🖼️ *Image Caption:*\n{caption}\n\n"
            f"🏷️ *Tags:* `{'` • `'.join(tags)}`"
        )
        await update.message.reply_text(response_text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Vision error: {e}")
        await update.message.reply_text(
            "❌ Could not process the image. Make sure the vision model is available."
        )


# ──────────────────────────────────────────────
# /history  — last 3 interactions
# ──────────────────────────────────────────────
async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    history = get_history(user_id)

    if not history:
        await update.message.reply_text("📭 No history yet. Try `/ask` or `/image`!")
        return

    lines = ["📜 *Your last interactions:*\n"]
    for i, item in enumerate(history, 1):
        emoji = "📚" if item["type"] == "ask" else "🖼️"
        lines.append(f"{emoji} *{i}. {item['type'].upper()}*")
        lines.append(f"   Q: _{item['query']}_")
        lines.append(f"   A: {item['answer'][:120]}...\n")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ──────────────────────────────────────────────
# /summarize  — summarize last interaction
# ──────────────────────────────────────────────
async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    history = get_history(user_id)

    if not history:
        await update.message.reply_text("📭 Nothing to summarize yet!")
        return

    last = history[-1]
    await update.message.reply_text(
        f"📝 *Summary of last interaction:*\n\n"
        f"*Type:* {last['type'].upper()}\n"
        f"*Input:* _{last['query']}_\n"
        f"*Response summary:* {last['answer'][:200]}",
        parse_mode="Markdown",
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("summarize", summarize_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("🚀 Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
