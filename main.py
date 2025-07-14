import sys
from os import getenv
import logging

import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from agent import graph

env = load_dotenv()
TOKEN = getenv("BOT_TOKEN")

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Для того, щоб почати діалог з майстром, просто введіть повідомлення)")


@dp.message()
async def message_handler(message: Message) -> None:
    """
    Handler will forward receive a message back to the sender

    By default, message handler will handle all message types (like a text, photo, sticker etc.)
    """
    try:
        state = {"user_input": message.text}
        print("INPUT: ", message.text)
        result = graph.invoke(state, {"configurable": {"thread_id": "1"}})
        print("RESULT: ", result)
        await message.answer(result["output"])

    except Exception as e:
        import traceback
        print("❌ Exception:", e)
        traceback.print_exc()
        await message.answer("❌ Сталася помилка: " + str(e))


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
    