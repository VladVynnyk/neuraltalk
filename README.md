# Neuraltalk

## Structure of project
1. main.py - main file of telegram bot
2. CalendarClient.py - module for Google Calendar
3. agent.py - module of AI agent
4. faq.txt - file of context for LLM

## How to setup?
1. `cd neuraltalk`
2. `python -m venv .venv`
3. `cd .venv/Scripts`
4. `activate`
5. `pip install req.txt`
6. create `.env` file and set variables: `BOT_TOKEN` and `OPENAI_API_KEY`
7. add file service-creds.json - it's file with credentials Service Account of Google
8. add calendar_id in file agent.py. This should be an email of user
9. `py main.py`

Also, you can add more information to faq.txt file, if you want to add more context