import os
import requests
from datetime import datetime
from mistralai.client import MistralClient

# Environment variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ACCUWEATHER_API_KEY = os.getenv("ACCUWEATHER_API_KEY")

MODEL = "mistral-small-latest"


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        },
        timeout=20
    )


def get_weather():
    try:
        # Default location can be changed later
        location_key = "Dhaka"

        url = (
            "https://dataservice.accuweather.com"
            "/forecasts/v1/daily/1day/"
            f"{location_key}"
        )

        params = {
            "apikey": ACCUWEATHER_API_KEY,
            "metric": "true"
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        if response.status_code != 200:
            return "Weather service unavailable."

        data = response.json()

        return str(data)[:500]

    except Exception:
        return "Weather unavailable."


def get_news():
    try:
        url = "https://api.thenewsapi.com/v1/news/top"

        params = {
            "api_token": NEWS_API_KEY,
            "search": "AI automation technology education jobs",
            "language": "en",
            "limit": 5
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        if response.status_code != 200:
            return "News unavailable."

        articles = response.json().get("data", [])

        result = []

        for item in articles:
            result.append(
                f"- {item.get('title')}"
            )

        return "\n".join(result)

    except Exception:
        return "News unavailable."


def create_ai_plan(weather, news):
    client = MistralClient(api_key=MISTRAL_API_KEY)

    prompt = f"""
You are a personal AI assistant.

Create a morning plan.

Include:

1. Good morning message
2. Weather summary
3. Important news
4. Today's priority tasks

Rules:

- Give list format
- Do not give exact clock schedule
- Mention estimated duration only
- Be practical

Weather:
{weather}

News:
{news}
"""

    response = client.chat.complete(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def main():
    weather = get_weather()
    news = get_news()

    plan = create_ai_plan(
        weather,
        news
    )

    message = f"""
Good Morning ☀️

Date:
{datetime.now().strftime('%d-%m-%Y')}

{plan}
"""

    send_telegram(message)


if __name__ == "__main__":
    main()
