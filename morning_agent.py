import os
import requests
from datetime import datetime
from mistralai import Mistral

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ACCUWEATHER_API_KEY = os.getenv("ACCUWEATHER_API_KEY")

MODEL = "mistral-small-latest"


def telegram_send(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
        timeout=20
    )


def get_news():
    try:
        r = requests.get(
            "https://api.thenewsapi.com/v1/news/top",
            params={
                "api_token": NEWS_API_KEY,
                "search": "AI automation technology education jobs",
                "language": "en",
                "limit": 5
            },
            timeout=20
        )
        data = r.json().get("data", [])
        return "\n".join(
            [f"- {x.get('title')}" for x in data]
        ) or "No news found."
    except Exception:
        return "News unavailable."


def get_weather():
    # Accuweather location key should be added later
    return "Weather information unavailable. Configure location key."


def create_plan(news, weather):
    client = Mistral(api_key=MISTRAL_API_KEY)

    result = client.chat.complete(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Create a practical morning plan. Use list format. Do not give exact schedule. Give estimated duration."
            },
            {
                "role": "user",
                "content": f"Weather:\n{weather}\n\nNews:\n{news}"
            }
        ]
    )

    return result.choices[0].message.content


def main():
    news = get_news()
    weather = get_weather()

    plan = create_plan(news, weather)

    telegram_send(
        f"Good Morning ☀️\n\nDate: {datetime.now().strftime('%d-%m-%Y')}\n\n{plan}"
    )


if __name__ == "__main__":
    main()
