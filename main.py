import os
import sys
import json
import html
import base64
import logging
import datetime
from email.mime.text import MIMEText
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import pytz
import requests
import uvicorn
from dotenv import load_dotenv

# Web Framework
from fastapi import FastAPI, Request, Response, status

# Telegram Bot Framework
from telegram import Update, Bot
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# AI Engine
from mistralai import Mistral

# Database ORM
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

# Background Scheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Google OAuth & Gmail API
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build

# ==============================================================================
# 1. CONFIGURATION & PRODUCTION LOGGING
# ==============================================================================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PersonalAIAssistant")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
ACCUWEATHER_API_KEY = os.getenv("ACCUWEATHER_API_KEY", "")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///assistant.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

TIMEZONE_STR = os.getenv("TIMEZONE", "Asia/Dhaka")
try:
    DHAKA_TZ = pytz.timezone(TIMEZONE_STR)
except Exception:
    DHAKA_TZ = pytz.timezone("Asia/Dhaka")

GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN", "")
GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID", "")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET", "")

PORT = int(os.getenv("PORT", 10000))

# ==============================================================================
# 2. TIMEZONE UTILITIES (STRICT ASIA/DHAKA -> UTC CONVERSION)
# ==============================================================================
def get_now_dhaka() -> datetime.datetime:
    return datetime.datetime.now(DHAKA_TZ)

def get_now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)

def to_utc_naive(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        dt = DHAKA_TZ.localize(dt)
    return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)

def to_dhaka_aware(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(DHAKA_TZ)

# ==============================================================================
# 3. DATABASE SETUP & ORM MODELS
# ==============================================================================
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    category = Column(String(100), default="General")
    priority = Column(String(50), default="Medium")
    estimated_duration = Column(String(50), nullable=True)
    status = Column(String(50), default="pending")
    created_at = Column(DateTime, default=lambda: get_now_utc().replace(tzinfo=None))
    completed_at = Column(DateTime, nullable=True)

    reminders = relationship("Reminder", back_populates="task", cascade="all, delete-orphan")

class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True)
    title = Column(String(255), nullable=False)
    category = Column(String(100), default="Personal")
    target_time = Column(DateTime, nullable=False)
    reminder_type = Column(String(50), default="one-time")
    deadline_date = Column(DateTime, nullable=True)
    repeat_pattern = Column(String(50), nullable=True)
    priority = Column(String(50), default="Medium")
    is_sent = Column(Boolean, default=False)
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=lambda: get_now_utc().replace(tzinfo=None))

    task = relationship("Task", back_populates="reminders")

class CompletedTask(Base):
    __tablename__ = "completed_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True)
    title = Column(String(255), nullable=False)
    category = Column(String(100), default="General")
    completed_at = Column(DateTime, default=lambda: get_now_utc().replace(tzinfo=None))

class DailyHistory(Base):
    __tablename__ = "daily_history"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=lambda: get_now_utc().replace(tzinfo=None))
    completed_count = Column(Integer, default=0)
    pending_count = Column(Integer, default=0)
    summary_text = Column(Text, nullable=True)

class NewsHistory(Base):
    __tablename__ = "news_history"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    source_url = Column(String(1000), nullable=True)
    published_date = Column(DateTime, default=lambda: get_now_utc().replace(tzinfo=None))

class EmailHistory(Base):
    __tablename__ = "email_history"

    id = Column(Integer, primary_key=True, index=True)
    recipient = Column(String(255), nullable=False)
    subject = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    status = Column(String(50), default="drafted")
    draft_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: get_now_utc().replace(tzinfo=None))
    sent_at = Column(DateTime, nullable=True)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization error: {e}", exc_info=True)
        raise e

# ==============================================================================
# 4. GMAIL SYSTEM (STATELESS OAUTH2 FOR CLOUD DEPLOYMENT)
# ==============================================================================
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose"
]

class GmailService:
    def __init__(self):
        self.service = None
        self._authenticate()

    def _authenticate(self):
        try:
            if GMAIL_REFRESH_TOKEN and GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET:
                creds = Credentials(
                    None,
                    refresh_token=GMAIL_REFRESH_TOKEN,
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=GMAIL_CLIENT_ID,
                    client_secret=GMAIL_CLIENT_SECRET,
                    scopes=GMAIL_SCOPES,
                )
                creds.refresh(GoogleRequest())
                self.service = build("gmail", "v1", credentials=creds)
                logger.info("Gmail API service authenticated via environment variables.")
            else:
                logger.warning("Gmail OAuth environment variables missing. Email service is offline.")
        except Exception as e:
            logger.error(f"Gmail authentication failed: {e}", exc_info=True)
            self.service = None

    def create_draft(self, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        if not self.service:
            self._authenticate()
            if not self.service:
                return {"success": False, "error": "Gmail service unauthenticated. Check credentials."}
        try:
            message = MIMEText(body)
            message["to"] = to_email
            message["subject"] = subject
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            body_payload = {"message": {"raw": raw}}
            draft = self.service.users().drafts().create(userId="me", body=body_payload).execute()
            return {"success": True, "draft_id": draft.get("id")}
        except Exception as e:
            logger.error(f"Gmail create_draft error: {e}", exc_info=True)
            return {"success": False, "error": "An error occurred while creating the Gmail draft."}

    def send_draft(self, draft_id: str) -> bool:
        if not self.service:
            self._authenticate()
            if not self.service:
                return False
        try:
            self.service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
            logger.info(f"Gmail draft {draft_id} dispatched successfully.")
            return True
        except Exception as e:
            logger.error(f"Gmail send_draft error for ID {draft_id}: {e}", exc_info=True)
            return False

    def get_important_emails(self, max_results: int = 5) -> List[Dict[str, str]]:
        if not self.service:
            self._authenticate()
            if not self.service:
                return []
        try:
            results = self.service.users().messages().list(
                userId="me", q="is:important label:INBOX", maxResults=max_results
            ).execute()
            messages = results.get("messages", [])
            summaries = []
            for msg in messages:
                m = self.service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
                headers = m.get("payload", {}).get("headers", [])
                subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject")
                sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown")
                snippet = m.get("snippet", "")
                summaries.append({"from": sender, "subject": subject, "snippet": snippet})
            return summaries
        except Exception as e:
            logger.error(f"Gmail get_important_emails error: {e}", exc_info=True)
            return []

gmail_service = GmailService()

# ==============================================================================
# 5. WEATHER SYSTEM (HTTPS ACCUWEATHER API)
# ==============================================================================
def get_weather_summary(city: str = "Dhaka") -> str:
    if not ACCUWEATHER_API_KEY:
        logger.error("ACCUWEATHER_API_KEY is not configured.")
        return "Weather service is currently unavailable."
    try:
        search_url = "https://dataservice.accuweather.com/locations/v1/cities/search"
        loc_res = requests.get(search_url, params={"apikey": ACCUWEATHER_API_KEY, "q": city}, timeout=10)
        loc_res.raise_for_status()
        loc_data = loc_res.json()
        if not loc_data:
            return f"Location '{city}' not found in weather database."

        location_key = loc_data[0]["Key"]
        cond_url = f"https://dataservice.accuweather.com/currentconditions/v1/{location_key}"
        cond_res = requests.get(cond_url, params={"apikey": ACCUWEATHER_API_KEY, "details": "true"}, timeout=10)
        cond_res.raise_for_status()
        cond_data = cond_res.json()
        if not cond_data:
            return "No weather conditions returned."

        current = cond_data[0]
        temp_c = current.get("Temperature", {}).get("Metric", {}).get("Value", "N/A")
        weather_text = current.get("WeatherText", "N/A")
        humidity = current.get("RelativeHumidity", "N/A")
        has_precip = current.get("HasPrecipitation", False)
        precip_type = current.get("PrecipitationType", "None")

        rain_info = f"Precipitation: {precip_type}" if has_precip else "No Rain"
        return f"{temp_c}°C, {weather_text} | Humidity: {humidity}% | {rain_info}"
    except requests.exceptions.RequestException as re:
        logger.error(f"AccuWeather request error: {re}")
        return "Weather service is temporarily unavailable."
    except Exception as e:
        logger.error(f"AccuWeather processing error: {e}", exc_info=True)
        return "An error occurred while fetching weather data."

# ==============================================================================
# 6. MISTRAL AI ENGINE & PROMPTS
# ==============================================================================
INTENT_PARSER_SYSTEM_PROMPT = """
You are a precise Natural Language Intent Parser engine for a Personal AI Assistant.
Translate user text into a strict JSON payload.

Supported Actions:
- create_task: Add a task to the backlog.
- create_reminder: Add a specific time-targeted alert or deadline reminder.
- list_tasks: Show active pending tasks.
- complete_task: Mark an active task as finished.
- send_email: Intent to draft or send an email.
- get_weather: Inquire about weather conditions.
- get_news: Inquire about AI/Tech/Education news.
- ask_clarification: If required information (like task name, target time, or email recipient) is missing or vague.
- unknown: When intent cannot be categorized.

Categories:
Homework, Assignment, Exam, Revision, Project, Application, Personal, Study, General

CRITICAL SAFETY RULES:
1. NEVER INVENT INFORMATION. If the user request is incomplete, return "action": "ask_clarification" and put the question in "body".
2. Reference Current Time: {current_time} (Timezone: {timezone}).
3. Always calculate absolute timestamps in "YYYY-MM-DD HH:MM" format.
4. For deadline requests, set "reminder_type": "deadline" and compute "datetime" (first reminder) and "deadline_date".
5. For repeats, set "reminder_type": "daily" or "weekly" and populate "repeat_pattern".
6. Return JSON ONLY without markdown wrapping, without backticks, and without code blocks.

Output JSON:
{{
  "action": "create_task" | "create_reminder" | "list_tasks" | "complete_task" | "send_email" | "get_weather" | "get_news" | "ask_clarification" | "unknown",
  "title": "<Extracted title, description, or subject>",
  "category": "<Category>",
  "priority": "Low" | "Medium" | "High",
  "datetime": "<YYYY-MM-DD HH:MM or null>",
  "deadline_date": "<YYYY-MM-DD HH:MM or null>",
  "reminder_type": "one-time" | "daily" | "weekly" | "deadline" | null,
  "repeat_pattern": "<pattern description or null>",
  "recipient": "<email address or null>",
  "body": "<email body content, or clarification question if ask_clarification, or null>",
  "estimated_duration": "<e.g. '2 hours', '45 minutes' or null>"
}}
"""

DAILY_PLANNER_SYSTEM_PROMPT = """
You are the Executive Daily Planner AI.
Generate a realistic, focused daily execution plan for the user based on previous performance, current weather, and pending items.

Strict Rules:
- DO NOT invent tasks. Only use the provided pending tasks.
- DO NOT assign rigid clock schedules (DO NOT write "07:00 AM - 08:00 AM").
- DO provide structured sequential priorities with estimated durations.
- If yesterday had low completions, reduce workload to prevent burnout.
- Be concise, direct, and encouraging.

Structure:
Good Morning!
[Weather Assessment]

Yesterday's Performance: [Brief summary]

Priority Plan for Today:
1. [Task Name]
   Estimated duration: [X hours/minutes]
2. [Task Name]
   Estimated duration: [X hours/minutes]

Motivational closing note.
"""

NEWS_CURATOR_SYSTEM_PROMPT = """
You are an expert AI & Technology News Editor.
Filter out all celebrity, entertainment, and clickbait items.
Keep ONLY high-impact items concerning:
- Artificial Intelligence & Machine Learning
- Automation & AI Agents
- Tech Industry Developments
- Higher Education Decisions, AI Job Market & High Demand Skills

Consolidate duplicates and select the top 3-5 most critical developments.
Do NOT invent news. Only use the provided feed.

Output Format for each item:
Title: <Clean Title>
What happened: <2-3 sentence factual summary>
Why it matters: <1-2 sentence contextual significance>
Source: <Source URL>
---
"""

class MistralAgent:
    def __init__(self):
        self.client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

    def parse_intent(self, user_message: str) -> Dict[str, Any]:
        if not self.client:
            logger.error("Mistral client is uninitialized.")
            return {"action": "unknown"}

        now_str = get_now_dhaka().strftime("%Y-%m-%d %H:%M")
        sys_prompt = INTENT_PARSER_SYSTEM_PROMPT.format(
            current_time=now_str,
            timezone=str(DHAKA_TZ)
        )
        try:
            response = self.client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            return json.loads(raw.strip())
        except Exception as e:
            logger.error(f"Mistral intent parse error: {e}", exc_info=True)
            return {"action": "unknown"}

    def generate_daily_plan(self, weather_info: str, completed_yesterday: list, pending_tasks: list) -> str:
        if not self.client:
            return "Good morning! Focus on executing your prioritized tasks today."
        prompt = f"""
        Weather: {weather_info}
        Completed Tasks Yesterday ({len(completed_yesterday)}): {[t.title for t in completed_yesterday]}
        Pending Tasks ({len(pending_tasks)}): {[f"{t.title} ({t.category})" for t in pending_tasks]}
        """
        try:
            response = self.client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": DAILY_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Mistral planner error: {e}", exc_info=True)
            return "Good morning! Focus on executing your key priorities today."

    def curate_news(self, raw_news_text: str) -> str:
        if not self.client:
            return "News curation service unavailable."
        try:
            response = self.client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": NEWS_CURATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Raw Feed Stream:\n{raw_news_text}"}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Mistral news curation error: {e}", exc_info=True)
            return "Unable to curate news items at this time."

agent = MistralAgent()

# ==============================================================================
# 7. NEWS SYSTEM (THENEWSAPI)
# ==============================================================================
def fetch_and_curate_news() -> str:
    if not NEWS_API_KEY:
        logger.error("NEWS_API_KEY is not configured.")
        return "News service is currently unavailable."
    try:
        url = "https://api.thenewsapi.com/v1/news/all"
        params = {
            "api_token": NEWS_API_KEY,
            "search": "AI OR automation OR technology OR education OR skills",
            "language": "en",
            "limit": 10,
        }
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        articles = res.json().get("data", [])
        if not articles:
            return "No relevant recent news items found."

        raw_feed = ""
        db = SessionLocal()
        try:
            for idx, a in enumerate(articles, 1):
                title = a.get("title", "No Title")
                desc = a.get("description", "")
                src_url = a.get("url", "")
                raw_feed += f"[{idx}] Title: {title}\nSummary: {desc}\nSource: {src_url}\n\n"
                
                nh = NewsHistory(title=title, summary=desc, source_url=src_url)
                db.add(nh)
            db.commit()
        except Exception as dbe:
            logger.error(f"Error persisting news history: {dbe}")
            db.rollback()
        finally:
            db.close()

        return agent.curate_news(raw_feed)
    except Exception as e:
        logger.error(f"News fetch/curate error: {e}", exc_info=True)
        return "Failed to fetch and aggregate news."

# ==============================================================================
# 8. SMART REMINDER ENGINE
# ==============================================================================
class ReminderEngine:
    @staticmethod
    def create_reminder(
        title: str,
        category: str,
        target_time: datetime.datetime,
        task_id: Optional[int] = None,
        reminder_type: str = "one-time",
        deadline_date: Optional[datetime.datetime] = None,
        repeat_pattern: Optional[str] = None,
        priority: str = "Medium",
    ) -> bool:
        if not title or not title.strip():
            logger.error("Reminder validation failed: title is empty.")
            return False
        if not target_time:
            logger.error("Reminder validation failed: target_time is missing.")
            return False

        db: Session = SessionLocal()
        try:
            target_utc_naive = to_utc_naive(target_time)
            deadline_utc_naive = to_utc_naive(deadline_date) if deadline_date else None
            now_utc_naive = get_now_utc().replace(tzinfo=None)

            if target_utc_naive < now_utc_naive:
                logger.warning("Attempted to set a reminder in the past. Ignored.")
                return False

            if reminder_type == "deadline" and deadline_utc_naive:
                offsets = [
                    (deadline_utc_naive - datetime.timedelta(days=7), "D-7"),
                    (deadline_utc_naive - datetime.timedelta(days=3), "D-3"),
                    (deadline_utc_naive - datetime.timedelta(days=1), "D-1"),
                    (deadline_utc_naive, "Deadline Day"),
                ]
                for fire_time, tag in offsets:
                    if fire_time > now_utc_naive:
                        rem = Reminder(
                            task_id=task_id,
                            title=f"[{tag}] {title.strip()}",
                            category=category,
                            target_time=fire_time,
                            reminder_type="deadline",
                            deadline_date=deadline_utc_naive,
                            priority=priority,
                            status="active",
                        )
                        db.add(rem)
            else:
                existing = db.query(Reminder).filter(
                    Reminder.task_id == task_id,
                    Reminder.title == title.strip(),
                    Reminder.target_time == target_utc_naive,
                    Reminder.status == "active"
                ).first()

                if existing:
                    logger.warning("Duplicate reminder prevented.")
                    return False

                rem = Reminder(
                    task_id=task_id,
                    title=title.strip(),
                    category=category,
                    target_time=target_utc_naive,
                    reminder_type=reminder_type,
                    deadline_date=deadline_utc_naive,
                    repeat_pattern=repeat_pattern,
                    priority=priority,
                    status="active",
                )
                db.add(rem)

            db.commit()
            return True
        except Exception as e:
            logger.error(f"Error persisting reminder: {e}", exc_info=True)
            db.rollback()
            return False
        finally:
            db.close()

    @staticmethod
    def get_pending_due_reminders() -> List[Reminder]:
        db: Session = SessionLocal()
        try:
            now_utc_naive = get_now_utc().replace(tzinfo=None)
            return (
                db.query(Reminder)
                .filter(
                    Reminder.target_time <= now_utc_naive,
                    Reminder.is_sent == False,
                    Reminder.status == "active",
                )
                .all()
            )
        finally:
            db.close()

    @staticmethod
    def post_process_reminder(reminder_id: int):
        db: Session = SessionLocal()
        try:
            rem = db.query(Reminder).filter(Reminder.id == reminder_id).first()
            if not rem:
                return
            if rem.repeat_pattern == "daily" or rem.reminder_type == "daily":
                rem.target_time = rem.target_time + datetime.timedelta(days=1)
                rem.is_sent = False
            elif rem.repeat_pattern == "weekly" or rem.reminder_type == "weekly":
                rem.target_time = rem.target_time + datetime.timedelta(weeks=1)
                rem.is_sent = False
            else:
                rem.is_sent = True
                rem.status = "completed"
            db.commit()
        except Exception as e:
            logger.error(f"Post-process reminder error for ID {reminder_id}: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()

    @staticmethod
    def cancel_reminders_by_task_id(task_id: int) -> int:
        db: Session = SessionLocal()
        try:
            rems = db.query(Reminder).filter(
                Reminder.task_id == task_id,
                Reminder.status == "active"
            ).all()
            count = len(rems)
            for r in rems:
                r.status = "cancelled"
            db.commit()
            return count
        except Exception as e:
            logger.error(f"Cancel reminders by task_id {task_id} error: {e}", exc_info=True)
            db.rollback()
            return 0
        finally:
            db.close()

# ==============================================================================
# 9. TELEGRAM BOT HANDLERS & SAFE FORMATTING
# ==============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 <b>Personal AI Assistant Online</b>\n\n"
        "<b>Commands:</b>\n"
        "/tasks - Active task list\n"
        "/reminders - Scheduled alerts\n"
        "/plan - Generate daily plan\n"
        "/news - Tech & AI curated news\n"
        "/review - Daily performance review\n"
        "/emails - Unread priority emails\n\n"
        "You can also chat in natural language (English / Bengali)."
    )
    if update.message:
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def tasks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = SessionLocal()
    try:
        tasks = db.query(Task).filter(Task.status == "pending").all()
        if not tasks:
            if update.message:
                await update.message.reply_text("📋 No pending tasks on your board.")
            return
        lines = ["📋 <b>Active Tasks:</b>"]
        for t in tasks:
            dur = f" ({html.escape(t.estimated_duration)})" if t.estimated_duration else ""
            lines.append(f"• [<b>{html.escape(t.category)}</b>] {html.escape(t.title)}{dur}")
        if update.message:
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Tasks command error: {e}", exc_info=True)
        if update.message:
            await update.message.reply_text("⚠️ Failed to retrieve tasks.")
    finally:
        db.close()

async def reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = SessionLocal()
    try:
        active_rems = db.query(Reminder).filter(Reminder.status == "active").order_by(Reminder.target_time.asc()).all()
        if not active_rems:
            if update.message:
                await update.message.reply_text("🔔 No active reminders.")
            return
        lines = ["🔔 <b>Scheduled Reminders:</b>"]
        for r in active_rems:
            loc_dt = to_dhaka_aware(r.target_time)
            lines.append(f"• {html.escape(r.title)} | <code>{loc_dt.strftime('%d-%b %I:%M %p')}</code> [<b>{html.escape(r.category)}</b>]")
        if update.message:
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Reminders command error: {e}", exc_info=True)
        if update.message:
            await update.message.reply_text("⚠️ Failed to retrieve reminders.")
    finally:
        db.close()

async def plan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("⚙️ Generating your AI Daily Plan...")
    await execute_morning_planner(context.bot)

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("📡 Fetching & curating news...")
    curated = fetch_and_curate_news()
    if update.message:
        await update.message.reply_text(curated, parse_mode=ParseMode.HTML)

async def review_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await execute_night_review(context.bot)

async def emails_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    emails = gmail_service.get_important_emails(max_results=3)
    if not emails:
        if update.message:
            await update.message.reply_text("No unread important emails found.")
        return
    lines = ["📬 <b>Important Emails:</b>\n"]
    for em in emails:
        lines.append(
            f"• <b>From:</b> {html.escape(em['from'])}\n"
            f"  <b>Subject:</b> {html.escape(em['subject'])}\n"
            f"  <b>Snippet:</b> {html.escape(em['snippet'])}\n"
        )
    if update.message:
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    db = SessionLocal()

    # Gmail Draft Confirmation Flow
    pending_draft = context.user_data.get("pending_draft")
    if pending_draft:
        if text.lower() in ["yes", "confirm", "send", "y"]:
            draft_id = pending_draft.get("draft_id")
            try:
                if gmail_service.send_draft(draft_id):
                    draft_rec = db.query(EmailHistory).filter(EmailHistory.draft_id == draft_id).first()
                    if draft_rec:
                        draft_rec.status = "sent"
                        draft_rec.sent_at = get_now_utc().replace(tzinfo=None)
                        db.commit()
                    await update.message.reply_text("🚀 Email sent successfully.")
                else:
                    await update.message.reply_text("❌ Failed to send email via Gmail API.")
            except Exception as ge:
                logger.error(f"Gmail confirmation dispatch error: {ge}", exc_info=True)
                await update.message.reply_text("❌ An error occurred during email transmission.")
            finally:
                context.user_data.pop("pending_draft", None)
                db.close()
            return
        elif text.lower() in ["no", "cancel", "stop", "n"]:
            draft_id = pending_draft.get("draft_id")
            try:
                draft_rec = db.query(EmailHistory).filter(EmailHistory.draft_id == draft_id).first()
                if draft_rec:
                    draft_rec.status = "cancelled"
                    db.commit()
                await update.message.reply_text("🚫 Email sending cancelled.")
            except Exception as ge:
                logger.error(f"Gmail cancellation error: {ge}", exc_info=True)
                await update.message.reply_text("🚫 Email operation cancelled.")
            finally:
                context.user_data.pop("pending_draft", None)
                db.close()
            return

    try:
        intent = agent.parse_intent(text)
        action = intent.get("action")

        if action == "ask_clarification":
            body = intent.get("body", "Please provide more details.")
            await update.message.reply_text(f"❓ {html.escape(body)}", parse_mode=ParseMode.HTML)

        elif action == "create_task":
            title = intent.get("title")
            if not title:
                await update.message.reply_text("⚠️ Could not determine the task title. Please specify.")
                return

            category = intent.get("category", "General")
            duration = intent.get("estimated_duration")
            priority = intent.get("priority", "Medium")

            task = Task(title=title, category=category, priority=priority, estimated_duration=duration, status="pending")
            db.add(task)
            db.commit()
            db.refresh(task)

            # Check if explicit time attached
            dt_str = intent.get("datetime")
            deadline_str = intent.get("deadline_date")
            rem_type = intent.get("reminder_type", "one-time")
            if dt_str or deadline_str:
                parsed_dt = DHAKA_TZ.localize(datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M")) if dt_str else None
                parsed_deadline = DHAKA_TZ.localize(datetime.datetime.strptime(deadline_str, "%Y-%m-%d %H:%M")) if deadline_str else None
                ReminderEngine.create_reminder(
                    title=title,
                    category=category,
                    target_time=parsed_dt or parsed_deadline,
                    task_id=task.id,
                    reminder_type=rem_type,
                    deadline_date=parsed_deadline,
                    priority=priority,
                )

            dur_str = f" (Est: {html.escape(duration)})" if duration else ""
            await update.message.reply_text(
                f"📝 Task added: <b>{html.escape(title)}</b> [<b>{html.escape(category)}</b>]{dur_str}",
                parse_mode=ParseMode.HTML
            )

        elif action == "create_reminder":
            title = intent.get("title")
            if not title:
                await update.message.reply_text("⚠️ Could not determine the reminder title. Please specify.")
                return

            category = intent.get("category", "Personal")
            priority = intent.get("priority", "Medium")
            dt_str = intent.get("datetime")
            deadline_str = intent.get("deadline_date")
            rem_type = intent.get("reminder_type", "one-time")
            pattern = intent.get("repeat_pattern")

            if not dt_str and not deadline_str:
                await update.message.reply_text("⚠️ Target time not detected. Please specify a clear date and time.")
                return

            try:
                parsed_dt = DHAKA_TZ.localize(datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M")) if dt_str else None
            except Exception:
                parsed_dt = None

            try:
                parsed_deadline = DHAKA_TZ.localize(datetime.datetime.strptime(deadline_str, "%Y-%m-%d %H:%M")) if deadline_str else None
            except Exception:
                parsed_deadline = None

            if not parsed_dt and not parsed_deadline:
                await update.message.reply_text("⚠️ Invalid date/time format received from parsing.")
                return

            task = Task(title=title, category=category, priority=priority, status="pending")
            db.add(task)
            db.commit()
            db.refresh(task)

            success = ReminderEngine.create_reminder(
                title=title,
                category=category,
                target_time=parsed_dt or parsed_deadline,
                task_id=task.id,
                reminder_type=rem_type,
                deadline_date=parsed_deadline,
                repeat_pattern=pattern,
                priority=priority,
            )
            if success:
                display_time = parsed_dt.strftime("%Y-%m-%d %I:%M %p") if parsed_dt else parsed_deadline.strftime("%Y-%m-%d")
                await update.message.reply_text(
                    f"🔔 Reminder scheduled: <b>{html.escape(title)}</b> for <code>{html.escape(display_time)}</code>",
                    parse_mode=ParseMode.HTML
                )
            else:
                await update.message.reply_text("❌ Failed to schedule reminder. Time might be in the past.")

        elif action == "complete_task":
            keyword = intent.get("title", text)
            task = db.query(Task).filter(Task.title.ilike(f"%{keyword}%"), Task.status == "pending").first()
            if task:
                task.status = "completed"
                task.completed_at = get_now_utc().replace(tzinfo=None)
                comp = CompletedTask(task_id=task.id, title=task.title, category=task.category)
                db.add(comp)
                ReminderEngine.cancel_reminders_by_task_id(task.id)
                db.commit()
                await update.message.reply_text(
                    f"✅ Marked complete: <b>{html.escape(task.title)}</b> and cleared associated reminders.",
                    parse_mode=ParseMode.HTML
                )
            else:
                await update.message.reply_text(f"Task matching '{html.escape(keyword)}' not found in active list.", parse_mode=ParseMode.HTML)

        elif action == "list_tasks":
            tasks = db.query(Task).filter(Task.status == "pending").all()
            if tasks:
                lines = ["📋 <b>Active Tasks:</b>"]
                for t in tasks:
                    lines.append(f"• [<b>{html.escape(t.category)}</b>] {html.escape(t.title)}")
                await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            else:
                await update.message.reply_text("No active tasks found.")

        elif action == "send_email":
            recipient = intent.get("recipient")
            subject = intent.get("title") or "No Subject"
            body = intent.get("body") or ""

            if not recipient:
                await update.message.reply_text("Recipient email address missing.")
                return

            draft_res = gmail_service.create_draft(recipient, subject, body)
            if draft_res.get("success"):
                draft_id = draft_res.get("draft_id")
                hist = EmailHistory(recipient=recipient, subject=subject, body=body, draft_id=draft_id, status="drafted")
                db.add(hist)
                db.commit()

                context.user_data["pending_draft"] = {"draft_id": draft_id, "recipient": recipient}
                confirm_msg = (
                    f"📧 <b>Draft Prepared in Gmail</b>\n\n"
                    f"<b>To:</b> <code>{html.escape(recipient)}</code>\n"
                    f"<b>Subject:</b> {html.escape(subject)}\n"
                    f"<b>Body:</b>\n{html.escape(body)}\n\n"
                    f"⚠️ Confirm dispatch? Reply <b>YES</b> to send or <b>NO</b> to cancel."
                )
                await update.message.reply_text(confirm_msg, parse_mode=ParseMode.HTML)
            else:
                await update.message.reply_text(f"❌ Failed to create draft: {html.escape(str(draft_res.get('error')))}", parse_mode=ParseMode.HTML)

        elif action == "get_weather":
            await update.message.reply_text(f"🌤 {get_weather_summary()}")

        elif action == "get_news":
            await update.message.reply_text("📡 Fetching updates...")
            curated = fetch_and_curate_news()
            await update.message.reply_text(curated, parse_mode=ParseMode.HTML)

        else:
            await update.message.reply_text("I could not clearly determine the action. Please try rephrasing.")

    except Exception as e:
        logger.error(f"Telegram message handler error: {e}", exc_info=True)
        await update.message.reply_text("⚠️ An error occurred processing your request. Please try again later.")
    finally:
        db.close()

def build_telegram_app() -> Optional[Application]:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN is missing in environment variables.")
        return None
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("tasks", tasks_command))
    app.add_handler(CommandHandler("reminders", reminders_command))
    app.add_handler(CommandHandler("plan", plan_command))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("review", review_command))
    app.add_handler(CommandHandler("emails", emails_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app

# ==============================================================================
# 10. AUTOMATION SCHEDULER (APSCHEDULER)
# ==============================================================================
async def execute_morning_planner(bot: Bot):
    if not TELEGRAM_CHAT_ID:
        return
    db = SessionLocal()
    try:
        yesterday_utc_naive = get_now_utc().replace(tzinfo=None) - datetime.timedelta(days=1)
        completed_yesterday = db.query(Task).filter(Task.status == "completed", Task.completed_at >= yesterday_utc_naive).all()
        pending_tasks = db.query(Task).filter(Task.status == "pending").all()
        weather_info = get_weather_summary()

        plan_content = agent.generate_daily_plan(
            weather_info=weather_info,
            completed_yesterday=completed_yesterday,
            pending_tasks=pending_tasks,
        )

        history = DailyHistory(
            completed_count=len(completed_yesterday),
            pending_count=len(pending_tasks),
            summary_text=plan_content,
        )
        db.add(history)
        db.commit()

        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=html.escape(plan_content), parse_mode=ParseMode.HTML)
        logger.info("Morning plan delivered successfully.")
    except Exception as e:
        logger.error(f"Morning planner error: {e}", exc_info=True)
    finally:
        db.close()

async def execute_news_broadcast(bot: Bot):
    if not TELEGRAM_CHAT_ID:
        return
    try:
        news_content = fetch_and_curate_news()
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🗞 <b>Daily Tech & AI Briefing</b>\n\n{news_content}", parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"News broadcast error: {e}", exc_info=True)

async def execute_night_review(bot: Bot):
    if not TELEGRAM_CHAT_ID:
        return
    db = SessionLocal()
    try:
        today_start_utc_naive = get_now_utc().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        completed_today = db.query(Task).filter(Task.status == "completed", Task.completed_at >= today_start_utc_naive).all()
        pending_tasks = db.query(Task).filter(Task.status == "pending").all()

        comp_lines = [f"• {html.escape(t.title)}" for t in completed_today] if completed_today else ["None"]
        pend_lines = [f"• {html.escape(t.title)}" for t in pending_tasks] if pending_tasks else ["None"]

        review_msg = (
            "🌙 <b>Daily Review</b>\n\n"
            f"<b>Completed:</b>\n" + "\n".join(comp_lines) + "\n\n"
            f"<b>Incomplete:</b>\n" + "\n".join(pend_lines) + "\n\n"
            f"<b>Carry Forward:</b> {len(pending_tasks)} task(s) remaining."
        )
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=review_msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Night review error: {e}", exc_info=True)
    finally:
        db.close()

async def execute_reminder_tick(bot: Bot):
    if not TELEGRAM_CHAT_ID:
        return
    try:
        due_reminders = ReminderEngine.get_pending_due_reminders()
        for r in due_reminders:
            msg = f"🔔 <b>REMINDER [{html.escape(r.category)}]</b>\n\n{html.escape(r.title)}"
            try:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.HTML)
                ReminderEngine.post_process_reminder(r.id)
            except Exception as send_err:
                logger.error(f"Failed to dispatch reminder ID {r.id}: {send_err}")
    except Exception as e:
        logger.error(f"Reminder tick error: {e}", exc_info=True)

def setup_scheduler(bot: Bot) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=DHAKA_TZ)
    # 06:00 AM Morning Planner
    scheduler.add_job(execute_morning_planner, CronTrigger(hour=6, minute=0, timezone=DHAKA_TZ), args=[bot])
    # 06:10 AM News Broadcast
    scheduler.add_job(execute_news_broadcast, CronTrigger(hour=6, minute=10, timezone=DHAKA_TZ), args=[bot])
    # 22:00 PM Night Review
    scheduler.add_job(execute_night_review, CronTrigger(hour=22, minute=0, timezone=DHAKA_TZ), args=[bot])
    # Check reminders every 60 seconds
    scheduler.add_job(execute_reminder_tick, "interval", seconds=60, args=[bot])
    return scheduler

# ==============================================================================
# 11. FASTAPI LIFESPAN & APPLICATION (RENDER COMPATIBLE)
# ==============================================================================
telegram_app: Optional[Application] = build_telegram_app()
scheduler_instance: Optional[AsyncIOScheduler] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler_instance
    init_db()

    if telegram_app:
        await telegram_app.initialize()
        await telegram_app.start()

        if WEBHOOK_URL:
            clean_url = WEBHOOK_URL.rstrip("/")
            full_webhook = f"{clean_url}/telegram/webhook"
            await telegram_app.bot.set_webhook(url=full_webhook)
            logger.info(f"Telegram webhook configured: {full_webhook}")
        else:
            logger.warning("WEBHOOK_URL not configured. Incoming webhooks won't trigger.")

        scheduler_instance = setup_scheduler(telegram_app.bot)
        scheduler_instance.start()
        logger.info("Scheduler started successfully.")
    else:
        logger.warning("TELEGRAM_BOT_TOKEN not provided. Telegram bot and Scheduler are idle.")

    yield

    if scheduler_instance and scheduler_instance.running:
        scheduler_instance.shutdown(wait=False)
    if telegram_app:
        await telegram_app.stop()
        await telegram_app.shutdown()
    logger.info("Application successfully dismantled.")

app = FastAPI(lifespan=lifespan)

@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {
        "status": "running",
        "service": "Personal AI Assistant"
    }

@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {
        "status": "healthy"
    }

@app.post("/telegram/webhook", status_code=status.HTTP_200_OK)
async def telegram_webhook(request: Request):
    if not telegram_app:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    try:
        payload = await request.json()
        update = Update.de_json(data=payload, bot=telegram_app.bot)
        if update:
            await telegram_app.process_update(update)
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Telegram webhook error: {e}", exc_info=True)
        # Always return 200 to prevent Telegram from infinitely retrying bad payloads
        return Response(status_code=status.HTTP_200_OK)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)