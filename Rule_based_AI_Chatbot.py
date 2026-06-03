"""
╔══════════════════════════════════════════════════════════════════╗
║         NEXUS AI — Enterprise Personal Assistant v2.0           ║
║         Refactored, Secured, Optimized & Feature-Rich           ║
╚══════════════════════════════════════════════════════════════════╝


Architecture:
  - SessionManager  : Centralised session-state bootstrap & guards
  - WeatherService  : Geocoding + forecast API wrapper
  - CommandRouter   : Intent detection + multi-step dialogue FSM
  - ToolBox         : Pure calculators / converters (no side effects)
  - DataManager     : In-memory CRUD for todos, notes, expenses
  - UIRenderer      : All Streamlit rendering helpers
  - RateLimiter     : Per-session request throttling
  - app()           : Main entry point
"""

# ─────────────────────────── stdlib ───────────────────────────────
import logging
import math
import platform
import random
import re
import string
import time
from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

# ─────────────────────────── third-party ──────────────────────────
import requests
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nexus_ai")


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
APP_NAME    = "NEXUS AI"
APP_VERSION = "2.0.0"
MAX_HISTORY = 200           # cap stored messages to prevent memory bloat
RATE_LIMIT  = 30            # max messages per minute per session
TIMEOUT_SEC = 6             # network call timeout

MOTIVATIONAL_QUOTES: List[str] = [
    "The only way to do great work is to love what you do. — Steve Jobs",
    "Quality is not an act, it is a habit. — Aristotle",
    "Precision beats power, and timing beats speed.",
    "Talk is cheap. Show me the code. — Linus Torvalds",
    "Stay hungry, stay foolish. — Steve Jobs",
    "In the middle of every difficulty lies opportunity. — Albert Einstein",
    "The secret of getting ahead is getting started. — Mark Twain",
    "It always seems impossible until it's done. — Nelson Mandela",
]

WELCOME_MSG = (
    f"👋 Welcome to **{APP_NAME}**! I am your intelligent personal assistant.\n\n"
    "Type **`help`** for a full command directory, or just start chatting!"
)

# Supported currency codes (offline fallback table vs. live API)
CURRENCY_TABLE: Dict[str, float] = {
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "PKR": 278.5,
    "INR": 83.1, "JPY": 149.8, "CAD": 1.36, "AUD": 1.53,
    "SAR": 3.75, "AED": 3.67, "CNY": 7.24, "CHF": 0.90,
}

# Habit categories
HABIT_ICONS: Dict[str, str] = {
    "exercise": "🏋️", "reading": "📚", "meditation": "🧘",
    "water": "💧", "sleep": "😴", "study": "📖", "coding": "💻",
    "diet": "🥗", "prayer": "🕌", "other": "✅",
}


# ══════════════════════════════════════════════════════════════════
# INPUT SANITISATION
# ══════════════════════════════════════════════════════════════════
def sanitise(text: str, max_len: int = 500) -> str:
    """
    Strip leading/trailing whitespace, remove control characters,
    and cap length to prevent injection or oversized payloads.
    """
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned.strip()[:max_len]


def safe_float(value: str) -> Tuple[bool, float]:
    """
    Attempt to parse a float. Returns (success, number).
    Blocks NaN and Infinity to prevent downstream math issues.
    """
    try:
        num = float(value.replace(",", "."))
        if not math.isfinite(num):
            return False, 0.0
        return True, num
    except (ValueError, AttributeError):
        return False, 0.0


# ══════════════════════════════════════════════════════════════════
# RATE LIMITER
# ══════════════════════════════════════════════════════════════════
class RateLimiter:
    """
    Sliding-window rate limiter stored in session state.
    Allows up to `limit` requests per 60-second window.
    """

    @staticmethod
    def _key() -> str:
        return "rl_timestamps"

    @classmethod
    def check(cls) -> bool:
        """Return True if request is allowed, False if rate-limited."""
        now = time.time()
        key = cls._key()
        if key not in st.session_state:
            st.session_state[key] = []
        # Keep only timestamps within the last 60 seconds
        window = [t for t in st.session_state[key] if now - t < 60]
        if len(window) >= RATE_LIMIT:
            st.session_state[key] = window
            return False
        window.append(now)
        st.session_state[key] = window
        return True


# ══════════════════════════════════════════════════════════════════
# SESSION MANAGER
# ══════════════════════════════════════════════════════════════════
class SessionManager:
    """
    Centralised session-state initialisation.
    All keys are typed and default-valued here — one source of truth.
    """

    DEFAULTS: Dict[str, Any] = {
        # Chat
        "messages":        [],
        "theme":           "dark",
        # Multi-step dialogue state machine
        "dialogue_mode":   None,   # str: current FSM node
        "dialogue_data":   {},     # dict: scratch data for current FSM
        # Productivity data
        "todos":           [],     # List[dict]: {text, done, created}
        "notes":           [],     # List[dict]: {text, created, tag}
        "expenses":        [],     # List[dict]: {amount, category, note, date}
        "habits":          {},     # Dict[str, List[date]]: habit → dates checked
        "goals":           [],     # List[dict]: {text, target, progress, unit}
        "pomodoro_start":  None,   # float | None: epoch of current session
        "pomodoro_count":  0,
        # Stats
        "total_messages":  0,
        "session_start":   datetime.now().isoformat(),
    }

    @classmethod
    def init(cls) -> None:
        for key, value in cls.DEFAULTS.items():
            if key not in st.session_state:
                # Deep-copy mutable defaults to avoid shared references
                if isinstance(value, (dict, list)):
                    st.session_state[key] = type(value)(value)
                else:
                    st.session_state[key] = value

        # Inject welcome message once
        if not st.session_state["messages"]:
            cls.push_bot(WELCOME_MSG)

    @staticmethod
    def push_user(text: str) -> None:
        msg = {"role": "user", "content": text, "ts": datetime.now().isoformat()}
        st.session_state.messages.append(msg)
        # Trim history
        if len(st.session_state.messages) > MAX_HISTORY:
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
        st.session_state.total_messages += 1

    @staticmethod
    def push_bot(text: str) -> None:
        msg = {"role": "assistant", "content": text, "ts": datetime.now().isoformat()}
        st.session_state.messages.append(msg)
        if len(st.session_state.messages) > MAX_HISTORY:
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    @staticmethod
    def reset_dialogue() -> None:
        st.session_state.dialogue_mode = None
        st.session_state.dialogue_data = {}

    @staticmethod
    def clear_chat() -> None:
        st.session_state.messages = []
        SessionManager.push_bot("🗑️ Chat cleared. Fresh start!")


# ══════════════════════════════════════════════════════════════════
# WEATHER SERVICE
# ══════════════════════════════════════════════════════════════════
class WeatherService:
    """
    Wraps Open-Meteo (free, no API key) geocoding + forecast endpoints.
    WMO weather-code descriptions included for human-readable output.
    """

    GEO_URL     = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    WMO_CODES: Dict[int, str] = {
        0: "☀️ Clear sky", 1: "🌤️ Mainly clear", 2: "⛅ Partly cloudy",
        3: "☁️ Overcast", 45: "🌫️ Foggy", 48: "🌫️ Icy fog",
        51: "🌦️ Light drizzle", 53: "🌦️ Moderate drizzle", 55: "🌧️ Dense drizzle",
        61: "🌧️ Slight rain", 63: "🌧️ Moderate rain", 65: "🌧️ Heavy rain",
        71: "🌨️ Slight snow", 73: "🌨️ Moderate snow", 75: "❄️ Heavy snow",
        80: "🌦️ Rain showers", 81: "🌧️ Moderate showers", 82: "⛈️ Violent showers",
        95: "⛈️ Thunderstorm", 96: "⛈️ Thunderstorm + hail", 99: "⛈️ Severe thunderstorm",
    }

    @st.cache_data(ttl=600, show_spinner=False)
    def fetch(_self, city: str) -> str:
        """Fetch current weather for a city. Results cached for 10 minutes."""
        city = sanitise(city, 100)
        if not city:
            return "⚠️ Please provide a valid city name."
        try:
            geo_resp = requests.get(
                _self.GEO_URL,
                params={"name": city, "count": 1, "language": "en"},
                timeout=TIMEOUT_SEC,
            )
            geo_resp.raise_for_status()
            results = geo_resp.json().get("results", [])
            if not results:
                return f"❌ City **'{city}'** not found. Please check the spelling."

            loc     = results[0]
            lat     = loc["latitude"]
            lon     = loc["longitude"]
            country = loc.get("country", "N/A")
            tz      = loc.get("timezone", "auto")

            w_resp = requests.get(
                _self.WEATHER_URL,
                params={
                    "latitude": lat, "longitude": lon,
                    "current_weather": True,
                    "hourly": "relativehumidity_2m,uv_index",
                    "daily": "sunrise,sunset,precipitation_sum",
                    "timezone": tz,
                },
                timeout=TIMEOUT_SEC,
            )
            w_resp.raise_for_status()
            data = w_resp.json()
            cw   = data["current_weather"]
            temp = cw.get("temperature", "N/A")
            wind = cw.get("windspeed", "N/A")
            wcode= int(cw.get("weathercode", 0))
            desc = _self.WMO_CODES.get(wcode, "🌡️ Unknown conditions")

            # Extract first hourly humidity reading
            hourly = data.get("hourly", {})
            humidity = hourly.get("relativehumidity_2m", [None])[0]
            uv       = hourly.get("uv_index", [None])[0]

            daily  = data.get("daily", {})
            rise   = daily.get("sunrise",  ["N/A"])[0]
            sset   = daily.get("sunset",   ["N/A"])[0]
            precip = daily.get("precipitation_sum", ["N/A"])[0]

            lines = [
                f"🌍 **{city.title()}, {country}** — Live Weather Report",
                f"{'─'*40}",
                f"🌡️ **Temperature:** {temp}°C",
                f"💨 **Wind Speed:** {wind} km/h",
                f"🌤️ **Condition:** {desc}",
            ]
            if humidity is not None:
                lines.append(f"💧 **Humidity:** {humidity}%")
            if uv is not None:
                lines.append(f"☀️ **UV Index:** {uv}")
            lines += [
                f"🌧️ **Precipitation (today):** {precip} mm",
                f"🌅 **Sunrise:** {rise}",
                f"🌇 **Sunset:** {sset}",
                f"🕒 *Updated: {datetime.now().strftime('%I:%M %p')}*",
            ]
            return "\n".join(lines)

        except requests.exceptions.ConnectionError:
            logger.warning("Weather fetch failed: no network.")
            return "🔌 Network unavailable. Please check your internet connection."
        except requests.exceptions.Timeout:
            logger.warning("Weather fetch timed out.")
            return "⏱️ Request timed out. The weather API is slow — try again shortly."
        except requests.exceptions.HTTPError as exc:
            logger.error("Weather HTTP error: %s", exc)
            return f"🚫 Weather API returned an error: {exc}"
        except (KeyError, ValueError, IndexError) as exc:
            logger.error("Weather parse error: %s", exc)
            return "⚠️ Unexpected response from the weather service. Please try again."


weather_svc = WeatherService()


# ══════════════════════════════════════════════════════════════════
# TOOLBOX  (pure functions — no Streamlit, no side effects)
# ══════════════════════════════════════════════════════════════════
class ToolBox:
    """
    Stateless utility methods. All inputs already validated before entry.
    """

    # ── Maths ──────────────────────────────────────────────────────
    @staticmethod
    def calculate(op: str, a: float, b: float) -> str:
        ops = {
            "sum":  ("Sum",       lambda x, y: x + y),
            "sub":  ("Difference",lambda x, y: x - y),
            "mul":  ("Product",   lambda x, y: x * y),
            "div":  ("Quotient",  lambda x, y: x / y),
            "pow":  ("Power",     lambda x, y: x ** y),
            "mod":  ("Remainder", lambda x, y: x % y),
        }
        if op not in ops:
            return "⚠️ Unknown operation."
        label, fn = ops[op]
        if op == "div" and b == 0:
            return "❌ Division by zero is undefined."
        if op == "mod" and b == 0:
            return "❌ Modulo by zero is undefined."
        try:
            result = fn(a, b)
            # Avoid ugly floats like 3.0 when result is whole
            fmt = int(result) if result == int(result) and abs(result) < 1e15 else f"{result:,.6g}"
            return f"🧮 **{label}:** `{a} → {b}` = **{fmt}**"
        except OverflowError:
            return "⚠️ Result is too large to compute."

    # ── BMI ────────────────────────────────────────────────────────
    @staticmethod
    def bmi(weight_kg: float, height_cm: float) -> str:
        if weight_kg <= 0 or height_cm <= 0:
            return "❌ Weight and height must be positive numbers."
        h_m  = height_cm / 100
        bmi  = weight_kg / (h_m ** 2)
        if   bmi < 18.5: cat = "🔵 Underweight"
        elif bmi < 25.0: cat = "🟢 Normal weight"
        elif bmi < 30.0: cat = "🟡 Overweight"
        else:            cat = "🔴 Obese"
        return (
            f"⚖️ **BMI Result**\n"
            f"• BMI Score: **{bmi:.1f}**\n"
            f"• Category: {cat}\n"
            f"• *(WHO classification)*"
        )

    # ── Age ────────────────────────────────────────────────────────
    @staticmethod
    def age(dob_str: str) -> str:
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"):
            try:
                dob   = datetime.strptime(dob_str.strip(), fmt).date()
                today = date.today()
                if dob > today:
                    return "❌ Date of birth cannot be in the future."
                years  = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                months = (today.month - dob.month) % 12
                days   = (today - dob.replace(year=today.year)).days % 30
                return (
                    f"🎂 **Age Calculator**\n"
                    f"• **{years} years**, {months} months, {days} days old\n"
                    f"• Born: {dob.strftime('%d %B %Y')}"
                )
            except ValueError:
                continue
        return "❌ Invalid date. Use formats like `25-04-1995` or `25/04/1995`."

    # ── Percentage ─────────────────────────────────────────────────
    @staticmethod
    def percentage(part: float, total: float) -> str:
        if total == 0:
            return "❌ Total cannot be zero."
        pct = (part / total) * 100
        return f"📊 **{part}** is **{pct:.2f}%** of **{total}**"

    # ── Loan EMI ───────────────────────────────────────────────────
    @staticmethod
    def emi(principal: float, annual_rate: float, months: int) -> str:
        if principal <= 0 or months <= 0:
            return "❌ Principal and tenure must be positive."
        if annual_rate < 0:
            return "❌ Interest rate cannot be negative."
        if annual_rate == 0:
            emi = principal / months
            return (
                f"🏦 **EMI Calculator (0% interest)**\n"
                f"• Monthly Payment: **{emi:,.2f}**\n"
                f"• Total Payment:   **{principal:,.2f}**"
            )
        r = annual_rate / (12 * 100)
        emi = (principal * r * (1 + r) ** months) / ((1 + r) ** months - 1)
        total = emi * months
        interest = total - principal
        return (
            f"🏦 **EMI Calculator**\n"
            f"• Monthly EMI:       **{emi:,.2f}**\n"
            f"• Total Payment:     **{total:,.2f}**\n"
            f"• Total Interest:    **{interest:,.2f}**\n"
            f"• Loan Tenure:       {months} months"
        )

    # ── Compound Interest ──────────────────────────────────────────
    @staticmethod
    def compound_interest(p: float, r: float, n: float, t: float) -> str:
        if p <= 0 or n <= 0 or t <= 0:
            return "❌ Principal, compounding frequency, and time must be positive."
        if r < 0:
            return "❌ Rate cannot be negative."
        amount   = p * (1 + r / (n * 100)) ** (n * t)
        interest = amount - p
        return (
            f"💰 **Compound Interest**\n"
            f"• Principal:   {p:,.2f}\n"
            f"• Rate:        {r}% p.a.\n"
            f"• Compounded:  {int(n)}× per year × {t} years\n"
            f"• **Final Amount:  {amount:,.2f}**\n"
            f"• **Interest Earned: {interest:,.2f}**"
        )

    # ── GPA ────────────────────────────────────────────────────────
    @staticmethod
    def gpa(grades_str: str) -> str:
        """
        Accept comma-separated grade points, e.g. '3.5, 4.0, 2.7, 3.8'.
        Returns weighted average GPA.
        """
        parts = [g.strip() for g in grades_str.split(",") if g.strip()]
        if not parts:
            return "❌ Please enter at least one grade point."
        valid = []
        for p in parts:
            ok, val = safe_float(p)
            if not ok or not (0.0 <= val <= 4.0):
                return f"❌ Invalid grade point: **{p}**. All values must be between 0.0 and 4.0."
            valid.append(val)
        gpa = sum(valid) / len(valid)
        if   gpa >= 3.7: grade = "A+ (Distinction)"
        elif gpa >= 3.3: grade = "A"
        elif gpa >= 3.0: grade = "A−"
        elif gpa >= 2.7: grade = "B+"
        elif gpa >= 2.3: grade = "B"
        elif gpa >= 2.0: grade = "B−"
        else:            grade = "Below B−"
        return f"🎓 **GPA:** {gpa:.2f} / 4.00 — {grade}"

    # ── Unit Converter ─────────────────────────────────────────────
    UNIT_CONVERSIONS: Dict[str, Tuple[str, str, float]] = {
        "c_to_f":   ("°C", "°F",  0.0),   # special-cased
        "f_to_c":   ("°F", "°C",  0.0),
        "kg_to_lb": ("kg", "lbs", 2.20462),
        "lb_to_kg": ("lbs","kg",  0.45359),
        "m_to_ft":  ("m",  "ft",  3.28084),
        "ft_to_m":  ("ft", "m",   0.30480),
        "km_to_mi": ("km", "mi",  0.62137),
        "mi_to_km": ("mi", "km",  1.60934),
        "l_to_gal": ("L",  "gal", 0.26417),
        "gal_to_l": ("gal","L",   3.78541),
    }

    @classmethod
    def convert_unit(cls, conv_key: str, value: float) -> str:
        if conv_key not in cls.UNIT_CONVERSIONS:
            return "⚠️ Unknown conversion type."
        from_u, to_u, factor = cls.UNIT_CONVERSIONS[conv_key]
        if conv_key == "c_to_f":
            result = (value * 9 / 5) + 32
        elif conv_key == "f_to_c":
            result = (value - 32) * 5 / 9
        else:
            result = value * factor
        return f"📐 **{value} {from_u}** = **{result:.4g} {to_u}**"

    # ── Currency Converter ─────────────────────────────────────────
    @staticmethod
    def convert_currency(amount: float, from_cur: str, to_cur: str) -> str:
        from_cur = from_cur.upper()
        to_cur   = to_cur.upper()
        if from_cur not in CURRENCY_TABLE:
            return f"❌ Currency **{from_cur}** not supported. Try: {', '.join(CURRENCY_TABLE)}"
        if to_cur not in CURRENCY_TABLE:
            return f"❌ Currency **{to_cur}** not supported. Try: {', '.join(CURRENCY_TABLE)}"
        usd_amount = amount / CURRENCY_TABLE[from_cur]
        result     = usd_amount * CURRENCY_TABLE[to_cur]
        return f"💱 **{amount:,.2f} {from_cur}** = **{result:,.2f} {to_cur}** *(offline rates)*"

    # ── Password Generator ─────────────────────────────────────────
    @staticmethod
    def generate_password(length: int = 16) -> str:
        length = max(8, min(64, length))
        # Guarantee at least one of each character class
        pool = (
            random.choice(string.ascii_lowercase)
            + random.choice(string.ascii_uppercase)
            + random.choice(string.digits)
            + random.choice("!@#$%^&*-_+=?")
            + "".join(
                random.choices(
                    string.ascii_letters + string.digits + "!@#$%^&*-_+=?",
                    k=length - 4,
                )
            )
        )
        chars = list(pool)
        random.shuffle(chars)
        return "".join(chars)

    # ── System Info ────────────────────────────────────────────────
    @staticmethod
    def system_info() -> str:
        try:
            import psutil  # optional — graceful fallback if absent
            cpu  = psutil.cpu_percent(interval=0.3)
            ram  = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            ram_used  = f"{ram.used  / 1e9:.1f} GB / {ram.total / 1e9:.1f} GB ({ram.percent}%)"
            disk_used = f"{disk.used / 1e9:.1f} GB / {disk.total/ 1e9:.1f} GB ({disk.percent}%)"
            cpu_info  = f"{cpu}%"
        except ImportError:
            cpu_info  = "psutil not installed"
            ram_used  = "psutil not installed"
            disk_used = "psutil not installed"

        return (
            f"💻 **System Diagnostics**\n"
            f"• OS:           {platform.system()} {platform.release()}\n"
            f"• Architecture: {platform.machine()}\n"
            f"• Processor:    {platform.processor() or 'N/A'}\n"
            f"• Python:       {platform.python_version()}\n"
            f"• CPU Usage:    {cpu_info}\n"
            f"• RAM:          {ram_used}\n"
            f"• Disk:         {disk_used}"
        )


# ══════════════════════════════════════════════════════════════════
# DATA MANAGER
# ══════════════════════════════════════════════════════════════════
class DataManager:
    """CRUD helpers for todos, notes, expenses, habits, goals."""

    # ── Todos ──────────────────────────────────────────────────────
    @staticmethod
    def add_todo(text: str) -> str:
        text = sanitise(text, 200)
        if not text:
            return "❌ Task text cannot be empty."
        st.session_state.todos.append(
            {"text": text, "done": False, "created": datetime.now().isoformat()}
        )
        return f"✅ Task added: **{text}** *(#{len(st.session_state.todos)})*"

    @staticmethod
    def list_todos() -> str:
        todos = st.session_state.todos
        if not todos:
            return "📋 Your to-do list is empty. Add tasks with `todo add`."
        lines = ["📋 **Your To-Do List:**\n"]
        for i, t in enumerate(todos, 1):
            mark = "✅" if t["done"] else "⬜"
            lines.append(f"{mark} **{i}.** {t['text']}")
        done_count = sum(1 for t in todos if t["done"])
        lines.append(f"\n*{done_count}/{len(todos)} completed*")
        return "\n".join(lines)

    @staticmethod
    def toggle_todo(idx: int) -> str:
        todos = st.session_state.todos
        if not (1 <= idx <= len(todos)):
            return f"❌ Invalid task number. You have {len(todos)} tasks."
        todos[idx - 1]["done"] = not todos[idx - 1]["done"]
        state = "✅ Done" if todos[idx - 1]["done"] else "⬜ Pending"
        return f"{state}: **{todos[idx - 1]['text']}**"

    @staticmethod
    def remove_todo(query: str) -> str:
        todos = st.session_state.todos
        if not todos:
            return "📋 Your to-do list is already empty."
        query = query.strip()
        # Try numeric index first
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(todos):
                removed = todos.pop(idx)
                return f"🗑️ Removed task: **{removed['text']}**"
            return f"❌ No task at position **{query}**."
        # Fuzzy name match
        q_lower = query.lower()
        for i, t in enumerate(todos):
            if q_lower in t["text"].lower():
                removed = todos.pop(i)
                return f"🗑️ Removed task: **{removed['text']}**"
        return f"❌ No task matching **'{query}'** found."

    # ── Notes ──────────────────────────────────────────────────────
    @staticmethod
    def add_note(text: str, tag: str = "general") -> str:
        text = sanitise(text, 1000)
        if not text:
            return "❌ Note cannot be empty."
        st.session_state.notes.append(
            {"text": text, "tag": tag, "created": datetime.now().strftime("%d %b %Y %I:%M %p")}
        )
        return f"📌 Note saved! *(#{len(st.session_state.notes)})*"

    @staticmethod
    def list_notes() -> str:
        notes = st.session_state.notes
        if not notes:
            return "📝 No notes saved yet. Use `save note` to add one."
        lines = [f"📝 **Saved Notes ({len(notes)}):**\n"]
        for i, n in enumerate(notes, 1):
            lines.append(f"**{i}.** 📌 {n['text']}\n   *{n['created']}  •  [{n.get('tag','general')}]*")
        return "\n".join(lines)

    @staticmethod
    def delete_note(idx: int) -> str:
        notes = st.session_state.notes
        if not (1 <= idx <= len(notes)):
            return f"❌ Invalid note number. You have {len(notes)} notes."
        removed = notes.pop(idx - 1)
        return f"🗑️ Deleted note: **{removed['text'][:60]}...**"

    # ── Expenses ───────────────────────────────────────────────────
    @staticmethod
    def add_expense(amount: float, category: str = "General", note: str = "") -> str:
        if amount <= 0:
            return "❌ Amount must be greater than zero."
        category = sanitise(category, 50) or "General"
        note     = sanitise(note, 200)
        st.session_state.expenses.append({
            "amount":   amount,
            "category": category,
            "note":     note,
            "date":     date.today().isoformat(),
        })
        total = sum(e["amount"] for e in st.session_state.expenses)
        return (
            f"💸 Expense logged: **{amount:,.2f}** [{category}]\n"
            f"   Note: {note or '—'}\n"
            f"   Running total: **{total:,.2f}**"
        )

    @staticmethod
    def view_expenses() -> str:
        expenses = st.session_state.expenses
        if not expenses:
            return "💳 No expenses logged yet. Use `expense add` to start tracking."
        total = sum(e["amount"] for e in expenses)
        # Group by category
        cats: Dict[str, float] = {}
        for e in expenses:
            cats[e["category"]] = cats.get(e["category"], 0) + e["amount"]
        lines = [f"💳 **Expense Report ({len(expenses)} entries)**\n"]
        for cat, amt in sorted(cats.items(), key=lambda x: -x[1]):
            pct = (amt / total) * 100
            lines.append(f"• {cat}: **{amt:,.2f}** ({pct:.1f}%)")
        lines.append(f"\n💰 **Total Spent: {total:,.2f}**")
        return "\n".join(lines)

    # ── Habits ─────────────────────────────────────────────────────
    @staticmethod
    def check_habit(name: str) -> str:
        name  = sanitise(name, 50).lower()
        if not name:
            return "❌ Habit name cannot be empty."
        today = date.today().isoformat()
        habits = st.session_state.habits
        if name not in habits:
            habits[name] = []
        if today in habits[name]:
            return f"⚡ You already checked off **{name}** today!"
        habits[name].append(today)
        streak = DataManager._streak(habits[name])
        icon   = HABIT_ICONS.get(name, "✅")
        return (
            f"{icon} **{name.title()}** checked for today!\n"
            f"   🔥 Current streak: **{streak} day{'s' if streak != 1 else ''}**"
        )

    @staticmethod
    def _streak(dates: List[str]) -> int:
        if not dates:
            return 0
        sorted_dates = sorted(set(dates), reverse=True)
        streak = 0
        check  = date.today()
        for d in sorted_dates:
            if date.fromisoformat(d) == check:
                streak += 1
                check  -= timedelta(days=1)
            else:
                break
        return streak

    @staticmethod
    def view_habits() -> str:
        habits = st.session_state.habits
        if not habits:
            return "🏋️ No habits tracked yet. Type `habit <name>` to start."
        lines = ["🏋️ **Habit Tracker**\n"]
        today = date.today().isoformat()
        for name, dates in habits.items():
            icon    = HABIT_ICONS.get(name, "✅")
            streak  = DataManager._streak(dates)
            checked = "✔️" if today in dates else "⬜"
            lines.append(
                f"{checked} {icon} **{name.title()}**  —  "
                f"🔥 {streak}-day streak  •  {len(dates)} total days"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# COMMAND ROUTER
# ══════════════════════════════════════════════════════════════════
class CommandRouter:
    """
    Two-layer routing:
      1. Multi-step dialogue FSM (handles ongoing conversations)
      2. Intent matcher (keyword / regex triggers → handler functions)

    All handlers receive `raw` (original casing) and `cmd` (lowercase),
    return a reply string.  Side-effects on session_state are allowed.
    """

    # ── Intent map: keyword → (handler, priority) ─────────────────
    # Checked in ascending priority order (lower = checked first)
    # Tuple: (match_fn, handler_fn)

    def route(self, raw: str) -> str:
        """Main routing entry point."""
        cmd = raw.lower().strip()

        # ── 1. Global override commands ────────────────────────────
        if cmd in ("clear", "reset", "cls"):
            SessionManager.clear_chat()
            SessionManager.reset_dialogue()
            return ""   # push_bot already called inside clear_chat

        if cmd in ("exit", "quit", "bye", "goodbye"):
            return "👋 Goodbye! Come back anytime. Have a great day!"

        # ── 2. Multi-step FSM ──────────────────────────────────────
        mode = st.session_state.dialogue_mode
        if mode:
            return self._handle_dialogue(raw, cmd, mode)

        # ── 3. Intent matching ─────────────────────────────────────
        return self._match_intent(raw, cmd)

    # ──────────────────────────────────────────────────────────────
    # DIALOGUE FSM
    # ──────────────────────────────────────────────────────────────
    def _handle_dialogue(self, raw: str, cmd: str, mode: str) -> str:
        data = st.session_state.dialogue_data

        # ── Math ───────────────────────────────────────────────────
        if mode == "math_a":
            ok, val = safe_float(cmd)
            if not ok:
                return "❌ Please enter a valid number for the first operand."
            data["a"] = val
            st.session_state.dialogue_mode = "math_b"
            return "🔢 Enter the **second number**:"

        if mode == "math_b":
            ok, val = safe_float(cmd)
            if not ok:
                return "❌ Please enter a valid number for the second operand."
            result = ToolBox.calculate(data["op"], data["a"], val)
            SessionManager.reset_dialogue()
            return result

        # ── BMI ────────────────────────────────────────────────────
        if mode == "bmi_weight":
            ok, val = safe_float(cmd)
            if not ok or val <= 0:
                return "❌ Enter a positive weight in kilograms, e.g. `70`."
            data["weight"] = val
            st.session_state.dialogue_mode = "bmi_height"
            return "📏 Enter your **height in cm**, e.g. `175`:"

        if mode == "bmi_height":
            ok, val = safe_float(cmd)
            if not ok or val <= 0:
                return "❌ Enter a positive height in cm, e.g. `175`."
            result = ToolBox.bmi(data["weight"], val)
            SessionManager.reset_dialogue()
            return result

        # ── Age ────────────────────────────────────────────────────
        if mode == "age_dob":
            result = ToolBox.age(cmd)
            SessionManager.reset_dialogue()
            return result

        # ── Percentage ─────────────────────────────────────────────
        if mode == "pct_part":
            ok, val = safe_float(cmd)
            if not ok:
                return "❌ Enter a valid number for the part."
            data["part"] = val
            st.session_state.dialogue_mode = "pct_total"
            return "🔢 Enter the **total** (the whole amount):"

        if mode == "pct_total":
            ok, val = safe_float(cmd)
            if not ok:
                return "❌ Enter a valid number for the total."
            result = ToolBox.percentage(data["part"], val)
            SessionManager.reset_dialogue()
            return result

        # ── EMI ────────────────────────────────────────────────────
        if mode == "emi_p":
            ok, val = safe_float(cmd)
            if not ok or val <= 0:
                return "❌ Enter a positive loan amount."
            data["p"] = val
            st.session_state.dialogue_mode = "emi_r"
            return "💹 Enter **annual interest rate** (% p.a.), e.g. `8.5`:"

        if mode == "emi_r":
            ok, val = safe_float(cmd)
            if not ok or val < 0:
                return "❌ Enter a non-negative rate."
            data["r"] = val
            st.session_state.dialogue_mode = "emi_n"
            return "📅 Enter **loan tenure in months**, e.g. `36`:"

        if mode == "emi_n":
            ok, val = safe_float(cmd)
            if not ok or val <= 0:
                return "❌ Enter a positive number of months."
            result = ToolBox.emi(data["p"], data["r"], int(val))
            SessionManager.reset_dialogue()
            return result

        # ── Compound Interest ──────────────────────────────────────
        if mode == "ci_p":
            ok, val = safe_float(cmd)
            if not ok or val <= 0:
                return "❌ Enter a positive principal amount."
            data["p"] = val
            st.session_state.dialogue_mode = "ci_r"
            return "💹 Enter **annual interest rate** (%), e.g. `6`:"

        if mode == "ci_r":
            ok, val = safe_float(cmd)
            if not ok or val < 0:
                return "❌ Rate must be non-negative."
            data["r"] = val
            st.session_state.dialogue_mode = "ci_n"
            return "🔄 **Compounding frequency** per year (e.g. `12` = monthly, `4` = quarterly, `1` = annual):"

        if mode == "ci_n":
            ok, val = safe_float(cmd)
            if not ok or val <= 0:
                return "❌ Enter a positive compounding frequency."
            data["n"] = val
            st.session_state.dialogue_mode = "ci_t"
            return "📅 Enter **time in years**, e.g. `5`:"

        if mode == "ci_t":
            ok, val = safe_float(cmd)
            if not ok or val <= 0:
                return "❌ Enter a positive number of years."
            result = ToolBox.compound_interest(data["p"], data["r"], data["n"], val)
            SessionManager.reset_dialogue()
            return result

        # ── GPA ────────────────────────────────────────────────────
        if mode == "gpa_grades":
            result = ToolBox.gpa(raw)
            SessionManager.reset_dialogue()
            return result

        # ── Unit Converter ─────────────────────────────────────────
        if mode == "unit_select":
            mapping = {
                "1": "c_to_f", "2": "f_to_c", "3": "kg_to_lb",
                "4": "lb_to_kg", "5": "m_to_ft",  "6": "ft_to_m",
                "7": "km_to_mi","8": "mi_to_km",  "9": "l_to_gal",
                "10":"gal_to_l",
            }
            if cmd not in mapping:
                return "❌ Please enter a number from **1 to 10**."
            data["conv"] = mapping[cmd]
            st.session_state.dialogue_mode = "unit_value"
            labels = dict(zip(mapping.values(), [
                "°C", "°F", "kg", "lbs", "m", "ft", "km", "mi", "L", "gal"
            ]))
            return f"🔢 Enter the value in **{labels[mapping[cmd]]}**:"

        if mode == "unit_value":
            ok, val = safe_float(cmd)
            if not ok:
                return "❌ Please enter a valid number."
            result = ToolBox.convert_unit(data["conv"], val)
            SessionManager.reset_dialogue()
            return result

        # ── Currency ───────────────────────────────────────────────
        if mode == "cur_from":
            cur = cmd.upper()
            if cur not in CURRENCY_TABLE:
                return f"❌ Unknown currency **{cur}**. Supported: {', '.join(CURRENCY_TABLE)}"
            data["from"] = cur
            st.session_state.dialogue_mode = "cur_to"
            return f"🔄 Convert **{cur}** to which currency? (e.g. `PKR`, `EUR`)"

        if mode == "cur_to":
            cur = cmd.upper()
            if cur not in CURRENCY_TABLE:
                return f"❌ Unknown currency **{cur}**. Supported: {', '.join(CURRENCY_TABLE)}"
            data["to"] = cur
            st.session_state.dialogue_mode = "cur_amount"
            return f"💰 Enter the **amount** in {data['from']}:"

        if mode == "cur_amount":
            ok, val = safe_float(cmd)
            if not ok or val <= 0:
                return "❌ Enter a positive amount."
            result = ToolBox.convert_currency(val, data["from"], data["to"])
            SessionManager.reset_dialogue()
            return result

        # ── Weather ────────────────────────────────────────────────
        if mode == "weather_city":
            result = weather_svc.fetch(raw)
            SessionManager.reset_dialogue()
            return result

        # ── Todo add ───────────────────────────────────────────────
        if mode == "todo_add":
            result = DataManager.add_todo(raw)
            SessionManager.reset_dialogue()
            return result

        if mode == "todo_remove":
            result = DataManager.remove_todo(raw)
            SessionManager.reset_dialogue()
            return result

        if mode == "todo_toggle":
            ok, val = safe_float(cmd)
            if not ok:
                return "❌ Enter the task number to toggle."
            result = DataManager.toggle_todo(int(val))
            SessionManager.reset_dialogue()
            return result

        # ── Note add ───────────────────────────────────────────────
        if mode == "note_add":
            result = DataManager.add_note(raw)
            SessionManager.reset_dialogue()
            return result

        if mode == "note_delete":
            if not cmd.isdigit():
                return "❌ Enter the note number to delete."
            result = DataManager.delete_note(int(cmd))
            SessionManager.reset_dialogue()
            return result

        # ── Expense add ────────────────────────────────────────────
        if mode == "expense_amount":
            ok, val = safe_float(cmd)
            if not ok or val <= 0:
                return "❌ Enter a positive expense amount."
            data["amount"] = val
            st.session_state.dialogue_mode = "expense_cat"
            return "🏷️ Enter a **category** (e.g. `Food`, `Transport`, `Bills`) or press Enter to skip:"

        if mode == "expense_cat":
            cat = raw.strip() or "General"
            data["cat"] = cat
            st.session_state.dialogue_mode = "expense_note"
            return "📝 Add a **note** for this expense (or press Enter to skip):"

        if mode == "expense_note":
            note   = raw.strip()
            result = DataManager.add_expense(data["amount"], data.get("cat", "General"), note)
            SessionManager.reset_dialogue()
            return result

        # ── Password length ────────────────────────────────────────
        if mode == "pwd_length":
            ok, val = safe_float(cmd)
            length  = int(val) if ok else 16
            pwd    = ToolBox.generate_password(length)
            SessionManager.reset_dialogue()
            return f"🔐 **Generated Password** ({length} chars):\n`{pwd}`\n\n*(Store it safely — it won't be shown again)*"

        # ── Pomodoro ───────────────────────────────────────────────
        if mode == "pomodoro_running":
            elapsed = time.time() - (st.session_state.pomodoro_start or time.time())
            remaining = max(0, 25 * 60 - elapsed)
            m, s = divmod(int(remaining), 60)
            if remaining == 0:
                st.session_state.pomodoro_count += 1
                st.session_state.pomodoro_start = None
                SessionManager.reset_dialogue()
                return (
                    f"⏰ **Pomodoro complete!** 🎉\n"
                    f"Take a 5-min break. Sessions completed: "
                    f"**{st.session_state.pomodoro_count}**"
                )
            return f"⏱️ Pomodoro running: **{m:02d}:{s:02d}** remaining."

        # ── Habit check ────────────────────────────────────────────
        if mode == "habit_name":
            result = DataManager.check_habit(raw)
            SessionManager.reset_dialogue()
            return result

        # Unknown FSM node — safe fallback
        logger.warning("Unknown dialogue mode: %s", mode)
        SessionManager.reset_dialogue()
        return "⚠️ Something went wrong. Resetting conversation state."

    # ──────────────────────────────────────────────────────────────
    # INTENT MATCHER
    # ──────────────────────────────────────────────────────────────
    def _match_intent(self, raw: str, cmd: str) -> str:

        # ── Greetings ──────────────────────────────────────────────
        if re.search(r"\b(hi|hello|hey|greetings|salaam|salam|howdy)\b", cmd):
            return f"👋 Hi there! How can I help you today? Type **`help`** to see all commands."

        # ── Wellbeing ──────────────────────────────────────────────
        if re.search(r"\b(how are you|how r u|you okay|you good)\b", cmd):
            return "🤖 I'm fully operational and ready to help! How are *you* doing?"

        # ── Help ───────────────────────────────────────────────────
        if re.search(r"\b(help|commands|menu|what can you do)\b", cmd):
            return self._help_text()

        # ── Date / Time ────────────────────────────────────────────
        if re.search(r"\btime\b", cmd):
            return f"🕐 Current time: **{datetime.now().strftime('%I:%M:%S %p')}**"
        if re.search(r"\b(date|today)\b", cmd):
            return f"📅 Today is **{datetime.now().strftime('%A, %d %B %Y')}**"
        if re.search(r"\bday\b", cmd):
            return f"📅 Today is **{datetime.now().strftime('%A')}**"

        # ── Weather ────────────────────────────────────────────────
        if re.search(r"\bweather\b", cmd):
            st.session_state.dialogue_mode = "weather_city"
            return "🌍 Which **city** would you like the weather for?"

        # ── Maths ──────────────────────────────────────────────────
        math_triggers = {
            r"\b(add|sum|addition|plus)\b":         "sum",
            r"\b(sub|subtract|subtraction|minus)\b":"sub",
            r"\b(mul|multiply|multiplication|times)\b":"mul",
            r"\b(div|divide|division)\b":            "div",
            r"\b(pow|power|exponent)\b":             "pow",
            r"\b(mod|modulus|remainder)\b":          "mod",
        }
        for pattern, op in math_triggers.items():
            if re.search(pattern, cmd):
                st.session_state.dialogue_mode = "math_a"
                st.session_state.dialogue_data = {"op": op}
                return f"🔢 Enter the **first number**:"

        # ── Calculators ────────────────────────────────────────────
        if re.search(r"\bbmi\b", cmd):
            st.session_state.dialogue_mode = "bmi_weight"
            return "⚖️ Enter your **weight in kg**, e.g. `70`:"

        if re.search(r"\bage\b", cmd):
            st.session_state.dialogue_mode = "age_dob"
            return "🎂 Enter your **date of birth** (e.g. `25-04-1995`):"

        if re.search(r"\b(percent|percentage|%)\b", cmd):
            st.session_state.dialogue_mode = "pct_part"
            return "📊 Enter the **part** (the smaller number):"

        if re.search(r"\b(emi|loan|mortgage)\b", cmd):
            st.session_state.dialogue_mode = "emi_p"
            return "🏦 Enter the **loan principal amount**, e.g. `500000`:"

        if re.search(r"\b(compound interest|ci)\b", cmd):
            st.session_state.dialogue_mode = "ci_p"
            return "💰 Enter the **principal amount**, e.g. `100000`:"

        if re.search(r"\bgpa\b", cmd):
            st.session_state.dialogue_mode = "gpa_grades"
            return "🎓 Enter your **grade points** separated by commas, e.g. `3.5, 4.0, 2.7`:"

        # ── Unit Converter ─────────────────────────────────────────
        if re.search(r"\b(convert|unit|units)\b", cmd):
            st.session_state.dialogue_mode = "unit_select"
            return (
                "📐 **Unit Converter** — Choose conversion:\n\n"
                "1. °C → °F  |  2. °F → °C\n"
                "3. kg → lbs |  4. lbs → kg\n"
                "5. m  → ft  |  6. ft → m\n"
                "7. km → mi  |  8. mi → km\n"
                "9. L  → gal | 10. gal → L\n\n"
                "*(Enter the number)*"
            )

        # ── Currency ───────────────────────────────────────────────
        if re.search(r"\b(currency|exchange|forex)\b", cmd):
            st.session_state.dialogue_mode = "cur_from"
            avail = ", ".join(CURRENCY_TABLE.keys())
            return f"💱 **Currency Converter**\nAvailable: {avail}\n\nEnter the **source currency** (e.g. `USD`):"

        # ── Password ───────────────────────────────────────────────
        if re.search(r"\b(password|passwd|pwd|keygen)\b", cmd):
            st.session_state.dialogue_mode = "pwd_length"
            return "🔐 How many characters? (8–64, default `16`):"

        # ── To-Do ──────────────────────────────────────────────────
        if re.search(r"\b(todo|task)\b", cmd):
            if re.search(r"\b(view|show|list|all)\b", cmd):
                return DataManager.list_todos()
            if re.search(r"\b(done|complete|finish|toggle|check)\b", cmd):
                st.session_state.dialogue_mode = "todo_toggle"
                return "✅ Enter the **task number** to toggle completion:"
            if re.search(r"\b(remove|delete|del)\b", cmd):
                if not st.session_state.todos:
                    return "📋 Your to-do list is empty."
                st.session_state.dialogue_mode = "todo_remove"
                return "🗑️ Enter the **task number or name** to remove:"
            # Default: add
            st.session_state.dialogue_mode = "todo_add"
            return "📋 What **task** would you like to add?"

        if re.search(r"\b(remove|delete)\b", cmd) and re.search(r"\btask\b", cmd):
            st.session_state.dialogue_mode = "todo_remove"
            return "🗑️ Enter the **task number or name** to remove:"

        # ── Notes ──────────────────────────────────────────────────
        if re.search(r"\bnote\b", cmd):
            if re.search(r"\b(view|show|list|all|read)\b", cmd):
                return DataManager.list_notes()
            if re.search(r"\b(delete|remove)\b", cmd):
                if not st.session_state.notes:
                    return "📝 No notes saved yet."
                st.session_state.dialogue_mode = "note_delete"
                return "🗑️ Enter the **note number** to delete:"
            st.session_state.dialogue_mode = "note_add"
            return "📝 Type your **note**:"

        if re.search(r"\b(save note|jot|write down)\b", cmd):
            st.session_state.dialogue_mode = "note_add"
            return "📝 Type your **note**:"

        # ── Expenses ───────────────────────────────────────────────
        if re.search(r"\b(expense|spend|spending|budget)\b", cmd):
            if re.search(r"\b(view|show|report|list|total|summary)\b", cmd):
                return DataManager.view_expenses()
            st.session_state.dialogue_mode = "expense_amount"
            return "💸 Enter the **expense amount**:"

        # ── Habits ─────────────────────────────────────────────────
        if re.search(r"\bhabit\b", cmd):
            if re.search(r"\b(view|show|list|all)\b", cmd):
                return DataManager.view_habits()
            st.session_state.dialogue_mode = "habit_name"
            return (
                f"🏋️ Which habit did you complete today?\n"
                f"Options: {', '.join(HABIT_ICONS.keys())}"
            )

        # ── Pomodoro ───────────────────────────────────────────────
        if re.search(r"\b(pomodoro|pomo|focus timer|work timer)\b", cmd):
            if st.session_state.pomodoro_start:
                elapsed  = time.time() - st.session_state.pomodoro_start
                remaining = max(0, 25 * 60 - elapsed)
                m, s = divmod(int(remaining), 60)
                return f"⏱️ Pomodoro in progress: **{m:02d}:{s:02d}** remaining."
            st.session_state.pomodoro_start = time.time()
            st.session_state.dialogue_mode  = "pomodoro_running"
            return (
                "🍅 **Pomodoro Timer Started!**\n"
                "⏱️ Focus for **25 minutes**.\n"
                "Type anything to check remaining time."
            )

        # ── Coin / Dice ────────────────────────────────────────────
        if re.search(r"\b(flip|coin|toss)\b", cmd):
            return f"🪙 Coin flip: **{random.choice(['Heads', 'Tails'])}!**"

        if re.search(r"\b(dice|roll|d6)\b", cmd):
            val = random.randint(1, 6)
            return f"🎲 Rolled: **{val}** {'(lucky!)' if val == 6 else ''}"

        # ── Quote ──────────────────────────────────────────────────
        if re.search(r"\b(quote|motivation|inspire|inspire me)\b", cmd):
            return f"💡 *\"{random.choice(MOTIVATIONAL_QUOTES)}\"*"

        # ── System ─────────────────────────────────────────────────
        if re.search(r"\b(system|sysinfo|specs|hardware|cpu|ram|disk)\b", cmd):
            return ToolBox.system_info()

        # ── Stats ──────────────────────────────────────────────────
        if re.search(r"\b(stats|statistics|dashboard|summary)\b", cmd):
            return self._session_stats()

        # ── Fallback ───────────────────────────────────────────────
        return (
            "🤔 I didn't quite understand that.\n\n"
            "Try typing **`help`** to see everything I can do, "
            "or rephrase your request."
        )

    # ──────────────────────────────────────────────────────────────
    # HELP TEXT
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _help_text() -> str:
        return """📖 **NEXUS AI — Command Directory**

🌤️ **Weather**
• `weather` — Live weather for any city

🧮 **Math & Calculators**
• `add / sub / mul / div / pow / mod` — Step-by-step calculator
• `bmi` — Body Mass Index calculator
• `age` — Age from date of birth
• `percentage` — Percentage of a total
• `emi` / `loan` — Monthly loan EMI calculator
• `compound interest` / `ci` — Compound interest calculator
• `gpa` — GPA from grade points

📐 **Converters**
• `convert` / `unit` — 10 unit conversions
• `currency` / `forex` — Offline currency exchange

📋 **Productivity**
• `todo` — Add a task
• `todo view` — List all tasks
• `todo done` — Toggle task complete
• `todo remove` — Delete a task
• `note` — Save a note
• `note view` — Read saved notes
• `note delete` — Remove a note
• `expense` — Log an expense
• `expense view` — Spending report
• `habit <name>` — Check off a habit today
• `habit view` — View all habits + streaks
• `pomodoro` — Start a 25-min focus timer

🔐 **Utilities**
• `password` — Secure password generator
• `coin` / `flip` — Flip a coin
• `dice` / `roll` — Roll a die
• `quote` — Motivational quote
• `system` — System diagnostics

📊 **Session**
• `stats` — Session statistics
• `clear` — Clear chat history
• `exit` / `quit` — Sign off"""

    @staticmethod
    def _session_stats() -> str:
        start = datetime.fromisoformat(st.session_state.session_start)
        duration = datetime.now() - start
        mins = int(duration.total_seconds() // 60)
        return (
            f"📊 **Session Statistics**\n"
            f"• Messages sent:  {st.session_state.total_messages}\n"
            f"• Session time:   {mins} minute{'s' if mins != 1 else ''}\n"
            f"• To-Do tasks:    {len(st.session_state.todos)}\n"
            f"• Notes saved:    {len(st.session_state.notes)}\n"
            f"• Expenses logged:{len(st.session_state.expenses)}\n"
            f"• Habits tracked: {len(st.session_state.habits)}\n"
            f"• Pomodoros done: {st.session_state.pomodoro_count}"
        )


# ══════════════════════════════════════════════════════════════════
# UI RENDERER
# ══════════════════════════════════════════════════════════════════
class UIRenderer:
    """All Streamlit UI code lives here — completely separate from logic."""

    CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --accent:   #58a6ff;
    --accent2:  #3fb950;
    --danger:   #f85149;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --user-bg:  #1f3a5f;
    --bot-bg:   #1a2332;
    --radius:   12px;
    --font:     'Space Grotesk', sans-serif;
    --mono:     'JetBrains Mono', monospace;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App header ── */
.nexus-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a2332 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.25rem 1.5rem 1rem;
    margin: -1rem -1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.nexus-logo {
    font-size: 2rem;
    animation: pulse 2.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1);    opacity: 1;   }
    50%       { transform: scale(1.08); opacity: 0.85; }
}
.nexus-title { font-size: 1.5rem; font-weight: 700; color: var(--accent); letter-spacing: -0.3px; }
.nexus-sub   { font-size: 0.78rem; color: var(--muted); margin-top: 1px; }
.nexus-badge {
    margin-left: auto;
    background: var(--accent2);
    color: #0d1117;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.75rem !important;
    padding: 0.9rem 1.1rem !important;
    animation: slide-in 0.2s ease;
}
@keyframes slide-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]) {
    border-left: 3px solid var(--accent2);
}
[data-testid="stChatMessage"][data-role="user"] {
    border-left: 3px solid var(--accent) !important;
    background: var(--user-bg) !important;
}

/* ── Input bar ── */
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; font-weight: 700; }

/* ── Code blocks ── */
code {
    font-family: var(--mono) !important;
    background: rgba(88,166,255,0.1) !important;
    color: #79c0ff !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: var(--font) !important;
    transition: opacity 0.15s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
"""

    @staticmethod
    def inject_css() -> None:
        st.markdown(UIRenderer.CSS, unsafe_allow_html=True)

    @staticmethod
    def render_header() -> None:
        st.markdown(
            f"""
<div class="nexus-header">
  <span class="nexus-logo">🤖</span>
  <div>
    <div class="nexus-title">NEXUS AI</div>
    <div class="nexus-sub">Enterprise Personal Assistant · v{APP_VERSION}</div>
  </div>
  <span class="nexus-badge">LIVE</span>
</div>
""",
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_sidebar() -> None:
        with st.sidebar:
            st.markdown("## 🤖 NEXUS AI")
            st.caption(f"Version {APP_VERSION}")
            st.divider()

            # ── Quick stats ───────────────────────────────────────
            st.markdown("### 📊 Session Stats")
            col1, col2 = st.columns(2)
            col1.metric("Messages", st.session_state.total_messages)
            col2.metric("Tasks", len(st.session_state.todos))
            col1.metric("Notes", len(st.session_state.notes))
            col2.metric("Expenses", len(st.session_state.expenses))
            st.divider()

            # ── Quick actions ─────────────────────────────────────
            st.markdown("### ⚡ Quick Actions")
            if st.button("🗑️ Clear Chat", use_container_width=True):
                SessionManager.clear_chat()
                SessionManager.reset_dialogue()
                st.rerun()

            if st.button("📋 View Tasks", use_container_width=True):
                reply = DataManager.list_todos()
                SessionManager.push_bot(reply)
                st.rerun()

            if st.button("📝 View Notes", use_container_width=True):
                reply = DataManager.list_notes()
                SessionManager.push_bot(reply)
                st.rerun()

            if st.button("💳 Expense Report", use_container_width=True):
                reply = DataManager.view_expenses()
                SessionManager.push_bot(reply)
                st.rerun()

            if st.button("🏋️ Habit Dashboard", use_container_width=True):
                reply = DataManager.view_habits()
                SessionManager.push_bot(reply)
                st.rerun()

            st.divider()

            # ── Export chat ───────────────────────────────────────
            st.markdown("### 💾 Export")
            if st.session_state.messages:
                export_lines = []
                for m in st.session_state.messages:
                    role = "You" if m["role"] == "user" else "NEXUS"
                    export_lines.append(f"[{m.get('ts','')}] {role}: {m['content']}")
                export_text = "\n\n".join(export_lines)
                st.download_button(
                    label="⬇️ Download Chat (.txt)",
                    data=export_text,
                    file_name=f"nexus_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            st.divider()

            # ── Dialogue status ───────────────────────────────────
            mode = st.session_state.dialogue_mode
            if mode:
                st.info(f"⚙️ Active flow: `{mode}`")
                if st.button("❌ Cancel Flow", use_container_width=True):
                    SessionManager.reset_dialogue()
                    SessionManager.push_bot("🔄 Operation cancelled.")
                    st.rerun()

            st.caption("Built with ❤️ using Streamlit + Open-Meteo")

    @staticmethod
    def render_messages() -> None:
        for msg in st.session_state.messages:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════
def app() -> None:
    """Application entry point."""

    # ── Page config (must be first Streamlit call) ────────────────
    st.set_page_config(
        page_title=f"{APP_NAME} — Personal Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Bootstrap ─────────────────────────────────────────────────
    SessionManager.init()
    UIRenderer.inject_css()

    # ── Layout ────────────────────────────────────────────────────
    UIRenderer.render_sidebar()
    UIRenderer.render_header()
    UIRenderer.render_messages()

    # ── Input ─────────────────────────────────────────────────────
    user_input = st.chat_input("Message NEXUS AI…")
    if not user_input:
        return

    raw = sanitise(user_input)
    if not raw:
        st.warning("⚠️ Empty or invalid message.")
        return

    # ── Rate limit check ──────────────────────────────────────────
    if not RateLimiter.check():
        st.error("⏱️ You're sending messages too quickly. Please wait a moment.")
        return

    # ── Log user message ──────────────────────────────────────────
    SessionManager.push_user(raw)

    # ── Route & get reply ─────────────────────────────────────────
    router = CommandRouter()
    try:
        reply = router.route(raw)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unhandled exception in router: %s", exc)
        reply = "⚠️ An unexpected error occurred. Please try again."
        SessionManager.reset_dialogue()

    # ── Log bot reply (if non-empty — clear returns "") ───────────
    if reply:
        SessionManager.push_bot(reply)

    st.rerun()


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app()
