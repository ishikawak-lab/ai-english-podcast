"""Configuration for AI English Podcast Generator."""
import os
from pathlib import Path

# ──────────────────────────────────────────────
# API Keys  (export these before running)
# ──────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
CALLMEBOT_PHONE: str = os.environ.get("CALLMEBOT_PHONE", "")   # e.g. "+60123456789"
CALLMEBOT_APIKEY: str = os.environ.get("CALLMEBOT_APIKEY", "")

# ──────────────────────────────────────────────
# Filesystem Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"          # temp WAV + final MP3
DOCS_DIR = BASE_DIR / "docs"             # GitHub Pages root
AUDIO_DIR = DOCS_DIR / "audio"           # served MP3 files
RSS_FEED_PATH = DOCS_DIR / "feed.xml"
LOG_PATH = BASE_DIR / "logs" / "podcast.log"

# ──────────────────────────────────────────────
# GitHub Pages / Podcast Metadata
# ──────────────────────────────────────────────
GITHUB_PAGES_BASE_URL: str = os.environ.get(
    "GITHUB_PAGES_BASE_URL",
    "https://ishikawak-lab.github.io/ai-english-podcast",
)
PODCAST_TITLE = "Ken's AI English Podcast"
PODCAST_DESCRIPTION = (
    "Daily 10-minute English lessons through AI, tech, Malaysia, Japan, "
    "real estate, startups, and early childhood education news."
)
PODCAST_AUTHOR = "Ken"
PODCAST_EMAIL: str = os.environ.get("PODCAST_EMAIL", "podcast@example.com")
PODCAST_LANGUAGE = "en"
PODCAST_CATEGORY = "Education"
MAX_FEED_EPISODES = 30   # keep the last N episodes in feed.xml

# ──────────────────────────────────────────────
# Gemini Models
# ──────────────────────────────────────────────
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MODEL_FALLBACKS = ["gemini-2.5-flash", "gemini-2.0-flash-lite"]
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_VOICE = "Kore"   # Options: Kore, Aoede, Charon, Fenrir, Puck

# ──────────────────────────────────────────────
# Audio Processing
# ──────────────────────────────────────────────
TARGET_LUFS = -19          # integrated loudness target (LUFS)
AUDIO_BITRATE = "128k"
AUDIO_SAMPLE_RATE = 44100
TTS_SAMPLE_RATE = 24000    # Gemini TTS native output rate (PCM 16-bit mono)

# ──────────────────────────────────────────────
# Article Scoring & Selection
# ──────────────────────────────────────────────
MAX_ARTICLES_TO_SELECT = 5
ARTICLES_PER_SOURCE = 5           # cap per RSS feed
FRESHNESS_CUTOFF_HOURS = 72       # ignore articles older than this
SCORING_WEIGHTS = {
    "impact": 0.30,
    "novelty": 0.25,
    "freshness": 0.25,
    "ken_fit": 0.20,
}

# ──────────────────────────────────────────────
# RSS Sources  (25 total across 6 topics)
# ──────────────────────────────────────────────
RSS_SOURCES = [
    # AI & Technology (5)
    {"url": "https://www.artificialintelligence-news.com/feed/",    "category": "AI"},
    {"url": "https://openai.com/news/rss.xml",                      "category": "AI"},
    {"url": "https://www.technologyreview.com/feed/",               "category": "AI"},
    {"url": "https://huggingface.co/blog/feed.xml",                 "category": "AI"},
    {"url": "https://feeds.feedburner.com/oreilly/radar",           "category": "AI"},
    # Malaysia (5)
    {"url": "https://www.malaymail.com/feed",                       "category": "Malaysia"},
    {"url": "https://www.thestar.com.my/rss/news",                  "category": "Malaysia"},
    {"url": "https://www.freemalaysiatoday.com/feed/",              "category": "Malaysia"},
    {"url": "https://www.nst.com.my/feed",                          "category": "Malaysia"},
    {"url": "https://www.malaysiakini.com/rss",                     "category": "Malaysia"},
    # Japan (4)
    {"url": "https://www3.nhk.or.jp/rss/news/cat0.xml",            "category": "Japan"},
    {"url": "https://japantoday.com/feed",                          "category": "Japan"},
    {"url": "https://www.japantimes.co.jp/feed",                    "category": "Japan"},
    {"url": "https://mainichi.jp/rss/etc/english.rss",              "category": "Japan"},
    # Real Estate (4)
    {"url": "https://www.propertyguru.com.my/property-news-feed",  "category": "RealEstate"},
    {"url": "https://www.edgeprop.my/rss",                          "category": "RealEstate"},
    {"url": "https://therealdeal.com/feed/",                        "category": "RealEstate"},
    {"url": "https://www.housingwire.com/feed/",                    "category": "RealEstate"},
    # Startups (4)
    {"url": "https://techcrunch.com/feed/",                         "category": "Startup"},
    {"url": "https://www.techinasia.com/feed",                      "category": "Startup"},
    {"url": "https://e27.co/feed/",                                 "category": "Startup"},
    {"url": "https://venturebeat.com/feed/",                        "category": "Startup"},
    # Early Childhood Education (3)
    {"url": "https://www.edsurge.com/feed",                         "category": "EarlyEd"},
    {"url": "https://www.naeyc.org/resources/blog/feed",            "category": "EarlyEd"},
    {"url": "https://www.childtrends.org/feed",                     "category": "EarlyEd"},
]

# ──────────────────────────────────────────────
# Day-of-Week Frameworks  (weekday() 0=Mon … 6=Sun)
# ──────────────────────────────────────────────
DAY_FRAMEWORKS = {
    0: {
        "name": "PREP",
        "description": "Point → Reason → Example → Point",
        "instruction": (
            "Use the PREP framework: open with your main Point, explain the Reason "
            "it matters, share a concrete Example drawn from today's stories, then "
            "restate the Point as a listener takeaway."
        ),
    },
    1: {
        "name": "PSI",
        "description": "Problem → Solution → Implementation",
        "instruction": (
            "Use the PSI framework: clearly define the Problem highlighted by today's "
            "news, present the Solution with supporting evidence, then walk through "
            "practical Implementation or next steps listeners can take."
        ),
    },
    2: {
        "name": "SCR",
        "description": "Situation → Complication → Resolution",
        "instruction": (
            "Use the SCR framework: set the Situation (background context), introduce "
            "the Complication (the challenge or conflict), then deliver the Resolution "
            "(outcome or what it signals for the future)."
        ),
    },
    3: {
        "name": "Before-After-Bridge",
        "description": "Before → After → Bridge",
        "instruction": (
            "Use the Before-After-Bridge framework: paint the Before picture (current "
            "state of affairs), describe the After vision (what the world looks like "
            "when this plays out), then explain the Bridge — the key insight, action, "
            "or trend connecting the two."
        ),
    },
    4: {
        "name": "SOAR",
        "description": "Situation → Obstacle → Action → Result",
        "instruction": (
            "Use the SOAR framework: describe the Situation, identify the Obstacle or "
            "challenge faced, explain the Action taken by key players in the news, and "
            "reveal the Result or expected impact."
        ),
    },
    5: {
        "name": "5Ws+H",
        "description": "Who / What / When / Where / Why / How",
        "instruction": (
            "Use the 5Ws+H framework: answer Who is involved, What happened, When "
            "it occurred, Where it took place, Why it matters to listeners, and How "
            "it will unfold or what listeners should watch next."
        ),
    },
    6: {
        "name": "Pyramid",
        "description": "Conclusion → Supporting arguments → Evidence",
        "instruction": (
            "Use the Inverted Pyramid framework: lead immediately with the main "
            "Conclusion, then layer in the Supporting arguments, and finally ground "
            "it with detailed Evidence and data from today's stories."
        ),
    },
}
