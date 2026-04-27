#!/usr/bin/env python3
"""AI English Podcast Generator.

Pipeline:
  1. Fetch articles from 25 RSS sources
  2. Score with Gemini 2.0 Flash (Impact/Novelty/Freshness/Ken-fit) → pick top 5
  3. Generate English script using the day's rhetorical framework
  4. Synthesize audio with Gemini TTS
  5. Normalise to target LUFS with ffmpeg
  6. Update GitHub Pages RSS feed
  7. Send WhatsApp notification via CallMeBot

Usage:
  python podcast_generator.py              # full production run
  python podcast_generator.py --dry-run   # fetch + score + print script; no I/O
  python podcast_generator.py --date 2026-04-28 --dry-run
"""

import argparse
import calendar
import datetime
import email.utils
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import feedparser
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import config

# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────

@dataclass
class Article:
    title: str
    url: str
    summary: str
    category: str
    published: datetime.datetime
    source_url: str
    impact: float = 0.0
    novelty: float = 0.0
    freshness: float = 0.0
    ken_fit: float = 0.0
    weighted_score: float = 0.0


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def setup_logging(dry_run: bool) -> logging.Logger:
    log = logging.getLogger("podcast")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.DEBUG)
    log.addHandler(ch)

    if not dry_run:
        config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

    return log


# ─────────────────────────────────────────────────────────────
# Step 1 — RSS Fetching
# ─────────────────────────────────────────────────────────────

def fetch_all_articles(log: logging.Logger) -> list[Article]:
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(hours=config.FRESHNESS_CUTOFF_HOURS)
    articles: list[Article] = []

    for source in config.RSS_SOURCES:
        url = source["url"]
        category = source["category"]
        try:
            _prev_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10)
            try:
                feed = feedparser.parse(url)
            finally:
                socket.setdefaulttimeout(_prev_timeout)
            count = 0
            for entry in feed.entries:
                if count >= config.ARTICLES_PER_SOURCE:
                    break
                pub = _parse_entry_date(entry)
                if pub and pub < cutoff:
                    continue
                if pub is None:
                    pub = now  # no date → treat as fresh

                title = (getattr(entry, "title", "") or "").strip()
                link = (getattr(entry, "link", "") or "").strip()
                if not title or not link:
                    continue

                summary = (
                    getattr(entry, "summary", "")
                    or getattr(entry, "description", "")
                    or ""
                )[:500].strip()

                articles.append(Article(
                    title=title,
                    url=link,
                    summary=summary,
                    category=category,
                    published=pub,
                    source_url=url,
                ))
                count += 1

            log.info(f"  {url}: {count} articles")
        except Exception as exc:
            log.warning(f"  SKIP {url}: {exc}")

    log.info(f"Total articles fetched: {len(articles)}")
    return articles


def _parse_entry_date(entry) -> Optional[datetime.datetime]:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            return datetime.datetime.fromtimestamp(
                calendar.timegm(t), tz=datetime.timezone.utc
            )
    return None


# ─────────────────────────────────────────────────────────────
# Step 2 — Scoring (Gemini 2.0 Flash, batch JSON)
# ─────────────────────────────────────────────────────────────

_SCORE_BATCH = 40   # articles per Gemini call

# Ken-fit baseline per category (used when Gemini scoring is unavailable)
_CATEGORY_KEN_FIT: dict[str, float] = {
    "AI": 9.0,
    "Malaysia": 9.0,
    "Japan": 8.5,
    "Startup": 8.0,
    "RealEstate": 7.5,
    "EarlyEd": 7.0,
}


def _extract_retry_delay(exc: Exception) -> float:
    """Pull the suggested retry delay from a 429 error message."""
    m = re.search(r'"retryDelay":\s*"(\d+)s"', str(exc))
    return float(m.group(1)) if m else 60.0


def _gemini_call_with_retry(fn, log: logging.Logger, max_retries: int = 2):
    """Call fn() and retry on 429.

    fn receives the active model name as its sole argument so callers can
    substitute the fallback model on each attempt.
    Primary model → GEMINI_MODEL_FALLBACKS[0] → GEMINI_MODEL_FALLBACKS[1] …
    """
    models = [config.GEMINI_MODEL] + config.GEMINI_MODEL_FALLBACKS
    for model in models:
        for attempt in range(max_retries):
            try:
                return fn(model)
            except genai_errors.ClientError as exc:
                err = str(exc)
                if "404" in err:
                    log.warning(f"Model {model} not found — trying next model …")
                    break  # skip to next model immediately
                if "429" not in err:
                    raise   # propagate non-quota errors
                if "limit: 0" in err:
                    log.warning(f"Model {model} has quota 0 — trying next model …")
                    break
                if attempt == max_retries - 1:
                    log.warning(f"Model {model} exhausted retries — trying next model …")
                    break
                delay = _extract_retry_delay(exc)
                log.warning(
                    f"Gemini 429 on {model}. "
                    f"Waiting {delay:.0f}s before retry {attempt + 1}/{max_retries} …"
                )
                time.sleep(delay)
    raise RuntimeError(
        "All Gemini models exhausted. Check API key quotas at https://ai.dev/rate-limit"
    )


def _apply_local_scores(articles: list[Article], log: logging.Logger) -> list[Article]:
    """Heuristic fallback when Gemini scoring is unavailable."""
    now = datetime.datetime.now(datetime.timezone.utc)
    w = config.SCORING_WEIGHTS
    for a in articles:
        hours_old = (now - a.published).total_seconds() / 3600
        if hours_old < 6:
            a.freshness = 10.0
        elif hours_old < 24:
            a.freshness = 8.0
        elif hours_old < 48:
            a.freshness = 6.0
        else:
            a.freshness = 4.0
        a.impact = 5.0
        a.novelty = 5.0
        a.ken_fit = _CATEGORY_KEN_FIT.get(a.category, 6.0)
        a.weighted_score = (
            w["impact"] * a.impact
            + w["novelty"] * a.novelty
            + w["freshness"] * a.freshness
            + w["ken_fit"] * a.ken_fit
        )
    log.info("Local fallback scoring applied.")
    return articles


def score_articles(
    articles: list[Article],
    client: genai.Client,
    log: logging.Logger,
) -> list[Article]:
    if not articles:
        return []

    now = datetime.datetime.now(datetime.timezone.utc)
    w = config.SCORING_WEIGHTS
    gemini_scored = 0

    for batch_start in range(0, len(articles), _SCORE_BATCH):
        batch = articles[batch_start : batch_start + _SCORE_BATCH]
        items_text = "\n\n".join(
            f"[{i}] Category: {a.category} | "
            f"Published: {((now - a.published).total_seconds() / 3600):.1f}h ago\n"
            f"    Title: {a.title}\n"
            f"    Summary: {a.summary[:200]}"
            for i, a in enumerate(batch)
        )

        prompt = (
            "Score each news article for an English podcast whose topics are: "
            "AI, Malaysia, Japan, real estate, startups, early childhood education "
            "(host persona: Ken — Japanese professional living in Malaysia).\n\n"
            "Four axes, each 1–10:\n"
            "  impact   : global or regional significance\n"
            "  novelty  : how surprising or unique the story is\n"
            "  freshness: recency (10=<6h, 8=<24h, 6=<48h, 4=<72h, 2=older)\n"
            "  ken_fit  : relevance to the podcast topics above\n\n"
            f"Articles:\n{items_text}\n\n"
            "Return a JSON array — one object per article, same order:\n"
            '[{"impact":X,"novelty":X,"freshness":X,"ken_fit":X}, ...]'
        )

        try:
            response = _gemini_call_with_retry(
                lambda m, p=prompt: client.models.generate_content(
                    model=m,
                    contents=p,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                ),
                log,
            )
            scores = json.loads(response.text)
            for j, a in enumerate(batch):
                if j >= len(scores):
                    break
                s = scores[j]
                a.impact = float(s.get("impact", 5))
                a.novelty = float(s.get("novelty", 5))
                a.freshness = float(s.get("freshness", 5))
                a.ken_fit = float(s.get("ken_fit", 5))
                a.weighted_score = (
                    w["impact"] * a.impact
                    + w["novelty"] * a.novelty
                    + w["freshness"] * a.freshness
                    + w["ken_fit"] * a.ken_fit
                )
                gemini_scored += 1
            log.info(
                f"Scored batch {batch_start}–{batch_start + len(batch) - 1} "
                f"({len(batch)} articles)"
            )
        except Exception as exc:
            log.error(f"Scoring batch {batch_start} failed: {exc}")

    # If Gemini couldn't score any articles, fall back to local heuristics
    if gemini_scored == 0:
        log.warning("Gemini scoring unavailable — using local heuristic fallback.")
        return _apply_local_scores(articles, log)

    return articles


def select_top_articles(
    articles: list[Article], log: logging.Logger
) -> list[Article]:
    ranked = sorted(articles, key=lambda a: a.weighted_score, reverse=True)
    selected = ranked[: config.MAX_ARTICLES_TO_SELECT]
    log.info("Selected articles:")
    for i, a in enumerate(selected, 1):
        log.info(
            f"  #{i} [{a.category}] score={a.weighted_score:.2f}  {a.title[:70]}"
        )
    return selected


# ─────────────────────────────────────────────────────────────
# Step 3 — Script Generation
# ─────────────────────────────────────────────────────────────

def generate_script(
    articles: list[Article],
    today: datetime.date,
    client: genai.Client,
    log: logging.Logger,
) -> str:
    fw = config.DAY_FRAMEWORKS[today.weekday()]
    day_name = today.strftime("%A, %B %d %Y")

    stories = "\n\n".join(
        f"Story {i}: [{a.category}] {a.title}\n"
        f"Summary : {a.summary[:350]}\n"
        f"URL     : {a.url}"
        for i, a in enumerate(articles, 1)
    )

    prompt = (
        f"You are Ken, host of 'Ken's AI English Podcast' — a daily 10-minute show "
        f"for Japanese professionals improving their English through global news.\n\n"
        f"Today is {day_name}. Today's rhetorical framework: {fw['name']}\n"
        f"Framework description: {fw['description']}\n"
        f"How to apply it: {fw['instruction']}\n\n"
        f"Today's {len(articles)} selected news stories:\n{stories}\n\n"
        "Write a complete podcast script (~1 000 words) in natural spoken English that:\n"
        "1. Opens with a warm greeting, introduces today's framework, and sets the theme\n"
        "2. Weaves all stories together using the framework structure\n"
        "3. Calls out 2 useful English expressions per story with brief in-context explanations\n"
        "4. Closes with a motivational sign-off and a teaser for tomorrow\n\n"
        "Rules: natural spoken tone, no stage directions, no markdown headers, "
        "no bullet points — continuous flowing prose the way a podcast host speaks."
    )

    response = _gemini_call_with_retry(
        lambda m, p=prompt: client.models.generate_content(
            model=m,
            contents=p,
            config=types.GenerateContentConfig(
                temperature=0.75,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        ),
        log,
    )
    script = response.text.strip()
    log.info(f"Script generated: {len(script):,} characters")
    return script


# ─────────────────────────────────────────────────────────────
# Step 4 — TTS Synthesis (Gemini)
# ─────────────────────────────────────────────────────────────

def synthesize_audio(
    script: str,
    raw_wav_path: Path,
    client: genai.Client,
    log: logging.Logger,
) -> None:
    log.info(f"TTS: model={config.TTS_MODEL}  voice={config.TTS_VOICE}")
    response = client.models.generate_content(
        model=config.TTS_MODEL,
        contents=script,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=config.TTS_VOICE
                    )
                )
            ),
        ),
    )

    audio_part = response.candidates[0].content.parts[0]
    pcm_bytes: bytes = audio_part.inline_data.data
    mime: str = audio_part.inline_data.mime_type   # e.g. "audio/l16;rate=24000"
    sample_rate = _parse_sample_rate(mime)

    _write_wav(pcm_bytes, raw_wav_path, sample_rate)
    log.info(
        f"Raw WAV written: {raw_wav_path.name}  "
        f"({len(pcm_bytes):,} bytes  {sample_rate} Hz)"
    )


def _parse_sample_rate(mime: str) -> int:
    if "rate=" in mime:
        try:
            return int(mime.split("rate=")[1].split(";")[0].strip())
        except ValueError:
            pass
    return config.TTS_SAMPLE_RATE


def _write_wav(pcm_data: bytes, path: Path, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)   # mono
        wf.setsampwidth(2)   # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# ─────────────────────────────────────────────────────────────
# Step 5 — Audio Normalisation (ffmpeg loudnorm)
# ─────────────────────────────────────────────────────────────

def normalize_audio(
    raw_wav: Path, out_mp3: Path, log: logging.Logger
) -> None:
    _require_ffmpeg(log)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_wav),
        "-af", f"loudnorm=I={config.TARGET_LUFS}:LRA=11:TP=-1.5",
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-b:a", config.AUDIO_BITRATE,
        str(out_mp3),
    ]
    log.info(
        f"Normalising: {raw_wav.name} → {out_mp3.name} "
        f"(target {config.TARGET_LUFS} LUFS)"
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr[-800:]}")
    log.info(f"MP3 written: {out_mp3}")


def _require_ffmpeg(log: logging.Logger) -> None:
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found. Install with: brew install ffmpeg")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Step 6 — GitHub Pages RSS Feed
# ─────────────────────────────────────────────────────────────

_ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"


def update_rss_feed(
    today: datetime.date,
    episode_title: str,
    episode_description: str,
    mp3_path: Path,
    log: logging.Logger,
) -> None:
    ET.register_namespace("itunes", _ITUNES_NS)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Copy MP3 into docs/audio/
    dest = config.AUDIO_DIR / mp3_path.name
    shutil.copy2(mp3_path, dest)
    audio_url = f"{config.GITHUB_PAGES_BASE_URL}/audio/{mp3_path.name}"
    file_size = dest.stat().st_size

    # Load or create feed skeleton
    if config.RSS_FEED_PATH.exists():
        tree = ET.parse(config.RSS_FEED_PATH)
        root = tree.getroot()
        channel = root.find("channel")
        if channel is None:
            raise ValueError("Malformed feed.xml: <channel> missing")
    else:
        root, channel = _new_feed_skeleton()
        tree = ET.ElementTree(root)

    # Build new <item>
    pub_rfc = email.utils.format_datetime(
        datetime.datetime.combine(
            today, datetime.time(), tzinfo=datetime.timezone.utc
        )
    )
    item = ET.Element("item")
    _sub(item, "title", episode_title)
    _sub(item, "link", audio_url)
    _sub(item, "description", episode_description)
    _sub(item, "pubDate", pub_rfc)
    _sub(item, "guid", audio_url, isPermaLink="true")
    ET.SubElement(
        item, "enclosure",
        url=audio_url,
        length=str(file_size),
        type="audio/mpeg",
    )
    ET.SubElement(item, f"{{{_ITUNES_NS}}}duration").text = (
        _duration_from_size(file_size)
    )

    # Insert before existing items so feed stays newest-first
    children = list(channel)
    first_item_idx = next(
        (i for i, c in enumerate(children) if c.tag == "item"),
        len(children),
    )
    channel.insert(first_item_idx, item)

    # Prune episodes beyond the configured cap
    for old in channel.findall("item")[config.MAX_FEED_EPISODES:]:
        channel.remove(old)

    _indent_xml(root)
    with open(config.RSS_FEED_PATH, "wb") as fh:
        fh.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(fh, encoding="utf-8", xml_declaration=False)

    log.info(f"RSS feed updated: {config.RSS_FEED_PATH}")


def _new_feed_skeleton():
    # Do NOT add xmlns:itunes manually — ET.register_namespace handles it
    root = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(root, "channel")
    _sub(channel, "title", config.PODCAST_TITLE)
    _sub(channel, "link", config.GITHUB_PAGES_BASE_URL)
    _sub(channel, "description", config.PODCAST_DESCRIPTION)
    _sub(channel, "language", config.PODCAST_LANGUAGE)
    ET.SubElement(channel, f"{{{_ITUNES_NS}}}author").text = config.PODCAST_AUTHOR
    ET.SubElement(channel, f"{{{_ITUNES_NS}}}explicit").text = "false"
    ET.SubElement(channel, f"{{{_ITUNES_NS}}}category", text=config.PODCAST_CATEGORY)
    return root, channel


def _sub(parent: ET.Element, tag: str, text: str = "", **attrib) -> ET.Element:
    el = ET.SubElement(parent, tag, **attrib)
    el.text = text
    return el


def _duration_from_size(size_bytes: int) -> str:
    # 128 kbps → 16 000 bytes/s
    seconds = max(0, size_bytes // 16_000)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        last_child = None
        for child in elem:
            _indent_xml(child, level + 1)
            last_child = child
        if last_child is not None and (
            not last_child.tail or not last_child.tail.strip()
        ):
            last_child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


# ─────────────────────────────────────────────────────────────
# Step 3b — Vocabulary Extraction
# ─────────────────────────────────────────────────────────────

def extract_vocab(
    script: str,
    client: genai.Client,
    log: logging.Logger,
) -> list[dict]:
    """Return vocabulary items highlighted in the script as a list of dicts."""
    prompt = (
        "The following is a podcast script for English learners. "
        "Extract every English expression or vocabulary item that was explicitly "
        "introduced or explained in the script.\n\n"
        "Return a JSON array — one object per item:\n"
        '[{"expression": "...", "meaning": "...", "example": "..."}]\n\n'
        f"Script:\n{script}"
    )
    try:
        response = _gemini_call_with_retry(
            lambda m, p=prompt: client.models.generate_content(
                model=m,
                contents=p,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            ),
            log,
        )
        items = json.loads(response.text)
        log.info(f"Vocabulary extracted: {len(items)} items")
        return items
    except Exception as exc:
        log.warning(f"Vocab extraction failed: {exc}")
        return []


# ─────────────────────────────────────────────────────────────
# Step 8 — Slack Notification (Incoming Webhook)
# ─────────────────────────────────────────────────────────────

_SLACK_BLOCK_MAX = 2800   # stay safely under the 3000-char block limit


def notify_slack(
    today: datetime.date,
    framework_name: str,
    articles: list[Article],
    script: str,
    vocab: list[dict],
    audio_url: str,
    rss_url: str,
    log: logging.Logger,
) -> None:
    if not config.SLACK_WEBHOOK_URL:
        log.warning("SLACK_WEBHOOK_URL not set — skipping Slack notification.")
        return

    blocks: list[dict] = []

    # ── Header ────────────────────────────────────────────────
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"Ken's AI English Podcast — {today}  |  {framework_name}",
        },
    })

    # ── Articles ──────────────────────────────────────────────
    articles_md = "\n".join(
        f"{i}. <{a.url}|{a.title}>  `[{a.category}]`  score: {a.weighted_score:.2f}"
        for i, a in enumerate(articles, 1)
    )
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*今日の選定記事 Top {len(articles)}*\n{articles_md}"},
    })

    blocks.append({"type": "divider"})

    # ── Script (chunked) ──────────────────────────────────────
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*台本全文*"},
    })
    for chunk_start in range(0, len(script), _SLACK_BLOCK_MAX):
        chunk = script[chunk_start : chunk_start + _SLACK_BLOCK_MAX]
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": chunk},
        })

    blocks.append({"type": "divider"})

    # ── Vocabulary ────────────────────────────────────────────
    if vocab:
        vocab_lines = "\n".join(
            f"• *{v.get('expression', '')}*  —  {v.get('meaning', '')}\n"
            f"  _例: {v.get('example', '')}_"
            for v in vocab
        )
        # Vocab list can also be long — chunk it
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*語彙・表現リスト*"},
        })
        for chunk_start in range(0, len(vocab_lines), _SLACK_BLOCK_MAX):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": vocab_lines[chunk_start : chunk_start + _SLACK_BLOCK_MAX],
                },
            })

    blocks.append({"type": "divider"})

    # ── Links ─────────────────────────────────────────────────
    blocks.append({
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": f":headphones: *音声*\n<{audio_url}|MP3を聴く>"},
            {"type": "mrkdwn", "text": f":rss: *RSSフィード*\n<{rss_url}|feed.xml>"},
        ],
    })

    # Slack caps at 50 blocks — trim gracefully
    if len(blocks) > 50:
        blocks = blocks[:49]
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "_（ブロック上限のため一部省略）_"},
        })

    payload = json.dumps({"blocks": blocks}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        config.SLACK_WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            log.info(f"Slack notification sent (HTTP {resp.status})")
    except Exception as exc:
        log.warning(f"Slack notification failed: {exc}")


# ─────────────────────────────────────────────────────────────
# Step 7 — WhatsApp Notification (CallMeBot)
# ─────────────────────────────────────────────────────────────

def notify_whatsapp(message: str, log: logging.Logger) -> None:
    if not config.CALLMEBOT_PHONE or not config.CALLMEBOT_APIKEY:
        log.warning("CALLMEBOT_PHONE or CALLMEBOT_APIKEY not set — skipping WhatsApp.")
        return

    url = (
        "https://api.callmebot.com/whatsapp.php"
        f"?phone={urllib.parse.quote(config.CALLMEBOT_PHONE)}"
        f"&text={urllib.parse.quote(message)}"
        f"&apikey={urllib.parse.quote(config.CALLMEBOT_APIKEY)}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            log.info(f"WhatsApp notification sent (HTTP {resp.status})")
    except Exception as exc:
        log.warning(f"WhatsApp notification failed: {exc}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI English Podcast Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  GEMINI_API_KEY          Google AI Studio API key\n"
            "  CALLMEBOT_PHONE         WhatsApp phone with country code (+601…)\n"
            "  CALLMEBOT_APIKEY        CallMeBot API key\n"
            "  GITHUB_PAGES_BASE_URL   Base URL of your GitHub Pages site\n"
            "  PODCAST_EMAIL           Contact email embedded in the RSS feed\n"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fetch articles, score, generate script, then print everything. "
            "No files written, no TTS, no WhatsApp."
        ),
    )
    p.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Override today's date (useful for testing a specific day's framework).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dry_run: bool = args.dry_run
    log = setup_logging(dry_run)

    today = (
        datetime.date.fromisoformat(args.date)
        if args.date
        else datetime.date.today()
    )
    fw_name = config.DAY_FRAMEWORKS[today.weekday()]["name"]
    log.info(
        f"{'[DRY RUN] ' if dry_run else ''}Generating episode for {today} "
        f"(framework: {fw_name})"
    )

    if not config.GEMINI_API_KEY:
        log.error("GEMINI_API_KEY is not set. Export it and re-run.")
        sys.exit(1)

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # 1. Fetch
    log.info("Step 1/7  Fetching RSS feeds …")
    articles = fetch_all_articles(log)
    if not articles:
        log.error("No articles fetched. Check RSS sources and network.")
        sys.exit(1)

    # 2. Score & select
    log.info("Step 2/7  Scoring with Gemini …")
    articles = score_articles(articles, client, log)
    selected = select_top_articles(articles, log)
    if not selected:
        log.error("No articles selected after scoring.")
        sys.exit(1)

    # 3. Script
    log.info("Step 3/8  Generating podcast script …")
    script = generate_script(selected, today, client, log)

    # 3b. Vocabulary extraction
    log.info("Step 3b/8  Extracting vocabulary list …")
    vocab = extract_vocab(script, client, log)

    # Dry-run exit after printing results
    if dry_run:
        print("\n" + "=" * 64)
        print(f"DRY RUN | {today} | {fw_name} framework")
        print("=" * 64)
        print("\nSELECTED ARTICLES:")
        for i, a in enumerate(selected, 1):
            print(f"  {i}. [{a.category:10s}] score={a.weighted_score:.2f}  {a.title}")
        print("\nGENERATED SCRIPT:\n")
        print(script)
        if vocab:
            print("\nVOCABULARY LIST:")
            for v in vocab:
                print(f"  • {v.get('expression')}: {v.get('meaning')}")
                print(f"    例: {v.get('example')}")
        print("\n[DRY RUN] No files written, no notifications sent.")
        return

    # 4. TTS
    log.info("Step 4/8  Synthesizing audio …")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = today.strftime("%Y-%m-%d")
    raw_wav = config.OUTPUT_DIR / f"{slug}_raw.wav"
    final_mp3 = config.OUTPUT_DIR / f"{slug}.mp3"

    synthesize_audio(script, raw_wav, client, log)

    # 5. Normalise
    log.info("Step 5/8  Normalising audio …")
    normalize_audio(raw_wav, final_mp3, log)
    raw_wav.unlink(missing_ok=True)

    # 6. RSS feed
    log.info("Step 6/8  Updating RSS feed …")
    episode_title = f"{today} | {fw_name} | Ken's AI English Podcast"
    episode_desc = "  |  ".join(
        f"[{a.category}] {a.title}" for a in selected
    )
    update_rss_feed(today, episode_title, episode_desc, final_mp3, log)

    audio_url = f"{config.GITHUB_PAGES_BASE_URL}/audio/{final_mp3.name}"
    rss_url = f"{config.GITHUB_PAGES_BASE_URL}/feed.xml"

    # 7. WhatsApp
    log.info("Step 7/8  Sending WhatsApp notification …")
    story_lines = "\n".join(
        f"  {i}. [{a.category}] {a.title}" for i, a in enumerate(selected, 1)
    )
    notify_whatsapp(
        f"New episode: {episode_title}\n\n"
        f"Stories:\n{story_lines}\n\n"
        f"Listen: {audio_url}",
        log,
    )

    # 8. Slack
    log.info("Step 8/8  Sending Slack notification …")
    notify_slack(
        today, fw_name, selected, script, vocab,
        audio_url, rss_url, log,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
