"""
dafont.py — باحث DaFont المطور مع تطبيع أسماء الخطوط وكاش محلي SQLite
====================================================================
يعالج المشكلة الأساسية في النسخة الأولى:
عندما كان WhatFontIs يرجع اسماً مثل 'NMY_Abril-Fatface Regular' أو 'FF Bebas Neue Pro Bold'،
كان البحث في dafont يفشل لأن DaFont يسجل اسم العائلة فقط ('Abril Fatface' أو 'Bebas Neue').
"""

import os
import re
import sqlite3
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

DAFONT_SEARCH_URL = "https://www.dafont.com/search.php?q={query}"
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "dafont_cache.sqlite3")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def normalize_font_name(raw_title: str) -> str:
    """
    ينظف اسم الخط من بادئات المسابك وأسماء الأوزان والأنماط ليطابق البحث القياسي.
    مثال: 'NMY_Abril-Fatface Regular Pro' -> 'Abril Fatface'
    """
    cleaned = re.sub(
        r"^(NMY_|FF_|GOOGLE_|ITC_|LHF_|MT_|URW_|ADOBE_|LINO_)",
        "",
        raw_title or "",
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[-_]+", " ", cleaned)
    cleaned = re.sub(
        r"\b(Regular|Bold|Black|Light|Medium|SemiBold|ExtraBold|Ultra|Heavy|Thin|"
        r"Italic|Oblique|Roman|Std|Pro|LT|MT|OT|W01|W02|W04|Display|Decorative)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 2 else (raw_title or "").strip()


def _init_cache_db():
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dafont_cache (
                clean_query TEXT PRIMARY KEY,
                search_url TEXT,
                font_page TEXT,
                download_url TEXT,
                matched_name TEXT,
                updated_at INTEGER
            )
            """
        )


def search_dafont_cached(font_name: str, timeout: int = 12) -> dict:
    """يبحث في الكاش المحلي أولاً، وإذا لم يجد الخط يبحث في DaFont بالاسم المطبّع ويحفظ النتيجة."""
    _init_cache_db()
    clean_name = normalize_font_name(font_name)
    cache_key = clean_name.lower()

    with sqlite3.connect(CACHE_DB_PATH) as conn:
        row = conn.execute(
            "SELECT search_url, font_page, download_url, matched_name FROM dafont_cache WHERE clean_query = ?",
            (cache_key,),
        ).fetchone()
        if row:
            return {
                "clean_query": clean_name,
                "search_url": row[0],
                "font_page": row[1],
                "download_url": row[2],
                "matched_name": row[3],
                "from_cache": True,
            }

    fresh = _scrape_dafont(clean_name, timeout=timeout)

    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO dafont_cache
            (clean_query, search_url, font_page, download_url, matched_name, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                fresh["search_url"],
                fresh["font_page"],
                fresh["download_url"],
                fresh["matched_name"],
                int(time.time()),
            ),
        )

    return fresh


def _scrape_dafont(clean_font_name: str, timeout: int = 12) -> dict:
    query = quote(clean_font_name)
    search_url = DAFONT_SEARCH_URL.format(query=query)

    result = {
        "clean_query": clean_font_name,
        "search_url": search_url,
        "font_page": None,
        "download_url": None,
        "matched_name": None,
        "from_cache": False,
    }

    try:
        r = requests.get(search_url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return result

    soup = BeautifulSoup(r.text, "html.parser")
    candidates = soup.find_all("a", href=re.compile(r"\.font(\?|$)"))
    if not candidates:
        return result

    best = candidates[0]
    href = best.get("href", "")
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = "https://www.dafont.com" + href
    elif not href.startswith("http"):
        href = "https://www.dafont.com/" + href

    result["font_page"] = href
    result["matched_name"] = best.get_text(strip=True) or clean_font_name

    # استخراج رابط التحميل المباشر .zip من صفحة الخط أو مباشرة من نتائج البحث
    dl_direct = soup.find("a", href=re.compile(r"dl\.dafont\.com/dl/\?f="))
    if dl_direct:
        dl_href = dl_direct.get("href", "")
        if dl_href.startswith("//"):
            dl_href = "https:" + dl_href
        result["download_url"] = dl_href
        return result

    try:
        r2 = requests.get(href, headers=HEADERS, timeout=timeout)
        r2.raise_for_status()
        soup2 = BeautifulSoup(r2.text, "html.parser")
        dl = soup2.find("a", href=re.compile(r"dl\.dafont\.com/dl/\?f="))
        if dl:
            dl_href = dl.get("href", "")
            if dl_href.startswith("//"):
                dl_href = "https:" + dl_href
            result["download_url"] = dl_href
    except requests.RequestException:
        pass

    return result
