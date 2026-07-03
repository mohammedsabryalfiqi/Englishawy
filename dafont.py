"""
بحث عن خط معيّن بالاسم داخل dafont.com ومحاولة استخراج رابط تحميل مباشر.

ملحوظة مهمة: الكود ده بيعتمد على تحليل صفحات HTML الحالية لموقع dafont.com.
لو الموقع غيّر شكل الصفحات (وده بيحصل من وقت للتاني في أي موقع)، ممكن تحتاج
تعدّل التعبيرات النمطية (regex) تحت. لو التحليل فشل، بيرجعلك على الأقل رابط
بحث صحيح في الموقع بدل ما يوقّف البوت بالكامل.
"""

import re
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

DAFONT_SEARCH_URL = "https://www.dafont.com/search.php?q={query}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def search_dafont(font_name: str, timeout: int = 15) -> dict:
    """
    يرجع dict فيه:
        search_url    : رابط نتائج البحث في dafont (دايمًا موجود)
        font_page     : رابط صفحة الخط لو اتلاقى
        download_url  : رابط تحميل مباشر (.zip) لو اتلاقى
        matched_name  : الاسم اللي ظهر في dafont (ممكن يختلف شوية عن اللي بحثنا بيه)
    """
    query = quote(font_name)
    search_url = DAFONT_SEARCH_URL.format(query=query)

    result = {
        "search_url": search_url,
        "font_page": None,
        "download_url": None,
        "matched_name": None,
    }

    try:
        r = requests.get(search_url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return result

    soup = BeautifulSoup(r.text, "html.parser")

    # صفحات تفاصيل الخط في dafont عادة بتنتهي بـ ".font"
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
    result["matched_name"] = best.get_text(strip=True) or font_name

    # حاول تدخل صفحة الخط نفسها وتلاقي رابط التحميل المباشر
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
