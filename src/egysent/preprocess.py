"""Light cleaning for Egyptian Arabic tweets, shared by the baseline and the transformer.

Kept on purpose: emojis, punctuation and negation words, because they carry sentiment.
Removed or normalized: links, user handles, diacritics, tatweel, letter variants,
and letters stretched for emphasis ("حلوووووو" -> "حلوو").
"""

from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ALEF = re.compile("[آأإٱ]")  # آ أ إ ٱ -> ا
_REPEATS = re.compile(r"(.)\1{2,}")  # 3+ of the same character
_SPACES = re.compile(r"\s+")

USER_TOKEN = "مستخدم"
URL_TOKEN = "رابط"


def clean_tweet(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _URL.sub(f" {URL_TOKEN} ", text)
    text = _MENTION.sub(f" {USER_TOKEN} ", text)
    # Hashtags often hold the opinion ("#قرار_ممتاز"), so keep their words.
    text = _HASHTAG.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = _ALEF.sub("ا", text)
    text = text.replace("ى", "ي")  # ى -> ي
    # Keep two copies of a stretched letter: "حلوووو" and "حلو" differ in tone.
    text = _REPEATS.sub(r"\1\1", text)
    return _SPACES.sub(" ", text).strip()
