import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egysent.preprocess import URL_TOKEN, USER_TOKEN, clean_tweet  # noqa: E402


def test_links_and_handles_become_tokens():
    assert clean_tweet("شوف ده https://t.co/abc @ahmed_1") == f"شوف ده {URL_TOKEN} {USER_TOKEN}"


def test_hashtag_words_are_kept():
    assert clean_tweet("#قرار_ممتاز جدا") == "قرار ممتاز جدا"


def test_stretched_letters_keep_two_copies():
    assert clean_tweet("حلوووووو اوي") == "حلوو اوي"
    assert clean_tweet("هههههههه") == "هه"


def test_diacritics_tatweel_and_letter_variants_are_normalized():
    assert clean_tweet("جَمِيـــلة") == "جميلة"
    assert clean_tweet("أحمد إبراهيم آمال") == "احمد ابراهيم امال"
    assert clean_tweet("على مستشفى") == "علي مستشفي"


def test_emojis_and_negation_are_kept():
    assert clean_tweet("مش حلو خالص 😡") == "مش حلو خالص 😡"


def test_whitespace_is_collapsed():
    assert clean_tweet("  كلام   كتير \n هنا ") == "كلام كتير هنا"
