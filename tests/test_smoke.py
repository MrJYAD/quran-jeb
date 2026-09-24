"""Smoke test for quran-jeb. No model needed for the exact-path checks.

    python tests/test_smoke.py          # plain
    python -m pytest tests/             # or with pytest

These numbers are the correctness contract of the project. If any of them
changes, something on the exact path changed — investigate before shipping.
"""
import io, json, os, sys, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import quran_index as qi          # noqa: E402
import quran_router as qr         # noqa: E402

ix = qi.get_index()


def ans(intent, text):
    return qr.answer(intent, text, ix)


def test_data_present():
    assert len(ix.words) == 77_432
    assert ix.page_count == 604
    assert len(ix.surahs) == 114


def test_known_counts():
    assert ans("بحث عن كلمة", "وين ذكرت كلمة الرحمن؟")[0]["count"] == 45
    assert ans("عدد مرات التكرار", "كم مرة تكررت كلمة الصلاة؟")[0]["count"] == 58
    assert ans("بحث عن كلمة", "من")[0]["count"] == 2763          # bare stopword is searchable
    assert ans("بحث عن كلمة", "وين جاء لفظ الصبر؟")[0]["match"] == "stem"
    r = ans("بحث عن كلمة", "وين ذكرت كلمة عبدالله")[0]        # one word today, two in the mushaf
    assert r["match"] == "split" and r["count"] == 2 and r["shown"] == "عبد الله"
    assert ans("عدد مرات التكرار", "كم مرة جاءت عبدالله")[0]["count"] == 2


def test_known_locations():
    p, k = ans("موقع الآية", "وش صفحة سورة 2 آية 255")
    assert k == "locate_ayah" and p["page"] == 42
    assert ans("معلومات السورة", "معلومات عن سورة الملك")[0]["ayah_count"] == 30
    assert ans("معلومات السورة", "كم آية في سورة الكهف؟")[0]["ayah_count"] == 110   # punctuation glued to the name
    assert ans("معلومات السورة", "سورة يس مكية ولا مدنية")[0]["revelation"] == "مكية"
    assert ans("محتويات الصفحة", "وش في صفحة 604")[0]["last"] == {"surah": 114, "ayah": 6}


def test_never_guesses_an_answer():
    # surah 1 has 7 ayahs: this must refuse, never resolve to سورة ص (page 453)
    p, k = ans("موقع الآية", "وش صفحة سورة 1 آية 99")
    assert k == "need_more" and "7" in p["need"]
    assert ix.surah_by_name("وش صفحة سورة 1 آية 99") is None
    assert ix.surah_by_name("صفحة") is None
    assert ix.surah_by_name("سورة ص") == 38
    for text in ("معلومات عن سورة 200", "وش في صفحة 999", "وش في صفحة 0"):
        p, k = ans("معلومات السورة" if "سورة" in text else "محتويات الصفحة", text)
        assert k == "need_more", text


def test_term_extraction():
    assert ans("بحث عن كلمة", "وين الرحمن بس")[0]["term"] == "الرحمن"
    p, _ = ans("بحث عن كلمة", "«بسم الله»")
    assert p["phrase"] and p["count"] > 0


def test_structural_override():
    assert qr.refine("عدد مرات التكرار", "كم آية في سورة الكهف", 0.8)[0] == "معلومات السورة"
    assert qr.refine("بحث عن كلمة", "كم مرة وردت كلمة الله", 0.7)[0] == "عدد مرات التكرار"
    assert qr.refine("معلومات السورة", "وش صفحة سورة 2 آية 255", 0.2)[0] == "موقع الآية"


def test_precheck():
    assert qr.precheck("") and qr.precheck("hello") and qr.precheck("ا" * 301)
    assert qr.precheck("وين الرحمن") is None


def test_http_if_running():
    """Only runs when the server is up; skips quietly otherwise."""
    url = os.environ.get("QURAN_JEB_URL", "http://127.0.0.1:8001")
    assert url.startswith(("http://", "https://")), "QURAN_JEB_URL must be http(s)"
    try:
        s = json.loads(urllib.request.urlopen(f"{url}/api/status", timeout=2).read())
    except Exception:                                         # noqa: BLE001
        print("  (server not running — HTTP checks skipped)")
        return
    if s.get("status") != "ready":
        print("  (server not ready — HTTP checks skipped)")
        return
    def post(raw):
        r = urllib.request.Request(f"{url}/api/quran", data=raw, headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(r, timeout=60).getcode()
        except urllib.error.HTTPError as e:
            return e.code
    assert post(b'{"text": 5}') == 400
    assert post(b'[1,2]') == 400
    assert post(b'{{{') == 400
    assert post(json.dumps({"text": "x" * 400}).encode()) == 400
    assert post(json.dumps({"text": "وين الرحمن"}, ensure_ascii=False).encode("utf-8")) == 200


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
