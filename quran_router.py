"""quran-jeb — Arabic intent routing for Quran queries.

The split that makes this safe:

    jeb decides WHAT you are asking.       (probabilistic, ~60 ms)
    The verified index decides the ANSWER. (exact, traceable)

The model never sees a question whose answer is Quranic text, and never
generates text of its own. Where the *shape* of the question is unambiguous
(two numbers and the word "ayah"; "كم مرة"), code decides the intent and the
UI says so. Where an answer does not exist (surah 200, ayah 1:99), code says
that too — it never falls back to a guess.
"""
import re
from quran_index import get_index, fold

# Intents. Bare labels on purpose: described labels were measured to make the
# model *worse* (77% -> 60% on 30 paraphrases). Keep them short.
INTENTS = {
    "بحث عن كلمة":        "word_lookup",
    "عدد مرات التكرار":   "count",
    "موقع الآية":         "locate_ayah",
    "محتويات الصفحة":     "page_contents",
    "معلومات السورة":     "surah_info",
}
QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": "ما هو قصد المستخدم من هذه العبارة؟",
        "criteria": {k: k for k in INTENTS},
    }
}

MAX_CHARS = 300
_ARABIC = re.compile(r"[ء-يٱ]")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def precheck(text):
    """Reject before the model runs. Returns an Arabic message, or None if fine."""
    if not isinstance(text, str) or not text.strip():
        return "اكتب سؤالاً أولاً"
    if len(text) > MAX_CHARS:
        return f"السؤال طويل — الحد {MAX_CHARS} حرفاً"
    if not _ARABIC.search(text):
        return "اكتب سؤالك بالعربية"
    return None


def _numbers(text):
    return [int(n) for n in re.findall(r"\d+", text.translate(_AR_DIGITS))]


_ORDINALS = {
    "الاولي": 1, "الثانيه": 2, "الثالثه": 3, "الرابعه": 4, "الخامسه": 5,
    "السادسه": 6, "السابعه": 7, "الثامنه": 8, "التاسعه": 9, "العاشره": 10,
}


def _ordinal(text):
    f = fold(text)
    for k, v in _ORDINALS.items():
        if k in f:
            return v
    return None


# ---------- term extraction (question side: fuzzy allowed) ----------

_QUOTED = re.compile(r'[«"“\'](.+?)[»"”\']')
_LEAD = r"(?:كلمة|كلمه|لفظ|اسم|عبارة|عباره)"
_VERB = r"(?:ذكر[ت]?|وردت?|ورد|تكرر[ت]?|جاء[ت]?|جات|نزل[ت]?|مذكور[ةه]?|مواضع)"
_AFTER = re.compile(rf"(?:{_LEAD}|{_VERB})\s+(?:{_LEAD}\s+)?(.{{2,60}}?)\s*[؟?.!]?$")
_STOP = {"في", "من", "على", "عن", "كم", "مرة", "مره", "مرات", "وين", "أين", "اين",
         "هل", "ما", "وش", "ايش", "الكريم", "القرآن", "القران", "المصحف", "بالقرآن",
         "بالقران", "كلمة", "كلمه", "لفظ", "اسم", "ذكر", "ذكرت", "وردت", "ورد",
         "تكرر", "تكررت", "جاء", "جاءت", "جات", "هي", "هو", "عدد", "بس", "يعني",
         "لي", "ابغى", "ابي", "ودي", "دور", "اطلع", "احسب", "مواضع", "مذكورة",
         "مذكوره", "آيات", "ايات", "أي", "اي", "الي", "اللي", "عبارة", "عباره"}


def _candidates(text):
    """Ordered guesses at the word being asked about, most likely first."""
    m = _AFTER.search(text.strip())
    tail = [t for t in m.group(1).split() if t not in _STOP] if m else []
    clean = re.sub(r"[؟?.,!،]", " ", text).split()
    rest = [t for t in clean if t not in _STOP and not t.translate(_AR_DIGITS).isdigit()]
    ordered = tail + [t for t in rest if t not in tail]
    if not ordered and len(clean) == 1:      # the whole query IS the word (e.g. "من")
        ordered = clean
    return ordered


def extract_term(text, ix):
    """Return (term, is_phrase). Prefers a candidate that exists in the text."""
    m = _QUOTED.search(text)
    if m:
        q = m.group(1).strip()
        return q, len(q.split()) > 1
    cands = _candidates(text)
    if not cands:
        return "", False
    # a multi-word tail that occurs verbatim is a phrase ("بسم الله")
    if len(cands) >= 2:
        phrase = " ".join(cands[:2])
        if ix.find_phrase(phrase)["count"]:
            return phrase, True
    for c in cands:
        if ix.by_fold.get(fold(c)):
            return c, False
    for c in cands:
        if ix.has_hits(c):
            return c, False
    return cands[0], False


# ---------- structural override (code decides where the shape is unambiguous) ----------

_HAS_AYAH = re.compile(r"آي[ةه]|\bاي[ةه]\b|الآي[ةه]|الاي[ةه]")
_HAS_PAGE = re.compile(r"صفح[ةه]|\bص\s*\d")
_HAS_SURAH = re.compile(r"سور[ةه]")
_COUNT = re.compile(r"كم\s+مر[ةه]|مرات|تكرر|عدد\s+مرات|احسب")
_SURAH_FACT = re.compile(r"كم\s+آي[ةه]|عدد\s+(?:آيات|الآيات|كلمات|الكلمات)|كم\s+كلم[ةه]|"
                         r"مكي[ةه]|مدني[ةه]|معلومات|تفاصيل|عرفني|تعريف")
_REF = re.compile(r"\b(\d{1,3})\s*[:：]\s*(\d{1,3})\b")


def refine(intent, text, confidence):
    """Deterministic override for shapes a classifier cannot see. Returns (intent, reason)."""
    t = text.translate(_AR_DIGITS)
    nums = _numbers(t)
    has_ayah, has_page, has_surah = _HAS_AYAH.search(t), _HAS_PAGE.search(t), _HAS_SURAH.search(t)

    if _REF.search(t) and not has_page:
        return "موقع الآية", "structure: surah:ayah reference"
    if has_ayah and (len(nums) >= 2 or (nums and has_surah) or (has_surah and _ordinal(t))):
        return "موقع الآية", "structure: ayah + surah + number"
    if has_page and len(nums) == 1 and not has_ayah:
        return "محتويات الصفحة", "structure: page + one number"
    if _COUNT.search(t):
        return "عدد مرات التكرار", "structure: كم مرة / تكرر"
    if has_surah and _SURAH_FACT.search(t) and not _COUNT.search(t):
        return "معلومات السورة", "structure: surah + fact word"
    if not has_ayah and not has_page and not has_surah and len(nums) == 2 and len(t.split()) <= 3:
        return "موقع الآية", "structure: two bare numbers"
    return intent, None


# ---------- answer (answer side: exact or a clear refusal) ----------

def _need(msg):
    return {"need": msg}, "need_more"


def answer(intent, text, ix=None):
    """Execute an intent against the exact index. Returns (payload, kind)."""
    ix = ix or get_index()
    t = text.translate(_AR_DIGITS)
    nums = _numbers(t)
    code = INTENTS.get(intent, "word_lookup")

    if code == "locate_ayah":
        m = _REF.search(t)
        if m:
            s, a = int(m.group(1)), int(m.group(2))
        elif len(nums) >= 2:
            s, a = nums[0], nums[1]
        else:
            s = ix.surah_by_name(t)
            a = nums[0] if nums else _ordinal(t)
            if s is None:
                return _need("أعطني رقم السورة ورقم الآية، مثل: سورة 2 آية 255")
            if a is None:
                return _need(f"أي آية من سورة {ix.surah_name(s)}؟ أعطني رقمها")
        if not ix.valid_surah(s):
            return _need(f"لا توجد سورة رقم {s} — أرقام السور من 1 إلى 114")
        if not 1 <= a <= ix.ayah_count(s):
            return _need(f"سورة {ix.surah_name(s)} فيها {ix.ayah_count(s)} آية فقط، ما فيها آية {a}")
        return ix.locate_ayah(s, a), "locate_ayah"

    if code == "page_contents":
        if not nums:
            return _need("أعطني رقم الصفحة، مثل: صفحة 604")
        p = nums[0]
        if not ix.valid_page(p):
            return _need(f"لا توجد صفحة {p} — الصفحات من 1 إلى {ix.page_count}")
        return ix.page_contents(p), "page_contents"

    if code == "surah_info":
        if nums:
            n = nums[0]
            if not ix.valid_surah(n):
                return _need(f"لا توجد سورة رقم {n} — أرقام السور من 1 إلى 114")
        else:
            n = ix.surah_by_name(t)
            if n is None:
                return _need("ما لقيت سورة بهذا الاسم — اكتب اسمها أو رقمها (1 إلى 114)")
        return ix.surah_info(n), "surah_info"

    term, is_phrase = extract_term(text, ix)
    if not term:
        return _need("أي كلمة تقصد؟ اكتبها بين علامتي تنصيص، مثل: «الرحمن»")
    if code == "count":
        r = ix.count_phrase(term) if is_phrase else ix.count_word(term)
        return {"term": term, **r}, "count"
    r = ix.find_phrase(term) if is_phrase else ix.find_word(term)
    return {"term": term, **r}, "word_lookup"
