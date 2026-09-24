"""Exact index over the vendored Quran text (Ḥafṣ).

Nothing here is probabilistic. Every answer is a lookup into `data/hafs.csv`,
which is source-verified and traceable to its printed KFGQPC edition (see
data/NOTICE.md). The model never touches this file; it only decides which of
these functions to call.

Rule for this file: fuzzy matching is allowed on the *question* side (finding
the word the user meant), never on the *answer* side. If a surah, ayah or page
does not exist, say so — do not guess a neighbour.
"""
import csv, io, json, os, re, unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# The text ships with this repo. Source, commit and licence: data/NOTICE.md
WORDS_FILE = os.path.join(HERE, "data", "hafs.csv")   # plain text: no LFS needed anywhere
CATALOG_FILE = os.path.join(HERE, "data", "catalog.json")

SURAH_COUNT = 114

# Arabic diacritics + tatweel: stripped only for MATCHING, never for display.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_AL = ("ال", "أل", "ٱل")


def fold(s: str) -> str:
    """Normalise for search only. Display always uses the original text."""
    s = unicodedata.normalize("NFC", s)
    s = _DIACRITICS.sub("", s)
    s = (s.replace("ٱ", "ا")          # alif wasla  -> alif
           .replace("آ", "ا")          # alif madda  -> alif
           .replace("أ", "ا").replace("إ", "ا")
           .replace("ى", "ي")          # alif maqsura -> ya
           .replace("ة", "ه"))         # ta marbuta   -> ha
    return s.strip()


def bare(s: str) -> str:
    """fold() and drop a leading definite article — for surah-name comparison."""
    s = fold(s)
    for p in _AL:
        if s.startswith(p):
            return s[len(p):].strip()
    return s


class QuranIndex:
    def __init__(self):
        self.words = []                    # row dicts, in mushaf order; index == position
        self.by_fold = defaultdict(list)   # folded form -> words
        self.by_ayah = {}                  # (surah, ayah) -> words
        self.by_page = {}                  # page -> words
        self.by_surah = {}                 # surah -> words
        self.surahs = {}                   # surah -> catalog entry
        self.page_count = 0
        self._load()

    # ---------- load ----------

    def _load(self):
        for p in (CATALOG_FILE, WORDS_FILE):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Quran data not found: {p}\n"
                    "Expected data/hafs.csv and data/catalog.json next to quran_index.py.")
        with io.open(CATALOG_FILE, encoding="utf-8") as f:
            for s in json.load(f)["surahs"]:
                self.surahs[s["number"]] = s

        with io.open(WORDS_FILE, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rec = {
                    "position": int(row["position"]),
                    "surah": int(row["surah"]),
                    "ayah": int(row["ayah"]),
                    "page": int(row["page"]),
                    "line": int(row["line"]),
                    "juz": int(row["juz"]),
                    "text": row["text"],
                    "imlai": row["rasm_imlai"],
                }
                rec["fold"] = fold(rec["imlai"])          # precomputed once, used by every scan
                rec["fold_u"] = fold(rec["text"])
                self.words.append(rec)
                for key in {rec["fold"], rec["fold_u"]}:
                    if key:
                        self.by_fold[key].append(rec)
                self.by_ayah.setdefault((rec["surah"], rec["ayah"]), []).append(rec)
                self.by_page.setdefault(rec["page"], []).append(rec)
                self.by_surah.setdefault(rec["surah"], []).append(rec)
        self.page_count = max(self.by_page)

    # ---------- ranges ----------

    def valid_surah(self, n):
        return isinstance(n, int) and 1 <= n <= SURAH_COUNT

    def valid_page(self, n):
        return isinstance(n, int) and 1 <= n <= self.page_count

    def ayah_count(self, surah):
        return self.surahs[surah]["ayah_count"]["hafs"]

    def surah_name(self, surah):
        return self.surahs[surah]["name_ar"]

    # ---------- word matching (question side: fuzzy allowed) ----------

    def _stem_hits(self, f):
        """Substring match on the stem inside prefixed forms (بالصبر، فاصبر، صبروا)."""
        stem = f[2:] if f.startswith("ال") and len(f) > 4 else f
        if len(stem) < 3:
            return []
        return [w for w in self.words if stem in w["fold"]]

    def _split_hits(self, f):
        """One written word that the mushaf spells as two consecutive words.

        عبدالله is written as one word everywhere today; the mushaf has عَبْدُ ٱللَّهِ.
        Try every split point; keep the one whose halves occur consecutively.
        Returns (hits, "first second") or ([], None).
        """
        best, best_phrase = [], None
        for i in range(2, len(f) - 1):
            a, b = f[:i], f[i:]
            if not self.by_fold.get(a) or not self.by_fold.get(b):
                continue
            hits = self._phrase_hits([a, b])
            if len(hits) > len(best):
                best, best_phrase = hits, f"{a} {b}"
        return best, best_phrase

    def has_hits(self, query):
        return bool(self._match(query)[0])

    def _match(self, query):
        """-> (hits, how, shown). `how` is exact | stem | split; `shown` is the
        form actually searched (differs from the query only for split)."""
        f = fold(query)
        if not f:
            return [], "exact", query
        exact = self.by_fold.get(f, [])
        if exact:
            return exact, "exact", query
        stem = self._stem_hits(f)
        if stem:
            return stem, "stem", query
        hits, phrase = self._split_hits(f)
        if hits:
            return hits, "split", phrase
        return [], "stem", query

    def _summarise(self, hits, how, phrase=False, shown=None):
        return {
            "count": len(hits),
            "match": how,
            "phrase": phrase,
            "shown": shown,                      # the form searched, when it differs from the query
            "surah_breakdown": [
                {"surah": s, "name": self.surah_name(s), "count": c}
                for s, c in Counter(h["surah"] for h in hits).most_common()
            ],
            "forms": [f"{t} ({c})" for t, c in
                      Counter(h["imlai"] for h in hits).most_common(8)],
            "hits": [self._decorate(h) for h in hits],
        }

    def find_word(self, query):
        """Every occurrence of a word, each with its exact address and its ayah."""
        hits, how, shown = self._match(query)
        return self._summarise(hits, how, phrase=(how == "split"),
                               shown=shown if how == "split" else None)

    def _phrase_hits(self, parts):
        n = len(self.words)
        hits = []
        for w in self.by_fold.get(parts[0], []):
            p = w["position"]
            if p + len(parts) > n:
                continue
            ok = all(parts[i] in (self.words[p + i]["fold"], self.words[p + i]["fold_u"])
                     for i in range(1, len(parts)))
            if ok:
                hits.append(w)
        return hits

    def find_phrase(self, phrase):
        """Consecutive-word match. Two or more words; exact forms only."""
        parts = [fold(t) for t in phrase.split() if fold(t)]
        if len(parts) < 2:
            return self.find_word(phrase)
        return self._summarise(self._phrase_hits(parts), "exact", phrase=True)

    def count_word(self, query):
        hits, how, shown = self._match(query)
        return {"count": len(hits), "match": how, "phrase": how == "split",
                "shown": shown if how == "split" else None,
                "surahs": len({h["surah"] for h in hits}),
                "pages": len({h["page"] for h in hits}),
                "forms": [f"{t} ({c})" for t, c in
                          Counter(h["imlai"] for h in hits).most_common(8)]}

    def count_phrase(self, phrase):
        r = self.find_phrase(phrase)
        return {"count": r["count"], "match": "exact", "phrase": True,
                "surahs": len({h["surah"] for h in r["hits"]}),
                "pages": len({h["page"] for h in r["hits"]}), "forms": []}

    # ---------- exact lookups (answer side: no guessing) ----------

    def locate_ayah(self, surah, ayah):
        ws = self.by_ayah.get((surah, ayah))
        if not ws:
            return None
        return {
            "surah": surah, "name": self.surah_name(surah), "ayah": ayah,
            "page": ws[0]["page"], "line": ws[0]["line"], "juz": ws[0]["juz"],
            "pages": sorted({w["page"] for w in ws}),
            "text": " ".join(w["text"] for w in ws),
            "word_count": len(ws),
        }

    def ayah_text(self, surah, ayah):
        return " ".join(w["text"] for w in self.by_ayah.get((surah, ayah), []))

    def page_contents(self, page):
        ws = self.by_page.get(page)
        if not ws:
            return None
        spans = sorted({(w["surah"], w["ayah"]) for w in ws})
        return {
            "page": page, "word_count": len(ws), "juz": ws[0]["juz"],
            "surahs": [{"surah": s, "name": self.surah_name(s)}
                       for s in sorted({w["surah"] for w in ws})],
            "first": {"surah": spans[0][0], "ayah": spans[0][1]},
            "last": {"surah": spans[-1][0], "ayah": spans[-1][1]},
        }

    def surah_info(self, surah):
        m = self.surahs.get(surah)
        if not m:
            return None
        ws = self.by_surah[surah]
        return {
            "surah": surah, "name": m["name_ar"], "name_en": m["name_en"],
            "revelation": "مكية" if m["revelation"] == "makki" else "مدنية",
            "ayah_count": m["ayah_count"]["hafs"], "word_count": m["word_count"],
            "pages": [ws[0]["page"], ws[-1]["page"]],
            "juz": sorted({w["juz"] for w in ws}),
        }

    _NAME_SKIP = {"سورة", "سوره", "عن", "معلومات", "في", "تفاصيل", "عرفني", "على", "كم",
                  "آية", "آيات", "ايه", "اية", "كلمات", "كلمة", "مكية", "مدنية", "مكيه",
                  "مدنيه", "ولا", "أو", "او", "وش", "ايش", "هل", "صفحة", "صفحه", "عدد",
                  "موقع", "وين", "أين", "اين", "الآية", "الايه", "من"}

    def surah_by_name(self, q):
        """Resolve a surah name in the question, or None.

        Exact match on the bare name first. A prefix match is allowed only in one
        direction — the *surah name* may start with a *query token* of three or
        more letters (الفاتح -> الفاتحة). The reverse (a query token starting with
        a surah name) is never allowed: it is how "صفحة" used to resolve to سورة ص.
        """
        q = re.sub(r"[؟?.,!،:؛\"«»()]", " ", q)              # punctuation is glued to Arabic words
        toks = [t for t in fold(q).split()
                if t and t not in self._NAME_SKIP and not t.isdigit()]
        if not toks:
            return None
        cands = {bare(t) for t in toks} | {bare(" ".join(toks))}
        cands.discard("")
        names = {n: bare(m["name_ar"]) for n, m in self.surahs.items()}
        for n, nm in names.items():
            if nm in cands:
                return n
        for n, nm in names.items():
            for c in cands:
                if len(c) >= 3 and nm.startswith(c):
                    return n
        return None

    def _decorate(self, w):
        return {
            "position": w["position"], "surah": w["surah"], "ayah": w["ayah"],
            "page": w["page"], "line": w["line"], "juz": w["juz"],
            "text": w["text"], "imlai": w["imlai"],
            "surah_name": self.surah_name(w["surah"]),
            "ayah_text": self.ayah_text(w["surah"], w["ayah"]),
        }


_index = None


def get_index():
    global _index
    if _index is None:
        _index = QuranIndex()
    return _index


if __name__ == "__main__":
    import sys, time
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    t = time.time()
    ix = get_index()
    print(f"loaded {len(ix.words):,} words, {ix.page_count} pages in {time.time()-t:.2f}s")
    for term in ("الرحمن", "الصبر", "الله"):
        t = time.time(); r = ix.find_word(term)
        print(f"{term:8} {r['count']:5} hits ({r['match']}) in {(time.time()-t)*1000:.0f} ms")
