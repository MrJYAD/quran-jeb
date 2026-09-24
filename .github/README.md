# quran-jeb — Arabic questions, exact Quranic answers

**Try it in the browser:** [huggingface.co/spaces/IJyad/quran-jeb](https://huggingface.co/spaces/IJyad/quran-jeb) · code: [github.com/MrJYAD/quran-jeb](https://github.com/MrJYAD/quran-jeb) · model: [IJyad/jeb](https://huggingface.co/IJyad/jeb)

Ask in plain Arabic — *وين ذكرت كلمة الرحمن؟* — and get every occurrence with its exact
address: surah, ayah, page, line, juz, and the full ayah with the word highlighted.

Two parts, kept strictly apart:

```
jeb (IJyad/jeb)   decides WHAT you are asking     probabilistic · ~60 ms on CPU · 5 intents
the index         decides the ANSWER              exact · traceable · 77,432 words
```

The model classifies the *shape of the question* into one of five intents. It never sees a
question whose answer is Quranic text, never generates a character of it, and never decides
what an ayah says. Every answer is a string lookup into a source-verified copy of the Ḥafṣ
muṣḥaf that ships in this repo.

## Run

```bash
pip install -r requirements.txt
python server.py            # → http://127.0.0.1:8001
```

- Python ≥ 3.10 (tested on 3.11 and 3.12). `transformers` 4.57 and 5.x both work.
- First run downloads the model weights (710 MB) into the Hugging Face cache; after that
  the model runs offline. The web fonts (IBM Plex) still need a connection and fall back to
  a system face without one.
- Needs about **2.5 GB of RAM** while running. One inference at a time — it is a
  single-user demo, not a service.
- Open **`127.0.0.1:8001`**, not `localhost`: on Windows, `localhost` resolves to IPv6 first
  and non-browser clients wait ~2 s per request. Browsers race both and are fine.
- `PORT=` changes the port. `HOST=0.0.0.0` exposes it beyond the machine — only do that on
  purpose; there is no authentication.

Test without the model (`python tests/test_smoke.py`); the HTTP checks run too if the server
is up.

### As a Hugging Face Space

The same repo runs as a Docker Space (`Dockerfile`, `app_port: 8001`). The free tier is
2 vCPU / 16 GB RAM: the **first request after a restart is a cold start** — the Space
downloads the 710 MB weights and loads them, which takes a minute or two, and the page shows
"جاري تحميل النموذج…" until then. The ~60 ms figure below is warm only. Set the Space's
hardware to CPU basic; nothing here needs a GPU.

## The Quran text

`data/hafs.csv` (every word with its address, Uthmani text and imlāʾī spelling) and
`data/catalog.json` (the 114 surahs) are copied verbatim from
[quran-ws/quran-text](https://github.com/quran-ws/quran-text), which derives them from the
King Fahd Complex `UthmanicHafs-v-3.0` release. Commit, checksums and licence
(CC BY 4.0) are in `data/NOTICE.md`. The index loads them read-only at startup in about
0.6 s. Nothing else is fetched.

## What it answers

| Intent | Example | Answer comes from |
|---|---|---|
| بحث عن كلمة | وين ذكرت كلمة الرحمن؟ | index → 45 exact positions |
| عدد مرات التكرار | كم مرة تكررت كلمة الصلاة؟ | index → 58 |
| موقع الآية | وش صفحة سورة ٢ آية ٢٥٥؟ | index → page 42 |
| محتويات الصفحة | وش في صفحة ٦٠٤؟ | index → surahs, juz, first and last ayah |
| معلومات السورة | كم آية في سورة الكهف؟ | catalog → 110 ayahs, pages, juz |

Also: a quoted phrase — `«بسم الله»` — is searched as consecutive words; a word with a
prefix or suffix (الصبر → بالصبر، فاصبر، صبروا) falls back to a stem match; and a word that
is written as one today but as two in the muṣḥaf (عبدالله → عَبْدُ ٱللَّهِ، رسولالله) is
split and matched as consecutive words. Each kind is labelled — *مطابقة تامة*, *مطابقة
بالجذر*, *كلمتان في المصحف* — so nobody mistakes which kind of match produced the number.

## Design rules

**The exact path never guesses.** If a surah, ayah or page does not exist, the reply says
so and names the valid range — سورة الفاتحة فيها 7 آية فقط — instead of resolving to the
nearest thing that does. Fuzzy matching is allowed only on the question side (which word
did you mean), never on the answer side.

**Structure overrides the model, visibly.** Some question shapes a classifier cannot see:
two numbers and the word آية, or كم مرة. Code recognises those and routes them, and the page
says so — it shows what the model answered, its confidence, and why it was overridden. On
30 paraphrases the model alone routes 70 %; with the rules, 30/30.

**Short labels, on purpose.** Giving the five intents descriptive sentences was measured and
made routing *worse* (77 % → 60 %); everything drifted toward "بحث عن كلمة". The labels stay
bare.

**Input is checked before the model runs.** Empty, non-Arabic, or over 300 characters is
refused with a plain message. Malformed requests get a 400, oversized ones a 413.

## Measured

| | |
|---|---|
| Index load | 0.6 s (77,432 words, 604 pages) |
| Intent classification | ~60 ms (CPU, 4 threads) |
| Word lookup, 45 hits (الرحمن) | ~1 ms |
| Word lookup, 2,763 hits (من) | ~12 ms |
| Whole request, warm | 60–100 ms at `127.0.0.1` |
| Intent routing, 30 paraphrases | model 70 % · model + rules 100 % |
| Lighthouse, desktop | accessibility 100 · best practices 100 · SEO 100 |

## What it deliberately does not do

- **Tafsir or meaning.** Out of scope for both the model and the data.
- **Tajweed.** `quran-tajweed` in the quran-ws stack has it annotated.
- **Other riwāyāt.** Ḥafṣ only. quran-text publishes six more in the same format, so
  adding them is a data drop, not a code change.
- **Anything generative about scripture.** By design.

## Files

| | |
|---|---|
| `quran_index.py` | the exact index — the only thing that produces an answer |
| `quran_router.py` | intents, input checks, term extraction, structural rules, range checks |
| `server.py` | `POST /api/quran`, `GET /api/status`; serves `static/` |
| `model.py` | the jeb loader, identical to the one in `IJyad/jeb` |
| `static/` | the page (Arabic, RTL, light and dark) |
| `data/` | the text, its licence, and `NOTICE.md` |
| `tests/test_smoke.py` | the numbers that must not change |

## Licence

Code: Apache 2.0 (`LICENSE`). Text: CC BY 4.0 from Quran.ws (`data/NOTICE.md`).
Model: [IJyad/jeb](https://huggingface.co/IJyad/jeb), Apache 2.0.
