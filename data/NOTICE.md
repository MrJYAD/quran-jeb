# Quran text — source and licence

The two data files in this folder are copied unmodified from the
[quran-ws/quran-text](https://github.com/quran-ws/quran-text) repository.

| file | what | sha256 (first 16) |
|---|---|---|
| `hafs.csv` | every word of the Ḥafṣ muṣḥaf with surah, ayah, page, line, juz, Uthmani text and imlāʾī spelling — 77,432 rows | `9458337a46e19c40` |
| `catalog.json` | the 114 surahs: Arabic/English names, revelation, ayah and word counts | `89ace55e18dd1b36` |

| | |
|---|---|
| Source repository | https://github.com/quran-ws/quran-text |
| Commit | `87d7691a0179dbb3cadbc2276581f0fdbbe476b1` (2026-09-13) |
| Stored form | uncompressed CSV (5.1 MB), so it needs no Git LFS on GitHub or the Hub; the gzip copied from upstream had sha256 `e1364d0869c146ac…` |
| Upstream text | KFGQPC `UthmanicHafs-v-3.0.zip`, sha256 `cdec7341b7c684e7…` (2026 release) |
| Riwāyah | Ḥafṣ ʿan ʿĀṣim |
| Licence | **CC BY 4.0** — full text in `LICENSE-CC-BY-4.0.txt` |

## Attribution

quran-text publishes its data under Creative Commons Attribution 4.0 International
**with a standing waiver of attribution for use inside a product**. This notice is kept
anyway, because the whole point of quran-jeb is that every answer is traceable to a
verified printed edition — the trail should be visible, not waived.

> Text data © Quran.ws, published as a waqf. Word-level data is derived by quran-text
> from the King Fahd Glorious Quran Printing Complex (KFGQPC) release named above.
> Waqf marks, sajdah lines and ۞ are held in a separate `marks` layer upstream and are
> not present in `hafs.csv`; see the upstream `docs/` for the reattachment rule.

## What quran-jeb does with it

Read-only. The text is loaded into an in-memory index at startup and searched by exact
(and stem) string match. No character of it is generated, rewritten, or passed through
the model. To refresh from upstream, replace the two files and update the checksums
and commit above.
