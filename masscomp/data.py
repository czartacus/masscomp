"""Real-language wordlists for the diagnostic probes (spec §13 tier 1 style: known relationships and
attested proto-languages, never a claim).

Reads a Lexibank / CLDF wordlist (forms.csv with CLTS-segmented IPA) and maps every segment onto the
prototype's inventory. Use the extended inventory (MASSCOMP_INVENTORY=extended, 40 segments) - the
mapping below is written for it; on the plan's 29 segments the extra symbols fall back to the nearest
of those. Rules, in order: a segment that is already in the inventory maps to itself; length,
nasalisation, velarisation, gemination, dental and syllabic marks are stripped; a few IPA variants are
folded (ɛ → e, ɪ → i, ʊ → u, ɔ → o, ɑ ɒ ɐ æ ʌ → a, ə ɯ → ɨ, ʏ → y, œ ɵ → ø, ɾ ʁ ɹ → r, ʋ β → v, ɣ → ʕ,
ɕ → ʃ, c → k, ɟ → g, ɡ → g); diphthongs are split into their vowels; anything left is reported and
the form dropped. Multi-word forms ("+" in the segmentation) and synonyms are dropped / the first
single-word form is kept, since the prototype has no lexeme sets.
"""
import csv, os, unicodedata, collections
from .segments import INDEX, SEGMENTS

_FOLD = {"ɛ": "e", "ɪ": "i", "ʊ": "u", "ɔ": "o", "ɑ": "a", "ɒ": "a", "ɐ": "a", "æ": "a", "ʌ": "a", "ə": "ɨ", "ɯ": "ɨ",
         "ʏ": "y", "œ": "ø", "ɵ": "ø", "ɘ": "ɨ", "ɜ": "ɨ", "ʉ": "ɨ",
         "ɾ": "r", "ʁ": "r", "ɹ": "r", "ʀ": "r", "ɻ": "r", "ʋ": "v", "β": "v", "ɣ": "ʕ", "ʁ̞": "r", "ɕ": "ʃ", "ʂ": "ʃ", "ʑ": "ʒ", "ʐ": "ʒ",
         "c": "k", "ɟ": "g", "ɡ": "g", "ɫ": "l", "ɭ": "l", "ɳ": "n", "ɖ": "d", "ʈ": "t", "ɸ": "f", "χ": "x", "ħ": "h", "ʔ": "ʔ",
         "ʍ": "w", "ɥ": "j", "ɰ": "w", "ɦ": "h", "ʝ": "j", "ɧ": "ʃ", "ɱ": "m", "ɬ": "l", "ɴ": "ŋ", "ʲ": "", "ʰ": "", "ʷ": ""}
_FALLBACK_29 = {"ʒ": "ʃ", "ɲ": "n", "ʎ": "l", "ts": "s", "dz": "z", "tʃ": "ʃ", "dʒ": "ʃ", "ç": "x", "y": "u", "ø": "o", "ɨ": "e"}
_STRIP_CATS = {"Mn", "Lm", "Sk"}          # combining marks, modifier letters, modifier symbols


def map_segment(seg):
    """One CLTS segment -> list of inventory indices (a diphthong gives two), or None if unmappable."""
    if seg in INDEX:
        return [INDEX[seg]]
    d = unicodedata.normalize("NFD", seg)
    base = "".join(ch for ch in d if unicodedata.category(ch) not in _STRIP_CATS and ch != "ː")
    base = unicodedata.normalize("NFC", base)
    if base in INDEX:
        return [INDEX[base]]
    if base in _FOLD:
        base = _FOLD[base]
        if base == "":
            return []
        if base in INDEX:
            return [INDEX[base]]
    if base in _FALLBACK_29 and _FALLBACK_29[base] in INDEX:
        return [INDEX[_FALLBACK_29[base]]]
    # affricates written as two letters, or diphthongs: map piecewise
    if len(base) >= 2:
        out = []
        for ch in base:
            m = map_segment(ch)
            if m is None:
                return None
            out.extend(m)
        return out
    return None


def load_cldf(cldf_dir, languages, concepts=None, max_len=12, min_len=1):
    """{language_id: {parameter_id: form tuple}} for the given language ids, plus a report dict.
    concepts: optional list of parameter ids to keep (default: every concept with a form in every language)."""
    forms_by = collections.defaultdict(dict)
    unmapped = collections.Counter(); dropped = collections.Counter(); kept = collections.Counter()
    with open(os.path.join(cldf_dir, "forms.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            L, c = r["Language_ID"], r["Parameter_ID"]
            if L not in languages or (concepts is not None and c not in concepts) or c in forms_by[L]:
                continue
            segs = r["Segments"].split()
            if "+" in segs or "_" in segs:
                dropped[(L, "multiword")] += 1
                continue
            out = []
            ok = True
            for s in segs:
                m = map_segment(s)
                if m is None:
                    unmapped[s] += 1; ok = False
                    break
                out.extend(m)
            if not ok:
                dropped[(L, "unmapped")] += 1
                continue
            if not (min_len <= len(out) <= max_len):
                dropped[(L, "length")] += 1
                continue
            forms_by[L][c] = tuple(out); kept[L] += 1
    common = set.intersection(*[set(forms_by[L]) for L in languages]) if languages else set()
    report = dict(kept={L: kept[L] for L in languages}, common_concepts=len(common), unmapped=unmapped.most_common(40),
                  dropped={f"{L}:{why}": n for (L, why), n in sorted(dropped.items())})
    return {L: dict(forms_by[L]) for L in languages}, sorted(common), report


def to_str(form):
    return "".join(SEGMENTS[i] for i in form)
