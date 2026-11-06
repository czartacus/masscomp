"""Segment inventory with binary feature vectors (plan §2).

Features (12 bits):
  0-2  place  (labial, coronal, dorsal)
  3-5  manner (stop, fricative, sonorant)
  6    voice
  7    nasal
  8    continuant
  9    high
  10   back
  11   round
EPS is the index of the empty output symbol (deletion), equal to len(SEGMENTS).
"""
import numpy as np

_C = [
    # name, place, manner, voice, nasal, cont, high, back, round
    ("p", "lab", "stop", 0, 0, 0, 0, 0, 0),
    ("b", "lab", "stop", 1, 0, 0, 0, 0, 0),
    ("t", "cor", "stop", 0, 0, 0, 0, 0, 0),
    ("d", "cor", "stop", 1, 0, 0, 0, 0, 0),
    ("k", "dor", "stop", 0, 0, 0, 0, 1, 0),
    ("g", "dor", "stop", 1, 0, 0, 0, 1, 0),
    ("f", "lab", "fric", 0, 0, 1, 0, 0, 0),
    ("v", "lab", "fric", 1, 0, 1, 0, 0, 0),
    ("θ", "cor", "fric", 0, 0, 1, 0, 0, 0),
    ("ð", "cor", "fric", 1, 0, 1, 0, 0, 0),
    ("s", "cor", "fric", 0, 0, 1, 1, 0, 0),
    ("z", "cor", "fric", 1, 0, 1, 1, 0, 0),
    ("ʃ", "cor", "fric", 0, 0, 1, 1, 1, 0),
    ("x", "dor", "fric", 0, 0, 1, 0, 1, 0),
    ("h", "dor", "fric", 0, 0, 1, 1, 1, 0),
    ("m", "lab", "son", 1, 1, 0, 0, 0, 0),
    ("n", "cor", "son", 1, 1, 0, 0, 0, 0),
    ("ŋ", "dor", "son", 1, 1, 0, 0, 1, 0),
    ("l", "cor", "son", 1, 0, 1, 0, 0, 0),
    ("r", "cor", "son", 1, 0, 1, 1, 0, 0),
    ("w", "lab", "son", 1, 0, 1, 1, 1, 1),
    ("j", "dor", "son", 1, 0, 1, 1, 0, 0),
    ("ʔ", "dor", "stop", 0, 0, 0, 0, 0, 0),
    ("ʕ", "dor", "fric", 1, 0, 1, 0, 1, 0),
]
_V = [
    # name, high, back, round  (vowels: place=none, manner=son, voice=1, cont=1)
    ("a", 0, 1, 0),
    ("e", 0, 0, 0),
    ("i", 1, 0, 0),
    ("o", 0, 1, 1),
    ("u", 1, 1, 1),
]

# Extended inventory for real-language probes (MASSCOMP_INVENTORY=extended): the plan's 29 segments
# keep their indices, these are appended. Affricates set both the stop and the fricative manner bit;
# the three free vowel cells of the (high, back, round) cube take the front rounded and central vowels.
_XC = [
    ("ʒ", "cor", "fric", 1, 0, 1, 1, 1, 0),
    ("ɲ", "cor", "son", 1, 1, 0, 1, 0, 0),
    ("ʎ", "cor", "son", 1, 0, 1, 0, 1, 0),
    ("ts", "cor", "affr", 0, 0, 1, 1, 0, 0),
    ("dz", "cor", "affr", 1, 0, 1, 1, 0, 0),
    ("tʃ", "cor", "affr", 0, 0, 1, 1, 1, 0),
    ("dʒ", "cor", "affr", 1, 0, 1, 1, 1, 0),
    ("ç", "dor", "fric", 0, 0, 1, 1, 0, 0),
]
_XV = [
    ("y", 1, 0, 1),
    ("ø", 0, 0, 1),
    ("ɨ", 1, 1, 0),
]
_EXTENDED = __import__("os").environ.get("MASSCOMP_INVENTORY", "plan") == "extended"
_ALL_C = _C + (_XC if _EXTENDED else [])
_ALL_V = _V + (_XV if _EXTENDED else [])
SEGMENTS = [c[0] for c in _C] + [v[0] for v in _V] + ([c[0] for c in _XC] + [v[0] for v in _XV] if _EXTENDED else [])
N_SEG = len(SEGMENTS)          # 29 (plan) or 40 (extended)
EPS = N_SEG                    # deletion symbol index
N_OUT = N_SEG + 1              # outputs include EPS
INDEX = {s: i for i, s in enumerate(SEGMENTS)}
CONSONANTS = [INDEX[c[0]] for c in _ALL_C]
VOWELS = [INDEX[v[0]] for v in _ALL_V]


def _feat_c(row):
    _, place, manner, voice, nasal, cont, high, back, rnd = row
    f = np.zeros(12, dtype=np.int8)
    f[{"lab": 0, "cor": 1, "dor": 2}[place]] = 1
    if manner == "affr":
        f[3] = 1; f[4] = 1
    else:
        f[3 + {"stop": 0, "fric": 1, "son": 2}[manner]] = 1
    f[6:12] = [voice, nasal, cont, high, back, rnd]
    return f


def _feat_v(row):
    _, high, back, rnd = row
    f = np.zeros(12, dtype=np.int8)
    f[5] = 1; f[6] = 1; f[8] = 1
    f[9:12] = [high, back, rnd]
    return f


FEATURES = np.array([_feat_c(r) for r in _C] + [_feat_v(r) for r in _V]
                    + ([_feat_c(r) for r in _XC] + [_feat_v(r) for r in _XV] if _EXTENDED else []))   # (N_SEG, 12)
# Hamming distance between segments, scaled to [0,1]
DIST = (FEATURES[:, None, :] != FEATURES[None, :, :]).sum(-1) / 12.0        # (N_SEG, N_SEG)


def _strength(f):
    """A consonantal-strength rank from the features (voiceless stop 7 ... vowel 0): the direction of
    change that is common in attested sound change is downward (lenition, loss); upward is rare."""
    place, stop, fric, son, voice, nasal, cont, high = f[0:3].any(), f[3], f[4], f[5], f[6], f[7], f[8], f[9]
    if not place:                              # vowel
        return 0
    if stop and fric:                          # affricate
        return 6 if not voice else 5
    if stop:
        return 7 if not voice else 6
    if fric:
        return 5 if not voice else 4
    if nasal:
        return 3
    if son and cont and high:                  # glide
        return 1
    return 2                                   # liquid


STRENGTH = np.array([_strength(f) for f in FEATURES])                           # (N_SEG,)
FORTITION = np.maximum(0, STRENGTH[None, :] - STRENGTH[:, None]).astype(float)  # (s, s'): strength gained by s -> s' (0 for lenition)


# Environments for conditioned rules: natural-class predicates on the input segments either side
# of a position (BOUNDARY = N_SEG at the start and end of a form). A rule (s, o, env) adds its
# weight to s -> o wherever the environment holds; several can hold at one position, so the
# environment of a position is a bitmask. These are the classic conditioning environments of
# attested sound change (initial / final position, intervocalic lenition, palatalisation before
# front vowels, post-nasal changes); ENV_MASK[prev, next] is the mask of a position.
ENV_NAMES = ["#_", "_#", "V_", "_V", "V_V", "C_", "_C", "_Vfront", "_Vhigh", "N_"]
N_ENV = len(ENV_NAMES)
BOUNDARY = N_SEG
_IS_V = np.zeros(N_SEG + 1, bool); _IS_V[VOWELS] = True
_IS_FRONT = np.zeros(N_SEG + 1, bool); _IS_FRONT[[v for v in VOWELS if FEATURES[v][10] == 0]] = True
_IS_HIGH = np.zeros(N_SEG + 1, bool); _IS_HIGH[[v for v in VOWELS if FEATURES[v][9] == 1]] = True
_IS_NAS = np.zeros(N_SEG + 1, bool); _IS_NAS[[i for i in range(N_SEG) if FEATURES[i][7] == 1]] = True


def env_mask(prev, nxt):
    """Bitmask of the environments that hold between input segments prev and nxt (BOUNDARY at the ends)."""
    m = 0
    if prev == BOUNDARY: m |= 1 << 0
    if nxt == BOUNDARY: m |= 1 << 1
    pv, nv = _IS_V[prev], _IS_V[nxt]
    if pv: m |= 1 << 2
    if nv: m |= 1 << 3
    if pv and nv: m |= 1 << 4
    if prev != BOUNDARY and not pv: m |= 1 << 5
    if nxt != BOUNDARY and not nv: m |= 1 << 6
    if _IS_FRONT[nxt]: m |= 1 << 7
    if _IS_HIGH[nxt]: m |= 1 << 8
    if _IS_NAS[prev]: m |= 1 << 9
    return m


ENV_MASK = np.array([[env_mask(p, n) for n in range(N_SEG + 1)] for p in range(N_SEG + 1)], dtype=np.int64)


def envs_of(prev, nxt):
    """The environment indices that hold at a position (for proposals and reports)."""
    m = int(ENV_MASK[prev, nxt])
    return [k for k in range(N_ENV) if m >> k & 1]


def to_str(form):
    return "".join(SEGMENTS[i] for i in form)


def from_str(s):
    return tuple(INDEX[ch] for ch in s)
