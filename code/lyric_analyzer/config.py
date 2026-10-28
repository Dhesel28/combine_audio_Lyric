"""
lyric_analyzer/config.py
========================
Word sets and signal weights for section-level and line-level scoring.
"""

# ── Hook / energy / sentiment word sets ───────────────────────────────────────

HOOK_WORDS = {
    "love", "baby", "feel", "yeah", "oh", "never", "always", "tonight",
    "heart", "wanna", "gonna", "need", "want", "know", "fire", "crazy",
    "miss", "cry", "pain", "soul", "dream", "broken", "beautiful", "hurt",
    "sorry", "lonely", "empty", "lost", "scared", "brave",
    "dance", "run", "fall", "rise", "stay", "fly", "let", "go", "burn",
    "mine", "kiss", "hold", "real", "together", "alone", "close", "touch",
    "forever", "everything", "nothing", "way", "eyes", "wild", "world",
    "ooh", "wow", "show", "glow", "flow", "free", "deep", "high", "life",
    "right", "light", "night", "fight", "bright", "tight",
}

ENERGY_WORDS = {
    "fire", "lit", "hype", "wild", "crazy", "turnt", "vibe", "bounce",
    "drop", "bang", "banger", "loud", "scream", "jump", "move", "go",
    "fast", "hard", "rage", "beast", "king", "queen", "boss", "flex",
    "drip", "swag", "pop", "rock", "hit", "unstoppable", "epic", "run",
    "rise", "shine", "burn", "blaze", "fly", "rush", "push", "break",
}

POSITIVE_WORDS = {
    "love", "happy", "joy", "smile", "beautiful", "wonderful", "amazing",
    "great", "good", "best", "perfect", "shine", "bright", "warm", "sweet",
    "dream", "hope", "win", "glory", "bless", "grace", "light", "gold",
    "free", "peace", "together", "forever", "always", "paradise", "heaven",
}

NEGATIVE_WORDS = {
    "hate", "pain", "cry", "tears", "hurt", "break", "lost", "alone",
    "dark", "cold", "dead", "fall", "fail", "lie", "cheat", "betray",
    "gone", "miss", "sad", "empty", "numb", "sorrow", "grief", "regret",
    "fear", "wrong", "bad", "worst", "hell", "die", "kill", "blood",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "as", "if", "so",
}

# ── Section-level weights (12 signals, sum to 1.0) ────────────────────────────
SECTION_WEIGHTS: dict[str, float] = {
    "hook_vocabulary":       0.15,
    "cross_song_repetition": 0.13,
    "internal_repetition":   0.12,
    "sentiment_extremity":   0.10,
    "end_rhyme_density":     0.09,
    "line_conciseness":      0.09,
    "syllable_consistency":  0.08,
    "phonetic_openness":     0.07,
    "pronoun_relatability":  0.07,
    "internal_rhyme":        0.05,
    "energy_markers":        0.03,
    "position_score":        0.02,
}
assert abs(sum(SECTION_WEIGHTS.values()) - 1.0) < 1e-9

# ── Line-level weights (6 signals, sum to 1.0) ────────────────────────────────
LINE_WEIGHTS: dict[str, float] = {
    "repetition":      0.28,
    "energy":          0.22,
    "rhyme":           0.15,
    "brevity":         0.13,
    "section_bonus":   0.12,
    "emotional_punch": 0.10,
}
assert abs(sum(LINE_WEIGHTS.values()) - 1.0) < 1e-9

# ── Section types that get line-level bonus ────────────────────────────────────
HOOK_SECTION_KEYS = {"chorus", "hook", "refrain", "bridge", "pre-chorus"}

# ── Default clip length (lines) ───────────────────────────────────────────────
CLIP_LINES = 5
