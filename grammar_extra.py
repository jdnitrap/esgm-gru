"""More hand-given English mechanics, layered on top of word_structure.py
without touching it -- same exact pattern (frozen, confirmed edges,
words wired to hub nodes for a category), just more dimensions than the
original 6 syntactic roles cover. Given for free, same reasoning as
ROLE_MAP: real structural facts about English, cheaper to state once
than to make the system re-derive from a corpus that's too sparse to
teach it (see the held-out accuracy experiment this replaces -- widening
the adjacency window only made things worse).

Every word below is a REAL word already in tiles.json -- nothing here
invents a category with no member. Words deliberately left out of a
given map (e.g. "you" from NUMBER) are a real "none" state, same
convention as ROLE_MAP's "not"/"yes"/"no".

Uses graph.grow() for hub nodes instead of hunting for free ids in the
cramped 262-298 range -- the reason that's now safe to do at all.

SYNTAX and MORPHOLOGY added 2026-09-14, from the mdbe/language_mechanics
worksheets merged into this repo (see EXPERIMENT_LOG.md) -- those two
mechanics are the ones that map cleanly onto ESGR's existing "one word
-> one hub" frozen-edge pattern. The other 10 mechanics in that
framework (Phonetics, Pragmatics, Discourse-the-mechanic [not to be
confused with this file's DISCOURSE dimension, which is really
Pragmatics-flavored], Sociolinguistics, etc.) are properties of a whole
utterance in context, not a fixed word->category fact, and don't fit
this pattern without real rethinking -- deliberately not attempted here.
"""
from word_structure import ROLE_MAP

TENSE_NAMES = ["PAST", "PRESENT"]
TENSE_MAP = {
    "sat": "PAST",
    "ran": "PAST",
    "is": "PRESENT",
    "likes": "PRESENT",
    "eats": "PRESENT",
}

NUMBER_NAMES = ["SINGULAR", "PLURAL"]
NUMBER_MAP = {
    "I": "SINGULAR", "he": "SINGULAR", "she": "SINGULAR", "it": "SINGULAR",
    "we": "PLURAL", "they": "PLURAL",
    # "you" deliberately absent: English collapses 2nd-person singular
    # and plural into one word -- a real ambiguity, not a gap to fill.
}

ANIMACY_NAMES = ["ANIMATE", "INANIMATE"]
ANIMACY_MAP = {
    "cat": "ANIMATE", "dog": "ANIMATE", "bird": "ANIMATE",
    "fish": "ANIMATE", "man": "ANIMATE",
    "mat": "INANIMATE", "house": "INANIMATE",
}

# "yes"/"no"/"not" are real words with no syntactic ROLE (word_structure.py
# leaves them role-less on purpose) -- but they aren't roleless in
# English grammar generally, they're just a DIFFERENT kind of category
# (discourse particles / negation) than NOUN/VERB/etc. Giving them their
# own dimension is the linguistically correct fix, not a patch onto ROLE.
#
# 2026-09-14: extended with real discourse-RELATION connectives, at
# explicit user request to look up published research rather than
# invent categories. Grounded in the Penn Discourse Treebank (PDTB),
# the standard annotation framework for this exact phenomenon: its
# four major semantic classes are TEMPORAL, CONTINGENCY, COMPARISON,
# and EXPANSION, each signaled by real, well-documented explicit
# connective words (PDTB3 alone catalogs dozens of contrast/concession
# connectives). This is the closest legitimate, citable, deterministic
# thing to "reasoning structure" that fits ESGR's hand-given-fact
# pattern -- cause/contrast/condition relations are lexically signaled,
# so detecting them is real parsing, not invented understanding.
DISCOURSE_NAMES = ["AFFIRM", "NEGATE", "NEGATOR",
                    "TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION"]
DISCOURSE_MAP = {
    "yes": "AFFIRM",
    "no": "NEGATE",
    "not": "NEGATOR",
    # TEMPORAL: sequence/synchrony connectives (PDTB)
    "then": "TEMPORAL", "after": "TEMPORAL", "before": "TEMPORAL",
    "when": "TEMPORAL", "until": "TEMPORAL", "once": "TEMPORAL",
    "meanwhile": "TEMPORAL",
    # CONTINGENCY: cause/condition connectives (PDTB)
    "because": "CONTINGENCY", "since": "CONTINGENCY", "so": "CONTINGENCY",
    "therefore": "CONTINGENCY", "thus": "CONTINGENCY", "if": "CONTINGENCY",
    "unless": "CONTINGENCY", "hence": "CONTINGENCY",
    # COMPARISON: contrast/concession connectives (PDTB)
    "but": "COMPARISON", "however": "COMPARISON", "although": "COMPARISON",
    "though": "COMPARISON", "while": "COMPARISON", "yet": "COMPARISON",
    "still": "COMPARISON", "whereas": "COMPARISON",
    # EXPANSION: elaboration/addition connectives (PDTB)
    "and": "EXPANSION", "also": "EXPANSION", "moreover": "EXPANSION",
    "furthermore": "EXPANSION", "additionally": "EXPANSION",
}

# Syntax: "How words combine into phrases and clauses. Who is the head?
# What depends on it?" (mdbe/language_mechanics_Column_key.csv). Real
# X-bar-theory distinction, not invented: NOUN/VERB/PRONOUN/PREPOSITION
# can each independently head a phrase (NP/VP/NP/PP); ARTICLE and
# ADJECTIVE only ever attach to and modify a head, never head one
# themselves. Derived FROM word_structure.ROLE_MAP, not hand-typed
# separately -- one word's syntax value is a fixed function of its role,
# so keeping ROLE_MAP as the single source of truth avoids the two ever
# silently disagreeing.
SYNTAX_NAMES = ["HEAD", "MODIFIER"]
_ROLE_TO_SYNTAX = {
    "NOUN": "HEAD", "VERB": "HEAD", "PRONOUN": "HEAD", "PREPOSITION": "HEAD",
    "ARTICLE": "MODIFIER", "ADJECTIVE": "MODIFIER",
}
SYNTAX_MAP = {word: _ROLE_TO_SYNTAX[role] for word, role in ROLE_MAP.items()
              if role in _ROLE_TO_SYNTAX}

# Morphology: "How words are built from smaller meaning pieces. What
# pieces is this word made of?" -- scoped to INFLECTION specifically
# (the worked example's "cat + -s (plural)" vs "sit + PAST -> sat
# (irregular)"), not derivation (a word like "consciousness" is built
# from "conscious" + "-ness", but that's a different morphological
# process and deliberately out of scope here). REGULAR = built by a
# predictable suffix rule; IRREGULAR = unpredictable form change
# (ablaut, suppletion). Bare/uninflected root forms (cat, dog, big...),
# and genuinely uncertain or non-inflectional cases (e.g. "physics" LOOKS
# like a plural but is a fixed lexical item, not physic+s; "does" is a
# real edge case left out rather than guessed) are deliberately left
# unmarked -- real "none" state, same convention as everywhere else in
# this file.
MORPHOLOGY_NAMES = ["REGULAR", "IRREGULAR"]
MORPHOLOGY_MAP = {
    # regular inflection: predictable -s suffix (plural or 3rd-singular present)
    "patterns": "REGULAR", "stories": "REGULAR", "billions": "REGULAR",
    "millions": "REGULAR", "choices": "REGULAR", "beliefs": "REGULAR",
    "years": "REGULAR", "characters": "REGULAR", "networks": "REGULAR",
    "times": "REGULAR", "likes": "REGULAR", "eats": "REGULAR",
    "becomes": "REGULAR", "reveals": "REGULAR",
    # irregular inflection: unpredictable form change (ablaut/suppletion)
    "sat": "IRREGULAR", "ran": "IRREGULAR", "is": "IRREGULAR",
    "are": "IRREGULAR", "was": "IRREGULAR", "has": "IRREGULAR",
    "these": "IRREGULAR", "written": "IRREGULAR",
}

DIMENSIONS = {
    "TENSE": (TENSE_NAMES, TENSE_MAP),
    "NUMBER": (NUMBER_NAMES, NUMBER_MAP),
    "ANIMACY": (ANIMACY_NAMES, ANIMACY_MAP),
    "DISCOURSE": (DISCOURSE_NAMES, DISCOURSE_MAP),
    "SYNTAX": (SYNTAX_NAMES, SYNTAX_MAP),
    "MORPHOLOGY": (MORPHOLOGY_NAMES, MORPHOLOGY_MAP),
}


def wire_grammar_extra(graph, tiles_path="tiles.json"):
    """NOT idempotent -- unlike add_fixed_edge(), graph.grow() unconditionally
    adds new nodes every call, so calling this twice on the same graph
    grows a second, orphaned set of hub nodes instead of reusing the
    first. Already run once against the real graph.json (2026-09-14);
    the resulting hub ids are saved in grammar_extra_hubs.json -- load
    that instead of calling this again for the real graph.

    Grows one hub node per category value across all four dimensions,
    then wires each mapped word to its hub with a frozen, confirmed
    edge -- identical mechanism to word_structure.py's word->role wiring,
    just a different set of hubs. Returns {dimension_name: {value_name:
    hub_node_id}} so a caller (or a future generation step) can look
    hubs up by name without re-deriving ids.
    """
    import json
    with open(tiles_path) as f:
        tiles = json.load(f)

    hub_ids = {}
    n_edges = 0
    for dim_name, (names, word_map) in DIMENSIONS.items():
        new_ids = graph.grow(len(names))
        hub_ids[dim_name] = dict(zip(names, new_ids))
        for word, value in word_map.items():
            if word in ("A", "space", "newline"):
                # Same guard word_structure.wire_word_structure() applies
                # to its own word_tiles -- these are reserved character
                # tile names, never real words, even if a hand-typed map
                # (like SYNTAX_MAP, derived from ROLE_MAP) has a stale
                # entry for one. Found by testing: without this, the
                # space CHARACTER got wired into SYNTAX=HEAD as if it
                # were the noun "space".
                continue
            node = tiles.get(word) or tiles.get(word.lower())
            if node is None:
                continue  # word not tiled yet -- skip, don't invent an id
            if node >= graph.n:
                # Same real gap found in word_structure.wire_word_structure():
                # tiles.json can reference a bigger, grown graph than the
                # one passed in here -- skip what doesn't fit rather than
                # crash (add_fixed_edge() itself now raises on this).
                continue
            hub_node = hub_ids[dim_name][value]
            graph.add_fixed_edge(node, hub_node, weight=1.0, trust=1.0)
            n_edges += 1
    return hub_ids, n_edges


def hub_node_ids(hubs_path="grammar_extra_hubs.json", discovered_path="discovered_dimensions.json"):
    """All node ids used as hubs across every dimension in this file
    PLUS every human-confirmed discover_dimension.py cluster, via the
    same load_hub_ids() merge head.py's tag table now reads from.
    Unlike CATEGORY_OFFSET/ROLE_OFFSET, these are NOT a fixed,
    predictable range -- grow() allocates them wherever the graph
    happened to be sized when wire_grammar_extra()/confirm_dimension()
    last ran -- so anything that computes a "reserved/used node id" set
    (shell.py's `tile` command, expand_vocab.auto_expand_vocab()) has
    to read this to know what to avoid. Real bug found by testing (2x):
    (1) without this at all, auto_expand_vocab() silently reassigned 13
    of 15 newly mined real words onto the exact node ids the
    TENSE/NUMBER/ANIMACY/DISCOURSE/SYNTAX/MORPHOLOGY hubs already live
    at; (2) after that fix but before this one only covered
    grammar_extra_hubs.json, shell.py's `tile` command reassigned node
    621 (DISCOVERED_ADJECTIVE_LIKE's hub) to a brand-new word tile,
    silently double-using a node that 5 other words already had a
    confirmed structural edge into. Returns an empty dict's worth of
    ids (empty set) if neither file exists yet."""
    hub_ids = load_hub_ids(hubs_path, discovered_path)
    return {nid for dim in hub_ids.values() for nid in dim.values()}


def load_hub_ids(hubs_path="grammar_extra_hubs.json", discovered_path="discovered_dimensions.json"):
    """The single, canonical way to load hub_ids -- merges the hand-
    named grammar_extra dimensions with any human-confirmed
    discover_dimension.py clusters, so every caller (head.py's tag
    table included) sees both without each having to know
    discovered_dimensions.json exists. Real gap found by testing:
    discover_dimension.py's confirm_dimension() already wires and
    persists real, confirmed graph edges for discovered clusters, but
    nothing merged them into the hub_ids every tag-building function
    reads from, so a confirmed discovery was a fact in graph.json that
    the tag table could never see. discovered_dimensions.json's shape
    is {dim_name: {"hub_id": id, "value_name": name, "words": [...]}}
    -- one value per discovered dimension so far; reshaped here to the
    same {dim_name: {value_name: hub_id}} shape grammar_extra_hubs.json
    already uses, so get_value() needs no special-casing to read either
    kind. Returns {} if neither file exists yet."""
    import json
    import os
    hub_ids = {}
    if os.path.exists(hubs_path):
        with open(hubs_path) as f:
            hub_ids = json.load(f)
    if os.path.exists(discovered_path):
        with open(discovered_path) as f:
            discovered = json.load(f)
        for dim_name, entry in discovered.items():
            hub_ids[dim_name] = {entry["value_name"]: entry["hub_id"]}
    return hub_ids


def get_value(graph, hub_ids, dim_name, word_node):
    """Mirror of word_structure.get_role() for any of these dimensions:
    returns the value name (e.g. "PAST") for a word node, or None if it
    has no confirmed, non-suspended edge into this dimension's hubs --
    a real 'none' state, not missing data. Suspended-but-confirmed
    edges are treated as none, matching get_role() -- see that
    docstring for why confirmed alone isn't enough."""
    for value_name, hub_node in hub_ids[dim_name].items():
        e = graph.find_edge(word_node, hub_node)
        if e is not None and bool(graph.confirmed[e]) and not bool(graph.suspended[e]):
            return value_name
    return None


def words_with_value(graph, hub_ids, dim_name, value_name):
    """Mirror of word_structure.words_with_role(): every word node with
    a confirmed edge into this specific hub."""
    hub_node = hub_ids[dim_name][value_name]
    incoming = (graph.dst == hub_node).nonzero().flatten().tolist()
    return [int(graph.src[e]) for e in incoming if bool(graph.confirmed[e])]
