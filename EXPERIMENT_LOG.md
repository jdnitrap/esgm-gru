# Experiment / Verification Log

## 2026-09-12 — Generation engine built, vocab/grammar expanded, real
corpus training

- Built the autoregressive generation engine (`sequence.py`). Found and
  fixed a real bug along the way: passive reinjection alone gets stuck
  repeating one word forever; actively querying `words_with_role()` for
  the current grammar slot and stimulating that pool directly is what
  makes generation actually work.
- Vocabulary expanded 13 → 31 words; 5 named grammar templates added
  (`simple`, `with_adjective`, `prepositional`, `pronoun_subject`,
  `pronoun_object`), selectable via `gen <grammar> <prompt>` in the
  shell.
- Real word-level training at scale: `mine_and_train.py` mined the
  actual 5MB Carbide corpus for genuine adjacent-word runs of the
  vocabulary (22,081 found) and trained on those directly — distinct
  from, and additional to, an earlier synthetic 8-sentence pass and an
  earlier byte-level-only pass over the same 5MB corpus (which never
  engaged word tiles at all).
- Full test suite passed at the time: `test_graph_sanity.py`,
  `test_fact_gate.py`, `test_integration_stress.py` (all 4 sub-checks),
  `test_sequence.py` (all 4 checks, including against the expanded
  vocab/grammars on the real trained graph).

## 2026-09-12 — Scale bug found and fixed: idle-edge trust decay with
no floor

**Symptom:** `active_fraction` (should hold steady at 0.05, the kWTA
quota fraction) dipped to ~0.037 under certain narrow single-region
stimulus patterns — but only on the real, massively-trained
`graph.json`. A fresh graph never showed it, meaning short smoke tests
had never caught this.

**Root cause:** edge trust (`tau`) decayed by a flat `-0.001` every
tick an edge went unused, with no floor, trending toward 0. Over the
real 5,000,000-tick byte-level corpus training run, mean tau collapsed
from 0.5 to 0.169 — pushing most non-frozen edges below the
`tau >= 0.3` participation threshold that gates message-passing
entirely. Under a stimulus touching only a handful of directly-relevant
nodes, too few edges remained trusted enough to propagate signal to
fill all k=15 quota slots with genuinely positive values — the kWTA
hard quota then included zero-valued "phantom" slots that don't count
as active, producing the dip. **A genuine scale bug** — works fine in
short smoke tests, breaks specifically at millions of ticks — not
something introduced by the generation-engine changes made the same
day.

**Fix:** idle decay now floors at the trust threshold itself
(`TAU_TRUST_THRESHOLD = 0.3`) instead of 0 —
`torch.clamp(self.tau - 0.001, min=TAU_TRUST_THRESHOLD)`. An edge that
goes quiet settles at "just barely trusted" rather than being fully
pruned; used edges still climb normally via `+0.01`; frozen/
contradiction-pair logic untouched. **Self-healing:** applying the fix
to the already-collapsed `graph.json` snaps affected edges back to 0.3
on their very next idle-decay tick — no need to touch the saved JSON
directly. **Verified:** `active_fraction` now holds exactly 0.0500
across 1500 mixed-stimulus stress-test ticks.

**Standing caution for future large-scale runs:** any future training
run that pushes well past where this one broke (millions of ticks)
should be re-checked for the same class of bug — unconditional
per-tick accumulation/decay with no floor — before assuming
short-test behavior generalizes.

## 2026-09-13 — Documentation pass, fresh test re-verification

Brought this repo's documentation up to the README+EXPERIMENT_LOG
standard used for the user's other split-out projects. Re-ran the full
test suite directly against the real, trained `graph.json` to confirm
nothing has regressed since 2026-09-12:

- `test_graph_sanity.py` — **PASS** (Hebbian consolidation: sustained
  stimulus keeps more originally-stimulated nodes active than brief
  stimulus, as expected)
- `test_fact_gate.py` — **PASS** (confirm/reject commit correctly;
  double-confirm is a safe no-op; no false-positive auto-confirms)
- `test_integration_stress.py` — **PASS** (full pipeline: propose →
  confirm → contradiction → suspend → decode → save/reload; frozen
  structure survives 1500 ticks of unrelated activity with 0 drift;
  full state reload matches exactly)
- `test_sequence.py` — **PASS** (real accumulated `graph.json` follows
  its grammar templates with zero fallbacks; every committed word's
  role independently re-verified against graph structure, not just the
  generation trace)

No code changes made this session — documentation only.

## 2026-09-13 — Reward-modulated learning, dynamic growth, real vocabulary
expansion 31 → 130 words, four bugs found and fixed

**Reward-modulated (three-factor) Hebbian learning.** `ESGRGraph.modulation`
(per-node, neutral=1.0) now scales the Hebbian weight update
(`eta*mod*x_u*x_v - lambda*w`); `graph.reward()`/`graph.punish()`, called
from `FactGate.confirm()`/`reject()`, nudge it. Neutral (mod=1.0, i.e. the
old formula exactly) unless something's actually been confirmed/rejected
nearby — verified byte-identical numbers on the existing sanity test.
Tuned against real data, not guessed: replayed the real corpus-mined
training data and the real graph's own already-trained state; amount=0.5
nearly pinned modulation at its ceiling the moment several real
confirmations landed close together, so tuned down to amount=0.3,
modulation_decay=0.97 (half-life ~23 ticks).

**Dynamic node growth (`ESGRGraph.grow()`) — the 300-node ceiling is no
longer architecturally fixed.** Append-only: new nodes + random-topology
edges, verified 0 frozen edges disturbed by growth. Wired into `tile`.
Also added `add_learnable_edge()` — an ordinary (non-frozen) edge can now
be created at runtime too, not just frozen ones via `add_fixed_edge`.

**`FactGate(auto_confirm=True)`** — an edge that clears sustained trust can
confirm itself with no human call, gated by a stricter
`auto_confirm_min_tau=0.95` floor than the normal 0.8 propose threshold.
Off by default; deliberately overrides "not a fact until confirm()" at
explicit user request.

**`supervise.py` — built, measured honestly, real negative result.** A
local, single-hop, ground-truth-corrected rule (not backprop — pushes
toward the real next word from real text even when it isn't currently
active). Measured against a proper held-out split of real corpus pairs
AND against a trivial "per-context majority vote" baseline: the graph
mechanism did NOT beat plain counting (51.8% vs 54.8% at the time).
Root cause: only 6/31 words ever appeared as a training "context" in real
adjacent-word pairs — severe vocabulary-driven data sparsity, not a flaw
in the rule. Widening the adjacency-gap tolerance (1/3/8/20 tried) made it
worse, not better (plateaued at 11/31 context words, baseline gap widened).
Left in the repo, working and honestly documented, currently unused
downstream.

**Vocabulary expansion 31 → 130 real words (`expand_vocab.py`), which
fixed the actual sparsity problem `supervise.py` couldn't:** mined the top
100 most-common real words in the corpus not already tiled, tiled via the
same free-slot-then-`grow()` path as the shell's `tile` command, extended
`word_structure.ROLE_MAP` with confident-only role assignments (genuinely
ambiguous words left without a role, same "real none" convention as the
original 31). Real coverage: 6/31 (19%) words with real training signal →
99/130 (76%). `graph.json`: 300→399 nodes, 2,667→4,244 edges.

**`grammar_extra.py` — four new hand-coded English mechanics dimensions**
beyond the original 6 syntactic roles, at explicit user direction ("there
is more language mechanics in the English system") after the `supervise.py`
finding: **TENSE** (PAST/PRESENT), **NUMBER** (SINGULAR/PLURAL, "you" left
out — real English ambiguity), **ANIMACY** (ANIMATE/INANIMATE),
**DISCOURSE** (AFFIRM/NEGATE/NEGATOR — finally gives "yes"/"no"/"not" a
real grammatical home). Same frozen-edge mechanism as `word_structure.py`;
uses `grow()` for hub nodes. Wired and queryable, not yet consumed by
`sequence.py`/`decode.py`.

**Four real bugs found and fixed, all verified against the full test
suite after each fix:**

1. **Tile/role-hub collision.** `tile`'s free-node search checked only
   `tiles.json`, not the category/role ranges — the next `tile` call
   would have silently overwritten the NOUN role hub, then the rest.
2. **"space" name collision.** The real English word "space" is also the
   reserved tiles.json key for the space *character* (byte 32) — mining
   logic excluded that key by name from "already have it," so the real
   word got mined as new and silently overwrote the character tile. Fixed
   the exclusion set (now ALL existing keys) and restored the value.
3. **Silent out-of-bounds edges.** `add_fixed_edge()` never validated
   `src`/`dst < graph.n` — a smaller graph given the (now bigger)
   `tiles.json` would silently create an edge to a nonexistent node, only
   crashing later, confusingly, inside `tick()`. Now raises immediately;
   `wire_word_structure()` skips what doesn't fit instead of relying on
   that.
4. **Latent min/max bug in `_enforce_sparsity`'s temperature path,
   exposed (not caused) by the vocabulary growth.**
   `pool_size = min(n, max(k*3, positive_count))` — the function's own
   docstring says the pool should be the generous multiple of k, OR every
   positive value if *fewer* — i.e. `min`, not `max`. It looked correct on
   the old sparse graph (positive_count rarely exceeded k*3, so the two
   operators agreed by coincidence); on the bigger, denser real graph,
   positive_count routinely hit 326/399, ballooning the softmax pool and
   collapsing temperature-based generation to 1/6 role matches across
   10/10 seeds tested. Fixed to `min`; reverified 10/10 seeds back to
   6/6. Worth extra scrutiny on similar `max()`/`min()` code whenever n or
   vocabulary size changes again — this class of bug is invisible at the
   scale it was written and tested against.

Full test suite (`test_graph_sanity.py`, `test_fact_gate.py`,
`test_sequence.py`, `test_integration_stress.py` against the real,
now-expanded `graph.json`) passes after every fix.

## 2026-09-14 — SYNTAX/MORPHOLOGY mechanics, self-expansion, and a real
byte-level trained head that reads and writes back through the graph

**`grammar_extra.py` gained SYNTAX and MORPHOLOGY**, from the merged
`mdbe/language_mechanics_*` worksheets (audited line-by-line first --
zero errors found there, unlike the byte-level `ASCII_Linguistics`
tables, which stay explicitly out of scope: real, verified errors,
built for a transformer/SSM/xLSTM's dense input layer, not ESGR). SYNTAX
is derived FROM `word_structure.ROLE_MAP` (not hand-typed separately --
one word's syntax is a fixed function of its role). Real bug found
wiring this to the live graph: a stale `"space": "NOUN"` entry left over
from the earlier vocabulary-expansion collision fix would have wired the
space CHARACTER into SYNTAX=HEAD; removed, and `wire_grammar_extra()`
now has the same reserved-name guard `wire_word_structure()` already had.

**Self-expansion, the "safe version":** `expand_vocab.auto_expand_vocab()`
is now reusable (not just a one-shot script) and wired into the shell as
`autoexpand [n]` -- frequency alone decides what's added, unattended, no
hand-picked word list. Real bug found on its first real test: 13 of 15
newly mined words collided with the grammar_extra hub nodes, because
those hubs are dynamically grown (no fixed offset like ROLE_OFFSET) and
nothing outside `grammar_extra.py` knew where they live. Fixed with one
shared source of truth, `grammar_extra.hub_node_ids()`, used by both
`autoexpand` and the `tile` command.

**A real, tested byte-level trained head (`head.py`), at explicit user
direction, layered on top of the graph without ever backpropagating into
it:**
- Rows = raw byte value (0x00-0xFF), matching Carbide's own token unit,
  not word-level.
- Per-byte input = a learned embedding concatenated with 32 real, fixed
  columns: the 6 `byte_identity.py` flags (always live) plus the 26
  word-level mechanics columns (ROLE/TENSE/NUMBER/ANIMACY/DISCOURSE/
  SYNTAX/MORPHOLOGY), which only turn on at the exact byte where a real
  tiled word completes -- causal only, never looks ahead.
- Ablation methodology replays Carbide's own (3-seed, `full` vs
  `embedding-only` vs `tags-only`): the mechanics columns gave a real,
  repeatable ~3-point accuracy gain across 3 seeds (35.4%/34.5%/35.1% ->
  38.4%/37.8%/38.5%), same controlled-comparison discipline as
  `carbide/MDBE_MANIFEST.md`.
- Single-token (bigram) head could never beat a plain count table --
  expected: one byte of context can't out-predict counting on the same
  one byte. `NextByteRNN` (a real GRU) was built specifically to test
  whether more context helps, and it does: beats the weak order-1
  baseline easily, and after warm-starting (see below), beat the much
  harder order-2 (previous-2-bytes) baseline for the first time --
  43.67% vs 42.69%.
- **Checkpointing with warm-start** (`head_checkpoint.py`): when the
  graph grows a new fixed column, the byte embedding table, GRU
  hidden-to-hidden weights, and output layer are provably unaffected by
  that (verified: copied byte-for-byte); only the GRU's input weight
  slice for the brand-new columns needs fresh init, since
  `build_tag_table()` always appends new columns after existing ones.
  Measured, not assumed: warm-start's first epoch (38.65%) already beat
  cold-start's third (33.81%).
- **The head teaches the graph back** (`retrain_head.py`,
  `supervise.py`'s existing `supervised_step()`, never a new backprop
  path): only on predictions that are both confident AND actually
  correct against real data -- 8,734 real edges reinforced this way in
  one test run, 0 frozen edges touched, graph node count unchanged.
- **`retrain_head` (shell command) + `uncertainty`**: a person decides
  when to fire a retrain, informed by a real, checkable signal (count of
  currently proposed-but-unconfirmed edges) -- nothing retrains
  automatically.

All of the above verified never touches `graph.w`/`tau`/`confirmed`/
`modulation`/`n`/edge count -- checked explicitly with before/after
tensor snapshots, not just by code inspection. Full existing test suite
still passes throughout.

## 2026-09-14 (session 2) — Real full-corpus training, a real weight
explosion bug, and a three-part fix for generation breaking under load

**Real training run, not a sample:** `run_real_training.py` warm-starts
from the existing checkpoint and trains on the full 5MB corpus (not the
300K-byte samples used for the ablations above). Held-out next-byte
accuracy: 88.6% -> 90.1% over successive real runs. Real, not assumed --
this is meaningfully higher than the sample-scale numbers, confirming
data volume (not just the mechanics columns) matters a lot here too.

**Real bug found: `supervise.py`'s `supervised_step()` had no ceiling on
weight growth.** Teaching back hundreds of thousands of real predictions
meant extremely common byte pairs (","->" ", "."->" ") got reinforced
so many times their weight reached **536** -- dwarfing frozen edges
(fixed at 1.0) and breaking word-level generation completely (temperature
sampling and even deterministic generation both collapsed to raw byte
repetition). Root cause: `tick()`'s own Hebbian growth is naturally
bounded by the `m_raw/(1+abs(m_raw))` saturation in message-passing;
`supervised_step()` writes directly with no such saturation. Fixed:
clamped to `graph.max_weight` (2.0). Verified: 1000 repeated teachings
of the same edge now cap at exactly 2.0.

**That fix alone was not enough -- a deeper, three-stage problem,
found only by systematically tracing the WHOLE pipeline stage by stage
(at explicit user direction, after two single-fix attempts each failed
in a different way and it became clear ad-hoc patching was chasing
symptoms downstream of each other):**

1. **`tick()`'s message-passing had no memory of its own recent
   history** -- a node's activation was just that tick's raw sum, so a
   burst of now-strengthened edges could cascade from near-zero to full
   strength in 1-2 ticks (measured: "the" stimulated at a stable 3.0
   collapsed under common vowels reaching 10-16 within two ticks).
   **Fix: `max_activation_rate`**, a leaky-bucket rate limiter on the
   message-passing term only (`raw = min(raw, self.x + rate)`,
   applied BEFORE external stimulus is added, so direct pokes are never
   rate-limited, only emergent internal cascades). Verified this alone
   does not weaken normal Hebbian learning (tracked edge still grew
   ~4x over 100 ticks under the limiter, matching unlimited growth).

2. **kWTA's `k` is one shared budget across the whole graph** -- with
   hundreds of now-strengthened byte-to-byte edges, enough of them
   climbing together (even individually rate-limited) could still fill
   all `k=20` seats collectively, crowding out word-level nodes even
   under direct, deliberate stimulation. **Fix: `split_sparsity_at`**
   -- splits the top-k selection into two fully independent pools
   (bytes 0-255, everything word-level 256+), each with its own quota,
   so byte-level volume can never take a word-level seat regardless of
   how many byte pathways are reinforced. Logic factored into a shared
   `_topk_select()` helper so both pools use identical, already-tested
   selection code.

3. **`decode()` has its OWN, completely separate shared budget** (a
   flat top-16 candidate cap) that neither of the above touches --
   found only by systematically tracing the exact same scenario through
   every stage rather than testing end-to-end output alone. A tiled word
   node could win its own kWTA seat and STILL get pushed out of
   `decode()`'s ranked candidate list by enough individually
   high-scoring byte candidates, especially in later steps of a
   sequence where the cascade has had more ticks to broaden (measured:
   "the" survived kWTA with activation 4.83 -- higher than the working
   first ARTICLE slot's 4.61 -- yet still failed to reach the candidate
   list, because more DISTINCT bytes had crossed high-score thresholds
   by that point in the sequence). **Fix: `decode(..., split_cap=True)`**
   -- same principle one layer up: tiled candidates get their own
   reserved half of the cap. Opt-in, default-off (existing callers
   unaffected); `sequence.py` opts in explicitly since that's the one
   pathway that needs it.

**Systematic verification, not spot-checks:** built one script that
traces every stage (kWTA activation after seed ticks, after step ticks,
decode candidate membership, final output) across all four combinations
(no fixes / rate-limit only / split-budget only / both together) on the
IDENTICAL reconstructed broken scenario. This is what actually revealed
neither single fix was sufficient and both were required together --
testing them one at a time in isolation had made each individually look
like a dead end. Only after adding the third fix (decode's own cap) did
the full 6-slot sequence reach 6/6 role matches across 8 seeds, both
temperature=0 and temperature=0.3, with 0 frozen edges disturbed --
verified with explicit before/after tensor snapshots each time, and the
real graph.json restored from backup after every failed attempt before
trying the next fix, so no broken intermediate state was ever left
persisted.

**Final real result, persisted:** graph.json now reflects a real,
complete training + teach-back cycle against the full corpus (439,251
edges reinforced, head at 90.1% held-out accuracy) with generation fully
working: `['the', 'choices', 'eats', 'under', 'the', 'existence']` --
6/6 grammar-correct, real vocabulary, real trained structure. Full test
suite passes.

## 2026-09-14 (session 2, continued) — more real training runs, vocab
330→340 words, real mechanics agreement, self-directed expansion

**Five real full-corpus training runs total now** (each ~80-110s on
the full 5MB corpus), held-out accuracy climbing 88.6% -> 90.4% with
diminishing returns as expected. Generation verified 6/6 after every
single one -- the 3-part fix above is holding up under repeated real
load, not just the first cycle.

**Vocabulary growth exposed a real scale-dependent regression in the
just-fixed rate limiter.** Growing 130 -> 330 words (n 412 -> 611) broke
generation again (4/6, not full collapse) with `max_activation_rate=2.0`
still active. Root cause: the SAME absolute rate cap that was
appropriately tuned for n=412 was too loose relative to the bigger,
denser n=611 graph. **Fixed by retuning, not redesigning: `rate=1.0`**
(tested against 0.5/1.0/1.5/2.0, only <=1.0 gave 5/5 seeds fully 6/6).
**Lesson for next time vocabulary/graph size changes significantly:
recheck whether `max_activation_rate` is still tuned right — it is NOT
automatically scale-invariant.**

**Real mechanics agreement wired into generation, not just stored.**
`sequence.py`'s `generate_sequence()` gained an optional `hub_ids`
param (default `None`, exact old behavior preserved) and
`_mechanics_bonus()`: when picking among role-matching candidates,
one that shares a real grammar agreement with what's already been
committed (VERB tense matching an earlier VERB's tense; PRONOUN number
matching an earlier NOUN's number) gets a scoring bonus. Verified
directly and precisely (not just by hoping emergent generation would
exercise it): "ran" (PAST) scores +2.0 after "sat" (PAST) was
committed; "is"/"likes" (PRESENT) score +0.0 in the same context;
`hub_ids=None` always scores 0 (mechanics off); a role with no
agreement rule (NOUN) always scores 0; no prior committed word of the
relevant role also scores 0. Added a `"two_actions"` grammar template
(`["PRONOUN","VERB","VERB"]`) specifically because none of the existing
5 templates have two VERB slots to exercise tense agreement on.

**Self-directed expansion, the real version, not just the "safe" manual
version from earlier this session.** New `autopilot [on|off]` shell
command: OFF by default (a person must explicitly opt in -- that single
choice is the entire safety gate), but once on, `tick`/`train`/`ask`
check a real signal (`uncertainty_signal()` -- count of proposed-but-
unconfirmed edges) after they run and, past a threshold, AUTOMATICALLY
mine new vocabulary, grow the graph, retrain the head, and save --
with no further human command. Verified working for real: `autopilot on
1` + `tick 20` produced `uncertainty=498 >= 1`, mined 10 new words, grew
the graph, retrained (0.9046 -> 0.9017, a small expected dip from
freshly-grown untrained words), and saved, entirely on its own. Full
test suite still passes afterward, generation still 6/6.

**Honest gap still open:** the newest ~210 words (mined via `autoexpand`
and `autopilot` this wave) don't have `ROLE_MAP` entries yet -- same
"real none" convention as before, not a bug, but it does mean
`words_with_role()`'s pools for existing roles haven't grown to include
most of the newest vocabulary. Hand-extending `ROLE_MAP` for ~200 more
words is real, tedious work that hasn't been done.

## 2026-09-14 (session 2, continued further) — 25-epoch heavy training,
ROLE_MAP gap closed, autoregressive generation, and 6 rounds of
dialogue fine-tuning that all failed the same honest way

**ROLE_MAP gap closed.** Hand-extended from ~115 to 247 entries
(covering the vocabulary above), same rule as always: assign only
where confident regardless of context, leave genuinely ambiguous words
(and/but/rather/etc) as a real "none". Full test suite still passes.

**Heaviest real training run yet:** 25 epochs (was 4) on the full 5MB
corpus, warm-started. Held-out accuracy 90.45% -> 91.23%, loss 0.378 ->
0.327, 391.7s. Teach-back wrote 446,897 confident-and-correct
predictions into the graph -- the largest volume yet, 0 frozen edges
disturbed, `n` unchanged. Real generation off the accumulated graph
still 6/6 grammar-correct afterward.

**Autoregressive generation is real now (`generate_bytes.py`).** The
trained head previously only ran teacher-forced; it can now sample its
own next byte and feed it back in, querying the graph live each step
(`word_at_position()`/`build_tag_table()` were already causal-only, so
this needed zero changes to be streaming-safe). Rigorously tested, not
just "it runs": 93.7% of generated tokens are valid English words,
100% of generated role-transitions matched a role-bigram that genuinely
occurs in the real corpus (mined from 262,485 real adjacent pairs),
stable over 1000-byte runs with no repetition collapse, 88.5%
word-validity even on seed words verified absent from the entire
training corpus. 0 frozen edges ever touched. ~67% of 4-word windows
ARE verbatim corpus recall though -- it leans heavily on memorized
phrasing, this is not free composition.

**Two new deterministic "speech mechanics" columns**, same hand-given-
fact pattern as `byte_identity.py`: `TURN` (QUESTION/ANSWER/none, a
forward scan for "Q:"/"A:" markers) and `QUESTION_FORM` (WH_WHAT/
WH_WHO/.../YES_NO/IMPERATIVE, detected from each question's real
opening word, held through its answer -- not reset at "A:", so the
signal is live exactly when the answer needs it). N_COLUMNS 32 -> 35 ->
45.

**Real discourse relations, grounded in published research (Penn
Discourse Treebank) instead of invented**, at explicit user direction
to search for prior art first. `DISCOURSE` extended from 3 values
(AFFIRM/NEGATE/NEGATOR) to 7, adding PDTB's standard four semantic
classes: TEMPORAL, CONTINGENCY, COMPARISON, EXPANSION. 28 real
connective words wired in, 22 newly tiled. N_COLUMNS 45 -> 49.

**Real self-dimension-discovery (`discover_dimension.py`)** -- the one
genuinely new-in-kind capability, not just another hand-given column,
at explicit user request ("make it have the ability to add dimension
to itself"). Uses the distributional hypothesis: clusters currently-
unassigned words by the ROLE of their real preceding/following
neighbor, mined from the actual corpus. Human-gated exactly like
`fact_gate.py` -- `propose_dimensions()` is read-only, `confirm_dimension()`
is the only thing that writes, called explicitly. Found and confirmed
two real clusters against the live graph: `DISCOVERED_ADJECTIVE_LIKE`
and `DISCOVERED_NOUN_LIKE`, both correctly rediscovering real
grammatical categories from pure context statistics, no ROLE_MAP hint
given. **Not yet wired into `head.py`'s N_COLUMNS/build_tag_table** --
confirmed in the graph and in `discovered_dimensions.json`, but the
trained head doesn't read them yet.

**Six rounds of dialogue fine-tuning (`dialogue_corpus.txt`,
`train_dialogue.py`), all honestly reported as failing to reach real
conversational coherence -- a genuine, diagnosed ceiling, not swept
under the rug:**
1. 51 pairs x6 repeats x200 epochs -> catastrophic overfitting
   (held-out accuracy dead flat 84.05%->84.03%, essay fluency
   96.3%->86.6%)
2. 203 pairs (66 facts x3 phrasings) + TURN column -> overfitting
   fixed, content still topically irrelevant to the question asked
3. + QUESTION_FORM column -> no improvement; confirmed DATA problem,
   not a missing-signal problem
4. 827 pairs (137 facts x6 phrasings), 40 epochs -> healthier curve,
   still climbing, worse output than round 2
5. same, 150 epochs -> accuracy plateaued cleanly (76.2%->82.7%) but
   real mode collapse: identical short garbled answers for different
   questions ("Pater. Theshs, better world.")
6. + PDTB discourse relations + discovered dimensions (N_COLUMNS->49),
   40 epochs -> held-out accuracy started at just 16% (vs ~77-84%
   every prior warm-start) because 22 newly-wired connective words are
   among the most common in English, shifting the tag distribution
   far more than any earlier column addition; climbed to 78% but still
   under-converged, conversation quality worse not better, and the
   essay-only checkpoint (never retrained on the new columns) broke
   down completely when loaded fresh.

**Final diagnosis, worked out with the user across the last several
turns of the session, not just my own guess:** the real problem was
never epoch count (tried 12/40/150/200, same collapse every time) --
it's that `dialogue_corpus.txt`'s answers are short, formulaic,
flashcard-style, one sentence in the same rhythm every time. The
essay-only head, by contrast, is genuinely fluent because the essay
corpus itself is rich, varied, natural prose. The direct analogy that
crystallized this (the user's own framing, confirmed against how
Claude itself is actually trained): pretraining on broad rich text
produces fluent continuation (== the essay-only head, already working
well); a SEPARATE fine-tuning stage on real prompt->response
DEMONSTRATIONS, written to the SAME quality bar as the pretraining
data, is what teaches instruction-following/conversation -- not just
more examples in a thin, repeated shape. `dialogue_corpus.txt` never
had that quality bar. Held-out next-BYTE accuracy is also NOT a
reliable proxy for conversational coherence at this scale -- round 5
proved that directly (accuracy climbed cleanly to 82.7% while output
degraded into repeated garbage).

**For a future session, in priority order if dialogue capability gets
picked up again:** (1) rewrite `dialogue_corpus.txt` with genuinely
rich, natural, multi-sentence answers instead of one-line facts --
don't just add more pairs in the same thin shape, that lever is
exhausted; (2) retrain `head_checkpoint.pt` itself (plain essay-only)
now that the graph has grown to 49 columns, since it currently breaks
down if loaded fresh, never having seen the new high-frequency
DISCOURSE columns; (3) wire `discovered_dimensions.json` into
`head.py` (never done this session); (4) consider sampling changes
(nucleus/top-k instead of plain multinomial, a repetition penalty) to
directly address the mode-collapse symptom, since it showed up
independent of dataset size or epoch count.

## 2026-09-14 (session 3) — renamed to ESGM, five verified fixes closing
gaps three independent AI reviews (Grok, Lumo, and a separate Claude
session) surfaced against this repo

**Renamed "Edge-State Graph Reasoner" to "Edge-State Graph Memory"
(ESGM), at the user's direction.** What the graph does --
propose/confirm/reject, contradiction-energy suspension -- is
consistency-tracking and belief revision, not inference or derivation
of new facts from existing ones; "Reasoner" overclaimed that. The
generation component is now named by contract, not architecture ("MDBE-
Conditioned Autoregressive Generator"), since it's explicitly meant to
be swappable -- this repo's concrete implementation is `head.py`'s GRU,
NOT an xLSTM (no xLSTM code exists anywhere in this repo; a real user
misunderstanding, carried for months, traced back to review documents
saying "the xLSTM / GRU is the mouth" that were never verified against
the actual code before being repeated). A sibling fork, ESGM-CARBIDE,
was created (`~/Downloads/esgm-carbide`, shares this repo's full commit
history) to eventually pair the same Memory component with a generator
built on Carbide's proven CPU-optimized SSM patterns instead of a GRU
-- as of the fork point, no Carbide-SSM code exists there yet, it's an
unmodified copy.

**Three independent AI reviews of this repo (Grok, Lumo, and a separate
Claude web session with no code access) were compared against each
other and against the live code.** Grok's and the separate Claude
session's reviews, once given the current repo instead of a stale
snapshot, converged independently on the same underlying complaint from
different files: real built capability that never reached the thing
that would consume it. Lumo's review, across three attempts, never
engaged with the graph/generator split at all despite it being the
README's lead section by the second attempt -- deprioritized for
architectural findings going forward. Five items from the two credible
reviews were implemented and verified this session:

1. **Wire hub_ids into shell.py's `gen` command.** `sequence.py`'s
   grammar-agreement code (`_AGREEMENT_RULES`/`_mechanics_bonus`) was
   fully built but dead in the real path -- `shell.py` always called
   `generate_sequence()` with `hub_ids=None`. One-line fix
   (`hub_ids=HUB_IDS`, loaded from `grammar_extra_hubs.json` at module
   load). Verified: manual `gen two_actions` run, full `test_sequence.py`
   pass.

2. **Resize `FactGate` on graph growth.** `consecutive_high_trust`/
   `proposed` were sized once at `__init__` and never grew with the
   graph, so edges added by `grow()`/`add_learnable_edge()`/
   `auto_expand_vocab()` after gate construction sat silently outside
   confirm/reject coverage. Added `FactGate.resync()`, called at every
   live call site that grows the graph (`shell.py`'s `tile` and
   `autoexpand` commands, the autopilot path). Verified by direct
   repro: `gate.step()` raised `RuntimeError: size of tensor a (317)
   must match size of tensor b (271)` pre-fix on a grown graph; clean
   after calling `resync()`.

3. **Found while implementing #4 below, not part of the original
   review list: `get_role()`/`get_value()`/`byte_columns()` checked
   only `confirmed`, never `suspended`.** Contradiction-suspension
   (`graph.py`'s `E_contr` auto-suspend) deliberately leaves `confirmed`
   untouched so a later `confirm()`/`reject()` has something to
   resolve -- so a confirmed-then-suspended edge kept reporting its
   role/value/category as if the fight never happened. `decode()`
   already treats suspend as silence; these three functions now match
   that (`confirmed and not suspended`). Verified by direct repro:
   `get_role()` returned `"NOUN"` on a manually-suspended edge pre-fix,
   `None` after.

4. **Wire the brain's veto into `generate_bytes.py`.** The README
   already stated this gap in words; nothing in the code closed it.
   Added `_word_is_vetoed()`: after each generated byte, if it just
   completed a real tiled word whose role or dimension edge is
   explicitly `rejected` or currently `suspended`, that word is
   stripped back out of the output and generation stops there. A word
   with no structural edge at all (an invented spelling) is NOT
   vetoed -- silence isn't an objection, only an explicit dispute is.
   Verified end-to-end with a fake model forced to always emit "cat":
   generated freely pre-reject (`'seed cat  '`), correctly stripped and
   stopped post-reject (`'seed '`, `stopped_early=True`).

5. **Wire `discover_dimension.py`'s confirmed clusters into `head.py`'s
   tag table** (item (3) named as not-done at the end of the previous
   session's entry above). Added `grammar_extra.load_hub_ids()`,
   merging `grammar_extra_hubs.json` with `discovered_dimensions.json`
   into one shape; every direct loader of the hubs file across the
   repo (8 files) switched to this one canonical loader. Extended
   `head.py`'s `WORD_DIMS` with the two currently-confirmed discovered
   dimensions (`DISCOVERED_ADJECTIVE_LIKE`, `DISCOVERED_NOUN_LIKE`).
   `N_COLUMNS` grew 49 -> 53; `head_checkpoint.pt` warm-started cleanly
   via the mechanism `head_checkpoint.py` already had for exactly this.
   Verified: `get_value()` returns `"ADJECTIVE_LIKE"` for a real
   confirmed word ("only"), tag table builds at the new width.

Full test suite (`test_graph_sanity`, `test_fact_gate`,
`test_integration_stress`, `test_sequence`, `test_generate_bytes`,
`test_word_generation`, `test_word_generation_deep`, `test_dialogue`)
re-run clean after all five changes.

**One more real bug found immediately after, by live testing via
`shell.py` rather than a unit test:** `hub_node_ids()`'s free-slot
exclusion set (used by `shell.py`'s `tile` command and
`expand_vocab.auto_expand_vocab()` to avoid reassigning a hub node to a
new word) only ever read `grammar_extra_hubs.json`, never
`discovered_dimensions.json` -- so `tile zephyr` reassigned node 621
(`DISCOVERED_ADJECTIVE_LIKE`'s hub, with 5 confirmed word edges already
wired into it) to a brand-new, unrelated word tile. Same class of
collision this function already exists to prevent for grammar_extra
hubs, just missing the newer file. Corrupted state was caught before
`save`, restored from a pre-test backup, never persisted. Fixed by
having `hub_node_ids()` build on the same `load_hub_ids()` merge
`head.py`'s tag table now reads from, so both sources are covered in
one place instead of two parallel exclusion lists. Verified: 621/622
now correctly excluded; re-running the exact `tile zephyr` command
grows a genuinely new node instead of colliding.

## 2026-09-14 (session 4) — learning rule modularized; Oja's rule tried
and rejected as the default, kept as a tested option

**`graph.py`'s weight-update formula extracted into `learning_rules.py`
(new file: `LearningRule` base class, `HebbianLearning`).** Previously
the Hebbian formula was hardcoded inline inside `tick()`, alongside
activation dynamics, k-WTA sparsity, and contradiction handling --
trying a different rule meant forking the whole file. `ESGRGraph`
gained a `learning_rule` constructor param (default `None` ->
`HebbianLearning()`, so every existing caller is unaffected).
**Verified byte-for-byte identical to the pre-refactor hardcoded
formula**: captured a golden reference (50 ticks, fixed seed/stimulus,
`w_sum`/`tau_sum`/full per-tick stats) before touching `tick()`, and
confirmed exact equality after.

**Real bug found by testing while verifying the refactor:**
`ESGRGraph.load_json()` builds via `__new__()`, bypassing `__init__()`
entirely -- it never set `learning_rule` on a loaded graph, so any
`tick()` on a graph loaded from `graph.json` crashed with
`AttributeError`. This broke `test_integration_stress.py` and
`test_sequence.py` immediately (both load the real graph). Fixed by
setting `g.learning_rule = HebbianLearning()` in `load_json()`,
matching the same "older saves predate X, default to old behavior"
pattern this function already uses for `modulation_decay`/
`max_activation_rate`/`split_sparsity_at`. Now regression-tested in
`test_learning_rules.py`.

**New `test_learning_rules.py`** (permanent, not a one-off script):
(1) default rule matches the golden pre-refactor reference exactly,
(2) a genuinely different rule (`FrozenLearning`, a null-op used only
to prove the plug point is real) swapped in at construction actually
changes `tick()`'s output -- proves this is really pluggable, not an
unused abstraction, (3) the `load_json()` bug above, regression-tested.

**Oja's rule (Oja, 1982) implemented as `OjaLearning`, then measured
against Hebbian on the real, accumulated `graph.json`** -- identical
seed and mixed-stimulus schedule as `test_integration_stress.py`
(bytes, MDBE-linked bytes, category nodes, word tiles, role hubs
rotating every tick), 1500 ticks:

| | Hebbian (default) | Oja |
|---|---|---|
| NaN ticks | 0 | 0 |
| Max \|weight\| reached | 64.93 | **15.68** |
| Final weight sum | 6506.0 | 4320.8 |
| Weight-floor (>=0) clamp hits | **0** | **1700** |
| Frozen structure disturbed | 0 | 0 |
| Runtime (1500 ticks) | 0.65s | 0.65s |

Oja's self-limiting property is real and measured: max weight
magnitude dropped >4x under identical stimulus, directly relevant to
the class of runaway-weight bug this project already hit once
(`w=536`, supervised_step()). **But it is not a clean win**: Oja's
decay term (`-w*x_v^2`) overshot below zero and needed the
non-negativity floor 1700 times in the same run Hebbian never needed
it once -- a real cost, not just a footnote, and evidence its
dynamics may be less stable at this graph's current `eta` than the
aggregate stats alone suggest.

**Decision: kept as a tested, available option, NOT switched to be
the default.** Every existing tuned constant in this system
(`eta`, `lam`, `max_weight`, the three-factor modulation amounts/decay,
the head checkpoint's warm-start assumptions) was calibrated against
Hebbian over many real sessions; switching the default is a much
bigger change than trying a rule, and nothing here is responding to a
live, active problem the way the original `w=536` bug was. If a real
weight-runaway problem resurfaces, or someone wants to run the larger,
real-training-data comparison this one synthetic-stimulus run doesn't
cover, `OjaLearning` is sitting there, tested, ready to try.

Full test suite (all 9 files including the new `test_learning_rules.py`)
re-run clean.

## 2026-09-15 (session 5) — the Oja comparison above was flawed; eta/lam
made genuinely rule-owned, and the corrected comparison tells a
materially different story

**Real gap found by the user, not a test:** "if one piece was
swappable but everything else depended on that piece" — the
`LearningRule` contract above only isolated the raw `Δw` formula.
`eta`/`lam` were still passed in from the graph on every call, and the
per-node outgoing-weight normalization (`w_norm_fires`) still ran
unconditionally in `tick()` regardless of which rule was active. So
the session-4 comparison wasn't really testing Oja's rule — it was
testing "Oja's formula, run with Hebbian's tuned `eta`, under
Hebbian's own external rescue mechanism." **That comparison's
conclusion ("Oja needed the floor-clamp 1700 times, a real cost") is
retracted, not just superseded.**

**Fixed properly:** `eta`/`lam` moved to be constructor args each
`LearningRule` owns itself (`HebbianLearning(eta=.., lam=..)`,
`OjaLearning(eta=..)` — Oja takes no `lam` at all now, since its own
`w*x_v^2` term IS the decay, not a supplement to a separate one).
`LearningRule` gained an optional `normalize()` hook (default: no-op)
that `HebbianLearning` overrides with the exact per-node rescale it
always had; `OjaLearning` doesn't override it, so nothing runs on top
of Oja's own dynamics unless it asks for it. `ESGRGraph.__init__`'s
`eta`/`lam` args still build the default `HebbianLearning` exactly as
before (byte-for-byte re-verified against the same golden reference).

**A second real bug found while fixing the first:** `save_json()`
persisted `self.eta`/`self.lam` (the graph's construction-time
values), not the actually-active rule's `eta`/`lam` -- so a
`HebbianLearning(eta=0.05)` plugged into a graph constructed with the
default `eta=0.01` would silently save and reload as `0.01`, losing
the real value. Fixed to read from `self.learning_rule` first,
falling back to the graph's own value only for a rule that doesn't
have one. Regression-tested in `test_learning_rules.py` (6 checks now,
up from 3: golden-reference match, a real swap changing behavior, the
`load_json()` crash fix, a rule's own `eta` genuinely overriding the
graph's, that override round-tripping through save/load correctly,
and `normalize()` actually firing under real stimulus).

**The corrected Hebbian-vs-Oja comparison** (same real `graph.json`,
same 1500-tick mixed-stimulus schedule as before), with Oja given its
own `eta` instead of Hebbian's:

| Oja `eta` | max \|weight\| | floor-clamp hits | NaN |
|---|---|---|---|
| 0.01 (Hebbian's borrowed value — the old, flawed test) | 15.68 | **1700** | 0 |
| 0.005 | 15.68 | **0** | 0 |
| 0.001 | 15.68 | **0** | 0 |
| 0.0005 | 15.68 | **0** | 0 |
| 0.0001 | 15.68 | **0** | 0 |

The floor-clamp problem disappears entirely the moment Oja isn't
forced to use Hebbian's `eta`. `max_w_seen` lands on exactly the same
value (15.683) across every `eta` tested — a real, explicable
property, not a coincidence: Oja's fixed point (`Δw=0` → `w* =
x_u/x_v`) doesn't depend on `eta` at all; `eta` only sets step size on
the way there. At `eta=0.01` each step overshoots that equilibrium and
has to correct back below zero, repeatedly hitting the floor; at any
smaller `eta` it converges smoothly, same destination, no overshoot.

**One more honest number this run surfaced that the first comparison
never tracked:** Hebbian's own `normalize()` (its external per-node
rescue mechanism) fired **31,234 times** over the same 1500 ticks.
Hebbian isn't naturally stable either — it leans on its own external
patch constantly. Properly-tuned Oja needed neither that patch nor the
floor-clamp, at any of the five `eta` values tried.

**Revised conclusion:** the original "keep Hebbian, Oja seemed less
stable" reasoning is wrong and retracted. Properly bundled, Oja looks
more self-sufficient than Hebbian on this graph, not less. **The
decision to keep `HebbianLearning` as the default is unchanged, but
now rests entirely on the reason the user gave directly — every other
tuned constant in this system was calibrated against Hebbian over many
real sessions, and switching the default is a bigger change than
trying a rule — not on any performance shortfall of Oja's, which this
corrected run does not show.**

Full test suite (all 9 files) re-run clean after both fixes.
