"""Verifies learning_rules.py's pluggable weight-update contract:
(1) the default (HebbianLearning) is byte-for-byte identical to the
hardcoded formula tick() used before this file existed, (2) a genuinely
different rule swapped in actually changes tick()'s behavior -- proving
this is really pluggable, not an unused abstraction wrapping one fixed
path, (3) ESGRGraph.load_json() (which bypasses __init__ via __new__())
still ends up with a working learning_rule -- regression test for a
real bug found by testing: load_json() crashed every tick() on a loaded
graph with AttributeError until load_json() was fixed to set it too,
(4) eta/lam are genuinely rule-owned, not borrowed from the graph --
regression test for a real gap the user found by inspection, not a
test: the first version of this contract passed the GRAPH's eta/lam
into whichever rule was plugged in, so "swapping" a rule really meant
"running a different formula with someone else's tuned constants" --
see EXPERIMENT_LOG.md for how this produced a misleading Hebbian-vs-Oja
comparison, (5) normalize() is a real, optional per-rule hook, not
something tick() always runs regardless of which rule is active.
"""
import torch
from graph import ESGRGraph
from learning_rules import LearningRule, HebbianLearning


def _run_ticks(g, n=50, seed=0):
    torch.manual_seed(seed)
    for i in range(n):
        ext = torch.zeros(g.n)
        if i % 5 == 0:
            ext[torch.randint(0, g.n, (3,))] = 3.0
        g.tick(external_input=ext, temperature=0.0)
    return g.w.sum().item(), g.tau.sum().item()


# [1] Default rule reproduces the exact pre-refactor golden values --
# captured from the hardcoded formula before learning_rules.py existed,
# on this exact seed/graph/stimulus schedule. A change here means the
# refactor altered real behavior, not just where the code lives.
g1 = ESGRGraph(n_nodes=300, mean_out_degree=8, seed=0)
w_sum, tau_sum = _run_ticks(g1)
GOLDEN_W_SUM = 546.9886474609375
GOLDEN_TAU_SUM = 1021.0134887695312
assert abs(w_sum - GOLDEN_W_SUM) < 1e-4, f"w_sum drifted: {w_sum} vs golden {GOLDEN_W_SUM}"
assert abs(tau_sum - GOLDEN_TAU_SUM) < 1e-4, f"tau_sum drifted: {tau_sum} vs golden {GOLDEN_TAU_SUM}"
print(f"[PASS] default HebbianLearning matches pre-refactor golden values exactly "
      f"(w_sum={w_sum:.4f}, tau_sum={tau_sum:.4f})")


# [2] A genuinely different rule (zero learning -- weights never move
# at all) swapped in at construction actually changes tick()'s output.
# Proves the plug point is real: graph.py contains no Hebbian-specific
# code path once learning_rule is swapped. Also exercises normalize():
# FrozenLearning doesn't override it, so the base class no-op must run
# (0 fires), not HebbianLearning's per-node rescale.
class FrozenLearning(LearningRule):
    """Deliberately extreme: weights never change, period. If this
    doesn't change tick()'s behavior vs. Hebbian, the graph isn't
    actually calling the plugged-in rule."""
    def update_weights(self, w, x_u, x_v, modulation, frozen, use_hard_clip, max_weight):
        return w.clone(), torch.zeros_like(w), 0


g2 = ESGRGraph(n_nodes=300, mean_out_degree=8, seed=0, learning_rule=FrozenLearning())
w_before = g2.w.clone()
_run_ticks(g2)
w_after = g2.w
assert torch.equal(w_before, w_after), (
    "FrozenLearning should leave every weight completely unchanged -- "
    "if this fails, either update_weights() or normalize() ran Hebbian's logic instead"
)
assert not torch.equal(g1.w, w_after), "swapped rule's output should differ from Hebbian's on the same graph/stimuli"
print("[PASS] a custom LearningRule (FrozenLearning) actually changes tick()'s behavior -- "
      "weights genuinely frozen (both update_weights() AND the default no-op normalize() respected), "
      "and diverge from the Hebbian run above")


# [3] load_json() (which bypasses __init__ via __new__()) still ends up
# with a working learning_rule. Real bug found by testing: this crashed
# with AttributeError on every tick() of a loaded graph until fixed.
g3 = ESGRGraph(n_nodes=50, mean_out_degree=6, seed=1)
g3.save_json("/tmp/test_learning_rules_scratch.json")
g4 = ESGRGraph.load_json("/tmp/test_learning_rules_scratch.json")
assert isinstance(g4.learning_rule, HebbianLearning), "load_json() must default to HebbianLearning"
g4.tick(external_input=torch.zeros(g4.n))  # must not raise AttributeError
print("[PASS] load_json() restores a working learning_rule (regression test for the __new__()-bypasses-__init__() bug)")


# [4] eta/lam are genuinely owned by the RULE, not the graph -- a rule
# constructed with different constants than the graph's own eta/lam
# produces different behavior than the graph's eta/lam would. Real gap
# found by the user (not a test) in the first version of this contract:
# eta/lam were passed in from the graph on every call, so any "swapped"
# rule was still running with someone else's tuned constants.
custom_rule = HebbianLearning(eta=5e-2, lam=1e-3)  # 5x the graph's own default eta
g5 = ESGRGraph(n_nodes=300, mean_out_degree=8, seed=0, eta=1e-2, lam=1e-3, learning_rule=custom_rule)
w5_sum, _ = _run_ticks(g5)
assert abs(w5_sum - GOLDEN_W_SUM) > 1.0, (
    "a rule with a different eta than the graph's own constructor eta should NOT reproduce "
    "the same result -- if it does, the rule isn't actually using its own eta"
)
print(f"[PASS] a rule's own eta (0.05) genuinely overrides the graph's constructor eta (0.01) -- "
      f"w_sum={w5_sum:.4f}, diverges from golden {GOLDEN_W_SUM:.4f} as expected")

# And round-trips correctly through save/load: a graph whose active
# rule has non-default eta/lam, saved and reloaded, must reconstruct
# HebbianLearning with THOSE values, not HebbianLearning()'s defaults.
g5.save_json("/tmp/test_learning_rules_scratch2.json")
g6 = ESGRGraph.load_json("/tmp/test_learning_rules_scratch2.json")
assert g6.learning_rule.eta == custom_rule.eta, (
    f"reloaded rule's eta should match the ACTIVE rule's real eta (0.05), not the graph's stale "
    f"construction-time eta (0.01) -- got {g6.learning_rule.eta}"
)
print(f"[PASS] the ACTIVE rule's real eta/lam (not the graph's possibly-stale construction-time "
      f"values) round-trips through save_json()/load_json() correctly (eta={g6.learning_rule.eta})")


# [5] normalize() is a real, optional per-rule hook: HebbianLearning's
# override actually fires (w_norm_fires > 0) under realistic sustained
# stimulus, and a rule that doesn't override it never triggers that
# rescale at all -- proven directly by [2] above already (FrozenLearning's
# weights stayed byte-identical, which couldn't happen if
# HebbianLearning's normalize() had run on top of it). Uses the real,
# accumulated graph.json with the same mixed-stimulus schedule
# test_integration_stress.py uses -- a fresh small random graph under
# modest stimulus doesn't reliably push any node's mean outgoing weight
# past the 2.0 threshold within a short run; the real graph's existing
# structure does.
from byte_identity import CATEGORY_OFFSET
from word_structure import ROLE_OFFSET, ROLE_NAMES

g7 = ESGRGraph.load_json("graph.json")
torch.manual_seed(0)
any_norm_fired = False
for t in range(500):
    stim = torch.zeros(g7.n)
    pattern = t % 5
    if pattern == 0:
        stim[torch.randint(0, 256, (8,))] = 2.0
    elif pattern == 1:
        stim[ord('c')] = 3.0; stim[ord('a')] = 3.0; stim[ord('t')] = 3.0
    elif pattern == 2:
        stim[CATEGORY_OFFSET:CATEGORY_OFFSET + 6] = 1.5
    elif pattern == 3:
        stim[262:275] = 1.5
    else:
        stim[ROLE_OFFSET:ROLE_OFFSET + len(ROLE_NAMES)] = 1.5
    info = g7.tick(external_input=stim)
    if info["w_norm_fires"] > 0:
        any_norm_fired = True
assert any_norm_fired, "HebbianLearning's normalize() should fire at least once over 500 ticks on the real graph"
print("[PASS] HebbianLearning.normalize() (the per-node outgoing-weight rescale) genuinely fires under real stimulus")

print("\nAll learning_rules checks passed.")
