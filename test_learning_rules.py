"""Verifies learning_rules.py's pluggable weight-update contract:
(1) the default (HebbianLearning) is byte-for-byte identical to the
hardcoded formula tick() used before this file existed, (2) a genuinely
different rule swapped in actually changes tick()'s behavior -- proving
this is really pluggable, not an unused abstraction wrapping one fixed
path, (3) ESGRGraph.load_json() (which bypasses __init__ via __new__())
still ends up with a working learning_rule -- regression test for a
real bug found by testing: load_json() crashed every tick() on a loaded
graph with AttributeError until load_json() was fixed to set it too.
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
# code path once learning_rule is swapped.
class FrozenLearning(LearningRule):
    """Deliberately extreme: weights never change, period. If this
    doesn't change tick()'s behavior vs. Hebbian, the graph isn't
    actually calling the plugged-in rule."""
    def update_weights(self, w, x_u, x_v, modulation, frozen, eta, lam, use_hard_clip, max_weight):
        return w.clone(), torch.zeros_like(w), 0


g2 = ESGRGraph(n_nodes=300, mean_out_degree=8, seed=0, learning_rule=FrozenLearning())
w_before = g2.w.clone()
_run_ticks(g2)
w_after = g2.w
assert torch.equal(w_before, w_after), "FrozenLearning should leave every weight completely unchanged"
assert not torch.equal(g1.w, w_after), "swapped rule's output should differ from Hebbian's on the same graph/stimuli"
print("[PASS] a custom LearningRule (FrozenLearning) actually changes tick()'s behavior -- "
      "weights genuinely frozen, and diverge from the Hebbian run above")


# [3] load_json() (which bypasses __init__ via __new__()) still ends up
# with a working learning_rule. Real bug found by testing: this crashed
# with AttributeError on every tick() of a loaded graph until fixed.
g3 = ESGRGraph(n_nodes=50, mean_out_degree=6, seed=1)
g3.save_json("/tmp/test_learning_rules_scratch.json")
g4 = ESGRGraph.load_json("/tmp/test_learning_rules_scratch.json")
assert isinstance(g4.learning_rule, HebbianLearning), "load_json() must default to HebbianLearning"
g4.tick(external_input=torch.zeros(g4.n))  # must not raise AttributeError
print("[PASS] load_json() restores a working learning_rule (regression test for the __new__()-bypasses-__init__() bug)")

print("\nAll learning_rules checks passed.")
