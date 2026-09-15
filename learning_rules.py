"""Pluggable weight-update rules for ESGRGraph.tick(), extracted so a
different rule (Oja's, eligibility traces, etc.) can be tried later by
swapping one object instead of forking graph.py. graph.py's own
activation dynamics, k-WTA sparsity, per-node outgoing-weight
normalization, tau/trust decay, and contradiction handling are NOT part
of this contract -- those stay fixed regardless of which learning rule
is plugged in; only the edge weight-update FORMULA is pluggable here.

HebbianLearning below is the exact formula graph.py always used,
extracted verbatim -- byte-for-byte identical behavior as the default,
verified against a golden pre-refactor reference (see
test_learning_rules.py). This is additive, not a behavior change.
"""
import torch


class LearningRule:
    """Contract: given the graph's current per-edge weight tensor and
    the tensors tick() already computed this step (pre/post-activation
    at each edge's src/dst, the reward-modulation term, which edges are
    frozen, and the graph's own eta/lam/clip settings), return the new
    weight tensor plus the raw delta (needed by tick() for its E_drift
    stat) and a clip-hit count. All arguments are full edge-indexed
    tensors -- this must stay vectorized, never a per-edge Python loop,
    to keep tick() usable at real graph scale."""

    def update_weights(self, w, x_u, x_v, modulation, frozen, eta, lam, use_hard_clip, max_weight):
        raise NotImplementedError


class HebbianLearning(LearningRule):
    """The rule this graph has always used, unchanged: Δw = eta * mod *
    x_u * x_v - lam * w, frozen edges never move, then clamped to
    [0, max_weight] (hard clip) or just [0, inf) (soft/no upper bound
    here -- the graph's own per-node outgoing-weight normalization,
    applied after this in tick(), is what actually bounds unclipped
    growth in that mode)."""

    def update_weights(self, w, x_u, x_v, modulation, frozen, eta, lam, use_hard_clip, max_weight):
        delta_w = eta * modulation * x_u * x_v - lam * w
        delta_w = torch.where(frozen, torch.zeros_like(delta_w), delta_w)
        w_raw = w + delta_w
        if use_hard_clip:
            w_clamped = w_raw.clamp(min=0.0, max=max_weight)
        else:
            w_clamped = w_raw.clamp(min=0.0)
        clip_hits = int((w_clamped != w_raw).sum().item())
        return w_clamped, delta_w, clip_hits


class OjaLearning(LearningRule):
    """Oja's rule (Oja, 1982): Δw = eta * (x_u*x_v - w*x_v^2) -- the
    same Hebbian co-activation term, but decay is proportional to w
    TIMES THE SQUARE of the post-synaptic activation, not a constant
    rate. Self-limiting by construction: as a weight grows, its own
    contribution to x_v grows too, feeding back into a bigger decay
    pull on itself -- the classic online algorithm for converging
    toward a principal eigenvector, not an ad-hoc external clamp.

    `lam` is accepted for interface compatibility but genuinely
    unused -- Oja's own w*x_v^2 term IS the decay; it isn't a
    supplement to a separate constant-rate one. Same reward-
    modulation, frozen-edge, and clamp handling as HebbianLearning
    (including keeping weights non-negative, matching this graph's
    excitatory-only activation model), so this is an apples-to-apples
    swap for comparison, not a differently-scoped rule."""

    def update_weights(self, w, x_u, x_v, modulation, frozen, eta, lam, use_hard_clip, max_weight):
        delta_w = eta * modulation * (x_u * x_v - w * x_v ** 2)
        delta_w = torch.where(frozen, torch.zeros_like(delta_w), delta_w)
        w_raw = w + delta_w
        if use_hard_clip:
            w_clamped = w_raw.clamp(min=0.0, max=max_weight)
        else:
            w_clamped = w_raw.clamp(min=0.0)
        clip_hits = int((w_clamped != w_raw).sum().item())
        return w_clamped, delta_w, clip_hits
