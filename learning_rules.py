"""Pluggable weight-update rules for ESGRGraph.tick(), extracted so a
different rule (Oja's, eligibility traces, etc.) can be tried later by
swapping one object instead of forking graph.py. graph.py's own
activation dynamics, k-WTA sparsity, tau/trust decay, and contradiction
handling are NOT part of this contract -- those stay fixed regardless
of which learning rule is plugged in.

Each rule owns its OWN tuning constants (eta, lam, or whatever else it
needs) as constructor args, rather than borrowing the graph's -- eta
and lam were previously graph-level and passed into update_weights()
call-by-call, which meant "swapping" a rule really meant "running a
different formula with someone else's tuned constants." Verified by
testing: this produced a misleading comparison (Oja's rule run with
Hebbian's eta, under Hebbian's own post-hoc normalization step, hit
its non-negativity floor 1700 times in a run Hebbian never needed it
once -- see EXPERIMENT_LOG.md). Real gap found by the user, not by a
test: "if one piece was swappable but everything else depended on that
piece," the swap isn't meaningful. graph.max_weight stays graph-level
on purpose (see update_weights()'s docstring) -- everything else that
was genuinely Hebbian-specific now lives on HebbianLearning itself.
"""
import torch


class LearningRule:
    """Contract: given the graph's current per-edge weight tensor and
    the tensors tick() already computed this step (pre/post-activation
    at each edge's src/dst, the reward-modulation term, which edges are
    frozen, and the graph's shared clip settings), return the new
    weight tensor plus the raw delta (needed by tick()'s E_drift stat)
    and a clip-hit count. All arguments are full edge-indexed tensors
    -- this must stay vectorized, never a per-edge Python loop, to keep
    tick() usable at real graph scale."""

    def update_weights(self, w, x_u, x_v, modulation, frozen, use_hard_clip, max_weight):
        raise NotImplementedError

    def normalize(self, w, frozen, src, n):
        """Optional post-update step, applied once per tick after
        update_weights() and its own clamp. Default: no-op -- a rule's
        own update_weights() is assumed sufficient unless it opts into
        more. Must return (new_w, fires_count): new_w the same shape as
        w, fires_count how many nodes this step actually rescaled (for
        tick()'s w_norm_fires stat; 0 for a no-op)."""
        return w, 0


class HebbianLearning(LearningRule):
    """The rule this graph has always used, unchanged: Δw = eta * mod *
    x_u * x_v - lam * w, frozen edges never move, then clamped to
    [0, max_weight] (hard clip) or just [0, inf) (soft/no upper clip
    here -- normalize() below is what actually bounds unclipped growth
    in that mode). eta/lam default to this graph's long-standing values
    (1e-2, 1e-3) so an un-configured HebbianLearning() is byte-for-byte
    identical to the old hardcoded formula."""

    def __init__(self, eta=1e-2, lam=1e-3):
        self.eta = eta
        self.lam = lam

    def update_weights(self, w, x_u, x_v, modulation, frozen, use_hard_clip, max_weight):
        delta_w = self.eta * modulation * x_u * x_v - self.lam * w
        delta_w = torch.where(frozen, torch.zeros_like(delta_w), delta_w)
        w_raw = w + delta_w
        if use_hard_clip:
            w_clamped = w_raw.clamp(min=0.0, max=max_weight)
        else:
            w_clamped = w_raw.clamp(min=0.0)
        clip_hits = int((w_clamped != w_raw).sum().item())
        return w_clamped, delta_w, clip_hits

    def normalize(self, w, frozen, src, n):
        """Per-source-node outgoing-weight normalize, moved here
        verbatim from tick(): scale = min(1, 2.0/(mean(w_out)+eps)) --
        only shrinks when the mean EXCEEDS 2.0, otherwise leaves
        weights untouched. Frozen edges excluded/untouched. This is a
        companion backstop specifically for Hebbian's unbounded-growth
        tendency (the 2.0 threshold matches max_weight's default, a
        second layer of defense on top of the per-edge clamp above) --
        not a universal necessity every rule needs, which is why it's
        a rule-specific override, not something tick() always runs.
        Note: the 2.0 here is a literal, not self.max_weight or the
        max_weight passed into update_weights() -- preserved exactly
        as the original code had it (a pre-existing quirk, not fixed
        here to keep this a pure extraction, not a behavior change)."""
        out_degree = torch.zeros(n)
        out_degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        w_for_norm = torch.where(frozen, torch.zeros_like(w), w)
        w_sum = torch.zeros(n)
        w_sum.index_add_(0, src, w_for_norm)
        mean_w_out = torch.where(out_degree > 0, w_sum / out_degree.clamp(min=1), torch.zeros(n))
        scale = torch.clamp(2.0 / (mean_w_out + 1e-8), max=1.0)
        fires = int((scale < 1.0 - 1e-9).sum().item())
        per_edge_scale = scale[src]
        w_final = torch.where(frozen, w, w * per_edge_scale)
        return w_final, fires


class OjaLearning(LearningRule):
    """Oja's rule (Oja, 1982): Δw = eta * (x_u*x_v - w*x_v^2) -- the
    same Hebbian co-activation term, but decay is proportional to w
    TIMES THE SQUARE of the post-synaptic activation, not a constant
    rate. Self-limiting by construction: as a weight grows, its own
    contribution to x_v grows too, feeding back into a bigger decay
    pull on itself -- the classic online algorithm for converging
    toward a principal eigenvector, not an ad-hoc external clamp.

    No `lam` parameter at all -- Oja's own w*x_v^2 term IS the decay,
    not a supplement to a separate constant-rate one; accepting a lam
    nobody uses would misrepresent the rule. Deliberately does NOT
    override normalize() -- Oja's rule is supposed to be self-
    normalizing on its own terms, so this tests that claim honestly
    rather than running Hebbian's companion rescale on top of it (see
    EXPERIMENT_LOG.md for why the first comparison, which did exactly
    that, gave a misleading result). Still keeps weights non-negative
    via the same clamp(min=0.0) as Hebbian, matching this graph's
    excitatory-only activation model -- that clamp is a basic safety
    bound every rule needs, not something Hebbian-specific."""

    def __init__(self, eta=1e-2):
        self.eta = eta

    def update_weights(self, w, x_u, x_v, modulation, frozen, use_hard_clip, max_weight):
        delta_w = self.eta * modulation * (x_u * x_v - w * x_v ** 2)
        delta_w = torch.where(frozen, torch.zeros_like(delta_w), delta_w)
        w_raw = w + delta_w
        if use_hard_clip:
            w_clamped = w_raw.clamp(min=0.0, max=max_weight)
        else:
            w_clamped = w_raw.clamp(min=0.0)
        clip_hits = int((w_clamped != w_raw).sum().item())
        return w_clamped, delta_w, clip_hits
