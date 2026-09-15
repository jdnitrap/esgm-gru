"""ESGR graph substrate. E_contr is now a first-class energy term
computed inside tick() itself for any registered contradiction pairs —
same function used by the logger and Task C, no separate script-level
rule. Update rule: dtau_i = -LR*tau_j (and symmetric for j) — this is
the ONLY thing that moves tau for a contradiction pair; the
"lower-tau-decays-faster" property falls out of the math (losing
-LR*tau_other is a bigger relative hit to whichever side already has
less), not a coded special case. Frozen edges in a pair never move;
if the other side is unfrozen it still decays against the frozen
edge's fixed tau.
"""
import json
import torch

from learning_rules import LearningRule, HebbianLearning

CONTRADICTION_LR = 0.05
TAU_TRUST_THRESHOLD = 0.3


class ESGRGraph:
    def __init__(self, n_nodes=8192, mean_out_degree=12, active_fraction=0.05,
                 eta=1e-2, lam=1e-3, seed=0, max_activation=5.0, max_weight=2.0,
                 use_hard_clip=False, modulation_decay=0.97, max_activation_rate=None,
                 split_sparsity_at=None, learning_rule: LearningRule = None):
        g = torch.Generator().manual_seed(seed)
        self.n = n_nodes
        self.active_fraction = active_fraction
        self.eta = eta
        self.lam = lam
        self.max_activation = max_activation
        self.max_weight = max_weight
        self.use_hard_clip = use_hard_clip
        # Pluggable weight-update rule -- see learning_rules.py. None
        # (default) is HebbianLearning(), the exact rule this graph
        # always used; every existing caller is unaffected unless it
        # opts into a different rule.
        self.learning_rule = learning_rule if learning_rule is not None else HebbianLearning()

        pairs = set()
        src_list, dst_list = [], []
        for u in range(n_nodes):
            degree = max(1, int(torch.normal(
                torch.tensor(float(mean_out_degree)), torch.tensor(3.0), generator=g).item()))
            targets = torch.randint(0, n_nodes, (degree * 2,), generator=g).tolist()
            added = 0
            for v in targets:
                if v == u or (u, v) in pairs:
                    continue
                pairs.add((u, v))
                src_list.append(u)
                dst_list.append(v)
                added += 1
                if added >= degree:
                    break
        self.src = torch.tensor(src_list, dtype=torch.long)
        self.dst = torch.tensor(dst_list, dtype=torch.long)
        E = self.src.shape[0]
        self.edge_index = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(self.src, self.dst))}

        self.w = torch.empty(E).uniform_(0.0, 0.5, generator=g)
        self.tau = torch.full((E,), 0.5)
        self.c = torch.empty(E).uniform_(0.01, 0.1, generator=g)
        self.last_use = torch.full((E,), -1000)
        self.frozen = torch.zeros(E, dtype=torch.bool)
        self.confirmed = torch.zeros(E, dtype=torch.bool)
        self.rejected = torch.zeros(E, dtype=torch.bool)
        self.suspended = torch.zeros(E, dtype=torch.bool)
        self.contradiction_pairs = []

        self.theta = torch.full((n_nodes,), 0.3)
        self.x = torch.zeros(n_nodes)
        self.tick_count = 0

        # Three-factor (reward-modulated) Hebbian plasticity. Plain
        # Hebbian (below) can't tell a meaningful co-firing pattern from
        # a coincidental one — it strengthens both identically. modulation
        # is a third, per-node input to the weight update, starting
        # neutral (1.0) everywhere, that FactGate nudges via reward()/
        # punish() when an edge actually gets confirmed or rejected. A
        # graph nobody ever confirms/rejects behaves byte-for-byte like
        # the plain two-factor rule — this is additive, not a replacement.
        self.modulation = torch.ones(n_nodes)
        self.modulation_decay = modulation_decay

        # Rate limiter on the message-passing (cascading, internal)
        # contribution to activation -- a leaky-bucket, same principle
        # network traffic shaping uses: let signal through at a bounded
        # rate instead of letting a burst fully land in one shot. Found
        # by testing (2026-09-14): teaching back hundreds of thousands
        # of real predictions strengthened enough ordinary byte-to-byte
        # edges that word-level activity, direct-poked at a stable 3.0,
        # got buried under cascading byte activity reaching 16 within
        # TWO ticks -- nothing bounded how fast a newly-strengthened
        # cluster of edges could resonate. None (default) disables this
        # entirely -- byte-for-byte identical to the old behavior --
        # so every existing test and prior run is unaffected unless a
        # caller opts in. Deliberately does NOT limit external_input
        # (a direct poke): that's added after this, at full strength,
        # unrate-limited -- only the emergent, internal cascade is
        # smoothed, not deliberate stimulation.
        self.max_activation_rate = max_activation_rate

        # Split kWTA budget -- see _enforce_sparsity()'s docstring.
        # None (default) preserves the exact original single-pool
        # behavior; every existing test and prior run is unaffected
        # unless a caller opts in.
        self.split_sparsity_at = split_sparsity_at

    def find_edge(self, src, dst):
        return self.edge_index.get((src, dst))

    def grow(self, n_new_nodes, mean_out_degree=8, seed=None):
        """Add n_new_nodes fresh nodes to the graph at runtime -- the
        thing that was architecturally impossible before: n was fixed
        forever at construction. Every existing node id, edge id, and
        piece of frozen/confirmed state is untouched; new edges are only
        ever APPENDED, never inserted, so no existing edge index moves
        and the "frozen edges never move" invariant (test_integration_
        stress.py) holds automatically.

        New nodes get the same random-topology OUTGOING wiring every
        original node got at construction (so they're ordinary,
        learnable graph citizens, not dead weight) -- but deliberately
        get no automatic INCOMING wiring from old nodes. For the actual
        use case this exists for (shell.py's `tile` command growing a
        node for a brand-new word), the real, meaningful connectivity is
        the caller then adding via add_fixed_edge()/wire_word_structure()
        -- letters in, role out -- exactly how all 31 existing word
        tiles work, not through random topology.

        Returns the list of new node ids.
        """
        g_gen = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        old_n = self.n
        new_n = old_n + n_new_nodes

        self.x = torch.cat([self.x, torch.zeros(n_new_nodes)])
        self.theta = torch.cat([self.theta, torch.full((n_new_nodes,), 0.3)])
        self.modulation = torch.cat([self.modulation, torch.ones(n_new_nodes)])

        pairs = set(self.edge_index.keys())
        src_list, dst_list = [], []
        for u in range(old_n, new_n):
            degree = max(1, int(torch.normal(
                torch.tensor(float(mean_out_degree)), torch.tensor(3.0), generator=g_gen).item()))
            targets = torch.randint(0, new_n, (degree * 2,), generator=g_gen).tolist()
            added = 0
            for v in targets:
                if v == u or (u, v) in pairs:
                    continue
                pairs.add((u, v))
                src_list.append(u)
                dst_list.append(v)
                added += 1
                if added >= degree:
                    break

        n_new_edges = len(src_list)
        start_idx = self.src.shape[0]
        self.src = torch.cat([self.src, torch.tensor(src_list, dtype=torch.long)])
        self.dst = torch.cat([self.dst, torch.tensor(dst_list, dtype=torch.long)])
        self.w = torch.cat([self.w, torch.empty(n_new_edges).uniform_(0.0, 0.5, generator=g_gen)])
        self.tau = torch.cat([self.tau, torch.full((n_new_edges,), 0.5)])
        self.c = torch.cat([self.c, torch.empty(n_new_edges).uniform_(0.01, 0.1, generator=g_gen)])
        self.last_use = torch.cat([self.last_use, torch.full((n_new_edges,), -1000)])
        self.frozen = torch.cat([self.frozen, torch.zeros(n_new_edges, dtype=torch.bool)])
        self.confirmed = torch.cat([self.confirmed, torch.zeros(n_new_edges, dtype=torch.bool)])
        self.rejected = torch.cat([self.rejected, torch.zeros(n_new_edges, dtype=torch.bool)])
        self.suspended = torch.cat([self.suspended, torch.zeros(n_new_edges, dtype=torch.bool)])
        for i, (s, d) in enumerate(zip(src_list, dst_list)):
            self.edge_index[(s, d)] = start_idx + i

        self.n = new_n
        return list(range(old_n, new_n))

    def register_contradiction(self, i, j):
        self.contradiction_pairs.append((i, j))

    def add_learnable_edge(self, src, dst, w=0.1, tau=0.5):
        """Create a real, ORDINARY (non-frozen) edge between two existing
        nodes if one doesn't already exist -- the gap add_fixed_edge()
        can't fill, since it always freezes. Needed for supervised
        correction (see supervise.py): pushing weight toward the real
        next word in real text requires an edge to push, even when
        nothing wired one there by chance. Append-only, same as grow()
        and add_fixed_edge() -- never disturbs an existing edge index."""
        existing = self.find_edge(src, dst)
        if existing is not None:
            return existing
        i = self.src.shape[0]
        self.src = torch.cat([self.src, torch.tensor([src])])
        self.dst = torch.cat([self.dst, torch.tensor([dst])])
        self.w = torch.cat([self.w, torch.tensor([w])])
        self.tau = torch.cat([self.tau, torch.tensor([tau])])
        self.c = torch.cat([self.c, torch.tensor([0.05])])
        self.last_use = torch.cat([self.last_use, torch.tensor([-1000])])
        self.frozen = torch.cat([self.frozen, torch.tensor([False])])
        self.confirmed = torch.cat([self.confirmed, torch.tensor([False])])
        self.rejected = torch.cat([self.rejected, torch.tensor([False])])
        self.suspended = torch.cat([self.suspended, torch.tensor([False])])
        self.edge_index[(src, dst)] = i
        return i

    def reward(self, node_ids, amount=0.3):
        """Bump modulation up for these nodes — called by FactGate when
        an edge touching them gets confirmed. Nudges nearby Hebbian
        updates to trust similar future co-firing more, for a while.
        amount=0.3 and modulation_decay=0.97 (half-life ~23 ticks) are
        tuned against the real corpus-mined training data, not guessed:
        0.5 nearly pinned modulation at its 3.0 ceiling the moment
        several real confirmations landed close together (14 edges in
        the live graph.json were already primed above the confirm
        threshold, and cleared it within the first few ticks of running
        step() for the first time), which collapses the signal to just
        "maxed or not." 0.3 peaked at 1.9 under the same real event,
        leaving headroom for the signal to stay graduated."""
        idx = torch.tensor(node_ids, dtype=torch.long)
        self.modulation[idx] = torch.clamp(self.modulation[idx] + amount, max=3.0)

    def punish(self, node_ids, amount=0.3):
        """Mirror of reward() — called by FactGate.reject()."""
        idx = torch.tensor(node_ids, dtype=torch.long)
        self.modulation[idx] = torch.clamp(self.modulation[idx] - amount, min=0.0)

    def propose(self, src):
        """Miss-propose only. Read-only — never sets confirmed/rejected,
        never writes graph.json, never touches tick() math.
        HIT: any outgoing edge that's confirmed, not rejected, not
        suspended. MISS: up to 5 unconfirmed, not-rejected outgoing
        edges, ranked by (tau*w) descending — candidates only.
        """
        out_edges = (self.src == src).nonzero().flatten().tolist()
        hits = [i for i in out_edges
                if bool(self.confirmed[i]) and not bool(self.rejected[i]) and not bool(self.suspended[i])]
        if hits:
            return {"status": "HIT", "edges": hits}
        candidates = [i for i in out_edges if not bool(self.confirmed[i]) and not bool(self.rejected[i])]
        candidates.sort(key=lambda i: (self.tau[i] * self.w[i]).item(), reverse=True)
        return {"status": "MISS", "edges": candidates[:5]}

    def add_fixed_edge(self, src, dst, weight=1.0, trust=1.0, confirmed=True):
        if src >= self.n or dst >= self.n:
            # Fail loudly and immediately, not later as a confusing
            # IndexError inside tick() -- found by testing: tiles.json
            # can now reference nodes from a grown, larger graph, and a
            # smaller fresh graph silently accepted an edge to a node it
            # doesn't have, only crashing on the next tick().
            raise ValueError(f"add_fixed_edge({src},{dst}): node id >= graph.n ({self.n})")
        if (src, dst) in self.edge_index:
            # Coincidental collision with a pre-existing random-topology
            # edge (category nodes are valid random targets too) — must
            # be UPGRADED to frozen/confirmed, not left as an ordinary
            # Hebbian edge. Found by testing: 7 such collisions moved
            # under Hebbian update despite being "MDBE" edges.
            i = self.edge_index[(src, dst)]
            self.w[i] = weight
            self.tau[i] = trust
            self.frozen[i] = True
            self.confirmed[i] = confirmed
            self.rejected[i] = False
            self.suspended[i] = False
            return i
        i = self.src.shape[0]
        self.src = torch.cat([self.src, torch.tensor([src])])
        self.dst = torch.cat([self.dst, torch.tensor([dst])])
        self.w = torch.cat([self.w, torch.tensor([weight])])
        self.tau = torch.cat([self.tau, torch.tensor([trust])])
        self.c = torch.cat([self.c, torch.tensor([0.0])])
        self.last_use = torch.cat([self.last_use, torch.tensor([-1000])])
        self.frozen = torch.cat([self.frozen, torch.tensor([True])])
        self.confirmed = torch.cat([self.confirmed, torch.tensor([confirmed])])
        self.rejected = torch.cat([self.rejected, torch.tensor([False])])
        self.suspended = torch.cat([self.suspended, torch.tensor([False])])
        self.edge_index[(src, dst)] = i
        return i

    def _topk_select(self, x: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
        """The actual top-k/stochastic-k selection, factored out of
        _enforce_sparsity() so split-budget mode (below) can apply the
        exact same, already-tested logic to two independent slices
        instead of duplicating it.

        temperature=0.0: deterministic hard top-k, byte-for-byte
        unchanged from the original single-pool implementation.

        temperature>0.0: stochastic — sample k without replacement,
        weighted by softmax(x/temperature), restricted to a real
        candidate pool first (see _enforce_sparsity's docstring for
        why: softmax over a mostly-zero field dilutes real signal).
        """
        if temperature <= 0.0:
            topk_vals, topk_idx = torch.topk(x, k)
            out = torch.zeros_like(x)
            out[topk_idx] = topk_vals
            return out
        pool_size = min(x.shape[0], min(k * 3, max(1, int((x > 0).sum().item()))))
        pool_vals, pool_idx = torch.topk(x, pool_size)
        probs = torch.softmax(pool_vals / temperature, dim=0)
        chosen = torch.multinomial(probs, min(k, pool_size), replacement=False)
        idx = pool_idx[chosen]
        out = torch.zeros_like(x)
        out[idx] = x[idx]
        return out

    def _enforce_sparsity(self, x: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
        """Hard quota, not a threshold club: ALWAYS keep exactly the top
        k = floor(active_fraction * n) values. theta is not used here;
        removing theta from the node update (see tick()) is what lets
        this quota actually bind at exactly k every tick instead of
        silently returning fewer.

        temperature=0.0 (default): the ORIGINAL deterministic hard
        top-k, byte-for-byte unchanged — every existing test, Task C,
        and training run above this line in the file's history depends
        on this exact determinism and is untouched.

        temperature>0.0: stochastic quota — sample k nodes WITHOUT
        replacement, weighted by softmax(x/temperature), instead of
        always taking the literal top-k. This is the fix for generation
        converging to a fixed repeating attractor: a fully deterministic
        selection rule has no way to ever move on from a stable point.
        Same role as sampling temperature in ordinary language-model
        decoding — applied here to WHICH NODES fire, not to a
        vocabulary distribution, since there is no vocabulary here.

        split_sparsity_at (constructor param, default None = disabled):
        when set to a node index, the quota is computed and applied
        SEPARATELY for [0, split) and [split, n) -- e.g. split=256
        gives byte nodes their own k and every word-level node (tiles,
        role hubs, category hubs, grammar_extra hubs) a completely
        separate k, so byte-level activity can never occupy a
        word-level seat no matter how much of it there is. Found
        necessary by testing (2026-09-14): a rate limiter alone (see
        max_activation_rate) slows how fast any ONE node's activation
        can climb, but does nothing to stop many DIFFERENT reinforced
        byte pathways from collectively filling a single shared budget
        -- teaching a trained head's real predictions back into the
        graph strengthened 497 ordinary edges, and even individually
        rate-limited, enough of them climbing together crowded out
        word-level generation entirely. Splitting the budget is the
        structural fix: capacity, not just speed.
        """
        if self.split_sparsity_at is None:
            k = max(1, int(self.active_fraction * self.n))
            return self._topk_select(x, k, temperature)
        split = self.split_sparsity_at
        k_a = max(1, int(self.active_fraction * split))
        k_b = max(1, int(self.active_fraction * (self.n - split)))
        out_a = self._topk_select(x[:split], k_a, temperature)
        out_b = self._topk_select(x[split:], k_b, temperature)
        return torch.cat([out_a, out_b])

    def tick(self, external_input: torch.Tensor = None, temperature: float = 0.0):
        clip_hits = 0
        nan_count = 0
        x = self.x if external_input is None else self.x + external_input

        trusted = self.tau >= TAU_TRUST_THRESHOLD
        m_raw = self.w[trusted] * x[self.src[trusted]]
        m = m_raw / (1.0 + m_raw.abs())  # bound the PRODUCT, not a clip on x
        sat_hits = int((m > 0.99).sum().item())
        agg = torch.zeros(self.n)
        agg.index_add_(0, self.dst[trusted], m)

        raw = torch.relu(agg)  # theta unused while hard quota is on
        if self.max_activation_rate is not None:
            raw = torch.min(raw, self.x + self.max_activation_rate)
        if external_input is not None:
            raw = raw + external_input  # a poked node fires this tick, before kWTA
        if self.use_hard_clip:
            clipped = raw.clamp(max=self.max_activation)
            clip_hits += int((clipped != raw).sum().item())
            raw = clipped
        x_new = self._enforce_sparsity(raw, temperature=temperature)

        x_u = x[self.src]
        x_v = x_new[self.dst]
        # mod is 1.0 for every edge until reward()/punish() has touched
        # src or dst, so this multiply is a no-op (identical to the old
        # two-factor rule) unless something has actually been confirmed
        # or rejected nearby.
        mod = 0.5 * (self.modulation[self.src] + self.modulation[self.dst])
        w_clamped, delta_w, rule_clip_hits = self.learning_rule.update_weights(
            self.w, x_u, x_v, mod, self.frozen, self.eta, self.lam, self.use_hard_clip, self.max_weight)
        clip_hits += rule_clip_hits

        # Per-source-node outgoing-weight normalize: scale = min(1, 2.0
        # / (mean(w_out)+eps)) — only shrinks when the mean EXCEEDS 2.0,
        # otherwise leaves weights untouched (typical mass can sit near
        # 2, not forced down to <=1). Frozen edges excluded/untouched.
        out_degree = torch.zeros(self.n)
        out_degree.index_add_(0, self.src, torch.ones_like(self.src, dtype=torch.float))
        w_for_norm = torch.where(self.frozen, torch.zeros_like(w_clamped), w_clamped)
        w_sum = torch.zeros(self.n)
        w_sum.index_add_(0, self.src, w_for_norm)
        mean_w_out = torch.where(out_degree > 0, w_sum / out_degree.clamp(min=1), torch.zeros(self.n))
        scale = torch.clamp(2.0 / (mean_w_out + 1e-8), max=1.0)
        w_norm_fires = int((scale < 1.0 - 1e-9).sum().item())
        per_edge_scale = scale[self.src]
        w_final = torch.where(self.frozen, w_clamped, w_clamped * per_edge_scale)
        self.w = w_final

        if torch.isnan(self.w).any() or torch.isnan(x_new).any():
            nan_count += 1  # detected honestly, never masked with nan_to_num

        used = trusted & (x_u > 0) & (x_v > 0)
        # Idle decay floors at TAU_TRUST_THRESHOLD, not 0. Found by
        # testing at real scale (5M-tick corpus run): unconditional
        # decay toward 0 on every not-used edge, every tick, silently
        # prunes almost the entire non-frozen graph once tick counts
        # reach the millions (mean tau collapsed to 0.169), starving
        # message-passing under narrow single-region stimuli and
        # forcing the kWTA quota to fill with zero-valued phantom
        # slots. An edge that goes quiet now settles at "just barely
        # trusted" instead of fully pruned; a used edge still climbs
        # normally above the floor via reinforcement.
        new_tau = torch.where(used, torch.clamp(self.tau + 0.01, max=1.0),
                               torch.clamp(self.tau - 0.001, min=TAU_TRUST_THRESHOLD))
        self.tau = torch.where(self.frozen, self.tau, new_tau)
        self.last_use = torch.where(used, torch.full_like(self.last_use, self.tick_count),
                                     self.last_use)

        # E_contr — the ONLY thing that touches tau below is this rule,
        # applied identically regardless of which edge was confirmed
        # first. No special-casing. E_contr > 0.5 sets suspended=True on
        # both edges (does NOT auto-resolve, does NOT stop tau decay —
        # only confirm()/reject() clears suspended).
        e_contr_total = 0.0
        newly_suspended = []
        for (i, j) in self.contradiction_pairs:
            if not (self.confirmed[i] and self.confirmed[j]):
                continue
            tau_i, tau_j = self.tau[i].item(), self.tau[j].item()
            e_contr = tau_i * tau_j
            e_contr_total += e_contr
            if e_contr > 0.5 and not (bool(self.suspended[i]) or bool(self.suspended[j])):
                self.suspended[i] = True
                self.suspended[j] = True
                newly_suspended.append((i, j))
            frozen_i, frozen_j = bool(self.frozen[i]), bool(self.frozen[j])
            if frozen_i and frozen_j:
                continue
            if not frozen_i:
                self.tau[i] = max(0.0, tau_i - CONTRADICTION_LR * tau_j)
            if not frozen_j:
                self.tau[j] = max(0.0, tau_j - CONTRADICTION_LR * tau_i)

        e_drift = (delta_w ** 2).sum().item()
        e_wire = self.c[used].sum().item()
        e_fire = (x_new > 0).float().sum().item()
        if torch.isnan(self.tau).any():
            nan_count += 1
        has_nan = nan_count > 0

        # modulation relaxes back toward neutral (1.0) every tick, so a
        # reward/punishment is a temporary nudge to nearby learning, not
        # a permanent multiplier.
        self.modulation = 1.0 + (self.modulation - 1.0) * self.modulation_decay

        self.x = x_new
        self.tick_count += 1
        return {"E_drift": e_drift, "E_wire": e_wire, "E_fire": e_fire, "E_contr": e_contr_total,
                "active_fraction": e_fire / self.n, "mean_trust": self.tau.mean().item(),
                "clip_hits": clip_hits, "nan": has_nan, "nan_count": nan_count,
                "sat_hits": sat_hits, "w_norm_fires": w_norm_fires, "newly_suspended": newly_suspended,
                "mean_modulation": self.modulation.mean().item()}

    def save_json(self, path):
        data = {
            "n": self.n, "eta": self.eta, "lam": self.lam,
            "active_fraction": self.active_fraction,
            "max_activation": self.max_activation, "max_weight": self.max_weight,
            "use_hard_clip": self.use_hard_clip,
            "modulation": self.modulation.tolist(), "modulation_decay": self.modulation_decay,
            "max_activation_rate": self.max_activation_rate,
            "split_sparsity_at": self.split_sparsity_at,
            "contradiction_pairs": self.contradiction_pairs,
            "edges": [
                {"src": int(s), "dst": int(d), "w": float(w), "tau": float(t), "c": float(c),
                 "last_use": int(lu), "frozen": bool(f), "confirmed": bool(cf), "rejected": bool(rj),
                 "suspended": bool(sp)}
                for s, d, w, t, c, lu, f, cf, rj, sp in zip(
                    self.src.tolist(), self.dst.tolist(), self.w.tolist(), self.tau.tolist(),
                    self.c.tolist(), self.last_use.tolist(), self.frozen.tolist(),
                    self.confirmed.tolist(), self.rejected.tolist(), self.suspended.tolist())
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(path):
        with open(path) as f:
            data = json.load(f)
        g = ESGRGraph.__new__(ESGRGraph)
        g.n = data["n"]
        g.eta, g.lam = data["eta"], data["lam"]
        g.active_fraction = data["active_fraction"]
        g.max_activation, g.max_weight = data["max_activation"], data["max_weight"]
        g.use_hard_clip = data["use_hard_clip"]
        # older saves predate modulation -- default to neutral (1.0) so
        # they load exactly as before, no behavior change on old files.
        g.modulation = torch.tensor(data["modulation"]) if "modulation" in data else torch.ones(data["n"])
        g.modulation_decay = data.get("modulation_decay", 0.97)
        # older saves predate the rate limiter -- default to None
        # (disabled), identical to their original behavior.
        g.max_activation_rate = data.get("max_activation_rate", None)
        # older saves predate the split-budget kWTA -- default to None
        # (disabled), identical to their original behavior.
        g.split_sparsity_at = data.get("split_sparsity_at", None)
        # learning_rule is never persisted (it's stateless code, not
        # data) -- every load gets the default HebbianLearning(), same
        # as every graph ever saved actually used. Real bug found by
        # testing: load_json() builds via __new__(), bypassing
        # __init__() entirely, so this was missing outright until now,
        # not just defaulted -- any tick() on a loaded graph crashed
        # with AttributeError.
        g.learning_rule = HebbianLearning()
        g.contradiction_pairs = [tuple(p) for p in data["contradiction_pairs"]]
        edges = data["edges"]
        g.src = torch.tensor([e["src"] for e in edges], dtype=torch.long)
        g.dst = torch.tensor([e["dst"] for e in edges], dtype=torch.long)
        g.w = torch.tensor([e["w"] for e in edges])
        g.tau = torch.tensor([e["tau"] for e in edges])
        g.c = torch.tensor([e["c"] for e in edges])
        g.last_use = torch.tensor([e["last_use"] for e in edges], dtype=torch.long)
        g.frozen = torch.tensor([e["frozen"] for e in edges], dtype=torch.bool)
        g.confirmed = torch.tensor([e["confirmed"] for e in edges], dtype=torch.bool)
        g.rejected = torch.tensor([e["rejected"] for e in edges], dtype=torch.bool)
        g.suspended = torch.tensor([e["suspended"] for e in edges], dtype=torch.bool)
        g.edge_index = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(g.src, g.dst))}
        g.theta = torch.full((g.n,), 0.3)
        g.x = torch.zeros(g.n)
        g.tick_count = 0
        return g
