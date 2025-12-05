import os
import re
import shutil
from _io import TextIOWrapper
from itertools import product


def get_services_from_prism_file(f: TextIOWrapper):
    SERVICE_RE = re.compile(r"^\s*const\s+int\s+c_([A-Za-z_]\w*)\s*=")

    pos = f.tell()
    try:
        services = []
        seen = set()

        for line in f:
            m = SERVICE_RE.match(line)
            if m:
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    services.append(name)

        return services
    finally:
        f.seek(pos)


def get_hat_variables_with_ranges_and_init(f: TextIOWrapper):
    CONST_RE = re.compile(
        r"^\s*const\s+(?:int|double)\s+([A-Za-z_]\w*)\s*=\s*([0-9.]+)\s*;"
    )

    HAT_DECL_RE = re.compile(
        r"^\s*([A-Za-z_]\w*)hat\s*:\s*"
        r"\[\s*([0-9A-Za-z_]+)\s*\.\.\s*([0-9A-Za-z_]+)\s*\]\s*"
        r"init\s+([A-Za-z_]\w*)"
    )

    pos = f.tell()
    try:
        constants = {}
        hat_vars = {}

        lines = f.readlines()

        for line in lines:
            m = CONST_RE.match(line)
            if m:
                name, value = m.groups()
                constants[name] = float(value) if "." in value else int(value)

        for line in lines:
            m = HAT_DECL_RE.match(line)
            if m:
                base, lo_raw, hi_raw, init_raw = m.groups()

                lo = int(lo_raw) if lo_raw.isdigit() else constants.get(lo_raw)
                hi = int(hi_raw) if hi_raw.isdigit() else constants.get(hi_raw)
                init = int(init_raw) if init_raw.isdigit() else constants.get(init_raw)

                if lo is None or hi is None:
                    raise ValueError(
                        f"Cannot resolve range for {base}hat: [{lo_raw}..{hi_raw}]"
                    )
                if init is None:
                    raise ValueError(
                        f"Cannot resolve init value for {base}hat: init {init_raw}"
                    )

                hat_vars[base] = {"range": (lo, hi), "init": init}

        return hat_vars
    finally:
        f.seek(pos)


class ParleyPlusURC:
    def __init__(
        self,
        infile,
        min_val=1,
        max_val=10,
        actions=("east", "west", "north", "south"),
        transition_after_update=False,  # <-- add
    ):
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.actions = list(actions)
        self.transition_after_update = bool(transition_after_update)

        with open(infile, "r") as f:
            self.services = get_services_from_prism_file(f)
            self.features = get_hat_variables_with_ranges_and_init(f)

        # stable order for decision naming and combo generation
        self.feature_names = list(self.features.keys())

        self.domains = []
        for name in self.feature_names:
            lo, hi = self.features[name]["range"]
            self.domains.append(range(lo, hi + 1))

        self.combinations = list(product(*self.domains))

    # ---------- naming helpers ----------

    def _decision_var(self, service: str, combo) -> str:
        parts = [f"{name}_{value}" for name, value in zip(self.feature_names, combo)]
        return f"{service}_decision_" + "_".join(parts)

    def _hat_guard(self, combo) -> str:
        return " & ".join(
            f"{name}hat={value}" for name, value in zip(self.feature_names, combo)
        )

    # ---------- pipeline ----------

    def transform_file(self, infile, outfile, popfile):
        TURN_START_RE = re.compile(r"^\s*module\s+Turn\b")
        ENDMODULE_RE = re.compile(r"^\s*endmodule\b")

        # 1) Copy base model, but:
        #    - drop original Turn module
        #    - drop fixed const int c_<service> = ...;
        with open(infile, "r") as fin, open(outfile, "w") as fout:
            skipping_turn = False
            for line in fin:
                if TURN_START_RE.match(line):
                    skipping_turn = True
                    continue
                if skipping_turn:
                    if ENDMODULE_RE.match(line):
                        skipping_turn = False
                    continue

                if any(
                    line.strip().startswith(f"const int c_{s}") for s in self.services
                ):
                    continue

                fout.write(line)

        # 3) Append added modules/decls
        with open(outfile, "a") as f:
            self.add_urc(f)
            self.add_turn(f)

        # 4) Population file (same number of evolvables in both modes here)
        with open(popfile, "w") as f:
            self.create_pop_file(f)

    # ---------- codegen (normal mode) ----------

    def add_urc(self, f: TextIOWrapper):
        names = list(self.features.keys())

        # evolve int <service>_decision_<obs>
        for service in self.services:
            for combo in product(*self.domains):
                parts = [f"{name}_{value}" for name, value in zip(names, combo)]
                label = "decision_" + "_".join(parts)
                f.write(
                    f"evolve int {service}_{label} [{self.min_val}..{self.max_val}];\n"
                )

        # init label from hat init values
        init_parts = [f"{name}_{info['init']}" for name, info in self.features.items()]
        init_label = "decision_" + "_".join(init_parts)

        f.write("\nmodule URC\n")
        for service in self.services:
            f.write(
                f"  c_{service} : [{self.min_val}..{self.max_val}] init {service}_{init_label};\n"
            )

        f.write("\n  // URC transitions\n")
        for combo in product(*self.domains):
            guard = " & ".join(
                f"{name}hat={value}" for name, value in zip(names, combo)
            )
            updates = " & ".join(
                f"(c_{service}'={service}_decision_"
                + "_".join(f"{name}_{value}" for name, value in zip(names, combo))
                + ")"
                for service in self.services
            )
            f.write(f"  [URC] {guard} -> {updates};\n")

        f.write("endmodule\n\n")

    def add_turn(self, f: TextIOWrapper):
        if not self.transition_after_update:
            # --- current behavior ---
            f.write("module Turn\n")
            f.write("  t : [0..2] init 0;\n")
            for a in self.actions:
                f.write(f"  [{a}] (t=0) -> (t'=1);\n")

            f.write("\n  [URC] (t=1) -> (t'=2);\n\n")

            for s in self.services:
                f.write(f"  [update_{s}] (t=2) -> (t'=0);\n")
            f.write("  [skip_update] (t=2) -> (t'=0);\n")
        else:
            f.write("module Turn\n")
            f.write("  t : [0..2] init 0;\n")

            # movement phase
            for a in self.actions:
                f.write(f"  [{a}] (t=0) -> (t'=1);\n")

            f.write("\n  // Update phase\n")
            # if an update happens, go to URC phase
            for s in self.services:
                f.write(f"  [update_{s}] (t=1) -> (t'=2);\n")
            # if we skip, go straight back to movement (no URC)
            f.write("  [skip_update] (t=1) -> (t'=0);\n")

            # URC phase (only reachable after update_*)
            f.write("\n  [URC] (t=2) -> (t'=0);\n")
        f.write("endmodule\n")

    # ---------- popfile ----------

    def create_pop_file(self, f: TextIOWrapper):
        num_vars = len(self.services) * len(self.combinations)
        for c in range(self.min_val, self.max_val + 1):
            f.write((" ".join([str(c)] * num_vars)) + "\n")


from _io import TextIOWrapper
from itertools import product


class ParleyUAMealy:
    def __init__(
        self,
        infile: str,
        min_val: int = 1,
        max_val: int = 10,
        actions=("east", "west", "north", "south"),
        internal_states: int = 10,
        transition_after_update: bool = False,
        range_split: bool = False,
    ):
        self.infile = infile
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.actions = list(actions)
        self.internal_states = int(internal_states)
        self.transition_after_update = bool(transition_after_update)
        self.range_split = bool(range_split)

        self._state_ranges = self._compute_state_ranges() if self.range_split else None


        with open(infile, "r") as f:
            self.services = get_services_from_prism_file(f)
            self.features = get_hat_variables_with_ranges_and_init(f)

        if not self.services:
            raise ValueError(
                "No services found (expected const int c_<service> = ...;)."
            )
        if not self.features:
            raise ValueError(
                "No *hat variables found (expected xhat/yhat/etc declarations)."
            )

        # stable order
        self.feature_names = list(self.features.keys())

        # observation domains
        self.domains = []
        for name in self.feature_names:
            lo, hi = self.features[name]["range"]
            self.domains.append(range(lo, hi + 1))
        self.combinations = list(product(*self.domains))

    # ---------- naming helpers ----------

    def _tr_act_symbol(self, action: str, s: int) -> str:
        return f"ua_tr_act_{action}_s{s}"

    def _tr_obs_symbol(self, s: int, combo) -> str:
        parts = [f"{name}_{val}" for name, val in zip(self.feature_names, combo)]
        return f"ua_tr_obs_s{s}_" + "_".join(parts)

    def _out_obs_symbol(self, service: str, s: int, combo) -> str:
        parts = [f"{name}_{val}" for name, val in zip(self.feature_names, combo)]
        return f"ua_out_obs_{service}_s{s}_" + "_".join(parts)

    # ---------- pipeline ----------

    def transform_file(self, infile: str, outfile: str, popfile: str):
        TURN_START_RE = re.compile(r"^\s*module\s+Turn\b")
        ENDMODULE_RE = re.compile(r"^\s*endmodule\b")

        with open(infile, "r") as fin, open(outfile, "w") as fout:
            skipping_turn = False

            for line in fin:
                # drop original Turn module completely
                if TURN_START_RE.match(line):
                    skipping_turn = True
                    continue
                if skipping_turn:
                    if ENDMODULE_RE.match(line):
                        skipping_turn = False
                    continue

                # drop fixed const counters: const int c_<service> = ...;
                if any(
                    line.strip().startswith(f"const int c_{s}") for s in self.services
                ):
                    continue

                fout.write(line)

        with open(outfile, "a") as f:
            self.add_ua(f)
            self.add_turn(f)

        with open(popfile, "w") as f:
            self.create_pop_file(f)

    # ---------- codegen ----------

    def add_ua(self, f: TextIOWrapper):
        f.write("\n// ===== ParleyUAMealy declarations =====\n")
        
        if self.range_split:  # <-- ADD
            f.write(f"evolve int ua_init_s [1..{self.internal_states}];\n")
        
        for service in self.services:
            f.write(
                f"evolve int ua_init_c_{service} "
                f"[{self.min_val}..{self.max_val}];\n"
            )
        f.write("\n")

        for s in range(1, self.internal_states + 1):
            for combo in self.combinations:
                f.write(
                    f"evolve int {self._tr_obs_symbol(s, combo)} [1..{self.internal_states}];\n"
                )
                for service in self.services:
                    lo, hi = self._out_range_for_state(s)
                    f.write(
                        f"evolve int {self._out_obs_symbol(service, s, combo)} "
                        f"[{lo}..{hi}];\n"
                    )

        f.write("\n")

        f.write("module UA\n")
        init_s = "ua_init_s" if self.range_split else "1"
        f.write(f"  ua_s : [1..{self.internal_states}] init {init_s};\n")


        for service in self.services:
            f.write(
                f"  c_{service} : [{self.min_val}..{self.max_val}] init ua_init_c_{service};\n"
            )

        # Observation transitions (Mealy)
        f.write("  // Observation transitions (Mealy)\n")

        # In speed_mode, fuse observation into update/skip_update labels
        fused_labels = [f"update_{svc}" for svc in self.services] + ["skip_update"]

        for s in range(1, self.internal_states + 1):
            for combo in self.combinations:
                guard = " & ".join(
                    f"{name}hat={val}" for name, val in zip(self.feature_names, combo)
                )

                updates = [f"(ua_s'={self._tr_obs_symbol(s, combo)})"]
                for service in self.services:
                    updates.append(
                        f"(c_{service}'={self._out_obs_symbol(service, s, combo)})"
                    )

                f.write(
                    f"  [URC] (ua_s={s}) & {guard} -> " + " & ".join(updates) + ";\n"
                )
        f.write("\n")
        f.write("endmodule\n\n")

    def add_turn(self, f: TextIOWrapper):
        f.write("module Turn\n")

        if not self.transition_after_update:
            # current behavior: move -> URC -> update/skip
            f.write("  t : [0..2] init 0;\n")
            for a in self.actions:
                f.write(f"  [{a}] (t=0) -> (t'=1);\n")
            f.write("\n  [URC] (t=1) -> (t'=2);\n")
            f.write("\n  // Update phase\n")
            for s in self.services:
                f.write(f"  [update_{s}] (t=2) -> (t'=0);\n")
            f.write("  [skip_update] (t=2) -> (t'=0);\n")

        else:
            # new behavior: move -> update/skip -> URC (only if an update happened)
            # We achieve this by only reaching t=2 via update_*; skip_update jumps back to 0.
            f.write("  t : [0..2] init 0;\n")
            for a in self.actions:
                f.write(f"  [{a}] (t=0) -> (t'=1);\n")

            f.write("\n  // Update phase\n")
            for s in self.services:
                f.write(f"  [update_{s}] (t=1) -> (t'=2);\n")
            f.write("  [skip_update] (t=1) -> (t'=0);\n")

            f.write("\n  // URC only after an update\n")
            f.write("  [URC] (t=2) -> (t'=0);\n")

        f.write("endmodule\n")

    # ---------- popfile (FIXED COUNTS) ----------
    def _evolved_act_tr_count(self) -> int:
        # This class does NOT emit action-transition evolvables (ua_tr_act_*)
        return 0

    def _evolved_obs_tr_count(self) -> int:
        # next-state + outputs; outputs are per service
        per_obs = 1 + len(self.services)
        return self.internal_states * len(self.combinations) * per_obs

    def _evolvable_count(self) -> int:
        extra = 1 if self.range_split else 0  # ua_init_s
        return extra + len(self.services) + self._evolved_obs_tr_count()

    def _evolvable_ranges_in_order(self):
        ranges = []

        # must match add_ua() declaration order
        if self.range_split:
            ranges.append((1, self.internal_states))  # ua_init_s

        for _svc in self.services:
            ranges.append((self.min_val, self.max_val))  # ua_init_c_<svc>

        for s in range(1, self.internal_states + 1):
            for _combo in self.combinations:
                ranges.append((1, self.internal_states))  # ua_tr_obs_...
                lo, hi = self._out_range_for_state(s)
                for _svc in self.services:
                    ranges.append((lo, hi))               # ua_out_obs_<svc>_s...
        return ranges


    def create_pop_file(self, f: TextIOWrapper):
        ranges = self._evolvable_ranges_in_order()
        expected = len(ranges)

        # keep your existing "cap" idea; clamping makes it safe
        cap = min(self.max_val, self.internal_states)

        def clamp(v, lo, hi):
            return lo if v < lo else hi if v > hi else v

        for c in range(1, cap + 1):
            row = [str(clamp(c, lo, hi)) for (lo, hi) in ranges]
            assert len(row) == expected
            f.write(" ".join(row) + "\n")


    def _compute_state_ranges(self):
        """Disjoint contiguous ranges covering [min_val..max_val].
        State 1 gets the HIGHEST values (confident = rare GPS).
        """
        R = self.max_val - self.min_val + 1
        K = self.internal_states
        if R < K:
            raise ValueError(
                f"range_split=True requires at least {K} distinct values in "
                f"[{self.min_val}..{self.max_val}] (got {R})."
            )

        base, rem = divmod(R, K)
        ranges = [None] * (K + 1)  # 1-indexed
        hi = self.max_val
        for s in range(1, K + 1):
            size = base + (1 if s <= rem else 0)
            lo = hi - size + 1
            ranges[s] = (lo, hi)   # s=1 highest chunk
            hi = lo - 1
        return ranges

    def _out_range_for_state(self, s: int):
        if not self.range_split:
            return (self.min_val, self.max_val)
        return self._state_ranges[s]
