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

    # ---------- popfile ----------

    def create_pop_file(self, f: TextIOWrapper):
        # 1) Header: names of all evolvable variables in declaration order
        header_names = []
        for service in self.services:
            for combo in self.combinations:
                header_names.append(self._decision_var(service, combo))

        # write header line
        f.write("\t".join(header_names) + "\n")

        # 2) Rows: one row per c, assigning that c to all evolvables
        num_vars = len(header_names)
        for c in range(self.min_val, self.max_val + 1):
            row = " ".join(str(c) for _ in range(num_vars))
            f.write(row + "\n")


from _io import TextIOWrapper
from itertools import product
import re


class ParleyPlusURCDist:
    def __init__(
        self,
        infile,
        min_val=1,
        max_val=10,
        actions=("east", "west", "north", "south"),
        transition_after_update=False,
    ):
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.actions = list(actions)
        self.transition_after_update = bool(transition_after_update)

        with open(infile, "r") as f:
            self.services = get_services_from_prism_file(f)
            self.features = get_hat_variables_with_ranges_and_init(f)

        if not self.services:
            raise ValueError(
                "No services found (expected const int c_<service> = ...;)"
            )
        if not self.features:
            raise ValueError(
                "No *hat variables found (expected xhat/yhat declarations)."
            )

        # stable order for decision naming and combo generation
        self.feature_names = list(self.features.keys())

        self.domains = []
        for name in self.feature_names:
            lo, hi = self.features[name]["range"]
            self.domains.append(range(lo, hi + 1))

        self.combinations = list(product(*self.domains))

        # number of buckets for each threshold distribution
        self.num_buckets = self.max_val - self.min_val + 1

    # ---------- naming helpers ----------

    def _decision_dist(self, service: str, combo) -> str:
        """Base name of the evolve distribution for this service & observation.

        The actual parameter names are base + index, e.g. ...ph1, ...ph2, ...
        """
        parts = [f"{name}_{value}" for name, value in zip(self.feature_names, combo)]
        return f"{service}_decision_" + "_".join(parts) + "ph"

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

        # 4) Population file – left empty so EvoChecker can random-initialise distributions
        with open(popfile, "w") as f:
            self.create_pop_file(f)

    # ---------- codegen (distribution-based URC) ----------

    def add_urc(self, f: TextIOWrapper):
        names = self.feature_names

        # 1) Declare evolve distributions instead of int parameters
        #    evolve distribution <service>_decision_<obs> [num_buckets];
        for service in self.services:
            for combo in self.combinations:
                dist_name = self._decision_dist(service, combo)
                f.write(f"evolve distribution {dist_name} [{self.num_buckets}];\n")

        f.write("\nmodule URC\n")
        # 2) c_<service> variables now just plain counters with some default init
        #    (first sampled value will come from the first URC step anyway)
        for service in self.services:
            f.write(
                f"  c_{service} : [{self.min_val}..{self.max_val}] init {self.max_val};\n"
            )

        f.write(
            "\n  // URC transitions: sample c_<service> from its distribution per observation\n"
        )
        # 3) For each observation, we create one command:
        #    [URC] guard -> dist : (c_s'=min) + dist : (c_s'=min+1) + ... ;
        #
        #    NOTE: to stay within EvoChecker’s syntax, we use exactly ONE distribution
        #    as the probability expression in each command. That means:
        #      - This works perfectly for a single service (typical gps case).
        #      - For multiple services, all c_<service> share the same sampled value.
        #        (If you want truly independent services, we need the multi-module
        #         synchronisation trick from the Mealy version.)
        for combo in self.combinations:
            guard = " & ".join(
                f"{name}hat={value}" for name, value in zip(names, combo)
            )

            for service in self.services:
                base = self._decision_dist(service, combo)
                branches = []
                # bins 1..num_buckets → values min_val..max_val
                for i, val in enumerate(range(self.min_val, self.max_val + 1), start=1):
                    branches.append(f"{base}{i} : (c_{service}'={val})")

                f.write(f"  [URC] {guard} -> " + " + ".join(branches) + ";\n")

        f.write("endmodule\n\n")

    def add_turn(self, f: TextIOWrapper):
        if not self.transition_after_update:
            # --- original behaviour: move -> URC -> update/skip ---
            f.write("module Turn\n")
            f.write("  t : [0..2] init 0;\n")
            for a in self.actions:
                f.write(f"  [{a}] (t=0) -> (t'=1);\n")

            f.write("\n  [URC] (t=1) -> (t'=2);\n\n")

            for s in self.services:
                f.write(f"  [update_{s}] (t=2) -> (t'=0);\n")
            f.write("  [skip_update] (t=2) -> (t'=0);\n")
        else:
            # --- variant: move -> update/skip, URC only after update ---
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
        """
        EvoChecker popfile for evolve distribution:
        Headers look like:
            gps_decision_x_0_y_01   gps_decision_x_0_y_02  ... gps_decision_x_0_y_0N

        i.e. the bin index is appended directly with no braces.
        """

        n = self.num_buckets  # number of bins per distribution

        # ---------- HEADER ----------
        header = []
        for service in self.services:
            for combo in self.combinations:
                dist_name = self._decision_dist(service, combo)
                # naive concatenation: dist_name + str(i)
                for i in range(1, n + 1):
                    header.append(f"{dist_name}{i}")

        f.write("\t".join(header) + "\n")

        # ---------- BODY ----------
        # n individuals: each is deterministic on a single bucket
        for k_idx in range(n):
            row_vals = []
            for _service in self.services:
                for _combo in self.combinations:
                    for b in range(n):
                        val = "1.0" if b == k_idx else "0.0"
                        row_vals.append(val)
            f.write(" ".join(row_vals) + "\n")


from _io import TextIOWrapper
from itertools import product
import re


class ParleyFSC:
    """
    EvoChecker FSC generator using 'evolve distribution dist [n];' and repeated 'dist' usage
    (as in the official FOREX example).

    - Threshold emission: per (service, ua_s) a 10-bin distribution over thresholds [min_val..max_val]
    - State transitions: per (ua_s, observation combo) a |S|-bin distribution over next ua_s'
    - Transition happens ONLY after an update (skip_update does not change ua_s)
    - Turn sequencing:
        t=0 move -> t=1 update/skip
        if update: UA_OUT_<svc> for each service, then UA_TR, then back to 0
        if skip: back to 0
    """

    def __init__(
        self,
        infile: str,
        min_val: int = 1,
        max_val: int = 10,
        actions=("east", "west", "north", "south"),
        internal_states: int = 10,
    ):
        self.infile = infile
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.actions = list(actions)
        self.internal_states = int(internal_states)

        # exactly 10 buckets expected
        self.bucket_values = list(range(self.min_val, self.max_val + 1))
        if len(self.bucket_values) != 10:
            raise ValueError(
                f"Expected exactly 10 threshold buckets, but got {len(self.bucket_values)} "
                f"from range [{self.min_val}..{self.max_val}]."
            )

        with open(infile, "r") as f:
            self.services = get_services_from_prism_file(f)
            self.features = get_hat_variables_with_ranges_and_init(f)

        if not self.services:
            raise ValueError(
                "No services found (expected const int c_<service> = ...;)"
            )
        if not self.features:
            raise ValueError(
                "No *hat variables found (expected xhat/yhat declarations)."
            )

        self.feature_names = list(self.features.keys())

        # observation domains (cartesian product of hat-variable ranges)
        self.domains = []
        for name in self.feature_names:
            lo, hi = self.features[name]["range"]
            self.domains.append(range(lo, hi + 1))
        self.combinations = list(product(*self.domains))

    # ---------------- naming helpers ----------------

    def _obs_id(self, combo) -> str:
        parts = [f"{name}_{val}" for name, val in zip(self.feature_names, combo)]
        return "_".join(parts)

    def _thr_dist(self, svc: str, s: int) -> str:
        return f"ua_thr_dist_{svc}_s{s}"

    def _tr_dist(self, s: int, obs: str) -> str:
        return f"ua_tr_dist_s{s}_{obs}"

    def _ua_out_label(self, svc: str) -> str:
        return f"UA_OUT_{svc}"

    def _ua_tr_label(self) -> str:
        return "UA_TR"

    # ---------------- pipeline ----------------

    def transform_file(self, infile: str, outfile: str):
        TURN_START_RE = re.compile(r"^\s*module\s+Turn\b")
        ENDMODULE_RE = re.compile(r"^\s*endmodule\b")

        with open(infile, "r") as fin, open(outfile, "w") as fout:
            skipping_turn = False

            for line in fin:
                # Remove original Turn module
                if TURN_START_RE.match(line):
                    skipping_turn = True
                    continue
                if skipping_turn:
                    if ENDMODULE_RE.match(line):
                        skipping_turn = False
                    continue

                # Remove fixed counters: const int c_<service> = ...;
                if any(
                    line.strip().startswith(f"const int c_{s}") for s in self.services
                ):
                    continue

                fout.write(line)

        with open(outfile, "a") as f:
            self.add_ua(f)
            self.add_turn(f)

    # ---------------- codegen: UA ----------------

    def add_ua(self, f: TextIOWrapper):
        S = self.internal_states
        combos = self.combinations

        f.write(
            "\n// ===== UA FSC (EvoChecker distributions) =====\n"
            "// Distributions are used by repeating the distribution name once per branch,\n"
            "// e.g., '-> dist : ... + dist : ... + dist : ...' (no indexing like dist(1)).\n\n"
        )

        # Threshold distributions: 10 bins per (service, state)
        f.write("// Threshold emission distributions (10 bins) per (service, ua_s)\n")
        for svc in self.services:
            for s in range(1, S + 1):
                f.write(f"evolve distribution {self._thr_dist(svc, s)} [10];\n")
        f.write("\n")

        # Transition distributions: S bins per (state, observation)
        f.write("// State-transition distributions (S bins) per (ua_s, observation)\n")
        for s in range(1, S + 1):
            for combo in combos:
                obs = self._obs_id(combo)
                f.write(f"evolve distribution {self._tr_dist(s, obs)} [{S}];\n")
        f.write("\n")
        f.write("evolve distribution ua_init_dist_<svc> [10];\n")

        # UA module
        f.write("module UA\n")
        f.write(f"  ua_s : [1..{S}] init 1;\n")
        for svc in self.services:
            f.write(
                f"  c_{svc} : [{self.min_val}..{self.max_val}] init {self.max_val};\n"
            )
        f.write("\n")

        # Threshold emission commands (only depends on ua_s)
        f.write("  // Threshold emission: sample c_<svc> based only on ua_s\n")
        for svc in self.services:
            lab = self._ua_out_label(svc)
            for s in range(1, S + 1):
                dist = self._thr_dist(svc, s)
                # Repeat 'dist' 10 times (bin1..bin10) – EvoChecker expands sequentially.
                branches = [f"{dist} : (c_{svc}'={val})" for val in self.bucket_values]
                f.write(f"  [{lab}] (ua_s={s}) -> " + " + ".join(branches) + ";\n")
            f.write("\n")

        # State transition commands (depends on ua_s and observation), only executed after update due to Turn
        f.write(
            "  // FSC state transition: after update, update ua_s based on (ua_s, observed hat vars)\n"
        )
        lab_tr = self._ua_tr_label()
        for s in range(1, S + 1):
            for combo in combos:
                guard_obs = " & ".join(
                    f"{name}hat={val}" for name, val in zip(self.feature_names, combo)
                )
                obs = self._obs_id(combo)
                dist = self._tr_dist(s, obs)
                # Repeat 'dist' exactly S times to map to next-states 1..S
                branches = [f"{dist} : (ua_s'={sp})" for sp in range(1, S + 1)]
                f.write(
                    f"  [{lab_tr}] (ua_s={s}) & {guard_obs} -> "
                    + " + ".join(branches)
                    + ";\n"
                )

        f.write("\nendmodule\n\n")

    # ---------------- Turn module ----------------

    def add_turn(self, f: TextIOWrapper):
        """
        Sequencing:
          t=0: move
          t=1: update/skip
          if update: UA_OUT_<svc> for each service, then UA_TR, then back to 0
          if skip: back to 0 (no UA_OUT, no UA_TR)
        """
        K = len(self.services)
        last = 2 + K  # UA_TR runs at t=last

        f.write("module Turn\n")
        f.write(f"  t : [0..{last}] init 0;\n")

        # Move phase
        for a in self.actions:
            f.write(f"  [{a}] (t=0) -> (t'=1);\n")

        f.write("\n  // Update phase\n")
        for svc in self.services:
            f.write(f"  [update_{svc}] (t=1) -> (t'=2);\n")
        f.write("  [skip_update] (t=1) -> (t'=0);\n")

        # UA_OUT phases
        f.write("\n  // UA threshold emission phases (only after an update)\n")
        for i, svc in enumerate(self.services):
            cur = 2 + i
            nxt = 3 + i
            f.write(f"  [{self._ua_out_label(svc)}] (t={cur}) -> (t'={nxt});\n")

        # UA_TR
        f.write("\n  // UA transition phase (only after an update)\n")
        f.write(f"  [{self._ua_tr_label()}] (t={last}) -> (t'=0);\n")

        f.write("endmodule\n")


from _io import TextIOWrapper
from itertools import product
import re


class ParleyFSCMealy:
    def __init__(
        self,
        infile: str,
        min_val: int = 1,
        max_val: int = 10,
        actions=("east", "west", "north", "south"),
        internal_states: int = 10,
    ):
        self.infile = infile
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.actions = list(actions)
        self.internal_states = int(internal_states)

        # exactly 10 buckets expected
        self.bucket_values = list(range(self.min_val, self.max_val + 1))
        if len(self.bucket_values) != 10:
            raise ValueError(
                f"Expected exactly 10 threshold buckets, but got {len(self.bucket_values)} "
                f"from range [{self.min_val}..{self.max_val}]."
            )

        with open(infile, "r") as f:
            self.services = get_services_from_prism_file(f)
            self.features = get_hat_variables_with_ranges_and_init(f)

        if not self.services:
            raise ValueError(
                "No services found (expected const int c_<service> = ...;)"
            )
        if not self.features:
            raise ValueError(
                "No *hat variables found (expected xhat/yhat declarations)."
            )

        self.feature_names = list(self.features.keys())

        # observation domains (cartesian product of hat-variable ranges)
        self.domains = []
        for name in self.feature_names:
            lo, hi = self.features[name]["range"]
            self.domains.append(range(lo, hi + 1))
        self.combinations = list(product(*self.domains))

    # ---------- naming helpers ----------

    def _obs_id(self, combo) -> str:
        parts = [f"{name}_{val}" for name, val in zip(self.feature_names, combo)]
        return "_".join(parts)

    # transition dist (ua_s, obs) -> ua_s'
    def _tr_dist(self, s: int, obs: str) -> str:
        return f"ua_tr_dist_s{s}_{obs}"

    # action dist (svc, ua_s, obs) -> threshold bucket
    def _act_dist(self, svc: str, s: int, obs: str) -> str:
        return f"ua_{svc}_act_dist_s{s}_{obs}"

    # initial threshold dist per service
    def _init_dist(self, svc: str) -> str:
        return f"ua_init_dist_{svc}"

    # labels
    def _urc_label(self, svc: str) -> str:
        # FSC update step after update_<svc>
        return f"UA_TR_{svc}"

    def _init_label(self, svc: str) -> str:
        # one-shot initialisation for c_<svc>
        return f"UA_INIT_{svc}"

    # ---------- pipeline ----------

    def transform_file(self, infile: str, outfile: str):
        TURN_START_RE = re.compile(r"^\s*module\s+Turn\b")
        ENDMODULE_RE = re.compile(r"^\s*endmodule\b")

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

                # remove const int c_<service>
                if any(
                    line.strip().startswith(f"const int c_{s}") for s in self.services
                ):
                    continue

                fout.write(line)

        with open(outfile, "a") as f:
            self.add_modules(f)
            self.add_turn(f)

    # ---------- codegen: FSC modules ----------

    def add_modules(self, f: TextIOWrapper):
        S = self.internal_states
        combos = self.combinations

        f.write(
            "\n// ===== FSC with two independent evolve distributions (via synchronisation) =====\n"
            "// We use:\n"
            "//  * ua_tr_dist_s<state>_<obs>   for next-state selection\n"
            "//  * ua_<svc>_act_dist_s<state>_<obs> for threshold selection\n"
            "//  * ua_init_dist_<svc>          for initial threshold of each service\n\n"
        )

        # --- declare distributions ---

        # Initial threshold distributions (per service, 10 buckets)
        f.write(
            "// Initial threshold distributions: per service -> c_<svc> in [min_val..max_val]\n"
        )
        for svc in self.services:
            f.write(f"evolve distribution {self._init_dist(svc)} [10];\n")
        f.write("\n")

        # Transition distributions: S bins per (state, observation)
        f.write("// State-transition distributions (S bins) per (ua_s, observation)\n")
        for s in range(1, S + 1):
            for combo in combos:
                obs = self._obs_id(combo)
                f.write(f"evolve distribution {self._tr_dist(s, obs)} [{S}];\n")
        f.write("\n")

        # Action/threshold distributions: 10 bins per (svc, state, observation)
        f.write(
            "// Action/threshold distributions (10 bins) per (svc, ua_s, observation)\n"
        )
        for svc in self.services:
            for s in range(1, S + 1):
                for combo in combos:
                    obs = self._obs_id(combo)
                    f.write(
                        f"evolve distribution {self._act_dist(svc, s, obs)} [10];\n"
                    )
            f.write("\n")

        # --- UA_TR module: owns ua_s (FSC state) ---
        f.write("module UA_TR\n")
        f.write(f"  ua_s : [1..{S}] init 1;\n\n")

        # FSC state transition: synchronised with Turn + UA_ACT through [UA_TR_<svc>]
        for svc in self.services:
            lab = self._urc_label(svc)
            for s in range(1, S + 1):
                for combo in combos:
                    guard_obs = " & ".join(
                        f"{name}hat={val}"
                        for name, val in zip(self.feature_names, combo)
                    )
                    obs = self._obs_id(combo)
                    dist = self._tr_dist(s, obs)
                    branches = [f"{dist} : (ua_s'={sp})" for sp in range(1, S + 1)]
                    f.write(
                        f"  [{lab}] (ua_s={s}) & {guard_obs} -> "
                        + " + ".join(branches)
                        + ";\n"
                    )
            f.write("\n")
        f.write("endmodule\n\n")

        # --- UA_ACT module: owns all c_<svc> and handles initial & update thresholds ---
        f.write("module UA_ACT\n")
        for svc in self.services:
            # init value is arbitrary within range; it will be overwritten in the init phase
            f.write(
                f"  c_{svc} : [{self.min_val}..{self.max_val}] init {self.max_val};\n"
            )
        f.write("\n")

        # Initial sampling of c_<svc>, one service per t-state, via [UA_INIT_<svc>]
        f.write(
            "  // Initial sampling of thresholds c_<svc> (one-shot at the beginning)\n"
        )
        for i, svc in enumerate(self.services):
            label = self._init_label(svc)
            dist = self._init_dist(svc)
            branches = [f"{dist} : (c_{svc}'={val})" for val in self.bucket_values]
            # Guard on t so only the matching Turn phase can trigger this
            f.write(f"  [{label}] (t={i}) -> " + " + ".join(branches) + ";\n")
        f.write("\n")

        # Threshold sampling after an update (depends on ua_s and observation)
        f.write("  // Threshold sampling after an update: per (svc, ua_s, obs)\n")
        for svc in self.services:
            lab = self._urc_label(svc)
            for s in range(1, S + 1):
                for combo in combos:
                    guard_obs = " & ".join(
                        f"{name}hat={val}"
                        for name, val in zip(self.feature_names, combo)
                    )
                    obs = self._obs_id(combo)
                    dist = self._act_dist(svc, s, obs)
                    branches = [
                        f"{dist} : (c_{svc}'={val})" for val in self.bucket_values
                    ]
                    f.write(
                        f"  [{lab}] (ua_s={s}) & {guard_obs} -> "
                        + " + ".join(branches)
                        + ";\n"
                    )
            f.write("\n")

        f.write("endmodule\n\n")

    # ---------- Turn module ----------

    def add_turn(self, f: TextIOWrapper):
        """
        Sequencing (K = #services):
          t in [0..K-1]: initialisation phases, one per service
              [UA_INIT_<svc_i>]: sample initial c_<svc_i>, then go to next init phase or move phase
          t = K       : movement phase
              [action] -> t=K+1
          t = K+1     : update/skip phase
              [update_<svc_i>] -> t = K+2+i
              [skip_update]    -> t = K
          t = K+2..K+1+K: FSC phase for each service
              [UA_TR_<svc_i>] -> t = K
        """
        K = len(self.services)

        init_start = 0
        init_end = K - 1  # inclusive
        move_state = K
        update_state = K + 1
        urc_start = K + 2  # K states: K+2 .. K+1+K
        max_t = K + 1 + K  # last URC state index

        f.write("module Turn\n")
        f.write(f"  t : [0..{max_t}] init 0;\n\n")

        # Initialisation phases: sample each c_<svc> once
        f.write(
            "  // Initialisation phases: sample c_<svc> once from ua_init_dist_<svc>\n"
        )
        for i, svc in enumerate(self.services):
            label = self._init_label(svc)
            cur = init_start + i
            if i < K - 1:
                nxt = cur + 1
            else:
                nxt = move_state  # after last init, start movement
            f.write(f"  [{label}] (t={cur}) -> (t'={nxt});\n")
        f.write("\n")

        # Movement phase
        f.write("  // Movement phase\n")
        for a in self.actions:
            f.write(f"  [{a}] (t={move_state}) -> (t'={update_state});\n")
        f.write("\n")

        # Update / skip phase
        f.write("  // Update / skip phase\n")
        for i, svc in enumerate(self.services):
            urc_state = urc_start + i
            f.write(f"  [update_{svc}] (t={update_state}) -> (t'={urc_state});\n")
        f.write(f"  [skip_update] (t={update_state}) -> (t'={move_state});\n\n")

        # FSC phase after each update_<svc>: synchronises UA_TR + UA_ACT
        f.write("  // FSC phase after each update_<svc>\n")
        for i, svc in enumerate(self.services):
            urc_state = urc_start + i
            f.write(
                f"  [{self._urc_label(svc)}] (t={urc_state}) -> (t'={move_state});\n"
            )

        f.write("endmodule\n")
