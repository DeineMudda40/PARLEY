from _io import TextIOWrapper
from itertools import product
import re


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


from _io import TextIOWrapper
from itertools import product
import re

# assumes these helpers are defined elsewhere:
# - get_services_from_prism_file(f)
# - get_hat_variables_with_ranges_and_init(f)


class ParleyFSCMealyCircle:
    """
    Deterministic Mealy-style FSC (NO distributions):

    - Internal controller state ua_s ∈ {1..S}.
    - For each (ua_s, observation) we choose a deterministic move in {left, stay, right}
      via an EvoChecker integer parameter:
          evolve int ua_move_s<s>_<obs> [1..3];

      bins: 1=left, 2=stay, 3=right

    - For each (service, ua_s, observation) we choose a deterministic threshold (action output)
      via an EvoChecker integer parameter:
          evolve int ua_thr_<svc>_s<s>_<obs> [min_val..max_val];

      During [UA_STEP_<svc>] we set:
          c_<svc>' = ua_thr_<svc>_s<s>_<obs>

    - For each service, an initial threshold is chosen once:
          evolve int ua_init_thr_<svc> [min_val..max_val];

    - Turn sequencing:

          t = 0: init phase
                 [UA_INIT_<svc>] -> t'=1  (set initial thresholds)

          t = 1: movement phase
                 [east]/[west]/[north]/[south] -> t'=2

          t = 2: controller phase
                 [UA_STEP_<svc>] -> t'=3  (set threshold for obs and update ua_s deterministically)

          t = 3: update/skip phase
                 [update_<svc>]  -> t'=1
                 [skip_update]   -> t'=1

    This yields a fully deterministic controller structure (given fixed evolved ints):
    - actions deterministic (thresholds are deterministic ints)
    - memory transitions deterministic (left/stay/right selected by deterministic ints)
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

        with open(infile, "r") as f:
            self.services = get_services_from_prism_file(f)
            self.features = get_hat_variables_with_ranges_and_init(f)

        if not self.services:
            raise ValueError("No services found (expected const int c_<service> = ...;)")
        if not self.features:
            raise ValueError("No *hat variables found (expected xhat/yhat declarations).")

        # stable order for naming & observation combinations
        self.feature_names = list(self.features.keys())

        # observation domains (cartesian product of hat-variable ranges)
        self.domains = []
        for name in self.feature_names:
            lo, hi = self.features[name]["range"]
            self.domains.append(range(lo, hi + 1))
        self.combinations = list(product(*self.domains))

    # ---------- naming helpers ----------

    def _obs_id(self, combo) -> str:
        # e.g. xhat,yhat → "x_3_y_5"
        parts = [f"{name}_{val}" for name, val in zip(self.feature_names, combo)]
        return "_".join(parts)

    def _ua_label(self, svc: str) -> str:
        return f"UA_STEP_{svc}"

    def _ua_init_label(self, svc: str) -> str:
        return f"UA_INIT_{svc}"

    def _init_thr(self, svc: str) -> str:
        # evolve int ua_init_thr_<svc> [min..max];
        return f"ua_init_thr_{svc}"

    def _thr_param(self, svc: str, s: int, combo) -> str:
        # evolve int ua_thr_<svc>_s<s>_<obs> [min..max];
        obs = self._obs_id(combo)
        return f"ua_thr_{svc}_s{s}_{obs}"

    def _move_param(self, s: int, combo) -> str:
        # evolve int ua_move_s<s>_<obs> [1..3];
        obs = self._obs_id(combo)
        return f"ua_move_s{s}_{obs}"

    # ---------- pipeline ----------

    def transform_file(self, infile: str, outfile: str, popfile: str):
        """
        - Removes original Turn module
        - Removes fixed const int c_<service> = ...;
        - Keeps the rest of the model
        - Appends UA (FSC) module and new Turn module
        - Writes a simple popfile with a small deterministic seed set (optional but handy)
        """
        TURN_START_RE = re.compile(r"^\s*module\s+Turn\b")
        ENDMODULE_RE = re.compile(r"^\s*endmodule\b")

        # 1) Copy base model without old Turn and const c_<svc>
        with open(infile, "r") as fin, open(outfile, "w") as fout:
            skipping_turn = False
            for line in fin:
                # drop original Turn module
                if TURN_START_RE.match(line):
                    skipping_turn = True
                    continue
                if skipping_turn:
                    if ENDMODULE_RE.match(line):
                        skipping_turn = False
                    continue

                # drop fixed counters: const int c_<service> = ...;
                if any(line.strip().startswith(f"const int c_{s}") for s in self.services):
                    continue

                fout.write(line)

        # 2) append UA (FSC) and Turn modules
        with open(outfile, "a") as f:
            self.add_ua(f)
            self.add_turn(f)

        # 3) popfile (optional, but you asked for an evochecker-ready script)
        with open(popfile, "w") as f:
            self.create_pop_file(f)

    # ---------- UA (FSC) module ----------

    def add_ua(self, f: TextIOWrapper):
        S = self.internal_states
        combos = self.combinations

        f.write(
            "\n// ===== UA FSC (deterministic Mealy controller via evolve int) =====\n"
            "// - ua_s is the internal controller state, 1..S\n"
            "// - ua_init_thr_<svc> picks initial thresholds\n"
            "// - ua_thr_<svc>_s<s>_<obs> picks deterministic threshold (action) per (svc,s,obs)\n"
            "// - ua_move_s<s>_<obs> picks deterministic move per (s,obs): 1=left, 2=stay, 3=right\n\n"
        )

        # --- declare evolve ints ---

        f.write("// Initial thresholds per service\n")
        for svc in self.services:
            f.write(f"evolve int {self._init_thr(svc)} [{self.min_val}..{self.max_val}];\n")
        f.write("\n")

        f.write("// Deterministic action-output (threshold) per (service, ua_s, observation)\n")
        for svc in self.services:
            for s in range(1, S + 1):
                for combo in combos:
                    p = self._thr_param(svc, s, combo)
                    f.write(f"evolve int {p} [{self.min_val}..{self.max_val}];\n")
        f.write("\n")

        f.write("// Deterministic memory update per (ua_s, observation): 1=left, 2=stay, 3=right\n")
        for s in range(1, S + 1):
            for combo in combos:
                m = self._move_param(s, combo)
                f.write(f"evolve int {m} [1..3];\n")
        f.write("\n")

        # --- UA module itself ---

        f.write("module UA\n")
        f.write(f"  ua_s : [1..{S}] init 1;\n")
        for svc in self.services:
            # overwritten during init + each controller step
            f.write(f"  c_{svc} : [{self.min_val}..{self.max_val}] init {self.max_val};\n")
        f.write("\n")

        # initial observation combo for hat variables
        init_combo = tuple(self.features[name]["init"] for name in self.feature_names)
        init_guard = " & ".join(
            f"{name}hat={val}" for name, val in zip(self.feature_names, init_combo)
        )

        # Init phase: set initial thresholds deterministically from evolve ints
        f.write("  // Init phase: set initial thresholds (deterministic)\n")
        for svc in self.services:
            lab_init = self._ua_init_label(svc)
            init_p = self._init_thr(svc)
            f.write(
                f"  [{lab_init}] (ua_s=1) & {init_guard} -> (c_{svc}'={init_p});\n"
            )
        f.write("\n")

        # Controller step: set action threshold + update memory deterministically based on (s, obs)
        f.write("  // Controller step: deterministic Mealy action + deterministic memory update\n")
        for svc in self.services:
            lab = self._ua_label(svc)
            for s in range(1, S + 1):
                left_s = s - 1 if s > 1 else S
                stay_s = s
                right_s = s + 1 if s < S else 1

                for combo in combos:
                    guard_obs = " & ".join(
                        f"{name}hat={val}" for name, val in zip(self.feature_names, combo)
                    )
                    thr = self._thr_param(svc, s, combo)
                    mv = self._move_param(s, combo)

                    # three deterministic branches guarded by mv (still deterministic because mv is fixed)
                    f.write(
                        f"  [{lab}] (ua_s={s}) & {guard_obs} & ({mv}=1) -> "
                        f"(c_{svc}'={thr}) & (ua_s'={left_s});\n"
                    )
                    f.write(
                        f"  [{lab}] (ua_s={s}) & {guard_obs} & ({mv}=2) -> "
                        f"(c_{svc}'={thr}) & (ua_s'={stay_s});\n"
                    )
                    f.write(
                        f"  [{lab}] (ua_s={s}) & {guard_obs} & ({mv}=3) -> "
                        f"(c_{svc}'={thr}) & (ua_s'={right_s});\n"
                    )
            f.write("\n")

        f.write("endmodule\n\n")

    # ---------- Turn module ----------

    def add_turn(self, f: TextIOWrapper):
        """
        Sequencing:

          t = 0: init phase
              [UA_INIT_<svc>]  -> t'=1

          t = 1: movement phase
              [east]/[west]/[north]/[south] -> t'=2

          t = 2: controller phase (FSC)
              [UA_STEP_<svc>] -> t'=3

          t = 3: update/skip phase
              [update_<svc>]  -> t'=1
              [skip_update]   -> t'=1
        """
        f.write("module Turn\n")
        f.write("  t : [0..3] init 0;\n\n")

        f.write("  // Init phase\n")
        for svc in self.services:
            f.write(f"  [{self._ua_init_label(svc)}] (t=0) -> (t'=1);\n")
        f.write("\n")

        f.write("  // Movement phase\n")
        for a in self.actions:
            f.write(f"  [{a}] (t=1) -> (t'=2);\n")
        f.write("\n")

        f.write("  // Controller phase\n")
        for svc in self.services:
            f.write(f"  [{self._ua_label(svc)}] (t=2) -> (t'=3);\n")
        f.write("\n")

        f.write("  // Update / skip phase\n")
        for svc in self.services:
            f.write(f"  [update_{svc}] (t=3) -> (t'=1);\n")
        f.write("  [skip_update] (t=3) -> (t'=1);\n\n")
        f.write("endmodule\n")

    # ---------- popfile ----------

    def create_pop_file(self, f: TextIOWrapper):
        """
        Popfile for EvoChecker evolve-int parameters.

        We output a small default population:
          - One individual per threshold k in [min_val..max_val]
          - All ua_init_thr_<svc> = k
          - All ua_thr_<svc>_s<*>_<obs> = k
          - All ua_move_s<*>_<obs> = 2  (stay)

        This mirrors your previous "same threshold everywhere" seeding, but now with evolve ints.
        You can (and probably should) expand this seeding later (e.g., vary moves).
        """
        S = self.internal_states
        combos = self.combinations

        # Build header in the exact order we generate values
        header = []

        # init thresholds
        for svc in self.services:
            header.append(self._init_thr(svc))

        # per-(svc,s,obs) thresholds
        for svc in self.services:
            for s in range(1, S + 1):
                for combo in combos:
                    header.append(self._thr_param(svc, s, combo))

        # per-(s,obs) moves
        for s in range(1, S + 1):
            for combo in combos:
                header.append(self._move_param(s, combo))

        f.write("\t".join(header) + "\n")

        # Body: one row per threshold seed
        for k in range(self.min_val, self.max_val + 1):
            row_vals = []

            # init thresholds
            for _svc in self.services:
                row_vals.append(str(k))

            # all ua_thr parameters
            for _svc in self.services:
                for _s in range(1, S + 1):
                    for _combo in combos:
                        row_vals.append(str(k))

            # all move parameters (default stay=2)
            for _s in range(1, S + 1):
                for _combo in combos:
                    row_vals.append("2")

            f.write("\t".join(row_vals) + "\n")
