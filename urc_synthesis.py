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


import re
from _io import TextIOWrapper


def get_hat_variables_with_ranges_and_init(f: TextIOWrapper):
    # regex to extract constants like: const int N = 9;
    CONST_RE = re.compile(
        r"^\s*const\s+(?:int|double)\s+([A-Za-z_]\w*)\s*=\s*([0-9.]+)\s*;"
    )

    # regex to extract hat declarations:
    #    xhat : [0..N] init xstart;
    HAT_DECL_RE = re.compile(
        r"^\s*([A-Za-z_]\w*)hat\s*:\s*"
        r"\[\s*([0-9A-Za-z_]+)\s*\.\.\s*([0-9A-Za-z_]+)\s*\]\s*"
        r"init\s+([A-Za-z_]\w*)"
    )

    pos = f.tell()
    try:
        constants = {}
        hat_vars = {}

        # read file into memory once
        lines = f.readlines()

        # Parse constants
        for line in lines:
            m = CONST_RE.match(line)
            if m:
                name, value = m.groups()
                if "." in value:
                    constants[name] = float(value)
                else:
                    constants[name] = int(value)

        # Parse hat variable declarations
        for line in lines:
            m = HAT_DECL_RE.match(line)
            if m:
                base, lo_raw, hi_raw, init_raw = m.groups()

                # resolve lower bound
                lo = int(lo_raw) if lo_raw.isdigit() else constants.get(lo_raw)

                # resolve upper bound
                hi = int(hi_raw) if hi_raw.isdigit() else constants.get(hi_raw)

                # resolve initial value
                init = int(init_raw) if init_raw.isdigit() else constants.get(init_raw)

                if lo is None or hi is None:
                    raise ValueError(
                        f"Cannot resolve range for {base}hat: [{lo_raw}..{hi_raw}]"
                    )

                if init is None:
                    raise ValueError(
                        f"Cannot resolve init value for {base}hat: init {init_raw}"
                    )

                hat_vars[base] = {
                    "range": (lo, hi),
                    "init": init,
                }

        return hat_vars

    finally:
        f.seek(pos)


class ParleyPlusURC:
    def __init__(
        self, infile, min_val=1, max_val=10, actions=["east", "west", "north", "south"]
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.actions = actions

        with open(infile, "r") as f:
            self.services = get_services_from_prism_file(f)
            self.features = get_hat_variables_with_ranges_and_init(f)

        self.domains = []
        for info in self.features.values():
            lo, hi = info["range"]
            self.domains.append(range(lo, hi + 1))

        # Precompute all combinations only once:
        self.combinations = list(product(*self.domains))

    def transform_file(self, infile, outfile, popfile):
        with open(infile, "r") as fin, open(outfile, "w") as fout:
            for line in fin:
                # Remove: const int c_gps = ...;
                if any(
                    line.strip().startswith(f"const int c_{s}") for s in self.services
                ):
                    continue
                fout.write(line)

        with open(outfile, "a") as f:
            self.add_urc(f)
            self.add_turn(f)

        with open(popfile, "w") as f:
            self.create_pop_file(f)

    def add_urc(self, f: TextIOWrapper):
        names = list(self.features.keys())

        # ------ Write evolvable decisions ------
        for service in self.services:
            for combo in product(*self.domains):

                parts = [f"{name}_{value}" for name, value in zip(names, combo)]
                label = "decision_" + "_".join(parts)

                f.write(
                    f"evolve int {service}_{label} [{self.min_val}..{self.max_val}];\n"
                )

        # ------ Compute init label ------
        init_parts = [f"{name}_{info['init']}" for name, info in self.features.items()]
        init_label = "decision_" + "_".join(init_parts)

        # ------ Write URC module ------
        f.write("\nmodule URC\n")
        for service in self.services:
            f.write(
                f"  c_{service} : [{self.min_val}..{self.max_val}] init {service}_{init_label};\n"
            )

        f.write("\n  // URC transitions\n")

        # ------ Write URC updates for each hat combination ------
        for combo in product(*self.domains):
            # build guard: xhat=3 & yhat=7
            guard = " & ".join(
                f"{name}hat={value}" for name, value in zip(names, combo)
            )

            # build RHS updates:
            # (c_gps'=decision_gps_x_3_y_7) & (c_cam'=decision_cam_x_3_y_7)
            updates = " & ".join(
                f"(c_{service}'={service}_decision_"
                + "_".join(f"{name}_{value}" for name, value in zip(names, combo))
                + ")"
                for service in self.services
            )

            f.write(f"  [URC] {guard} -> {updates};\n")

        f.write("endmodule\n\n")

    def add_turn(self, f: TextIOWrapper):
        f.write("module Turn\n")
        f.write("  t : [0..2] init 0;\n")
        for a in self.actions:
            f.write(f"  [{a}] (t=0) -> (t'=1);\n")

        f.write("\n  [URC] (t=1) -> (t'=2);\n\n")

        # IMPORTANT: must match Knowledge labels: [update_<service>]
        for s in self.services:
            f.write(f"  [update_{s}] (t=2) -> (t'=0);\n")

        f.write("  [skip_update] (t=2) -> (t'=0);\n")
        f.write("endmodule\n")

    def create_pop_file(self, f):
        num_vars = len(self.services) * len(self.combinations)
        for c in range(self.min_val, self.max_val + 1):  # include max
            f.write((" ".join([str(c)] * num_vars)) + "\n")


class ParleyUA:
    """
    UA transform with separate action/observation transitions:

    - Internal mode: ua_s in [1..internal_states]
    - Each mode represents counter values c_<service> (fixed/evolved depending on 'setting')
    - Action transition: on each action label [east]/[west]/..., update ua_snext as f(ua_s, action)
    - Observation transition: ONLY after an update, update ua_snext as g(ua_s, xhat, yhat, ...)
    - Apply step [UA_APPLY] sets ua_s := ua_snext and assigns c_<service> based on ua_s
    """

    def __init__(
        self,
        infile: str,
        min_val: int = 1,
        max_val: int = 10,
        actions=("east", "west", "north", "south"),
        internal_states: int = 10,
        setting: str = "fixed",  # "fixed" or "evolve"
        trigger_on_action: bool = True,  # enable action transitions (s,action)->s'
        obs_only_after_update: bool = True,  # if True: obs transition only after update_* (not after skip_update)
    ):
        if setting not in ("fixed", "evolve"):
            raise ValueError("setting must be 'fixed' or 'evolve'")

        self.infile = infile
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.actions = list(actions)
        self.internal_states = int(internal_states)
        self.setting = setting
        self.trigger_on_action = bool(trigger_on_action)
        self.obs_only_after_update = bool(obs_only_after_update)

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

        # fixed mapping count
        self.fixed_count = self.max_val - self.min_val + 1

    def transform_file(self, infile: str, outfile: str, popfile: str):
        # Copy while removing fixed const counters: const int c_<service> = ...;
        with open(infile, "r") as fin, open(outfile, "w") as fout:
            for line in fin:
                if any(
                    line.strip().startswith(f"const int c_{s}") for s in self.services
                ):
                    continue
                fout.write(line)

        with open(outfile, "a") as f:
            self.add_ua(f)
            self.add_turn(f)

        with open(popfile, "w") as f:
            self.create_pop_file(f, num_lines=100)

    # ------------ naming helpers ------------

    def _state_c_symbol(self, service: str, state: int) -> str:
        return f"ua_c_{service}_s{state}"

    def _tr_act_symbol(self, action: str, s: int) -> str:
        # ua_tr_act_east_s3
        return f"ua_tr_act_{action}_s{s}"

    def _tr_obs_symbol(self, s: int, combo) -> str:
        # ua_tr_obs_s3_x_2_y_7 ...
        parts = [f"{name}_{val}" for name, val in zip(self.feature_names, combo)]
        return f"ua_tr_obs_s{s}_" + "_".join(parts)

    # ------------ codegen ------------

    def add_ua(self, f: TextIOWrapper):
        f.write("\n// ===== ParleyUA declarations =====\n")
        f.write(
            f"evolve int ua_init_state [1..{min(self.internal_states,self.max_val)}];\n\n"
        )

        # Per-state counter values for each service
        for service in self.services:
            for s in range(1, self.internal_states + 1):
                is_fixed = self.setting == "fixed" and s <= min(
                    self.internal_states, self.fixed_count
                )
                if is_fixed:
                    c_val = self.min_val + (s - 1)
                    f.write(
                        f"const int {self._state_c_symbol(service, s)} = {c_val};\n"
                    )
                else:
                    f.write(
                        f"evolve int {self._state_c_symbol(service, s)} "
                        f"[{self.min_val}..{self.max_val}];\n"
                    )
        f.write("\n")

        # Action transitions: (s, action) -> s'
        if self.trigger_on_action:
            for a in self.actions:
                for s in range(1, self.internal_states + 1):
                    f.write(
                        f"evolve int {self._tr_act_symbol(a, s)} [1..{self.internal_states}];\n"
                    )
            f.write("\n")

        # Observation transitions: (s, xhat, yhat, ...) -> s'  (only used after update)
        for s in range(1, self.internal_states + 1):
            for combo in self.combinations:
                f.write(
                    f"evolve int {self._tr_obs_symbol(s, combo)} [1..{self.internal_states}];\n"
                )
        f.write("\n")

        # UA module
        f.write("module UA\n")
        f.write(f"  ua_s : [1..{self.internal_states}] init ua_init_state;\n")
        f.write(f"  ua_snext : [1..{self.internal_states}] init ua_init_state;\n")
        for service in self.services:
            # init arbitrary; Turn starts by applying correct values immediately
            f.write(
                f"  c_{service} : [{self.min_val}..{self.max_val}] init {self.min_val};\n"
            )
        f.write("\n")

        # Action-driven next-state update: happens on the action labels, depends only on ua_s and the action
        if self.trigger_on_action:
            f.write("  // Action transitions: (ua_s, action) -> ua_snext\n")
            for a in self.actions:
                for s in range(1, self.internal_states + 1):
                    f.write(
                        f"  [{a}] (ua_s={s}) -> (ua_snext'={self._tr_act_symbol(a, s)});\n"
                    )
            f.write("\n")

        # Observation-driven next-state update: explicit UA step, depends on ua_s and hat-variables
        f.write(
            "  // Observation transitions (only fired by Turn after an update): (ua_s, obs) -> ua_snext\n"
        )
        for s in range(1, self.internal_states + 1):
            for combo in self.combinations:
                guard = " & ".join(
                    f"{name}hat={val}" for name, val in zip(self.feature_names, combo)
                )
                f.write(
                    f"  [UA_TRANS_OBS] (ua_s={s}) & {guard} -> (ua_snext'={self._tr_obs_symbol(s, combo)});\n"
                )
        f.write("\n")

        # Apply step: set ua_s := ua_snext and assign c_<service> based on ua_snext
        f.write("  // Apply: commit state and counters based on ua_snext\n")
        for s_next in range(1, self.internal_states + 1):
            updates = [f"(ua_s'={s_next})"]
            for service in self.services:
                updates.append(
                    f"(c_{service}'={self._state_c_symbol(service, s_next)})"
                )
            f.write(
                f"  [UA_APPLY] (ua_snext={s_next}) -> " + " & ".join(updates) + ";\n"
            )
        f.write("endmodule\n\n")

    def add_turn(self, f: TextIOWrapper):
        """
        Turn sequencing.

        If trigger_on_action=True:
          t=4: UA_APPLY (initial apply) -> t=0
          t=0: action -> t=1
          t=1: UA_APPLY (apply action-transition result) -> t=2
          t=2: update_* -> t=3
               skip_update -> t=0   (if obs_only_after_update)
               skip_update -> t=3   (otherwise)
          t=3: UA_TRANS_OBS -> t=4
          t=4: UA_APPLY -> t=0

        If trigger_on_action=False:
          t=3: UA_APPLY (initial apply) -> t=0
          t=0: action -> t=1
          t=1: update_* -> t=2
               skip_update -> t=0   (if obs_only_after_update)
               skip_update -> t=2   (otherwise)
          t=2: UA_TRANS_OBS -> t=3
          t=3: UA_APPLY -> t=0
        """
        if self.trigger_on_action:
            f.write("module Turn\n")
            f.write("  t : [0..4] init 4;\n")

            for a in self.actions:
                f.write(f"  [{a}] (t=0) -> (t'=1);\n")

            f.write("\n  // Apply action-transition result\n")
            f.write("  [UA_APPLY] (t=1) -> (t'=2);\n")

            f.write("\n  // Update phase\n")
            for s in self.services:
                f.write(f"  [update_{s}] (t=2) -> (t'=3);\n")

            if self.obs_only_after_update:
                f.write("  [skip_update] (t=2) -> (t'=0);\n")
            else:
                f.write("  [skip_update] (t=2) -> (t'=3);\n")

            f.write("\n  // Observation transition (only after update if configured)\n")
            f.write("  [UA_TRANS_OBS] (t=3) -> (t'=4);\n")

            f.write("\n  // Apply obs-transition result (also runs once at start)\n")
            f.write("  [UA_APPLY] (t=4) -> (t'=0);\n")
            f.write("endmodule\n")
            return

        # trigger_on_action = False
        f.write("module Turn\n")
        f.write("  t : [0..3] init 3;\n")

        for a in self.actions:
            f.write(f"  [{a}] (t=0) -> (t'=1);\n")

        f.write("\n  // Update phase\n")
        for s in self.services:
            f.write(f"  [update_{s}] (t=1) -> (t'=2);\n")

        if self.obs_only_after_update:
            f.write("  [skip_update] (t=1) -> (t'=0);\n")
        else:
            f.write("  [skip_update] (t=1) -> (t'=2);\n")

        f.write("\n  [UA_TRANS_OBS] (t=2) -> (t'=3);\n")
        f.write("  [UA_APPLY] (t=3) -> (t'=0);\n")
        f.write("endmodule\n")

    # ------------ seed popfile ------------

    def _evolved_c_count(self) -> int:
        count = 0
        for _service in self.services:
            for s in range(1, self.internal_states + 1):
                is_fixed = self.setting == "fixed" and s <= min(
                    self.internal_states, self.fixed_count
                )
                if not is_fixed:
                    count += 1
        return count

    def _evolved_act_tr_count(self) -> int:
        if not self.trigger_on_action:
            return 0
        return len(self.actions) * self.internal_states

    def _evolved_obs_tr_count(self) -> int:
        return self.internal_states * len(self.combinations)

    def _evolvable_count(self) -> int:
        # ua_init_state + evolved c-values + (optional) action transitions + obs transitions
        return (
            1
            + self._evolved_c_count()
            + self._evolved_act_tr_count()
            + self._evolved_obs_tr_count()
        )

    def create_pop_file(self, f: TextIOWrapper, num_lines: int = 100):
        """
        Seed population where each row is filled with a constant value:
        row 1 -> all 1s
        row 2 -> all 2s
        ...
        up to min(max_val, self.state_count) (falls back to internal_states if state_count not present).
        """
        expected = self._evolvable_count()

        # prefer self.state_count if you added it; otherwise use internal_states
        state_cap = getattr(self, "state_count", self.internal_states)
        cap = min(self.max_val, state_cap)

        for c in range(1, cap + 1):
            values = [str(c)] * expected
            f.write(" ".join(values) + "\n")


from _io import TextIOWrapper
from itertools import product


class ParleyUAMealy:
    """
    Mealy-style UA (output on obs-transition):

    - Internal mode: ua_s in [1..internal_states]
    - Action transition (optional): (ua_s, action) -> ua_snext      [no output]
    - Observation transition (after update only): (ua_s, obs) -> (ua_snext, cnext_<service>)
    - Apply step [UA_APPLY] commits ua_s := ua_snext and c_<service> := cnext_<service>

    Notes:
    - c_<service> depends on (s, obs) because it is chosen on the obs transition.
    - For multi-service updates, the obs transition occurs after whichever update fired.
    """

    def __init__(
        self,
        infile: str,
        min_val: int = 1,
        max_val: int = 10,
        actions=("east", "west", "north", "south"),
        internal_states: int = 10,
        trigger_on_action: bool = True,
        obs_only_after_update: bool = True,
    ):
        self.infile = infile
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.actions = list(actions)
        self.internal_states = int(internal_states)
        self.trigger_on_action = bool(trigger_on_action)
        self.obs_only_after_update = bool(obs_only_after_update)

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
        # Copy while removing fixed const counters: const int c_<service> = ...;
        with open(infile, "r") as fin, open(outfile, "w") as fout:
            for line in fin:
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
        for service in self.services:
            f.write(
                f"evolve int ua_init_c_{service} "
                f"[{self.min_val}..{self.max_val}];\n"
            )
        f.write("\n")

        if self.trigger_on_action:
            for a in self.actions:
                for s in range(1, self.internal_states + 1):
                    f.write(
                        f"evolve int {self._tr_act_symbol(a, s)} [1..{self.internal_states}];\n"
                    )
            f.write("\n")

        for s in range(1, self.internal_states + 1):
            for combo in self.combinations:
                f.write(
                    f"evolve int {self._tr_obs_symbol(s, combo)} [1..{self.internal_states}];\n"
                )
                for service in self.services:
                    f.write(
                        f"evolve int {self._out_obs_symbol(service, s, combo)} "
                        f"[{self.min_val}..{self.max_val}];\n"
                    )
        f.write("\n")

        inline_commit = not self.obs_only_after_update

        f.write("module UA\n")
        f.write(f"  ua_s : [1..{self.internal_states}] init 1;\n")

        for service in self.services:
            f.write(
                f"  c_{service} : [{self.min_val}..{self.max_val}] init ua_init_c_{service};\n"
            )

        if not inline_commit:
            # old buffered style
            f.write(f"  ua_snext : [1..{self.internal_states}] init 1;\n")
            for service in self.services:
                f.write(
                    f"  cnext_{service} : [{self.min_val}..{self.max_val}] init ua_init_c_{service};\n"
                )
        f.write("\n")

        # Action transitions
        if self.trigger_on_action:
            f.write("  // Action transitions\n")
            for a in self.actions:
                for s in range(1, self.internal_states + 1):
                    if inline_commit:
                        # directly update ua_s
                        f.write(
                            f"  [{a}] (ua_s={s}) -> (ua_s'={self._tr_act_symbol(a, s)});\n"
                        )
                    else:
                        # buffered update
                        f.write(
                            f"  [{a}] (ua_s={s}) -> (ua_snext'={self._tr_act_symbol(a, s)});\n"
                        )
            f.write("\n")

        # Observation transitions
        f.write("  // Observation transitions (Mealy)\n")
        for s in range(1, self.internal_states + 1):
            for combo in self.combinations:
                guard = " & ".join(
                    f"{name}hat={val}" for name, val in zip(self.feature_names, combo)
                )

                if inline_commit:
                    updates = [f"(ua_s'={self._tr_obs_symbol(s, combo)})"]
                    for service in self.services:
                        updates.append(
                            f"(c_{service}'={self._out_obs_symbol(service, s, combo)})"
                        )
                else:
                    updates = [f"(ua_snext'={self._tr_obs_symbol(s, combo)})"]
                    for service in self.services:
                        updates.append(
                            f"(cnext_{service}'={self._out_obs_symbol(service, s, combo)})"
                        )

                f.write(
                    f"  [UA_TRANS_OBS] (ua_s={s}) & {guard} -> "
                    + " & ".join(updates)
                    + ";\n"
                )
        f.write("\n")

        # Apply step only in buffered mode
        if not inline_commit:
            f.write("  // Apply: commit ua_s and c_<service>\n")
            apply_updates = ["(ua_s'=ua_snext)"] + [
                f"(c_{service}'=cnext_{service})" for service in self.services
            ]
            f.write("  [UA_APPLY] true -> " + " & ".join(apply_updates) + ";\n")

        f.write("endmodule\n\n")

    def add_turn(self, f: TextIOWrapper):
        inline_commit = (not self.obs_only_after_update)

        if inline_commit:
            # No UA_APPLY steps exist in this mode.
            f.write("module Turn\n")
            f.write("  t : [0..2] init 0;\n")

            # Action
            for a in self.actions:
                f.write(f"  [{a}] (t=0) -> (t'=1);\n")

            f.write("\n  // Update phase\n")
            for s in self.services:
                f.write(f"  [update_{s}] (t=1) -> (t'=2);\n")
            f.write("  [skip_update] (t=1) -> (t'=2);\n")

            f.write("\n  // Observation transition\n")
            f.write("  [UA_TRANS_OBS] (t=2) -> (t'=0);\n")
            f.write("endmodule\n")
            return

        # --------- existing behavior (buffered style, keeps UA_APPLY steps) ---------

        if self.trigger_on_action:
            f.write("module Turn\n")
            f.write("  t : [0..4] init 4;\n")

            for a in self.actions:
                f.write(f"  [{a}] (t=0) -> (t'=1);\n")

            f.write("\n  // Apply action-transition result (state only)\n")
            f.write("  [UA_APPLY] (t=1) -> (t'=2);\n")

            f.write("\n  // Update phase\n")
            for s in self.services:
                f.write(f"  [update_{s}] (t=2) -> (t'=3);\n")

            if self.obs_only_after_update:
                f.write("  [skip_update] (t=2) -> (t'=0);\n")
            else:
                f.write("  [skip_update] (t=2) -> (t'=3);\n")

            f.write("\n  // Observation transition\n")
            f.write("  [UA_TRANS_OBS] (t=3) -> (t'=4);\n")

            f.write("\n  // Apply obs-transition output (also runs once at start)\n")
            f.write("  [UA_APPLY] (t=4) -> (t'=0);\n")
            f.write("endmodule\n")
            return

        # trigger_on_action = False
        f.write("module Turn\n")
        f.write("  t : [0..3] init 3;\n")

        for a in self.actions:
            f.write(f"  [{a}] (t=0) -> (t'=1);\n")

        f.write("\n  // Update phase\n")
        for s in self.services:
            f.write(f"  [update_{s}] (t=1) -> (t'=2);\n")

        if self.obs_only_after_update:
            f.write("  [skip_update] (t=1) -> (t'=0);\n")
        else:
            f.write("  [skip_update] (t=1) -> (t'=2);\n")

        f.write("\n  [UA_TRANS_OBS] (t=2) -> (t'=3);\n")
        f.write("  [UA_APPLY] (t=3) -> (t'=0);\n")
        f.write("endmodule\n")


    # ---------- popfile ----------

    def _evolved_act_tr_count(self) -> int:
        if not self.trigger_on_action:
            return 0
        return len(self.actions) * self.internal_states

    def _evolved_obs_tr_count(self) -> int:
        # next-state + outputs; outputs are per service
        per_obs = 1 + len(self.services)
        return self.internal_states * len(self.combinations) * per_obs

    def _evolvable_count(self) -> int:
        # ua_init_state + (optional) action transitions + obs transitions+outputs
        return 1 + self._evolved_act_tr_count() + self._evolved_obs_tr_count()

    def create_pop_file(self, f: TextIOWrapper):
        """
        Seed population where each row is filled with a constant value:
        row 1 -> all 1s
        row 2 -> all 2s
        ...
        up to min(max_val, internal_states).
        """
        expected = self._evolvable_count()
        cap = min(self.max_val, self.internal_states)

        for c in range(1, cap + 1):
            f.write(" ".join([str(c)] * expected) + "\n")
