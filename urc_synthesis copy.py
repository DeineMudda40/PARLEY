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
        speed_mode: bool = False,
    ):
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.actions = list(actions)
        self.speed_mode = bool(speed_mode)

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

                if any(line.strip().startswith(f"const int c_{s}") for s in self.services):
                    continue

                fout.write(line)

        # 2) If speed_mode: rewrite Knowledge (remove due_* formula and re-guard update/skip)
        if self.speed_mode:
            with open(outfile, "r", encoding="utf-8") as f:
                text = f.read()
            text = self._rewrite_knowledge_speed_mode(text)
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(text)

        # 3) Append added modules/decls
        with open(outfile, "a") as f:
            if self.speed_mode:
                self.add_decisions_only(f)
                self.add_turn_speed(f)
            else:
                self.add_urc(f)
                self.add_turn(f)

        # 4) Population file (same number of evolvables in both modes here)
        with open(popfile, "w") as f:
            self.create_pop_file(f)

    # ---------- speed_mode rewrite ----------

    def _rewrite_knowledge_speed_mode(self, text: str) -> str:
        # Remove all "formula due_<service> = ..." lines (they reference c_<service>).
        text = re.sub(r"(?m)^\s*formula\s+due_[A-Za-z_]\w*\s*=.*;\s*\n", "", text)

        # Extract Knowledge module
        m = re.search(r"(?ms)^module\s+Knowledge\b.*?^endmodule\b", text)
        if not m:
            raise ValueError("Could not find 'module Knowledge ... endmodule' to rewrite.")

        block = m.group(0)

        # Capture RHS of existing update and skip commands (so we preserve your effects).
        update_rhs = {}
        for s in self.services:
            mu = re.search(
                rf"(?m)^\s*\[update_{re.escape(s)}\].*?->\s*(.*?)\s*;\s*$",
                block,
            )
            if not mu:
                raise ValueError(f"Could not find [update_{s}] command in Knowledge.")
            update_rhs[s] = mu.group(1).strip()

        ms = re.search(r"(?m)^\s*\[skip_update\].*?->\s*(.*?)\s*;\s*$", block)
        if not ms:
            raise ValueError("Could not find [skip_update] command in Knowledge.")
        skip_rhs = ms.group(1).strip()

        # Remove existing update/skip command lines (single-line commands assumed).
        new_lines = []
        for line in block.splitlines(True):
            if re.match(r"^\s*\[(update_|skip_update)", line):
                continue
            new_lines.append(line)

        # Insert generated guarded commands just before endmodule
        end_idx = None
        for i, line in enumerate(new_lines):
            if re.match(r"^\s*endmodule\b", line):
                end_idx = i
                break
        if end_idx is None:
            raise ValueError("Malformed Knowledge module: missing endmodule.")

        gen = []
        gen.append("\n  // speed_mode: inline per-observation thresholds (no URC module/state)\n")

        # Priority semantics: earlier services in self.services have higher priority
        for combo in self.combinations:
            hat_guard = self._hat_guard(combo)

            # updates
            higher_due_exprs = []
            for s in self.services:
                due_s = f"(step>={self._decision_var(s, combo)})"
                if higher_due_exprs:
                    not_higher = " & !(" + " | ".join(higher_due_exprs) + ")"
                else:
                    not_higher = ""
                gen.append(
                    f"  [update_{s}] {hat_guard} & {due_s}{not_higher} -> {update_rhs[s]};\n"
                )
                higher_due_exprs.append(due_s)

            # skip (no due service true) - use conjunctive form to keep guards disjoint and simple
            all_not_due = " & ".join(
                f"(step<{self._decision_var(s, combo)})" for s in self.services
            )
            gen.append(f"  [skip_update] {hat_guard} & {all_not_due} -> {skip_rhs};\n")

        new_lines.insert(end_idx, "".join(gen))
        new_block = "".join(new_lines)

        # Replace in full text
        return text[: m.start()] + new_block + text[m.end() :]

    # ---------- codegen (normal mode) ----------

    def add_urc(self, f: TextIOWrapper):
        names = list(self.features.keys())

        # evolve int <service>_decision_<obs>
        for service in self.services:
            for combo in product(*self.domains):
                parts = [f"{name}_{value}" for name, value in zip(names, combo)]
                label = "decision_" + "_".join(parts)
                f.write(f"evolve int {service}_{label} [{self.min_val}..{self.max_val}];\n")

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
            guard = " & ".join(f"{name}hat={value}" for name, value in zip(names, combo))
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

        for s in self.services:
            f.write(f"  [update_{s}] (t=2) -> (t'=0);\n")
        f.write("  [skip_update] (t=2) -> (t'=0);\n")
        f.write("endmodule\n")

    # ---------- codegen (speed_mode) ----------

    def add_decisions_only(self, f: TextIOWrapper):
        # Same evolve vars as normal mode, but no URC module/state at all.
        for service in self.services:
            for combo in product(*self.domains):
                f.write(
                    f"evolve int {self._decision_var(service, combo)} "
                    f"[{self.min_val}..{self.max_val}];\n"
                )
        f.write("\n")

    def add_turn_speed(self, f: TextIOWrapper):
        # 2-phase: action -> update/skip. No [URC] phase.
        f.write("module Turn\n")
        f.write("  t : [0..1] init 0;\n")
        for a in self.actions:
            f.write(f"  [{a}] (t=0) -> (t'=1);\n")

        f.write("\n")
        for s in self.services:
            f.write(f"  [update_{s}] (t=1) -> (t'=0);\n")
        f.write("  [skip_update] (t=1) -> (t'=0);\n")
        f.write("endmodule\n")

    # ---------- popfile ----------

    def create_pop_file(self, f: TextIOWrapper):
        num_vars = len(self.services) * len(self.combinations)
        for c in range(self.min_val, self.max_val + 1):
            f.write((" ".join([str(c)] * num_vars)) + "\n")


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
        speed_mode:bool=False,
    ):
        self.infile = infile
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.actions = list(actions)
        self.internal_states = int(internal_states)
        self.trigger_on_action = bool(trigger_on_action)
        self.obs_only_after_update = bool(obs_only_after_update)
        self.speed_mode=speed_mode

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
        inline_commit = not self.obs_only_after_update

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
