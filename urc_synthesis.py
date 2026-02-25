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