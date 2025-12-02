import csv
import json
import dijkstra
from _io import TextIOWrapper


class Robot_Problem:
    def __init__(self, map_params, map_array):
        self.start_pos = (map_params["startX"], map_params["startY"])
        self.target_pos = (map_params["targetX"], map_params["targetY"])
        self.p = map_params["p"]
        self.fix_cs = map_params["fix_cs"]

        self.update_names = map_params["update_names"]
        self.update_costs = map_params["update_costs"]
        self.update_effects = map_params["update_effects"]

        self.action_names = map_params["action_names"]
        self.action_knowledge_effects = map_params["action_knowledge_effects"]
        self.action_effects = map_params["action_effects"]
        self.action_costs = map_params["action_costs"]

        self.mapSize = len(map_array)

        transposed = list(zip(*map_array))  # Transpose the matrix
        self.map_data = [
            row[::-1] for row in transposed
        ]  # Reverse the order of each row

        self.obstacles = []
        for x in range(0, self.mapSize):
            for y in range(0, self.mapSize):
                if int(self.map_data[x][y]) > 9:
                    self.obstacles.append([x, y])

        _d = dijkstra.compute_directions(self.map_data, self.target_pos)
        self.d = list(zip(*_d))

        self.gps_error = float(map_params.get("gps_error", 0.0))  # e.g. 0.2

        # after obstacles are computed:
        self.obstacle_set = set((ox, oy) for ox, oy in self.obstacles)

    def to_file(self, prism_file):
        open(prism_file, "w").close()

        with open(prism_file, "a") as f:
            self.preambel(f)
            self.robot(f)
            self.adaptation_mape_controller(f)
            self.knowledge(f)
            self.turn(f)
            self.rewards(f)

        print("Finished map!")

    def preambel(self, f: TextIOWrapper):
        f.write("dtmc\n")
        for update_name, fc in zip(self.update_names, self.fix_cs):
            f.write(f"const int c_{update_name} = {fc};\n")
        f.write(f"const int N = {self.mapSize - 1};\n")
        f.write(f"const int xstart = {self.start_pos[0]};\n")
        f.write(f"const int ystart = {self.start_pos[1]};\n")
        f.write(f"const int xtarget = {self.target_pos[0]};\n")
        f.write(f"const int ytarget = {self.target_pos[1]};\n")
        f.write(f"const double p = {round(self.p,3)};\n")
        f.write(f"const double gps_err = {round(self.gps_error, 6)};\n")
        f.write("const double gps_off = gps_err/4;\n")

        # formula for obstacles
        f.write("formula hasCrashed = (1=0) ")
        for x, y in self.obstacles:
            f.write("| (x={0} & y={1}) ".format(str(x), str(y)))
        f.write(";\n\n")

    def robot(self, f: TextIOWrapper):
        f.write("module Robot \n")
        f.write("  x : [0..N] init xstart;\n")
        f.write("  y : [0..N] init ystart;\n")
        f.write("  crashed : [0..1] init 0;\n\n")
        for action_name, action_effect in zip(self.action_names, self.action_effects):
            f.write(f"[{action_name}] true -> \n {action_effect}\n")

        f.write("\n")
        # Crash update synchronised with the update phase (t=1) or skip
        for name in self.update_names:
            f.write(f"  [update_{name}] hasCrashed  -> (crashed'=1);\n")
            f.write(f"  [update_{name}] !hasCrashed -> (crashed'=crashed);\n")

        if self.update_names:
            f.write("  [skip_update] hasCrashed  -> (crashed'=1);\n")
            f.write("  [skip_update] !hasCrashed -> (crashed'=crashed);\n")

        f.write("endmodule\n\n")

    def adaptation_mape_controller(self, f: TextIOWrapper):
        f.write("module Adaptation_MAPE_controller\n")
        for x in range(self.mapSize):
            for y in range(self.mapSize):
                direction = int(self.d[y][x])
                if direction < 4:
                    f.write(f"  [{self.action_names[direction]}] ")
                    f.write(f"(xhat={x}) & (yhat={y}) -> true;\n")
        f.write("endmodule\n\n")

    def knowledge(self, f: TextIOWrapper):
        for name in self.update_names:
            f.write(f"formula due_{name} = (step>=c_{name});\n")

        f.write("module Knowledge\n")
        f.write("  xhat : [0..N] init xstart;\n")
        f.write("  yhat : [0..N] init ystart;\n")
        f.write("  step : [1..20] init 1;\n\n")

        for d, effect in zip(self.action_names, self.action_knowledge_effects):
            f.write(f"  [{d}] true -> {effect};\n\n")

        f.write("\n")
        higher_due = []
        for name, effect in zip(self.update_names, self.update_effects):
            if higher_due:
                cond_not_higher = " & !(" + " | ".join(higher_due) + ")"
            else:
                cond_not_higher = ""

            # Special-case imperfect GPS
            if name == "gps" and self.gps_error > 0.0:
                for tx in range(self.mapSize):
                    for ty in range(self.mapSize):
                        branches = self._gps_branches_for_true_xy(tx, ty)
                        f.write(
                            f"  [update_{name}] due_{name}{cond_not_higher} & (x={tx}) & (y={ty}) -> \n"
                            f"    {branches}\n"
                        )
            else:
                # default (perfect) update logic for other services
                f.write(
                    f"  [update_{name}] due_{name}{cond_not_higher} -> "
                    f"{effect} & (step'=1);\n"
                )

            higher_due.append(f"due_{name}")

        if self.update_names:
            f.write(
                "  [skip_update] !("
                + " | ".join([f"due_{n}" for n in self.update_names])
                + ") -> (step'=step+1);\n"
            )
        else:
            f.write("  [skip_update] true -> (step'=step+1);\n")
        f.write("endmodule\n\n")

    def rewards(self, f: TextIOWrapper):
        f.write('rewards "cost" \n')
        for action_name, action_cost in zip(self.action_names, self.action_costs):
            f.write(f"  [{action_name}] true : {action_cost}; \n")
        for update_name, update_cost in zip(self.update_names, self.update_costs):
            f.write(f"  [update_{update_name}] true : {update_cost};\n")
        f.write("endrewards \n\n")

    def turn(self, f: TextIOWrapper):
        f.write("module Turn\n")
        f.write("  t : [0..1] init 0;\n")

        # Movement phase: actions only when t = 0
        for a in self.action_names:
            f.write(f"  [{a}] (t=0) -> (t'=1);\n")

        f.write("\n")
        # Update phase: one of the updates or skip when t = 1
        for name in self.update_names:
            f.write(f"  [update_{name}] (t=1) -> (t'=0);\n")
        f.write("  [skip_update] (t=1) -> (t'=0);\n\n")

        f.write("endmodule\n\n")

    def _gps_branches_for_true_xy(self, tx: int, ty: int) -> str:
        """
        Return a PRISM probabilistic update for the GPS update when true position is (tx,ty).
        Method 1: each valid 4-neighbour gets gps_off; invalid mass snaps back to (tx,ty).
        """
        candidates = [(tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)]
        valid = []
        for nx, ny in candidates:
            if (
                0 <= nx < self.mapSize
                and 0 <= ny < self.mapSize
                and (nx, ny) not in self.obstacle_set
            ):
                valid.append((nx, ny))

        v = len(valid)
        parts = [f"(1 - gps_off*{v}): (xhat'={tx}) & (yhat'={ty}) & (step'=1)"]
        for nx, ny in valid:
            parts.append(f"gps_off: (xhat'={nx}) & (yhat'={ny}) & (step'=1)")

        return " + \n    ".join(parts) + ";"


def generate_robot_model(i, param_file="input.json"):
    prism_file = f"Applications/EvoChecker-master/models/model_{i}.prism"

    with open(param_file, "r") as file:
        params = json.load(file)

    map_array = []
    with open(f"maps/map_{i}.csv", "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            map_array.append(row)

    rp = Robot_Problem(params, map_array)

    rp.to_file(prism_file)
