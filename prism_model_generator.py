import csv
import json
import dijkstra
from _io import TextIOWrapper


class Robot_Problem:
    def __init__(self, map_params, map_array):
        self.start_pos = (map_params["startX"], map_params["startY"])
        self.target_pos = (map_params["targetX"], map_params["targetY"])
        self.p = map_params["p"]
        self.fix_cs=map_params["fix_cs"]

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

    def to_file(self, prism_file):
        open(prism_file, "w").close()

        with open(prism_file, "a") as f:
            self.preambel(f)
            self.robot(f)
            self.adaptation_mape_controller(f)
            self.knowledge(f)
            self.rewards(f)

        print("Finished map!")

    def preambel(self, f: TextIOWrapper):
        f.write("dtmc\n")
        for update_name,fc in zip(self.update_names,self.fix_cs):
            f.write(f"const int c_{update_name} = {fc};\n")
        f.write(f"const int N = {self.mapSize - 1};\n")
        f.write(f"const int xstart = {self.start_pos[0]};\n")
        f.write(f"const int ystart = {self.start_pos[1]};\n")
        f.write(f"const int xtarget = {self.target_pos[0]};\n")
        f.write(f"const int ytarget = {self.target_pos[1]};\n")
        f.write(f"const double p = {round(self.p,3)};\n")
        # formula for obstacles
        f.write("formula hasCrashed = (1=0) ")
        for x, y in self.obstacles:
            f.write("| (x={0} & y={1}) ".format(str(x), str(y)))
        f.write(";\n\n")

    def robot(self, f: TextIOWrapper):
        f.write("module Robot \n")
        f.write("  x : [0..N] init xstart;\n")
        f.write("  y : [0..N] init ystart;\n")
        f.write("  move_ready : [0..1] init 1;\n")
        f.write("  crashed : [0..1] init 0;\n\n")
        for action_name,action_effect in zip(self.action_names,self.action_effects):
            f.write(f"[{action_name}] (move_ready=1) -> \n {action_effect}\n")

        f.write("\n")
        f.write(
            "  [check] (move_ready=0) & hasCrashed -> (crashed'=1) & (move_ready'=1); \n"
        )
        f.write("  [check] (move_ready=0) & !hasCrashed -> (move_ready'=1); \n")
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
            f.write(f"formula due_{name} = (ready=0) & (step>=c_{name});\n")

        f.write("module Knowledge\n")
        f.write("  xhat : [0..N] init xstart;\n")
        f.write("  yhat : [0..N] init ystart;\n")

        f.write("  step : [1..20] init 1;\n\n")
        f.write("  ready : [0..1] init 1;\n")

        for d, effect in zip(self.action_names, self.action_knowledge_effects):
            f.write(f"  [{d}] ready=1 -> {effect} & (ready'=0);\n\n")

        f.write("\n")
        higher_due = []
        for name, effect in zip(self.update_names, self.update_effects):
            if higher_due:
                f.write(
                    f"  [update_{name}] due_{name} & !("
                    + " | ".join(higher_due)
                    + f") -> {effect} & (step'=1) & (ready'=1);\n"
                )
            else:
                f.write(
                    f"  [update_{name}] due_{name} -> {effect} & (step'=1) & (ready'=1);\n"
                )
            higher_due.append(f"due_{name}")
        f.write("\n")

        if self.update_names:
            f.write(
                "  [skip_update] (ready=0) & !("
                + " | ".join([f"due_{n}" for n in self.update_names])
                + ") -> (ready'=1) & (step'=step+1);\n"
            )
        else:
            f.write("  [skip_update] (ready=0) -> (ready'=1) & (step'=step+1);\n")
        f.write("endmodule\n\n")

    def rewards(self, f:TextIOWrapper):
        f.write('rewards "cost" \n')
        for action_name, action_cost in zip(self.action_names,self.action_costs):
            f.write(f"  [{action_name}] true : {action_cost}; \n")
        for update_name,update_cost in zip(self.update_names,self.update_costs):
            f.write(f"  [update_{update_name}] true : {update_cost};\n")
        f.write("endrewards \n\n")


def generate_robot_model(i, param_file="input.json"):
    prism_file = f"Applications/EvoChecker-master/models/model_{i}.prism"

    with open(param_file, "r") as file:
        params = json.load(file)

    map_array=[]
    with open(f"maps/map_{i}.csv", "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            map_array.append(row)

    rp=Robot_Problem(params,map_array)

    rp.to_file(prism_file)