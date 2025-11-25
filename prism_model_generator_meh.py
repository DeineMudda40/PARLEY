import csv
import json
import dijkstra

startX = 0
startY = 0
targetX = 4
targetY = 4
p = 0.01
directions = ["west", "east", "south", "north"]
directions_effects = [
    "(xhat'=max(xhat-1, 0))",
    "(xhat'=min(xhat+1, N))",
    "(yhat'=max(yhat-1, 0))",
    "(yhat'=min(yhat+1, N))",
]
error_precision = 10
obstacles = []

updates = [5]  # cost of updates

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)
corridor = 1

prism_file = ""
# period of updates
period = 1


def write_sensor_helpers():
    N = mapSize - 1
    obs = {(int(ox), int(oy)) for (ox, oy) in obstacles}

    def free(x, y):
        return 0 <= x <= N and 0 <= y <= N and (x, y) not in obs

    # For each (x,y), check if east is possible: (x+1,y) is free
    east_free  = [ (x,y) for x in range(N)     for y in range(N+1) if free(x+1, y) ]
    west_free  = [ (x,y) for x in range(1,N+1) for y in range(N+1) if free(x-1, y) ]
    north_free = [ (x,y) for y in range(N)     for x in range(N+1) if free(x,   y+1) ]
    south_free = [ (x,y) for y in range(1,N+1) for x in range(N+1) if free(x,   y-1) ]

    def disj(pairs):
        if not pairs:
            return "false"
        return " | ".join(f"(x={x} & y={y})" for (x,y) in pairs)

    east_ok  = disj(east_free)
    west_ok  = disj(west_free)
    north_ok = disj(north_free)
    south_ok = disj(south_free)

    with open(prism_file, "a") as f:
        f.write("// Direction OK predicates using free tiles\n")
        f.write(f"formula east_ok  = {east_ok};\n")
        f.write(f"formula west_ok  = {west_ok};\n")
        f.write(f"formula north_ok = {north_ok};\n")
        f.write(f"formula south_ok = {south_ok};\n\n")

        f.write("formula k_neighbors = "
                "(east_ok?1:0) + (west_ok?1:0) + "
                "(north_ok?1:0) + (south_ok?1:0);\n\n")






def build_map(filename):
    n = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            n.append(row)
    global mapSize
    global map_data
    mapSize = len(n)
    transposed = list(zip(*n))  # Transpose the matrix
    map_data = [row[::-1] for row in transposed]  # Reverse the order of each row
    global obstacles
    obstacles = []
    for x in range(0, mapSize):
        for y in range(0, mapSize):
            if int(map_data[x][y]) > 9:
                obstacles.append([x, y])
    # for j in range(0, mapSize):
    #    map_data.append([i[j] for i in n])


def preambel():
    with open(prism_file, "a") as f:
        f.write("dtmc\n")
        f.write(f"const int c = {period};\n")
        f.write("const int N=" + str(mapSize - 1) + ";\n")
        f.write("const int xstart = " + str(startX) + ";\n")
        f.write("const int ystart = " + str(startY) + ";\n")
        f.write("const int xtarget = " + str(targetX) + ";\n")
        f.write("const int ytarget = " + str(targetY) + ";\n")
        f.write("const double p = " + str(p) + ";\n \n")
        # formula for obstacles
        f.write("formula hasCrashed = (1=0) ")
        for x, y in obstacles:
            f.write("| (x={0} & y={1}) ".format(str(x), str(y)))
        f.write(";\n\n")


def robot():
    with open(prism_file, "a") as f:
        f.write("module Robot \n")
        f.write("  x : [0..N] init xstart;\n")
        f.write("  y : [0..N] init ystart;\n")
        f.write("  move_ready : [0..1] init 1;\n")
        f.write("  crashed : [0..1] init 0;\n\n")
        f.write(
            "  [east] (move_ready=1) -> \n"
            "    (1-3*p): (x'=min(x+1, N)) & (move_ready'=0) + \n"
            "    p: (y'=min(y+1, N)) & (move_ready'=0) + \n"
            "    p: (y'=max(y-1, 0)) & (move_ready'=0) + \n"
            "    p: (x'=max(x-1, 0)) & (move_ready'=0); \n"
        )

        f.write(
            "  [west] (move_ready=1) -> \n"
            "    p: (x'=min(x+1, N)) & (move_ready'=0) + \n"
            "    p: (y'=min(y+1, N)) & (move_ready'=0) + \n"
            "    p: (y'=max(y-1, 0)) & (move_ready'=0) + \n"
            "    (1-3*p): (x'=max(x-1, 0)) & (move_ready'=0); \n"
        )

        f.write(
            "  [north] (move_ready=1) -> \n"
            "    p: (x'=min(x+1, N)) & (move_ready'=0) + \n"
            "    (1-3*p): (y'=min(y+1, N)) & (move_ready'=0) + \n"
            "    p: (y'=max(y-1, 0)) & (move_ready'=0) + \n"
            "    p: (x'=max(x-1, 0)) & (move_ready'=0); \n"
        )

        f.write(
            "  [south] (move_ready=1) -> \n"
            "    p: (x'=min(x+1, N)) & (move_ready'=0) + \n"
            "    p: (y'=min(y+1, N)) & (move_ready'=0) + \n"
            "    (1-3*p): (y'=max(y-1, 0)) & (move_ready'=0) + \n"
            "    p: (x'=max(x-1, 0)) & (move_ready'=0); \n"
        )
        f.write("\n")
        f.write(
            "  [check] (move_ready=0) & hasCrashed -> (crashed'=1) & (move_ready'=1); \n"
        )
        f.write("  [check] (move_ready=0) & !hasCrashed -> (move_ready'=1); \n")
        f.write("endmodule\n\n")


def adaptation_mape_controller(d):
    with open(prism_file, "a") as f:
        f.write("module Adaptation_MAPE_controller\n")
        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])
                if direction < 4:
                    f.write("  [{0}] ".format(directions[direction]))
                    f.write("(xhat={0}) & (yhat={1}) -> true;\n".format(str(x), str(y)))
        f.write("endmodule\n\n")


def knowledge(uncertainty_aware=False):
    with open(prism_file, 'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n')
        f.write('  yhat : [0..N] init ystart;\n')
        f.write('  step : [1..20] init 1;\n\n')

        f.write(f'  p_err : [0..{error_precision-1}] init 0;\n')
        f.write('  p_drawn : [0..1] init 0;\n\n')
            
        
        
        f.write('  ready : [0..1] init 1;\n')
        for d, effect in zip(directions, directions_effects):
            f.write(f'  [{d}] ready=1 -> {effect} & (ready\'=0);\n')

        

        f.write('\n  [pick_p] step>=c & ready=0 & p_drawn=0 ->\n')     
        for i in range(error_precision-1):
            f.write(f"  1.0/({error_precision}) : (p_err'={i})  & (p_drawn'=1) +\n")
        f.write(f"  1.0/({error_precision}) : (p_err'={error_precision-1})  & (p_drawn'=1);\n\n")

        p_err_over_E = f'(p_err*1.0)/{error_precision}'

        f.write('  [update] step>=c & ready=0 & p_drawn=1 & (k_neighbors > 0) ->\n')
        f.write(f'    (1 - {p_err_over_E}) : (xhat\'=x) & (yhat\'=y) & (step\'=1) & (ready\'=1) & (p_drawn\'=0)\n')
        f.write('  + ')
        f.write(f'    ({p_err_over_E}) * ((east_ok ? 1.0 : 0.0)/(k_neighbors*1.0))'
                ' : (xhat\'=min(x+1, N)) & (yhat\'=y) & (step\'=1) & (ready\'=1) & (p_drawn\'=0)\n')
        f.write('  + ')
        f.write(f'    ({p_err_over_E}) * ((west_ok ? 1.0 : 0.0)/(k_neighbors*1.0))'
                ' : (xhat\'=max(x-1, 0)) & (yhat\'=y) & (step\'=1) & (ready\'=1) & (p_drawn\'=0)\n')
        f.write('  + ')
        f.write(f'    ({p_err_over_E}) * ((north_ok ? 1.0 : 0.0)/(k_neighbors*1.0))'
                ' : (xhat\'=x) & (yhat\'=min(y+1, N)) & (step\'=1) & (ready\'=1) & (p_drawn\'=0)\n')
        f.write('  + ')
        f.write(f'    ({p_err_over_E}) * ((south_ok ? 1.0 : 0.0)/(k_neighbors*1.0))'
                ' : (xhat\'=x) & (yhat\'=max(y-1, 0)) & (step\'=1) & (ready\'=1) & (p_drawn\'=0);\n\n')

        f.write('  [skip_update] step<c & ready=0 -> (ready\'=1) & (step\'=step+1);\n')
        f.write('endmodule\n\n')


def rewards():
    with open(prism_file, "a") as f:
        f.write('rewards "cost" \n')
        f.write("  [east] true : 1; \n")
        f.write("  [west] true : 1; \n")
        f.write("  [north] true : 1; \n")
        f.write("  [south] true : 1; \n")
        f.write("  [update] true : 5;\n")
        f.write("endrewards \n\n")

        # f.write('label \"mission_success\" = (x=xtarget) & (y=ytarget) & (!hasCrashed);\n')


def read_params_from_file():
    with open("input.json", "r") as file:
        params = json.load(file)
    global startX, startY, targetX, targetY, map_file, p, updates
    startX = params["startX"]
    startY = params["startY"]
    targetX = params["targetX"]
    targetY = params["targetY"]
    p = params["p"]
    map_file = params["map_file"]
    updates = params["updates"]


# i depicts which map should be used
def generate_model(i, uncertainty_aware=False):
    global prism_file
    prism_file = "Applications/EvoChecker-master/models/model_" + str(i) + ".prism"
    read_params_from_file()
    build_map("maps/map_" + str(i) + ".csv")
    target_pos = (targetX, targetY)
    _d = dijkstra.compute_directions(map_data, target_pos)
    # we have to transpose the matrix in the end, similar to what we did when reading the map_data in the first place
    d = list(zip(*_d))  # Transpose the matrix

    open(prism_file, "w").close()
    preambel()
    write_sensor_helpers()
    robot()
    adaptation_mape_controller(d)
    knowledge(uncertainty_aware)
    rewards()

    print("finished map " + str(i))
