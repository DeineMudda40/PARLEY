import os
import re
import shutil


def _get_range(name, possible_decisions):
    if isinstance(possible_decisions, dict):
        return possible_decisions.get(name, possible_decisions.get("default", (1, 10)))
    return possible_decisions  # backward compat with list/tuple


def _remove_global_defaults(file_path, global_evolvables):
    if not global_evolvables:
        return
    with open(file_path, "r") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        skip = False
        for name in global_evolvables:
            # match e.g.  "const int d "  or "const int g\t" but not "const int direction"
            pattern = rf"^\s*const\s+int\s+{name}(\s|=)"
            if re.match(pattern, line):
                skip = True
                break
        if not skip:
            cleaned.append(line)

    with open(file_path, "w") as f:
        f.writelines(cleaned)


def manipulate_prism_model(
    input_path,
    output_path,
    possible_decisions={"default": (1, 10)},
    decision_variables=[],
    global_evolvables=[],
    before_actions=["east", "west", "north", "south"],
    after_actions=["update", "skip_update"],
    module_name="Knowledge",
    baseline=False,
    initial_pop_file="Set",
):
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output files cannot be the same.")

    shutil.copyfile(input_path, output_path)

    _remove_global_defaults(output_path, global_evolvables)

    variables, estimates = get_variables(input_path, decision_variables)

    counter_ids = remove_counter_from_module(output_path)

    add_controller(
        output_path,
        estimates,
        variables,
        possible_decisions,
        baseline,
        initial_pop_file,
        counter_ids,
        global_evolvables,
    )

    add_turn(output_path, before_actions, after_actions)


def get_variables(prism_model_path, decision_variables):
    int_constants_pattern = re.compile(r"const\s+int\s+(\w+)\s*=\s*(-?\s*\d+)\s*;")
    int_constants = {}

    with open(prism_model_path, "r") as prism_model_file:
        for line in prism_model_file:
            matches = int_constants_pattern.finditer(line)
            for match in matches:
                int_constants[match.group(1)] = int(match.group(2).replace(" ", ""))

    int_variable_declaration_pattern = re.compile(
        r"(\w+)\s*:\s*\[(-?\s*\w+)\s*\.\.\s*(-?\s*\w+)\]\s*init\s*(-?\s*\w+)\s*;"
    )
    _vars = []
    _bel = []

    with open(prism_model_path, "r") as prism_model_file:
        for line in prism_model_file:
            matches = int_variable_declaration_pattern.finditer(line)
            for match in matches:
                if match.group(1)[-3:] == "hat":
                    _bel.append(match.group(1))
                elif match.group(1) not in decision_variables:
                    continue
                lower_limit = __get_limit(
                    match.group(2).replace(" ", ""), int_constants
                )
                upper_limit = __get_limit(
                    match.group(3).replace(" ", ""), int_constants
                )
                _vars.append([match.group(1), lower_limit, upper_limit])

    return _vars, _bel


def __get_limit(string, constants):
    if not string.lstrip("-").isdigit():
        return constants[string]
    else:
        return int(string)


def remove_counter_from_module(output_path):
    removed_counters = []
    new_lines = []
    pattern = re.compile(r"^\s*const\s+int\s+c\d*\s*=\s*\d+\s*;")
    with open(output_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if pattern.match(line):
            match = re.search(r"c(\d*)", line)
            if match:
                counter_id = match.group(1) or str(len(removed_counters) + 1)
                removed_counters.append(int(counter_id))
        else:
            new_lines.append(line)

    with open(output_path, "w") as file:
        file.writelines(new_lines)
    return sorted(removed_counters)


def add_controller(
    file_path,
    estimates,
    variables,
    possible_decisions,
    baseline,
    initial_pop_file,
    counter_ids,
    global_evolvables=[],
):
    combinations = generate_combinations_list(variables)
    __add_controller_prefix(
        file_path, possible_decisions, combinations, variables, baseline, counter_ids,global_evolvables
    )

    with open(file_path, "a") as file:
        # Loop through counter_ids and initialize them with the correct decision variables
        for idx, counter_id in enumerate(counter_ids):
            # Use the first combination for initialization of the counter
            first_combination = combinations[0]
            decision_var = f"decision"
            for var in first_combination:
                decision_var += f"_{var}"
            decision_var += f"_{counter_id-1}"

            # Initialize the counter with the appropriate decision variable
            file.write(f"  c : [1..10] init {decision_var};\n")
            # file.write(f'  c{counter_id} : [1..10] init {decision_var};\n')

        # Add URC logic to handle combinations and decisions for counters
        for combination in combinations:
            new_line = "  [URC] "
            for c, estimate in zip(combination, estimates):
                new_line += f"{estimate}={c} & "
            new_line = new_line.rstrip(" & ")

            # Generate updates for each counter based on combinations and dynamic decision variables
            """updates = [
                f"(c{counter_id}'=decision" + ''.join([f'_{c}' for c in combination]) + f'_{i})'
                for i, counter_id in enumerate(counter_ids)
            ]"""
            updates = [
                f"(c'=decision" + "".join([f"_{c}" for c in combination]) + f"_{i})"
                for i, counter_id in enumerate(counter_ids)
            ]
            new_line += " -> " + " & ".join(updates) + ";\n"
            file.write(new_line)

        # Finalize URC module
        file.write("endmodule\n")

    __generate_initial_population(initial_pop_file, possible_decisions, combinations,global_evolvables)


def __add_controller_prefix(
    file_path, possible_decisions, combinations, variables, baseline, counter_ids,global_evolvables=[]
):
    with open(file_path, "a") as file:
        # Add global evolvable variables
        if global_evolvables:
            for name in global_evolvables:
                if baseline:
                    file.write(f"const int {name} = 1;\n")
                else:
                    low, high = _get_range(name, possible_decisions)
                    file.write(f"evolve int {name} [{low}..{high}];\n")
        # Loop through combinations and create decision variables for each combination and counter
        for combination in combinations:
            # Loop through the counter ids to handle decision variables for each counter
            for counter_id in counter_ids:
                # Start the line with "evolve int decision" or "const int decision" based on baseline
                if baseline:
                    new_line = f"const int decision"
                else:
                    new_line = f"evolve int decision"

                # Append the combination of variables to the decision variable name
                for var in range(0, len(variables)):
                    new_line += f"_{combination[var]}"

                # Append the counter_id at the end of the variable name
                new_line += f"_{counter_id-1}"

                # Set the decision range: either constant (baseline) or dynamic range (evolve)
                if baseline:
                    new_line += "=1;"
                else:
                    low, high = _get_range("default", possible_decisions)
                    new_line += f" [{low}..{high}];"

                # Write the declaration for the decision variable
                file.write("\n" + new_line)

        # Add the constant one for potential other uses
        file.write("const int one=1;\n")

        # Begin the URC module declaration
        file.write("\nmodule URC\n")


def add_turn(file_path, before_actions, after_actions):
    with open(file_path, "a") as file:
        file.write("module Turn\n")
        file.write("  t : [0..2] init 0;\n")
        for action in before_actions:
            file.write(f"  [{action}] (t=0) -> (t'=1);\n")
        file.write("\n")
        file.write("  [URC] (t=1) -> (t'=2);\n")
        file.write("\n")
        for action in after_actions:
            file.write(f"  [{action}] (t=2) -> (t'=0);\n")
        if len(after_actions) == 0:
            file.write("  [] (t=2) -> (t'=0);\n")
        file.write("endmodule\n")


def generate_combinations_list(variables):
    result = []

    def generate_combinations_recursive(current_combination, remaining_variables):
        if not remaining_variables:
            result.append(tuple(current_combination))
            return
        current_variable = remaining_variables[0]
        for value in range(current_variable[1], current_variable[2] + 1):
            generate_combinations_recursive(
                current_combination + [value], remaining_variables[1:]
            )

    generate_combinations_recursive([], variables)
    return result


def __generate_initial_population(file_path, possible_decisions, combinations,global_evolvables):
    with open(file_path, "w") as file:
        low, high = _get_range("default", possible_decisions)
        for c in range(low, high):
            new_line = ""
            for _ in range(len(combinations)+len(global_evolvables)):
                new_line += f"{c} "
            file.write(new_line + "\n")
