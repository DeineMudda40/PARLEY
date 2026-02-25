import re
import stormpy


# Same two properties, but now we don't need shell-escaping quotes
PROPERTIES = (
    'P=? [ F (x=xtarget & y=ytarget & crashed=0) ]',
    'R{"cost"} =? [ C<=200 ]'
)


def compute_baseline(infile, period, outfile="out.prism"):
    """
    1. Rewrite all `const int c_* = ...;` to use the given `period`.
    2. Save to `outfile` (default: out.prism).
    3. Use stormpy to check the two properties on that model.
    4. Return the results as a single tab-separated string,
       e.g. "0.1234\t56.78\t", matching the old script’s style.
    """
    # --- Step 1: rewrite constants in the PRISM file ---
    pattern = re.compile(r"^\s*const\s+int\s+(c_[A-Za-z_]\w*)\s*=\s*[-0-9]+\s*;")

    with open(infile, "r") as fi, open(outfile, "w") as fo:
        for line in fi:
            m = pattern.match(line)
            if m:
                const_name = m.group(1)  # e.g. "c_gps"
                fo.write(f"const int {const_name} = {period};\n")
            else:
                fo.write(line)

    # --- Step 2: parse the (modified) PRISM program with stormpy ---
    prism_program = stormpy.parse_prism_program(outfile)

    resultline = ""

    # --- Step 3: model check each property and collect the result ---
    for prop_str in PROPERTIES:
        # Parse property for this PRISM program
        props = stormpy.parse_properties_for_prism_program(prop_str, prism_program)

        if len(props) == 0:
            raise RuntimeError(f"Could not parse property: {prop_str}")

        # Build model (like calling PRISM once per property)
        model = stormpy.build_model(prism_program, props)

        # Run model checking
        result = stormpy.model_checking(model, props[0])

        # Get value for initial state (equivalent to "Result: <value>" in PRISM)
        initial_state = model.initial_states[0]
        value = result.at(initial_state)

        # Append to tab-separated string, as in the original script
        resultline += str(value) + "\t"

    # If you want to delete out.prism afterwards, uncomment:
    # os.remove(outfile)

    return resultline
