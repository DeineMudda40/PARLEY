import os
import sys
import subprocess
import re

#prism_bin = 'Applications/prism/bin/prism' if sys.platform == "darwin" else 'prism'  # use local prism if not OS X
prism_bin = "prism"   # example: your real linux prism

properties = ('\'P=?[F(x=xtarget & y=ytarget & crashed=0)]\'', '\'R{"cost"}=?[C<=200]\'')
# properties = ('\'R{"service"} =? [F counter=iterations]\'', '\'R{"cost"} =? [F counter=iterations]\'')
command = f'{prism_bin} -maxiters 50000 out.prism -pf '


def compute_baseline(infile, period, outfile="out.prism"):
    pattern = re.compile(r"^\s*const\s+int\s+(c_[A-Za-z_]\w*)\s*=\s*[-0-9]+\s*;")

    with open(infile, "r") as fi, open(outfile, "w") as fo:
        for line in fi:
            m = pattern.match(line)
            if m:
                const_name = m.group(1)  # e.g. "c_gps"
                fo.write(f"const int {const_name} = {period};\n")
            else:
                fo.write(line)
    resultline = ''
    for prop, i in zip(properties, range(0, len(properties))):
        # Execute the command and capture the output
        try:
            result = subprocess.run(
                command + prop,
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture standard error
                shell=True,  # Use shell for command execution
                universal_newlines=True,  # Return output as text (str)
            )
            # Check if the command was successful (return code 0)
            if result.returncode == 0:
                # Capture standard output and standard error
                stdout = result.stdout

                # Print or process the captured output as needed
                # Find and print the line that starts with "Result:"
                lines = stdout.splitlines()
                for line in lines:
                    if line.startswith("Result:"):
                        resultline += line.split(' ')[1] + '\t'
                        # print("PRISM: " + prop + ' = ' + str(result))
                        break  # Stop searching after finding the first matching line
            else:
                print(f"Command failed with return code {result.returncode}")
                print(result.stdout)
                print(result.stderr)

        except Exception as e:
            print(f"An error occurred: {e}")
    #os.remove("out.prism")
    return resultline
