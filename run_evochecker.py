import os
from multiprocessing import Pool, cpu_count
import shutil


def run_task(args,uncertainty_aware=False):
    start_dir = os.getcwd()
    os.chdir("Applications/EvoChecker-master")
    os.environ["LD_LIBRARY_PATH"] = "libs/runtime"
    i, rep, suffix = args

    folder_to_delete = "data/ROBOT{0}_REP{1}{2}".format(i, rep, suffix)
    if os.path.exists(folder_to_delete):
        shutil.rmtree(folder_to_delete)
        print(f"Deleted folder: {folder_to_delete}")
    else:
        print(f"Folder not found, skipping delete: {folder_to_delete}")

    path = "./{0}_{1}.properties".format(str(i), str(rep))
    open(path, "w").close()
    with open(path, "a") as f:
        f.write("PROBLEM = ROBOT{0}_REP{1}{2}\n".format(str(i), str(rep), suffix))
        f.write(
            "       MODEL_TEMPLATE_FILE = models/model_{0}_umc.prism\n".format(str(i))
        )
        f.write("       PROPERTIES_FILE = ../../robot.pctl\n")
        f.write("       ALGORITHM = NSGAII\n")
        f.write("       POPULATION_SIZE = 100\n")
        f.write("       MAX_EVALUATIONS = 10000\n")
        f.write(f"       PROCESSORS = {cpu_count()}\n")  # cpu_count()
        f.write("       PLOT_PARETO_FRONT = false\n")
        f.write("       VERBOSE = true\n")
        f.write("       LOAD_SEED = false\n")
        #f.write("       SEED_FILE = data/ROBOT10/Front\n")
        f.write("       EVOCHECKER_TYPE = NORMAL\n")
        f.write("       EVOCHECKER_ENGINE = PRISM\n")
        f.write("       INIT_PORT = 55{0}\n".format(str(i)))
    # Note: INIT_PORT doesn't have an effect https://github.com/gerasimou/EvoChecker/issues/11

    # os.system("java -Xms1g -Xmx4g -jar ./target/EvoChecker-1.1.0.jar " + path)
    """os.environ["JAVA_TOOL_OPTIONS"] = (
        "-Xms512m -Xmx1g -XX:+UseG1GC "
        "-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=./prismexec.hprof "
        "-XX:+ExitOnOutOfMemoryError -Xlog:gc*"
    )"""

    os.system("java -jar ./target/EvoChecker-1.1.1.jar " + path)

    os.chdir(start_dir)


def run(map_, replications, suffix="",uncertainty_aware=False):
    # Number of parallel processes
    num_processes = 1  # cpu_count()

    # available maps
    rep_values = range(replications)  # 10 replications

    # Create a list of tuples with all combinations of i and rep
    tasks = [(map_, rep, suffix) for rep in rep_values]

    run_task(tasks[0],uncertainty_aware)
    """with Pool(num_processes) as pool:
        pool.map(run_task, tasks)"""
