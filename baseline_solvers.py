import argparse
from ortools.sat.python import cp_model
import gurobipy as gp
import os
import cplex
from cplex.callbacks import IncumbentCallback, MIPCallback
import pyscipopt as scp
import time
from helper import *
# gp.setParam('LogToConsole', 0)

def gurobi(ins_name, timelimit=3600, threads=1, log_path=None):
    SAVE_DIR = f'./results/QPLIB/gurobi_timelimit{timelimit}_threads{threads}'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    id = os.path.basename(ins_name)
    if os.path.exists(f'{SAVE_DIR}/results_{id}.txt'):
        print('Already exists')
        return 0
    m = gp.read(ins_name)
    if log_path is not None:
        m.Params.LogFile = log_path
    m.Params.Threads = threads
    m.Params.TimeLimit = timelimit
    m.Params.NonConvex = 2
    m.Params.Seed = 0
    incumbents = []
    timing = []
    def callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            timing.append(model.cbGet(gp.GRB.Callback.RUNTIME))
            incumbents.append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ))
    m.optimize(callback)

    id = os.path.basename(ins_name)
    with open(f'{SAVE_DIR}/results_{id}.txt', 'w') as f:
        f.write(str([timing, incumbents]))
    return 0


class MyEvent(scp.Eventhdlr):
    def eventinit(self):
        self.timing = []
        self.abs_timing = []
        self.incumbents = []
        self.start_time = time.monotonic()
        self.model.catchEvent(scp.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        self.sol_found_time = time.monotonic()
        sol = self.model.getBestSol()
        obj = self.model.getSolObjVal(sol)
        self.timing.append(self.sol_found_time - self.start_time)
        self.abs_timing.append(self.sol_found_time)
        self.incumbents.append(obj)


def scip(ins_name, time_limit=3600, threads=1, log_path=None):
    SAVE_DIR = f'./results/QPLIB/scip_timelimit{time_limit}_threads{threads}'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    id = os.path.basename(ins_name)
    if os.path.exists(f'{SAVE_DIR}/results_{id}.txt'):
        print('Already exists')
        return 0
    model = scp.Model()
    model.readProblem(ins_name)
    model.hideOutput(True)
    model.setParam("limits/time", time_limit)
    model.setParam('parallel/maxnthreads', threads)
    if log_path is not None:
        model.setLogfile(log_path)
    event = MyEvent()
    model.includeEventhdlr(
        event,
        "",
        ""
    )
    model.optimize()
    id = os.path.basename(ins_name)
    with open(f'{SAVE_DIR}/results_{id}.txt', 'w') as f:
        f.write(str([event.timing, event.incumbents]))
    return 0
class SolutionRecorder(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.start = time.perf_counter()
        self.times = []
        self.objective_values = []

    def on_solution_callback(self):
        self.__solution_count += 1
        t = time.perf_counter()
        self.times.append(t - self.start)
        objective_value = self.ObjectiveValue()
        self.objective_values.append(objective_value)

    def result(self):
        return self.times, self.objective_values


def cp_sat(ins, timelimit, threads):
    SAVE_DIR = f'./results/QPLIB/cpsat_timelimit{timelimit}_threads{threads}'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    id = os.path.basename(ins)
    if os.path.exists(f'{SAVE_DIR}/results_{id}.txt'):
        print('Already exists')
        return 0
    A, b, Q, _ = parse_lp(ins, torch.float)

    model = cp_model.CpModel()
    cp_vars = [model.NewIntVar(0, 1, f'v_{i}') for i in range(Q.shape[0])]
    get_quad_var_by_name = {}
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            z = model.NewIntVar(0, 1, f'z_{i, j}')
            model.AddMultiplicationEquality(z, cp_vars[i], cp_vars[j])
            get_quad_var_by_name[f'z_{i, j}'] = z

    for c_idx in range(A.shape[0]):
        lin_expr = sum(int(A[c_idx, v_idx].item()) * cp_vars[v_idx] for v_idx in range(A.shape[1]))
        model.add(lin_expr == int(b[c_idx].item()))

    # obj_expr = sum(Q[i, j].item() * cp_vars[i] * cp_vars[j] / 2 for i in range(Q.shape[0]) for j in range(Q.shape[1]))
    obj_expr = sum(0.5 * Q[i, j].item() * get_quad_var_by_name[f'z_{i, j}'] for i in range(Q.shape[0]) for j in range(Q.shape[1]))
    model.Minimize(obj_expr)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = threads
    solver.parameters.max_time_in_seconds = timelimit
    solution_recorder = SolutionRecorder()
    status = solver.Solve(model, solution_recorder)

    id = os.path.basename(ins)
    with open(f'{SAVE_DIR}/results_{id}.txt', 'w') as f:
        f.write(str(solution_recorder.result()))

    return 0


def solve_cplex(ins_name, timelimit, threads):
    SAVE_DIR = f'./results/QPLIB/cplex_timelimit{timelimit}_threads{threads}'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    id = os.path.basename(ins_name)
    if os.path.exists(f'{SAVE_DIR}/results_{id}.txt'):
        print('Already exists')
        return 0
    model = cplex.Cplex(ins_name)
    model.parameters.timelimit.set(timelimit)
    model.parameters.threads.set(threads)
    # model.parameters.mip.display.set(0)
    model.parameters.optimalitytarget.set(3)
    timing, incumbents = [], []
    start = time.perf_counter()
    class MyIncumbentCallback(IncumbentCallback):
        def __call__(self):
            # Get the incumbent objective value
            timing.append(time.perf_counter() - start)
            incumbents.append(self.get_objective_value())
    model.register_callback(MyIncumbentCallback)
    model.solve()
    id = os.path.basename(ins_name)
    with open(f'{SAVE_DIR}/results_{id}.txt', 'w') as f:
        f.write(str([timing, incumbents]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="gurobi")
    parser.add_argument("--timelimit", type=int, default=3600)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()


    # gp.setParam('LogToConsole', 0)
    for file in os.listdir('./instance/QPLIB'):
        ins_name = os.path.join('./instance/QPLIB', file)
        print(ins_name)
        m = scp.Model()
        m.readProblem(ins_name)
        if args.solver == 'gurobi':
            gurobi(ins_name, args.timelimit, args.threads)
        elif args.solver == 'scip':
            scip(ins_name, args.timelimit, args.threads)
        elif args.solver == 'cplex':
            solve_cplex(ins_name, args.timelimit, args.threads)
        elif args.solver == 'cpsat':
            cp_sat(ins_name, args.timelimit, args.threads)

