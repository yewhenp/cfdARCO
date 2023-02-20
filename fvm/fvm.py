from mesh import *

import numpy as np
import tqdm

class Variable:
    def __init__(self, mesh, initial, boundary_conditions, name=""):
        self.name = name
        if mesh is not None:
            self.mesh: Quadrangle2DMesh = mesh
            self.current = initial
            self.boundary_conditions = boundary_conditions
            self.history = []

    def set_bound(self):
        self.current = self.boundary_conditions(self.mesh, self.current)

    def add_history(self):
        self.history.append(self.current.copy())

    def __repr__(self):
        return self.name

    def __add__(self, other):
        return _SubVariable(self, other, lambda x, y: x + y, "add")

    def __radd__(self, other):
        return _SubVariable(self, other, lambda x, y: y + x, "add")

    def __sub__(self, other):
        return _SubVariable(self, other, lambda x, y: x - y, "add")

    def __neg__(self):
        return _SubVariable(self, -1, lambda x, y: x * y, "neg")

    def __rsub__(self, other):
        return _SubVariable(self, other, lambda x, y: y - x, "add")

    def __mul__(self, other):
        return _SubVariable(self, other, lambda x, y: x * y, "mul")

    def __rmul__(self, other):
        return _SubVariable(self, other, lambda x, y: y * x, "mul")

    def __truediv__(self, other):
        return _SubVariable(self, other, lambda x, y: x / y, "rdiv")

    def __rtruediv__(self, other):
        return _SubVariable(self, other, lambda x, y: y / x, "rdiv")

    def evaluate_first_order(self, **kwargs):
        raise NotImplemented

    def evaluate_second_order(self, **kwargs):
        raise NotImplemented

    def extract(self, left_part, **kwargs):
        return left_part

    def evaluate(self, **kwargs):
        return self.current.copy()

    def set_current(self, current, **kwargs):
        self.current = current

    def get_history(self):
        return self.history

    def solve(self, **kwargs):
        return Equation.EqSolver.solve_dt(equation=kwargs["equation"], time_var=self, set_var=self, dt=kwargs["dt"])


class DT(Variable):
    class UpdatePolicies:
        @staticmethod
        def constant_value(timesteps, time_s, **kwargs):
            return time_s / timesteps

        @staticmethod
        def CourantFriedrichsLewy(CFL, space_vars, deltas, **kwargs):
            with np.errstate(divide='ignore'):
                dt = CFL / np.sum([np.amax(space_vars[0].current) / deltas[0], np.amax(space_vars[1].current) / deltas[1]])
            if np.isinf(dt):
                dt = CFL * np.sum(deltas)
            return dt

    def __init__(self, update_fn=UpdatePolicies.constant_value, **params):
        super().__init__(None, None, None)
        self.name = "dt"
        self._dt = 0
        self.update_fn = update_fn
        self.params = params

    def evaluate(self, **kwargs):
        return kwargs["dt"]

    def update(self, **kwargs):
        self._dt = self.update_fn(**{**self.params, **kwargs})

    @property
    def dt(self):
        return self._dt


class Variable2d(Variable):
    pass
    # def evaluate_second_order(self, **kwargs):
    #     for i in range(self.mesh.num_nodes):
    #


class _SubVariable(Variable):
    def __init__(self, left_operand, right_operand, op, name):
        super().__init__(None, None, None)
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.op = op
        self.name = name

    def evaluate(self, **kwargs):
        if isinstance(self.left_operand, Variable):
            left_eval = self.left_operand.evaluate(**kwargs)
        else:
            left_eval = self.left_operand
        if isinstance(self.right_operand, Variable):
            right_eval = self.right_operand.evaluate(**kwargs)
        else:
            right_eval = self.right_operand
        return self.op(left_eval, right_eval)


class _SubVariableSecondOrder(Variable):
    def __init__(self, var, direction):
        super().__init__(None, None, None)
        self.var = var
        self.direction = direction

    def evaluate(self, **kwargs):
        return self.var.evaluate_second_order(direction=self.direction)

    def extract(self, left_part, **kwargs):
        ret1, ret2 = self.var.get_shift_second_order(direction=self.direction)
        return -(left_part * self.var.get_delta(direction=self.direction)**2 - ret1 - ret2) / 2

    def get_history(self):
        return self.var.get_history()

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

class _DT(Variable):
    def __init__(self, var):
        super().__init__(None, None, None)
        self.var = var

    def extract(self, left_part, **kwargs):
        return (kwargs["dt"] * left_part) + self.var.current

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

    def solve(self, **kwargs):
        return Equation.EqSolver.solve_dt(equation=kwargs["equation"], time_var=self, set_var=self.var, dt=kwargs["dt"])


class _Laplass(Variable):
    def __init__(self, var):
        super().__init__(None, None, None)
        self.var = var

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

    def evaluate(self, **kwargs):
        ret = np.zeros_like(self.var.current)

        for i in range(self.var.mesh.num_nodes):
            node = self.var.mesh.nodes[i]
            summ = 0.0
            if not node.is_boundary():
                qq = 0
                for edge_id in node.edges_id:
                    edge = self.var.mesh.edged[edge_id]
                    n1 = self.var.mesh.nodes[edge.nodes_id[0]]
                    n2 = self.var.mesh.nodes[edge.nodes_id[1]]

                    fi = (self.var.current[n1.id] + self.var.current[n2.id]) / 2

                    norm = edge.normal
                    if qq == 0 and np.any(norm) < 0:
                        norm *= -1
                    if qq == 1 and np.any(norm) > 0:
                        norm *= -1
                    if qq == 2 and np.any(norm) > 0:
                        norm *= -1
                    if qq == 3 and np.any(norm) < 0:
                        norm *= -1

                    flux = np.sum(fi * norm * edge.area)
                    summ += flux
                    qq += 1

            summ /= node.volume
            ret[i] = summ

        return ret


def d2dx(var):
    return _SubVariableSecondOrder(var, direction="x")

def d2dy(var):
    return _SubVariableSecondOrder(var, direction="y")

def d1t(var):
    return _DT(var)


def laplass(var):
    return _Laplass(var)


class Equation:
    class EqSolver:
        @staticmethod
        def solve_dt(equation, time_var, set_var, dt: DT):
            current = equation.evaluate()
            extracted = time_var.extract(current, dt=dt.dt)
            set_var.set_current(extracted, set_boundaries=0)

    def __init__(self, timesteps):
        self.timesteps = timesteps

    def evaluate(self, all_vars, equation_system, dt: DT):
        for t in tqdm.trange(self.timesteps):
            dt.update()

            for i in range(len(all_vars)):
                all_vars[i].set_bound()

            for equation in equation_system:
                left_part, _, right_part = equation
                left_part.solve(equation=right_part, dt=dt)

            for i in range(len(all_vars)):
                all_vars[i].add_history()

        return [varr.get_history() for varr in all_vars]




