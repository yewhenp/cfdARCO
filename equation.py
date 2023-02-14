import numpy as np
import tqdm
import operator


class Variable:
    def __init__(self, initial, boundary_conditions, deltas, name=""):
        self.name = name
        if initial is not None:
            self.current = initial.copy()
            self.boundary_conditions = boundary_conditions
            self.deltas = deltas
            self.history = []

    def set_bound(self):
        self.current = self.boundary_conditions(self.current)

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
        if kwargs["set_boundaries"]:
            self.history.append(self.current.copy())

    def get_history(self):
        return self.history

    def get_delta(self, **kwargs):
        return self.deltas[0]

    def get_deltas(self):
        return self.deltas

    def get_shift_first_order(self, **kwargs):
        return self.current

    def get_shift_second_order(self, **kwargs):
        return self.current, self.current

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


class Variable1d(Variable):
    def evaluate_first_order(self, **kwargs):
        arr_value = self.current.copy()
        arr_value[1:] = (arr_value[1:] - arr_value[:-1]) / (2*self.deltas[0])
        return arr_value

    def evaluate_second_order(self, **kwargs):
        arr_value = self.current.copy()
        arr_value[1:-1] = (arr_value[2:] - 2*arr_value[1:-1] + arr_value[:-2]) / (self.deltas[0] ** 2)
        return arr_value

    def get_delta(self, **kwargs):
        return self.deltas[0]

    def get_shift_first_order(self, **kwargs):
        ret = np.zeros_like(self.current)
        ret[1:] = self.current[1:]
        return ret

    def get_shift_second_order(self, **kwargs):
        ret1 = np.zeros_like(self.current)
        ret1[2:] = self.current[2:]
        ret2 = np.zeros_like(self.current)
        ret2[:-2] = self.current[:-2]
        return ret1, ret2



class Variable2d(Variable):
    def evaluate_first_order(self, **kwargs):
        arr_value = np.zeros_like(self.current)
        if kwargs["direction"] == "y":
            arr_value[1:-1, 1:-1] = (self.current[2:, 1:-1] - self.current[:-2, 1:-1]) / (2*self.deltas[0])
        if kwargs["direction"] == "x":
            arr_value[1:-1, 1:-1] = (self.current[1:-1, 2:] - self.current[1:-1, :-2]) / (2*self.deltas[1])
        return arr_value

    def evaluate_second_order(self, **kwargs):
        # arr_value = self.current.copy()
        arr_value = np.zeros_like(self.current)
        if kwargs["direction"] == "y":
            arr_value[1:-1, 1:-1] = (self.current[2:, 1:-1] - 2*self.current[1:-1, 1:-1] + self.current[:-2, 1:-1]) / (self.deltas[0] ** 2)
        if kwargs["direction"] == "x":
            arr_value[1:-1, 1:-1] = (self.current[1:-1, 2:] - 2*self.current[1:-1, 1:-1] + self.current[1:-1, :-2]) / (self.deltas[1] ** 2)
        return arr_value

    def get_delta(self, **kwargs):
        if kwargs["direction"] == "x":
            return self.deltas[0]
        if kwargs["direction"] == "y":
            return self.deltas[1]

    def get_shift_first_order(self, **kwargs):
        if kwargs["direction"] == "x":
            ret = np.zeros_like(self.current)
            ret[1:, :] = self.current[1:, :]
        if kwargs["direction"] == "y":
            ret = np.zeros_like(self.current)
            ret[:, 1:] = self.current[:, 1:]
        return ret

    def get_shift_second_order(self, **kwargs):
        if kwargs["direction"] == "x":
            ret1 = np.zeros_like(self.current)
            ret1[2:, :] = self.current[2:, :]
            ret2 = np.zeros_like(self.current)
            ret2[:-2, :] = self.current[:-2, :]
        if kwargs["direction"] == "y":
            ret1 = np.zeros_like(self.current)
            ret1[:, 2:] = self.current[:, 2:]
            ret2 = np.zeros_like(self.current)
            ret2[:, :-2] = self.current[:, :-2]
        return ret1, ret2


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


class _SubVariableFirstOrder(Variable):
    def __init__(self, var, direction):
        super().__init__(None, None, None)
        self.var = var
        self.direction = direction

    def evaluate(self, **kwargs):
        return self.var.evaluate_first_order(direction=self.direction)

    def extract(self, left_part, **kwargs):
        return -(left_part * self.var.get_delta(direction=self.direction) - self.var.get_shift_first_order(direction=self.direction))

    def get_history(self):
        return self.var.get_history()

    def get_delta(self, **kwargs):
        return self.var.get_delta(direction=self.direction)

    def get_shift_first_order(self):
        return self.var.get_shift_first_order(direction=self.direction)

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)


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

    def get_shift_second_order(self):
        return self.var.get_shift_second_order(direction=self.direction)

    def get_history(self):
        return self.var.get_history()

    def get_delta(self, **kwargs):
        return self.var.get_delta(direction=self.direction)

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

class _D2T(Variable):
    def __init__(self, var):
        super().__init__(None, None, None)
        self.var = var

    def extract(self, left_part, **kwargs):
        if len(self.var.history) < 2:
            ret = ((kwargs["dt"]**2) * left_part[1:-1, 1:-1]) + self.var.current[1:-1, 1:-1]
        else:
            ret = ((kwargs["dt"]**2) * left_part[1:-1, 1:-1]) - self.var.history[-2][1:-1, 1:-1] + 2*self.var.current[1:-1, 1:-1]
        aaa = np.zeros_like(left_part)
        aaa[1:-1, 1:-1] = ret
        return aaa

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
        aaa = self.var.current[:-2, 1:-1] + self.var.current[2:, 1:-1] + self.var.current[1:-1, :-2] + self.var.current[1:-1, 2:] - 4 * self.var.current[1:-1, 1:-1]
        ret = self.var.current.copy()
        ret[1:-1, 1:-1] = aaa
        return ret

    def extract(self, left_part, **kwargs):
        deltas = self.var.get_deltas()
        var_old = self.var.current.copy()
        var_xy_ = (var_old[2:, 1:-1] + var_old[:-2, 1:-1]) * deltas[0] ** 2 + (var_old[1:-1, 2:] + var_old[1:-1, :-2]) * deltas[1] ** 2
        var_xy = np.zeros_like(var_old)
        var_xy[1:-1, 1:-1] = var_xy_
        left_part_xy = left_part * deltas[0] ** 2 * deltas[1] ** 2
        return (var_xy + left_part_xy) / (2 * (deltas[0] ** 2 + deltas[1] ** 2))

    def solve(self, **kwargs):
        return Equation.EqSolver.solve_iteratively(equation=kwargs["equation"], time_var=self, set_var=self.var, dt=kwargs["dt"])


def d1dx(var):
    return _SubVariableFirstOrder(var, direction="x")

def d1dy(var):
    return _SubVariableFirstOrder(var, direction="y")

def d2dx(var):
    return _SubVariableSecondOrder(var, direction="x")

def d2dy(var):
    return _SubVariableSecondOrder(var, direction="y")

def d1t(var):
    return _DT(var)

def d2t(var):
    return _D2T(var)

def laplass(var):
    return _Laplass(var)

class Equation:
    class EqSolver:
        @staticmethod
        def solve_dt(equation, time_var, set_var, dt: DT):
            current = equation.evaluate()
            extracted = time_var.extract(current, dt=dt.dt)
            set_var.set_current(extracted, set_boundaries=0)

        @staticmethod
        def solve_iteratively(equation, time_var, set_var, dt: DT):
            error = 1
            tol = 1e-3
            qq = 0

            while error > tol:
                qq += 1
                old_val = set_var.current.copy()
                current = equation.evaluate(dt=dt.dt)
                extracted = time_var.extract(current, dt=dt.dt)
                error = np.amax(abs(extracted - old_val))
                set_var.set_current(extracted, set_boundaries=0)
                set_var.set_bound()

                if qq > 500:
                    tol *= 10

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




