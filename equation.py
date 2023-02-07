import numpy as np
import tqdm
import operator


class Variable:
    def __init__(self, initial, boundary_conditions, deltas):
        if initial is not None:
            self.current = initial.copy()
            self.boundary_conditions = boundary_conditions
            self.current = self.boundary_conditions(self.current)
            self.deltas = deltas
            self.history = []
            self.current = self.boundary_conditions(self.current)

    def __add__(self, other):
        return _SubVariable(self, other, operator.add, "add")

    def __mul__(self, other):
        return _SubVariable(self, other, operator.mul, "mul")

    def evaluate_first_order(self, **kwargs):
        raise NotImplemented

    def evaluate_second_order(self, **kwargs):
        raise NotImplemented

    def extract(self, left_part, **kwargs):
        return left_part

    def evaluate(self):
        return self.current.copy()

    # def dt(self, current, dt):
    #     self.current = ((dt * current) + self.current).copy()
    #     self.current = self.boundary_conditions(self.current)
    #     self.history.append(self.current.copy())

    def set_current(self, current, **kwargs):
        self.current = current
        if kwargs["set_boundaries"]:
            self.history.append(self.current.copy())
            self.current = self.boundary_conditions(self.current)

    def get_history(self):
        return self.history

    def get_delta(self, **kwargs):
        return self.deltas[0]

    def get_shift_first_order(self, **kwargs):
        return self.current

    def get_shift_second_order(self, **kwargs):
        return self.current, self.current


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
        # ret = self.boundary_conditions(ret)
        return ret

    def get_shift_second_order(self, **kwargs):
        ret1 = np.zeros_like(self.current)
        ret1[2:] = self.current[2:]
        ret2 = np.zeros_like(self.current)
        ret2[:-2] = self.current[:-2]
        # ret1 = self.boundary_conditions(ret1)
        # ret2 = self.boundary_conditions(ret2)
        return ret1, ret2



class Variable2d(Variable):
    def evaluate_first_order(self, **kwargs):
        arr_value = np.zeros_like(self.current)
        if kwargs["direction"] == "x":
            arr_value[1:-1, 1:-1] = (self.current[2:, 1:-1] - self.current[:-2, 1:-1]) / (2*self.deltas[0])
            # arr_value[1:, :] = (arr_value[1:, :] - arr_value[:-1, :]) / (2*self.deltas[0])
        if kwargs["direction"] == "y":
            arr_value[1:-1, 1:-1] = (self.current[1:-1, 2:] - self.current[1:-1, :-2]) / (2*self.deltas[1])
            # arr_value[:, 1:] = (arr_value[:, 1:] - arr_value[:, :-1]) / (2*self.deltas[1])
        return arr_value

    def evaluate_second_order(self, **kwargs):
        # arr_value = self.current.copy()
        arr_value = np.zeros_like(self.current)
        if kwargs["direction"] == "x":
            arr_value[1:-1, 1:-1] = (self.current[2:, 1:-1] - 2*self.current[1:-1, 1:-1] + self.current[:-2, 1:-1]) / (self.deltas[0] ** 2)
            # arr_value[1:-1, 1:-1] = (arr_value[2:, 1:-1] - 2*arr_value[1:-1, 1:-1] + arr_value[:-2, 1:-1]) / (self.deltas[0] ** 2)
        if kwargs["direction"] == "y":
            arr_value[1:-1, 1:-1] = (self.current[1:-1, 2:] - 2*self.current[1:-1, 1:-1] + self.current[1:-1, :-2]) / (self.deltas[1] ** 2)
            # arr_value[1:-1, 1:-1] = (arr_value[1:-1, 2:] - 2*arr_value[1:-1, 1:-1] + arr_value[1:-1, :-2]) / (self.deltas[1] ** 2)
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
        # ret = self.boundary_conditions(ret)
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
        # ret1 = self.boundary_conditions(ret1)
        # ret2 = self.boundary_conditions(ret2)
        return ret1, ret2


class _SubVariable(Variable):
    def __init__(self, left_operand, right_operand, op, name):
        super().__init__(None, None, None)
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.op = op
        self.name = name

    def evaluate(self):
        if isinstance(self.left_operand, Variable):
            left_eval = self.left_operand.evaluate()
        else:
            left_eval = self.left_operand
        if isinstance(self.right_operand, Variable):
            right_eval = self.right_operand.evaluate()
        else:
            right_eval = self.right_operand
        return self.op(left_eval, right_eval)

    def extract(self, left_part, **kwargs):
        if self.name == "add":
            if isinstance(self.left_operand, _SubVariableSecondOrder) and isinstance(self.right_operand, _SubVariableSecondOrder):
                ret1_l, ret2_l = self.left_operand.get_shift_second_order()
                ret1_r, ret2_r = self.right_operand.get_shift_second_order()
                delta_l = self.left_operand.get_delta()
                delta_r = self.right_operand.get_delta()
                factor = 1 / (2 / delta_l ** 2 + 2 / delta_r ** 2)
                p2_xy = (ret2_l + ret2_l) / (delta_l**2) + (ret2_r + ret2_r) / (delta_r**2)
                return p2_xy * factor - (factor / kwargs["dt"]) * left_part
                # return (left_part * (delta_l**2) * (delta_r**2) - delta_r*(ret1_l + ret2_l) - delta_l*(ret1_r + ret2_r)) / (-2*delta_l -2*delta_r)
                # return (left_part - 0.5*((ret1_l + ret2_l) / (delta_l**2) + (ret1_r + ret2_r) / (delta_r**2))) * (delta_l**2 * delta_r**2) / (2 * (delta_r + delta_l))
                # return (left_part - 0.5 * (ret1_l + ret2_l + ret1_r + ret2_r))  / 2
        raise NotImplementedError


class _SubVariableFirstOrder(Variable):
    def __init__(self, var, direction):
        super().__init__(None, None, None)
        self.var = var
        self.direction = direction

    def evaluate(self):
        return self.var.evaluate_first_order(direction=self.direction)

    # def dt(self, current, dt):
    #     return self.var.dt(current, dt)

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

    def evaluate(self):
        return self.var.evaluate_second_order(direction=self.direction)

    # def dt(self, current, dt):
    #     return self.var.dt(current, dt)

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


# class _SubVariableFourthOrder(Variable):
#     def __init__(self, var, direction):
#         super().__init__(None, None, None)
#         self.var = var
#         self.direction = direction
#
#     def evaluate(self):
#         return self.var.evaluate_second_order(direction=self.direction)
#
#     # def extract(self, left_part, **kwargs):
#     #     ret1, ret2 = self.var.get_shift_second_order(direction=self.direction)
#     #     return -(left_part * self.var.get_delta(direction=self.direction)**2 - ret1 - ret2) / 2
#
#     def get_shift_second_order(self):
#         return self.var.get_shift_second_order(direction=self.direction)
#
#     def get_history(self):
#         return self.var.get_history()
#
#     def get_delta(self, **kwargs):
#         return self.var.get_delta(direction=self.direction)
#
#     def set_current(self, current):
#         self.var.set_current(current)


class _DT(Variable):
    def __init__(self, var):
        super().__init__(None, None, None)
        self.var = var

    def extract(self, left_part, **kwargs):
        return (kwargs["dt"] * left_part) + self.var.current

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

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


class _Laplass(Variable):
    def __init__(self, var):
        super().__init__(None, None, None)
        self.var = var

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

    def evaluate(self):
        aaa = self.var.current[:-2, 1:-1] + self.var.current[2:, 1:-1] + self.var.current[1:-1, :-2] + self.var.current[1:-1, 2:] - 4 * self.var.current[1:-1, 1:-1]
        ret = self.var.current.copy()
        ret[1:-1, 1:-1] = aaa
        return ret


def d1dx(var):
    return _SubVariableFirstOrder(var, direction="x")

def d1dy(var):
    return _SubVariableFirstOrder(var, direction="y")

def d2dx(var):
    return _SubVariableSecondOrder(var, direction="x")

def d2dy(var):
    return _SubVariableSecondOrder(var, direction="y")

def dt(var):
    return _DT(var)

def d2t(var):
    return _D2T(var)

def laplass(var):
    return _Laplass(var)

class Equation:
    def __init__(self, timesteps = 200, time_s = 2):
        self.dt = time_s / timesteps
        self.timesteps = timesteps
        self.time_s = time_s

    def evaluate(self, all_vars, time_vars, equations):
        for t in tqdm.trange(self.timesteps):
            # CFL = 0.8
            # with np.errstate(divide='ignore'):
            #     dt = CFL / np.sum([np.amax(all_vars[0].current) / all_vars[0].deltas[0], np.amax(all_vars[1].current) / all_vars[1].deltas[1]])
            # # Escape condition if dt is infinity due to zero velocity initially
            # if np.isinf(dt):
            #     dt = CFL * (all_vars[0].deltas[0] + all_vars[1].deltas[1])
            # self.dt = dt

            for i in range(len(equations)):
                if i == 2:
                    qq = 1
                else:
                    qq = 1
                for _ in range(qq):
                    current = equations[i].evaluate()
                    extracted = time_vars[i].extract(current, dt=self.dt)
                    all_vars[i].set_current(extracted, set_boundaries=1)
        return [varr.get_history() for varr in all_vars]




