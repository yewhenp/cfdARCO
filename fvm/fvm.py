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
        def CourantFriedrichsLewy(CFL, space_vars, **kwargs):
            with np.errstate(divide='ignore'):
                dt = CFL / np.sum([np.amax(space_vars[0].current) / 1, np.amax(space_vars[1].current) / 1])
            if np.isinf(dt):
                dt = CFL * 2
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
            if not node.is_boundary():
                # nabla_TF = 0.0
                # for edge_id in node.edges_id:
                #     edge = self.var.mesh.edged[edge_id]
                #     n1 = self.var.mesh.nodes[edge.nodes_id[0]]
                #     n2 = self.var.mesh.nodes[edge.nodes_id[1]]
                #     if n1.id != node.id:
                #         n1, n2 = n2, n1
                #     Cf = edge.center_coords - n1.center_coords
                #     fF = n2.center_coords - edge.center_coords
                #     Cf_n = np.sqrt(np.sum((Cf) ** 2))
                #     fF_n = np.sqrt(np.sum((fF) ** 2))
                #     g = Cf_n / (Cf_n + fF_n)
                #     val = g*self.var.current[n1.id] + (1-g)*self.var.current[n2.id]
                #     nabla_TF += val * edge.area
                #
                # nabla_TF /= node.volume

                summ = 0.0
                for edge_id in node.edges_id:
                    edge = self.var.mesh.edged[edge_id]
                    n1 = self.var.mesh.nodes[edge.nodes_id[0]]
                    n2 = self.var.mesh.nodes[edge.nodes_id[1]]
                    if n1.id != node.id:
                        n1, n2 = n2, n1

                    FC_v = n2.center_coords - n1.center_coords
                    dist_between_nodes = np.sqrt(np.sum((FC_v) ** 2))
                    FC_v_e = FC_v / dist_between_nodes

                    fi = (self.var.current[n1.id] - self.var.current[n2.id]) / dist_between_nodes
                    Sf = edge.normal * edge.area  # https://habr.com/ru/post/276193/
                    Ef = (FC_v_e * Sf) * FC_v_e
                    ort_term = fi * Ef

                    # Tf = Sf - Ef
                    # Cf = edge.center_coords - n1.center_coords
                    # fF = n2.center_coords - edge.center_coords
                    # Cf_n = np.sqrt(np.sum((Cf) ** 2))
                    # fF_n = np.sqrt(np.sum((fF) ** 2))
                    # g = Cf_n / (Cf_n + fF_n)
                    #
                    # nabla_TC = 0.0
                    # for edge_id_ in self.var.mesh.nodes[n2.id].edges_id:
                    #     edge_ = self.var.mesh.edged[edge_id_]
                    #     n1_ = self.var.mesh.nodes[edge_.nodes_id[0]]
                    #     if len(edge_.nodes_id) > 1:
                    #         n2_ = self.var.mesh.nodes[edge_.nodes_id[1]]
                    #     else:
                    #         n2_ = self.var.mesh.nodes[edge_.nodes_id[0]]
                    #
                    #     if n1_.id != n2:
                    #         n1_, n2_ = n2_, n1_
                    #     Cf_ = edge_.center_coords - n1_.center_coords
                    #     fF_ = n2_.center_coords - edge_.center_coords
                    #     Cf_n_ = np.sqrt(np.sum((Cf_) ** 2))
                    #     fF_n_ = np.sqrt(np.sum((fF_) ** 2))
                    #     g_ = Cf_n_ / (Cf_n_ + fF_n_)
                    #     val_ = g_ * self.var.current[n1_.id] + (1 - g_) * self.var.current[n2_.id]
                    #     nabla_TC += val_ * edge_.area
                    #
                    # nabla_TC /= self.var.mesh.nodes[n2.id].volume
                    #
                    # nabla_Tf = g*nabla_TC + (1-g)*nabla_TF
                    # cross_dif_term = nabla_Tf * Tf

                    # flux = np.sum(ort_term + cross_dif_term)
                    flux = np.sum(ort_term)
                    summ += flux

                summ /= node.volume
                ret[i] = summ

        return ret

    def extract(self, left_part, **kwargs):
        deltas = self.var.get_deltas()
        var_old = self.var.current.copy()
        var_xy_ = (var_old[2:, 1:-1] + var_old[:-2, 1:-1]) * deltas[0] ** 2 + (var_old[1:-1, 2:] + var_old[1:-1, :-2]) * deltas[1] ** 2
        var_xy = np.zeros_like(var_old)
        var_xy[1:-1, 1:-1] = var_xy_
        left_part_xy = left_part * deltas[0] ** 2 * deltas[1] ** 2
        return (var_xy + left_part_xy) / (2 * (deltas[0] ** 2 + deltas[1] ** 2))


class _Grad(Variable):
    def __init__(self, var, clc_x=1, clc_y=1):
        super().__init__(None, None, None)
        self.var = var
        self.clc_x = clc_x
        self.clc_y = clc_y
        self.clc = np.asarray([clc_x, clc_y])

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

    def evaluate(self, **kwargs):
        ret = np.zeros_like(self.var.current)

        for i in range(self.var.mesh.num_nodes):
            node = self.var.mesh.nodes[i]
            if not node.is_boundary():
                summ = 0.0
                for edge_id in node.edges_id:
                    edge = self.var.mesh.edged[edge_id]
                    n1 = self.var.mesh.nodes[edge.nodes_id[0]]
                    n2 = self.var.mesh.nodes[edge.nodes_id[1]]
                    if n1.id != node.id:
                        n1, n2 = n2, n1

                    # dist_between_nodes = np.sqrt(np.sum((n1.center_coords - n2.center_coords) ** 2))
                    fi = (self.var.current[n1.id] + self.var.current[n2.id]) / 2

                    norm = edge.normal * self.clc
                    flux = np.sum(fi * norm * edge.area)
                    summ += flux

                summ /= node.volume
                ret[i] = summ

        return ret


def d1t(var):
    return _DT(var)


def laplass(var):
    return _Laplass(var)


def d1dx(var):
    return _Grad(var, clc_x=1, clc_y=0)

def d1dy(var):
    return _Grad(var, clc_x=0, clc_y=1)

def grad(var):
    return _Grad(var, clc_x=1, clc_y=1)


class Equation:
    class EqSolver:
        @staticmethod
        def solve_dt(equation, time_var, set_var, dt: DT):
            current = equation.evaluate(dt=dt)
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




