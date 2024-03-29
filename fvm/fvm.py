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
            self._cache = None
            self._cache_valid = False
            self._grad_cache = None
            self._grad_cache_valid = False

    @property
    def cache_valid(self):
        return self._cache_valid

    def set_bound(self):
        self.current = self.boundary_conditions(self.mesh, self.current)

    def add_history(self):
        self.history.append(self.current.copy())

    def estimate_grads(self):
        if self._grad_cache_valid:
            return self._grad_cache

        current = self.current
        ret = np.zeros((self.mesh.num_nodes, 2), dtype=np.float64)

        for i in range(self.mesh.num_nodes):
            node = self.mesh.nodes[i]
            summ = np.asarray([0.0, 0.0], dtype=np.float64)
            for idx, edge_id in enumerate(node.edges_id):
                edge = self.mesh.edged[edge_id]
                if len(edge.nodes_id) > 1:
                    n1 = self.mesh.nodes[edge.nodes_id[0]]
                    n2 = self.mesh.nodes[edge.nodes_id[1]]
                else:
                    n1 = self.mesh.nodes[edge.nodes_id[0]]
                    n2 = self.mesh.nodes[edge.nodes_id[0]]
                if n1.id != node.id:
                    n1, n2 = n2, n1

                fi = (current[n1.id] + current[n2.id]) / 2
                summ += fi * n1.normals[idx] #* edge.area
            ret[i] = summ / node.volume

        self._grad_cache_valid = True
        self._grad_cache = ret
        return ret

    @property
    def dx(self):
        return _GradEstimated(self, 1, 0)

    @property
    def dy(self):
        return _GradEstimated(self, 0, 1)

    @property
    def grid(self):
        grid = np.zeros((self.mesh.x, self.mesh.y), dtype=np.float64)
        for x_ in range(self.mesh.x):
            for y_ in range(self.mesh.y):
                grid[x_, y_] = self.current[self.mesh.coord_fo_idx(x_, y_)]
        return grid

    def get_interface_vars_first_order(self, clc, **kwargs):
        if self._cache_valid:
            return self._cache

        current = self.current
        grads = self.estimate_grads()
        ret = np.zeros((self.mesh.num_nodes, 4, 3), dtype=np.float64)

        for i in range(self.mesh.num_nodes):
            node = self.mesh.nodes[i]
            for idx, edge_id in enumerate(node.edges_id):
                edge = self.mesh.edged[edge_id]
                if len(edge.nodes_id) > 1:
                    n1 = self.mesh.nodes[edge.nodes_id[0]]
                    n2 = self.mesh.nodes[edge.nodes_id[1]]
                else:
                    n1 = self.mesh.nodes[edge.nodes_id[0]]
                    n2 = self.mesh.nodes[edge.nodes_id[0]]
                if n1.id != node.id:
                    n1, n2 = n2, n1

                n1_to_mid = n1.vectors_in_edges_directions[idx]
                n2_to_mid = n2.vectors_in_edges_directions_by_id[edge_id]

                fi_n1 =  grads[n1.id] @ n1_to_mid + current[n1.id]
                fi_n2 =  grads[n2.id] @ n2_to_mid + current[n2.id]

                ret[i][idx][0] = (fi_n1 + fi_n2) / 2

                if idx == 0:
                    ret[i][idx][1] = fi_n2
                    ret[i][idx][2] = fi_n1
                elif idx == 1:
                    ret[i][idx][1] = fi_n1
                    ret[i][idx][2] = fi_n2
                elif idx == 2:
                    ret[i][idx][1] = fi_n1
                    ret[i][idx][2] = fi_n2
                elif idx == 3:
                    ret[i][idx][1] = fi_n2
                    ret[i][idx][2] = fi_n1

        self._cache_valid = True
        self._cache = ret
        return ret


    def __repr__(self):
        return self.name

    def __add__(self, other):
        return _SubVariable(self, other, lambda x, y: x + y, "add")

    def __radd__(self, other):
        return _SubVariable(self, other, lambda x, y: y + x, "add")

    def __sub__(self, other):
        return _SubVariable(self, other, lambda x, y: x - y, "sub")

    def __neg__(self):
        return _SubVariable(self, -1, lambda x, y: x * y, "neg")

    def __rsub__(self, other):
        return _SubVariable(self, other, lambda x, y: y - x, "sub")

    def __mul__(self, other):
        return _SubVariable(self, other, lambda x, y: x * y, "mul")

    def __rmul__(self, other):
        return _SubVariable(self, other, lambda x, y: y * x, "mul")

    def __truediv__(self, other):
        return _SubVariable(self, other, lambda x, y: x / y, "div")

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
        aaa = to_grid(current)
        self.current = current
        self._cache_valid = False
        self._grad_cache_valid = False

    def get_history(self):
        return self.history

    def solve(self, **kwargs):
        return Equation.EqSolver.solve_dt(equation=kwargs["equation"], time_var=self, set_var=self, dt=kwargs["dt"])


class _GradEstimated(Variable):
    def __init__(self, var, clc_x=1, clc_y=1):
        super().__init__(None, None, None)
        self.var = var
        self.clc_x = clc_x
        self.clc_y = clc_y
        self.mask_x = np.asarray([[0, 1, 0, 1]], dtype=np.float64)
        self.mask_y = np.asarray([[1, 0, 1, 0]], dtype=np.float64)
        self.clc = np.asarray([clc_x, clc_y], dtype=np.float64)

    def evaluate(self, **kwargs):
        grads = self.var.estimate_grads()
        if self.clc_x and self.clc_y:
            return grads[:,:,0] + grads[:,:,1]
        if self.clc_x:
            return grads[:,0]
        if self.clc_y:
            return grads[:,1]


class DT(Variable):
    class UpdatePolicies:
        @staticmethod
        def constant_value(timesteps, time_s, **kwargs):
            return time_s / timesteps

        @staticmethod
        def CourantFriedrichsLewy(CFL, space_vars, **kwargs):
            u, v, p, rho, gamma, l = space_vars
            dl = 1 / l

            with np.errstate(divide='ignore'):
                dt = CFL * np.min(dl / (np.sqrt(gamma * p.current / rho.current) + np.sqrt(u.current ** 2 + v.current ** 2)))
            if np.isinf(dt):
                dt = CFL * 2
            return dt

    def __init__(self, update_fn=UpdatePolicies.constant_value, **params):
        super().__init__(None, None, None)
        self.name = "dt"
        self._dt = 0
        self.update_fn = update_fn
        self.params = params

    @property
    def cache_valid(self):
        return False


    def evaluate(self, **kwargs):
        return self._dt

    def update(self, **kwargs):
        self._dt = self.update_fn(**{**self.params, **kwargs})

    @property
    def dt(self):
        return self._dt


class Variable2d(Variable):
    pass


class _SubVariable(Variable):
    def __init__(self, left_operand, right_operand, op, name):
        super().__init__(None, None, None)
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.op = op
        self.name = name
        self._cache_valid = False
        self._cache = None

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

    @property
    def current(self):
        return self.evaluate()

    @property
    def mesh(self):
        if isinstance(self.left_operand, Variable):
            return self.left_operand.mesh
        if isinstance(self.right_operand, Variable):
            return self.right_operand.mesh

    @property
    def cache_valid(self):
        if isinstance(self.left_operand, Variable):
            self._cache_valid = self._cache_valid and self.left_operand.cache_valid
        if isinstance(self.right_operand, Variable):
            self._cache_valid = self._cache_valid and self.right_operand.cache_valid
        return self._cache_valid

    def get_interface_vars_first_order(self, **kwargs):
        if self.cache_valid:
            return self._cache

        if isinstance(self.left_operand, Variable):
            left_eval = self.left_operand.get_interface_vars_first_order(**kwargs)
        else:
            left_eval = self.left_operand
        if isinstance(self.right_operand, Variable):
            right_eval = self.right_operand.get_interface_vars_first_order(**kwargs)
        else:
            right_eval = self.right_operand

        res = self.op(left_eval, right_eval)

        # self._cache_valid = True
        self._cache = res
        return res


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
        return kwargs["dt"] * left_part + self.var.current

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

    def solve(self, **kwargs):
        return Equation.EqSolver.solve_dt(equation=kwargs["equation"], time_var=self, set_var=self.var, dt=kwargs["dt"])


class _Laplass(Variable):
    def __init__(self, var, clc_x=1, clc_y=1):
        super().__init__(None, None, None)
        self.var = var
        self.clc_x = clc_x
        self.clc_y = clc_y
        self.mask_x = np.asarray([[0, 1, 0, 1]], dtype=np.float64)
        self.mask_y = np.asarray([[1, 0, 1, 0]], dtype=np.float64)
        self.clc = np.asarray([clc_x, clc_y], dtype=np.float64)

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

    def evaluate(self, **kwargs):
        ret = np.zeros_like(self.var.current, dtype=np.float64)

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

                    FC_v = n2.center_coords - n1.center_coords
                    dist_between_nodes = np.sqrt(np.sum((FC_v) ** 2, dtype=np.float64), dtype=np.float64)
                    FC_v_e = FC_v / dist_between_nodes

                    fi = (self.var.current[n1.id] - self.var.current[n2.id]) / dist_between_nodes
                    Sf = edge.normal * edge.area * self.clc  # https://habr.com/ru/post/276193/
                    Ef = (FC_v_e * Sf) * FC_v_e
                    ort_term = fi * Ef

                    flux = np.sum(ort_term, dtype=np.float64)
                    summ += flux

                summ /= node.volume
                ret[i] = summ

        return ret


class _Grad(Variable):
    def __init__(self, var, clc_x=1, clc_y=1):
        super().__init__(None, None, None)
        self.var = var
        self.clc_x = clc_x
        self.clc_y = clc_y
        self.mask_x = np.asarray([[0, 1, 0, 1]], dtype=np.float64)
        self.mask_y = np.asarray([[1, 0, 1, 0]], dtype=np.float64)
        self.clc = np.asarray([clc_x, clc_y], dtype=np.float64)

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

    def evaluate(self, **kwargs):
        current_interface_gradients = self.var.get_interface_vars_first_order(clc=self.clc)
        current_interface_gradients_star = current_interface_gradients[:,:,0]

        grad_x = current_interface_gradients_star[:,:] * self.var.mesh.normal_x
        grad_y = current_interface_gradients_star[:,:] * self.var.mesh.normal_y

        res_x = np.sum(grad_x, axis=1, dtype=np.float64)
        res_y = np.sum(grad_y, axis=1, dtype=np.float64)

        if self.clc_x and self.clc_y:
            return res_x + res_y
        if self.clc_x:
            return res_x
        if self.clc_y:
            return res_y


class _Stab(Variable):
    def __init__(self, var, clc_x=1, clc_y=1):
        super().__init__(None, None, None)
        self.var = var
        self.clc_x = clc_x
        self.clc_y = clc_y
        self.mask_x = np.asarray([[0, 1, 0, 1]], dtype=np.float64)
        self.mask_y = np.asarray([[1, 0, 1, 0]], dtype=np.float64)
        self.clc = np.asarray([clc_x, clc_y], dtype=np.float64)

    def set_current(self, current, **kwargs):
        self.var.set_current(current, **kwargs)

    def evaluate(self, **kwargs):
        current_interface_gradients = self.var.get_interface_vars_first_order(clc=self.clc)
        current_interface_gradients_star = (current_interface_gradients[:, :, 2] - current_interface_gradients[:, :, 1]) / 2
        grad_x = current_interface_gradients_star[:,:] * self.var.mesh.normal_x
        grad_y = current_interface_gradients_star[:,:] * self.var.mesh.normal_y

        res_x = np.sum(grad_x, axis=1, dtype=np.float64)
        res_y = np.sum(grad_y, axis=1, dtype=np.float64)

        if self.clc_x and self.clc_y:
            return res_x + res_y
        if self.clc_x:
            return res_x
        if self.clc_y:
            return res_y



def d1t(var):
    return _DT(var)


def laplass(var):
    return _Laplass(var)


def d1dx(var):
    return _Grad(var, clc_x=1, clc_y=0)

def d1dy(var):
    return _Grad(var, clc_x=0, clc_y=1)

def stab_x(var):
    return _Stab(var, clc_x=1, clc_y=0)

def stab_y(var):
    return _Stab(var, clc_x=0, clc_y=1)

def d2dx(var):
    return _Laplass(var, clc_x=1, clc_y=0)

def d2dy(var):
    return _Laplass(var, clc_x=0, clc_y=1)

def grad(var):
    return _Grad(var, clc_x=1, clc_y=1)


def to_grid(arr):
    def coord_fo_idx(x, y):
        return x * 10 + y

    grid = np.zeros((10, 10), dtype=np.float64)
    for x_ in range(10):
        for y_ in range(10):
            grid[x_, y_] = arr[coord_fo_idx(x_, y_)]
    return grid


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
        t_val = 0
        for t in tqdm.trange(self.timesteps):
            dt.update()

            for i in range(len(all_vars)):
                all_vars[i].set_bound()

            t_val += dt.dt
            print(t_val)

            for equation in equation_system:
                left_part, _, right_part = equation
                left_part.solve(equation=right_part, dt=dt)

            for i in range(len(all_vars)):
                all_vars[i].set_bound()

            for i in range(len(all_vars)):
                all_vars[i].add_history()

        return [varr.get_history() for varr in all_vars]




