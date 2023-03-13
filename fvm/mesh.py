from typing import List

import numpy as np


class Vertex2D:
    def __init__(self, x=0, y=0):
        self.id = 0
        self.coords = np.asarray([x, y], dtype=np.float64)

    def compute(self):
        pass

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    def __repr__(self):
        return f"Vertex2D({self.x}, {self.y})"


class Edge2D:
    def __init__(self, vertexes_1_idx=0, vertexes_2_idx=0):
        self.mesh = None
        self.id = 0
        self.vertexes_id = np.asarray([vertexes_1_idx, vertexes_2_idx])
        self.nodes_id = []
        self.normal = np.asarray([0, 0], dtype=np.float64)
        self.t = np.asarray([0, 0], dtype=np.float64)
        self.center_coords = np.asarray([0, 0], dtype=np.float64)
        self.area = 0

    def compute(self):
        p1_coords = self.mesh.vertexes[self.vertexes_id[0]].coords
        p2_coords = self.mesh.vertexes[self.vertexes_id[1]].coords
        self.center_coords = (p1_coords + p2_coords) / 2
        self.normal = np.asarray([
            -(p2_coords[1] - p1_coords[1]),
            (p2_coords[0] - p1_coords[0]),
        ], dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.t = np.asarray([
            (p2_coords[0] - p1_coords[0]),
            (p2_coords[1] - p1_coords[1]),
        ])
        self.t = self.t / np.linalg.norm(self.t)
        self.area = np.sqrt(np.sum((p2_coords - p1_coords) ** 2, dtype=np.float64), dtype=np.float64)

    def __repr__(self):
        return f"Edge2D(area={self.area}, p1=({self.mesh.vertexes[self.vertexes_id[0]].coords}), p2=({self.mesh.vertexes[self.vertexes_id[1]].coords}), is_bound={len(self.nodes_id) < 2})"


class Quadrangle2D:
    def __init__(self, e1=0, e2=0, e3=0, e4=0, v1=0, v2=0, v3=0, v4=0):
        self.mesh = None
        self.id = 0
        self.edges_id = np.asarray([e1,e2,e3,e4])
        self.vertexes_id = np.asarray([v1,v2,v3,v4])
        self.center_coords = np.asarray([0, 0], dtype=np.float64)
        self.vectors_in_edges_directions = np.zeros((4, 2), dtype=np.float64)
        self.normals = []
        self.volume = 0

    @property
    def x(self):
        return self.center_coords[0]

    @property
    def y(self):
        return self.center_coords[1]

    def compute(self):
        p1_coords = self.mesh.vertexes[self.vertexes_id[0]].coords
        p2_coords = self.mesh.vertexes[self.vertexes_id[1]].coords
        p3_coords = self.mesh.vertexes[self.vertexes_id[2]].coords
        p4_coords = self.mesh.vertexes[self.vertexes_id[3]].coords
        self.center_coords = (p1_coords + p2_coords + p3_coords + p4_coords) / 4
        for i in range(4):
            edge = self.mesh.edged[self.edges_id[i]]
            direction_vector = np.asarray([
                (edge.center_coords[0] - self.center_coords[0]),
                (edge.center_coords[1] - self.center_coords[1]),
            ], dtype=np.float64)
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            self.vectors_in_edges_directions[i] = direction_vector
        self.volume = 0.5 * (
            p1_coords[0] * p2_coords[1] + p2_coords[0] * p3_coords[1] + p3_coords[0] * p4_coords[1] + p4_coords[0] * p1_coords[1]
            -
            (p1_coords[1] * p2_coords[0] + p2_coords[1] * p3_coords[0] + p3_coords[1] * p4_coords[0] + p4_coords[1] * p1_coords[0] )
        )
        for i in range(len(self.vertexes_id) - 1):
            v0_id, v1_id = self.vertexes_id[i], self.vertexes_id[i+1]
            v0, v1 = self.mesh.vertexes[v0_id], self.mesh.vertexes[v1_id]
            dx = v1.x - v0.x
            dy = v1.y - v0.y
            self.normals.append(np.asarray([dy, -dx], dtype=np.float64))
        v0_id, v1_id = self.vertexes_id[-1], self.vertexes_id[0]
        v0, v1 = self.mesh.vertexes[v0_id], self.mesh.vertexes[v1_id]
        dx = v1.x - v0.x
        dy = v1.y - v0.y
        self.normals.append(np.asarray([dy, -dx], dtype=np.float64))


    def __repr__(self):
        return f"Quadrangle2D(volume={self.volume}, p1=({self.mesh.vertexes[self.vertexes_id[0]].coords}), p2=({self.mesh.vertexes[self.vertexes_id[1]].coords}), p3=({self.mesh.vertexes[self.vertexes_id[2]].coords}), p4=({self.mesh.vertexes[self.vertexes_id[3]].coords}))"

    def is_boundary(self):
        for edge_id in self.edges_id:
            edge = self.mesh.edged[edge_id]
            if len(edge.nodes_id) != 2:
                return True
        return False


class Quadrangle2DMesh:
    def __init__(self, x, y, lx, ly):
        self.num_nodes = x * y
        self.x = x
        self.y = y
        self.lx = lx
        self.ly = ly
        self.dx = lx / x
        self.dy = ly / y
        self.vertexes: List[Vertex2D] = []
        self.edged: List[Edge2D] = []
        self.nodes: List[Quadrangle2D] = []
        self.volumes: np.ndarray = np.zeros(self.num_nodes, dtype=np.float64)
        self._init_internals()

    def coord_fo_idx(self, x, y):
        return x * self.x + y

    def idx_to_coord(self, idx):
        return idx // self.x, idx % self.x

    def get_meshgrid(self):
        x, y = np.linspace(0, self.lx, self.x + 1, dtype=np.float64), np.linspace(0, self.ly, self.y + 1, dtype=np.float64)
        yv, xv = np.meshgrid(x, y, dtype=np.float64)
        for node in self.nodes:
            node_y, node_x = self.idx_to_coord(node.id)
            xv[node_x, node_y] = self.vertexes[node.vertexes_id[0]].coords[0]
            yv[node_x, node_y] = self.vertexes[node.vertexes_id[0]].coords[1]
            xv[node_x, node_y + 1] = self.vertexes[node.vertexes_id[1]].coords[0]
            yv[node_x, node_y + 1] = self.vertexes[node.vertexes_id[1]].coords[1]
            xv[node_x + 1, node_y + 1] = self.vertexes[node.vertexes_id[2]].coords[0]
            yv[node_x + 1, node_y + 1] = self.vertexes[node.vertexes_id[2]].coords[1]
            xv[node_x + 1, node_y] = self.vertexes[node.vertexes_id[3]].coords[0]
            yv[node_x + 1, node_y] = self.vertexes[node.vertexes_id[3]].coords[1]
        return yv, xv

    def _init_internals(self):
        for x_ in range(self.x):
            for y_ in range(self.y):
                i = self.coord_fo_idx(x_, y_)

                if i == 0:
                    v1, v2, v3, v4 = Vertex2D(0, 0), Vertex2D(self.dx, 0), Vertex2D(self.dx, self.dy), Vertex2D(0, self.dy)
                    for j, v in enumerate([v1, v2, v3, v4]):
                        v.id = len(self.vertexes)
                        self.vertexes.append(v)

                    e1, e2, e3, e4 = Edge2D(v1.id, v2.id), Edge2D(v2.id, v3.id), Edge2D(v3.id, v4.id), Edge2D(v4.id, v1.id)
                    for j, e in enumerate([e1, e2, e3, e4]):
                        e.id = len(self.edged)
                        self.edged.append(e)

                    n = Quadrangle2D(e1.id, e2.id, e3.id, e4.id, v1.id, v2.id, v3.id, v4.id)
                    n.id = len(self.nodes)
                    self.nodes.append(n)

                    for j, e in enumerate([e1, e2, e3, e4]):
                        if n.id not in e.nodes_id:
                            e.nodes_id.append(n.id)

                elif y_ == 0:
                    v2, v3 = Vertex2D((x_ + 1) * self.dx, 0), Vertex2D((x_ + 1) * self.dx, self.dy)
                    for j, v in enumerate([v2, v3]):
                        v.id = len(self.vertexes)
                        self.vertexes.append(v)

                    node_left_id = self.coord_fo_idx(x_ - 1, y_)
                    v1 = self.vertexes[self.nodes[node_left_id].vertexes_id[1]]
                    v4 = self.vertexes[self.nodes[node_left_id].vertexes_id[2]]
                    e1, e2, e3 = Edge2D(v1.id, v2.id), Edge2D(v2.id, v3.id), Edge2D(v3.id, v4.id)
                    for j, e in enumerate([e1, e2, e3]):
                        e.id = len(self.edged)
                        self.edged.append(e)

                    e4 = self.edged[self.nodes[node_left_id].edges_id[1]]
                    n = Quadrangle2D(e1.id, e2.id, e3.id, e4.id, v1.id, v2.id, v3.id, v4.id)
                    n.id = len(self.nodes)
                    self.nodes.append(n)

                    for j, e in enumerate([e1, e2, e3, e4]):
                        if n.id not in e.nodes_id:
                            e.nodes_id.append(n.id)

                        if len(e.nodes_id) > 2:
                            print("AAAA")

                elif x_ == 0:
                    v3, v4 = Vertex2D(self.dx, (y_ + 1) * self.dy), Vertex2D(0, (y_ + 1) * self.dy)
                    for j, v in enumerate([v3, v4]):
                        v.id = len(self.vertexes)
                        self.vertexes.append(v)

                    node_bottom_id = self.coord_fo_idx(x_, y_ - 1)
                    v1 = self.vertexes[self.nodes[node_bottom_id].vertexes_id[3]]
                    v2 = self.vertexes[self.nodes[node_bottom_id].vertexes_id[2]]
                    e2, e3, e4 = Edge2D(v2.id, v3.id), Edge2D(v3.id, v4.id), Edge2D(v4.id, v1.id)
                    for j, e in enumerate([e2, e3, e4]):
                        e.id = len(self.edged)
                        self.edged.append(e)

                    e1 = self.edged[self.nodes[node_bottom_id].edges_id[2]]
                    n = Quadrangle2D(e1.id, e2.id, e3.id, e4.id, v1.id, v2.id, v3.id, v4.id)
                    n.id = len(self.nodes)
                    self.nodes.append(n)

                    for j, e in enumerate([e1, e2, e3, e4]):
                        if n.id not in e.nodes_id:
                            e.nodes_id.append(n.id)

                        if len(e.nodes_id) > 2:
                            print("AAAA")

                else:
                    v3 = Vertex2D((x_+1)*self.dx, (y_+1)*self.dy)
                    v3.id = len(self.vertexes)
                    self.vertexes.append(v3)

                    node_left_id = self.coord_fo_idx(x_ - 1, y_)
                    node_bottom_id = self.coord_fo_idx(x_, y_ - 1)

                    v1 = self.vertexes[self.nodes[node_left_id].vertexes_id[1]]
                    v2 = self.vertexes[self.nodes[node_bottom_id].vertexes_id[2]]
                    v4 = self.vertexes[self.nodes[node_left_id].vertexes_id[2]]
                    e2, e3 = Edge2D(v2.id, v3.id), Edge2D(v3.id, v4.id)
                    for j, e in enumerate([e2, e3]):
                        e.id = len(self.edged)
                        self.edged.append(e)

                    e1 = self.edged[self.nodes[node_bottom_id].edges_id[2]]
                    e4 = self.edged[self.nodes[node_left_id].edges_id[1]]
                    n = Quadrangle2D(e1.id, e2.id, e3.id, e4.id, v1.id, v2.id, v3.id, v4.id)
                    n.id = len(self.nodes)
                    self.nodes.append(n)

                    for j, e in enumerate([e1, e2, e3, e4]):
                        if n.id not in e.nodes_id:
                            e.nodes_id.append(n.id)

                        if len(e.nodes_id) > 2:
                            print("AAAA")

                if len(self.edged[3].nodes_id) > 1:
                    print(self.edged[3].nodes_id)
                    print("Noooo")

    def visualize(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()

        vertexs = []
        for vertex in self.vertexes:
            if vertex.id not in vertexs:
                vertexs.append(vertex.id)
                G.add_node(vertex.id, pos=(vertex.x, vertex.y))

        print("vertexs = ", len(vertexs))

        colors = []
        edges = []
        for edge in self.edged:
            if edge.id not in edges:
                G.add_edge(edge.vertexes_id[0], edge.vertexes_id[1])
                edges.append(edge.id)
                if len(edge.nodes_id) == 1:
                    colors.append('blue')
                    print(edge.id)
                elif len(edge.nodes_id) == 2:
                    colors.append('black')
                else:
                    print("asdfv")

        print("edges = ", len(edges))
        print("colors = ", len(colors))

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_nodes(G, pos=pos)
        nx.draw_networkx_edges(G, edge_color=colors, pos=pos)
        plt.show()


    def compute(self):
        for list_entry in [self.vertexes, self.edged, self.nodes]:
            for elem in list_entry:
                elem.mesh = self
                elem.compute()
        for idx, node in enumerate(self.nodes):
            self.volumes[idx] = node.volume

    @staticmethod
    def _norm_idx_to_glob(node_idx, local_norm_idx):
        return node_idx * 4 + local_norm_idx



if __name__ == '__main__':
    nX = 60
    nY = 60
    lX = 60
    lY = 60
    mesh = Quadrangle2DMesh(nX, nY, lX, lY)
    last_dist = 2
    for x in range(nX):
        node_id = mesh.coord_fo_idx(x, 0)
        vrtx_id = mesh.vertexes[mesh.nodes[node_id].vertexes_id[1]].id
        mesh.vertexes[vrtx_id].coords[1] += (x + 1) * last_dist
        for y in range(nY):
            node_id = mesh.coord_fo_idx(x, y)
            vrtx_id = mesh.vertexes[mesh.nodes[node_id].vertexes_id[2]].id
            mesh.vertexes[vrtx_id].coords[1] += (x+1) * last_dist
            last_dist -= (1 / (nX + nY)) / (x+1)

    mesh.compute()
    mesh.visualize()

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # data = np.zeros((mesh.x, mesh.y))
    # for node in mesh.nodes:
    #     if node.is_boundary():
    #         data[int(node.center_coords[0]), int(node.center_coords[1])] = 99
    #     else:
    #         data[int(node.center_coords[0]), int(node.center_coords[1])] = -99
    # sns.heatmap(data, vmax=100, vmin=-100, cmap="crest")
    # plt.show()
