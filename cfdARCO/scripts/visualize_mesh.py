import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import subprocess
import json


def plot_mesh(mesh_nodes, values):
    template = """
    set terminal pngcairo size 400,400
    set output "image-rectangles.png"
    
    set palette maxcolors 1024
    set style fill transparent solid 0.9 noborder
    set xrange [0:1]
    set yrange [0:1]
    set cbrange [0:1024]
    set size ratio 1
    unset key
    
    {}
    
    plot 0
    """

    cmap_name = "autumn_r"
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=3)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)


    def get_rect_value(x1, x2, x3, x4, y1, y2, y3, y4, value):
        rgb = scalarMap.to_rgba(value)
        r = "{:02x}".format(int(rgb[0] * 255))
        g = "{:02x}".format(int(rgb[1] * 255))
        b = "{:02x}".format(int(rgb[2] * 255))
        rect_template = f'set object polygon from {x1},{y1} to {x2},{y2} to {x3},{y3} to {x4},{y4} to {x1},{y1} fc rgb "#{r}{g}{b}" fillstyle solid 1.0 border lt -1'
        return rect_template

    all_polys = []

    i = 0
    for ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) in mesh_nodes:
        all_polys.append(get_rect_value(x1, x2, x3, x4, y1, y2, y3, y4, values[i]))
        i += 1

    template = template.replace("{}", "\n".join(all_polys))
    with open("mesh.gnuplot", "w") as filee:
        filee.write(template)

    subprocess.run(["gnuplot", "mesh.gnuplot"])


if __name__ == '__main__':
    with open("/home/yevhen/Documents/cfdARCO/cfdARCO/dumps/strange_mesh2.json") as filee:
        mesh_json = json.load(filee)

    mesh = []
    values = []

    for node in mesh_json["nodes"]:
        node_repr = []
        for v_id in node["vertexes"]:
            node_repr.append(mesh_json["vertexes"][v_id])
        mesh.append(node_repr)
        values.append(.5)

    plot_mesh(mesh, values)
