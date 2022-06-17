
from itertools import count
from helper import create_tsp_graph,plot_tsp_graph,bitstring_to_path,cost
from classical import solve_tsp_classical
from alternating_operator import get_expectation,analyse_result,create_qaoa_circ,filter_unique_paths
from scipy.optimize import minimize
from qiskit.tools.visualization import plot_histogram
from qiskit import transpile,Aer
from qaoa import get_expectation_qaoa,create_classical_qaoa_circ
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open(".\\data\\alternating_operator_counts", "rb") as fp:   # Unpickling
    alternating_operator_counts = pickle.load(fp)

with open(".\\data\\G", "rb") as fp:   # Unpickling
    G = pickle.load(fp)


# get labels and costs of all measured paths
path_dic = {}
for shot in alternating_operator_counts:
    for bitstring,count in shot.items():
        if bitstring not in path_dic:
            path_dic[bitstring] = cost(G,bitstring_to_path(bitstring))

# sort dic by cost 
path_dic = filter_unique_paths(G,dict(sorted(path_dic.items(), key=lambda item: item[1])))

def format_counts_for_plot(path_dic,counts):
    sorted_counts = {}
    for bitstring,_ in path_dic.items():
        if bitstring in counts:
            sorted_counts[bitstring] = counts[bitstring]
        else : 
            sorted_counts[bitstring] = 0

    return sorted_counts

# format all shots, such that they are ordered and can be plotted
for i,shot in enumerate(alternating_operator_counts):
    alternating_operator_counts[i] = format_counts_for_plot(path_dic,shot)



fig,ax = plt.subplots()
first = filter_unique_paths(G,alternating_operator_counts[0])

bar = ax.bar([bitstring_to_path(bitstring, return_as_string=True) for bitstring in first.keys()],first.values())
labels = [bitstring_to_path(bitstring, return_as_string=True) for bitstring in first.keys()]
ax.set_xticklabels(labels,rotation=60)   
bar.patches[0].set_color("r")


def update_plot(step):

    data = filter_unique_paths(G,alternating_operator_counts[step])
    ax.set_title(f"Iteration: {step}")
    for count, single_bar in zip(data.values(),bar.patches):
        single_bar.set_height(count)

    return

anim = animation.FuncAnimation(fig, update_plot,repeat=False, frames=50)
plt.show()


# save video 
now = datetime.datetime.now()
datetime_string = "%d-%m-%Y_%H-%M"
save_path = f"C:\\Users\\t-sdobers\\Desktop\\quantum_alternating_operator\\animations\\{now.strftime(datetime_string)}_{G.shape[0]}_cities.mp4"

print(f"saving to : {save_path}")
writervideo = animation.FFMpegWriter(fps=60) 
anim.save(save_path, writer=writervideo)