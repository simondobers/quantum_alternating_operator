
import os
from alternating_operator import filter_unique_paths
from helper import bitstring_to_path,cost
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use('seaborn')
algorithm = "alternating_operator" # "alternating_operator"  or "qaoa" 


with open(f".\\data\\{algorithm}_counts", "rb") as fp:   # Unpickling
    alternating_operator_counts = pickle.load(fp)

with open(".\\data\\G", "rb") as fp:   # Unpickling
    G = pickle.load(fp)


# get labels and costs of all measured paths
path_dic = {}
for shot in alternating_operator_counts:
    for bitstring,count in shot.items():
        path = bitstring_to_path(bitstring)

        # check if path valid 
        if path is not None:
            path_cost = round(cost(G,path),3)

            if path_cost not in path_dic:
                path_dic[path_cost] = bitstring

# sort dic by cost 
path_dic = {val:key for key,val in path_dic.items()}
path_dic = dict(sorted(path_dic.items(), key=lambda item: item[1]))


def format_counts_for_plot(path_dic,counts):
    sorted_counts = {}
    counts = filter_unique_paths(G,counts,unique_cost_dict=path_dic)
    cnt = 0
    for bitstring,_ in path_dic.items():
        if bitstring in counts:
            cnt += counts[bitstring]
            if bitstring not in sorted_counts:
                sorted_counts[bitstring] = counts[bitstring]/1024
            else : 
                sorted_counts[bitstring] += counts[bitstring]/1024

        else : 
            sorted_counts[bitstring] = 0
    print(cnt)
    return sorted_counts

# format all shots, such that they are ordered and can be plotted
for i,shot in enumerate(alternating_operator_counts):
    alternating_operator_counts[i] = format_counts_for_plot(path_dic,shot)


fig,ax = plt.subplots()
first = filter_unique_paths(G,alternating_operator_counts[0])

bar = ax.bar([bitstring_to_path(bitstring, return_as_string=True) for bitstring in first.keys()],first.values())
labels = [bitstring_to_path(bitstring, return_as_string=True) for bitstring in first.keys()]
ax.set_xticklabels(labels,rotation=45)   
bar.patches[0].set_color("g")
ax.set_ylim([0,1.])

def update_plot(step):

    data = alternating_operator_counts[step]
    ax.set_title(f"Iteration: {step}")
    for count, single_bar in zip(data.values(),bar.patches):
        single_bar.set_height(count)

    return

anim = animation.FuncAnimation(fig, update_plot,repeat=False, frames=len(alternating_operator_counts))
plt.show()


# save video 
now = datetime.datetime.now()
datetime_string = "%d-%m-%Y_%H-%M"
save_path = f"{os.getcwd()}\\animations\\{now.strftime(datetime_string)}_{algorithm}_{G.shape[0]}_cities.mp4"

print(f"saving to : {save_path}")
writervideo = animation.FFMpegWriter(fps=4) 
anim.save(save_path, writer=writervideo)