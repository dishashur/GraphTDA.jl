
include("GraphTDA.jl")
using SparseArrays
G = Symmetric(sprand(10,10,0.4))
lens = rand(10,7)
labels = [];



verbose = false
min_group_size = 5
max_split_size = 100
min_component_group = 5
overlap = 0.025

#To do a full analysis
gtda_obj = GraphTDA.analyzepredictions(G,lens,labels = labels,overlap = overlap,min_group_size=min_group_size,max_split_size=max_split_size,min_component_group=min_component_group)
reebgraph = GraphTDA.getreebgraph(gtda_obj)

#Just the Reeb graph
reebgraph = GraphTDA.getreebgraph(lens)