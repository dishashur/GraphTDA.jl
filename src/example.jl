
include("GraphTDA.jl")
using NPZ, SparseArrays
G = sparse(npzread("mnist_test_G.npy"));
U = npzread("mnist_test_U.npy");
labels = npzread("y.npy");
preds = U[:,1:100];


verbose = false
min_group_size = 5
max_split_size = 100
min_component_group = 5
overlap = 0.025

k = GraphTDA.analyzepredictions(G,preds,labels = labels,overlap = overlap,min_group_size=min_group_size,max_split_size=max_split_size,min_component_group=min_component_group)

timetaken = GraphTDA.getcomputetime(k)