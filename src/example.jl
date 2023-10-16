
include("GraphTDA.jl")
using SparseArrays, LinearAlgebra, StableRNGs
n = 1000
G = sprand(StableRNG(1), n,n,15/n)
fill!(A.nzval, 1)
G = max.(G,G')
G = G - Diagonal(G); 
##
lens = rand(n, 5);


verbose = false
min_group_size = 5
max_split_size = 100
min_component_group = 5
overlap = 0.025

#To do a full analysis
gtda_obj = GraphTDA.analyzepredictions(lens,G,labels = labels,overlap = overlap,min_group_size=min_group_size,max_split_size=max_split_size,min_component_group=min_component_group)
reebgraph = GraphTDA.reebgraphof(gtda_obj)

#Just the Reeb graph
reebgraph = GraphTDA.reebgraphof(lens)