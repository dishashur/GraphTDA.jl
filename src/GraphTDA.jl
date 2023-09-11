

module GraphTDA

using SparseArrays, Statistics, DataStructures, MatrixNetworks, Plots, LinearAlgebra, JLD2, StatsBase
include("mainalg.jl")
export gnl, gtdagraph!, sGTDA, toy, canonicalize_graph, error_prediction!, smooth_lenses!


include("interface.jl")
export analyzepredictions, getreeberrors, getnodeerrors, getreebcomposition, getnodecomposition, getreebgraph, getprojectedgraph, getcomputetime, savereebs

end
