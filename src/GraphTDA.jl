module GraphTDA

#using Pkg; Pkg.add.(["JSON","Statistics", "SparseArrays", "DataStructures", "MatrixNetworks", "LinearAlgebra", "StatsBase", "JLD2"])

using SparseArrays, JSON, Statistics, DataStructures, MatrixNetworks, LinearAlgebra, JLD2, StatsBase
include("mainalg.jl")
export gnl, gtdagraph!, sGTDA, toy, canonicalize_graph, error_prediction!, smooth_lenses!


include("interface.jl")
export analyzepredictions, reeberrorsof, nodeerrorsof, reebcompositionof, nodecompositionof, reebgraphof, 
projectedgraphof, computetimeof, savereebs

end
