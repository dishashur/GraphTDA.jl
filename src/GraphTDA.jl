
module GraphTDA

include("GraphTDAAlgo.jl")
export gtdagraph!, sGTDA, toy, canonicalize_graph, error_prediction!, smooth_lenses!


include("GraphTDAinterface.jl")
export gnl, analyzepredictions, getreeberrors, getnodeerrors, getreebcomposition, getnodecomposition, getreebgraph, getprojectedgraph, getcomputetime

end