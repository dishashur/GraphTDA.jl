using .mainalg

"""
analyzepredictions
---
Initializes all the necessary structures for and executes algorithm 1 from the paper

Inputs
---
- Required: lens = x
- Optional: - graph= G
            - min_group_size (s1 in paper) : if the number of nodes comprising a reebnode is lesser than this, then that reebnode is rejected;
            used for filtering tiny reebnodes and merging reebnodes
            - max_split_size (K in paper) : if a component has more than these many nodes, its split further
(Rule of thumb for the above 2 is to start from [1,3] and then increase according to performance)
            - min_component_group (s2 in paper) : if the connected component in the reeb graph don't have at least these many reebnodes, that reebnode is rejected
(Rule of thumb dictates this as 1 so that all reebnodes are considered)
            - overlap : width of the overlapping gap to be considered
            - labels_to_eval
            - labels
            - batch_size
            - knn
Outputs
---
- gtda object for the final reeb graph with fields
    - A_reeb: Projected graph
    - G_reeb: Reeb graph
    - greeb_orig: Original reeb graph with possibly disconnected component
    - reeb2node: List of original nodes within each reebnode
    - node2reeb: List of reebnodes a particular node is a part of
    - reebtime: Time taken for computing the reeb graph
    - error measures calculated on the basis of provided labels
- the original labels (if labels are given), or a blank list
"""
analyzepredictions(X::Matrix{Tx},A::SparseMatrixCSC{Ta, Int64};kwargs...) where {Tx,Ta<:Any} = analyzepredictions(X,G=A;kwargs...) 

function analyzepredictions(X::Matrix{Tx};G = sparse([],[],[]),trainlen=0,testlen=0,labels_to_eval = [i for i in 1:size(X,2)],labels=[],
    extra_lens=nothing,alpha=0.5,batch_size=10000,known_nodes=nothing,knn=5,nsteps_preprocess=5,
    min_group_size = 5,max_split_size = 100,min_component_group = 5,overlap = 0.025,nsteps_mixing=10,
    is_merging=true,split_criteria="diff",split_thd=0,is_normalize=true,is_standardize=false,merge_thd=1.0,
    max_split_iters=200,max_merge_iters=10,degree_normalize_preprocess=1,degree_normalize_mixing=1,verbose=false) where {Tx}

    A = _intlzrbgrph(X,G=G,labels_to_eval=labels_to_eval,labels=labels,knn=knn,batch_size=batch_size)

    gtda = gtdagraph!(A,max_split_size = max_split_size,overlap = overlap,min_group_size=min_group_size,min_component_group = min_component_group
     ,alpha = alpha,nsteps_preprocess=nsteps_preprocess,extra_lens=extra_lens,is_merging=is_merging,split_criteria=split_criteria,split_thd=split_thd,
     is_normalize=is_normalize,is_standardize=is_standardize,merge_thd=merge_thd,max_split_iters=max_split_iters,max_merge_iters=max_merge_iters,
     degree_normalize_preprocess=degree_normalize_preprocess,verbose=verbose);


#NOTE: Charlie's contrived example of how you could split one function into two, where one modifies the loaded input.      
#     return A, gtda, analyzepredictions(A, gtda,args..)...
#end
#
#     function analyzepredictions!(A,gtda, args...)
#        return 

    if length(A.origlabels) > 0
        train_nodes = [i for i in range(1,trainlen)]
        val_nodes = []
        test_nodes = [i for i in range(trainlen+1,trainlen+testlen)]
        train_mask = zeros(size(A.G,1))
        train_mask[train_nodes] .= 1
        val_mask = zeros(size(A.G,1))
        val_mask[val_nodes] .= 1
        test_mask = zeros(size(A.G,1))
        test_mask[test_nodes] .= 1

	    error_prediction!(A,gtda,train_mask = train_mask,val_mask = val_mask,nsteps=nsteps_mixing,alpha = alpha,known_nodes=known_nodes,degree_normalize=degree_normalize_mixing)
        
	    @info "prediction error" sum(gtda.sample_colors_mixing)
    return gtda, A.labels
    else
        @info "Original labels are missing, unable to calculate prediction error"
	return gtda,[]
    end
end


function reeberrorsof(A::sGTDA)
    if A.node_colors_class !== nothing
        return A.node_colors_class,A.node_colors_class_truth,A.node_colors_error,A.node_colors_uncertainty,A.node_colors_mixing
    else
        @info "No labels provided, unable to calculate prediction error"
    end
end

function nodeerrorsof(A::sGTDA)
    if A.sample_colors_mixing !== nothing
        return A.sample_colors_mixing, A.sample_colors_uncertainty, A.sample_colors_error
        @show sum(A.sample_colors_mixing)
    else
        @info "No labels provided, unable to calculate prediction error"
    end

end

#reeb composition gives the number of nodes in comprising a reeb node
reebcompositionof(A::sGTDA) = A.reeb2node
reebcompositionof(X::Tx,A::SparseMatrixCSC{Ta, Int64};kwargs...) where {Tx,Ta<:Any} = reebcompositionof(X,G=A;kwargs...)
function reebcompositionof(X::Union{Matrix{Float64},Matrix{Float32}};G = sparse([],[],[]),kwargs...) 
    return reebcompositionof(analyzepredictions(X;G=G,kwargs...))
end


#node composition gives the reeb node indices that each node is a part of
nodecompositionof(A::sGTDA) = A.node2reeb
nodecompositionof(X::Tx,A::SparseMatrixCSC{Ta, Int64};kwargs...) where {Tx,Ta<:Any} = nodecompositionof(X,G=A;kwargs...)
function nodecompositionof(X::Union{Matrix{Float64},Matrix{Float32}};G = sparse([],[],[]),kwargs...) 
    return nodecompositionof(analyzepredictions(X;G=G,kwargs...))
end

#reeb graph is the graph made up of the reeb nodes
reebgraphof(A::sGTDA) = A.G_reeb
reebgraphof(X::Tx,A::SparseMatrixCSC{Ta, Int64};kwargs...) where {Tx,Ta<:Any} = reebgraphof(X,G=A;kwargs...)
function reebgraphof(X::Union{Matrix{Float64},Matrix{Float32}};G = sparse([],[],[]),kwargs...) 
    return reebgraphof(analyzepredictions(X;G=G,kwargs...))
end


#projected graph is the reeb graph expanded to the node view
projectedgraphof(A::sGTDA) = A.A_reeb
projectedgraphof(X::Tx,A::SparseMatrixCSC{Ta, Int64};kwargs...) where {Tx,Ta<:Any} = projectedgraphof(X,G=A;kwargs...)
function projectedgraphof(X::Union{Matrix{Float64},Matrix{Float32}};G = sparse([],[],[]),kwargs...) 
    return projectedgraphof(analyzepredictions(X;G=G,kwargs...))
end


computetimeof(A::sGTDA) = A.reebtime

function savereebs(A::sGTDA,filepath::String)
    x,y,z = findnz(A.greeb_orig)
    i,j,v = findnz(projectedgraphof(A))
    reebcomp = reebcompositionof(A)
    p,q,r = findnz(reebgraphof(A))
    if "sample_colors_mixing" in fieldnames(typeof(A)) 
    reeblabels = reeberrorsof(A)[1]
    reebtruelabels = reeberrorsof(A)[2]
    save("$filepath.jld2",Dict("reebgraph"=>(p,q,r),"reebcomps"=>reebcomp,"projected"=>(i,j,v),"greeb_orig"=>(x,y,z),"error"=>[A.sample_colors_mixing if "sample_colors_mixing" in fieldnames(typeof(A)) else [] end ],"reebcolors"=>reeblabels, "givencolors"=>reebtruelabels,"sample_colors_mixing"=> [A.sample_colors_mixing if "sample_colors_mixing" in fieldnames(typeof(A)) else [] end ]))
    else
    reeblabels = []
    reebtruelabels = []
    save("$filepath.jld2",Dict("reebgraph"=>(p,q,r),"reebcomps"=>reebcomp,"projected"=>(i,j,v),"greeb_orig"=>(x,y,z)))
    end
    
end
