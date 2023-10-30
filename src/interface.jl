

using .mainalg

analyzepredictions(X::Union{Matrix{Float64},Matrix{Float32}},A::SparseMatrixCSC{Float64, Int64};kwargs...) = analyzepredictions(X,G=A;kwargs...)

function analyzepredictions(X;G = sparse([],[],[]),trainlen=0,testlen=0,labels_to_eval = [i for i in range(1,size(X,2))],labels=[],
    extra_lens=nothing,alpha=0.5,batch_size=10000,known_nodes=nothing,knn=5,nsteps_preprocess=5,
    min_group_size = 5,max_split_size = 100,min_component_group = 5,overlap = 0.025,nsteps_mixing=10,
    is_merging=true,split_criteria="diff",split_thd=0,is_normalize=true,is_standardize=false,merge_thd=1.0,
    max_split_iters=200,max_merge_iters=10,degree_normalize_preprocess=1,degree_normalize_mixing=1,verbose=false)

    A = _intlzrbgrph(X,G=G,labels_to_eval=labels_to_eval,labels=labels,knn=knn,batch_size=batch_size)

    gtda = gtdagraph!(A,max_split_size = max_split_size,overlap = overlap,min_group_size=min_group_size,min_component_group = min_component_group
     ,alpha = alpha,nsteps_preprocess=nsteps_preprocess,extra_lens=extra_lens,is_merging=is_merging,split_criteria=split_criteria,split_thd=split_thd,
     is_normalize=is_normalize,is_standardize=is_standardize,merge_thd=merge_thd,max_split_iters=max_split_iters,max_merge_iters=max_merge_iters,
     degree_normalize_preprocess=degree_normalize_preprocess,verbose=verbose)


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
        return gtda
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
    else
        @info "No labels provided, unable to calculate prediction error"
    end

end

#reeb composition gives the number of nodes in comprising a reeb node
reebcompositionof(A::sGTDA) = A.reeb2node
reebcompositionof(X::Union{Matrix{Float64},Matrix{Float32}},A::SparseMatrixCSC{Float64, Int64};kwargs...) = reebcompositionof(X,G=A;kwargs...)
function reebcompositionof(X::Union{Matrix{Float64},Matrix{Float32}};G = sparse([],[],[]),kwargs...) 
    return reebcompositionof(analyzepredictions(X;G=G,kwargs...))
end


#node composition gives the reeb node indices that each node is a part of
nodecompositionof(A::sGTDA) = A.node2reeb
nodecompositionof(X::Union{Matrix{Float64},Matrix{Float32}},A::SparseMatrixCSC{Float64, Int64};kwargs...) = nodecompositionof(X,G=A;kwargs...)
function nodecompositionof(X::Union{Matrix{Float64},Matrix{Float32}};G = sparse([],[],[]),kwargs...) 
    return nodecompositionof(analyzepredictions(X;G=G,kwargs...))
end

#reeb graph is the graph made up of the reeb nodes
reebgraphof(A::sGTDA) = A.G_reeb
reebgraphof(X::Union{Matrix{Float64},Matrix{Float32}},A::SparseMatrixCSC{Float64, Int64};kwargs...) = reebgraphof(X,G=A;kwargs...)
function reebgraphof(X::Union{Matrix{Float64},Matrix{Float32}};G = sparse([],[],[]),kwargs...) 
    return reebgraphof(analyzepredictions(X;G=G,kwargs...))
end


#projected graph is the reeb graph expanded to the node view
projectedgraphof(A::sGTDA) = A.A_reeb
projectedgraphof(X::Union{Matrix{Float64},Matrix{Float32}},A::SparseMatrixCSC{Float64, Int64};kwargs...) = projectedgraphof(X,G=A;kwargs...)
function projectedgraphof(X::Union{Matrix{Float64},Matrix{Float32}};G = sparse([],[],[]),kwargs...) 
    return projectedgraphof(analyzepredictions(X;G=G,kwargs...))
end


computetimeof(A::sGTDA) = A.reebtime

function savereebs(A::sGTDA,filepath::String)
    i,j,v = findnz(projectedgraphof(A))
    reebcomp = reebcompositionof(A)
    p,q,r = findnz(reebgraphof(A))
    save("$filepath.jld2",Dict("reebgraph"=>(p,q,r),"reebcomps"=>reebcomp,"projected"=>(i,j,v)))
end
