
include("GraphTDAAlgo.jl")



function _intlzrbgrph(X::Matrix{Float64};G = sparse([],[],[]),labels = [],labels_to_eval=[i for i=1:size(X,2)],knn=5,batch_size=256)
    if length(findnz(G)[3]) == 0
        G = canonicalize_graph(X,knn,batch_size)
    end
    return gnl(G,X,labels,labels_to_eval)
end


analyzepredictions(A::SparseMatrixCSC{Float64, Int64},X::Matrix{Float64};kwargs...) = analyzepredictions(X,G=A;kwargs...)

function analyzepredictions(X;G = sparse([],[],[]),trainlen=0,testlen=0,labels_to_eval = [i for i in range(1,size(X,2))],labels=[],
    extra_lens=nothing,alpha=0.5,batch_size=10000,known_nodes=nothing,knn=5,nsteps_preprocess=5,
    min_group_size = 5,max_split_size = 100,min_component_group = 5,overlap = 0.025,nsteps_mixing=10,
    is_merging=true,split_criteria="diff",split_thd=0,is_normalize=true,is_standardize=false,merge_thd=1.0,
    max_split_iters=200,max_merge_iters=10,degree_normalize_preprocess=1,degree_normalize_mixing=1,verbose=false)

    A = _intlzrbgrph(X,G=G,labels_to_eval=labels_to_eval,labels=labels,knn=knn,batch_size=batch_size)

    gtda = gtdagraph(A,max_split_size = max_split_size,overlap = overlap,min_group_size=min_group_size,min_component_group = min_component_group
     ,alpha = alpha,nsteps_preprocess=nsteps_preprocess,extra_lens=extra_lens,is_merging=is_merging,split_criteria=split_criteria,split_thd=split_thd,
     is_normalize=is_normalize,is_standardize=is_standardize,merge_thd=merge_thd,max_split_iters=max_split_iters,max_merge_iters=max_merge_iters,
     degree_normalize_preprocess=degree_normalize_preprocess,verbose=verbose)


    if length(labels) > 0
        train_nodes = [i for i in range(1,trainlen)]
        val_nodes = []
        test_nodes = [i for i in range(trainlen+1,trainlen+testlen)]
        train_mask = zeros(size(A.G,1))
        train_mask[train_nodes] .= 1
        val_mask = zeros(size(A.G,1))
        val_mask[val_nodes] .= 1
        test_mask = zeros(size(A.G,1))
        test_mask[test_nodes] .= 1

	error_prediction!(A,gtda,train_mask = train_mask,val_mask = val_mask,
        nsteps=nsteps_mixing,alpha = alpha,known_nodes=known_nodes,degree_normalize=degree_normalize_mixing)
        
	@info "prediction error" sum(gtda.sample_colors_mixing)

     else
         @info "Original labels is missing, unable to calculate prediction error"

     end

    return gtda
end


function getreeberrors(A::GraphTDA.sGTDA)
    if A.node_colors_class !== nothing
        return A.node_colors_class,A.node_colors_class_truth,A.node_colors_error,A.node_colors_uncertainty,A.node_colors_mixing
    else
        @info "No labels provided, unable to calculate prediction error"
    end
end

function getnodeerrors(A::GraphTDA.sGTDA)
    if A.sample_colors_mixing !== nothing
        return A.sample_colors_mixing, A.sample_colors_uncertainty, A.sample_colors_error
    else
        @info "No labels provided, unable to calculate prediction error"
    end

end

function getreebcomposition(A::GraphTDA.sGTDA)
    return A.reeb2node
end

function getnodecomposition(A::GraphTDA.sGTDA)
    return A.node2reeb
end

function getreebgraph(A::GraphTDA.sGTDA)
    return A.G_reeb
end

function getprojectedgraph(A::GraphTDA.sGTDA)
    return A.A_reeb
end

function getcomputetime(A::GraphTDA.sGTDA)
    return A.reebtime
end

