# GraphTDA

[![Build Status](https://github.com/dishashur/GraphTDA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dishashur/GraphTDA.jl/actions/workflows/CI.yml?query=branch%3Amain)


This is a Julia based implementation for the framework introduced in [Topological Structure of Complex Prediction](https://arxiv.org/abs/2207.14358). 


GraphTDA for probing into the model and the dataset 
As pointed out in the paper, some data points close to those marked as Ankle Boots, have been predicted as something else - thus hinting at a problem in the model that generates the lens or the dataset itself.

<img src="https://github.com/dishashur/GraphTDA.jl/blob/main/src/jureebpredicted-Ankle_boot.png?raw=true" width="150" height="120">
<img src="https://github.com/dishashur/GraphTDA.jl/blob/main/src/jureeboriginal-Ankle_boot.png?raw=true" width="150" height="120">
<img src="https://github.com/dishashur/GraphTDA.jl/blob/main/src/juprojected-Ankle_boot.png?raw=true" width="150" height="120">

GraphTDA for biological developmental data

<figure>
  <img src="https://github.com/dishashur/GraphTDA.jl/blob/main/src/409b2_2000_400_1.png?raw=true" width="150" height="120">
  <figcaption>Reeb-projected graph for human brain organoids 409b2 development at 7 time points from [Attraction-Repulsion Spectrum in Neighbor Embeddings](https://github.com/berenslab/ne-spectrum). Notice the segregation of the later stages from the earlier stages</figcaption>
</figure>


<figure>
    <img src="https://github.com/dishashur/GraphTDA.jl/blob/main/src/2000_3500_1_drl.png?raw=true" width="150" height="120">
  <figcaption>Reeb-projected graph for zebrafish embryo development at 7 time points from [Attraction-Repulsion Spectrum in Neighbor Embeddings](https://github.com/berenslab/ne-spectrum)</figcaption>
</figure>

<figure>
    <img src="https://github.com/dishashur/GraphTDA.jl/blob/main/src/src/Ery.png?raw=true" width="150" height="120">
  <figcaption>Reeb-projected graph of hematopoietic development data in Mouse from [Transcriptional Heterogeneity and Lineage Commitment in Myeloid Progenitors.](https://pubmed.ncbi.nlm.nih.gov/26627738/)</figcaption>
</figure>




For X := Prediction matrix aka lens, G: Input graph (options), A: sGTDA object (output), this code offers the following different functions. 

1. analyzepredictions - Outputs the (1) prediction error if labels are provided, (2) the sGTDA object containing all the details of the structure
    1.1 analyzepredictions(X)
    1.2 analyzepredictions(X,G)
    1.3 analyzepredictions(X;G = sparse([],[],[]),trainlen=0,testlen=0,labels_to_eval = [i for i in range(1,size(X,2))],labels=[],
    extra_lens=nothing,alpha=0.5,batch_size=10000,known_nodes=nothing,knn=5,nsteps_preprocess=5,
    min_group_size = 5,max_split_size = 100,min_component_group = 5,overlap = 0.025,nsteps_mixing=10,
    is_merging=true,split_criteria="diff",split_thd=0,is_normalize=true,is_standardize=false,merge_thd=1.0,
    max_split_iters=200,max_merge_iters=10,degree_normalize_preprocess=1,degree_normalize_mixing=1,verbose=false)

2. reeberrorsof - Outputs node_colors_class: , 
                        node_colors_class_truth: ,
                        node_colors_error: ,
                        node_colors_uncertainty: ,
                        node_colors_mixing:

3. nodeerrorsof - Outputs sample_colors_mixing: Prediction error for each node,
                          sample_colors_uncertainty: uncertainty for each node ,
                          sample_colors_error: Error in nodes according to given prelabels

4. reebcompositionof - Outputs the list of nodes part of each reeb node
    4.1 reebcompositionof(A)
    4.2 reebcompositionof(X)
    4.3 reebcompositionof(X,G)

5. nodecompositionof - Outputs the list of reeb nodes that include each node
    5.1 nodecompositionof(A)
    5.2 nodecompositionof(X)
    5.3 nodecompositionof(X,G)

6. reebgraphof - Outputs the reebgaph
    6.1 reebgraphof(A)
    6.2 reebgraphof(X)
    6.3 reebgraphof(X,G)

7. projectedgraphof - Outputs the connectivity of the original nodes according to the reebgraph
    7.1 projectedgraphof(A)
    7.2 projectedgraphof(X)
    7.3 projectedgraphof(X,G)

8. computetimeof(A) - Outputs the compute time of the sGTDA object A

9. savereebs(A,filepath) - Saves the reeb composition and the adjacency matrices of the reeb graph and the projected graph in (i,j,v) format
        Eg: ``` using Delimited File, SparseArrays
                myreeb = load("savedreeb.jld2")
                reebgraph = sparse(myreeb["reebgraph"][1],myreeb["reebgraph"][2],myreeb["reebgraph"][3])
                projectedgraph = sparse(myreeb["projected"][1],myreeb["projected"][2],myreeb["projected"][3])
                reebcomponents = myreeb["reebcomps"]
             ```
            

10. gnl - Immutable structure with fields as follows
    gnl.G = A
    gnl.preds = X
    gnl.origlabels 
    gnl.labels 
    gnl.labels_to_eval

11. gtdagraph! - Outputs the sGTDA object containing all the details of the structure sGTDA described below after calculating the reeb graph. Required argument: gnl object

12. sGTDA - Mutable structure with fields as follows
            A_reeb::SparseMatrixCSC{Float64, Int64}
            G_reeb::SparseMatrixCSC{Float64, Int64}
            greeb_orig::SparseMatrixCSC{Float64, Int64}
            reeb2node::Vector{Vector{Int64}}
            node2reeb::DefaultDict{Int64, Vector{Float64}, DataType}
            reebtime::Float64
        
            node_colors_class::Matrix{Float64}
            node_colors_class_truth::Matrix{Float64}
            node_colors_error::Vector{Float64}
            node_colors_uncertainty::Vector{Float64}
            node_colors_mixing::Vector{Float64}
            sample_colors_mixing::Vector{Float64}
            sample_colors_uncertainty::Matrix{Float64}
            sample_colors_error::Vector{Float64}

13. canonicalize_graph - Outputs the adjacency matrix from the prediction matrix if the input graph is missing. Reuired arguments are the prediction matrix, number of nearest neighbors and batch size.

14. error_prediction! - Updates the sGTDA object with the error values and returns them in the following order
    node_colors_class, node_colors_class_truth, node_colors_error, node_colors_uncertainty, node_colors_mixing,
    sample_colors_mixing, sample_colors_error, sample_colors_uncertainty

15. smooth_lenses! - Smoothen the lenses according to the description in the paper
    15.1 smooth_lenses!(X)
    15.2 smooth_lenses!(X,G)
    15.3 smooth_lenses!(X,G,labelsto_eval)
    15.4 smooth_lenses!(gnlobject)



This code can be used to analyze embeddings from 3 different procedures : -

1. Diffusion - analyzepredictions(G,X;kwargs...) 
    _Required_ arguments:
    - **G**: The graph structure. We assume this is the adjacency matrix as a sparse CSC matrix that represents an undirected graph. 
    - **X**: The lenses with each lens in a column as a matrix type. These are the embedding vectors obtained from the procedure.
    - **labels**: This is necessary only if the procedure is attempting to analyze predictions; by default, it is an empty list.

    _Optional_ arguments:
    - **labels_to_eval**: The labels/embedding vectors that need to be evaluated. This is the number of coloumns by default.
    - **max_split_size**: This is the largest number of nodes considered inside a reebnode.
    - **overlap**: This values decides the closeness in prediction for data nodes to be considered a part of the same reebnode.
    - **node_size_thd**: This is the smallest number of nodes a reebnode needs to include.
    - **split_thd**: This is the smallest difference between the probabilities the data nodes should have to be considered with different reebnodes.
    - **component_size_thd**: This is the minimum number of original data a component should include to be considered as a reebnode.


2. Neural Networks - analyzepredictions(X;kwargs...) 
     _Required_ arguments:
    - **X**: These are theembedding vectors returned by the neural network.
    - **trainlen** : Number of train examples
    - **testlen** : Number of test examples
    - **labels**: This is necessary only if the procedure is attempting to analyze the error in predictions; by default, it is an empty list.

    _Optional_ arguments:
    - **G**: The graph structure. We assume this is the adjacency matrix as a sparse CSC matrix that represents an undirected graph. 
    - **Same** as in analyzepredictions() 

3. Graph Neural Networks - analyzepredictions(X;kwargs...) 
     _Required_ arguments:
    - **X**: These are the embedding vectors returned by the graph neural network.
    - **trainlen** : Number of train examples
    - **testlen** : Number of test examples
    - **labels**: This is necessary only if the procedure is attempting to analyze the error in predictions; by default, it is an empty list.

    _Optional_ arguments:
    - **G**: The graph structure. We assume this is the adjacency matrix as a sparse CSC matrix that represents an undirected graph. 
    - **Same** as in analyzepredictions() 


