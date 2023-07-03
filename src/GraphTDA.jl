module GraphTDA
using SparseArrays, Statistics, DataStructures, MatrixNetworks, Plots, LinearAlgebra, JLD2, StatsBase

mutable struct sGTDA
     G::SparseMatrixCSC{Float64, Int64}
     preds::Matrix{Float64} #this is the embedding vector/lens
     labels::Vector{Any} 
     labels_to_eval::Vector{Int64} 
     pre_lbs::Vector{Float64}
     pre_ubs::Vector{Float64}
     bin_sizes::Vector{Float64}
     lbs::Matrix{Float64}
     ubs::Matrix{Float64}
     component_records_all::Dict{Any, Any}
     final_components_all::Dict{Any, Any}
     component_records::Dict{Any, Any}
     final_components::Dict{Any, Any}
     final_components_unique::Dict{Any, Any}
     component_counts::Vector{Int64}
     split_lens::Dict{Any, Any}
     component_id_map::DefaultDict{Int64, Vector{Float64}, DataType}
     node_assignments::Vector{Vector{Int}}
     node_assignments_tiny_components::Vector{Vector{Int}}
     final_components_filtered::Dict{Any, Any}
     final_components_removed::Dict{Any, Any}
     edges_to_merge::Vector{Int}
     A_reeb::SparseMatrixCSC{Float64, Int64}
     G_reeb::SparseMatrixCSC{Float64, Int64}
     An::SparseMatrixCSC{Float64, Int64}
     total_mixing_all::Matrix{Float64}
     filtered_nodes::Vector{Int64}
     node_colors_class::Matrix{Float64}
     node_colors_class_truth::Matrix{Float64}
     node_colors_error::Vector{Float64}
     node_colors_uncertainty::Vector{Float64}
     node_colors_mixing::Vector{Float64}
     sample_colors_mixing::Vector{Float64}
     sample_colors_uncertainty::Matrix{Float64}
     sample_colors_error::Vector{Float64}
     reeb2node::DefaultDict{Int64, Vector{Float64}, DataType}
     node2reeb::DefaultDict{Int64, Vector{Float64}, DataType}
     sGTDA() = new()
end

function toy()
    return 5.0
end

function feat_sim(features,train_features,k)
    similarity = features*train_features
    indices = [sortperm(similarity[i,:],rev=true)[1:k] for i=1:size(similarity,1)]
    distances = [similarity[i,j] for (i,j) in enumerate(indices)]
    return distances, indices
end

function feat_sim_batched(features,train_features,k,batch_size)
    overall_distances = nothing
    overall_indices = nothing
    for i in Vector(1:batch_size:size(train_features,1))
        si = i
        ei = min(si+batch_size-1,size(train_features,1))
        train_features_batch = train_features[si:ei,:]'
        similarity = features*train_features_batch
        indices = [sortperm(similarity[i,:],rev=true)[1:k] for i=1:size(similarity,1)]
        distances = [similarity[i,j] for (i,j) in enumerate(indices)]
       
        @show indices = indices .+ si - 1
        if overall_distances === nothing
            overall_distances = distances
            overall_indices = indices
        else
            tmp_distances = cat(overall_distances,distances,dims=2)
            tmp_indices = cat(overall_indices,indices,dims=2)
            new_indices = [sortperm(tmp_distances[i,:],rev=true)[1:k] for i=1:size(tmp_distances,1)]
            overall_distances = [tmp_distances[i,j] for (i,j) in enumerate(new_indices)]
            overall_indices = [tmp_indices[i,j] for (i,j) in enumerate(new_indices)]
        end
     end
    return overall_distances, overall_indices
end


function canonicalize_graph(X,knn,batch_size;thd=0,batch_training=false,batch_size_training=50000)
    ei,ej = [],[]
    for i in Vector(1:batch_size:size(X,1))
          start_i = i
          end_i = min(start_i+batch_size-1,size(X,1))	
	  
	  if batch_training
               distances,batch_indices = feat_sim_batched(X[start_i:end_i,:],X,knn+1,batch_size_training)
          else
               distances,batch_indices = feat_sim(X[start_i:end_i,:],transpose(X),knn+1)
          end
          for xi in range(1,size(batch_indices,1))
               cnt = 0
               for (j,xj) in enumerate(batch_indices[xi])
                    if ((xi-1+start_i != xj) && (distances[xi][j] >= thd) && (cnt < knn))
                         append!(ei,xi-1+start_i)
                         append!(ej,xj)
                         cnt += 1
                    end
               end
          end
        distances = nothing
        batch_indices = nothing
     end
    n = size(X,1)
    A_knn = sparse(ei,ej,ones(length(ei)),n,n)
    A_knn = make_graph_symmetric(A_knn,n)
    return A_knn
end

function _is_overlap(x,y)
     for i=1:length(x)
          xi = x[i]
          yi = y[i]
          if (xi[1] >= yi[1] && xi[1] <= yi[2]) || (yi[1] >= xi[1] && yi[1] <= xi[2])
              continue
          else
              return false
          end
     end
     return true
end

#_function: internal usage - not to be exported

function _compute_bin_lbs(A::sGTDA,inner_id,overlap,col_id,nbins)
    if (inner_id > nbins) || (inner_id < 1)
          return Inf
     end
     curr_lb = A.pre_lbs[col_id]+A.bin_sizes[col_id]*(inner_id-1)
     if inner_id != 1
          curr_lb -= overlap*A.bin_sizes[col_id]
     end
     return curr_lb
end

function _compute_bin_ubs(A::sGTDA,inner_id,overlap,col_id,nbins)
     if (inner_id > nbins) || (inner_id < 1)
          return -Inf
     end
     curr_ub = A.pre_lbs[col_id]+A.bin_sizes[col_id]*inner_id
     if inner_id != nbins
          curr_ub += overlap*A.bin_sizes[col_id]
     end
     return curr_ub
end


 

function smooth_lenses!(A::sGTDA;alpha=0.5,nsteps=3,normalize=true,extra_lens=nothing,standardize=false,degree_normalize=1,verbose=false)
     @info "Preprocessing lens"
     Ar = A.G .> 0
     Ar = Float64.(Ar)
     degs = vec(sum(Ar,dims=1))
     dinv = 1 ./ degs
     dinv[dinv .== Inf] .= 0
     Dinv = SparseArrays.spdiagm(size(Ar,1),size(Ar,1),dinv)
     if degree_normalize == 1
          An = Dinv*Ar
     elseif degree_normalize == 2
          An = Ar*Dinv
     elseif degree_normalize == 3
          An = (sqrt.(Dinv))*Ar*(sqrt.(Dinv))
     else
          An = Ar
     end
     total_mixing_all = copy(A.preds)
     init_mixing = copy(A.preds)
     if extra_lens !== nothing
          total_mixing_all = hcat(total_mixing_all,extra_lens)
          init_mixing = hcat(init_mixing,extra_lens)
     end
     for _ in range(1,nsteps)
         total_mixing_all = (1-alpha)*init_mixing + alpha*(An*total_mixing_all)
     end
     selected_col = A.labels_to_eval
     if extra_lens !== nothing
          selected_col += [x for x in range(A.preds.size[2],total_mixing_all.size[2])] 
     end
     M = copy(total_mixing_all[:,selected_col])
     if standardize
          for i in range(1,size(M,2))
               M[:,i] = (M[:,i]-mean(M[:,i]))/std(M[:,i])
          end
     end
     if normalize
          for i in range(1,size(M,2))
               if maximum(M[:,i]) != minimum(M[:,i])
                    M[:,i] = (M[:,i] .- minimum(M[:,i]))./(maximum(M[:,i])-minimum(M[:,i]))
               end
          end
     end
     return M,Ar
end

function _clustering_single_col_pyramid(A::sGTDA,M,filter_cols,nbins;lbs=nothing,ubs=nothing)
     A.pre_lbs = zeros(length(filter_cols))
     A.pre_ubs = zeros(length(filter_cols))
     A.bin_sizes = zeros(length(filter_cols))
     if lbs != nothing
         A.lbs = lbs
     else
          A.lbs = minimum(M,dims = 1)
     end
     if ubs != nothing
         A.ubs = ubs
     else
         A.ubs = maximum(M,dims = 1)
     end
     for (i,col) in enumerate(filter_cols)
          A.pre_lbs[i] = A.lbs[col]
          A.pre_ubs[i] = A.ubs[col]
          A.bin_sizes[i] = (A.ubs[col]-A.lbs[col])/nbins
     end
end

function _find_bins_pyramid(A::sGTDA,M,filter_cols,overlap,nbins)
        bin_nums = 1
        bin_key_map = Dict()
        bin_map = Dict()
        bins = DefaultDict{Int,Vector{Float64}}(Vector{Float64})
        all_assignments = [[[] for _ in range(1,length(filter_cols))] for _ in range(1,size(M,1))]
	for (j,col) in enumerate(filter_cols)
          bin_size = A.bin_sizes[j]
          inner_id = Int64.(floor.((M[:,col] .- A.pre_lbs[j]) ./ bin_size)) .+ 1
          boundary = [i for i in range(1,length(M[:,col])) if M[i,col] ==  A.pre_ubs[j]]
          inner_id[boundary] .= nbins
          bin_ids = inner_id
          for (i,bin_id) in enumerate(bin_ids)
               append!(all_assignments[i][j],bin_id)
          end
          bin_lbs = []
          bin_ubs = []
          for t in range(1,nbins)
               append!(bin_lbs,_compute_bin_lbs(A,t,overlap[1],j,nbins))
               append!(bin_ubs,_compute_bin_ubs(A,t,overlap[2],j,nbins))
          end
          new_inner_id = inner_id .+ 1
          valid_ids = [i for i=1:length(new_inner_id) if (new_inner_id[i]>=1)&&(new_inner_id[i]<=nbins)]
          valid_bin_ids = new_inner_id[valid_ids]
          filtered_ids = [i for i=1:length(valid_ids) if M[valid_ids[i],col] >= bin_lbs[valid_bin_ids[i]]]
	     valid_ids = valid_ids[filtered_ids]
          valid_bin_ids = valid_bin_ids[filtered_ids]
          for (i,valid_id) in enumerate(valid_ids)
               append!(all_assignments[valid_id][j],valid_bin_ids[i])
          end
          new_inner_id = inner_id .- 1
          valid_ids = [i for i=1:length(new_inner_id) if (new_inner_id[i]>=1)&&(new_inner_id[i]<=nbins)]
	     valid_bin_ids = new_inner_id[valid_ids]
	     filtered_ids = [i for i=1:length(valid_ids) if M[valid_ids[i],col] <= bin_ubs[valid_bin_ids[i]]]
	     valid_ids = valid_ids[filtered_ids]
          valid_bin_ids = valid_bin_ids[filtered_ids]
          for (i,valid_id) in enumerate(valid_ids)
               append!(all_assignments[valid_id][j],valid_bin_ids[i])
          end
     end
	
     for i=1:size(M,1)
          for (_,bin_key) in enumerate(all_assignments[i]...)
	    	     if !(bin_key in keys(bin_key_map))
                    bin_key_map[bin_key] = bin_nums
                    bin_map[bin_nums] = bin_key
                    push!(bins[bin_nums],i)
                    bin_nums += 1
               else
                    assigned_id = bin_key_map[bin_key]
                    push!(bins[assigned_id],i)
               end
          end
     end
     return bins,bin_map
end


function filtering(A::sGTDA,M,filter_cols;nbins=2,overlap=(0.05,0.05),kwargs...)
     _clustering_single_col_pyramid(A,M,filter_cols,nbins;kwargs...)
     return _find_bins_pyramid(A,M,filter_cols,overlap,nbins)
end

function find_components(A::SparseMatrixCSC{Float64,Int64};size_thd=100,verbose=false)
    if (nnz(A) == 0) && (size_thd == 0)
        return [i for i=1:size(A,1)], [[i] for i=1:size(A,1)]
    end
    sc = scomponents(A)
    components = [[] for _ in range(1,sc.number)]
    for (i,l) in enumerate(sc.map)
    	append!(components[l],i)
    end
    selected_components = [c for c in components if length(c)>size_thd]
    selected_nodes = [j for c in selected_components for j in c]
    sort!(unique!(selected_nodes))
    return selected_nodes, selected_components
end

function graph_clustering(G,bins;component_size_thd=10)
     graph_clusters = DefaultDict{Int64,Vector{Vector{Int64}}}([])
     for key in keys(bins)
          points = Int64.(bins[key])
          if length(points) < component_size_thd
               continue
          end
          Gr = copy(G[points,:][:,points])
          _,components = find_components(Gr,size_thd = component_size_thd) 
	     for component in components
	       push!(graph_clusters[key],[points[node] for node in component])
          end
     end    
     return graph_clusters
end

function grouping(A::sGTDA,Ar,M,slice_columns,curr_level)
     all_G_sub = []
     all_M_sub = []
     this1 = time()
     for component_id in curr_level
          component = A.component_records[component_id]
          G_sub = copy(Ar[component,:][:,component])
          if slice_columns
               M_sub = copy(M[component,:][:,filter_cols])
          else
               M_sub = copy(M[component,:])
          end
          append!(all_G_sub,[G_sub])
          append!(all_M_sub,[M_sub])
     end 
     groupingtime = time() - this1
     @info "Grouping took  " groupingtime
     return all_G_sub, all_M_sub
end

function find_reeb_nodes!(A::sGTDA,M,Ar;filter_cols=nothing,nbins_pyramid=2,overlap=(0.5,0.5),smallest_component=50,component_size_thd=0,split_criteria="diff",split_thd=0.01,max_iters=50,verbose=false)
	A.component_counts = []
     _,components = find_components(Ar,size_thd=0)
     @info "Initial number of components" length(components)
     curr_level = []
     num_final_components = 0
     num_total_components = 0
     A.component_records = Dict{Int64,Int64}()
     A.final_components =  Dict{Int64,Int64}()
     A.split_lens = Dict{Int64,Int64}()
     A.component_id_map = DefaultDict{Int,Vector{Float64}}(Vector{Float64})

     if typeof(overlap) !== Tuple{Float64, Float64}
          @assert(overlap > 0)
          @assert(overlap < 1)
          overlap = (0,overlap)
     end

     for component in components
          A.component_records[num_total_components] = component
          if length(component) > smallest_component
             append!(curr_level,num_total_components)
          else
              A.final_components[num_final_components] = component
              num_final_components += 1
          end     
          num_total_components += 1
     end
     append!(A.component_counts,length(A.component_records))
     iters = 0
     function worker(M_sub,G_sub,component_id,nbins_pyramid,overlap,component_size_thd,split_thd)
          if split_criteria == "std"
               diffs = std(M_sub, dims = 1)
          else
               diffs = maximum(M_sub,dims=1) - minimum(M_sub,dims=1)
          end
          col_to_filter = argmax(diffs)[2]
          largest_diff = diffs[col_to_filter]
          if largest_diff < split_thd
               return [i for i in range(1,size(M_sub,1))],component_id,largest_diff,col_to_filter
          else
	  	        bins,_ = filtering(A,M_sub,[col_to_filter],nbins=nbins_pyramid,overlap=overlap)
		        graph_clusters = graph_clustering(G_sub,bins,component_size_thd=component_size_thd)
          end
          return graph_clusters,component_id,largest_diff,col_to_filter
     end
     slice_columns = true
     if length(setdiff([i for i in range(1,size(M,2))],filter_cols)) == 0
          slice_columns = false
     end
     splitstart = time()
     while (length(curr_level) > 0) && (iters < max_iters) 
          iters += 1
          if verbose
               @info "Iteration " iters
               @info "Components to split" length(curr_level) 
          end
          sizes = []
          new_level = []
          all_G_sub, all_M_sub = grouping(A,Ar,M,slice_columns,curr_level)
          process_order = sort([(-1*size(all_M_sub[i],1),i) for i in range(1,length(all_M_sub))])
          min_largest_diff = Float64(Inf)
          max_largest_diff = -1*Float64(Inf)
          for (_,i) in process_order
               graph_clusters,component_id,largest_diff,col_to_filter = worker(all_M_sub[i],all_G_sub[i],curr_level[i],nbins_pyramid,overlap,component_size_thd,split_thd)
	          min_largest_diff = min(min_largest_diff,largest_diff)
               max_largest_diff = max(max_largest_diff,largest_diff)
               A.split_lens[component_id] = col_to_filter
               component = A.component_records[component_id]

               if largest_diff < split_thd
                    A.final_components[num_final_components] = component
                    num_final_components += 1
                    num_total_components += 1
                    append!(sizes,length(component))
               else
                    for components in values(graph_clusters)
                         for new_component in components
                              append!(sizes,length(new_component))
                              A.component_records[num_total_components] = component[new_component]
                              append!(A.component_id_map[component_id],num_total_components)
                              if (length(new_component) > smallest_component)
                                   append!(new_level,num_total_components)
                              else
                                   A.final_components[num_final_components] = component[new_component]
                                   num_final_components += 1
                              end
                              num_total_components += 1
                         end
                    end
               end
          end #end of for loop
          curr_level = new_level
          append!(A.component_counts,length(A.component_records))
          if verbose
               @info "Min/max largest difference: " min_largest_diff, max_largest_diff
               @info "New components sizes:" counter(sizes)
               @info  "Time taken by splitting" splittime
               @info "At the end of while loop " length(curr_level)
          end
     end #end of while loop that check length(curr_level)>0
     splittime = time() - splitstart
     @info "Splitting took time " splittime
     if length(curr_level) > 0
          if iters >= max_iters
               @info "Stopped early, try increasing max number of iterations for splitting"
          end
          for i in curr_level
               A.final_components[num_final_components] = A.component_records[i]
               num_final_components += 1
          end
     end
     @info "After the while loop num_total_components" num_total_components
     @info "num_final_components" num_final_components
	_remove_duplicate_components(A)
end

function _remove_duplicate_components(A::sGTDA)
     all_c = sort(unique!([sort(A.final_components[key]) for key in keys(A.final_components)]))
     A.final_components_unique = Dict(i=>c for (i,c) in enumerate(all_c))
end

function filter_tiny_components!(A::sGTDA,Ar;node_size_thd=10,verbose=false)
     nodes = []
     for val in values(A.final_components_unique)
          for node in val
               push!(nodes,node)
          end
     end
     nodes = unique!(sort(nodes))
     @info "Number of samples included before filtering:" length(nodes)
     all_keys = keys(A.final_components_unique)
     filtered_keys = []
     removed_keys = []
     A.node_assignments = [[] for _ in range(1,size(Ar,1))]
     A.node_assignments_tiny_components = [[] for _ in range(1,size(Ar,1))]
     for key in all_keys
          if length(A.final_components_unique[key]) > node_size_thd
               for node in A.final_components_unique[key]
                    append!(A.node_assignments[node],key)
               end
               append!(filtered_keys,key)
          else
               for node in A.final_components_unique[key]
                    append!(A.node_assignments_tiny_components[node],key)
               end
               append!(removed_keys,key)
          end
     end
     A.final_components_removed = Dict(key => A.final_components_unique[key] for key in removed_keys)
     A.final_components_filtered = Dict(key => A.final_components_unique[key] for key in filtered_keys)
     nodes = []
     for val in values(A.final_components_filtered)
          for node in val
               push!(nodes,node)
          end
     end
     nodes = unique!(sort(nodes))
     @info "Number of samples included after filtering:" length(nodes)
end



function merge_reeb_nodes!(A::sGTDA,Ar,M;niters=1,node_size_thd=10,edges_dists=[])
     num_components = length(A.final_components_filtered) + length(A.final_components_removed)
     function worker(tmp_edges_dists,nodes,k1)
          closest_neigh = -1
          iii = findnz(tmp_edges_dists)
          neighs = iii[2]
          valid_neighs = setdiff(neighs,nodes)
          tmp_edges_dists = tmp_edges_dists[:,valid_neighs]
          iii =  findnz(tmp_edges_dists)
          if length(tmp_edges_dists.nzval) > 0
               closest_neigh_id = argmin(tmp_edges_dists.nzval)
               closest_neigh = iii[2][closest_neigh_id]
               closest_neigh = valid_neighs[closest_neigh]
          end
          return closest_neigh, k1
     end
     modified = true
     A.edges_to_merge = []
     @info "Now merging reeb nodes"
     for _ in range(1,niters)
          if modified
               modified = false
          else
               break
          end
          merging_ei = []
          merging_ej = []
          keys_to_check = keys(A.final_components_removed) 
          processed_list = []
          for k1 in keys_to_check
               push!(processed_list,worker(edges_dists[A.final_components_removed[k1],:],A.final_components_removed[k1],k1))
          end
          for (closest_neigh,k1) in processed_list
               if  closest_neigh != -1
                    components_to_connect = cat(A.node_assignments[closest_neigh],A.node_assignments_tiny_components[closest_neigh],dims = 1)
                    if length(components_to_connect) > 0
                         sizes = []
                         for c in components_to_connect
                              if c in keys(A.final_components_filtered)
                                   append!(sizes,length(A.final_components_filtered[c]))
                              else
                                   append!(sizes,length(A.final_components_removed[c]))
                              end
                         end
                         component_to_connect = components_to_connect[argmin(sizes)]
                         append!(merging_ei,component_to_connect)
                         append!(merging_ej,k1)
                         append!(A.edges_to_merge,(component_to_connect,k1))
                         modified = true
                    end
               end
          end
          merging_map = sparse(merging_ei,merging_ej,ones(length(merging_ei)),num_components,num_components)
          merging_map = make_graph_symmetric(merging_map,num_components)
          merging_tiny_nodes!(A,merging_map,node_size_thd=node_size_thd)
          @info "After merging tiny components " length(A.final_components_filtered)
     end

end



function merging_tiny_nodes!(A::sGTDA,merging_map;node_size_thd=10,verbose=false)
     keys_to_remove = []
     _,components_to_merge = find_components(merging_map,size_thd=1)
     if verbose
          @info "tiny reebnodes to be connected further" size(merging_map)
          @info "number of reeb components to merge"  length(components_to_merge)
     end
     for component_to_merge in components_to_merge
          component_to_connect = component_to_merge[1]
          for k in component_to_merge
               if k in keys(A.final_components_filtered)
                    component_to_connect = k
               end
          end
          new_component = []
          for k in component_to_merge
               if k == component_to_connect
                    continue
               end
               nodes = A.final_components_removed[k]
               if component_to_connect in keys(A.final_components_filtered)
                    append!(A.final_components_filtered[component_to_connect],nodes)
                    A.final_components_filtered[component_to_connect] = sort(unique!(A.final_components_filtered[component_to_connect]))
                    append!(keys_to_remove,k)
                    for node in nodes
                         append!(A.node_assignments[node],component_to_connect)
                         setdiff!(A.node_assignments_tiny_components[node],[k])
                    end
               else
                    append!(new_component,nodes)
               end
          end
          if !(component_to_connect in keys(A.final_components_filtered))
               append!(new_component,A.final_components_removed[component_to_connect])
               new_component = sort(unique!(new_component))
               if verbose
                    @info "new_component " new_component
                    @info "node_size_thd " node_size_thd
               end
               if length(new_component) > node_size_thd
                    for k in component_to_merge
                        nodes = A.final_components_removed[k]
                        append!(keys_to_remove,k)
                        for node in nodes 
                            append!( A.node_assignments[node],component_to_connect)
                            setdiff!(A.node_assignments_tiny_components[node],[k])
                        end
                    end
                    A.final_components_filtered[component_to_connect] = new_component
               else
                    for k in component_to_merge
                         nodes = A.final_components_removed[k]
                         if k != component_to_connect
                              append!(keys_to_remove,k)
                              for node in nodes 
                                   setdiff!(A.node_assignments_tiny_components[node],[k])
                                   append!( A.node_assignments_tiny_components[node],component_to_connect)
                              end
                         end
                    end
                    A.final_components_removed[component_to_connect] = new_component
               end
          end
     end
     for k in keys_to_remove
          delete!(A.final_components_removed,k)
     end
     nodes = []
     for val in values(A.final_components_filtered)
          append!(nodes,val)
     end
     nodes = sort(unique!(nodes))
     if verbose
          print("Number of samples included after merging:", length(nodes))
     end
end



function error_prediction!(A::sGTDA;alpha=0.5,nsteps=10,pre_labels=nothing,known_nodes=nothing,train_mask=[],val_mask=[],verbose=false)

     if known_nodes === nothing
          known_mask_np = train_mask+val_mask
          known_nodes =  [i for i=1:length(known_mask_np) if known_mask_np[i] != 0]
     else
          known_mask_np = zeros(size(Ar,1))
          known_mask_np[known_nodes] = 1.0
     end

     if pre_labels === nothing
          pre_labels = [i[2] for i in argmax(A.preds,dims=2)]
     end


     max_key = maximum([key for key in keys(A.final_components_filtered)])
     
     A.node_colors_class = zeros(max_key,size(A.preds,2))
     A.node_colors_class_truth = zeros(max_key,size(A.preds,2))
     A.node_colors_error = zeros(max_key)
     A.node_colors_uncertainty = zeros(max_key)
     A.node_colors_mixing = zeros(max_key)
     A.sample_colors_mixing = zeros(size(A.preds,1))
     uncertainty = 1 .- maximum(A.preds,dims=2)
     A.sample_colors_uncertainty = uncertainty
     A.sample_colors_error = zeros(size(A.preds,1))


     training_node_labels = zeros((size(A.G,1),size(A.preds,2)))

     if known_nodes !== nothing
          for node in known_nodes
               training_node_labels[node,A.labels[node]+1] = 1
          end
     end
     A.total_mixing_all = copy(training_node_labels)
     
     for _ in range(1,nsteps)
          A.total_mixing_all = (1-alpha)*training_node_labels + alpha*(A.An*A.total_mixing_all)
     end
     
     for i in range(1,size(A.total_mixing_all,1))
          if sum(A.total_mixing_all[i,:]) > 0
               d = A.total_mixing_all[i,pre_labels[i]]/sum(A.total_mixing_all[i,:])
               A.sample_colors_mixing[i] = 1-d
          else
               A.sample_colors_mixing[i] = uncertainty[i]
          end
          A.sample_colors_error[i] = 1-(pre_labels[i] == (A.labels[i]+1))
          A.sample_colors_uncertainty[i] = uncertainty[i]
     end
     
     if verbose
          @info "this is the sum of the prediction error: " sum(A.sample_colors_mixing)
     end
     
     for key in keys(A.final_components_filtered)
          component = A.final_components_filtered[key]
          component_label_cnt = StatsBase.countmap(A.labels[component])
          for l in keys(component_label_cnt)
              A.node_colors_class_truth[key,l+1] = component_label_cnt[l]
          end
          component_label_cnt = StatsBase.countmap(pre_labels[component])
          for l in keys(component_label_cnt)
              A.node_colors_class[key,l] = component_label_cnt[l]
          end
          if length(component) > 0
              A.node_colors_error[key] = mean(A.sample_colors_error[component])
              A.node_colors_uncertainty[key] = mean(A.sample_colors_uncertainty[component])
              A.node_colors_mixing[key] = mean(A.sample_colors_mixing[component])
          end
     end
     
     return A.sample_colors_mixing
end 


function get_reebgroups!(A::sGTDA;verbose=false)
     nodes = []
     A.node2reeb = DefaultDict{Int,Vector{Float64}}(Vector{Float64})
     A.reeb2node = DefaultDict{Int,Vector{Float64}}(Vector{Float64})
     A.filtered_nodes = sort(unique!(intersect(A.filtered_nodes,[k for k in keys(A.final_components_filtered)])))
     for i in A.filtered_nodes
          append!(A.reeb2node[i],A.final_components_filtered[i])
	     A.reeb2node[i] = sort(unique!(A.reeb2node[i]))
          component = A.final_components_filtered[i]
          for noodles in component
               append!(A.node2reeb[noodles],i)
          end
          append!(nodes,component)
     end
     nodes = sort(unique!(nodes))
     
     if verbose
          @info "Final number of reebnodes " length(A.filtered_nodes)
          @info "Number of original nodes included after merging reeb components " length(nodes)
          @info "Number of reebnodes" length(A.reeb2node)
     end
     
     return A.reeb2node, A.node2reeb
end



function make_graph_symmetric(A_tmp,dim)
    temp = A_tmp+A_tmp'
    i = findnz(temp)
    A_tmp = sparse(i[1],i[2],Float64.(i[3] .> 0),dim,dim)
end

function gives_removed_components(A::sGTDA, A_tmp, size_thd, reeb_component_thd)
    _,components_left1 = find_components(A_tmp,size_thd=size_thd)
    components_removed = []
    A.filtered_nodes = []
    for c in components_left1
        if length(c) > reeb_component_thd
            append!(A.filtered_nodes,c)
        else
            for node in c 
                if node in keys(A.final_components_filtered)
                        append!(components_removed,c)
                        break
                end
            end
        end
    end
    return components_removed
end


function build_reeb_graph!(A::sGTDA,M;reeb_component_thd=10,max_iters=10,is_merging=true,edges_dists=nothing,verbose=false,degree_normalize = 1)
     all_edge_index = [[], []]
     extra_edges = [[],[]]
     @info "Building reeb graph"

     reeb_dim = maximum(keys(A.final_components_filtered))
     ei,ej = [],[]
     for (key,val) in A.final_components_filtered
          append!(ei,[key for _=1:length(val)])
	     append!(ej,val)
     end
     bipartite_g = sparse(ei,ej,ones(length(ei)),reeb_dim,size(M,1))
     bipartite_g_t = sparse(bipartite_g')
     ei,ej = [],[]
     for i in keys(A.final_components_filtered) 
          neighs = findnz(bipartite_g_t[findnz(bipartite_g[i,:])[1],:])[2]
	     setdiff!(neighs,i)
          append!(all_edge_index[1],[i for _ in range(1,length(neighs))])
          append!(all_edge_index[2],neighs)
     end
     A_tmp = sparse(all_edge_index[1],all_edge_index[2],ones(length(all_edge_index[1])),reeb_dim,reeb_dim)
     A_tmp = make_graph_symmetric(A_tmp,reeb_dim)
     components_removed = gives_removed_components(A, A_tmp, 0, reeb_component_thd)
     curr_iter = 0
     modified = true
     if verbose
          @info "length(components_removed)" length(components_removed)
     end
     while (modified) && (is_merging) && (length(components_removed) > 0) && (curr_iter < max_iters)
          modified = false
          curr_iter += 1
          for cr in components_removed
               nodes_removed = []
               for key in cr
                    append!(nodes_removed,A.final_components_filtered[key])
               end
	     
               tmp_edges_dists = edges_dists[nodes_removed,:] 
	          neighs = sparse(tmp_edges_dists').rowval
               valid_neighs = setdiff(neighs,nodes_removed)
	          tmp_edges_dists = tmp_edges_dists[:,valid_neighs]
	       
               key_to_connect = -1
               closest_neigh = -1
	       
               if size(tmp_edges_dists.nzval,1) > 0
                    kkk = findnz(tmp_edges_dists)
	               closest_neigh_id = argmin(kkk[3])
		          closest_neigh = kkk[2][closest_neigh_id]
                    closest_neigh = valid_neighs[closest_neigh]
		          rfmt = sparse(tmp_edges_dists')
		          node_to_connect = nodes_removed[searchsorted(rfmt.colptr,argmin(findnz(rfmt)[3])).start-1]
                    key_to_connect = minimum(intersect(A.node_assignments[node_to_connect],cr))
		     end

               if closest_neigh != -1
                    component_to_connect,modified = connect_the_components(A,closest_neigh)
		          append!(all_edge_index[1],key_to_connect)
                    append!(all_edge_index[2],component_to_connect)
                    append!(extra_edges[1],node_to_connect)
                    append!(extra_edges[2],closest_neigh)
               end
          end
          if verbose
               @info "length(all_edge_index[1])" length(all_edge_index[1])
               @info "length(all_edge_index[2])" length(all_edge_index[2])
          end
          A_tmp = sparse(all_edge_index[1],all_edge_index[2],ones(length(all_edge_index[1])),reeb_dim,reeb_dim)
	     A_tmp = make_graph_symmetric(A_tmp,reeb_dim)

          if verbose
               @info "length(findnz(A_tmp)[1])" length(findnz(A_tmp)[1])
          end
	     
          components_removed = gives_removed_components(A, A_tmp, 0, reeb_component_thd)
     end

     A.G_reeb = A_tmp[A.filtered_nodes,:][:,A.filtered_nodes]

     #now that we have the gtda graph - we see the projected graph A_reeb
     reeb_components = find_components(A.G_reeb,size_thd=0)[2]
     ei,ej = [],[]
     Ar = A.G
     for reeb in reeb_components
          for rnode in reeb
               if rnode in keys(A.final_components_filtered)
                    nodes = A.final_components_filtered[rnode]
                    nodes = sort(unique!(nodes))
                    mapping = Dict(i => k for (i,k) in enumerate(nodes))
                    sub_A = Ar[nodes,:][:,nodes]
                    temp = findnz(sub_A)
                    for (i,j) in zip(temp[1],temp[2])
                         append!(ei,mapping[i])
                         append!(ej,mapping[j])
                    end
               end
          end
     end
     if extra_edges !== nothing
          append!(ei,extra_edges[1])
          append!(ej,extra_edges[2])
     end
     A.A_reeb = sparse(ei,ej,ones(length(ei)),size(Ar,1),size(Ar,2))
     A.A_reeb = make_graph_symmetric(A.A_reeb,size(Ar,1))
     degs = sum(A.A_reeb, dims = 1)
     dinv = vec(1 ./ degs)
     dinv[dinv .== Inf] .= 0
     Dinv = SparseArrays.spdiagm(size(A.A_reeb,1),size(A.A_reeb,2),dinv)
     if degree_normalize == 1
          A.An = Dinv*A.A_reeb
     elseif degree_normalize == 2
          A.An = A.A_reeb*Dinv
     elseif degree_normalize == 3
          A.An = sqrt.(Dinv) * A.A_reeb * sqrt.(Dinv)
     else
          A.An = A.A_reeb
     end
     
     return A_tmp,extra_edges
end



function connect_the_components(A::sGTDA,closest_neigh;verbose=false)
     components_to_connect = A.node_assignments[closest_neigh]
     if verbose
          @info "components_to_connect" components_to_connect
          @info "closent_neigh" closest_neigh
     end
     if length(components_to_connect) > 0
          sizes = []
          for c in components_to_connect
               if c in keys(A.final_components_filtered)
                    append!(sizes,length(A.final_components_filtered[c]))
               else
                    append!(sizes,length(A.final_components_removed[c]))
               end
          end
          component_to_connect = components_to_connect[argmin(sizes)]
	     modified = true
     end
     return component_to_connect,modified
end

function select_merge_edges(A_knn,M;merge_thd=1.0)
     Au = sparse(UpperTriangular(A_knn))
     ei,ej = findnz(Au)[1:2]
     e = []
     n = size(A_knn,1)
     for i in Vector(1:10000:length(Au.nzval))
          start_i = length(e)+1
          end_i = min(start_i+10000,length(Au.nzval))
          append!(e,maximum(abs.(M[ei[start_i:end_i],:] .- M[ej[start_i:end_i],:]),dims=2))
	end
	e = Float64.(e)
     selected_edges = [i for i=1:length(e) if e[i] < merge_thd]
     @info "number of selected edges" length(selected_edges)
     edges_dists = sparse(ei[selected_edges],ej[selected_edges],e[selected_edges],n,n)
     edges_dists = edges_dists+edges_dists'
     return edges_dists
end

function gtdagraph!(gtda::sGTDA;overlap = 0.025,max_split_size = 100,min_group_size=5,min_component_group=5,alpha=0.5,nsteps_preprocess=5,extra_lens=nothing,is_merging=true,split_criteria="diff",split_thd=0,is_normalize=true,is_standardize=false,merge_thd=1.0,max_split_iters=200,max_merge_iters=10,nprocs=1,degree_normalize_preprocess=1,degree_normalize_mixing=1,verbose=false)

     M,Ar = smooth_lenses!(gtda,extra_lens=extra_lens,alpha=alpha,nsteps = nsteps_preprocess,normalize = is_normalize,standardize = is_standardize,degree_normalize=degree_normalize_preprocess)
     this = time()
     
     find_reeb_nodes!(gtda,M,Ar,smallest_component=max_split_size,filter_cols=[i for i in range(1,size(M,2))],overlap=overlap,component_size_thd=0,split_criteria=split_criteria,split_thd=split_thd,max_iters=max_split_iters,verbose=verbose)
     
     if verbose
          @info "Initial number of reebnodes found " length(gtda.final_components_unique)
     end

     filter_tiny_components!(gtda,Ar,node_size_thd=min_group_size)

     if verbose
          @info "After filtering tiny components " length(gtda.final_components_filtered)
     end

     edges_dists = select_merge_edges(gtda.G,M,merge_thd = merge_thd)

     merge_reeb_nodes!(gtda,Ar,M,niters=max_merge_iters,node_size_thd=min_group_size,edges_dists=edges_dists)

     if verbose
          @info "After merging " length(gtda.final_components_filtered)
     end

     
     g_reeb_orig,extra_edges = build_reeb_graph!(gtda,M,reeb_component_thd=min_component_group,max_iters=max_merge_iters,is_merging=is_merging,edges_dists=edges_dists,verbose=verbose,degree_normalize = degree_normalize_mixing)
     reebgroups,nodereebs = get_reebgroups!(gtda)
     computereebtime = time()-this
     if verbose
          @info "Building reebgraph took time " computereebtime
     end
    
    
    GTDA_record = (G_reeb = gtda.G_reeb,
          A_reeb = gtda.A_reeb,
          reeb_group = reebgroups,
          node_in_reebs = nodereebs,
          extra_edges = extra_edges,
          timetaken = computereebtime,
          M = M)
          
    return GTDA_record, g_reeb_orig

end




function analyzediffusion(G,X;labels_to_eval = [i for i in range(1,size(preds,2))],labels=[],extra_lens=nothing,alpha=0.5,known_nodes=nothing,nsteps_preprocess=5,min_group_size = 5,max_split_size = 100,min_component_group = 5,overlap = 0.025,nsteps_mixing=10,is_merging=true,split_criteria="diff",split_thd=0,is_normalize=true,is_standardize=false,merge_thd=1.0,max_split_iters=200,max_merge_iters=10,degree_normalize_preprocess=1,degree_normalize_mixing=1,verbose=false)
     gtda = GTDA.sGTDA()
     gtda.G = G
     gtda.preds = X
     gtda.labels_to_eval = labels_to_eval
     gtda.labels = labels
     

     GTDA_record, greeb_orig = gtdagraph!(gtda,max_split_size = max_split_size,overlap = overlap,min_group_size=min_group_size,min_component_group = min_component_group
     ,alpha = alpha,nsteps_preprocess=nsteps_preprocess,extra_lens=extra_lens,is_merging=is_merging,split_criteria=split_criteria,split_thd=split_thd,
     is_normalize=is_normalize,is_standardize=is_standardize,merge_thd=merge_thd,max_split_iters=max_split_iters,max_merge_iters=max_merge_iters,
     degree_normalize_preprocess=degree_normalize_preprocess,degree_normalize_mixing=degree_normalize_mixing,verbose=verbose)

     err = 0
     if length(labels)
          err =  error_prediction!(gtda,nsteps=nsteps_mixing,alpha = alpha,known_nodes=known_nodes)
     end
     return GTDA_record, greeb_orig, err

end


function analyzeGNN(X,trainlen,testlen;labels_to_eval = [i for i in range(1,size(preds,2))],labels=[],extra_lens=nothing,G = nothing,alpha=0.5,batch_size=10000,known_nodes=nothing,knn=5,nsteps_preprocess=5,min_group_size = 5,max_split_size = 100,min_component_group = 5,overlap = 0.025,nsteps_mixing=10,is_merging=true,split_criteria="diff",split_thd=0,is_normalize=true,is_standardize=false,merge_thd=1.0,max_split_iters=200,max_merge_iters=10,degree_normalize_preprocess=1,degree_normalize_mixing=1,verbose=false)
     if G === nothing
          G = canonicalize_graph(X,knn,batch_size)
     end
     gtda = GTDA.sGTDA()
     gtda.G = G
     gtda.preds = X
     gtda.labels_to_eval = labels_to_eval
     gtda.labels = labels
     
     GTDA_record, greeb_orig = gtdagraph!(gtda,max_split_size = max_split_size,overlap = overlap,min_group_size=min_group_size,min_component_group = min_component_group
     ,alpha = alpha,nsteps_preprocess=nsteps_preprocess,extra_lens=extra_lens,is_merging=is_merging,split_criteria=split_criteria,split_thd=split_thd,
     is_normalize=is_normalize,is_standardize=is_standardize,merge_thd=merge_thd,max_split_iters=max_split_iters,max_merge_iters=max_merge_iters,
     degree_normalize_preprocess=degree_normalize_preprocess,degree_normalize_mixing=degree_normalize_mixing,verbose=verbose)

    
     train_nodes = [i for i in range(1,trainlen)]
     val_nodes = []
     test_nodes = [i for i in range(trainlen,trainlen+testlen)]
     train_mask = zeros(size(G,1))
     train_mask[train_nodes] = 1
     val_mask = zeros(size(G,1))
     val_mask[val_nodes] = 1
     test_mask = zeros(size(G,1))
     test_mask[test_nodes] = 1
     
     err = 0
     if length(labels)
          err =  error_prediction!(gtda,nsteps=nsteps_mixing,alpha = alpha,known_nodes=known_nodes,train_mask=train_mask,val_mask = val_mask)
     end
     return GTDA_record, greeb_orig, err
end

function analyzeNN(X,trainlen,testlen;labels_to_eval = [i for i in range(1,size(preds,2))],labels=[],extra_lens=nothing,G = nothing,alpha=0.5,batch_size=10000,known_nodes=nothing,knn=5,nsteps_preprocess=5,min_group_size = 5,max_split_size = 100,min_component_group = 5,overlap = 0.025,nsteps_mixing=10,is_merging=true,split_criteria="diff",split_thd=0,is_normalize=true,is_standardize=false,merge_thd=1.0,max_split_iters=200,max_merge_iters=10,degree_normalize_preprocess=1,degree_normalize_mixing=1,verbose=false)
     if G === nothing
          G = canonicalize_graph(X,knn,batch_size)
     end
     gtda = GTDA.sGTDA()
     gtda.G = G
     gtda.preds = X
     gtda.labels_to_eval = labels_to_eval
     gtda.labels = labels
     
     
     GTDA_record, greeb_orig = gtdagraph!(gtda,max_split_size = max_split_size,overlap = overlap,min_group_size=min_group_size,min_component_group = min_component_group
     ,alpha = alpha,nsteps_preprocess=nsteps_preprocess,extra_lens=extra_lens,is_merging=is_merging,split_criteria=split_criteria,split_thd=split_thd,
     is_normalize=is_normalize,is_standardize=is_standardize,merge_thd=merge_thd,max_split_iters=max_split_iters,max_merge_iters=max_merge_iters,
     degree_normalize_preprocess=degree_normalize_preprocess,degree_normalize_mixing=degree_normalize_mixing,verbose=verbose)

    
     train_nodes = [i for i in range(1,trainlen)]
     val_nodes = []
     test_nodes = [i for i in range(trainlen,trainlen+testlen)]
     train_mask = zeros(size(G,1))
     train_mask[train_nodes] = 1
     val_mask = zeros(size(G,1))
     val_mask[val_nodes] = 1
     test_mask = zeros(size(G,1))
     test_mask[test_nodes] = 1
     
     err = 0
     if length(labels)
          err =  error_prediction!(gtda,nsteps=nsteps_mixing,alpha = alpha,known_nodes=known_nodes,train_mask=train_mask,val_mask = val_mask)
     end
     return GTDA_record, greeb_orig, err

end


export gtdagraph!, sGTDA, toy, canonicalize_graph, error_prediction!, smooth_lenses!, analyzeNN, analyzediffusion

end #end of module

