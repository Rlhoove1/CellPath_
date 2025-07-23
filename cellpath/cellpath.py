import cellpath.visual as visual
import cellpath.clustering as clust
import cellpath.path as path
import cellpath.nn as nn
import cellpath.benchmark as bmk

import numpy as np  
import pandas as pd
import scvelo as scv
import scipy.sparse as sparse

def pairwise_distances(x, y):
    """\
    Description
        Calculate the pairwise distance given two feature matrices x and y
    Parameters
    ----------
    x
        feature matrix of dimension (n_obs_x, n_features)
    y
        feature matrix of dimension (n_obs_y, n_features)
    """
    x_norm = np.sum(x**2, axis = 1)[:, None]
    y_norm = np.sum(y**2, axis = 1)[None, :]
    
    dist = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    return np.where(dist <= 0.0, 0.0, dist)

class CellPath():
    def __init__(self, adata, preprocess = True, **kwargs):
        """\
        Description
            Initialize cellpath object
        
        Parameters
        ----------       
        adata
            Anndata object, store the dataset
        preprocess
            dataset is processed or not, boolean
        **kwargs
            additional parameters for preprocessing (using scvelo)
        """
        self.adata = adata
        if not sparse.issparse(adata.X):
            self.adata.X = sparse.csr_matrix(self.adata.X)
        
        if preprocess != True:
            print("preprocessing data using scvelo...")

            _kwargs = {
                "min_shared_genes": 20,
                "n_top_genes": 1000,
                "n_pcs": 30,
                "n_neighbors": 30,
                "velo_mode": "stochastic"
            }
            
            _kwargs.update(kwargs)

            # store the raw count before processing, for subsequent de analysis
            self.adata.layers["raw"] = self.adata.X.copy()

            scv.pp.filter_and_normalize(data = self.adata, 
                                        min_shared_counts=_kwargs["min_shared_genes"], 
                                        n_top_genes=_kwargs["n_top_genes"])

            scv.pp.moments(data = self.adata, n_pcs = _kwargs["n_pcs"], n_neighbors = _kwargs["n_neighbors"])

            if _kwargs["velo_mode"] == "stochastic":
                scv.tl.velocity(data = self.adata, model = _kwargs["velo_mode"])

            elif _kwargs["velo_mode"] == "dynamical":
                scv.tl.recover_dynamics(data = self.adata)
                scv.tl.velocity(data = self.adata, model = _kwargs["velo_mode"])

                # remove nan genes
                _velo_matrix = self.adata.layers['velocity'].copy()
                _genes_subset = ~np.isnan(_velo_matrix).any(axis=0)
                self.adata._inplace_subset_var(_genes_subset)

            else:
                raise ValueError("`velo_mode` can only be dynamical or stochastic")

    def meta_cell_construction(self, flavor="k-means", n_clusters=None, resolution=30, **kwarg):
        """\
        Construct meta-cells within predefined clusters.
    
        Parameters
        ----------
        flavor : str
            Clustering algorithm for meta-cell construction ("k-means" or "leiden").
        n_clusters : int or None
            Number of meta-cells per cluster (for k-means). If None, defaults to len(cluster) // 10.
        resolution : float
            Resolution parameter (for leiden clustering).
        """
    
        _kwargs = {
            "include_unspliced": True,
            "standardize": True,
            "n_comps": 30, 
            "kernel": "rbf",
            "alpha": 1,
            "gamma": 0.3,
            "verbose": True,
            "seed": 0
        }
        _kwargs.update(kwarg)
    
        if "clusters" not in self.adata.obs:
            raise ValueError("Predefined clusters not found in `adata.obs['clusters']`")
    
        meta_cell_labels = []
    
        for cluster in self.adata.obs["clusters"].unique():
            idx = self.adata.obs["clusters"] == cluster
            adata_sub = self.adata[idx].copy()
    
            if flavor == "k-means":
                n_sub = n_clusters or max(2, len(adata_sub) // 10)
                X = adata_sub.obsm["X_pca"] if "X_pca" in adata_sub.obsm else adata_sub.X.toarray()
                kmeans = KMeans(n_clusters=n_sub, random_state=_kwargs["seed"], n_init="auto").fit(X)
                sub_labels = [f"{cluster}_{i}" for i in kmeans.labels_]
    
            elif flavor == "leiden":
                import scanpy as sc
                sc.pp.neighbors(adata_sub, n_pcs=_kwargs["n_comps"])
                sc.tl.leiden(adata_sub, resolution=resolution, key_added="meta_cell_tmp")
                sub_labels = [f"{cluster}_{lab}" for lab in adata_sub.obs["meta_cell_tmp"]]
    
            else:
                raise ValueError(f"Unsupported clustering flavor: {flavor}")
    
            meta_cell_labels.extend(pd.Series(sub_labels, index=adata_sub.obs_names))
    
        # Assign meta-cell labels to adata
        self.adata.obs["meta_cell"] = pd.Series(meta_cell_labels)
    
        # Create meta-cell expression/velocity representation
        self.groups = self.adata.obs["meta_cell"]
        self.X_clust, self.velo_clust = clust.meta_cells(
            self.adata,
            kernel=_kwargs["kernel"],
            alpha=_kwargs["alpha"],
            gamma=_kwargs["gamma"]
        )
    
        if _kwargs["verbose"]:
            print(f"Meta-cell construction complete. Number of meta-cells: {self.X_clust.shape[0]}")

    
    def meta_cell_graph(self, k_neighs = 10, pruning = True, **kwargs):
        """\
        Description
            meta-cell level graph construction

        Parameters
        ----------
        k_neighs
            Number of neighbors for the neighborhood graph, default 10
        pruning
            Pruning the network or not, boolean variable, affect the continuity of the trajectory. 
            False (less fragmented paths) for most of the cases.
        """        
        _kwargs = {
            "symm": True,
            "scaling": 3,
            "distance_scalar": 0.5,
            "verbose": True
        }
        _kwargs.update(kwargs)

        _adj_matrix, _dist_matrix = nn.NeighborhoodGraph(self.X_clust, k_neighs = k_neighs, symm = _kwargs["symm"], pruning = pruning)
        self.adj_assigned = nn.assign_weights(connectivities = _adj_matrix, distances = _dist_matrix, X = self.X_clust, 
                                                               velo_pca = self.velo_clust, scaling = _kwargs["scaling"], 
                                                               distance_scalar = _kwargs["distance_scalar"], threshold = 0.0)

        self.max_weight = (_kwargs["scaling"] * (1 + _kwargs["distance_scalar"]))**_kwargs["scaling"]
        if _kwargs["verbose"] == True:
            print("Meta-cell level neighborhood graph constructed")
    
    def meta_paths_finding(self, threshold = 0.5, cutoff_length = 5, length_bias = 0.7, mode = "fast", **kwargs):
        """\
        Description
            meta-cell level trajectory finding

        Parameters
        ----------
        threshold
            Cut-off quality score, equals to threshold * max_weight
        cutoff_length
            The cutoff length (lower bound) of inferred trajectory
        length_bias
            The bias on the path length for greedy selection
        mode
            The path finding algorithm. ``fast'': dijkstra, ``exact'': floydWarshall, default fast.
        """ 
        _kwargs = {
            "max_trajs": None,
            "verbose": True,
            "root_cell_indeg":[0,1,2],
        }

        _kwargs.update(kwargs)
        if mode == "fast":
            self.paths, self.opt = path.dijkstra_paths(adj = self.adj_assigned.copy(), indeg = _kwargs["root_cell_indeg"])
        elif mode == "exact":
            self.paths, self.opt = path.floyd_warshall(adj = self.adj_assigned.copy())
        else:
            raise ValueError("mode can only be ``fast'' or ``exact''.")

        n_metacells = int(np.max(self.groups)+1)
        self.greedy_order, self.paths = path.greedy_selection(nodes = n_metacells, paths = self.paths,opt_value = self.opt, threshold = threshold, 
                                                              max_w=self.max_weight, cut_off=cutoff_length, 
                                                              verbose = _kwargs["verbose"], length_bias = length_bias, 
                                                              max_trajs = _kwargs["max_trajs"])

    def _cells_insertion(self, num_trajs = None, prop_insert = 1):
        """\
        Description
            Inserting unassigned cells

        Parameters
        ----------
        num_trajs
            Number of trajectories
        verbose
            Output result
        prop_insert
            Parameters for the number of cells incorporated
        """     
        if num_trajs == None:
            num_trajs = len(self.greedy_order)
        elif num_trajs > len(self.greedy_order):
            print(len(self.greedy_order))

        # assigned cells
        cell_pools = set([])
        clust_pools = set([])

        for i in range(num_trajs):
            traj = []
            for index in self.paths[self.greedy_order[i]]: 
                clust_pools.add(index)  

                # find the cells corresponding to the cluster in greedy paths
                group_i = np.where(self.groups == index)[0]
                # ordering the cells
                diff = self.adata.obsm['X_pca'][group_i,:] - self.X_clust[index,:] 
                group_i = group_i[np.argsort(np.dot(diff, self.velo_clust[index,:])/np.linalg.norm(self.velo_clust[index,:],2))]

                # traj store all the cells in the trajectory/greedy paths
                traj = np.append(traj, group_i)

            # incorporate all the cells in traj
            cell_pools = cell_pools.union(set(traj))
        
        # uncovered cells
        cell_uncovered = list(set([x for x in range(self.adata.n_obs)]) - cell_pools)

        # cell_uncovered by meta-cell distance matrix calculation
        X_pca = self.adata.obsm["X_pca"]
        uncovered_pca = X_pca[cell_uncovered, :]

        covered_clust = self.X_clust[list(clust_pools), :]
        pdist = pairwise_distances(uncovered_pca, covered_clust)

        threshold = prop_insert * np.max(pdist)

        pdist = np.where(pdist <= threshold, pdist, np.inf)
        

        # choose meta-cell for each cell
        # meta_cells = [list(clust_pools)[x] for i, x in enumerate(np.argmin(pdist, axis = 1)) if pdist[i, x] != np.inf]
        
        meta_cells = []
        indices = []

        for idx in range(pdist.shape[0]):
            x = np.argmin(pdist[idx,:])
            if pdist[idx, x] != np.inf:
                meta_cells.append(list(clust_pools)[x])
                indices.append(cell_uncovered[idx])

        print("number of cells: " + str(len(indices)))
        # assign cells to the closest meta-cell
        self.groups[indices] = meta_cells

    def first_order_pt(self, num_trajs = None, verbose = True, prop_insert = 0):
        """\
        Description
            cell level pseudo-time inference using first order approximation

        Parameters
        ----------
        num_trajs
            Number of trajectories
        verbose
            Output result
        prop_insert
            The proportion of cells to be incorporated, default 0(no cell to be inserted)
        """ 
        if num_trajs == None:
            num_trajs = len(self.greedy_order)
        elif num_trajs > len(self.greedy_order):
            print(len(self.greedy_order))
            num_trajs = len(self.greedy_order)
            # raise ValueError("number of trajectory to be selected larger than maximum number")
        self.pseudo_order = pd.DataFrame(data = np.nan, index = self.adata.obs.index, columns = ["traj_" + str(x) for x in range(num_trajs)]) 

        if prop_insert > 0:
            # inserting uncovered cells to the nearby meta-cells
            self._cells_insertion(num_trajs = num_trajs, prop_insert = prop_insert)


        for i in range(num_trajs):
            traj = np.array([])

            # the traj is already sorted by enumerating process
            for index in self.paths[self.greedy_order[i]]:
                # find the cells corresponding to the cluster in greedy paths
                group_i = np.where(self.groups == index)[0]
                # ordering the cells
                diff = self.adata.obsm['X_pca'][group_i,:] - self.X_clust[index,:] 
                group_i = group_i[np.argsort(np.dot(diff, self.velo_clust[index,:])/np.linalg.norm(self.velo_clust[index,:],2))]

                # traj store all the cells in the trajectory/greedy paths
                traj = np.append(traj, group_i)  
                
            traj = traj.astype('int')

            for pt, curr_cell in enumerate(traj):
                # self.pseudo_order.loc["cell_"+str(curr_cell),"traj_"+str(i)] = pt
                
                self.pseudo_order.loc[self.adata.obs.index.values[curr_cell],"traj_"+str(i)] = pt

        if verbose:
            print("Cell-level pseudo-time inferred")


    def all_in_one(self, flavor = "leiden", resolution = 30, num_metacells = None, 
                   n_neighs = 13, pruning = True, threshold = 0.5, cutoff_length = 5, length_bias = 0.7, mode = "exact", 
                   num_trajs = None, prop_insert = 0, **kwargs):
        """\
        Description
            run CellPath in one function

        Parameters
        ----------
        num_metacells
            number of meta cells, default cell number/10
        n_neighs
            number of neighbors for the neighborhood graph, default 10
        include_unspliced
            Boolean, whether include unspliced count or not
        standardize
            Standardize before pca, boolean
        threshold
            Cut-off quality score, quality score is no larger than threshold * max_weight
        cutoff_length
            The cutoff length (lower bound) of inferred trajectory
        length_bias
            The bias on the path length for greedy selection
        mode 
            The mode of path finding algorithm, include ``fast'' and ``exact'', default ``exact''
        num_trajs
            Number of trajectories 
        prop_insert
            Proportion of cells to be incorporated in insertion, no cell to be inserted if set to 0, default 0
        """ 

        # self.meta_cell_construction(n_clusters = num_metacells, include_unspliced = include_unspliced,
        #                             standardize = standardize, **kwargs)
        self.meta_cell_construction(flavor = flavor, n_clusters = num_metacells, resolution = resolution, **kwargs)

        self.meta_cell_graph(k_neighs = n_neighs, **kwargs)
        self.meta_paths_finding(threshold = threshold, cutoff_length = cutoff_length, length_bias = length_bias, mode = mode, **kwargs)
        self.first_order_pt(num_trajs = num_trajs, prop_insert = prop_insert)
