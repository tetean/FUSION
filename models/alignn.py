import os
import csv
import copy
import random
import sys
import pickle
import math
from typing import Optional, Tuple, Union, Literal, List, Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pymatgen.core.structure import Structure
from collections import OrderedDict, defaultdict

from jarvis.core.atoms import Atoms, get_supercell_dims
from jarvis.core.specie import Specie, chem_data, get_node_attributes
from jarvis.core.utils import random_colors
from jarvis.analysis.structure.neighbors import NeighborsAnalysis

import dgl
from dgl.data import DGLDataset
import dgl.function as fn
from dgl.nn import AvgPooling
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

tqdm.pandas()

def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    # if a site has too few neighbors, increase the cutoff radius
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    # print ('cutoff=',all_neighbors)
    if min_nbrs < max_neighbors:
        # print("extending cutoff radius!", attempt, cutoff, id)
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1

        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )
    # build up edge list
    # NOTE: currently there's no guarantee that this creates undirected graphs
    # An undirected solution would build the full edge list where nodes are
    # keyed by (index, image), and ensure each edge has a complementary edge

    # indeed, JVASP-59628 is an example of a calculation where this produces
    # a graph where one site has no incident edges!

    # build an edge dictionary u -> v
    # so later we can run through the dictionary
    # and remove all pairs of edges
    # so what's left is the odd ones out
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        # max_dist = distances[max_neighbors - 1]

        # keep all edges out to the neighbor shell of the k-th neighbor
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        # keep track of cell-resolved edges
        # to enforce undirected graph construction
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges


def build_undirected_edgedata(
    atoms=None,
    edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            # if np.linalg.norm(d)!=0:
            # print ('jv',dst_image,d)
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
    u, v, r = (np.array(x) for x in (u, v, r))
    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())

    return u, v, r


def radius_graph(
    atoms=None,
    cutoff=5,
    bond_tol=0.5,
    id=None,
    atol=1e-5,
    cutoff_extra=3.5,
):
    """Construct edge list for radius graph."""

    def temp_graph(cutoff=5):
        """Construct edge list for radius graph."""
        cart_coords = torch.tensor(atoms.cart_coords).type(
            torch.get_default_dtype()
        )
        frac_coords = torch.tensor(atoms.frac_coords).type(
            torch.get_default_dtype()
        )
        lattice_mat = torch.tensor(atoms.lattice_mat).type(
            torch.get_default_dtype()
        )
        # elements = atoms.elements
        X_src = cart_coords
        num_atoms = X_src.shape[0]
        # determine how many supercells are needed for the cutoff radius
        recp = 2 * math.pi * torch.linalg.inv(lattice_mat).T
        recp_len = torch.tensor(
            [i for i in (torch.sqrt(torch.sum(recp**2, dim=1)))]
        )
        maxr = torch.ceil((cutoff + bond_tol) * recp_len / (2 * math.pi))
        nmin = torch.floor(torch.min(frac_coords, dim=0)[0]) - maxr
        nmax = torch.ceil(torch.max(frac_coords, dim=0)[0]) + maxr
        # construct the supercell index list

        all_ranges = [
            torch.arange(x, y, dtype=torch.get_default_dtype())
            for x, y in zip(nmin, nmax)
        ]
        cell_images = torch.cartesian_prod(*all_ranges)

        # tile periodic images into X_dst
        # index id_dst into X_dst maps to atom id as id_dest % num_atoms
        X_dst = (cell_images @ lattice_mat)[:, None, :] + X_src
        X_dst = X_dst.reshape(-1, 3)
        # pairwise distances between atoms in (0,0,0) cell
        # and atoms in all periodic image
        dist = torch.cdist(
            X_src, X_dst, compute_mode="donot_use_mm_for_euclid_dist"
        )
        # u, v = torch.nonzero(dist <= cutoff, as_tuple=True)
        # print("u1v1", u, v, u.shape, v.shape)
        neighbor_mask = torch.bitwise_and(
            dist <= cutoff,
            ~torch.isclose(
                dist,
                torch.tensor([0]).type(torch.get_default_dtype()),
                atol=atol,
            ),
        )
        # get node indices for edgelist from neighbor mask
        u, v = torch.where(neighbor_mask)
        # print("u2v2", u, v, u.shape, v.shape)
        # print("v1", v, v.shape)
        # print("v2", v % num_atoms, (v % num_atoms).shape)

        r = (X_dst[v] - X_src[u]).float()
        # gk = dgl.knn_graph(X_dst, 12)
        # print("r", r, r.shape)
        # print("gk", gk)
        v = v % num_atoms
        g = dgl.graph((u, v))
        return g, u, v, r

    g, u, v, r = temp_graph(cutoff)
    while (g.num_nodes()) != len(atoms.elements):
        try:
            cutoff += cutoff_extra
            g, u, v, r = temp_graph(cutoff)
            print("cutoff", id, cutoff)
            print(atoms)

        except Exception as exp:
            print("Graph exp", exp)
            pass
        return u, v, r

    return u, v, r


###
def radius_graph_old(
    atoms=None,
    cutoff=5,
    bond_tol=0.5,
    id=None,
    atol=1e-5,
):
    """Construct edge list for radius graph."""
    cart_coords = torch.tensor(atoms.cart_coords).type(
        torch.get_default_dtype()
    )
    frac_coords = torch.tensor(atoms.frac_coords).type(
        torch.get_default_dtype()
    )
    lattice_mat = torch.tensor(atoms.lattice_mat).type(
        torch.get_default_dtype()
    )
    # elements = atoms.elements
    X_src = cart_coords
    num_atoms = X_src.shape[0]
    # determine how many supercells are needed for the cutoff radius
    recp = 2 * math.pi * torch.linalg.inv(lattice_mat).T
    recp_len = torch.tensor(
        [i for i in (torch.sqrt(torch.sum(recp**2, dim=1)))]
    )
    maxr = torch.ceil((cutoff + bond_tol) * recp_len / (2 * math.pi))
    nmin = torch.floor(torch.min(frac_coords, dim=0)[0]) - maxr
    nmax = torch.ceil(torch.max(frac_coords, dim=0)[0]) + maxr
    # construct the supercell index list

    all_ranges = [
        torch.arange(x, y, dtype=torch.get_default_dtype())
        for x, y in zip(nmin, nmax)
    ]
    cell_images = torch.cartesian_prod(*all_ranges)

    # tile periodic images into X_dst
    # index id_dst into X_dst maps to atom id as id_dest % num_atoms
    X_dst = (cell_images @ lattice_mat)[:, None, :] + X_src
    X_dst = X_dst.reshape(-1, 3)

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic image
    dist = torch.cdist(
        X_src, X_dst, compute_mode="donot_use_mm_for_euclid_dist"
    )
    # u, v = torch.nonzero(dist <= cutoff, as_tuple=True)
    # print("u1v1", u, v, u.shape, v.shape)
    neighbor_mask = torch.bitwise_and(
        dist <= cutoff,
        ~torch.isclose(
            dist, torch.tensor([0]).type(torch.get_default_dtype()), atol=atol
        ),
    )
    # get node indices for edgelist from neighbor mask
    u, v = torch.where(neighbor_mask)
    # print("u2v2", u, v, u.shape, v.shape)
    # print("v1", v, v.shape)
    # print("v2", v % num_atoms, (v % num_atoms).shape)

    r = (X_dst[v] - X_src[u]).float()
    # gk = dgl.knn_graph(X_dst, 12)
    # print("r", r, r.shape)
    # print("gk", gk)
    return u, v % num_atoms, r

def _get_attribute_lookup(atom_features: str = "cgcnn"):
    """Build a lookup array indexed by atomic number."""
    max_z = max(v["Z"] for v in chem_data.values())

    # get feature shape (referencing Carbon)
    template = get_node_attributes("C", atom_features)

    features = np.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features)

        if x is not None:
            features[z, :] = x

    return features
###

def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))
    # print (r1,r1.shape)
    # print (r2,r2.shape)
    # print (bond_cosine,bond_cosine.shape)
    return {"h": bond_cosine}

class Graph(object):
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = True,
        # use_canonize: bool = False,
        use_lattice_prop: bool = False,
        cutoff_extra=3.5,
    ):
        """Obtain a DGLGraph for Atoms object."""
        # print('id',id)
        if neighbor_strategy == "k-nearest":
            edges = nearest_neighbor_edges(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
            )
            u, v, r = build_undirected_edgedata(atoms, edges)
        elif neighbor_strategy == "radius_graph":
            # print('HERE')
            # import sys
            # sys.exit()
            u, v, r = radius_graph(
                atoms, cutoff=cutoff, cutoff_extra=cutoff_extra
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        # elif neighbor_strategy == "voronoi":
        #    edges = voronoi_edges(structure)

        # u, v, r = build_undirected_edgedata(atoms, edges)

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            # if include_prdf_angles:
            #    feat=feat+list(prdf[ii])+list(adf[ii])
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        g = dgl.graph((u, v))
        g.edata["r"] = r
        vol = atoms.volume
        g.ndata["V"] = torch.tensor([vol for ii in range(atoms.num_atoms)])
        g.ndata["coords"] = torch.tensor(atoms.cart_coords)
        if use_lattice_prop:
            lattice_prop = np.array(
                [atoms.lattice.lat_lengths(), atoms.lattice.lat_angles()]
            ).flatten()
            # print('lattice_prop',lattice_prop)
            g.ndata["extra_features"] = torch.tensor(
                [lattice_prop for ii in range(atoms.num_atoms)]
            ).type(torch.get_default_dtype())
        # print("g", g)
        # g.edata["V"] = torch.tensor(
        #    [vol for ii in range(g.num_edges())]
        # )
        # lattice_mat = atoms.lattice_mat
        # g.edata["lattice_mat"] = torch.tensor(
        #    [lattice_mat for ii in range(g.num_edges())]
        # )
        z = copy.deepcopy(node_features)
        g.ndata["atomic_number"] = z
        node_features = node_features.type(torch.IntTensor).squeeze()
        features = _get_attribute_lookup("cgcnn")
        f = torch.tensor(features[node_features]).type(torch.FloatTensor)
        if g.num_nodes() == 1:
            f = f.unsqueeze(0)
        g.ndata["atom_features"] = f
        if compute_line_graph:
            # construct atomistic line graph
            # (nodes are bonds, edges are bond pairs)
            # and add bond angle cosines as edge features
            lg = g.line_graph(shared=True)
            lg.apply_edges(compute_bond_cosines)
            return [g, lg]
        else:
            return g

    @staticmethod
    def from_atoms(
        atoms=None,
        get_prim=False,
        zero_diag=False,
        node_atomwise_angle_dist=False,
        node_atomwise_rdf=False,
        features="basic",
        enforce_c_size=10.0,
        max_n=100,
        max_cut=5.0,
        verbose=False,
        make_colormap=True,
    ):
        """
        Get Networkx graph. Requires Networkx installation.

        Args:
             atoms: jarvis.core.Atoms object.

             rcut: cut-off after which distance will be set to zero
                   in the adjacency matrix.

             features: Node features.
                       'atomic_number': graph with atomic numbers only.
                       'cfid': 438 chemical descriptors from CFID.
                       'cgcnn': hot encoded 92 features.
                       'basic':10 features
                       'atomic_fraction': graph with atomic fractions
                                         in 103 elements.
                       array: array with CFID chemical descriptor names.
                       See: jarvis/core/specie.py

             enforce_c_size: minimum size of the simulation cell in Angst.
        """
        if get_prim:
            atoms = atoms.get_primitive_atoms
        dim = get_supercell_dims(atoms=atoms, enforce_c_size=enforce_c_size)
        atoms = atoms.make_supercell(dim)

        adj = np.array(atoms.raw_distance_matrix.copy())

        # zero out edges with bond length greater than threshold
        adj[adj >= max_cut] = 0

        if zero_diag:
            np.fill_diagonal(adj, 0.0)
        nodes = np.arange(atoms.num_atoms)
        if features == "atomic_number":
            node_attributes = np.array(
                [[np.array(Specie(i).Z)] for i in atoms.elements],
                dtype="float",
            )
        if features == "atomic_fraction":
            node_attributes = []
            fracs = atoms.composition.atomic_fraction_array
            for i in fracs:
                node_attributes.append(np.array([float(i)]))
            node_attributes = np.array(node_attributes)

        elif features == "basic":
            feats = [
                "Z",
                "coulmn",
                "row",
                "X",
                "atom_rad",
                "nsvalence",
                "npvalence",
                "ndvalence",
                "nfvalence",
                "first_ion_en",
                "elec_aff",
            ]
            node_attributes = []
            for i in atoms.elements:
                tmp = []
                for j in feats:
                    tmp.append(Specie(i).element_property(j))
                node_attributes.append(tmp)
            node_attributes = np.array(node_attributes, dtype="float")
        elif features == "cfid":
            node_attributes = np.array(
                [np.array(Specie(i).get_descrp_arr) for i in atoms.elements],
                dtype="float",
            )
        elif isinstance(features, list):
            node_attributes = []
            for i in atoms.elements:
                tmp = []
                for j in features:
                    tmp.append(Specie(i).element_property(j))
                node_attributes.append(tmp)
            node_attributes = np.array(node_attributes, dtype="float")
        else:
            print("Please check the input options.")
        if node_atomwise_rdf or node_atomwise_angle_dist:
            nbr = NeighborsAnalysis(
                atoms, max_n=max_n, verbose=verbose, max_cut=max_cut
            )
        if node_atomwise_rdf:
            node_attributes = np.concatenate(
                (node_attributes, nbr.atomwise_radial_dist()), axis=1
            )
            node_attributes = np.array(node_attributes, dtype="float")
        if node_atomwise_angle_dist:
            node_attributes = np.concatenate(
                (node_attributes, nbr.atomwise_angle_dist()), axis=1
            )
            node_attributes = np.array(node_attributes, dtype="float")

        # construct edge list
        uv = []
        edge_features = []
        for ii, i in enumerate(atoms.elements):
            for jj, j in enumerate(atoms.elements):
                bondlength = adj[ii, jj]
                if bondlength > 0:
                    uv.append((ii, jj))
                    edge_features.append(bondlength)

        edge_attributes = edge_features

        if make_colormap:
            sps = atoms.uniq_species
            color_dict = random_colors(number_of_colors=len(sps))
            new_colors = {}
            for i, j in color_dict.items():
                new_colors[sps[i]] = j
            color_map = []
            for ii, i in enumerate(atoms.elements):
                color_map.append(new_colors[i])
        return Graph(
            nodes=nodes,
            edges=uv,
            node_attributes=np.array(node_attributes),
            edge_attributes=np.array(edge_attributes),
            color_map=color_map,
        )

    def to_networkx(self):
        """Get networkx representation."""
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        for i, j in zip(self.edges, self.edge_attributes):
            graph.add_edge(i[0], i[1], weight=j)
        return graph

    @property
    def num_nodes(self):
        """Return number of nodes in the graph."""
        return len(self.nodes)

    @property
    def num_edges(self):
        """Return number of edges in the graph."""
        return len(self.edges)

    @classmethod
    def from_dict(self, d={}):
        """Constuct class from a dictionary."""
        return Graph(
            nodes=d["nodes"],
            edges=d["edges"],
            node_attributes=d["node_attributes"],
            edge_attributes=d["edge_attributes"],
            color_map=d["color_map"],
            labels=d["labels"],
        )

    def to_dict(self):
        """Provide dictionary representation of the Graph object."""
        info = OrderedDict()
        info["nodes"] = np.array(self.nodes).tolist()
        info["edges"] = np.array(self.edges).tolist()
        info["node_attributes"] = np.array(self.node_attributes).tolist()
        info["edge_attributes"] = np.array(self.edge_attributes).tolist()
        info["color_map"] = np.array(self.color_map).tolist()
        info["labels"] = np.array(self.labels).tolist()
        return info

    def __repr__(self):
        """Provide representation during print statements."""
        return "Graph({})".format(self.to_dict())

    @property
    def adjacency_matrix(self):
        """Provide adjacency_matrix of graph."""
        A = np.zeros((self.num_nodes, self.num_nodes))
        for edge, a in zip(self.edges, self.edge_attributes):
            A[edge] = a
        return A


class Standardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: dgl.DGLGraph):
        """Apply standardization to atom_features."""
        g = g.local_var()
        h = g.ndata.pop("atom_features")
        g.ndata["atom_features"] = (h - self.mean) / self.std
        return g

class StructureDataset(DGLDataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        graphs: Sequence[dgl.DGLGraph],
        line_graphs: Sequence[dgl.DGLGraph],
        targets=None,
        transform=None,
        line_graph=False,
        classification=False,
        ids=None,
    ):
        """Pytorch Dataset for atomistic graphs.

        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        `target_grad`: For fitting forces etc.
        `target_atomwise`: For fitting bader charge on atoms etc.
        """
        self.graphs = graphs
        self.line_graphs = line_graphs
        self.line_graph = line_graph
        self.ids = ids
        self.labels = targets
        self.transform = transform

        if classification:
            self.labels = torch.stack(self.labels, dim=0).view(-1).long()
            print("Classification dataset.", self.labels)

    def __len__(self):
        """Get length."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]
        cif_id = self.ids[idx]
        # id = self.ids[idx]
        if self.transform:
            g = self.transform(g)

        if self.line_graph:
            return g, self.line_graphs[idx], label, cif_id

        return g, label, cif_id

    @staticmethod
    def collate(samples):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels, batch_cif_ids = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.stack(labels, dim=0), batch_cif_ids

    @staticmethod
    def collate_line_graph(samples):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels, batch_cif_ids = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        return (batched_graph, batched_line_graph), torch.stack(labels, dim=0), batch_cif_ids

def load_alignn_data(root_dir: str='data/', task: str=None, config: dict=None):
    assert os.path.exists(root_dir), 'root_dir does not exist!'
    id_prop_file = os.path.join(root_dir, task, 'targets.csv')
    assert os.path.exists(id_prop_file), 'targets.csv does not exist!'
    if os.path.exists(os.path.join(root_dir, task, "alignn_data.pkl")) == True:
        with open(os.path.join(root_dir, task, "alignn_data.pkl"), 'rb') as f:
            data = pickle.load(f)
    else:
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            id_prop_data = [row for row in reader]
        data = []
        for d in id_prop_data:
            cif_id, target = d
            crystal = Atoms.from_cif(os.path.join(root_dir, task, cif_id + '.cif'))
            crystal = crystal.to_dict()
            structure = (
                Atoms.from_dict(crystal) if isinstance(crystal, dict) else crystal
            )
            g = Graph.atom_dgl_multigraph(
                structure,
                cutoff=config["cutoff"],
                cutoff_extra=config["cutoff_extra"],
                atom_features="atomic_number",
                max_neighbors=config["max_neighbors"],
                compute_line_graph=True,
                use_canonize=config["use_canonize"],
                neighbor_strategy="k-nearest",
                id=cif_id,
            )
            data.append([g, torch.Tensor([float(target)]), cif_id])
        with open(os.path.join(root_dir, task, "alignn_data.pkl"), 'wb') as f:
            pickle.dump(data, f)

    return data

def get_alignn_train_val_test_loader(dataset, train_indexs=None,
                              val_indexs=None, test_indexs=None,
                              batch_size=256, return_test=True,
                              num_workers=0, pin_memory=False):

    total_size = len(dataset)
    train_indice = []
    val_indice = []
    test_indice = []
    for i in range(total_size):
        cif_id = dataset[i][2]
        _cif_id = int(cif_id)
        if _cif_id in train_indexs:
            train_indice.append(i)
        elif _cif_id in val_indexs:
            val_indice.append(i)
        elif _cif_id in test_indexs:
            test_indice.append(i)
        else:
            print("Can't find data which cif_id is %d in dataset", _cif_id)


    train_dataset = [dataset[x][0][0] for x in train_indice]
    train_line_graphs = [dataset[x][0][1] for x in train_indice]
    train_targets = [dataset[x][1] for x in train_indice]
    train_ids = [dataset[x][2] for x in train_indice]

    val_dataset = [dataset[x][0][0] for x in val_indice]
    val_line_graphs = [dataset[x][0][1] for x in val_indice]
    val_targets = [dataset[x][1] for x in val_indice]
    val_ids = [dataset[x][2] for x in val_indice]

    test_dataset = [dataset[x][0][0] for x in test_indice]
    test_line_graphs = [dataset[x][0][1] for x in test_indice]
    test_targets = [dataset[x][1] for x in test_indice]
    test_ids = [dataset[x][2] for x in test_indice]

    train_data = StructureDataset(
        train_dataset,
        train_line_graphs,
        targets=train_targets,
        line_graph=True,
        ids=train_ids,
    )
    val_data = StructureDataset(
        val_dataset,
        val_line_graphs,
        targets=val_targets,
        line_graph=True,
        ids=val_ids,
    )
    test_data = StructureDataset(
        test_dataset,
        test_line_graphs,
        targets=test_targets,
        line_graph=True,
        ids=test_ids,
    )

    collate_fn = train_data.collate_line_graph

    train_loader = GraphDataLoader(
        # train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = GraphDataLoader(
        # val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if return_test:
        test_loader = GraphDataLoader(
            # DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )

class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """
    def __init__(
        self,
        atom_input_features: int = 92,
        edge_input_features: int = 80,
        triplet_input_features: int = 40,
        hidden_features: int = 256,
        embedding_features: int = 64,
        alignn_layers: int = 4,
        gcn_layers: int = 4,
        evidential="False",
        classification=False,
    ):
        """Initialize class with number of input features, conv layers."""
        super(ALIGNN, self).__init__()

        self.classification = classification
        self.evidential = evidential
        self.atom_embedding = MLPLayer(
            atom_input_features, hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=edge_input_features,
            ),
            MLPLayer(edge_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=triplet_input_features,
            ),
            MLPLayer(triplet_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    hidden_features,
                    hidden_features,
                )
                for idx in range(alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    hidden_features, hidden_features
                )
                for idx in range(gcn_layers)
            ]
        )

        self.readout = AvgPooling()
        self.out_act = nn.Softplus()

        if self.classification:
            self.fc = nn.Linear(hidden_features, 1)
            self.softmax = nn.Sigmoid()
            # self.softmax = nn.LogSoftmax(dim=1)
        elif self.evidential == "True":
            self.fc = nn.Linear(hidden_features, 4)
        else:
            self.fc = nn.Linear(hidden_features, 1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        min_val = 1e-6
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        r = g.edata["r"]

        bondlength = torch.norm(r, dim=1)
        # mask = bondlength >= self.config.inner_cutoff
        # bondlength[mask]=float(1.1)

        y = self.edge_embedding(bondlength)
        # y = self.edge_embedding(bondlength)
        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)
        # norm-activation-pool-classify
        h = self.readout(g, x)
        h = self.dropout(h)
        out = self.fc(h)

        if self.classification:
            # out = torch.max(out,dim=1)
            out = self.softmax(out)
        if self.evidential=="True":
            if out.shape[0] == 4:
                out = torch.unsqueeze(out, 0)
            out = out.view(out.shape[0], -1, 4)
            mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(out, 1, dim=-1)]
            return mu, self.out_act(logv)+ min_val, self.out_act(logalpha)+ min_val + 1, self.out_act(logbeta)+ min_val
        else:
            return torch.squeeze(out)