import os
import re
import networkx as nx
import pandas as pd
from os.path import join
from src.data.processor.pathways.gmt_reader import GMT

# reactome_base_dir = REACTOM_PATHWAY_PATH
ppi_base_dir = os.path.join(os.getcwd(), 'data/protein')
gene_base_dir = os.path.join(os.getcwd(), 'data/genes')
relations_file_name = '9606.protein.links.full.v12.0.txt'
protein_name = '9606.protein.info.v12.0.txt'
gene_file_name = 'breast_expressed_genes_and_cancer_genes_10000.csv'


def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def complete_network(G, n_leveles=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles) #get subgraph of neighbors centered at node root within 5 radius. 
    # terminal_nodes = [n for n, d in sub_graph.degree() if d == 1]
    # terminal_nodes = [n for n, d in sub_graph.degree() if d < 5] #8
    terminal_nodes = [n for n, d in nx.bfs_tree(sub_graph, 'root').out_degree() if d == 0]
    distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph


def get_nodes_at_level(net, distance):
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))

    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)

def get_top_node(net, id2name, name2id):
    # return top node based on special rule.
    top_num = max(len(net.node) // 100 * 3, 2)
    valid_node_name = [id2name[id] for id in net.node]

    # degree
    degree_dict = dict(net.degree())
    sorted_nodes_by_degree = sorted(degree_dict, key=lambda x:degree_dict[x], reverse=True)
    
    # betweenness centrality
    betweenness_dict = nx.approximate_current_flow_betweenness_centrality(net)
    sorted_nodes_by_betweenness_centrality = sorted(betweenness_dict, key=lambda x: betweenness_dict[x], reverse=True)

    # contribution from genecard
    df_gene = pd.read_csv(join(gene_base_dir, gene_file_name))
    gene_list = df_gene['genes'].to_list()
    sorted_nodes_by_contribution = [name2id[gene] for gene in gene_list if gene in valid_node_name]

    return list(
        set(sorted_nodes_by_degree[:top_num]) &
        set(sorted_nodes_by_betweenness_centrality[:top_num]) &
        set(sorted_nodes_by_contribution[:top_num])
    )

def get_layers_from_net(net, n_levels):
    layers = []
    dis2nodes = {}
    for i in range(n_levels + 1):
        dis2nodes[i] = get_nodes_at_level(net, i)
    
    for i in range(n_levels):
        dict = {}
        for n in dis2nodes[i]:
            n_name = re.sub('_copy.*', '', n)
            next = net.neighbors(n)
            dict[n_name] = [
                re.sub('_copy.*', '', nex) for nex in next 
                if nex in dis2nodes[i+1]
            ]
        layers.append(dict)
    return layers


class Protein():

    def __init__(self, genes):
        self.protein_name = self.load_names(genes)
        self.name2id = self.protein_name.set_index(['name'])['protein_id'].to_dict()
        self.id2name = self.protein_name.set_index(['protein_id'])['name'].to_dict()
        self.hierarchy = self.load_hierarchy()

    def load_names(self, genes):
        filename = join(ppi_base_dir, protein_name)
        df = pd.read_csv(filename, sep='\t')
        # df.columns = ['protein_id', 'name', 'protein_size', 'annotation']
        df.rename(columns={
                '#string_protein_id': 'protein_id', 'preferred_name': 'name'
            }, 
            inplace=True
        )
        # remove rows that don't contain gene in genes set
        df = df[df['name'].isin(genes)]
        df.reset_index(drop=True, inplace=True)
        return df

    def load_hierarchy(self):
        if not os.path.exists(join(ppi_base_dir, 'selected_links.txt')):
            filename = join(ppi_base_dir, relations_file_name)
            df = pd.read_csv(filename, sep=' ', low_memory=False)

            # remove rows that don't contain gene in genes set
            select_genes_id = list(set(self.id2name.keys()))
            df = df[
                (df['protein1'].isin(select_genes_id)) &
                (df['protein2'].isin(select_genes_id))
            ]

            print(f"The minimum is {df['combined_score'].min()} and the maximum is {df['combined_score'].max()}", )
            # remove lower confident edges
            confidence = 0.7
            max_value = df['combined_score'].max()
            min_value = df['combined_score'].min()
            threshold = (max_value - min_value) * confidence + min_value
            df = df[df['combined_score' ] > threshold]

            # keep one edge only
            df['flag'] = df.apply(
                lambda row:
                    row['protein1'] + row['protein2'] if row['protein1'] < row['protein2'] else  row['protein2'] + row['protein1'],
                axis=1
            )
            df = df[~df['flag'].duplicated(keep='first')]
            df.drop('flag', axis=1, inplace=True)
            print(df['combined_score'].value_counts())

            df['node1'] = df['protein1'].replace(self.id2name)
            df['node2'] = df['protein2'].replace(self.id2name)
            df.to_csv(join(ppi_base_dir, 'selected_links.txt'), index=False)
        else:
            filename = join(ppi_base_dir, 'selected_links.txt')
            df = pd.read_csv(filename)
        
        # df.columns = ['child', 'parent']
        df = df[['protein1', 'protein2', 'combined_score']]
        df.rename(columns={'protein1': 'child', 'protein2': 'parent'}, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


class ProteinNetwork():

    def __init__(self, genes):
        self.protein = Protein(genes)  # low level access to reactome pathways and genes
        self.netx = self.get_networkx()

    def get_roots(self):

        roots = get_nodes_at_level(self.netx, distance=1)
        return roots

    # get a undirected graph representation of the ppi hierarchy
    def get_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.protein.hierarchy
        net = nx.from_pandas_edgelist(hierarchy, 'child', 'parent')
        components = list(nx.connected_components(net))
        largest_component = max(components, key=len)
        largest_subgraph = net.subgraph(largest_component).copy()
        net = largest_subgraph
        net.name = 'ppi'
        roots = get_top_node(net, self.protein.id2name, self.protein.name2id)
        # roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):

        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')

        return G

    def get_completed_network(self, n_levels):
        G = complete_network(self.netx, n_leveles=n_levels)
        return G

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = complete_network(G, n_leveles=n_levels)
        return G

    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]

        # get the last layer (genes level)
        # terminal_nodes = [n for n, d in net.degree() if d < 5]  # set of terminal pathways
        terminal_nodes = get_nodes_at_level(net, n_levels)

        filter_gene = list(self.protein.name2id.values())
        hierarchy = self.protein.hierarchy[['child', 'parent']]

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            # construct relationships between leaf nodes and filter genes
            neig_gene = list(set(self.netx.adj[pathway_name].keys()))
            # neig_gene1 = hierarchy[hierarchy['child'] == pathway_name]['parent'].unique()
            # neig_gene2 = hierarchy[hierarchy['parent'] == pathway_name]['child'].unique()
            genes = [self.protein.id2name[gene] for gene in neig_gene if gene in filter_gene]
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes

        layers.append(dict)
        return layers
