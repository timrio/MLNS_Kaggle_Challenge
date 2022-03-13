import networkx as nx
from collections import Counter


def create_articles_graph(train_set,information_df):
    """
    in this case, it is a directed graph
    """
    nodes = list(information_df.new_ID.unique())

    # we only keep existing edges
    edges = set(train_set.query("label==1").apply(lambda x: (x.node1,x.node2), axis = 1))

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return(G)

def create_co_authorship_graph(information_df, authors2idx):
    """
    undirected graph
    """
    pre_edges = list(information_df.authors_id.apply(lambda x : [(x[i],x[j]) for i in range(len(x)) for j in range(len(x)) if i>j]))
    authors_edges = [edge for list_edge in pre_edges for edge in list_edge]
    authors_edges_dict = Counter(authors_edges)

    G = nx.Graph()
    G.add_nodes_from(authors2idx.values())
    G.add_weighted_edges_from([(a,b,weight) for (a,b),weight in authors_edges_dict.items()])

    return(G)

def create_authors_co_citation_graph(train_set, information_df, authors2idx):
    """
    directed graph
    """
    co_citation = (train_set
    .query('label==1')
    .merge(information_df,how = 'left', left_on = ["node1"], right_on = 'new_ID')
    [['node1',	'node2', 'authors_id']]
    .rename(columns = {"authors_id":'authors1'})
    .merge(information_df,how = 'left', left_on = ["node2"], right_on = 'new_ID')
    [['node1',	'node2', 'authors1',"authors_id"]]
    .rename(columns = {"authors_id":'authors2'})
    )

    co_citation_list = list(co_citation.apply(lambda x: [(auth1,auth2) for auth1 in x.authors1 for auth2 in x.authors2 if auth1!=auth2  if auth1!='' if auth2!='' ], axis = 1))
    edges_list = [edge for edge_list in co_citation_list for edge in edge_list]
    authors_citation_edges_dict = Counter(edges_list)

    G = nx.DiGraph()
    G.add_nodes_from(authors2idx.values())
    G.add_weighted_edges_from([(a,b,weight) for (a,b),weight in authors_citation_edges_dict.items()])

    return(G)