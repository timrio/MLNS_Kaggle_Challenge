import networkx as nx
import numpy as np


def Jaccard(graph, edge):

    inter_size = len(list(nx.common_neighbors(graph, edge[0], edge[1])))
    union_size = len(set(graph[edge[0]]) | set(graph[edge[1]]))
    try:
        jacard = inter_size / union_size
    except:
        jacard = np.nan

    return jacard

def AdamicAdar(graph, edge):

    inter_list = nx.common_neighbors(graph, edge[0], edge[1])
    try:
        adamic_adar =  sum( [1/np.log(graph.degree(node)) for node in inter_list])
    except:
        adamic_adar = np.nan
    
    return adamic_adar

def preferential_attachement(graph, edge):
    pa = graph.degree(edge[0]) * graph.degree(edge[1])
    return pa


def common_journal(information_df, node1, node2):
    journal1 = str(information_df[information_df.new_ID==node1].journal_name.values[0])
    journal2 = str(information_df[information_df.new_ID==node2].journal_name.values[0])

    if journal1 == '' or journal2 == '':
        return(-1)
    elif journal1 == journal2:
        return(1)
    else:
        return(0)