import networkx as nx
import community
import matplotlib.pyplot as plt
import numpy as np

A = np.loadtxt('F:\\Pycharm\\PycharmProjects\\DGSIC\\data\\simNet40.txt', dtype=int)
G = nx.from_numpy_matrix(A)
NodeNum = len(A)

def showGraph():
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    node_labels = nx.get_node_attributes(G, 'name1')
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, 'name2')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def CM(G):
    if len(G) == 0:
        return 0
    return max([len(n) for n in nx.connected_components(G)])
def SortByDGSIC(G):
    G1 = nx.Graph(G)
    inNum = len(G1.nodes())
    rank=np.zeros(inNum,int)
    for i in range(inNum):
        rank[i]=-1
    remove_num = 0
    while remove_num < 0.5*len(A):
        gcc = G1.subgraph(max(nx.connected_components(G1), key=len))
        Agcc = nx.adjacency_matrix(gcc).todense()
        NodeNum=len(Agcc)
        for shell in range(NodeNum):
            partition = community.best_partition(gcc)
            comm = np.zeros(inNum,int)
            commValue=list(partition.values())
            commNum=list(partition.keys())
            len_comm=len(commNum)
            for j in range(NodeNum):
                comm[j]=-1
            for j in range(len_comm):
                comm[commNum[j]]=commValue[j]
            maxComm = 0
            for node1 in gcc.nodes():
                if comm[node1] > maxComm:
                    maxComm =comm[node1]
            commNum=maxComm+1
            CI = np.zeros(inNum,float)
            comm1 = np.zeros(shape=(commNum,inNum))
            for j in range(commNum):
                for k in gcc.nodes:
                    comm1[j][k]=-1
            for i in range(commNum):
                count=0
                G2 = nx.Graph()
                for j in gcc.nodes:
                    if comm[j]==i:
                        comm1[i][count]=j
                        G2.add_node(j)
                        count+=1
                for j in gcc.nodes:
                    aj = int(comm1[i][j])
                    for k in gcc.nodes:
                        ak=int(comm1[i][k])
                        if A[aj][ak]==1 and aj!=-1 and ak!=-1:
                            G2.add_edge(j, k)
                BC = nx.betweenness_centrality(G2)
                BCL = list(BC.values())
                for q in range(count):
                    c=int(comm1[i][q])
                    CI[c]=BCL[q]
            CO=np.zeros(inNum,float)
            for node in gcc.nodes:
                count=0
                for neighbor in gcc.neighbors(node):
                    if comm[node] != comm[neighbor]:
                        count+=1
                CO[node]=count
            SIC=np.zeros(inNum,float)
            for j in range(inNum):
                SIC[j]=-1
            for node in gcc.nodes:
                SIC[node]=pow(CO[node],(1+CI[node]))
            remove_list = np.argsort(SIC)
            rank1 = remove_list[::-1]
            print(remove_list)
        num = rank1[0]
        print(remove_num)
        print(num)
        rank[remove_num]=num
        G1.remove_node(num)
        remove_num=remove_num+1
    print(rank)
    return rank

SortByDGSIC(G)

