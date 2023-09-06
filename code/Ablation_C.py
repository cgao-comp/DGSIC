import math
import networkx as nx
import community
from community import community_louvain
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import pearsonr
import copy
from matplotlib.pyplot import MultipleLocator

A = np.loadtxt('F:\\Pycharm\\PycharmProjects\\DGSIC\\data\\simNet40.txt', dtype=int)
G = nx.from_numpy_matrix(A)
NodeNum = len(A)


def showGraph():
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    node_labels = nx.get_node_attributes(G, 'name1')
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, 'name2')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def CM(G):
    if len(G) == 0:
        return 0
    return max([len(n) for n in nx.connected_components(G)])


def SortBySIC(G):
    partition = community.best_partition(G)
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0
    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]

    comm = list(partition.values())
    maxComm = 0
    for node1 in range(NodeNum):
        if comm[node1] > maxComm:
            maxComm = comm[node1]
    commNum = maxComm + 1

    CI = np.zeros(NodeNum, float)
    comm1 = np.zeros(shape=(commNum, NodeNum))
    for j in range(commNum):
        for k in range(NodeNum):
            comm1[j][k] = -1
    for i in range(commNum):
        count = 0
        G1 = nx.Graph()
        for j in range(NodeNum):
            if comm[j] == i:
                comm1[i][count] = j
                G1.add_node(j)
                count += 1

        for j in range(NodeNum):
            aj = int(comm1[i][j])
            for k in range(NodeNum):
                ak = int(comm1[i][k])
                if A[aj][ak] == 1 and aj != -1 and ak != -1:
                    G1.add_edge(j, k)

        BC = nx.betweenness_centrality(G1)
        # print(BC)
        BCL = list(BC.values())
        for q in range(count):
            c = int(comm1[i][q])
            CI[c] = BCL[q]

    CO = np.zeros(NodeNum, float)
    for node in G.nodes():
        count = 0
        for neighbor in G.neighbors(node):
            if comm[node] != comm[neighbor]:
                count += 1
        CO[node] = count

    SIC = np.zeros(NodeNum, float)
    for node in G.nodes():
        SIC[node] = pow(CO[node], (1 + CI[node]))
    rank = np.argsort(SIC)
    rank1 = rank[::-1]
    print(SIC)
    print('SIC' + str(rank1))
    return rank1


def SortByDGSIC(G):
    G1 = nx.Graph(G)
    inNum = len(G1.nodes())
    rank = np.zeros(inNum, int)
    for i in range(inNum):
        rank[i] = -1
    remove_num = 0
    while remove_num < 0.9 * len(A):
        gcc = G1.subgraph(max(nx.connected_components(G1), key=len))
        Agcc = nx.adjacency_matrix(gcc).todense()
        NodeNum = len(Agcc)
        for shell in range(NodeNum):
            partition = community.best_partition(gcc)
            size = float(len(set(partition.values())))
            pos = nx.spring_layout(gcc)
            count = 0
            comm = np.zeros(inNum, int)
            # print(NodeNum)
            commValue = list(partition.values())
            commNum = list(partition.keys())
            len_comm = len(commNum)
            # print(commNum)
            for j in range(NodeNum):
                comm[j] = -1
            for j in range(len_comm):
                comm[commNum[j]] = commValue[j]

            maxComm = 0
            for node1 in gcc.nodes():
                if comm[node1] > maxComm:
                    maxComm = comm[node1]
            commNum = maxComm + 1

            CI = np.zeros(inNum, float)
            comm1 = np.zeros(shape=(commNum, inNum))
            for j in range(commNum):
                for k in gcc.nodes:
                    comm1[j][k] = -1
            for i in range(commNum):
                count = 0
                G2 = nx.Graph()
                for j in gcc.nodes:
                    if comm[j] == i:
                        comm1[i][count] = j
                        G2.add_node(j)
                        count += 1
                # print(count)
                # print(G1.nodes)
                for j in gcc.nodes:
                    aj = int(comm1[i][j])
                    for k in gcc.nodes:
                        ak = int(comm1[i][k])
                        if A[aj][ak] == 1 and aj != -1 and ak != -1:
                            G2.add_edge(j, k)
                BC = nx.betweenness_centrality(G2)
                # print(BC)
                BCL = list(BC.values())
                for q in range(count):
                    c = int(comm1[i][q])
                    CI[c] = BCL[q]
            CO = np.zeros(inNum, float)
            for node in gcc.nodes:
                count = 0
                for neighbor in gcc.neighbors(node):
                    if comm[node] != comm[neighbor]:
                        count += 1
                CO[node] = count
            SIC = np.zeros(inNum, float)
            for j in range(inNum):
                SIC[j] = -1
            for node in gcc.nodes:
                SIC[node] = pow(CO[node], (1 + CI[node]))
            remove_list = np.argsort(SIC)
            rank1 = remove_list[::-1]

        num = rank1[0]

        rank[remove_num] = num
        G1.remove_node(num)
        remove_num = remove_num + 1

    print(SIC)
    print('DGSIC' + str(rank))
    return rank


def SortByGSIC(G):
    G1 = nx.Graph(G)
    inNum = len(G1.nodes())
    rank = np.zeros(inNum, int)
    for i in range(inNum):
        rank[i] = -1
    remove_num = 0
    while remove_num < 0.9 * len(A):

        gcc = G1.subgraph(max(nx.connected_components(G1), key=len))

        Agcc = nx.adjacency_matrix(gcc).todense()
        NodeNum = len(Agcc)

        for shell in range(NodeNum):

            partition = community.best_partition(gcc)
            size = float(len(set(partition.values())))
            pos = nx.spring_layout(gcc)
            count = 0
            comm = np.zeros(inNum, int)
            # print(NodeNum)
            commValue = list(partition.values())
            commNum = list(partition.keys())
            len_comm = len(commNum)
            # print(commNum)
            for j in range(NodeNum):
                comm[j] = -1
            for j in range(len_comm):
                comm[commNum[j]] = commValue[j]

            maxComm = 0
            for node1 in gcc.nodes():
                if comm[node1] > maxComm:
                    maxComm = comm[node1]
            commNum = maxComm + 1

            CI = np.zeros(inNum, float)
            comm1 = np.zeros(shape=(commNum, inNum))
            for j in range(commNum):
                for k in gcc.nodes:
                    comm1[j][k] = -1
            for i in range(commNum):
                count = 0
                G2 = nx.Graph()
                for j in gcc.nodes:
                    if comm[j] == i:
                        comm1[i][count] = j
                        G2.add_node(j)
                        count += 1

                for j in gcc.nodes:
                    aj = int(comm1[i][j])
                    for k in gcc.nodes:
                        ak = int(comm1[i][k])
                        if A[aj][ak] == 1 and aj != -1 and ak != -1:
                            G2.add_edge(j, k)
                BC = nx.betweenness_centrality(G2)
                # print(BC)
                BCL = list(BC.values())
                for q in range(count):
                    c = int(comm1[i][q])
                    CI[c] = BCL[q]
            CO = np.zeros(inNum, float)
            for node in gcc.nodes:
                count = 0
                for neighbor in gcc.neighbors(node):
                    if comm[node] != comm[neighbor]:
                        count += 1
                CO[node] = count

            SIC = np.zeros(inNum, float)
            for j in range(inNum):
                SIC[j] = -1
            for node in gcc.nodes:
                SIC[node] = pow(CO[node], (1 + CI[node]))

            remove_list = np.argsort(SIC)
            rank1 = remove_list[::-1]

        num = rank1[0]

        rank[remove_num] = num
        G1.remove_node(num)
        remove_num = remove_num + 1

    print(rank)

    return rank


def SortByDSIC(G):
    G1 = nx.Graph(G)
    inNum = len(G1.nodes())

    rank = np.zeros(inNum, int)
    for shell in range(inNum):

        partition = community.best_partition(G1)
        size = float(len(set(partition.values())))
        pos = nx.spring_layout(G1)
        count = 0
        for com in set(partition.values()):
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]

        comm = np.zeros(NodeNum, int)

        commValue = list(partition.values())
        commNum = list(partition.keys())
        len_comm = len(commNum)

        for j in range(NodeNum):
            comm[j] = -1
        for j in range(len_comm):
            comm[commNum[j]] = commValue[j]
        # print(comm)
        maxComm = 0
        for node1 in G1.nodes:
            if comm[node1] > maxComm:
                maxComm = comm[node1]
        commNum = maxComm + 1
        # print(commNum)
        CI = np.zeros(NodeNum, float)
        comm1 = np.zeros(shape=(commNum, NodeNum))
        for j in range(commNum):
            for k in G1.nodes:
                comm1[j][k] = -1
        for i in range(commNum):
            count = 0
            G2 = nx.Graph()
            for j in G1.nodes:
                if comm[j] == i:
                    comm1[i][count] = j
                    G2.add_node(j)
                    count += 1

            for j in G1.nodes:
                aj = int(comm1[i][j])
                for k in G1.nodes:
                    ak = int(comm1[i][k])
                    if A[aj][ak] == 1 and aj != -1 and ak != -1:
                        G2.add_edge(j, k)

            BC = nx.betweenness_centrality(G2)

            BCL = list(BC.values())
            for q in range(count):
                c = int(comm1[i][q])
                CI[c] = BCL[q]

        CO = np.zeros(NodeNum, float)
        for node in G1.nodes():
            count = 0
            for neighbor in G1.neighbors(node):
                if comm[node] != comm[neighbor]:
                    count += 1
            CO[node] = count

        SIC = np.zeros(NodeNum, float)
        for j in range(NodeNum):
            SIC[j] = -1
        for node in G1.nodes():
            SIC[node] = pow(CO[node], (1 + CI[node]))

        remove_list = np.argsort(SIC)
        rank1 = remove_list[::-1]

        num = rank1[0]

        rank[shell] = num
        G1.remove_node(num)

    print(SIC)
    print('DSIC' + str(rank))
    return rank

rank1 = SortBySIC(G)
rank2 = SortByDSIC(G)
rank3 = SortByDGSIC(G)

def IC_model(G, Infect):

    G1=nx.Graph(G)
    Num = len(Infect)

    Step = list()
    step_by_step = list()
    step_by_step_temp = [Num]
    c = np.zeros(NodeNum, float)

    for i in range(len(Infect)):
        G1.remove_node(Infect[i])
    for ite in range(4000):
        G2 = nx.Graph(G1)

        Infect_temp=Infect.copy()
        Infect_all=Infect.copy()
        while len(Infect_temp)!=0:
            temp=list()
            for node in Infect_temp:
                for nei in G.neighbors(node):
                    if nei not in Infect_all and random.uniform(0,1)<(1/(G2.degree(nei)+1)):
                        temp.append(nei)
                        G2.remove_node(nei)
                        Infect_all.append(nei)

            Infect_temp=temp

            step_by_step_temp.append(len(Infect_all))

        step_by_step.append(step_by_step_temp.copy())

        step_by_step_temp.clear()

        step_by_step_temp.append(len(Infect))


        Step.append(len(Infect_all))


    max_length=0

    for step in step_by_step:
        if len(step)>max_length:
            max_length=len(step)
    #每一次
    for i in range(len(step_by_step)):

        temp=step_by_step[i][len(step_by_step[i])-1]
        for j in range(max_length-len(step_by_step[i])):

            step_by_step[i].append(temp)

    temp=step_by_step[0]
    for i in range(1, len(step_by_step)):

        temp=np.sum([temp, step_by_step[i]], axis=0)


    return [(i/4000)/G.number_of_nodes() for i in temp]




def getIC(G,num):
    Infect1= []
    Infect2 =  []
    Infect3 =  []

    for i in range(num):
        Infect1.append(rank1[i])
        Infect2.append(rank2[i])
        Infect3.append(rank3[i])

    print(IC_model(G,Infect1))
    print(IC_model(G, Infect2))
    print(IC_model(G,Infect3))





getIC(G,16)