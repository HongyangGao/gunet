import networkx as nx
from collections import defaultdict


def get_indic(ind_file):
    with open(ind_file) as f:
        lines = f.readlines()
    indic = {}
    g_sizes = [0]*int(lines[-1])
    for i, line in enumerate(lines):
        g_id = int(line)-1
        indic[i] = g_id
        g_sizes[g_id] += 1
    return indic, g_sizes


def get_labels(label_file):
    with open(label_file) as f:
        lines = f.readlines()
    return {i: int(line)-1 for i, line in enumerate(lines)}


def trans_graphs(g_file, A_file, labels, id_dict, g_sizes):
    with open(g_file, 'w') as f:
        f.write('%s\n' % len(labels))
    with open(A_file) as f:
        lines = f.readlines()
    edges_list = [defaultdict(list) for _ in range(len(labels))]

    for line in lines:
        i, j = list(map(int, line.split(',')))
        g_id = id_dict[i-1]
        edges_list[g_id][i-1].append(j-1)
        edges_list[g_id][j-1].append(i-1)  # use set to remove duplicate

    acc_g_size = 0
    for i in range(len(labels)):
        print('=========> graph', i, acc_g_size)
        label = labels[i]
        edges = edges_list[i]
        write_graph(g_file, acc_g_size, label, edges)
        acc_g_size += g_sizes[i]


def write_graph(g_file, acc_g_size, label, edges):
    with open(g_file, 'a+') as f:
        f.write('%s %s\n' % (len(edges), label))
        num_nodes = len(edges)
        for i in range(len(edges)):
            neighbors = sorted(set(edges[acc_g_size+i]))
            neighbors = [n_id - acc_g_size for n_id in neighbors]
            neighbors = [n_id for n_id in neighbors if n_id < num_nodes]
            f.write('0 %s %s\n' % (
                len(neighbors), ' '.join(map(str, neighbors))))


def load_data(g_file):
    print('loading data')
    label_dict = {}
    feat_dict = {}

    acc_line_num = 0
    with open(g_file, 'r') as f:
        n_g = int(f.readline().strip())
        acc_line_num += 1
        for i in range(n_g):
            row = f.readline().strip().split()
            acc_line_num += 1
            num_node, label = [int(w) for w in row]
            if label not in label_dict:
                mapped = len(label_dict)
                label_dict[label] = mapped
            g = nx.Graph()
            node_tags = []
            n_edges = 0
            for j in range(num_node):
                g.add_node(j)
                row = f.readline().strip().split()
                acc_line_num += 1
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    row = [int(w) for w in row]
                else:
                    row = [int(w) for w in row[:tmp]]
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
            assert len(g) == num_node


if __name__ == '__main__':
    name = 'REDDIT-MULTI-12K'
    g_file = '%s.txt' % name
    id_dict, g_sizes = get_indic('%s/%s_graph_indicator.txt' % (name, name))
    labels = get_labels('%s/%s_graph_labels.txt' % (name, name))
    trans_graphs(
        g_file, '%s/%s_A.txt' % (name, name), labels,
        id_dict, g_sizes)
    load_data(g_file)
    # print(id_dict)
