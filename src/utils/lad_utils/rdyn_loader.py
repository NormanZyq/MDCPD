import pickle


def load_edgelist(file_path):
    # 似乎需要返回一个snapshot的列表？
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()

    graphs_directed = []
    for g in data['graphs']:
        graphs_directed.append(g.to_directed())

    return graphs_directed
