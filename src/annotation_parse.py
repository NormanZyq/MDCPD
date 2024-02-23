#%%
import pandas as pd


anno_path = 'data/annotation/annotation0108.xlsx'
data_path = 'data/annotation/data0107.csv'
data = pd.read_csv(data_path)
annotation = pd.read_excel(anno_path, usecols='[A:D]')
names, auto, manual = annotation['节点名称'], annotation['自动标注'], annotation['人工结果']
annotation_dict = {}
for i in range(len(names)):
    name = names[i]
    if auto[i] == '-1':
        target = manual[i]
    else:
        target = auto[i]
    annotation_dict[name] = target      # name->with what
nodes = set(data['source']) | set(data['target'])
annotated = set()
unknown_nodes = set()

communities = []       # a list of set

#%%
for n in nodes:
    if n not in annotation_dict:
        unknown_nodes.add(n)
        continue
    for c in communities:
        if annotation_dict[n] in c:
            c.add(n)
            annotated.add(n)
            break
        else:
            communities.append({n})
            annotation.add(n)
            break




