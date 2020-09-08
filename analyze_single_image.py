import networkx as nx
import pandas as pd
from pathlib import Path
from utils.process_images import *
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 160
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')

k_nn = np.array([[1,1,1],[1,0,1],[1,1,1]])
k_cross = np.array([[0,1,0],[1,0,1],[0,1,0]])

data_dir = Path(r'D:\Docs\Python_code\ParkinsonsSketch\178338_401677_bundle_archive\drawings')

draw_df = pd.DataFrame({'path': list(data_dir.glob('*/*/*/*.png'))})
draw_df['img_id'] = draw_df['path'].map(lambda x: x.stem)
draw_df['disease'] = draw_df['path'].map(lambda x: x.parent.stem)
draw_df['validation'] = draw_df['path'].map(lambda x: x.parent.parent.stem)
draw_df['activity'] = draw_df['path'].map(lambda x: x.parent.parent.parent.stem)
print(draw_df.shape, 'images loaded')

# TODO: Error when break in curve with no path to dest node
# Error images for single curve 'V15PO03' 'V14PE01'
c_id = 'V10HE03' #draw_df.sample()['img_id'].values[0]# V03PO05   'V04HE01'#'V07PO02' #'V04PE01'
print(f'Image processing: {c_id}')
img_path = draw_df.loc[draw_df['img_id'] == c_id]['path'].values[0]
activity = draw_df.loc[draw_df['img_id'] == c_id]['activity'].values[0]
print(f'Actvity: {activity}')
thresh_img = read_and_thresh(img_path, resize=False)
clean_img = closing(label_sort(thresh_img) > 0, disk(1))
img_skel = skeleton_drawing(clean_img)
#print(sum_pixels(img_skel))
fig, m_axs = plt.subplots(1, 1, figsize = (5,5), dpi=150)
m_axs.imshow(clean_img, interpolation='none')
m_axs.axis('off')
plt.show()
fig.tight_layout()

# nearest neighbours
img_nn = nearest_neighbours(img_skel, k_nn)
branch_label_img = create_branches(img_skel, k_nn)
img_cleaned_nn, img_cleaned_label = clean_labelled(img_nn, branch_label_img)

fig, m_axs = plt.subplots(1, 1, figsize = (5,5), dpi=150)
m_axs.imshow(img_cleaned_label, interpolation='none', norm=mpl.colors.PowerNorm(gamma=0.25))
m_axs.axis('off')
plt.show()
fig.tight_layout()

nodes_3 = np.where(img_cleaned_nn == 3) # nodes position
nodes_3 = list(zip(*nodes_3))
nodes_1 = np.where(img_cleaned_nn == 1) # nodes position
nodes_1 = list(zip(*nodes_1))
all_nodes = nodes_1 + nodes_3

node_edges = get_edges(img_cleaned_label, all_nodes) #edges at each node
# correction for any empty values in node_edges, but need node_edges unchanged
node_edges_for_edge_int = [x for x in node_edges if len(x) > 0]
edge_int = list(range(1,max(map(max, node_edges_for_edge_int))+1))
edges = get_edges_as_nodes(edge_int, node_edges)
edge_lengths = get_edge_length(img_cleaned_label, edge_int)

# create graph
G = nx.MultiGraph()
edge_number = G.add_edges_from(edges)
# add ID and weight
for i,(j,k) in enumerate(edges):
    G[j][k][edge_number[i]]['edge_id'] = i+1
    G[j][k][edge_number[i]]['weight'] = edge_lengths[i]

if activity == 'wave':
    source, dest = get_wave_source_dest(nodes_1) # ALL EP
else:
    source, dest = get_spiral_source_dest(nodes_1, clean_img.shape)

source = source + 1
dest = dest + 1 # for G, index starts at 1

# draw source and destination node targets
img_copy = draw_nodes(img_cleaned_nn, [all_nodes[source - 1], all_nodes[dest - 1]])

fig, m_axs = plt.subplots(1, 1, figsize = (5,5), dpi=150)
m_axs.imshow(img_copy, interpolation='none', norm=mpl.colors.PowerNorm(gamma=0.25))
m_axs.axis('off')
plt.show()
fig.tight_layout()

# contains the weight and edge_ident number to be used when getting edge id
# paths_info = list(map(np.array,zip(*[get_weights_and_edges(G,path) for i,path in enumerate(nx.all_simple_paths(G, source, dest)) if i % 2 == 0])))
# paths_info[0] == weights
# paths_info[1] == edge_ident_id

simple_path_edge_labels = []
simple_path_weights_list = []
img_node_list = []
for j,path in enumerate(nx.all_simple_paths(G, source, dest)):
    if j % 2 == 0:
#         print(j)
#         print(path, len(path))
#         print(list(zip(path[:-1], path[1:])), len(list(zip(path[:-1], path[1:]))))
        path_info = get_weights_and_edges(G,path)
#         print(path_info[1], len(path_info[1]))
        simple_path_edge_labels.append(get_edge_labels(G, path, path_info[1]))
        simple_path_weights_list.append(path_info[0])
        img_node_list.append(get_node_labels(path, path_info[1]))

# display each single curve image
# i = 0
# fig, m_axs = plt.subplots(9, 2, figsize=(18, 18), dpi=300)
# for c_ax, c_row in zip(m_axs.flatten(), simple_path_edge_labels):
#     img_cleaned_sketch = np.isin(img_cleaned_label, c_row).astype(np.uint8)
#     img_cleaned_sketch = img_cleaned_label * img_cleaned_sketch
#
#     c_ax.imshow(img_cleaned_sketch, interpolation='none', cmap='magma', norm=mpl.colors.PowerNorm(gamma=0.5))
#     c_ax.axis('off')
#     c_ax.set_title(f'{i},{simple_path_weights_list[i]}')
#     i += 1
#
# plt.show()

heavy_path_idx = np.argmax(simple_path_weights_list, axis=0)
heaviest_path_nodes = list(nx.all_simple_paths(G, source, dest))[heavy_path_idx]
print('Nodes:', heaviest_path_nodes)
heaviest_path_edge_labels = simple_path_edge_labels[heavy_path_idx]
print('Edge Labels:', heaviest_path_edge_labels) #Label on image
print('Length:', simple_path_weights_list[heavy_path_idx])


# cleaned labeled
img_cleaned_sketch = np.isin(img_cleaned_label, heaviest_path_edge_labels).astype(np.uint8)
img_cleaned_sketch = img_cleaned_label * img_cleaned_sketch

fig, m_axs = plt.subplots(1,1, figsize=(5,5), dpi=150)
m_axs.imshow(img_cleaned_sketch, interpolation='none', cmap='nipy_spectral')
m_axs.axis('off')
plt.show()
# fig.savefig(r'images\label_edges_cleaned.png', dpi=150, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format=None,
#         transparent=False, bbox_inches=None, pad_inches=0.0, metadata=None)
fig.tight_layout()


img_skel_single = np.copy(img_cleaned_sketch)
# add back in intersection points to image
for i in img_node_list[heavy_path_idx]:
    x,y = all_nodes[i-1]
    img_skel_single[x,y] = 1
img_skel_single = np.where(img_skel_single > 0, 255,0).astype(np.uint8)

fig, m_axs = plt.subplots(1,1, figsize=(5,5), dpi=150)
m_axs.imshow(img_skel_single, interpolation='none', cmap='nipy_spectral')
m_axs.axis('off')
plt.show()
fig.tight_layout()

# draw nodex of path
img_copy = draw_nodes(img_skel_single, [all_nodes[x-1] for x in img_node_list[heavy_path_idx]], labels=img_node_list[heavy_path_idx])
fig, m_axs = plt.subplots(1, 1, figsize = (5,5), dpi=150)
m_axs.imshow(img_copy, interpolation='none', cmap='nipy_spectral', norm=mpl.colors.PowerNorm(gamma=0.25))
m_axs.axis('off')
plt.show()
# fig.savefig(r'images\label_edges_cleaned_nodes.png', dpi=150, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format=None,
#         transparent=False, bbox_inches=None, pad_inches=0.0, metadata=None)
fig.tight_layout()

## create potential time series of curve with cellular automata
# padding = (2,2)
# tmp_img = np.copy(img_skel_single)
# tmp_img = np.pad(tmp_img, padding, 'constant', constant_values=(0, 0))
# ca_img = np.zeros_like(tmp_img)
# source_xy = all_nodes[source-1]
# dest_xy = all_nodes[dest-1]
#
# ca_xy = source_xy
# t = 0
# ca_img[ca_xy] = 1
# for m in range(100):
#     t += 1
#     x = ca_xy[0] + padding[0]
#     y = ca_xy[1] + padding[1]
#     # check nn. Calc distance of non zero cells. take min index. Get new x,y
#     nn = np.asarray([tmp_img[(x-1), (y+1)], tmp_img[(x+0), (y+1)], tmp_img[(x+1), (y+1)],
#                      tmp_img[(x-1), (y+0)]                       , tmp_img[(x+1), (y+0)],
#                      tmp_img[(x-1), (y-1)], tmp_img[(x+0), (y-1)], tmp_img[(x+1), (y-1)]])
#     nn = np.where(nn > 0, 1,0)
#     print(nn)
#     pos_arr = np.asarray([[(x + 1), (y + 1)], [(x + 0), (y + 1)], [(x + 1), (y + 1)],
#                           [(x - 1), (y + 0)],                     [(x + 1), (y + 0)],
#                           [(x - 1), (y - 1)], [(x + 0), (y - 1)], [(x + 1), (y - 1)]])
#     new_x, new_y = pos_arr[np.argwhere(nn > 0)[0]][0]
#     print(ca_xy, new_x,new_y)
#     ca_img[new_x,new_y] = t
#     ca_xy = (new_x - padding[0], new_y - padding[0])
#
# # ca_img[ca_xy] = 100
# fig, m_axs = plt.subplots(1, 1, figsize = (5,5), dpi=150)
# m_axs.imshow(ca_img, interpolation='none', cmap='nipy_spectral', norm=mpl.colors.PowerNorm(gamma=0.25))
# m_axs.axis('off')
# plt.show()
