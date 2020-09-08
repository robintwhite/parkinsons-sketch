from skimage.io import imread
from skimage.filters import threshold_yen as thresh_func
from skimage.filters import median
from skimage.morphology import disk
from skimage.morphology import label as sk_label #ndimage has function label as well with different call
from skimage.morphology import closing
from skimage.morphology import medial_axis
from skimage.morphology import skeletonize
from skimage import draw
from scipy import ndimage
import numpy as np


k_nn = np.array([[1,1,1],[1,0,1],[1,1,1]])
k_cross = np.array([[0,1,0],[1,0,1],[0,1,0]])

def process_imread(in_path, resize=True):
    """read images, invert and scale them"""
    c_img = 1.0 - imread(in_path, as_gray=True)
    max_dim = np.max(c_img.shape)
    if not resize:
        return c_img
    if c_img.shape == (256, 256):
        return c_img
    if max_dim > 256:
        big_dim = 512
    else:
        big_dim = 256
    """ pad with zeros and center image, sizing to either 256 or 512"""
    out_img = np.zeros((big_dim, big_dim), dtype='float32')
    c_offset = (big_dim - c_img.shape[0]) // 2
    d_offset = c_img.shape[0] + c_offset

    e_offset = (big_dim - c_img.shape[1]) // 2
    f_offset = c_img.shape[1] + e_offset
    out_img[c_offset:d_offset, e_offset:f_offset] = c_img[:(d_offset - c_offset), :(f_offset - e_offset)]
    return out_img


def read_and_thresh(in_path, resize=True):
    c_img = process_imread(in_path, resize=resize)
    c_img = (255*c_img).clip(0, 255).astype('uint8')
    c_img = median(c_img, disk(1))
    c_thresh = thresh_func(c_img)
    return c_img>c_thresh


def label_sort(in_img, cutoff=0.01):
    total_cnt = np.sum(in_img>0)
    lab_img = sk_label(in_img)
    new_image = np.zeros_like(lab_img)
    remap_index = []
    for k in np.unique(lab_img[lab_img>0]):
        cnt = np.sum(lab_img==k) # get area of labelled object
        if cnt>total_cnt*cutoff:
            remap_index+=[(k, cnt)]
    sorted_index = sorted(remap_index, key=lambda x: -x[1]) # reverse sort - largest is first
    for new_idx, (old_idx, idx_count) in enumerate(sorted_index, 1): #enumerate starting at id 1
        new_image[lab_img==old_idx] = new_idx
    return new_image


def stroke_thickness_img(in_img):
    skel, distance = medial_axis(in_img, return_distance=True)
    skeleton = skeletonize(in_img)
    # Distance to the background for pixels of the skeleton
    return distance * skeleton


def stroke_thickness(in_img):
    skel, distance = medial_axis(in_img, return_distance=True)
    skeleton = skeletonize(in_img)
    # Distance to the background for pixels of the skeleton
    dist = distance * skeleton
    return dist[dist>0]


def skeleton_drawing(in_img):
    edge_img = skeletonize(in_img, method='lee')
    return (edge_img > 0).astype(np.uint8)


def nearest_neighbours(in_img, k):
    a = np.where(in_img != 0, 1, 0)
    a_nn = ndimage.convolve(a, k, mode='constant', cval=0.0)
    return (a_nn * a).astype(np.uint8)


def number_of_end_points(in_img, k_nn):
    img_skel = skeleton_drawing(in_img)
    img_nn = nearest_neighbours(img_skel, k_nn)
    img_ep = (img_nn == 1).astype(np.uint8)
    return np.sum(img_ep[img_ep != 0])


def create_branches(in_img, k_nn):
    # input: skeleton image 0, 1
    a = np.where(in_img != 0, 1, 0)
    a_nn = nearest_neighbours(in_img, k_nn)
    intersections = np.where(a_nn >= 3, 1, 0)
    a_no_branches = np.where(intersections, 0, 1) * a # mask at intersections
    a_keep = sk_label(a_no_branches) # remove branches
    return a_keep#((a_keep + intersections) > 0).astype(np.uint8)


def sum_nearest_neighbours(in_img, k):
    a = np.where(in_img > 0, 1, 0)
    a_nn = ndimage.convolve(in_img, k, mode='constant', cval=0.0)
    return (a_nn * a).astype(np.uint8)


def sum_pixels(in_img):
    return np.sum(np.where(in_img > 0, 1, 0))


def get_edges(labelled_img, nodes):
    img_x, img_y = labelled_img.shape
    z_nn = labelled_img
    edges = []
    for x,y in nodes:
        nn = np.asarray([z_nn[(x-1) % img_x, (y+1) % img_y],z_nn[x+0, (y+1) % img_y],z_nn[(x+1) % img_x, (y+1) % img_y],
                         z_nn[(x-1) % img_x,  y+0],                                  z_nn[(x+1) % img_x,  y+0],
                         z_nn[(x-1) % img_x, (y-1) % img_y],z_nn[x+0, (y-1) % img_y],z_nn[(x+1) % img_x, (y-1) % img_y]])
        edges.append(nn[nn != 0])

    return edges


def clean_labelled(in_img, label_img):
    k_nn = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    k_cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    img_2_conn_sum = sum_nearest_neighbours(in_img, k_nn)
    img_1_conn_sum = sum_nearest_neighbours(in_img, k_cross)

    tmp_nodes = np.where(img_2_conn_sum >= 8)
    tmp_nodes = list(zip(*tmp_nodes))
    edges_around_nodes = get_edges(label_img, tmp_nodes)
    for i, (x, y) in enumerate(tmp_nodes):
        if len(edges_around_nodes[i]) > 0:
            label_img[x, y] = edges_around_nodes[i]

    tmp_nn = (img_1_conn_sum >= 3) * (img_1_conn_sum <= 5)  # incorrect intersections
    img_cleaned_nn = np.where(tmp_nn == True, 2, in_img)  # reset to 'corrected' NN to value of 2
    return img_cleaned_nn, label_img


def number_of_intersection_points(in_img, k_nn):
    img_skel = skeleton_drawing(in_img)
    branch_label_img = create_branches(img_skel, k_nn)
    img_nn = nearest_neighbours(img_skel, k_nn)
    img_cleaned_nn, img_cleaned_label = clean_labelled(img_nn, branch_label_img)
    img_inter = (img_cleaned_nn == 3).astype(np.uint8)
    return np.sum(img_inter[img_inter != 0])


# for each edge, check each node
#### Get which nodes are connected by edges
def get_edges_as_nodes(edge_int, node_edges):
    edges = []
    for edge_id in edge_int:
        connection = []
        for node_id in range(len(node_edges)): #j,k in enumerate try
            #print([k for k, e in enumerate(node_edges[j]) if e == i])
            e = np.where(node_edges[node_id] == edge_id)[0]
            if e.size > 0:
                #print(e, edge_id, node_id+1) #position around node, edge id, connecting to node j
                connection.append(node_id+1)
        if len(connection) == 1:
            edges.append(tuple((connection[0],connection[0]))) #TEST
        else:
            edges.append(tuple(connection))
    return edges
    #edge ID is index (+1), nodes connected by edge. Need only 2. See above


def get_edge_length(label_img, edge_list):
    edge_length = []
    for edge in edge_list:
        length = np.sum(np.where(label_img == edge, 1, 0))
        edge_length.append(length)
    return edge_length


def get_cleaned_nn_and_label(in_img, k_nn):
    img_skel = skeleton_drawing(in_img)
    branch_label_img = create_branches(img_skel, k_nn)
    img_nn = nearest_neighbours(img_skel, k_nn)
    img_cleaned_nn, img_cleaned_label = clean_labelled(img_nn, branch_label_img)
    return img_cleaned_nn, img_cleaned_label


def get_wave_source_dest(all_ep):
    all_ep_x, all_ep_y = zip(*all_ep)
    source = all_ep_y.index(min(all_ep_y))
    dest = all_ep_y.index(max(all_ep_y))
    return source, dest


def get_spiral_source_dest(al_ep, img_shape):
    # approximation by node closest to the center of image
    c_x, c_y = img_shape
    c_x, c_y = c_x // 2, c_y // 2
    all_ep_x, all_ep_y = zip(*al_ep)
    source = al_ep.index(min(al_ep, key=lambda c: (c[0] - c_x)**2 + (c[1] - c_y)**2))
    dest = al_ep.index(max(al_ep, key=lambda c: (c[0] - c_x)**2 + (c[1] - c_y)**2))
    return source, dest


def draw_nodes(img, nodes, r=2, labels = None):
    img_copy = np.copy(img)
    # source node
    if labels == None:
        for i, (x,y) in enumerate(nodes):
            rr,cc = draw.disk((x,y),r)
            img_copy[rr % img.shape[0], cc % img.shape[1]] = int(i+1)
    else:
        for i, (x,y) in enumerate(nodes):
            rr,cc = draw.disk((x,y),r)
            img_copy[rr % img.shape[0], cc % img.shape[1]] = int(labels[i])
    return img_copy


def get_weights_and_edges(G, path):
    edges_x, edges_y, ident, weights = zip(*list(G.edges(data='weight', keys=True)))
    edge_arr_x = np.array(list(edges_x))
    edge_arr_y = np.array(list(edges_y))

    weight_total = 0
    edge_ident = []
    for i, (u, v) in enumerate(zip(path[:-1], path[1:])):
        index = np.where(np.logical_and(edge_arr_x == u, edge_arr_y == v))[0]
        if len(index) == 0:
            index = np.where(np.logical_and(edge_arr_x == v, edge_arr_y == u))[0]
        # weight_total += max([weights[x] for x in index])
        maxi = 0
        for x in index:
            tmp = weights[x]
            if tmp >= maxi:
                maxi = tmp
                indi = ident[x]

        weight_total += maxi
        edge_ident.append(indi)
    return weight_total, edge_ident


# get edge labels from given node path
def get_edge_labels(G, node_path, edge_ident):
    edge_list = []
    for i, (u, v) in enumerate(zip(node_path[:-1], node_path[1:])):
        edge_list.append(G.edges[u,v,edge_ident[i]]['edge_id'])
    return edge_list


def get_node_labels(node_path, edge_ident):
    i = 0
    node_list = []
    for ident in edge_ident:
        if ident == 0:
            node_list.append(node_path[i])
            i += 1
        else:
            i = i+1
            node_list.append(node_path[i])
            node_list.append(node_path[i]+1)
            i += 1
    node_list.append(node_path[-1])
    node_list = list(set(node_list))
    return node_list


# https://stackoverflow.com/questions/24806174/is-there-an-opposite-inverse-to-numpy-pad-function
def unpad(dens, pad):
    """
    Input:  dens   -- np.ndarray(shape=(nx,ny,nz))
            pad    -- np.array(px,py,pz)

    Output: pdens -- np.ndarray(shape=(nx-px,ny-py,nz-pz))
    """

    nx, ny, nz = dens.shape
    pl = pad // 2
    pr = pad - pl

    pdens = dens[pl[0]:nx-pr[0],
            pl[1]:ny-pr[1],
            pl[2]:nz-pr[2]]

    return pdens

