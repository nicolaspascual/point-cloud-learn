import re
from collections import namedtuple
import pandas as pd
from io import StringIO

# Cluster = namedtuple('Cluster', [
#         'cluster_id', 'x', 'y', 'z', 'n_points', 'n_order', 'volume', 'positive_volume',
#         'negative_volume', 'area_2d', 'klass', 'trust_level', 'p_all_mean', 'p_all_sigma',
#         'p_1_mean', 'p_1_sigma', 'p_0_mean', 'p_0_sigma', 'p_2_mean', 'p_2_sigma',
#         'n_clusters_origin', 'n_clusters_destination', 'color_origin_index', 'red_origin_mean',
#         'red_origin_sigma', 'green_origin_mean', 'green_origin_sigma', 'blue_origin_mean',
#         'blue_origin_sigma', 'intensity_origin_mean', 'intensity_origin_sigma',
#         'color_reference_index', 'red_reference_mean', 'red_reference_sigma',
#         'green_reference_mean', 'green_reference_sigma', 'blue_reference_mean',
#         'blue_reference_sigma', 'intensity_destination_mean', 'intensity_destination_sigma',
#         'origin_file', 'destination_file'
#     ])
# ClusterRepresentativePoint = namedtuple('ClusterRepresentativePoint', [])
Cluster = namedtuple('Cluster', ['id'] + [f'label_{i}' for i in range(31)])
ClusterRepresentativePoint = namedtuple('ClusterRepresentativePoint', ['clusterId'] + [f'label_{i}' for i in range(51)])

def load_data(file_name):
    with open(file_name, 'r') as file:
        clusters = []
        representative_points = []
        current_cluster_id = -1
        for line in file.readlines()[::-1]:
            if re.compile(r'^#cluster: \d+').match(line):
                cluster = parse_cluster(line)
                current_cluster_id = cluster.id
                clusters.append(cluster)
            else:
                representative_points.append(
                    parse_point(line, current_cluster_id)
                )
    cluster_str = StringIO('\n'.join([','.join(cluster) for cluster in clusters]))
    repre_str = StringIO('\n'.join([','.join(point) for point in representative_points]))
    return pd.read_csv(cluster_str, index_col=0, names=Cluster._fields),\
        pd.read_csv(repre_str, names=ClusterRepresentativePoint._fields)


def parse_cluster(line):
    line = line.replace('#cluster:', '')
    fields = re.compile(r'\s+').split(line.strip())
    return Cluster(*fields)

def parse_point(line, cluster_id):
    fields = [cluster_id] + re.compile(r'\s+').split(line.strip())
    return ClusterRepresentativePoint(*fields)