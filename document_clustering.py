from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from vector_models import generate_tfidf_ngram_ls, generate_tfidf_ngram_rs, generate_tfidf_ngram_combined
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd
import pickle, json


def perform_multidimensional_scaling(tfidf_matrix):
    dist = 1 - cosine_similarity(tfidf_matrix)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    return xs, ys


def generate_kmeans_clustering_ls():
    num_clusters = 4
    tfidf_matrix = generate_tfidf_ngram_ls()
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    xs, ys = perform_multidimensional_scaling(tfidf_matrix)

    jfile = open('data/ls_session_enum_tfidf.json', 'r')
    label_map = json.load(jfile).items()
    titles = [x[0] for x in sorted(label_map, key=lambda x: x[1])]
    jfile.close()

    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
    cluster_names = {0: 'Cluster 1',
                     1: 'Cluster 2',
                     2: 'Cluster 3',
                     3: 'CLuster 4',
                     4: 'Cluster 5'}

    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))
    groups = df.groupby('label')
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')
    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)
    plt.savefig('data/ls_kmeans_clusters.png', dpi=200)


def generate_kmeans_clustering_rs():
    num_clusters = 4
    tfidf_matrix = generate_tfidf_ngram_rs()
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    xs, ys = perform_multidimensional_scaling(tfidf_matrix)

    jfile = open('data/rs_session_enum_tfidf.json', 'r')
    label_map = json.load(jfile).items()
    titles = [x[0] for x in sorted(label_map, key=lambda x: x[1])]
    jfile.close()

    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
    cluster_names = {0: 'Cluster 1',
                     1: 'Cluster 2',
                     2: 'Cluster 3',
                     3: 'Cluster 4',
                     4: 'Cluster 5'}

    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))
    groups = df.groupby('label')
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')
    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)
    plt.savefig('data/rs_kmeans_clusters.png', dpi=200)


def generate_kmeans_clustering_combined():
    num_clusters = 5
    tfidf_matrix = generate_tfidf_ngram_combined()
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    xs, ys = perform_multidimensional_scaling(tfidf_matrix)

    jfile = open('data/combined_session_enum_tfidf.json', 'r')
    label_map = json.load(jfile).items()
    titles = [x[0] for x in sorted(label_map, key=lambda x: x[1])]
    jfile.close()

    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
    cluster_names = {0: 'Cluster 1',
                     1: 'Cluster 2',
                     2: 'Cluster 3',
                     3: 'Cluster 4',
                     4: 'Cluster 5'}

    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))
    groups = df.groupby('label')
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')
    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)
    plt.savefig('data/combined_kmeans_clusters.png', dpi=200)


def generate_ward_clustering_ls():
    tfidf_matrix = generate_tfidf_ngram_ls()
    jfile = open('data/ls_session_enum_tfidf.json', 'r')
    label_map = json.load(jfile).items()
    titles = [x[0] for x in sorted(label_map, key=lambda x: x[1])]
    jfile.close()

    linkage_matrix = linkage(tfidf_matrix, 'ward')  # define the linkage_matrix using ward clustering pre-computed distances
    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=titles)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout
    plt.savefig('data/ls_ward_clusters.png', dpi=200)


def generate_ward_clustering_rs():
    tfidf_matrix = generate_tfidf_ngram_rs()
    jfile = open('data/rs_session_enum_tfidf.json', 'r')
    label_map = json.load(jfile).items()
    titles = [x[0] for x in sorted(label_map, key=lambda x: x[1])]
    jfile.close()

    linkage_matrix = linkage(tfidf_matrix, 'ward')  # define the linkage_matrix using ward clustering pre-computed distances
    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=titles)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout
    plt.savefig('data/rs_ward_clusters.png', dpi=200)


def generate_ward_clustering_combined():
    tfidf_matrix = generate_tfidf_ngram_combined()
    jfile = open('data/combined_session_enum_tfidf.json', 'r')
    label_map = json.load(jfile).items()
    titles = [x[0] for x in sorted(label_map, key=lambda x: x[1])]
    jfile.close()

    linkage_matrix = linkage(tfidf_matrix, 'ward')  # define the linkage_matrix using ward clustering pre-computed distances
    fig, ax = plt.subplots(figsize=(15, 30))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=titles)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout
    plt.savefig('data/combined_ward_clusters.png', dpi=200)


generate_kmeans_clustering_ls()