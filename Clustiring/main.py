# for loading/processing the images
# for everything else
import os
import pickle
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
# models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import cv2

PATH = r"../GAN/data/captcha_dataset_1"
#PATH = r"../../GAN_Captcha_Solver/data/captcha_dataset"
# change the working directory to the path where the images are located
os.chdir(PATH)

# image = cv2.imread('yml_00000905052022Y2kNk4Di3e.png', cv2.IMREAD_GRAYSCALE)
# print(len(image.shape))

# this list holds all the image filename
captchas = []

# creates a ScandirIterator aliased as files
with os.scandir() as files:
    # loops through each file in the directory
    for file in files:
        if file.name.endswith('.png'):
            # adds only the image files to the flowers list
            captchas.append(file.name)

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


data = {}
P = r"vectors.pkl"

# lop through each image in the dataset
for captcha in captchas:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(captcha, model)
        data[captcha] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except FileNotFoundError:
        with open(P, 'wb') as file:
            pickle.dump(data, file)

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))
feat = feat.reshape(-1, 4096)

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
CLUSTER_NUMBER = 30
kmeans = KMeans(n_clusters=CLUSTER_NUMBER, random_state=22)
kmeans.fit(x)

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)


# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize=(25, 25))
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')


# this is just incase you want to see which value for k might be the best
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)

    sse.append(km.inertia_)
# PATH = r"../../GAN_Captcha_Solver/data/captcha_dataset"
# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
for i in range(0, CLUSTER_NUMBER):
    os.makedirs(f"../clusters_/cluster_{i}", exist_ok=True)
    for image in groups[i]:
        # view_cluster(i)
        copyfile(image, f"../clusters_/cluster_{i}/{image}")
plt.show()


