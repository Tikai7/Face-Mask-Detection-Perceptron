from PIL import Image
from tqdm import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
import Ann


with_mask_test = glob.glob("dataset/test_with/*.jpg")
without_mast_test = glob.glob("dataset/test_without/*.jpg")
with_mask = glob.glob("dataset/with_mask/*.jpg")
without_mask = glob.glob("dataset/without_mask/*.jpg")


def preprocess_data(folder_name, classe):
    array = convert(folder_name)
    # plt.imshow(array[0])
    # plt.show()
    # ---- Normalisation des données méthode MinMax ------ X = X-Xmin / Xmax - Xmin
    array = reshape_array(array)

    if classe == 1:
        y_train = np.ones((array.shape[0], 1))
    else:
        y_train = np.zeros((array.shape[0], 1))

    return array, y_train

# ------- Convertie TOUT notre dataset de .jpg en tableau Numpy


def convert(folder_name):
    folder_name_list = []

    for image in tqdm(folder_name):
        data = Image.open(image)
        gs_image = data.convert(mode='L')
        gs_image_resized = gs_image.resize((64, 64))
        gs_image_resized.save(image)
        folder_name_list.append(plt.imread(image))

    return np.array(folder_name_list)

# ------- Reshape les tableaux numpy pour avoir 2 Dimensions


def reshape_array(array):
    return array.reshape(array.shape[0], -1)


# ------- Fait le preprocessing des données(Reshape + convertion)
X_WM_train, Y_WM_train = preprocess_data(with_mask, 1)
X_WTM_train, Y_WTM_train = preprocess_data(without_mask, 0)
X_WM_test, Y_WM_test = preprocess_data(with_mask_test, 1)
X_WTM_test, Y_WTM_test = preprocess_data(without_mast_test, 0)


print(X_WM_train.shape)
print(X_WTM_train.shape)
print(X_WM_test.shape)
print(X_WTM_test.shape)

# ------- Creation du train_set et test_set
X_train = np.concatenate((X_WM_train, X_WTM_train), axis=0)
Y_train = np.concatenate((Y_WM_train, Y_WTM_train), axis=0)
X_test = np.concatenate((X_WM_test, X_WTM_test), axis=0)
Y_test = np.concatenate((Y_WM_test, Y_WTM_test), axis=0)

# ----------------------------------------- Normalisation
X_test = X_test/X_train.max()
X_train = X_train/X_train.max()


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# -------- Lancement de l'entrainement
W, b = Ann.Ann(X_train, Y_train, X_test, Y_test, 0.01, 100)

with open("params.txt", "w") as fichier:
    for row in W:
        np.savetxt(fichier, row)

with open("params_b.txt", "w") as fichier_1:
    np.savetxt(fichier_1, b)
