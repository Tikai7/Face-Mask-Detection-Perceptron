import Ann
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob


def preprocess_image():
    folder_name_list = []

    images = input("Donnez le nom de l'image : ")
    ext = input("Donnez nom extention : ")
    im = glob.glob(f"images/{images}.{ext}")

    for image in im:
        data = Image.open(image)
        gs_image = data.convert(mode='L')
        gs_image_resized = gs_image.resize((64, 64))
        gs_image_resized.save(image)
        folder_name_list.append(plt.imread(image))

    X = np.array(folder_name_list)
    X = X.reshape(X.shape[0], -1)/X.max()
    W = np.loadtxt("params.txt")
    W = W.reshape((X.shape[1], 1))
    b = np.array([0.23048032])
    return X, W, b


X, W, b = preprocess_image()

print("Prediction en cours...")
y_pred, proba = Ann.predict(X, W, b)

if y_pred:
    print(f"Masque non detecté : {proba}")
else:
    print(f"Masqué detecté {1-proba}")
