# import cv2
# import requests
# import numpy as np
# import pickle
# import os
# import base64

# url = "http://localhost:8080/api/gethog"
# def img2vec(img):
#     resized_img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
#     v, buffer = cv2.imencode(".jpg", resized_img)
#     img_str = base64.b64encode(buffer).decode('utf-8')
#     image_data_string = "data:image/jpeg;base64," + img_str
#     params = {"item_str": image_data_string}
#     response = requests.get(url, params=params)
#     return response.json()


# path = '../train/'
# carvectors = []
# y = []
# for sub in os.listdir(path):
#     for fn in os.listdir(os.path.join(path,sub)):
#         img_file_name = os.path.join(path,sub)+"/"+fn
#         img = cv2.imread(img_file_name)
#         res = img2vec(img)
#         vec = list(res["message"])
#         vec.append(sub)
#         carvectors.append(vec)
#     y.append(sub)

# print(y)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(carvectors, y, test_size=0.2)
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# clf = DecisionTreeClassifier()
# clf = clf.fit(X_train,y_train)

# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# write_path = "carvectors.pkl"
# pickle.dump(carvectors, open(write_path,"wb"))
# print("data preparation is done")

import cv2
import requests
import numpy as np
import pickle
import os
import base64

url = "http://localhost:5000 /api/gethog"
def img2vec(img):
    resized_img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
    v, buffer = cv2.imencode(".jpg", resized_img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_data_string = "data:image/jpeg;base64," + img_str
    params = {"item_str": image_data_string}
    # response = requests.get(url, params=params)
    return params
demo = img2vec("15.jpg")
print(demo)

# path = '..\\Cars Dataset\\train' 
# carvectors = []
# for sub in os.listdir(path):
#     for fn in os.listdir(os.path.join(path,sub)):
#         img_file_name = os.path.join(path,sub)+"/"+fn
#         img = cv2.imread(img_file_name)
#         res = img2vec(img)
#         vec = list(res["message"])
#         vec.append(sub)
#         carvectors.append(vec)
# write_path = "carvectors_train.pkl"
# pickle.dump(carvectors, open(write_path,"wb"))
# print("data preparation is done")