import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random

dataset = pd.read_csv('E:/celeba-dataset/list_attr_celeba.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,[0,21]].values
print(y[0][1])

DATADIR="E:/celeba-dataset/img_align_celeba/img_align_celeba"
categories=["Female","Male"]

training_data=[]
img_size=50

def create_training_data():
    i=0
    m=0
    f=0
    for img in os.listdir(DATADIR):
        try:
            if(y[i][1]==1 or (y[i][1]==-1 and f<20946)):
                img_array=cv2.imread(os.path.join(DATADIR,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(img_size,img_size))
                if y[i][1]==1 :
                    gender=1
                    m=m+1
                else:
                    gender=0
                    f=f+1
                # i=i+1
                training_data.append([new_array,gender])
        except Exception as e:
            pass
        finally:
            if(i%100==0):
                print(i)
            i=i+1
            if i==50000:
                print("breaking")
                break
    print(m)
    print(f)

create_training_data()
print(training_data[0])


random.shuffle(training_data)

X=[]
y=[]

for a,b in training_data:
    X.append(a)
    y.append(b)

X=np.array(X).reshape(-1,img_size,img_size,1)
# X=X/255.0
f=open("X_pickle1.pkl","wb")
pickle.dump(X,f)
f.close()

w=open("y_pickle1.pkl","wb")
pickle.dump(y,w)
w.close()