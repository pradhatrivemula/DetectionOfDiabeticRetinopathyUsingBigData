import numpy as np
import cv2
import logging
from sklearn import metrics
import glob
from sklearn import svm
from matplotlib import pyplot as plt
import mahotas
import sklearn
import sklearn.preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score




xfinal =np.ones(152128)
yfinal = np.ones(1)

#to extract features from images using hu_moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

#to extract features from images using haralick
def fd_haralick(image):    
    # convert the image to grayscale		
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
 

#loading images from the un effected folder and extracting features from them
#assign the label 0 to the un effected images for classification; the labels are stored in yfinal
filepath='/Users/pradhatrivemula/Desktop/un effected/*'    
nm=0
for filepath in glob.iglob(filepath):
        x=cv2.imread(filepath)
        x1=cv2.cvtColor(x,cv2.COLOR_BGR2YCR_CB)
        Y,Cr,Cb=cv2.split(x1)
        Y=cv2.equalizeHist(Y)
        x2=cv2.merge((Y,Cr,Cb))
        image=cv2.cvtColor(x2,cv2.COLOR_YCR_CB2BGR)
        nm=nm+1
        x123 = np.hstack([ fd_haralick(image), fd_hu_moments(image)])
        b=len(x123)
        c=152128-b
        nx123 = np.pad(x123, (0, c), 'constant')
        xfinal=np.vstack((xfinal,nx123))
        y=[0]
        yfinal=np.vstack((yfinal,y))


#loading images from the effected folder and extracting features from them
#assign the label 1 to the effected images for classification; the labels are stored in yfinal
filepath='/Users/pradhatrivemula/Desktop/effected/*'
nm=0
for filepath in glob.iglob(filepath):
        x=cv2.imread(filepath)
        x1=cv2.cvtColor(x,cv2.COLOR_BGR2YCR_CB)
        Y,Cr,Cb=cv2.split(x1)
        Y=cv2.equalizeHist(Y)
        x2=cv2.merge((Y,Cr,Cb))
        image=cv2.cvtColor(x2,cv2.COLOR_YCR_CB2BGR)
        nm=nm+1
        x123 = np.hstack([ fd_haralick(image), fd_hu_moments(image)])
        b=len(x123)
        c=152128-b
        nx123 = np.pad(x123, (0, c), 'constant')
        xfinal=np.vstack((xfinal,nx123))
        y=[1]
        yfinal=np.vstack((yfinal,y))

#loading images from the test folder and extracting features from them
xtest =np.ones(152128)
filepath='/Users/pradhatrivemula/Desktop/test/*'
nm=0
for filepath in glob.iglob(filepath):
        x=cv2.imread(filepath)
        x1=cv2.cvtColor(x,cv2.COLOR_BGR2YCR_CB)
        Y,Cr,Cb=cv2.split(x1)
        Y=cv2.equalizeHist(Y)
        x2=cv2.merge((Y,Cr,Cb))
        image=cv2.cvtColor(x2,cv2.COLOR_YCR_CB2BGR)
        nm=nm+1
        #cv2.imwrite('/home/pradhatri/openface/demos/diseasefilter/PATIENT '+str(fn)+'/'+str(nm)+'.JPG', image)
        #print('retinal images of test filtered and saved.....')
        #print(len(xtest))
        x123 = np.hstack([ fd_haralick(image), fd_hu_moments(image)])
        b=len(x123)
        c=152128-b
        nx123 = np.pad(x123, (0, c), 'constant')
        xtest=np.vstack((xtest,nx123))
         
# load the actual labels of the test dataset into y_test   
with open('/Users/pradhatrivemula/Desktop/test_retina_img_details.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([filename,filename[-5]])
        y_test.append(float(filename[-5]))
	csvFile.close()

#fitting the features and the labels into the random Forest Classifier and training the model
model = RandomForestClassifier()
# fit the model on the whole dataset
model.fit(xfinal, np.ravel(yfinal,order='C'))

#predicting the labels for the test dataset
ypred = model.predict(xtest)

print("Accuracy:",metrics.accuracy_score(y_test, ypred ))
print("average precision score:",average_precision_score(y_test, ypred ))

