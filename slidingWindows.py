import numpy as np
import matplotlib.image as mpimage
import skimage.color as color

def greyscaleReadImage(imageName):
    return np.array(color.rgb2gray(mpimage.imread(imageName)))

def slidingWindows(size, image, step):
    image.transpose()
    numWindows = (((image.shape[0]-size)/step)+1) * (((image.shape[1]-size)/step)+1)
    allWindows = np.zeros((numWindows,400))
    counter = 0
    for i in range(0,image.shape[0]-size,step):
        for j in range(0,image.shape[1]-size,step):
            allWindows[counter,:] = np.resize(rebin(image[i:i+size,j:j+size],(20,20)),400)
            counter +=1
    np.save("slidingWindows.npy",allWindows)
            
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

slidingWindows(80, greyscaleReadImage("target.jpg"),2)
