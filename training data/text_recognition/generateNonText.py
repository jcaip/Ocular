import argparse 
import numpy as np
import matplotlib.image as mpimage
import skimage.color as color

def greyscaleReadImage(imageName):
    return np.array(color.rgb2gray(mpimage.imread(imageName)))

def slidingWindows(size, imageFile, step):
    image = loadImage(imageFile)
    image.transpose()
    numWindows = (((image.shape[0]-size)/step)+1) * (((image.shape[1]-size)/step)+1)
    allWindows = np.zeros((numWindows,400))
    counter = 0
    for i in range(0,image.shape[0]-size,step):
        for j in range(0,image.shape[1]-size,step):
            allWindows[counter,:] = np.resize(rebin(image[i:i+size,j:j+size],(20,20)),400)
            counter +=1
    allBool = np.zeros(numWindows)
    boolFile = imageFile+".bool.npy"
    saveFile = imageFile+".npy"

    np.save(boolFile,allBool)
    np.save(saveFile,allWindows)
            
def loadImage(imageFileName):
    return np.array(color.rgb2gray(mpimage.imread(imageFileName)))


def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

parser = argparse.ArgumentParser(description = "get non-text data")
parser.add_argument('windowSize',metavar='W', type=int,help='The size of the sliding window')
parser.add_argument('windowStep',metavar='S', type=int,help='The size of the sliding window step')
parser.add_argument('filename', metavar='filename', type=str,help='filename')
args = parser.parse_args();
slidingWindows(args.windowSize,args.filename,args.windowStep)

