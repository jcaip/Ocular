import os
import numpy
import matplotlib.image as mpimg
import skimage.color as color

def grabImageFiles(): #returns an array that contains all box files for processing.
    imageFiles =[]
    files = os.listdir(os.curdir)
    for f in files:
        if 'gif' in f:
            imageFiles.append(f)
    return imageFiles

def greyscaleImageFile(imgFile):
    return color.rgb2gray(mpimg.imread(imgFile))

def imagePreProcessing():
    
    allImages = grabImageFiles()
    trainingData = numpy.zeros(((len(allImages)),400))
    trainingLabels = numpy.chararray(len(allImages))    
    for i in range(0,len(allImages)):
        trainingData[i,:] = numpy.resize(numpy.array(greyscaleImageFile(allImages[i])),400)
        trainingLabels[i] = allImages[i][5]
    numpy.save("trainingData.npy",trainingData)
    numpy.save("trainingLabels.npy",trainingLabels)

imagePreProcessing()

