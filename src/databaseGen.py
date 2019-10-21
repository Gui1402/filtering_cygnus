import numpy as np
import sys
sys.path.insert(0,'../trackGeneration')
import tGenfunctions
import pickle
import time
import h5py 
from datetime import date

def imageCreating(img,alpha,nMean,nStd,shape):
    #iMax = 135   ## image max
    #iPed = 99    ## image pedestal

    imgT         = img > 0.1 # a visual truth only to apply poisson to signal pixels
    img          = img/img.max()   ## normalize by max
    ##Contrast applying
    img          = alpha*img  ## apply contrast in track
    ##poisson distribuition
    row,col      = np.where(imgT==True)  ## get positions 
    img[row,col] = np.random.poisson(lam = img[row,col],size = (1,len(img[row,col]))) ## poisson draw of each pixel
    ## noise generating
    noiseGen     = np.random.normal(loc = nMean,scale = nStd)
    rebin        = 1
    xx,yy        = np.meshgrid(range(int(1024-shape[0]/2),int(1024+(shape[0]/2)),rebin),range(int(1024-shape[0]/2),int(1024+shape[0]/2),rebin))
    noiseGen     = noiseGen[xx,yy]
    imNoised     = noiseGen + img ## output image
    
    
    return imNoised   


def loadDatapy2(filename):
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='latin1')[0] 
    return d    
##INPUTS
lGen       = loadDatapy2('../trackGeneration/fitL.pickle')  ## draw a width
cGen       = loadDatapy2('../trackGeneration/fitC.pickle')  ## draw a lenght
hf         = h5py.File('../noiseCaracterization/noise_data.h5', 'r')
hf.keys()
imMean     = np.array(hf.get('mean'))
imStd      = np.array(hf.get('std'))
runNumber  = 818
nMean,nStd = imMean[:,:,runNumber - 817],imStd[:,:,runNumber - 817]
del imMean,imStd,runNumber


  


nRuns    = 1000
imShape  = (256,256)
f        = h5py.File('../../bases/db_gen_' + str(date.today()) + 'Run_' + str(nRuns)+'.h5', 'w')
l        = np.abs(lGen.sample(nRuns))  ## get only the abs of w
c        = np.abs(cGen.sample(nRuns))  ## get only the abs of c
data     = np.append(l,c,axis=1)      # concatenate data
data     = data[data[:,1] < 120,:]
Xarr     = f.create_dataset('x_train',(len(data),imShape[0]*imShape[1]), chunks=True)
Yarr     = f.create_dataset('y_train',(len(data),imShape[0]*imShape[1]), chunks=True)
Aarr     = f.create_dataset('alpha',(len(data),), chunks=True)
print('\n' + str(len(data)) + ' images are going to be generated')
for count,param in list(enumerate(data)):
## Generating tracks
    print(param)
    start = time.time()
    print('\n Making image:'+str(count))        
    img               = tGenfunctions.trackGen(param,False,False,None,None) 
    alpha             = 35*np.random.rand()
    imgReal           = imageCreating(img,alpha,nMean,nStd,shape = imShape)
    #y_train[count,:]  = img.ravel()
    #x_train[count,:]  = imgReal.ravel()
    Yarr[count,:]     = img.ravel()
    Xarr[count,:]     = imgReal.ravel()
    Aarr[count]     = alpha
    del imgReal,img,alpha
    


print(' Done! \n time elapsed : '+ str(time.time()-start))
hf.close()


        
     


