import cv2
import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.filters.rank import median
from skimage.morphology import disk

# coding: utf-8

# In[8]:


def clahe_preprocessing(img,clip_param = 0.01, grid_param = None, nbin_param = 256,debug_option = 'off'):
    """scikit-learn clahe algorithm

    Parameters
    ----------
    image : (M, N[, C]) ndarray / (pixel / color space)
        Input image.

    kernel_size: integer or list-like, optional
        Defines the shape of contextual regions used in the algorithm. If iterable is passed, it must have the same number of elements as image.ndim (without color channel). If integer, it is broadcasted to each image dimension. By default, kernel_size is 1/8 of image height by 1/8 of its width.

    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more contrast).

    nbins : int, optional
        Number of gray bins for histogram (“data range”).
        
    Returns
    -------
        clahe image
    
    Example
    -------

    """
    norm_img = cv2.normalize(img_as_float(img), None, 0.0, 1.0, cv2.NORM_MINMAX) # Convert to normalized floating point
    if grid_param == None:
        clahe = skimage.exposure.equalize_adapthist(norm_img, clip_limit = clip_param,nbins = nbin_param)
    else:
        clahe = skimage.exposure.equalize_adapthist(norm_img, grid_param, clip_limit = clip_param, nbins = nbin_param)
    
    if debug_option == 'on':
        print("data type img : {} clahe : {}".format(img.dtype,clahe.dtype))
        
        plt.hist(clahe.ravel(),256,[clahe.min(),clahe.max()])
        plt.title('CLAHE Histogram')
        plt.show()
        
        plt.axis("off")
        plt.title("CLAHE IMAGE")
        plt.imshow(clahe,cmap='gray')
        plt.show()
    
    return clahe


# In[9]:


def correction_non_uniform_back(img, blur_size,debug_option = 'off'):
    """non uniform background subtraction.
    by this paper: 
    Automated segmentation of the optic nerve head for diagnosis of glaucoma
    
    R.Chrastek, M.Wolf, K.Donath, H.Niemann, D.Paulus, T. Hothorn, B.Lausen, R.Lammer, C.Y.Mardin, G.Michelson
    
    MEDICAL IMAGE ANALYSIS
    
    correction of non-uniform illuminaition.

    Parameters
    ----------
    image : (M, N[, C]) 2D numpy array / (pixel / color space)
        2D numpy array image
    
    blur_size: median blur kernel size
        recommended kernel size is 30~40.
        it it up to your image.
    
    debug_option : for debugging.
        debugging option == 'on'
            show histogram & show data tape
            (default = 'off')
            
    Returns
    -------
        uniform background image 't'
    
    Example
    -------

    """
    if len(img.shape) == 2:
        corImg = img
        backImg = median(corImg,disk(blur_size))
        maxGrayVal = np.max(backImg)
        
        col,row = corImg.shape[0], corImg.shape[1]
        rList = np.zeros((col,row))

        for i in range(col):
            for j in range(row):
                if backImg[i,j] != 0:
                    rList[i,j] = maxGrayVal / backImg[i,j]
                    
        meanVal =np.mean(backImg,dtype = 'int') 
        c = maxGrayVal - meanVal
        p = np.multiply(img,rList)
        t = p - c
        for i in range(col):
            for j in range(row):
                if t[i][j] >= 255:
                    t[i][j] = 255
                elif t[i][j] <=0:
                    t[i][j] = 0
        
        if debug_option == 'on':
            print('data type \n')
            print('corImg : {}, backImg : {}, resultImg : {}'.format(corImg.dtype,backImg.dtype,t.dtype))
            
            plt.hist(t.ravel(),256,[t.min(),t.max()])
            plt.title('Histogram')
            plt.show()
            
        return t
        
    elif img.shape[2] == 3:
        corImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        backImg = median(corImg,disk(blur_size))
        maxGrayVal = np.max(backImg)
        
        col,row = corImg.shape[0], corImg.shape[1]
        rList = np.zeros((col,row))

        for i in range(col):
            for j in range(row):
                if backImg[i,j] != 0:
                    rList[i,j] = maxGrayVal / backImg[i,j]
                    
        meanVal =np.mean(backImg,dtype = 'int') 
        c = maxGrayVal - meanVal
        p = np.multiply(corImg,rList)
        t = p - c
        for i in range(col):
            for j in range(row):
                if t[i][j] >= 255:
                    t[i][j] = 255
                elif t[i][j] <=0:
                    t[i][j] = 0
        if debug_option == 'on':
            print('data type \n')
            print('corImg : {}, backImg : {}, resultImg : {}'.format(corImg.dtype,backImg.dtype,t.dtype))

            plt.hist(t.ravel(),256,[t.min(),t.max()])
            plt.title('Histogram')
            plt.show()            
        return t
    else:
        print('Color Space Error!')


# In[10]:


def non_uniform_back(img, blur_size,debug_option = 'off'):
    """non uniform background subtraction.

    Parameters
    ----------
    image : (M, N[, C]) 2D numpy array / (pixel / color space)
        2D numpy array image
    
    blur_size : median blur kernel size
        recommended kernel size is 30~40.
        it it up to your image.
      
    debug_option : for debugging.
        debugging option == 'on'
            show histogram & show data tape
            (default = 'off')
            
    Returns
    -------
        uniform background image 't'
    
    Example
    -------

    """
    if len(img.shape) == 2:
        nUniImg = img
        backImg = median(nUniImg,disk(51))
        meanVal = np.mean(nUniImg)
        shadeResult = np.divide(nUniImg,backImg) * meanVal
        col,row = nUniImg.shape[0],nUniImg.shape[1]
        for i in range(col):
            for j in range(row):
                if shadeResult[i,j] <= 0:
                    shadeResult[i,j] = 0
                elif shadeResult[i,j] >=255:
                    shadeResult[i,j] = 255
                    
        if debug_option == 'on':
            print('data type \n')
            print('Img : {}, backImg : {}, resultImg : {}'.format(nUniImg.dtype,backImg.dtype,t.dtype))

            plt.hist(shadeResult.ravel(),256,[shadeResult.min(),shadeResult.max()])
            plt.title('Histogram')
            plt.show()
            
        return shadeResult
    
    elif img.shape[2] == 3:
        nUniImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        backImg = median(nUniImg,disk(51))
        meanVal = np.mean(nUniImg)
        shadeResult = np.divide(nUniImg,backImg) * meanVal
        col,row = nUniImg.shape[0],nUniImg.shape[1]
        for i in range(col):
            for j in range(row):
                if shadeResult[i,j] <= 0:
                    shadeResult[i,j] = 0
                elif shadeResult[i,j] >=255:
                    shadeResult[i,j] = 255
        
        if debug_option == 'on':
            print('data type \n')
            print('Img : {}, backImg : {}, resultImg : {}'.format(nUniImg.dtype,backImg.dtype,t.dtype))

            plt.hist(shadeResult.ravel(),256,[shadeResult.min(),shadeResult.max()])
            plt.title('Histogram')
            plt.show()
                    
        return shadeResult
    
    else:
        print('Color Space Error!')

