
# coding: utf-8

# In[ ]:


def show_on_jupyter(img,color= None,title=None):
    import matplotlib.pyplot as plt
    import cv2
    """Show img on jupyter notebook. No return Value
    
    You should check the img's color space.
    I just consider about RGB color space & 1 ch color space(like green ch, gray space, ...)
    
    using matplotlib
    
    Parameters
    ----------
    img : 2-D Array
        numpy 2-D array
        opencv / sklearn / plt are avaliable.
        float / uint8 data type.
        
    color : string
        'gray' or 'None'
        'gray' means that img has a 1 ch.
        'None' means that img has a RGB ch.
        (default: None)
        
    title : string
        decide img's title
        (default : None)
        
    Returns
    -------
        No return value.
    
    Example
    -------
    >>> img = cv2.imread(img_path)
    >>> show_on_jupyter(img)
    
    img has a 1 ch
    >>> img = cv2.imread(img_path)
    >>> show_on_jupyter(img,'gray')
    """
    if color == 'gray':
        plt.axis("off")
        plt.title(title)
        plt.imshow(img,cmap=color)
        plt.show()
    elif color == None:
        plt.axis("off")
        plt.title(title)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        print("Gray or None")


# In[2]:


def show_histogram(img,color = None, dtype = 'int'):
    import cv2
    import matplotlib.pyplot as plt
    """Show histogram on jupyter notebook
    
    I consider about img's color space and data type.
    
    color space : RGB / 1ch color space(also gray)
    data type : uint8 or float64
    
    so, you should check your img's color space and data type.
    
    Parameters
    ----------
    img : 2-D Array
        numpy 2-D array
        opencv / sklearn / plt are avaliable.
        float / uint8 data type.
        
    color : string
        'gray' or 'None'
        'gray' means that img has a 1 ch.
        'None' means that img has a RGB ch.
        (default: None)
        
    dtype :  string
        'int' or 'float'
        img's data type
        float : [min,max] and divided 256 values
        int = [0,256]
        (default : int)
        
    Returns
    -------
        No return value.
    
    Example
    -------
    >>> img = cv2.imread(img_path)
    >>> show_histogram(img)
    
    img has a 1 ch
    >>> img = cv2.imread(img_path)
    >>> show_histogram(img,'gray')
    
    1ch img & float img
    >>> img = cv2.imread(img_path)
    >>> show_histogram(img,'gray','float')
    """
    if (color == None) and (dtype =='int'):
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.show()
    if (color == None) and (dtype =='float'):
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,1])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.show()
    elif (color == 'gray')and (dtype=='float') :
        plt.hist(img.ravel(),256,[img.min(),img.max()])
        plt.title('Histogram for gray scale picture')
        plt.show()
    elif (color == 'gray') and (dtype == 'int'):
        plt.hist(img.ravel(),256,[0,256])
        plt.title('Histogram for gray scale picture')
        plt.show()
    else:
        print('check your parameter')


# In[171]:


def sub2ind( sizes, multi_index ):
    """
    Map a d-dimensional index to the scalar index of the equivalent flat array
    Example:
    | 1,1  1,2  1,3 |     | 1  4  7 | 
    | 2,1  2,2  2,3 | --> | 2  5  8 |
    | 3,1  3,2  3,3 |     | 3  6  9 |      
    """
    num_dims = sizes.shape[0]
    index = 0
    shift = 1
    for i in range( num_dims ):
        index += shift * multi_index[i]
        shift *= sizes[i]
    return index+1


# In[250]:


def strel_line(length,degree):
    import numpy as np
    from numpy import pi,cos,sin,tan,fix
    
    
    """Project matlab to python
    
    matlab function equal to strel("line",length,degree)
    but just consider 2D array. not 3D array.
    
    implement 'line' structure element.
    and you can rotate it using degree parameter


    Parameters
    ----------
    length : int
        decide line's length.
        length is always odd number.
        Example
        -------
        >>> length = 3
        >>> [1,1,1]
        >>> length = 5
        >>> [1,1,1,1,1]

    degree : int
        rotate degree.
       
    Returns
    -------
        line structure element
    
    Example
    -------
    >>> temp = strel_line(5,30)
    >>> temp2 = strel_line(5,0)
    >>> openImg = morphology.open(img,temp) 
    """

    deg90 = degree % 90
    if deg90 > 45:
        alpha = pi * (90 - deg90) / 180
    else:
        alpha = pi * deg90 / 180

    center = (length -1) /2 

    c =int(round (center * cos(alpha)) + 1)
    r =int(round (center * sin(alpha)) + 1)
    
    line = np.zeros((r,c),dtype = 'int')
    #print(line)
    m = tan(alpha)
    x = np.int64( np.arange(1,c+1) )
    y = np.int64(r - fix(np.multiply(m,x-0.5)) )
    temp= np.array([r,c])
    
    idx = []
    for i in range(len(x)):
        idx.append( sub2ind(temp,(y[i]-1,x[i]-1)) )
    
    temp2 = np.ravel(line)
    for i in idx:
        temp2[i-1] = 1
    
    line = temp2.reshape(r,c,order = 'F')
    #print(line)
    
    z = np.zeros((r-1,c),dtype='int')
    lineStrip = line[0,0:c-1]
    lineRest = line[1:r,0:c-1]
    
    #print("fuck",lineStrip[::-1])
    #print(lineStrip)
    
    tempArray = np.concatenate((z, lineRest[::-1,::-1]), axis=1)
    tempArray2 = np.concatenate((lineStrip,np.array([1]), lineStrip[::-1]), axis=0)
    tempArray3 = np.concatenate((lineRest,z[::-1,::-1]),axis=1)
    se = np.vstack((tempArray,tempArray2,tempArray3))
    #print(se) 
    sect = int( fix( (degree % 180) / 45 ))
    #print(sect)
    
    #np.concatenate((a, b.T), axis=1)
    #np.concatenate((a, b.T), axis=1)
    if sect == 1:
        se = np.transpose(se)
    elif sect == 2:
        se = np.rot90(se,1)
    elif sect == 3:
        se = np.fliplr(se)
    
    #print(se)
    return se

def denormalized_img(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    
    """Normalized image's range convert to previous range.
        
        Parameters
        ----------
        image : float image.
            In my case, I always use this parameter with sk-learn image's img_as_float() function.
        from_min : float
            In many case, the range is 0.0
        from_max : float
            In many case, the range is 1.0
        to_min : int
            In many case, the range is 0 (to_min) to 255 (to_max)
        to_max : int
            In many case, the range is 0 (to_min) to 255 (to_max)

        Returns
        -------
        denormalized Image.
        
        Example
        -------
        >>> img = denormalized_img(img,from_min, from_max, to_min, to_max)
        >>> #now, img is denormalized!
        """
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

