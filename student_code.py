import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
import matplotlib
from scipy.stats import itemfreq
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC



def get_tiny_images(image_paths):
  """
  This feature is inspired by the simple tiny images used as features in
  80 million tiny images: a large dataset for non-parametric object and
  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

  To build a tiny image feature, simply resize the original image to a very
  small square resolution, e.g. 16x16. You can either resize the images to
  square while ignoring their aspect ratio or you can crop the center
  square portion out of each image. Making the tiny images zero mean and
  unit length (normalizing them) will increase performance modestly.

  Useful functions:
  -   cv2.resize
  -   use load_image(path) to load a RGB images and load_image_gray(path) to
      load grayscale images

  Args:
  -   image_paths: list of N elements containing image paths

  Returns:
  -   feats: N x d numpy array of resized and then vectorized tiny images
            e.g. if the images are resized to 16x16, d would be 256
  """
  # dummy feats variable
  feats = []
  feats=np.zeros((len(image_paths),256))
  for x,y in enumerate(image_paths):
    image1 = load_image_gray(y)
    image2 = cv2.resize(image1, (16,16))
    image_mean = np.mean(image2)
    normalized_image = image2/image_mean

    flat_image=np.ndarray.flatten(normalized_image)
    feats[x,:]=flat_image
  print(feats.shape)
  #print(len(feats[1,:]))




  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return feats

def build_vocabulary(image_paths, vocab_size):
  """
  This function will sample SIFT descriptors from the training images,
  cluster them with kmeans, and then return the cluster centers.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
          -  frames is a N x 2 matrix of locations, which can be thrown away
          here (but possibly used for extra credit in get_bags_of_sifts if
          you're making a "spatial pyramid").
          -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
  -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
          http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
               the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
               cluster_centers is a K x d matrix of cluster centers. This is
               your vocabulary.

  Args:
  -   image_paths: list of image paths.
  -   vocab_size: size of vocabulary

  Returns:
  -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
  """
  # Load images from the training set. To save computation time, you don't
  # necessarily need to sample from all images, although it would be better
  # to do so. You can randomly sample the descriptors from each image to save
  # memory and speed up the clustering. Or you can simply call vl_dsift with
  # a large step size here, but a smaller step size in get_bags_of_sifts.
  #
  # For each loaded image, get some SIFT features. You don't have to get as
  # many SIFT features as you will in get_bags_of_sift, because you're only
  # trying to get a representative sample here.
  #
  # Once you have tens of thousands of SIFT features from many training
  # images, cluster them with kmeans. The resulting centroids are now your
  # visual word vocabulary.

  dim = 128      # length of the SIFT descriptors that you are going to compute.
  vocab = np.zeros((vocab_size,dim))
  count=len(image_paths)
  #count=1
  SIFT = []
  #print(np.shape(image_paths))

  """
  
  a =load_image_gray(image_paths[2])
  print(np.shape(a))
  a1 = a.astype('float32')
  frames, descriptors = vlfeat.sift.dsift(a1,15,8,fast=True)
  print(np.shape(descriptors))
  print(np.shape(frames))
  """
  #counts=4
  SIFT=np.zeros((1,128))
  

  
  for y in range(count):
    a =load_image_gray(image_paths[y])
    #print(a)
    #print(np.shape(a))
    #print(type(a))
    #a1 = a.astype('float32')
    #print(np.shape(a1))
    #print(type(a1))
    frames, descriptors = vlfeat.sift.dsift(a,step=15,fast=True)
    #print("hi")
    #print(descriptors)
    #print(np.shape(descriptors))
    
    #print(np.shape(descriptors))
    #SIFT.append(descriptors)
    SIFT=np.vstack((SIFT,descriptors))
    #size=np.shape(descriptors)
    #var=size[1]
    #a1=np.random.permutation(var)
    #print(np.shape(SIFT))
  #print(np.shape(SIFT))
  #print(SIFT)
  bh=np.shape(SIFT)
  bh1=bh[0]
  SIFT=SIFT[1:bh1,:]
  #print(np.shape(SIFT))
  b1 = np.float32(SIFT)
  #print(b1)
  cluster_centers = vlfeat.kmeans.kmeans(b1, vocab_size)
  #print(np.shape(cluster_centers))
  #print(cluster_centers)
  vocab=cluster_centers
  #print(vocab)
  print(np.shape(vocab))

  




    
 
  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return vocab

def get_bags_of_sifts(image_paths, vocab_filename):
  """
  This feature representation is described in the handout, lecture
  materials, and Szeliski chapter 14.
  You will want to construct SIFT features here in the same way you
  did in build_vocabulary() (except for possibly changing the sampling
  rate) and then assign each local feature to its nearest cluster center
  and build a histogram indicating how many times each cluster was used.
  Don't forget to normalize the histogram, or else a larger image with more
  SIFT features will look very different from a smaller version of the same
  image.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
          (but possibly used for extra credit in get_bags_of_sifts if you're
          making a "spatial pyramid").
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
  -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """
  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)
  print(vocab.dtype)
  print(vocab.shape)
  # dummy features variable

  feats = []
  #print(vocab)
  
  a1=len(image_paths)
  #a1=1
  y=np.shape(vocab)
  y_cor=y[0]
  feats=np.zeros((a1,y_cor))
  for y in range(a1):
    a =load_image_gray(image_paths[y])
    frames, descriptors = vlfeat.sift.dsift(a,step=10,fast=True)
    descriptors1=np.float32(descriptors)
    #print(np.shape(descriptors1))
    assignments = vlfeat.kmeans.kmeans_quantize(descriptors1,vocab)
    #print(assignments)
    #print(np.shape(assignments))
    vg=np.shape(vocab)
    vg1=vg[0]
    vg2=vg1+2
    #print(vg2)
    hist,bin_edge=np.histogram(assignments,bins=np.arange(1,vg2))
    #print(np.shape(hist))
    hist=np.transpose([hist])
    hist=hist.T
    #print(np.shape(hist))
    #print(hist)
    nm=np.linalg.norm(hist)
    #print(nm)
    histn =hist/nm
    feats[y,:]=histn
  print(np.shape(feats))
  #print(feats)




    
   

  
  return feats

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean'):

  """
  This function will predict the category for every test image by finding
  the training image with most similar features. Instead of 1 nearest
  neighbor, you can vote based on k nearest neighbors which will increase
  performance (although you need to pick a reasonable value for k).

  Useful functions:
  -   D = sklearn_pairwise.pairwise_distances(X, Y)
        computes the distance matrix D between all pairs of rows in X and Y.
          -  X is a N x d numpy array of d-dimensional features arranged along
          N rows
          -  Y is a M x d numpy array of d-dimensional features arranged along
          N rows
          -  D is a N x M numpy array where d(i, j) is the distance between row
          i of X and row j of Y

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating
          the ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  -   metric: (optional) metric to be used for nearest neighbor.
          Can be used to select different distance functions. The default
          metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
          well for histograms

  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  test_labels = []
  mat = sklearn_pairwise.pairwise_distances(train_image_feats,test_image_feats)
  n=np.shape(mat)
  #short=mat[1,:]
  #print(short)
  #bab=np.argmax(short)
  #print(bab)
  #bab1=np.argmin(short)
  #print(bab1)
  #b=np.argsort(short,axis=0)
  #print(b)
  d=n[0]
  train=np.asarray(train_labels)
  #print(train.shape)
  
  b=np.zeros((1500,1500))
  for i in range(d):
    f=np.argsort(mat[i,:],axis=0)
    b[i,:]=np.asarray(f)
  #print(b.shape)
  #print(b[1,:])
  #print(np.max(b[:,1]))
  index=b[:,0]
  #print(index.shape)
  #print(type(index))
  #index1=index.reshape((1, -1))
  #print(index1.shape)
  #print(index1)
  #index1=index.reshape((1, -1))
  #print(type(index))
  #index1=np.asarray(index)
  #print(index1.shape)
  #print(type(index1))
  #inal=np.zeros((1500,1))
  final=list()
  for i in range(0,1500):
    var=index[i]
    #rnt(var)
    #rint(type(var))
    var=int(var)
    #rint(type(var))
    #var1=int(var)
    final1=train[var]
    #rint(final)
    final.append(final1)
  #print(len(final))
  #print(final)
  #print(type(final))

  #final=train_labels[index1]
    #print("done")
   # b.append(b)
  #print(b)

    #b=sorted(enumerate(mat[i,:]), key=lambda x: x[0])
  #print(b.shape)
  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return  final

def svm_classify(train_image_feats, train_labels, test_image_feats):
  """
  This function will train a linear SVM for every category (i.e. one vs all)
  and then use the learned linear classifiers to predict the category of
  every test image. Every test feature will be evaluated with all 15 SVMs
  and the most confident SVM will "win". Confidence, or distance from the
  margin, is W*X + B where '*' is the inner product or dot product and W and
  B are the learned hyperplane parameters.

  Useful functions:
  -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  -   svm.fit(X, y)
  -   set(l)

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  # categories
  categories = list(set(train_labels))
  #print(len(categories))
  #print(categories[0])
  train=train_labels
  #print(train)
  #print(np.shape(train))
  training=np.zeros((1300,200))
  training=train_image_feats[0:1300,:]
  st=np.shape(training)
  print("Training set=",st)
  train_labels1=train_labels[0:1300]
  print("training_label_size=",np.shape(train_labels1))
  w1=np.zeros((15,200))
  b1=np.zeros((15,1))
  """
  for i in range(len(categories)):
    str1=categories[i]
    label=[]
    for j in range(0,1300):
      str2=train_labels[j]
      if str1 == str2:
        label.append(1)
      else:
        label.append(-1)
    svm = LinearSVC(C=1)
    #svm=sklearn.svm.SVC(C=1,kernel='rbf')
    svm.fit(train_image_feats[0:1300,:],label)
    w1[i,:]=svm.coef_
    b1[i,:]=svm.intercept_
  print(np.shape(w1))
  print(np.shape(b1))
  """


  # construct 1 vs all SVMs for each category
  svms = {cat: LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5) for cat in categories}
  #print(svms)
  #test_labels = np.zeros((1500,))
  validation=np.zeros((200,200))
  validation=train_image_feats[1300:1500,:]
  print("validation set=",np.shape(validation))

  test_labels= []
  test_lab=np.zeros((1,1500))
  w=np.zeros((15,200))
  b=np.zeros((15,1))
  
  for i in range(len(categories)):
    str1=categories[i]
    label=[]
    for j in range(len(train_labels)):
      str2=train_labels[j]
      if str1 == str2:
        label.append(1)
      else:
        label.append(-1)
    
    #arr=np.asarray([label])
    #print(np.shape(arr))
    #arr=arr.T
    #print(np.shape(arr))
    svm = LinearSVC(C=1)
    #svm=sklearn.svm.SVC(C=1,kernel='rbf')
    svm.fit(train_image_feats,label)
    w[i,:]=svm.coef_
    b[i,:]=svm.intercept_
  #print(w)
  #print(b)
  print(np.shape(test_image_feats))
  
  for i in range(len(test_image_feats)):
    conf=np.zeros((15,1))
    for j in range(len(categories)):
      conf[j,:]=np.dot(w[j,:],test_image_feats[i,:])+b[j,:]
    #print(conf.T)
    #print(np.shape(conf.T))
    ind=np.argmax(conf,axis=0)
    #print(ind)
    ind=int(ind)
    test_labels.append(categories[ind])
  print(np.shape(test_labels))




    #test_lab=np.concatenate((test_lab,arr),axis=0)

  #test_labels=test_lab[1:16,:]
  #print(np.shape(test_labels))
  #test_labels=test_labels.T
  #print(np.shape(test_labels))
  
  #print(list(test_labels[0,:]))

  """
  for i in range(len(categories)):
    svm = LinearSVC(C=1)
    svm.fit(train_image_feats, test_labels[:,i])
  """
  
   


    

   


  


  #for i in range(len(categories)):


  

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return test_labels
