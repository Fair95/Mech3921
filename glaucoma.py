import smtplib
import email.utils
import cv2
import tensorflow as tf
import numpy as np
import argparse
import time
import cv
import subprocess as sp
import os
import sys
import getpass
from matplotlib import pyplot as plt
from email.mime.text import MIMEText
from sendmail import send_mail
to_email = raw_input('Recipient: ')
cv2.destroyAllWindows()

testPath = sys.argv[1]
print "the testing image has the path:"+sys.argv[1]

image_size = 100

def rotate(image,degrees):
    
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    
    #rotate the image by 180 degrees
    M = cv2.getRotationMatrix2D(center, degrees, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
#img = cv2.imread('/Users/fair/Desktop/MECH3921/IBM/Training/Glaucoma Images/Im256.jpg')
#new_img = cv2.resize(img,(28,28))
#print np.size(img,0)
#print np.size(img,1)
#print np.size(new_img,0)
#print np.size(new_img,1)
#img = cv2.resize(img,(image_size,image_size))
#cv2.imshow('image',img)
#cv2.imshow('frame',new_img)
#rotate(img,90)
#cv2.destroyAllWindows()
#cv2.destroyWindow('image')


glaucomaData = []
glaucomaLabels = []
counter = 0
for i in os.listdir("/Users/fair/Desktop/MECH3921/IBM/Training/Glaucoma Images"):
    if i.endswith(".jpg"): 
        
        
        image = cv2.imread("/Users/fair/Desktop/MECH3921/IBM/Training/Glaucoma Images/" + i)
        
        new_image = cv2.resize(image,(image_size,image_size))
        
        glaucomaData.append(new_image)
        glaucomaLabels.append([0,1])
        
        angle = 90
        while angle < 360:
            
            image = rotate(new_image,angle)
            
            glaucomaData.append(image)
            glaucomaLabels.append([0,1])
            angle = angle + 90
    
        #cv2.imshow('frame',image)
        #print np.size(image,0)
        #print np.size(image,1)
        #print np.size(image,2)
       
        counter = counter + 1
        continue
    else:
        continue
healthyData = []
healthyLabels = []
counter = 0
for i in os.listdir("/Users/fair/Desktop/MECH3921/IBM/Training/Normal"):
    if i.endswith(".jpg"): 
        
        
        image = cv2.imread("/Users/fair/Desktop/MECH3921/IBM/Training/Normal/" + i)
        new_image = cv2.resize(image,(image_size,image_size))
        
        healthyData.append(new_image)
        healthyLabels.append([1,0])
        
        angle = 90
        while angle < 360:
            
            image = rotate(new_image,angle)
            
            healthyData.append(image)
            healthyLabels.append([1,0])
            angle = angle + 90

        #cv2.imshow('frame',image)
        #print np.size(image,0)
        #print np.size(image,1)
        #print np.size(image,2)
        counter = counter + 1
        continue
    else:
        continue

print "number of glaucoma images is",len(glaucomaData)
print "number of healthy images is",len(healthyData)
print "dimention of healthy images is", np.asarray(healthyData).shape
print "dimention of henlthy labels is", np.asarray(healthyLabels).shape

# Join two set of data and convert type of data to float32
full_data = np.concatenate((glaucomaData,healthyData),0).astype(np.float32)
full_labels = np.concatenate((glaucomaLabels,healthyLabels),0).astype(np.float32)
print "dimention of labels after concatenating is" , full_labels.shape
print "dimention of data after concatenating is" , full_data.shape
# print full_labels
# Always create same random array?
np.random.seed(133)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
full_data,full_labels = randomize(full_data,full_labels)

# What does this do?
def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#full_data, full_labels = shuffle_in_unison_inplace(full_data,full_labels)

# Calculate the accuracy by comparing prediction and labels, [1,0] or [0,1]
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# Divide dataset into 2, one for training, the other for validation.
# But why 1600:1820? Should be 1600:1628?
train_dataset = full_data[0:1600]
train_labels = full_labels[0:1600]
valid_dataset = full_data[1600:1820]
valid_labels = full_labels[1600:1820]

####################### Lets just be cheap and try it at 28*28.

# Initialize the variables for Cnn

# two classes: glaucoma and health
num_labels = 2
# 3 channels for each image
num_channels = 3 # grayscale

batch_size = 10
patch_size = 7
depth = 16
num_hidden = 64

num_steps = 20
# 1820 /5 is 364
graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  #tf_test_dataset = tf.constant(test_dataset)
  
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  #the reason its /4 is because we've max pooled twice 
    #halved twice (i.e /4)
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size / 4 * image_size / 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  # the way conv2d works is data is the input, layer1weights is the filter, [1,strides,strides,1]
    # is the strides throughtout the dimensions, first and 4th must be the same in 4D,  1. 
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  logits = model(tf_train_dataset)
  probabilities = tf.nn.softmax(logits)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  #optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
  optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
 # test_prediction = tf.nn.softmax(model(tf_test_dataset))
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  saver = tf.train.Saver()
  print "Initialized"
  for step in xrange(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 10 == 0):
      print "Minibatch loss at step", step, ":", l
      print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
      print "Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels)
      #print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)
    # Save variables to .ckpt file
    saver.save(session, "trained_variables.ckpt")
#  print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)
session.close()
testData = []
        
for i in os.listdir(testPath):
    if i.endswith(".jpeg"): 
        
        image = cv2.imread(testPath +'/' + i)
        
        new_image = cv2.resize(image,(image_size,image_size))
        
        testData.append(new_image)
        
        continue
    else:
        continue        
        
### Evaluate some new data!
with tf.Session(graph=graph) as sess:
      saver = tf.train.Saver()
      saver.restore(sess, "trained_variables.ckpt")
      print "Initialized"
      
      testSet = np.asarray(testData).astype(np.float32)
      tests = testSet[0:10, :, :, :]
      #print batch_data.shape
     
      #print testSet[0, :, :, :].shape
      
      feed_dict={tf_train_dataset:tests}
      predictions = sess.run(probabilities,
                       feed_dict)
      print predictions
      glau = 0
      heal = 0
      classifications = []
      for i in xrange(len(predictions)):
            classNo = np.argmax(predictions,1)[i]
            if classNo == 1:
		glau = glau + 1
                classifications.append("glaucoma")
            elif classNo ==0:
		heal = heal + 1
                classifications.append("healthy")
                
      print classifications
      if glau > heal:
		result = "Thanks for using our app!\nThe result of your diagnosis is glaucoma, the possibility is " + str(glau/10.0)
      else:
		result = "Thanks for using our app!\nThe result of your diagnosis is healthy, the possibility is " + str(heal/10.0)
send_mail('qili2960@uni.sydney.edu.au', to_email, 'Your Daignosis result', result, username='qili2960@uni.sydney.edu.au', password='Alex.811')
