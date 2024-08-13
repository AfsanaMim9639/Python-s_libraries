{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "af754899-398d-46e8-a921-77ca15b7be65",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c1d18ce4-4f11-431d-8779-61a5f3cb8a03",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
      "\u001b[1m11490434/11490434\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 0us/step\n"
     ]
    }
   ],
   "source": [
    "(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b158c9b3-fdeb-43a8-892b-28b51c62c6d8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "60000"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "88104f10-bbb6-4194-93ae-cf6e078e09bd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10000"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1ea52005-d038-4525-afb8-36f7e0358351",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(28, 28)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[0].shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8554f0d7-654a-493c-aa5a-7aaab199780a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,\n",
       "         18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,\n",
       "        253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,\n",
       "        253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,\n",
       "        253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,\n",
       "        205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,\n",
       "         90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,\n",
       "        190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,\n",
       "        253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,\n",
       "        241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "         81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,\n",
       "        148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,\n",
       "        253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,\n",
       "        253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,\n",
       "        195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,\n",
       "         11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0]], dtype=uint8)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7bc49b42-c90b-4103-821c-4bdf3ded37f9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x2cbb2ef04d0>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 480x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.matshow(X_train[2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "03a963eb-5768-4f6f-8a9f-496134c5ce7a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train[2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b6c5a590-9770-4a33-a13b-79802683af66",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 0, 4, 1, 9], dtype=uint8)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "8dbf9bbb-106f-477a-9919-4bacd2189512",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "e050cefa-25e2-40ad-88ab-b96c75bef08c",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = X_train/255\n",
    "X_test = X_test/255"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "a6251304-e375-4584-a496-8060b35dede7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.01176471, 0.07058824, 0.07058824,\n",
       "        0.07058824, 0.49411765, 0.53333333, 0.68627451, 0.10196078,\n",
       "        0.65098039, 1.        , 0.96862745, 0.49803922, 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.11764706, 0.14117647,\n",
       "        0.36862745, 0.60392157, 0.66666667, 0.99215686, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.88235294, 0.6745098 ,\n",
       "        0.99215686, 0.94901961, 0.76470588, 0.25098039, 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.19215686, 0.93333333, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.98431373, 0.36470588, 0.32156863,\n",
       "        0.32156863, 0.21960784, 0.15294118, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.07058824, 0.85882353, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.77647059,\n",
       "        0.71372549, 0.96862745, 0.94509804, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.31372549, 0.61176471,\n",
       "        0.41960784, 0.99215686, 0.99215686, 0.80392157, 0.04313725,\n",
       "        0.        , 0.16862745, 0.60392157, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.05490196,\n",
       "        0.00392157, 0.60392157, 0.99215686, 0.35294118, 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.54509804, 0.99215686, 0.74509804, 0.00784314,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.04313725, 0.74509804, 0.99215686, 0.2745098 ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.1372549 , 0.94509804, 0.88235294,\n",
       "        0.62745098, 0.42352941, 0.00392157, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.31764706, 0.94117647,\n",
       "        0.99215686, 0.99215686, 0.46666667, 0.09803922, 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.17647059,\n",
       "        0.72941176, 0.99215686, 0.99215686, 0.58823529, 0.10588235,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.0627451 , 0.36470588, 0.98823529, 0.99215686, 0.73333333,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.97647059, 0.99215686, 0.97647059,\n",
       "        0.25098039, 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.18039216,\n",
       "        0.50980392, 0.71764706, 0.99215686, 0.99215686, 0.81176471,\n",
       "        0.00784314, 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.15294118, 0.58039216, 0.89803922,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.98039216, 0.71372549,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.09411765, 0.44705882, 0.86666667, 0.99215686, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.78823529, 0.30588235, 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.09019608, 0.25882353,\n",
       "        0.83529412, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "        0.77647059, 0.31764706, 0.00784314, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.07058824, 0.67058824, 0.85882353, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.76470588, 0.31372549,\n",
       "        0.03529412, 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.21568627,\n",
       "        0.6745098 , 0.88627451, 0.99215686, 0.99215686, 0.99215686,\n",
       "        0.99215686, 0.95686275, 0.52156863, 0.04313725, 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.53333333,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.83137255, 0.52941176,\n",
       "        0.51764706, 0.0627451 , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ]])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "9f9c4f64-42b8-48c6-b789-5c33505b367f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 784)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train_flattend=X_train.reshape(len(X_train), 28*28)\n",
    "x_train_flattend.shape\n",
    "\n",
    "x_test_flattend=X_test.reshape(len(X_test), 28*28)\n",
    "x_test_flattend.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "47dd63dd-e843-4e9f-b377-47845ca35b4b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.01176471, 0.07058824, 0.07058824,\n",
       "       0.07058824, 0.49411765, 0.53333333, 0.68627451, 0.10196078,\n",
       "       0.65098039, 1.        , 0.96862745, 0.49803922, 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.11764706, 0.14117647, 0.36862745, 0.60392157,\n",
       "       0.66666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.99215686, 0.88235294, 0.6745098 , 0.99215686, 0.94901961,\n",
       "       0.76470588, 0.25098039, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.19215686, 0.93333333,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.98431373, 0.36470588,\n",
       "       0.32156863, 0.32156863, 0.21960784, 0.15294118, 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.07058824, 0.85882353, 0.99215686, 0.99215686,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.77647059, 0.71372549,\n",
       "       0.96862745, 0.94509804, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.31372549, 0.61176471, 0.41960784, 0.99215686, 0.99215686,\n",
       "       0.80392157, 0.04313725, 0.        , 0.16862745, 0.60392157,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.05490196,\n",
       "       0.00392157, 0.60392157, 0.99215686, 0.35294118, 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.54509804,\n",
       "       0.99215686, 0.74509804, 0.00784314, 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.04313725, 0.74509804, 0.99215686,\n",
       "       0.2745098 , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.1372549 , 0.94509804, 0.88235294, 0.62745098,\n",
       "       0.42352941, 0.00392157, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.31764706, 0.94117647, 0.99215686, 0.99215686, 0.46666667,\n",
       "       0.09803922, 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.17647059,\n",
       "       0.72941176, 0.99215686, 0.99215686, 0.58823529, 0.10588235,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.0627451 , 0.36470588,\n",
       "       0.98823529, 0.99215686, 0.73333333, 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.97647059, 0.99215686,\n",
       "       0.97647059, 0.25098039, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.18039216, 0.50980392,\n",
       "       0.71764706, 0.99215686, 0.99215686, 0.81176471, 0.00784314,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.15294118,\n",
       "       0.58039216, 0.89803922, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.98039216, 0.71372549, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.09411765, 0.44705882, 0.86666667, 0.99215686, 0.99215686,\n",
       "       0.99215686, 0.99215686, 0.78823529, 0.30588235, 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.09019608, 0.25882353, 0.83529412, 0.99215686,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.77647059, 0.31764706,\n",
       "       0.00784314, 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.07058824, 0.67058824, 0.85882353,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.76470588,\n",
       "       0.31372549, 0.03529412, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.21568627, 0.6745098 ,\n",
       "       0.88627451, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.95686275, 0.52156863, 0.04313725, 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.53333333, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.83137255, 0.52941176, 0.51764706, 0.0627451 , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        ])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train_flattend[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "82de0b4e-8542-4775-b500-57b291becaad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 2ms/step - accuracy: 0.8020 - loss: 0.7420\n",
      "Epoch 2/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 2ms/step - accuracy: 0.9117 - loss: 0.3106\n",
      "Epoch 3/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 2ms/step - accuracy: 0.9211 - loss: 0.2853\n",
      "Epoch 4/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 2ms/step - accuracy: 0.9225 - loss: 0.2756\n",
      "Epoch 5/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 2ms/step - accuracy: 0.9267 - loss: 0.2619\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x2cbb74c07a0>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = keras.Sequential([\n",
    "    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')\n",
    "])\n",
    "model.compile(\n",
    "    optimizer='adam', \n",
    "    loss='sparse_categorical_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "model.fit(x_train_flattend, y_train, epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "3a40c71c-4225-4f33-ac8d-e4a51466cc0b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9149 - loss: 0.3040\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.26706430315971375, 0.9265000224113464]"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(x_test_flattend, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "b173ec30-6e64-405b-a965-5ea639b4a42a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x2cbb099b0e0>"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 480x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.matshow(X_test[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "cda16712-8ad0-4e45-a353-73c899db1670",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([4.0315166e-01, 4.4164322e-03, 9.9968749e-01, 3.7934422e-01,\n",
       "       8.1326113e-10, 8.4695691e-01, 8.7140000e-01, 6.6082713e-13,\n",
       "       1.8813339e-01, 2.8531375e-09], dtype=float32)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_predicted= model.predict(x_test_flattend)\n",
    "y_predicted[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "c4fc5d74-2547-497d-9537-4234f85160b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.argmax(y_predicted[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "40f91c60-11f6-4464-8436-30241e707ce2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[7, 2, 1, 0, 4]"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_predicted_labels=[np.argmax(i) for i in y_predicted]\n",
    "y_predicted_labels[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "1717774e-f8ff-4abe-896f-947975e9bc34",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([7, 2, 1, 0, 4], dtype=uint8)"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "3b161590-1d73-4711-adda-4e9e735c56d4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(10, 10), dtype=int32, numpy=\n",
       "array([[ 964,    0,    2,    2,    0,    5,    4,    2,    1,    0],\n",
       "       [   0, 1110,    3,    2,    0,    1,    4,    2,   13,    0],\n",
       "       [   4,    7,  938,   15,    7,    2,   11,    8,   35,    5],\n",
       "       [   1,    0,   22,  927,    1,   19,    2,    6,   21,   11],\n",
       "       [   1,    1,    5,    2,  911,    0,   10,    3,    9,   40],\n",
       "       [   8,    2,    6,   40,    8,  775,    9,    4,   33,    7],\n",
       "       [  11,    3,    9,    1,    7,   17,  905,    2,    3,    0],\n",
       "       [   1,    6,   25,    9,    9,    1,    0,  923,    2,   52],\n",
       "       [   7,    7,    7,   22,    9,   26,    8,    5,  872,   11],\n",
       "       [  11,    7,    1,   11,   22,    5,    0,    7,    5,  940]])>"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)\n",
    "cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "852963b3-c01a-4773-9349-b10e118d5fe5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(95.72222222222221, 0.5, 'Truth')"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAxkAAAJaCAYAAABDWIqJAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAACjJ0lEQVR4nOzdd1wT5x8H8E+YQlgq24m7oqLiQkWrouK27m2dddbROqjbUnHPWrVWxVlntY46cQuK4kJBcINMkY0yk98f/kxJwcklB8nn3de9Xs3d5fI5SQ6efJ/nOYlcLpeDiIiIiIhIIDpiByAiIiIiIs3CRgYREREREQmKjQwiIiIiIhIUGxlERERERCQoNjKIiIiIiEhQbGQQEREREZGg2MggIiIiIiJBsZFBRERERESCYiODiIiIiIgEpSd2AFV4c+o3sSOIwrTjL2JHICIVkYgdQCRysQMQkcpkZ0aIHeG9suKeqO219C0rqO211ImVDCIiIiIiEpRGVjKIiIiIiL6YLEfsBEUeKxlERERERCQoVjKIiIiIiHKTy8ROUOSxkkFERERERIJiJYOIiIiIKDcZKxkFxUoGEREREREJipUMIiIiIqJc5ByTUWCsZBARERERkaBYySAiIiIiyo1jMgqMlQwiIiIiIhIUKxlERERERLlxTEaBsZJBRERERESCYiWDiIiIiCg3WY7YCYo8VjKIiIiIiEhQbGQQEREREZGg2F2KiIiIiCg3DvwuMFYyiIiIiIhIUKxkEBERERHlxpvxFRgrGUREREREJCg2Mj5BWnomFh+4gHazN6Ph5F8xaPle3HserbTPk+h4TNhwGE2nrEOjH9ai35I/ERWfnOdYcrkcY387hNrjV+HsncfqOgWVGj1qMB6FXkVq8mP4Xj6C+vVqix1JpaZNHQc/32NIeBWCyBd3cGD/JlSpUlHsWCqnref9jra9z2fNmoyszAilJTDwgtix1G7qlLHIzozAsqXzxI6iUtr6+XZt2hCHDnoj7FkAsjMj0LlzW7EjqZW2Xdc+h1wuU9uiqdjI+ATzdp3B1Qdh8BzUFvs8BsClWlmM+vUgYhJTAQDhLxMxZMU+lLcpgT++74590/tjpHtDGOrn7Y2249wtQKLuM1Cdnj07Y+mSOfjZcznqN3THnbtB+OfYTlhZlRQ7mso0c22Edeu2oolrJ7i37wt9PX0cP7YLxsZGYkdTKW09b0A73+cAcO/+A5QuU1uxfP11V7EjqVU9ZyeMGD4Ad+4GiR1F5bT18y2VGuPu3SCMnzBD7Chqp63XNVIfiVwul4sdQmhvTv0m2LHSM7PRZMpvWDGiE5rVcFCs77v4TzSpXg7jOjbGtC3Hoaerg18GffgbkAcvXuL7DYexa0ofuM34A8uHd0RLJ+G+KTLt+Itgx/pUvpeP4PqNO5gwcSYAQCKR4NmT61j72xYsXrJW7XnEYGlZAtGRgWjRshsuXb4mdhy10abzLgzvc3V/NzFr1mR06eyOevXbqPmVlYn1C0oqNcZ1/5MYP/4n/OTxPW7fCcIPP84RKY36adPn+53szAh06zEUhw+fFDuKWhSG61p2ZoRaXudLZDz0VdtrGVZurLbXUidRKxlxcXFYvHgxvvnmG7i4uMDFxQXffPMNlixZgpcvX4oZTSFHJkOOTA5DfV2l9Yb6urj1OBIymRyX7j9FOWsLjF57EC08fseApbvzdIV6k5mFn7aegEfPr2FpJlXnKaiMvr4+6tatBZ+zlxTr5HI5fM5eRqNGziImUy9zczMAQHxCorhB1Exbzlub3+eVKjng+bMAhDzwxbata1CmjL3YkdRmzeoFOP6Pj9LPXZtoy+dbW2nzdY3UR7RGxvXr11GlShWsXr0a5ubmaNasGZo1awZzc3OsXr0a1apVw40bN8SKpyAtZoBaDnb4/YQ/YpNSkSOT4dj1B7j7NBpxyWmIT32N1xlZ2Hz6Bhp/VQ7rxn6DlrUq4odNR3Hj4QvFcZb+dRFODnZoUUtz+rhaWpaAnp4eYmPilNbHxr6ErY2VSKnUSyKRYPnSebhyxR/374eIHUdttOm8tfV97u9/C8OGT0LHTgMwbrwHypcvi3NnD8LERDO+JPmQXr06o06dGvhpppfYUUShTZ9vbaWt17XPIpepb9FQok1hO378ePTs2RPr16+HRKLcEUAul2PUqFEYP348/Pz8PnicjIwMZGRkKK2TZWbB0EBfsKy/DGyDubvOoM3MTdDVkaBaaWu4O1dBcHgsZP/vbfZ1zQoY2LIuAKBaaSvceRqF/ZcDUa9yaZwPfAL/0HDsmdZPsExUOKxZvQCOjlXRvMU3YkdRK209b21y8uQ5xf8HBgbD3/8WHj+6hp49OmGL924Rk6lW6dL2WLFsPtzb983zu0Vb8PNNREIQrZFx584deHt752lgAG+/RZk0aRLq1Knz0eN4eXlh3jzlWT9+GtAeMwd2ECxrGSsLbJrQA28yspCangkrcymmbv4HpUqao7jUCHo6OqhoqzxQysG2BG49jgQA+IeG40VcElynrlfa58dNx1Cnoj02TeghWFZ1iouLR3Z2NqxtLJXWW1tbITqmcHR3U6VVKz3Rob0bWrTqhoiIKLHjqI22nbe2v8/fSUpKxsOHT1CxUnmxo6hU3bo1YWNjhevXTijW6enpwdW1EcaO+RbGJg6QafD8+dr2+dZWvK59AlmO2AmKPNG6S9na2sLf3/+92/39/WFjY/PR43h4eCApKUlpmdJbNQMVjQz1YWUuRfLrdPg+eI6va1WAvp4uqpezwbPYBKV9n8cmwq6EKQBgaOt62De9P/ZM66dYAODHbs0wv39rlWRVh6ysLNy8eRctWzRVrJNIJGjZoimuXg0QMZnqrVrpia5d3NG6bS88exYudhy10cbz1ub3eW5SqTEqVCiH6KhYsaOo1Nmzl+FUpyWc67dRLNdv3MauPw/CuX4bjW9gaNvnW1vxukbqIFol48cff8TIkSMREBCAVq1aKRoUMTEx8PHxwcaNG7F06dKPHsfQ0BCGhoZK694I2FUKAHyDn0Mul6O8dXGExSVixaHLcLApgS6NqgMAvm1VF1O3HEfdiqVQv0pp+AY9x8V7T/DH990BAJZm0nwHe9sWN0UpS3NBs6rbilUbsWXTCgTcvIvr12/h+/EjIJUawXvrHrGjqcya1QvQt09XdOs+FCkpqbD5f//VpKQUpKeni5xOdbT1vAHtfJ8vWjgLR4+dRljYC9jb2WL27B+QkyPD7j2HxI6mUqmpaXnGIbxOe41XrxI0enyCtn6+pVJjVKr078yRDuXLwsnJEfHxCQgPjxQxmepp43Xts2jwWAl1Ea2RMXbsWFhaWmLFihX47bffkJPztiylq6sLZ2dneHt7o1evXmLFU5LyJgNrjvgiJjEV5saGaOVUCeM6NYa+7tsZp1o6VcLM3i2x6fR1LD5wHuWsi2PpsA6oU7GUyMlVb9++w7CyLIG5s3+Era0V7ty5jw4dByA2Nu7jTy6iRo8aDAA463NAaf3QYZOwbfteMSKphbaeN6Cd7/NSpe2wY/talCxZHC9fxuOKrz+aunZCXFy82NFIBbT1813P2Qk+Z/YrHi9bOhcAsHXbXgwbPkmkVOqhjdc1Uq9CcZ+MrKwsxMW9fVNbWlpCX79glQgh75NRlIhxnwwiUg8NuofnZxH9FxQRqUyhvk/GfR+1vZahYyu1vZY6iVbJyE1fXx92dnZixyAiIiIiIgEUikYGEREREVGhwTEZBSbqHb+JiIiIiEjzsJFBRERERESCYncpIiIiIqLcNPieOOrCSgYREREREQmKlQwiIiIiolzk8hyxIxR5rGQQEREREZGgWMkgIiIiIsqNU9gWGCsZREREREQkKFYyiIiIiIhy4+xSBcZKBhERERERCYqVDCIiIiKi3Dgmo8BYySAiIiIiIkGxkkFERERElJuM98koKFYyiIiIiIhIUKxkEBERERHlxjEZBcZKBhERERFREXDx4kV06tQJ9vb2kEgkOHTokNJ2uVyO2bNnw87ODkZGRnBzc8PDhw+V9omPj0f//v1hZmYGCwsLDBs2DKmpqUr73L17F66urihWrBjKlCmDxYsXf3ZWNjKIiIiIiHKTydS3fIa0tDQ4OTlh7dq1+W5fvHgxVq9ejfXr1+PatWuQSqVo27Yt0tPTFfv0798f9+/fx+nTp3H06FFcvHgRI0eOVGxPTk5GmzZtUK5cOQQEBGDJkiWYO3cufv/998/KKpHL5fLPekYR8ObUb2JHEIVpx1/EjkBEKiIRO4BINO4XFBEpZGdGiB3hvdKv7lHbaxVr1PuLnieRSHDw4EF07doVwNsqhr29PX744Qf8+OOPAICkpCTY2NjA29sbffr0QXBwMKpXr47r16+jXr16AIATJ06gffv2ePHiBezt7bFu3TrMmDED0dHRMDAwAABMnz4dhw4dwoMHDz45HysZRERERES5yWVqWzIyMpCcnKy0ZGRkfHbkp0+fIjo6Gm5ubop15ubmaNiwIfz8/AAAfn5+sLCwUDQwAMDNzQ06Ojq4du2aYp9mzZopGhgA0LZtW4SEhCAhIeGT87CRQUREREQkEi8vL5ibmystXl5en32c6OhoAICNjY3SehsbG8W26OhoWFtbK23X09NDiRIllPbJ7xi5X+NTaOTsUtrabehN5CWxI4jCyN5V7AhEKsduQ6QNdCTa2TFQpnk914u+zxwrURAeHh6YPHmy0jpDQ0O1vb6qaGQjg4iIiIioKDA0NBSkUWFrawsAiImJgZ2dnWJ9TEwMateurdgnNjZW6XnZ2dmIj49XPN/W1hYxMTFK+7x7/G6fT8HuUkRERERERZyDgwNsbW3h4+OjWJecnIxr167BxcUFAODi4oLExEQEBAQo9jl79ixkMhkaNmyo2OfixYvIyspS7HP69GlUrVoVxYsX/+Q8bGQQEREREeVWSKewTU1Nxe3bt3H79m0Abwd73759G2FhYZBIJJg4cSI8PT1x+PBhBAYGYtCgQbC3t1fMQPXVV1/B3d0dI0aMgL+/P65cuYJx48ahT58+sLe3BwD069cPBgYGGDZsGO7fv489e/Zg1apVebp0fQy7SxERERERFQE3btxAixYtFI/f/eE/ePBgeHt7Y+rUqUhLS8PIkSORmJiIpk2b4sSJEyhWrJjiOTt37sS4cePQqlUr6OjooHv37li9erViu7m5OU6dOoWxY8fC2dkZlpaWmD17ttK9ND6FRt4nQ8+glNgRRMGB30REVJRx4Ld2Kcz3yXhz0Vttr2XU7Fu1vZY6sbsUEREREREJit2liIiIiIhyU+MUtpqKlQwiIiIiIhIUKxlERERERLnJWckoKFYyiIiIiIhIUKxkEBERERHlxjEZBcZKBhERERERCYqVDCIiIiKi3Dgmo8BYySAiIiIiIkGxkkFERERElBvHZBQYKxlERERERCQoVjKIiIiIiHLjmIwCYyWDiIiIiIgExUoGEREREVFuHJNRYKxkEBERERGRoNjIENDoUYPxKPQqUpMfw/fyEdSvV1vsSJ/sxu1AjJ06By0690eNJu3gc9FXafvp81cwYuJPaNKuF2o0aYcHoY/zHGPf3//g23FT0bB1N9Ro0g7JKal59klKTsG0uYvQsHU3uLTtgVleK/D69RuVnZcqfDdyEG4GnEZ83APExz3A5YuH4d62hdixVG7a1HHw8z2GhFchiHxxBwf2b0KVKhXFjqU2Rfnz/SVcmzbEoYPeCHsWgOzMCHTu3FbsSKKYOmUssjMjsGzpPLGjqJS2fL6bNm2Ig39twbOnN5CZ8SLP+3rWzMkIvHseCfGhiIm+h+PH/0T9+nVESqt62nZdI/ViI0MgPXt2xtIlc/Cz53LUb+iOO3eD8M+xnbCyKil2tE/y5k06qlaqgBk/jMl/e3o66tZyxKTRQ997jPT0DDRtWA8jBvV57z7T5i3Go6dh2LhyAdYunouA2/cwd/HqAudXp4iIKMyY4YUGjdqhoUt7nDt/BX8d2Izq1auIHU2lmrk2wrp1W9HEtRPc2/eFvp4+jh/bBWNjI7GjqVxR/3x/CanUGHfvBmH8hBliRxFNPWcnjBg+AHfuBokdReW05fP97n09YcLMfLc/fPgEEybORF1nN7Ro0Q3Pn73AP8d2wtKyhJqTqp42Xtc+i0ymvkVDSeRyuVzsEELTMyil9tf0vXwE12/cwYSJby9cEokEz55cx9rftmDxkrVqyfAm8pIgx6nRpB1Wec1Cq2aN82yLiIpB2x7fYv+WX1HtPd9y+d+8i6Hjp8H3xD6YmZoo1j9+FoYu/b/D7j9WocZXb/8gv3z1Bkb/OBs+B7fD+gsvbEb2rl/0PCHFRt/DtOme2OK9W+woamNpWQLRkYFo0bIbLl2+JnYclSoMn28xZWdGoFuPoTh8+KTYUdRGKjXGdf+TGD/+J/zk8T1u3wnCDz/OETuW2oj1+daRSNT2WpkZL9Cj57APvq9NTU3wKu4B2rr3xrlzV1SWRSbCn2KF4bqWnRmhltf5Em+OrVTbaxl1mKi211InVjIEoK+vj7p1a8Hn7L9/5MvlcvicvYxGjZxFTFa43LkXDDNTE0UDAwAa1asDHR0J7gY9EDHZl9PR0UGvXp0hlRrj6rUAseOolbm5GQAgPiFR3CAqxs+3dlqzegGO/+Oj9HPXJtry+f4QfX19DB/eH4mJSbirYdUsXtc+gVymvkVDFepGRnh4OIYOfX/3nMLC0rIE9PT0EBsTp7Q+NvYlbG2sREpV+MS9SkAJC3OldXp6ujA3NUVcfIJIqb5MjRrVkBgfitepT/HbrwvRo+dwBAc/FDuW2kgkEixfOg9Xrvjj/v0QseOoFD/f2qdXr86oU6cGfprpJXYUUWjT5zs/7du3QvyrEKQkP8b340egXft+ePWqaP2O+hhe10gdCvUUtvHx8di6dSs2b9783n0yMjKQkZGhtE4ul0OixpIraZ+QkMdwrt8G5mam6N69AzZvWomWbt21pqGxZvUCODpWRfMW34gdhUhQpUvbY8Wy+XBv3zfP7xZtoe2f7/PnfVG/QVuULFkCw4b2w65d69C0aSe8fPlK7GikTho8VkJdRG1kHD58+IPbnzx58tFjeHl5Yd485Vk/JDomkOiaFSjb54iLi0d2djasbSyV1ltbWyE65qXachR2liWLIz4xSWlddnYOklJSYFmiuEipvkxWVhYeP34GALh5KxD1nGtj/LjhGDN2mrjB1GDVSk90aO+GFq26ISIiSuw4KsfPt3apW7cmbGyscP3aCcU6PT09uLo2wtgx38LYxAEyDf7jQ9s+3/l5/foNHj9+hsePn8Hf/ybu37+EId/20ajxV7yukTqI2sjo2rUrJBIJPjT2/GMVCQ8PD0yePFlpXfGS1QTJ96mysrJw8+ZdtGzRVDGATCKRoGWLpvht3Ra1ZinMnGp8heSUVNx/8BCO1SoDAK4F3IZMJket6ur9mQlNR0cHhoYGYsdQuVUrPdG1iztate6JZ8/CxY6jFvx8a5ezZy/DqU5LpXV/bFyOkJDHWLJ0rcY3MLTt8/0pdHQkMDQ0FDuGoHhd+wQaPFZCXURtZNjZ2eG3335Dly5d8t1++/ZtODt/eACSoaFhng+/GF2lVqzaiC2bViDg5l1cv34L348fAanUCN5b96g9y5d4/foNwl5EKh5HRMbgQehjmJuZws7WGknJKYiKjkVs3Nty8dOwFwDeVicsS76d2i/uVTziXiUojvPw8TNIjY1gZ2sNczNTVCxfFk0b1cPcRaswe8p4ZGVnY8GKdWjn1vyLZ5YSwy+e03HixDmEhUfA1NQEfft0RfPmLmjfoZ/Y0VRqzeoF6NunK7p1H4qUlFTY/L/fblJSCtLT00VOp1pF/fP9JaRSY1Sq5KB47FC+LJycHBEfn4Dw8MgPPLNoS01NyzMO4XXaa7x6laDR4xO05fMtlRqjUsXyisfly5eBU63qiE9IxKtXCfCY/j2OHD2N6OgYlCxZAqNHDUYpe1scOHBUvNAqoo3XNVIvUaew7dy5M2rXro358+fnu/3OnTuoU6fOZ39zJMYUtgAwZvS3+GHyaNjaWuHOnfuYOGk2/K/fUtvrF2QK23fTzv5Xl3Zu+GXmDzh07DRmLlieZ/voof0xdtgAAMDaTTuwbvPOPPt4/jQZXTu0BvD2Zny/LP8N5y9fg46OBG5fN8FPE0cXaC52dU9h+/uGpWjZoins7KyRlJSCwMBgLFm6Fmd8NHsWmvdNNTh02CRs275XzWnUT+zPt7o1b+YCnzP786zfum0vhg2fJEIi8fic3qfxU9gWls+3qqewbdbMBWdO78uzftu2vRg7zgPbt/2K+vXrwNKyOF69SkBAwB0s8FqNgIA7Ks0lxhS2gPjXtUI9he3BhWp7LaNvpqvttdRJ1EbGpUuXkJaWBnd393y3p6Wl4caNG2jevPlnHVesRobYhLpPRlFTGO6TQUREBafO+2QUJmI1MsTGRsZbmtrIELW7lKvrh/84lEqln93AICIiIiIqEI7JKLBCfZ8MIiIiIiIqegr1fTKIiIiIiNROg2eSUxdWMoiIiIiISFCsZBARERER5cZKRoGxkkFERERERIJiJYOIiIiIKDctnVZYSKxkEBERERGRoFjJICIiIiLKjWMyCoyVDCIiIiIiEhQbGUREREREJCh2lyIiIiIiyo3dpQqMlQwiIiIiIhIUKxlERERERLnJWckoKFYyiIiIiIhIUKxkEBERERHlxjEZBcZKBhERERERCYqVDCIiIiKi3ORysRMUeaxkEBERERGRoFjJICIiIiLKjWMyCoyVDCIiIiIiEhQrGUREREREubGSUWBsZGgQY3tXsSOIImXfBLEjiMKi9xqxI4hCpqUXfm0dgqgjkYgdQRRyLR10WkzPQOwIoniTlSF2BCLBsZFBRERERJQb7/hdYByTQUREREREgmIlg4iIiIgoF7lMO7ssComVDCIiIiIiEhQrGUREREREuWnpJCNCYiWDiIiIiIgExUYGEREREREJit2liIiIiIhy4xS2BcZKBhERERERCYqVDCIiIiKi3DiFbYGxkkFERERERIJiJYOIiIiIKDdOYVtgrGQQEREREZGgWMkgIiIiIsqNlYwCYyWDiIiIiIgExUoGEREREVFucs4uVVCsZBARERERkaBYySAiIiIiyo1jMgqMlQwiIiIiIhIUKxlERERERLnxjt8FxkqGCkydMhbZmRFYtnSe2FFU6mHoVWRlRuRZVq/6RexoBZKWnonFh6+i3YLdaPiTNwatPYJ74S8V29eduomuS/aj0YytcJ2zHd/9fhyBYbFKx3j+MgkTvU/j67k70GTWNnz721FcfxSp7lMpkKZNG+KvA5vx9MkNZKSHo3OntkrbN25cjoz0cKXlyOHtIqVVHU19n3/MtKnj4Od7DAmvQhD54g4O7N+EKlUqih1LcE2bNsTBv7bg2dMbyMx4gc6dld/nXbu0w7FjOxEVGYjMjBdwqlVdpKSqpaOjg7lzpyA0xA/JSY/wIPgKfvppotixBDdseH/4XvsHL6Lu4EXUHZw5ux+t2zRXbD92fBeS054oLStWeYqYWDVmzZqc55oWGHhB7FikYVjJEFg9ZyeMGD4Ad+4GiR1F5Vwat4eurq7isaNjNZw8sRv7DxwVMVXBzdt/GY9iEuDZpzmszKQ4dvMRRm08jgM/dIeNuRTlrMwxvasLSpcwRXpWDnZeuofRf5zA4ak9UcLECAAwfssplLU0w+/ftYehni52Xr6P8VtO4+j0nrA0NRb5DD+N1NgIdwOD4b11L/bt3ZjvPidPnsOIkT8oHmdkZKorntpo6vv8Y5q5NsK6dVtxI+A29PT04Dl/Oo4f24WaTl/j9es3YscTjFRqjLt3g+DtvQf79v2R73bfK9exf/9RbFi/RISE6jFlylh8N3IQhg6biKCgEDg7O+GPjcuRnJSMX9duFjueYCIiojB39mI8fvQMEokEfft3w597NqBp4054EPwQALBl85/4xXOF4jlvXqeLFVel7t1/AHf3PorH2dnZIqYphOQck1FQbGQISCo1xrZtv2LU6Kn4yeN7seOoXFxcvNLjqVPG4dGjp7h40U+kRAWXnpUNn3vPsGKwG5wr2AEARrepi4vBYdjnF4xx7vXQvo7yt7k/dGqIg9dD8TAqAQ0rGyEhLR1hccmY29MVVexKAAAmtKuHvX7BeBSdUGQaGSdPncfJU+c/uE9GRiZiYl5+cJ+iThPf55+iQ6cBSo+HDp+I6MhAONethUuXr4mUSngnT57DyZPn3rt9564DAIBy5UqrK5IoXBrVw5EjJ3H8uA8A4PnzF+jduwvq168tbjCBnTh+Vunxz/OWYfjw/qhfv46ikfHmTTpiY+LEiKdWOdk5Gn/9JnGxu5SA1qxegOP/+MDn7CWxo6idvr4++vXrBu+te8SOUiA5OTLkyOQw1FNufxvq6+HWs5g8+2dl5+DAtRCYFDNAFfu3DQoLY0OUtzLHkYCHeJOZhewcGfZfC0EJk2KoXspSLeehLs2aNUJ42C0E3j2PNasXoEQJC7EjqZSmvM+/hLm5GQAgPiFR3CCkEn5Xb6BFi6aoXLkCAKBWrepo0rgBTnygAVbU6ejooHuPjjCWGsHf/6Zifa9enfH0+Q1cvX4cc+ZNgZFRMRFTqk6lSg54/iwAIQ98sW3rGpQpYy92pMJFJlffoqFEr2S8efMGAQEBKFGiBKpXV+7rmp6ejr1792LQoEHvfX5GRgYyMjKU1snlckgkEpXkfZ9evTqjTp0aaOTSQa2vW1h06eIOCwszbNu2V+woBSItZoBa5azxu88tOFibo6SpEU7cfoK7z2NRpqSZYr+LQWGYtusc0rOyYWlqjPUj3FFc+vYXkUQiwYYR7TBp6xk0nrUNOhIJSkiN8NuwtjAzNhTr1AR36tR5/H3oOJ4+C0fFCuUwf/5UHP57O5o17wKZhk79pynv888lkUiwfOk8XLnij/v3Q8SOQyqwePGvMDMzwb3AC8jJyYGuri5mzV6EP/88KHY0wVV3rIozZ/ejWDFDpKa+Rv++oxHy4BEAYN/ewwgPj0BUVCxq1KiGeT9PReXKFTCg32iRUwvL3/8Whg2fhNDQx7C1tcasmZNx7uxB1K7TEqmpaWLHIw0haiMjNDQUbdq0QVhYGCQSCZo2bYrdu3fDzu5tN5WkpCQMGTLkg40MLy8vzJunPMBaomMCia7Ze54hvNKl7bFi2Xy4t++bp8GjLYZ82wcnTp5DVFTeb/uLml/6NMfcvZfQ5pfd0NWRoFqpknCvXQHBEf+Wz+tXssOeid8gMS0df/mHYOqOs9gxvjNKmBhBLpfD65AvipsUw+bRHVFMTxd/XQ/B996nsXN8F1iZFY3uUh+zb99hxf/fv/8AgfeC8SD4Cpo3d8G5c1dETKY6mvQ+/xxrVi+Ao2NVNG/xjdhRSEV69uyEvn26YeCgsQgKCoWTkyOWLZ2HqKgYbN++T+x4gnoY+gRNXTrCzMwUXb5ph/UblqCde1+EPHgE7y27FfsF3Q9BdHQsjv6zEw4OZfH0aZiIqYWVu4tgYGAw/P1v4fGja+jZoxO2eO/+wDO1h1xDvyxTJ1G7S02bNg01atRAbGwsQkJCYGpqiiZNmiAs7NM/yB4eHkhKSlJaJDqmKkydV926NWFjY4Xr104g/fVzpL9+jubNG2P8uKFIf/0cOjqa3SutbNlSaNXKFZs37xI7iiDKlDTDptEd4Oc5CCd+6oOd47sgO0eGUiX+fV8ZGeijrKUZapWzxtyertDV0cFB/1AAgP+jKFwMDsei/i1Qp7wNviptiRnfNIGhnh6OBDwU67RU7unTMLx8+QoVK5YXO4pKaNr7/FOtWumJDu3d4NamJyIiosSOQyqy0GsWliz5FXv3Hsa9ew+wc+cBrFq9EVOnjhM7muCysrLw5Mlz3L59D/PmLEHgvQcYPebbfPe9cf02AKBCxXLqCyiCpKRkPHz4BBUrlRc7CmkQUSsZvr6+OHPmDCwtLWFpaYkjR45gzJgxcHV1xblz5yCVSj96DENDQxgaKndBUXdXqbNnL8OpTkuldX9sXI6QkMdYsnStxnYdeWfw4N6IjY3DP//4iB1FUEYG+jAy0Efy6wz4hkZgYvv6791XLpcjMzsHwNvB4wCg85/3oY5EAplcc/telipli5IliyM6KvbjOxdBmvo+/5BVKz3RtYs7WrXuiWfPwsWOQypkbGwE2X/6hufk5Gj8l2QAoKMjgaGhQb7bav5/yuLoaM0eIC2VGqNChXLYufOA2FFIg4jayHjz5g30cg2wlUgkWLduHcaNG4fmzZtj166i8Y1hampann7Kr9Ne49WrBI3vvyyRSDB4UG9s37EPOTk5YscRhG/IC8gBlLcyR1hcMlYc84eDtTm61K+CN5lZ2OhzB19XLwtLMyMkpmVgj28QYpNfo3UtBwBArXLWMDMywKw9FzHSrTaK6evhwLUQRCSkwLVaGXFP7jNIpcZKVYny5cugVq3qSEhIRHx8ImbOmISDh/5BTMxLVKhQDgt++QmPHz/DqdOaN9e6Jr7PP2bN6gXo26crunUfipSUVNjYWAEAkpJSkJ6uOVN6SqXGqPSf97lTreqIT0hEeHgkihe3QNky9rCztwUAxb1ComNeatTMPMeOncb06d8jLDwCQUEhqF27BiZOGAnvrZrVdWbOvCk4feo8XoRHwsTUBD17dYarayN80+VbODiURc9enXHq5HnExyfAsUY1LFw0E5cvXcP9ew/Eji6oRQtn4eix0wgLewF7O1vMnv0DcnJk2L3nkNjRCg8NHpCtLqI2MqpVq4YbN27gq6++Ulr/66+/AgA6d+4sRiz6DK1auaJcudLw9tac2XZS0jOx5vgNxCSlwdzYEK1qlse4tvWgr6sDmUyGZy8T8cP2h0hMS4eFcTE4lrHE5tEdUMm2OACguLQY1g5ri19PBmDk78eRnSNDRRsLrBzshqr2JUU+u0/n7FwLp0/92xd7yZI5AIBt2/dh/PifULPmVxgwoAcsLMwQGRUDnzMXMXfeUmRmat69MjTxff4xo0cNBgCc9VH+ZnPosEnYtl1zBr47OzvhzOl/3+dLl8wFAGzbthfDR0xGx46tsemPf++ZsHPnOgDAzz8vx8+ey9WaVZUmTJyJeXOnYs3qBbC2LonIyBhs/GMHPHPdL0ITWFmVxIaNy2Bra4Xk5BTcuxeCb7p8i3NnL6NUKTt83aIJxowdAmOpMSJeROHvv09gyaK1YscWXKnSdtixfS1KliyOly/jccXXH01dO+WZspuoICRyuXj9N7y8vHDp0iX8888/+W4fM2YM1q9f/9ndjfQMSgkRr8hRbyexwiN53wSxI4jCovcasSOIQtO7H76Ptn6n9t9uh9pCxF/NojLS15wZ+D7HmyztnDQmKzNC7AjvleY54OM7CUQ6c4faXkudRO1s6eHh8d4GBgD89ttvWvsHBRERERFRUSX6fTKIiIiIiAoVjskoMM2fNoKIiIiIiNSKlQwiIiIiotzYXb/AWMkgIiIiIiJBsZJBRERERJQbx2QUGCsZRERERERFQE5ODmbNmgUHBwcYGRmhYsWK+Pnnn5WmvZbL5Zg9ezbs7OxgZGQENzc3PHz4UOk48fHx6N+/P8zMzGBhYYFhw4YhNTVV0KxsZBARERER5SaXqW/5DIsWLcK6devw66+/Ijg4GIsWLcLixYuxZs2/985avHgxVq9ejfXr1+PatWuQSqVo27Yt0tPTFfv0798f9+/fx+nTp3H06FFcvHgRI0eOFOyfD2B3KSIiIiKiIsHX1xddunRBhw4dAADly5fHn3/+CX9/fwBvqxgrV67EzJkz0aVLFwDAtm3bYGNjg0OHDqFPnz4IDg7GiRMncP36ddSrVw8AsGbNGrRv3x5Lly6Fvb29IFlZySAiIiIiyk0mV9uSkZGB5ORkpSUjI/+7wDdu3Bg+Pj4IDQ0FANy5cweXL19Gu3btAABPnz5FdHQ03NzcFM8xNzdHw4YN4efnBwDw8/ODhYWFooEBAG5ubtDR0cG1a9cE+ydkI4OIiIiISCReXl4wNzdXWry8vPLdd/r06ejTpw+qVasGfX191KlTBxMnTkT//v0BANHR0QAAGxsbpefZ2NgotkVHR8Pa2lppu56eHkqUKKHYRwjsLkVERERElItcjffJ8PDwwOTJk5XWGRoa5rvv3r17sXPnTuzatQuOjo64ffs2Jk6cCHt7ewwePFgdcT8ZGxlERERERCIxNDR8b6Piv6ZMmaKoZgBAzZo18fz5c3h5eWHw4MGwtbUFAMTExMDOzk7xvJiYGNSuXRsAYGtri9jYWKXjZmdnIz4+XvF8IbC7FBERERFRbmock/E5Xr9+DR0d5T/fdXV1Ift/5cXBwQG2trbw8fFRbE9OTsa1a9fg4uICAHBxcUFiYiICAgIU+5w9exYymQwNGzb80n+xPFjJICIiIiIqAjp16oRffvkFZcuWhaOjI27duoXly5dj6NChAACJRIKJEyfC09MTlStXhoODA2bNmgV7e3t07doVAPDVV1/B3d0dI0aMwPr165GVlYVx48ahT58+gs0sBbCRQURERERUJKxZswazZs3CmDFjEBsbC3t7e3z33XeYPXu2Yp+pU6ciLS0NI0eORGJiIpo2bYoTJ06gWLFiin127tyJcePGoVWrVtDR0UH37t2xevVqQbNK5LlvEagh9AxKiR1BFBKxA4gked8EsSOIwqL3mo/vpIFkahyMV5ho3IX6E+lItPPKpoG/mj+Jkf6n9UvXNG+y8p+uVNNlZUaIHeG9Uqd8o7bXMllyUG2vpU4ck0FERERERIJidykiIiIiotzk2lk1FxIrGUREREREJChWMoiIiIiIcvvMqWUpLzYyNIi2fhxK9FkrdgRRJO74TuwIojDtt07sCKLQ09EVO4IosmU5YkcQhXYOdwcyc7LFjiAKiZZOcECajY0MIiIiIqJc5KxkFBjHZBARERERkaBYySAiIiIiyo2VjAJjJYOIiIiIiATFSgYRERERUW4y3iejoFjJICIiIiIiQbGSQURERESUG8dkFBgrGUREREREJChWMoiIiIiIcmMlo8BYySAiIiIiIkGxkkFERERElItczkpGQbGSQUREREREgmIlg4iIiIgoN47JKDBWMoiIiIiISFBsZBARERERkaDYXYqIiIiIKDd2lyowVjKIiIiIiEhQrGQQEREREeUiZyWjwFjJICIiIiIiQbGSQURERESUGysZBcZKBhERERERCYqNDAG4Nm2IQwe9EfYsANmZEejcua3YkdRq9KjBeBR6FanJj+F7+Qjq16stdiRB/fjjGFy+fBixsffx/HkA9u79HZUrV1BsL17cHMuXz8OdO2cRHx+C0FBfLFs2F2ZmpiKm/nxpGVlYfOw62i35Cw3n7MKgDSdw70UcACArR4aVJ26ix+ojaDR3F1ov3I+Z+64gNvm14vnXn0Sj9ozt+S7vjlMUacvnu2nTBjhwYDOePLmO9PQwdOrURmm7tbUlNm5chidPriM+PgSHD29DxYrlxQmrQtOmjoOf7zEkvApB5Is7OLB/E6pUqSh2LLWwt7fFVu/ViI66h+SkR7h18wyc69YS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a6mTqI2M9evXo0yZMujQoUO+23/66SfExsbijz/++KzjinmfDFI/dd8no7AQupFRVIh5nwwxqfM+GYWJ2PfJEIu2dvpV530yChN13iejMCnUjYzuX6vttYofOK+211InUf86GzVq1Ae3L1jAO08SERERERU1HJVJRERERESC0s5+JkRERERE78GB3wXHSgYREREREQmKlQwiIiIioty0cyy+oFjJICIiIiIiQbGSQURERESUi5bOKiwoVjKIiIiIiEhQrGQQEREREeXGSkaBsZJBRERERESCYiWDiIiIiCgXjskoOFYyiIiIiIhIUKxkEBERERHlxkpGgbGSQUREREREgmIlg4iIiIgoF47JKDhWMoiIiIiISFCsZBARERER5cJKRsGxkkFERERERIJiJYOIiIiIKBdWMgqOlQwiIiIiIhIUKxlERERERLnJJWInKPLYyKAiLzsnW+wIojDtt07sCKJI2Tpc7AiiMB38h9gRRKEj0c5f9DK5XOwIopBpaR8VI31DsSMQCY7dpYiIiIiISFCsZBARERER5aKlRTVBsZJBRERERESCYiWDiIiIiCgXuUw7x4MJiZUMIiIiIiISFCsZRERERES5cExGwbGSQUREREREgmIlg4iIiIgoFzlvxldgrGQQEREREZGgWMkgIiIiIsqFYzIKjpUMIiIiIiISFCsZRERERES58D4ZBcdKBhERERERCYqVDCIiIiKiXORysRMUfaxkEBERERGRoFjJICIiIiLKhWMyCo6VDCIiIiIiEhQrGUREREREubCSUXCsZBARERERkaDYyCAiIiIiIkGxuxQRERERUS6cwrbgWMkQgGvThjh00BthzwKQnRmBzp3bih1JLbT1vGfNmoyszAilJTDwgtixVG7a1HHw8z2GhFchiHxxBwf2b0KVKhXFjlVgaRlZWHz8JtqtOIyGnvsw6I/TuBfxSrHdJygco7adQ/NFf6H23N14EJWQ5xj7bzzCsC0+aLJgP2rP3Y3kN5nqPAWVGj1qMB6FXkVq8mP4Xj6C+vVqix1JUE2bNsTBv7bg2dMbyMx4ke91bM7sH/H8WQCSEh/h+PE/UamSgwhJVeu7kYNwM+A04uMeID7uAS5fPAz3ti3EjqUWJiZSLF06Fw9DryIp8REunD8EZ2cnsWOpzKTJ3yEp9TG8Fs1UrDM0NMDS5XPx9PkNRETf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Y4dGjjXKrVRpO+zYvhYlSxbHy5fxuOLrj6aunfJM2V2UTGtRHb/5PsSCs0FIeJ0JKxND9KhZBiMbVVLa72RIFAA53KvZ5TnG1edxCE98jfDE12i78bzStluT3FWYXljOzk44c/rf7n5Ll8wFAGzbthfDR0xGx46tsemPf9/jO3e+nar6f+3de1hU5d4+8HsYZBxgEE8goIBIKpqSQPCiFqnkYRvh9tLIbIei7l1BgaQmlaKhorU1zfyBpuIByWOSocYmSpTUDUKYRxTzHIL8VBCM08x6/2g3m3lRU5mZJ2fuz3WtP2bNYtb9wJphvvNdz5qEhCVImLfEqFn/LDgno+VkksA+zaxZs7By5UqEhoYiOzsbYWFhSEtLQ1xcHGQyGebPn48xY8ZgyZKHO8BbWZnWOcJEd2MpF37dBiHUGtPsHP0RucXdJ6Oauka1aRRy9GAqP737xSVMnX30DtERhKiv+/N+GOXvHGS0feX9kmO0fRmT0Hcpc+fOhVKpxKFDhzBlyhTMnDkT3t7emDFjBu7cuYOQkBAkJCSIjEhEREREZkZiJ6PFhHYyDIWdDDIH7GSYF3YyyBywk2Fe/sydjKednzXavvJ/2W+0fRmTeb5LISIiIiK6BxP8DN7ozOuafEREREREZHDsZBARERERNcGrS7UcOxlERERERKRX7GQQERERETXBORktx04GERERERHpFTsZRERERERNcE5Gy7GTQUREREREesVOBhERERFRE/zG75ZjJ4OIiIiIiPSKRQYREREREekVT5ciIiIiImpCw0vYthg7GUREREREpFfsZBARERERNcGJ3y3HTgYREREREekVOxlERERERE1wTkbLsZNBRERERER6xU4GEREREVETnJPRcuxkEBERERGRXrGTQURERETUBOdktJxJFhkymUx0BCEkM31CmOeoAbVGLTqCEHILuegIQjSqG0VHEMJcn9/m+V8MaPP2dtERhKjaFi06ApHemWSRQURERET0qDgno+U4J4OIiIiIiPSKnQwiIiIioiY4J6Pl2MkgIiIiIiK9YieDiIiIiKgJzsloOXYyiIiIiIhIr9jJICIiIiJqQpI0oiM89tjJICIiIiIivWKRQUREREREesXTpYiIiIiImtBw4neLsZNBRERERER6xU4GEREREVETEr+Mr8XYySAiIiIiIr1iJ4OIiIiIqAnO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uev2ffr0gUajQU5OjvZ0qaZ+76So1Wrtul69ekGhUODSpUv37IB4eXlh165dOusOHz78x4MkIiK6C078JiIykPHjx6NDhw4IDQ3FgQMHcP78eezbtw9vv/02rly5AgCIjo7GwoULkZ6ejtOnT+PNN9+873dcuLu7Izw8HBEREUhPT9c+5tatWwEAbm5ukMlkyMjIwPXr11FdXQ2VSoVp06Zh6tSpWL9+Pc6dO4fCwkIsX74c69evBwC8/vrrOHv2LKZPn47i4mKkpaVh3bp1hv4VERGRiWKRQURkINbW1ti/fz9cXV0xevRoeHl5YdKkSaitrdV2Nt555x387W9/Q3h4OAIDA6FSqfDXv/71vo+blJSEMWPG4M0330TPnj0xZcoU1NTUAABcXFwwd+5czJw5E46OjoiKigIAJCQkYNasWUhMTISXlxeGDx+O3bt3o2vXrgAAV1dX7NixA+np6fD29kZycjIWLFhgwN8OERGZMpl0r9mFREREREREj4CdDCIiIiIi0isWGUREREREpFcsMoiIiIiISK9YZBARERERkV6xyCAiIiIiIr1ikUFERERERHrFIoOIiIiIiPSKRQYREREREekViwwiIiIiItIrFhlERERERKRXLDKIiIiIiEivWGQQEREREZFe/S/q0Ulzjnbk5gAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x700 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sn\n",
    "plt.figure(figsize=(10, 7))\n",
    "sn.heatmap(cm, annot=True, fmt='d')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel(\"Truth\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "62c31d8d-8b9d-498c-9c42-1540c99c7b71",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 3ms/step - accuracy: 0.8794 - loss: 0.4457\n",
      "Epoch 2/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 3ms/step - accuracy: 0.9622 - loss: 0.1286\n",
      "Epoch 3/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 3ms/step - accuracy: 0.9739 - loss: 0.0882\n",
      "Epoch 4/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 3ms/step - accuracy: 0.9811 - loss: 0.0633\n",
      "Epoch 5/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 3ms/step - accuracy: 0.9842 - loss: 0.0515\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x2cbd402c620>"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = keras.Sequential([\n",
    "    keras.layers.Dense(100, input_shape=(784,), activation='relu'),\n",
    "    keras.layers.Dense(10, activation='sigmoid')\n",
    "])\n",
    "model.compile(\n",
    "    optimizer='adam', \n",
    "    loss='sparse_categorical_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "model.fit(x_train_flattend, y_train, epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "3b6278e6-9ed0-48f4-928d-79d268132785",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9694 - loss: 0.0957\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.07982977479696274, 0.9746999740600586]"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(x_test_flattend, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "bc91ea7a-536d-432d-b8d8-942ae9e9ef6d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(95.72222222222221, 0.5, 'Truth')"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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+++/8fjx45eu60tERERERKbtnU6XKlKkiM7YC7VajebNm+uso1arIZPJoFQqhU1IRERERCSi/J3nQ6/yTp2MI0eOGDoHEREREREZiXfqZDRp0kTz/3FxcShZsuRLV5VSq9W4d++esOmIiIiIiESW30vL0svyfZ8MLy8vPHr06KX5KSkpvE8GERERERHl/xK2/4y9+K+MjAydS2MRERERERVGalYy9PbOnYxRo0YBeHGTj4kTJ8LGxkazTKlU4syZM5rblRMRERERkel6507G+fPnAbyoZERHR8PS0lKzzNLSEt7e3hg9erTwCYmIiIiIRMSrS+nvnTsZ/1xh6quvvsKiRYvg4OBgsFBERERERFR45XtMxpo1awyRg4iIiIioQODVpfSX705Gs2bN3rj88OHD7x2GiIiIiIgKv3x3Mry9vXUe5+bm4sKFC7h06RL69OkjWDAiejP5K67yZgrsuwRJHUESaXPbSR1BEg5jdksdgcjg8lRKqSPQf/DqUvrLdycjKOjVX/BTpkxBRkaG3oGIiIiIiKhwy/fN+F6nV69eWL16tVCbIyIiIiKShEotE20yVoJ1MsLCwngzPiIiIiIiyv/pUp06ddJ5rFar8fDhQ5w9exYTJ04ULBgRERERkRR4mwz95buT4ejoqPNYLpfjgw8+wLRp09CiRQvBghERERERUeGUr06GUqnEV199herVq6No0aKGykRERERERIVYvsZkmJmZoUWLFnj69KmB4hARERERSYsDv/WX74Hf1apVw61btwyRhYiIiIiIjEC+OxkzZszA6NGjsWfPHjx8+BBpaWk6ExERERFRYaZWy0SbjNU7j8mYNm0avv32W7Rp0wYA0L59e8i07jisVqshk8mgVPKulUREREREpuydOxlTp07FoEGDcOTIEUPmISIiIiKSlErqAEbgnTsZavWLKwY3adLEYGGIiIiIiKjwy9clbLVPjyIiIiIiMkZq8G9efeWrk1GxYsW3djRSUlL0CkRERERERIVbvjoZU6dOfemO30RERERExkSlljpB4ZevTkb37t3h6upqqCxERERERGQE3rmTwfEYRERERGQKVByTobd3vhnfP1eXIiIiIiIiepN3rmSoVLxiMBEREREZP15dSn/vXMkgIiIiIiJ6F+xkCODrgb1xLvIgUpJjkJIcgxPHdqFVy6ZSxxLN4EF9cOPa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EgAiIyNHHv91aVVQUBABAQHAP2Y+iouLOXDgAElJSSPPOXToEOXl5SxatIjS0lJmzJjBxo0bAYiOjqajo4NXX33Vg2dNREQmCpUMEZGr0NDQgL+/P5cuXcLtdvPggw/y0ksvkZWVRVxc3Kj7ML744gu6u7ux2+2jjjEwMEBPTw/nz5/nzJkzLFiwYOSxSZMmkZiYeMWSqV+5XC6sViuLFi3605m7u7u5ePEid95556jtQ0NDzJs3D4Bjx46NygGMFBIREZGrpZIhInIVkpOTKS0txWazERoayqRJ//w36ufnN2rfvr4+EhIS2LFjxxXHmTp16l8a39fX96qf09fXB8C+ffsICwsb9ZiPj89fyiEiIvJHVDJERK6Cn58fUVFRf2rf+Ph4du3aRVBQENdff/1v7hMSEsKnn37K7bffDsDly5dpb28nPj7+N/ePi4vD7XbT2to6slzqX/06kzI8PDyybdasWfj4+PDtt9/+7gxIbGwse/fuHbXt8OHD//lFioiI/Abd+C0i4iXp6elMmTKFZcuWcfDgQU6cOEFLSwtr1qzhu+++AyAnJ4dXXnmF+vp6vvrqK1avXv2H33Fx880343Q6yczMpL6+fuSYNTU1AISHh2OxWGhoaODcuXP09fVht9tZv34969atY/v27fT09HD06FG2bNnC9u3bAXjiiSf4+uuvycvLo6uri+rqaqqqqrx9ikREZJxSyRAR8ZLrrruOtrY2HA4Hy5cvJzY2llWrVjEwMDAys/H000/z8MMP43Q6SUpKwm63c9999/3hcUtLS7n//vtZvXo1MTExPP744/T39wMQFhbGhg0byM/PZ9q0aWRnZwNQWFjICy+8QElJCbGxsaSkpLBv3z4iIiIAcDgc1NbWUl9fz9y5cykrK6O4uNiLZ0dERMYzi/F7dxeKiIiIiIj8BZrJEBERERERj1LJEBERERERj1LJEBERERERj1LJEBERERERj1LJEBERERERj1LJEBERERERj1LJEBERERERj1LJEBERERERj1LJEBERERERj1LJEBERERERj1LJEBERERERj1LJEBERERERj/o7fm72zQhxNj0AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1000x700 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "y_predicted= model.predict(x_test_flattend)\n",
    "y_predicted_labels=[np.argmax(i) for i in y_predicted]\n",
    "cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)\n",
    "\n",
    "plt.figure(figsize=(10, 7))\n",
    "sn.heatmap(cm, annot=True, fmt='d')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel(\"Truth\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "3731ad1b-b648-429d-a1f2-a58c9d217eed",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Admin\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\keras\\src\\layers\\reshaping\\flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(**kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 3ms/step - accuracy: 0.8773 - loss: 0.4466\n",
      "Epoch 2/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 3ms/step - accuracy: 0.9603 - loss: 0.1366\n",
      "Epoch 3/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 3ms/step - accuracy: 0.9736 - loss: 0.0907\n",
      "Epoch 4/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 3ms/step - accuracy: 0.9798 - loss: 0.0698\n",
      "Epoch 5/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 3ms/step - accuracy: 0.9837 - loss: 0.0541\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x2cbd717eb40>"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = keras.Sequential([\n",
    "    keras.layers.Flatten(input_shape = (28,28)),\n",
    "    keras.layers.Dense(100, activation='relu'),\n",
    "    keras.layers.Dense(10, activation='sigmoid')\n",
    "])\n",
    "model.compile(\n",
    "    optimizer='adam', \n",
    "    loss='sparse_categorical_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "model.fit(X_train, y_train, epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c542b02f-413c-492b-84b9-62a2ca9f534e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
