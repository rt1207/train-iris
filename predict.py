#!/usr/bin/env python
#coding:utf-8

# python predict.py -m model.npz -i '5.9 3.0 5.1 1.8'

from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import numpy as np

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def predict(model, x_test):
    x = chainer.Variable(x_test)

    h1 = F.dropout(F.relu(model.predictor.l1(x)))
    h2 = F.dropout(F.relu(model.predictor.l2(h1)))
    y = model.predictor.l3(h2)

    return np.argmax(y.data)

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--model', '-m', type=str, default='')
    parser.add_argument('--input', '-i', default='', type=str)
    args = parser.parse_args()

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.model == '' or args.input == '':
        print('Error: missingArgumentException')
    else:
        with open(args.model, 'rb') as i:

            model = L.Classifier(MLP( 4, args.unit, 2))
            serializers.load_npz(i, model)

            if len(args.input.split()) == 4:
                x_test = args.input.split() # [5.9,3.0,5.1,1.8]
                print( predict( model, np.array([x_test], dtype=np.float32)/10) )

            else:
                print('Error: invalidInput')

if __name__ == '__main__':
    main()
