#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

"""
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

# import data


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

batchsize = 100
n_epoch = 20
n_units = 1000

# Prepare dataset
print('prepare databset')

import csv
x, y = []. []
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[4] == 'Iris-setosa':
            y.append(0)
        else:
            y.append(1)
        del row[4]
        x.append([float(i) for i in row])

x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.int32)
x /= 10

print x,y

N = 60
x_train, x_test = np.split( x.astype(np.float32), [N])
y_train, y_test = np.split( y.astype(np.int32), [N])
N_test = y_test.size

# Prepare multi-layer perceptron model
model = chainer.FunctionSet(l1=F.Linear(4, n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, 1))
if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

# Neural net architecture


def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = x_train[perm[i:i + batchsize]]
        y_batch = y_train[perm[i:i + batchsize]]
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        if epoch == 1 and i == 0:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = x_test[i:i + batchsize]
        y_batch = y_test[i:i + batchsize]
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

