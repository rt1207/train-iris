#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
from pymongo import MongoClient

import numpy as np
import random
from pprint import pprint

#connection
connect = MongoClient('localhost', 27017)

#testdbを取得
db = connect.mydb
#次のような記述もOK db = con['test']

#db名を出力
print "db name is = "
print db.name

# collections
users = db.users
acts = db.activations

# test data
# ユーザー情報にひも付けて、最適な配信時間を予測する
if users.find().count() == 0:
    users.save({'uid':10, 'gender': 1, 'student': 1, 'working': 1})

# Activated: True, acv_rate_at_lessonが高いものが教師データ
if acts.find().count() == 0:
    acts.save({'uid':10, 'acv_rate': random.randint(0, 30), 'acv_rate_at_lesson': 100, 'most_login_hour': 21, 'hours_to_lesson': 24*3, 'activated': 1})
    acts.save({'uid':10, 'acv_rate': random.randint(80, 100), 'acv_rate_at_lesson': 90, 'most_login_hour': 21, 'hours_to_lesson': 1, 'activated': 0})

# setup datasets
# np.arrays
for data in acts.find({'uid':10}):
    del data['_id']
    print data.values()
