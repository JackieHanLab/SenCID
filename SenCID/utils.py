# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:41:03 2022

@author: admin
"""

import os

def Mkdir(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print('make dir:', pathname)
