#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr 24 13:09:23 2026

@author: david
"""

import numpy as np
from train import train
from datetime import datetime
from tempfile import gettempdir

input_noises = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
n_neurons = np.array([50])
saves = []
 
for n_neuron in n_neurons:
    for input_noise in input_noises:
        logdir: str = datetime.now().strftime(f"{gettempdir()}/%y%m%d-%H%M%S")
        train(logdir = logdir, input_noise=input_noise, neurons = int(n_neuron))
        saves.append(logdir.replace("/tmp/", "") + "_ruda")


print(saves)