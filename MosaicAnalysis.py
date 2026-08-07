# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:03:56 2023
@author: David
"""

import matplotlib
import numpy as np
import torch
from util import find_last_cp, find_last_model
from pathlib import Path
import time

parent_path = str(Path.cwd().parent) + '/'
saves_path = parent_path + "saves/"

save = '260512-042335_ruda'

epoch = None

n_clusters_global = 1 #Best value is 1.
n_comps_global = 1 #Best value is 1. 

start = time.time()

matplotlib.use("QtAgg")

def DoG_from_params(rr,a,b,c,):
    return np.exp(-a * rr) - c*np.exp(-b * rr)

class Analysis():
    
        #Retrieves variables from the cp and model files. 
        def __init__(self, save, path = saves_path, epoch = None):
            self.path = path + save + "/"
            self.save = save
            
            if epoch == None:
                last_cp_file = find_last_cp(self.path)
                last_model_file = find_last_model(self.path)
                epoch = int(last_model_file[6:-3])
            else:
                last_cp_file = "checkpoint-" + str(epoch) + ".pt"
                last_model_file = "model-" + str(epoch) + ".pt"
            self.max_iteration = int(last_cp_file[11:-3])
            print(self.max_iteration)
            self.cp = torch.load(self.path + last_cp_file)
            self.input_noise = self.cp['args']['input_noise']
            self.output_noise = self.cp['args']['output_noise']
            self.learning_rate = self.cp['args']['learning_rate']
            self.model = torch.load(self.path + last_model_file)
            self.kernel_size = self.cp['args']['kernel_size']
            self.n_neurons = self.cp['args']['neurons']
            self.w_flat = self.model.encoder.W.cpu().detach().numpy()
            self.w_flat = np.swapaxes(self.w_flat,0,1)
            self.w = np.reshape(self.w_flat, [self.n_neurons, self.kernel_size, self.kernel_size])
            self.W = self.w
            
            
        
        def compute_center_surround(self):
            center_surround_ratio = []
            for n in range(self.n_neurons):
                ON_sum = np.sum(np.clip(self.W[n,:,:],0,np.inf))
                OFF_sum = abs(np.sum(np.clip(self.W[n,:,:],-np.inf,0)))
                if np.max(self.W[n,:,:]) > abs(np.min(self.W[n,:,:])):
                    center_sum = ON_sum
                    surround_sum = OFF_sum
                    ratio = surround_sum/center_sum
                else:
                    center_sum = OFF_sum
                    surround_sum = ON_sum
                    ratio = surround_sum/center_sum
                    
                center_surround_ratio.append(ratio)
            self.center_surround_ratio = center_surround_ratio
        
        #Retrieves the DoG parameters (a, b, c, kernel_centers) from the saved model. 
        def get_DoG_params(self):
            self.a = self.model.encoder.shape_function.a.cpu().detach().numpy()
            self.b = self.model.encoder.shape_function.b.cpu().detach().numpy()
            self.c = self.model.encoder.shape_function.c.cpu().detach().numpy()
   
            self.kernel_centers = self.cp['model_state_dict']['encoder.kernel_centers'].cpu().detach().numpy()
        

        def get_pathways(self, n_clusters, from_pca = False):
                self.pathway = []
                for n in range(self.n_neurons):
                    Wn = self.W[n,:,:]
                    if np.max(Wn) > abs(np.min(Wn)):
                        self.pathway.append('ON')
                    else:
                        self.pathway.append('OFF')
        
        #Do zero_crossings but for a fraciton of the maximum value instead.
        def half_crossings(self, cross_min):
            self.half_cross = []
            step_size = 0.01
            for n in range(self.n_neurons):
                
                dog_rf, x = self.DoG_neuron(n, x_range=15, step_size=step_size)
                max_value = np.max(dog_rf)
                half_value = max_value * cross_min
                dog_rf_right = dog_rf[x >= 0]
                has_crossed = dog_rf_right <= half_value
                half_cross = np.argmax(has_crossed)
                self.half_cross.append(half_cross*step_size)
            self.half_cross = np.array(self.half_cross)
        
        #Makes 1D gaussian from the median a b and c parameters across the population. 
        def DoG_median(self, x_range = 10, step_size = 0.01):
            a = np.median(self.a)
            b = np.median(self.b)
            c = np.median(self.c)
            x_neg = np.arange(-x_range, 0, step_size)
            x_pos = np.arange(0,x_range,step_size)
            rr = np.append(x_neg**2,x_pos**2)
            dog_rf = DoG_from_params(rr,a,b,c)
            dog_rf = dog_rf / np.linalg.norm(dog_rf)
            
            return dog_rf, np.append(x_neg,x_pos)
            
        #Makes 1D gaussian from the median a b and c parameters of an individual neuron. 
        def DoG_neuron(self, n, x_range = 10, step_size = 0.01):
            a = np.median(self.a[n])
            b = np.median(self.b[n])
            c = np.median(self.c[n])
            x_neg = np.arange(-x_range, 0, step_size)
            x_pos = np.arange(0,x_range,step_size)
            rr = np.append(x_neg**2,x_pos**2)
            dog_rf = DoG_from_params(rr,a,b,c)
            dog_rf = dog_rf / np.linalg.norm(dog_rf)
            
            return dog_rf, np.append(x_neg,x_pos)
            
        
        def __call__(self, n_comps = None, n_clusters = None):
            if n_comps is None:
                n_comps = n_comps_global
                
            if n_clusters is None:
                n_clusters = n_clusters_global

            self.get_DoG_params()
            self.get_pathways(n_clusters)
            self.compute_center_surround()
            matplotlib.use("QtAgg")
                
test = Analysis(save, saves_path, epoch)
test(n_comps_global, n_clusters_global)
