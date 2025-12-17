# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:03:19 2023

@author: admin
"""
import numpy as np
import random
import os
import re

def extract_magnitude(filename):
    match = re.search(r'CNB(\d+)\.vtk$', filename)
    if match:
        return int(match.group(1))

    return None

def get_sorted_files(address):
    files = os.listdir(address)
    cnb_files = [file for file in files if re.match(r'CNB\d+\.vtk$', file)]
    sorted_files = sorted(cnb_files, key=extract_magnitude)

    return sorted_files

def save_three_dimension(fai, kappa, num_t, address, savepath, seed="None"):
    data_dict = {}
    if seed == "None":
        for i in range(len(kappa)):
            dataset = np.empty((len(fai),num_t,128,128,128),dtype=np.float32)
            for j in range(len(fai)):
                address_fs = address+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])
                file_n = get_sorted_files(address_fs)
                for k in range(num_t):
                    address_f = os.path.join(address_fs, file_n[k])
                    with open(address_f,'r',encoding='UTF-8') as data:
                        lines = data.readlines()
                        lines = lines[10:]
                        lines = np.array(lines)
                        lines = np.reshape(lines, (128,128,128))
                        dataset[j,k] = lines
            key_name = "kappa"+str(kappa[i])
            data_dict[key_name] = dataset
    else:
        for i in range(len(kappa)):
            index = -1
            dataset = np.empty((len(fai)*len(seed),num_t,128,128,128),dtype=np.float32)
            for j in range(len(fai)):
                for m in range(len(seed)):
                    index += 1
                    address_fs = address+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])+\
                    "_seed="+str(seed[m])+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])+\
                    "_seed="+str(seed[m])
                    file_n = get_sorted_files(address_fs)
                    for k in range(num_t):
                        address_f = os.path.join(address_fs, file_n[k])
                        with open(address_f,'r',encoding='UTF-8') as data:
                            lines = data.readlines()
                            lines = lines[10:]
                            lines = np.array(lines)
                            lines = np.reshape(lines, (128,128,128))
                            dataset[index,k] = lines
            key_name = "kappa"+str(kappa[i])
            data_dict[key_name] = dataset

    np.savez(savepath, **data_dict)

    return 0

def save_two_dimension(fai, kappa, num_t, address, savepath, seed="None"):
    data_dict = {}
    if seed == "None":
        for i in range(len(kappa)):
            dataset = np.empty((len(fai),num_t,128,128),dtype=np.float32)
            for j in range(len(fai)):
                address_fs = address+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])
                file_n = get_sorted_files(address_fs)
                for k in range(num_t):
                    address_f = os.path.join(address_fs, file_n[k])
                    with open(address_f,'r',encoding='UTF-8') as data:
                        lines = data.readlines()
                        lines = lines[10:]
                        lines = np.array(lines)
                        lines = np.reshape(lines, (128,128))
                        dataset[j,k] = lines
            key_name = "kappa"+str(kappa[i])
            data_dict[key_name] = dataset
    else:
        for i in range(len(kappa)):
            index = -1
            dataset = np.empty((len(fai)*len(seed),num_t,128,128),dtype=np.float32)
            for j in range(len(fai)):
                for m in range(len(seed)):
                    index += 1
                    address_fs = address+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])+\
                    "_seed="+str(seed[m])+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])+\
                    "_seed="+str(seed[m])
                    file_n = get_sorted_files(address_fs)
                    for k in range(num_t):
                        address_f = os.path.join(address_fs, file_n[k])
                        with open(address_f,'r',encoding='UTF-8') as data:
                            lines = data.readlines()
                            lines = lines[10:]
                            lines = np.array(lines)
                            lines = np.reshape(lines, (128,128))
                            dataset[index,k] = lines
            key_name = "kappa"+str(kappa[i])
            data_dict[key_name] = dataset

    np.savez(savepath, **data_dict)

    return 0

def save_one_dimension(fai, kappa, num_t, address, savepath, seed="None"):
    data_dict = {}
    if seed == "None":
        for i in range(len(kappa)):
            dataset = np.empty((len(fai),num_t,128),dtype=np.float32)
            for j in range(len(fai)):
                address_fs = address+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])+\
                            "/fai="+str(fai[j])+"_kappa="+str(kappa[i])
                file_n = get_sorted_files(address_fs)
                for k in range(num_t):
                    address_f = os.path.join(address_fs, file_n[k])
                    with open(address_f,'r',encoding='UTF-8') as data:
                        lines = data.readlines()
                        lines = lines[10:]
                        lines = np.array(lines)
                        lines = np.reshape(lines, (128))
                        dataset[j,k] = lines
            key_name = "kappa"+str(kappa[i])
            data_dict[key_name] = dataset
    else:
        for i in range(len(kappa)):
            index = -1
            dataset = np.empty((len(fai)*len(seed),num_t,128),dtype=np.float32)
            for j in range(len(fai)):
                for m in range(len(seed)):
                    index += 1
                    address_fs = address+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])+\
                    "_seed="+str(seed[m])+"/fai="+str(fai[j])+"_kappa="+str(kappa[i])+\
                    "_seed="+str(seed[m])
                    file_n = get_sorted_files(address_fs)
                    for k in range(num_t):
                        address_f = os.path.join(address_fs, file_n[k])
                        with open(address_f,'r',encoding='UTF-8') as data:
                            lines = data.readlines()
                            lines = lines[10:]
                            lines = np.array(lines)
                            lines = np.reshape(lines, (128))
                            dataset[index,k] = lines
            key_name = "kappa"+str(kappa[i])
            data_dict[key_name] = dataset

    np.savez(savepath, **data_dict)

    return 0

if __name__=="__main__":
    fai = np.round(np.linspace(0.325,0.675,8),3)
    kappa = np.linspace(4.5,11.5,8)
    seed = np.linspace(1,10,10).astype(int)
    num_t = 100
    address = "/user"
    savepath = "/user/train.npz"
    
    save_two_dimension(fai, kappa, num_t, address, savepath, seed)
        
