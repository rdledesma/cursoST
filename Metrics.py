#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:57:37 2023

@author: solar
"""
import numpy as np
def mbe(true, pred):
    mbe_loss = np.sum(pred - true)/true.size
    return mbe_loss


def mae(true, pred):
    mbe_loss = np.sum(abs(pred - true))/true.size
    return mbe_loss

def rmsd(true, pred):
    return np.sqrt(sum((pred - true) ** 2) / true.size)


def rmbe(true, pred):
    mbe_loss = np.mean(pred  - true)
    return mbe_loss/ true.mean() * 100


def rmae(true, pred):
    mbe_loss = np.sum(abs(pred - true))/true.size
    return mbe_loss/ true.mean() * 100

def rrmsd(true, pred):
    return np.sqrt(sum((pred - true) ** 2) / true.size)  / true.mean() * 100

