#!/usr/bin/env python

import h5py
import numpy as np

U = np.array([3.0, 3.0, 3.0, 3.0])
xmu = np.array([1.5, 1.5, 1.5, 1.5])
t = np.array([[ 0.0, -1.0, -1.0,  0.3],
              [-1.0,  0.0,  0.3, -1.0],
              [-1.0,  0.3,  0.0, -1.0],
              [ 0.3, -1.0, -1.0,  0.0]])

Ns = len(U)
sectors = np.array([[Ns/2,Ns/2],])


data = h5py.File("input.h5", "w");

hop_g = data.create_group("sectors")
hop_g.create_dataset("values", data=sectors)

hop_g = data.create_group("hopping")
hop_g.create_dataset("values", data=t)

int_g = data.create_group("interaction")
int_ds = int_g.create_dataset("values", shape=(Ns,), data=U)

int_g = data.create_group("chemical_potential")
int_ds = int_g.create_dataset("values", shape=(Ns,), data=xmu)
