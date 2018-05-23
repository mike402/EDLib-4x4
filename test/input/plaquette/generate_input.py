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

dmorbs = np.array([[0, 1],
                   [3, 1],
                   [1, 2],
                   [3, 0]])
fulldmorbs = np.array([0, 1, 2, 3])


data = h5py.File("input.h5", "w");

hop_g = data.create_group("sectors")
hop_g.create_dataset("values", data=sectors)

hop_g = data.create_group("hopping")
hop_g.create_dataset("values", data=t)

int_g = data.create_group("interaction")
int_ds = int_g.create_dataset("values", shape=(Ns,), data=U)

int_g = data.create_group("chemical_potential")
int_ds = int_g.create_dataset("values", shape=(Ns,), data=xmu)

for ii in range(dmorbs.shape[0]):
  data.create_group("DensityMatrix" + str(ii) + "_orbitals").create_dataset("values", data=dmorbs[ii])

fdmo_g = data.create_group("FullDensityMatrix_orbitals")
fdmo_g.create_dataset("values", data=fulldmorbs)
