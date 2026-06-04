#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include <cufft.h>

#ifndef __MUTUALDIPOLE_CUH__
#define __MUTUALDIPOLE_CUH__

#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

// Kernel driver for the calculations called by MutualDipole.cc
cudaError_t gpu_ZeroForce(unsigned int Ntotal, // total number of particles
			  Scalar4 *d_force, // pointer to the particle forces 
			  unsigned int block_size); // number of threads per block

cudaError_t gpu_ComputeForce(Scalar4 *d_pos, // pointer to particle positions
				 Scalar *d_radii, // pointer to particle radii
				 unsigned int *d_radii_tag, // pointer to particle radii tag
			     Scalar *d_conductivity, // pointer to particle conductivities 
			     Scalar3 *d_dipole, // pointer to particle dipoles
			     Scalar3 *d_extfield, // pointer to external field evaluated at particle centers
			     Scalar3 field, // external field
			     Scalar3 gradient, // external field gradient
			     Scalar4 *d_force, // pointer to the particle forces
			     unsigned int Ntotal, // total number of particles
                 unsigned int group_size, // number of particles in active group
			     int *d_group_membership_tag, // pointer to particle membership and index in active group
                 unsigned int *d_group_members, // pointer to indices of particles in active group
				 int *d_group_tag, 
				 unsigned int *d_tag, 
                 const BoxDim& box, // simulation box
                 unsigned int block_size, // number of threads per block
			     Scalar xi, // Ewald splitting parameter
			     Scalar errortol, // error tolerance
			     Scalar3 eta, // spectral splitting parameter
			     Scalar rc, // real spcae cutoff radius
			     const int Nx, // number of grid nodes in the x dimension
			     const int Ny, // number of grid nodes in the y dimension
			     const int Nz, // number of grid nodes in the z dimension
			     Scalar3 gridh, // grid spacing
			     int P, // number of grid nodes over which to spread and contract
			     Scalar4 *d_gridk, // pointer to wave vector and scaling on grid
			     CUFFTCOMPLEX *d_gridX, // pointer to x component of grid
			     CUFFTCOMPLEX *d_gridY, // pointer to y component of grid
			     CUFFTCOMPLEX *d_gridZ, // pointer to z component of grid
			     cufftHandle plan, // plan for the FFTs
			     int Ntable, // number of entries in the real space table
			     Scalar drtable, // real space table spacing
			     Scalar4 *d_fieldtable, // pointer to real space field coefficient table
			     Scalar4 *d_forcetable, // pointer to real space force coefficient table
				 unsigned int radii_types,
                 const unsigned int *d_nlist, // pointer to neighbor list
                 const unsigned int *d_head_list, // pointer to head list used to access entries in the neighbor list
			     const unsigned int *d_n_neigh, // pointer to number of neighbors of each particle
			     int constantdipoleflag);  // indicates whether or not to turn off the mutual dipole functionality

// Kernel called by PotentialWrapper.cuh
cudaError_t ComputeField(Scalar4 *d_pos, // pointer to particle posisitons
			 Scalar *d_radii, 
			 unsigned int *d_radii_tag,
			 Scalar *d_conductivity, // pointer to particle conductivities
			 Scalar3 *d_dipole, // pointer to particle dipoles
			 Scalar3 *d_extfield, // pointer to external field at particle centers
			 unsigned int group_size, // number of particles in active
			 int *d_group_membership_tag, // pointer to particle membership and index in active group 
			 unsigned int *d_group_members, // pointer to indices of particles in active group
			 unsigned int *d_tag, 
			 int *d_group_tag,
			 const BoxDim& box, // simulation box
			 unsigned int block_size, // number of threads to use per block
			 Scalar xi, // Ewald splitting parameter
			 Scalar3 eta, // spectral splitting parameter
			 Scalar rc, // real space cutoff radius
			 const int Nx, // number of grid nodes in the x dimension
			 const int Ny, // number of grid nodes in the y dimension
			 const int Nz, // number of grid nodes in the z dimension
			 Scalar3 gridh, // grid spacing
			 int P, // number of grid nodes over which to spread and contract
			 Scalar4 *d_gridk, // pointer to wave vector and scaling on grid
			 CUFFTCOMPLEX *d_gridX, // pointer to x component of wave space grid
			 CUFFTCOMPLEX *d_gridY, // pointer to y component of wave space grid
			 CUFFTCOMPLEX *d_gridZ, // pointer to z component of wave space grid
			 cufftHandle plan, // plan for the FFTs
			 int Ntable, // number of entries in the real space table
			 Scalar drtable, // spacing between table entries
			 Scalar4 *d_fieldtable, // pointer to real space field table
			 unsigned int radii_types, 
			 const unsigned int *d_nlist, // pointer to neighbor list
			 const unsigned int *d_head_list, // pointer to head list used to access entries in the neighbor list
			 const unsigned int *d_n_neigh); // pointer to number of neighbors of each particle

#endif
