#include "MutualDipole.cuh"
#include <stdio.h>
#include <cusp/linear_operator.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// command to convert floats or doubles to integers
#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

// Construct a class wrapper to use the grand potential matrix as a matrix-free method in CUSP.
class cuspPotential : public cusp::linear_operator<float,cusp::device_memory>
{
public:

    typedef cusp::linear_operator<float,cusp::device_memory> super;  // size of the linear operator

    Scalar4 *d_pos;  // pointer to particle positions
    Scalar *d_radii;
    Scalar *d_conductivity;  // pointer to particle conductivity

    int group_size;  // number of active particles
    int *d_group_membership_tag;  // pointer to particle membership and index in active group
    unsigned int *d_group_members;  // pointer to indices of particles in the active group
    unsigned int *d_tag;
    int *d_group_tag;
    const BoxDim& box;  // simulation box

    int block_size;  // number of threads to use per block

    Scalar xi;  // Ewald splitting parameter
    Scalar3 eta;  // spectral splitting parameter
    Scalar rc;  // real space cut-off radius
    const int Nx;  // number of grid nodes in x direction
    const int Ny;  // number of grid nodes in y direction
    const int Nz;  // number of grid nodes in z direction
    Scalar3 gridh;  // grid spacing
    int P;  // number of grid nodes over which to spread and contract

    Scalar4 *d_gridk;  // pointer to wave vector and scaling corresponding to each grid point
    CUFFTCOMPLEX *d_gridX;  // pointer to x component of grid
    CUFFTCOMPLEX *d_gridY;  // pointer to y component of grid
    CUFFTCOMPLEX *d_gridZ;  // pointer to z component of grid
    cufftHandle plan;  // plan for cuFFT
 
    const unsigned int *d_nlist;  // pointer to neighbor list
    const unsigned int *d_head_list;  // pointer to head list used to access entries in the neighbor list
    const unsigned int *d_n_neigh;  // pointer to number of neighbors of each particle

    int Ntable;  // number of entries in real space table
    Scalar drtable;  // real space table spacing
    Scalar4 *d_fieldtable;  // real space field table
    unsigned int ntypes;

    // constructor
    cuspPotential(Scalar4 *d_pos,
          Scalar *d_radii, 
          unsigned int *d_radii_tag, 
		  Scalar *d_conductivity,
		  int group_size,
		  int *d_group_membership_tag,
		  unsigned int *d_group_members,
          unsigned int *d_tag,
          int *d_group_tag, 
		  const BoxDim& box,
		  int block_size,
		  Scalar xi,
          Scalar3 eta,
          Scalar rc,
          const int Nx,
          const int Ny,
          const int Nz,
          Scalar3 gridh,
          int P,
		  Scalar4 *d_gridk,
          CUFFTCOMPLEX *d_gridX,
          CUFFTCOMPLEX *d_gridY,
          CUFFTCOMPLEX *d_gridZ,
		  cufftHandle plan,
          int Ntable,
          Scalar drtable,
		  Scalar4 *d_fieldtable,
          unsigned int radii_types,
          const unsigned int *d_nlist,
		  const unsigned int *d_head_list,
		  const unsigned int *d_n_neigh)
          : super(3*group_size,3*group_size), 
		  d_pos(d_pos),
          d_radii(d_radii),
          d_radii_tag(d_radii_tag),
		  d_conductivity(d_conductivity),
          group_size(group_size),
		  d_group_membership_tag(d_group_membership_tag),
		  d_group_members(d_group_members),
          d_tag(d_tag),
          d_group_tag(d_group_tag),
		  box(box),
		  block_size(block_size),
		  xi(xi),
          eta(eta),
          rc(rc),
          Nx(Nx),
          Ny(Ny),
          Nz(Nz),
          gridh(gridh),
          P(P),
		  d_gridk(d_gridk),
		  d_gridX(d_gridX),
		  d_gridY(d_gridY),
		  d_gridZ(d_gridZ),
          plan(plan),
          Ntable(Ntable),
          drtable(drtable),
		  d_fieldtable(d_fieldtable),
          radii_types(radii_types),
		  d_nlist(d_nlist),
		  d_head_list(d_head_list),
		  d_n_neigh(d_n_neigh){}


    // Perform the linear matrix multiplication operation E0 = M*S
    template <typename VectorType1,
             typename VectorType2>
    void operator()( VectorType1& S, VectorType2& E0 )
    {
        // Obtain a raw pointer to device memory for input (dipole) and output (particle external field) arrays
        float *S_ptr = (float *) thrust::raw_pointer_cast(&S[0]);
        float *E0_ptr = (float *) thrust::raw_pointer_cast(&E0[0]);

	    // Cast to a Scalar3 pointer
	    Scalar3 *S_ptr2 = (Scalar3 *) &S_ptr[0];
	    Scalar3 *E0_ptr2 = (Scalar3 *) &E0_ptr[0];

        // Compute E0 = M*S
	    ComputeField( d_pos, d_radii, d_radii_tag, d_conductivity, S_ptr2, E0_ptr2, group_size, d_group_membership_tag, d_group_members, d_tag, d_group_tag, box, block_size, xi, eta, rc, Nx, Ny, Nz, gridh, P, d_gridk, d_gridX, d_gridY, d_gridZ, plan, Ntable, drtable, d_fieldtable, radii_types, d_nlist, d_head_list, d_n_neigh);

    }
};