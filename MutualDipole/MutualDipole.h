#include <hoomd/ForceCompute.h>
#include <cufft.h>
#include <hoomd/md/NeighborList.h>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>

#ifndef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

#ifndef __MUTUALDIPOLE_H__
#define __MUTUALDIPOLE_H__

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

// Declares the MutualDipole class.
class MutualDipole : public ForceCompute {

    public:
        // Constructor
        MutualDipole(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
	              	 std::shared_ptr<NeighborList> nlist,
				  	 std::vector<int> &group_tag,
				  	 std::vector<float> &conductivity,
					 std::vector<float> &radii, 
					 std::vector<unsigned int> &radii_tag, 
		      		 std::vector<float> &field,
		      		 std::vector<float> &gradient,
	              	 Scalar xi,
		      		 Scalar errortol,
		      		 std::string fileprefix,
		      		 int period,
		      		 int constantdipoleflag,
		      		 unsigned int t0);

	// Destructor
    virtual ~MutualDipole();

	// Set parameters needed for the calculations
	void SetParams();

	// Update the external field
	void UpdateField(std::vector<float> &field,
			 		 std::vector<float> &gradient);

	// Update simulation parameters
	void UpdateParameters(std::vector<int> &group_tag, 
						  std::vector<float> &conductivity, 
						  std::vector<float> &radii,
						  std::vector<unsigned int> &radii_tag,
				  		  std::vector<float> &field,
			      		  std::vector<float> &gradient,
			      		  std::string fileprefix,
			      		  int period,
			      		  int constantdipoleflag,
			      		  unsigned int t0);

	// Performs the dipole and force calculation
    virtual void computeForces(unsigned int timestep);

	// Write particle positions, dipoles, and forces to file
	void OutputData(unsigned int timestep);

    protected:

	std::shared_ptr<ParticleGroup> m_group;		// active group of particles on which to perform the calculation
	std::shared_ptr<NeighborList> m_nlist;    	// neighbor list

	int m_Ntotal;					// total number of particles
	int m_group_size;				// number of particles in the active group

	GPUArray<int> m_group_tag;
	GPUArray<int> m_group_membership_tag;		// active group membership list
	GPUArray<Scalar> m_conductivity;		// particle conductivities
	GPUArray<Scalar> m_radii;			// particle radii by type
	GPUArray<unsigned int> m_radii_tag;
	unsigned int m_radii_types;			// number of particle radii types

	Scalar3 m_field;				// applied external field
	Scalar3 m_gradient;				// applied field gradient

	GPUArray<Scalar3> m_extfield;			// external field at particle positions as computed by particle dipoles
	GPUArray<Scalar3> m_dipole;			// particle dipoles

	Scalar m_xi;               			// Ewald splitting parameter
	Scalar m_errortol;				// error tolerance
	Scalar3 m_eta;					// spectral splitting parameter
	Scalar m_rc;                           		// real space cutoff

	int m_Nx;                                	// number of grid points in x direction
	int m_Ny;                                	// number of grid points in y direction
	int m_Nz;                                	// number of grid points in z direction

	Scalar3 m_gridh;				// grid spacing
	int m_P;					// number of grid points over which to spread and contract

	GPUArray<Scalar4> m_gridk;      		// wave vector corresponding to each grid point

	GPUArray<CUFFTCOMPLEX> m_gridX;      		// x component of the grid
	GPUArray<CUFFTCOMPLEX> m_gridY;      		// y component of the grid
	GPUArray<CUFFTCOMPLEX> m_gridZ;      		// z component of the grid

	cufftHandle m_plan;                    		// used for the fast Fourier transformations performed on the GPU

	int m_Ntable;                           	// number of entries in real space tables
	Scalar m_drtable;                           	// real space table spacing

	GPUArray<Scalar4> m_fieldtable;                 // real space table for the field 
	GPUArray<Scalar4> m_forcetable;			// real space table for the force

	std::string m_fileprefix;			// output file prefix
	int m_period;					// frequency with which to write output files
	unsigned int m_t0;				// initial timestep

	int m_constantdipoleflag;			// 0 indicates mutual dipole model (particles can mutually polarize each other)
										// 1 indicates constant dipole model (particles are only polarized by external field) 

    };

// Exports the MutualDipole class to python
void export_MutualDipole(pybind11::module& m);

#endif
