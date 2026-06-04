#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "MutualDipole.h"
#include "MutualDipole.cuh"
#include "PotentialWrapper.cuh"
#include <stdio.h>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <set>

#define PI 3.1415926535897932

using namespace std;

// Constructor for the MutualDipole class
MutualDipole::MutualDipole(std::shared_ptr<SystemDefinition> sysdef, // system this method will act on; must not be NULL
				std::shared_ptr<ParticleGroup> group, // group of particles in which to compute the force
			  	std::shared_ptr<NeighborList> nlist, // neighbor list
				std::vector<int> &group_tag, 
				std::vector<float> &conductivity, // particle conductivities
				std::vector<float> &radii, // particle radius corresponding to radii tag
				std::vector<unsigned int> &radii_tag,
			  	std::vector<float> &field, // imposed external field
				std::vector<float> &gradient, // imposed external field gradient
			  	Scalar xi, // Ewald splitting parameter
				Scalar errortol, // error tolerance
				std::string fileprefix,  // output file name prefix
				int period,  // output file period
				int constantdipoleflag,  // indicates whether to turn off the mutual dipole functionality
				unsigned int t0) // initial timestep
			  	: ForceCompute(sysdef),
			  	m_group(group),
			  	m_nlist(nlist),
			  	m_xi(xi),
				m_errortol(errortol),
				m_fileprefix(fileprefix),
				m_period(period),
				m_constantdipoleflag(constantdipoleflag),
				m_t0(t0)
{
    m_exec_conf->msg->notice(5) << "Constructing MutualDipole" << std::endl;

	// only one GPU is supported
	if (!m_exec_conf->isCUDAEnabled())
	{
		m_exec_conf->msg->error() << "Creating a MutualDipole when CUDA is disabled" << std::endl;
		throw std::runtime_error("Error initializing MutualDipole");
	}

	// Set the field and field gradient
	m_field = make_scalar3(field[0], field[1], field[2]);
	m_gradient = make_scalar3(gradient[0], gradient[1], gradient[2]);

	// Get the group size and total number of particles
	m_group_size = m_group->getNumMembers();
	m_Ntotal = m_pdata->getN();

	// group_tag
	GPUArray<int> n_group_tag(m_Ntotal, m_exec_conf);
	m_group_tag.swap(n_group_tag);
	ArrayHandle<int> h_group_tag(m_group_tag, access_location::host, access_mode::read);
	for (unsigned int i = 0; i < m_Ntotal; ++i ){
		h_group_tag.data[i] = group_tag[i];
	}
	
	// Extract the particle conductivities
	GPUArray<Scalar> n_conductivity(m_group_size, m_exec_conf);
	m_conductivity.swap(n_conductivity);
	ArrayHandle<Scalar> h_conductivity(m_conductivity, access_location::host, access_mode::readwrite);
	for (unsigned int i = 0; i < m_group_size; ++i ){
		h_conductivity.data[i] = conductivity[i];
	}

	// Extract the particle radii
	GPUArray<Scalar> n_radii(m_group_size, m_exec_conf);
	m_radii.swap(n_radii);
	ArrayHandle<Scalar> h_radii(m_radii, access_location::host, access_mode::read);
	for (unsigned int i = 0; i < m_group_size; ++i ){
		h_radii.data[i] = radii[i];
	}

	// radii tag
	GPUArray<unsigned int> n_radii_tag(m_group_size, m_exec_conf);
	m_radii_tag.swap(n_radii_tag);
	ArrayHandle<unsigned int> h_radii_tag(m_radii_tag, access_location::host, access_mode::read);
	for (unsigned int i = 0; i < m_group_size; ++i ){
		h_radii_tag.data[i] = radii_tag[i];
	}
}

// Destructor for the MutualDipole class
MutualDipole::~MutualDipole() {

    m_exec_conf->msg->notice(5) << "Destroying MutualDipole" << std::endl;
	cufftDestroy(m_plan);
}

// Compute and set parameters needed for the calculations.  This step is computed only once when the MutualDipole class is created or on each call to update_parameters if system parameters change. 
void MutualDipole::SetParams() {

	////// Compute parameters associated with the numerical method.

	const BoxDim& box = m_pdata->getBox(); // simulation box
	Scalar3 L = box.getL();  // box dimensions

	m_rc = sqrtf(-logf(m_errortol))/m_xi;  // real space cutoff radius
	Scalar kcut = 2.0*m_xi*sqrtf(-logf(m_errortol));  // wave space cutoff
	m_Nx = int(ceil(1.0 + L.x*kcut/PI));  // number of grid nodes in the x direction
	m_Ny = int(ceil(1.0 + L.y*kcut/PI));  // number of grid nodes in the y direction
	m_Nz = int(ceil(1.0 + L.z*kcut/PI));  // number of grid nodes in the z direction

	// Get a list of 5-smooth integers between 8 and 512^3 (can be written as (2^a)*(3^b)*(5^c); i.e. only prime factors of 2, 3, and 5)
	std::vector<int> Mlist;
	for ( int ii = 0; ii < 28; ++ii ){
		int pow2 = 1;
		for ( int i = 0; i < ii; ++i ){
			pow2 *= 2;
		}
		for ( int jj = 0; jj < 18; ++jj ){
			int pow3 = 1;
			for ( int j = 0; j < jj; ++j ){
				pow3 *= 3;
			}
			for ( int kk = 0; kk < 12; ++kk ){
				int pow5 = 1;
				for ( int k = 0; k < kk; ++k ){
					pow5 *= 5;
				}
				int Mcurr = pow2 * pow3 * pow5;
				if ( Mcurr >= 8 && Mcurr <= 512*512*512 ){
					Mlist.push_back(Mcurr);
				}
			}
		}
	}

	// Sort the list from lowest to highest
	std::sort(Mlist.begin(), Mlist.end());

	// Get the length of the list
	const int nmult = Mlist.size();	

	// Set the number of grid points to be a 5-smooth integer for most efficient FFTs
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nx <= Mlist[ii]){
			m_Nx = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Ny <= Mlist[ii]){
			m_Ny = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nz <= Mlist[ii]){
			m_Nz = Mlist[ii];
			break;
		}
	}

	// Throw an error if there are too many grid nodes
	if (m_Nx*m_Ny*m_Nz > 512*512*512){
		printf("ERROR: Total number of grid nodes is larger than 512^3.  Calculation may fail due to GPU memory limitations. Try decreasing the error tolerance, decreasing the Ewald splitting parameter, or decreasing the size of the simulation box.\n");
		printf("Nx = %i \n", m_Nx);
		printf("Ny = %i \n", m_Ny);
		printf("Nz = %i \n", m_Nz);
		printf("Total = %i \n", m_Nx*m_Ny*m_Nz);

		exit(EXIT_FAILURE);
	}

	// Additional parameters
	m_gridh = L / make_scalar3(m_Nx,m_Ny,m_Nz); // grid spacing
	m_P = ceil(-2.0*logf(m_errortol)/PI);  // number of grid nodes over which to support spreading and contracting kernels
	m_eta = m_P*m_gridh*m_gridh*m_xi*m_xi/PI; // spectral splitting parameter controlling decay of spreading and contracting kernels

	// Cannot support more nodes than the grid size
	if (m_P > m_Nx) m_P = m_Nx;
	if (m_P > m_Ny) m_P = m_Ny;		     
	if (m_P > m_Nz) m_P = m_Nz;

	// Print summary to command line output
	printf("\n");
	printf("\n");
	m_exec_conf->msg->notice(2) << "--- ????? Parameters ---" << std::endl;
	m_exec_conf->msg->notice(2) << "Active group size: " << m_group_size << std::endl;
	m_exec_conf->msg->notice(2) << "Box dimensions: " << L.x << ", " << L.y << ", " << L.z << std::endl;
	m_exec_conf->msg->notice(2) << "Ewald parameter xi: " << m_xi << std::endl;
	m_exec_conf->msg->notice(2) << "Error tolerance: " << m_errortol << std::endl;
	m_exec_conf->msg->notice(2) << "Real space cutoff: " << m_rc << std::endl;
	m_exec_conf->msg->notice(2) << "Wave space cutoff: " << kcut << std::endl;
	m_exec_conf->msg->notice(2) << "Grid nodes in each dimension: " << m_Nx << ", " << m_Ny << ", " << m_Nz << std::endl;
	m_exec_conf->msg->notice(2) << "Grid spacing: " << m_gridh.x << ", " << m_gridh.y << ", " << m_gridh.z << std::endl;
	m_exec_conf->msg->notice(2) << "Spectral parameter eta: " << m_eta.x << ", " << m_eta.y << ", " << m_eta.z << std::endl;
	m_exec_conf->msg->notice(2) << "Support P: " << m_P << std::endl;
	printf("\n");
	printf("\n");

	////// Wave space grid

	// Create plan for FFTs on the GPU
	cufftPlan3d(&m_plan, m_Nx, m_Ny, m_Nz, CUFFT_C2C);

	// Initialize array for the wave vector and scaling factor at each grid point
	GPUArray<Scalar4> n_gridk(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridk.swap(n_gridk);
	ArrayHandle<Scalar4> h_gridk(m_gridk, access_location::host, access_mode::readwrite);

	// Initialize arrays for x, y, and z, components of the wave space grid
	GPUArray<CUFFTCOMPLEX> n_gridX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridX.swap(n_gridX);
	GPUArray<CUFFTCOMPLEX> n_gridY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridY.swap(n_gridY);
	GPUArray<CUFFTCOMPLEX> n_gridZ(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridZ.swap(n_gridZ);

	// Populate array with wave space vectors and scalings
	for (int i = 0; i < m_Nx; i++) {  // loop through x dimension
		for (int j = 0; j < m_Ny; j++) {  // loop through y dimension
			for (int k = 0; k < m_Nz; k++) {  // loop through z dimension

				// Linear index into grid array
				int idx = i*m_Ny*m_Nz + j*m_Nz + k;

				// Wave vector components go from -2*PI*N/2 to 2*PI*N/2
				h_gridk.data[idx].x = ((i < (m_Nx+1)/2) ? i : i - m_Nx) * 2.0*PI/L.x;
				h_gridk.data[idx].y = ((j < (m_Ny+1)/2) ? j : j - m_Ny) * 2.0*PI/L.y;
				h_gridk.data[idx].z = ((k < (m_Nz+1)/2) ? k : k - m_Nz) * 2.0*PI/L.z;

				// Wave vector magnitude and magnitude squared
				Scalar k2 = h_gridk.data[idx].x*h_gridk.data[idx].x + h_gridk.data[idx].y*h_gridk.data[idx].y + h_gridk.data[idx].z*h_gridk.data[idx].z;
				Scalar kmag = sqrt(k2);

				// Term in the exponential of the scaling factor
				Scalar etak2 = (1.0-m_eta.x)*h_gridk.data[idx].x*h_gridk.data[idx].x + (1.0-m_eta.y)*h_gridk.data[idx].y*h_gridk.data[idx].y + (1.0-m_eta.z)*h_gridk.data[idx].z*h_gridk.data[idx].z;

				// Scaling factor used in wave space sum.  The k = 0 term is excluded.
				if (i == 0 && j == 0 && k == 0){
					h_gridk.data[idx].w = 0.0;
				}
				else{
					// Divided by total number of grid nodes due to the ifft conventions in cuFFT.  
					h_gridk.data[idx].w = 9.0*pow( (sin(kmag)/k2 - cos(kmag)/kmag) ,2)*expf(-etak2/(4.0*m_xi*m_xi))/(k2*k2) / Scalar(m_Nx*m_Ny*m_Nz);
				} // end scaling if statement
				
			} // end z dimension loop (k)
		} // end y dimension loop (j)
	} // end x dimension loop (i)

	////// Real space tables
	// Will need 2 dimensional array, (m_Ntable+1)*(m_Ntable+1)

	// Parameters for the real space table
	m_drtable = double(0.001); // table spacing
	m_Ntable = m_rc/m_drtable - 1; // number of entries in the table

	ArrayHandle<int> h_group_tag(m_group_tag, access_location::host, access_mode::read);
	ArrayHandle<Scalar> h_radii(m_radii, access_location::host, access_mode::read);
	ArrayHandle<unsigned int> h_radii_tag(m_radii_tag, access_location::host, access_mode::read);

	// Count how many unique radii_tag values exist
	std::set<unsigned int> unique_radii_tags;
	for (unsigned int i = 0; i < m_group_size; ++i)
	{
		unique_radii_tags.insert(h_radii_tag.data[i]);
	}
	m_radii_types = static_cast<unsigned int>(unique_radii_tags.size());

	const unsigned int table_width = m_Ntable + 1;
	const unsigned int table_size = table_width * m_radii_types * m_radii_types;

	// initialize the field real space table
	GPUArray<Scalar4> n_fieldtable(table_size, m_exec_conf); // GPUArray<Scalar4> n_fieldtable((m_Ntable + 1) * m_radii_types * m_radii_types, m_exec_conf);
	m_fieldtable.swap(n_fieldtable);
	ArrayHandle<Scalar4> h_fieldtable(m_fieldtable, access_location::host, access_mode::readwrite);

	// initialize the force real space table
	GPUArray<Scalar4> n_forcetable(table_size, m_exec_conf); // GPUArray<Scalar4> n_forcetable((m_Ntable + 1) * m_radii_types * m_radii_types, m_exec_conf);
	m_forcetable.swap(n_forcetable);
	ArrayHandle<Scalar4> h_forcetable(m_forcetable, access_location::host, access_mode::readwrite);

	// xi values
	double xi = m_xi;
	double xi2 = pow(xi,2);
	double xi3 = pow(m_xi,3);
	double xi4 = pow(m_xi,4);
	double xi5 = pow(m_xi,5);
	double xi6 = pow(m_xi,6);

	// Fill the real space tables
	// Need to add a doule loop for a_i and a_j
	// User need to provide radii and the types
	// Looping through radii tag
	// a_i and a_j come from the user-provided radii vector matching radii_tag

	for (unsigned int i = 0; i < m_radii_types; ++i)
	{
		for (unsigned int j = 0; j < m_radii_types; ++j)
		{
			//const double a_i = double(h_radii.data[h_radii_tag[i]]);
			//const double a_j = double(h_radii.data[h_radii_tag[j]]);
			unsigned int radii_tag_i = h_radii_tag.data[i];
			unsigned int radii_tag_j = h_radii_tag.data[j];

			const double a_i = double(h_radii.data[radii_tag_i]);
			const double a_j = double(h_radii.data[radii_tag_j]);

			const unsigned int pair_offset = (i * m_radii_types + j) * table_width;

			double ai2 = pow(a_i,2);
			double ai3 = pow(a_i,3);

			double aj2 = pow(a_j,2);
			double aj3 = pow(a_j,3);

			double aipaj = a_i + a_j;
			double aimaj = a_i - a_j;

			double aipaj2 = pow(aipaj,2);
			double aipaj3 = pow(aipaj,3);
			double aipaj4 = pow(aipaj,4);

			double aimaj2 = pow(aimaj,2);
			double aimaj3 = pow(aimaj,3);
			double aimaj4 = pow(aimaj,4);

		for (unsigned int table_i = 0; table_i <= m_Ntable; table_i++)
		{
			const unsigned int table_idx = pair_offset + table_i;
			const unsigned int table_idx_next = pair_offset + table_i + 1;

			double field_1_Irr = 0.0;
			double field_2_Irr = 0.0;
			double field_3_Irr = 0.0;
			double field_4_Irr = 0.0;
			double field_5_Irr = 0.0;
			double field_6_Irr = 0.0;
			double field_7_Irr = 0.0;
			double field_8_Irr = 0.0;

			double field_1_rr = 0.0;
			double field_2_rr = 0.0;
			double field_3_rr = 0.0;
			double field_4_rr = 0.0;
			double field_5_rr = 0.0;
			double field_6_rr = 0.0;
			double field_7_rr = 0.0;
			double field_8_rr = 0.0;

			double force_1_Irr = 0.0;
			double force_2_Irr = 0.0;
			double force_3_Irr = 0.0;
			double force_4_Irr = 0.0;
			double force_5_Irr = 0.0;
			double force_6_Irr = 0.0;
			double force_7_Irr = 0.0;
			double force_8_Irr = 0.0;

			double force_1_rr = 0.0;
			double force_2_rr = 0.0;
			double force_3_rr = 0.0;
			double force_4_rr = 0.0;
			double force_5_rr = 0.0;
			double force_6_rr = 0.0;
			double force_7_rr = 0.0;
			double force_8_rr = 0.0;

			// Particle separation corresponding to current table entry
			double dist = (i + 1) * m_drtable;		
			double dist2 = pow(dist,2);
			double dist3 = pow(dist,3);
			double dist4 = pow(dist,4);
			double dist5 = pow(dist,5);
			double dist6 = pow(dist,6);

			// // Exponentials and complimentary error functions
			// double expp = exp(-(dist+2)*(dist+2)*xi2);		
			// double expm = exp(-(dist-2)*(dist-2)*xi2);
			// double exp0 = exp(-dist2*xi2);
			// double erfp = erfc((dist+2)*xi);
			// double erfm = erfc((dist-2)*xi);
			// double erf0 = erfc(dist*xi);

			// Exponentials and complimentary error functions
			double exp_1 = exp(-pow((dist+aipaj),2)*xi2);
			double exp_2 = exp(-pow((dist-aipaj),2)*xi2);
			double exp_3 = exp(-pow((dist-aimaj),2)*xi2);
			double exp_4 = exp(-pow((dist+aimaj),2)*xi2);
			double erf_5 = erfc((dist+aipaj)*xi);
			double erf_6 = erfc((dist-aipaj)*xi);
			double erf_7 = erfc((dist-aimaj)*xi);
			double erf_8 = erfc((dist+aimaj)*xi);

			// Derivative of the exponentials and complimentary error functions
			double dexp_1 = -2.0*(aipaj+dist)*exp(-pow((aipaj+dist),2)*xi2)*xi2;
			double dexp_2 = -2.0*(-aipaj+dist)*exp(-pow((-aipaj+dist),2)*xi2)*xi2;
			double dexp_3 = -2.0*(-aimaj+dist)*exp(-pow((-aimaj+dist),2)*xi2)*xi2;
			double dexp_4 = -2.0*(aimaj+dist)*exp(-pow((aimaj+dist),2)*xi2)*xi2;
			double derf_5 = -((2.0*exp(-pow((aipaj+dist),2)*xi2)*xi)/pow(PI,0.5));
			double derf_6 = -((2.0*exp(-pow((-aipaj+dist),2)*xi2)*xi)/pow(PI,0.5));
			double derf_7 = -((2.0*exp(-pow((-aimaj+dist),2)*xi2)*xi)/pow(PI,0.5));
			double derf_8 = -((2.0*exp(-pow((aimaj+dist),2)*xi2)*xi)/pow(PI,0.5));

			// // Field table; I-rr component
			// double exppolyp = 1.0/(1024.0*pow(PI,1.5)*xi5*dist3)*(4.0*xi4*dist5 - 8.0*xi4*dist4 + 8.0*xi2*(2.0-7.0*xi2)*dist3 - 8.0*xi2*(3.0+2.0*xi2)*dist2 + (3.0-12.0*xi2+32.0*xi4)*dist + 2.0*(3.0+4.0*xi2-32.0*xi4));
			// double exppolym = 1.0/(1024.0*pow(PI,1.5)*xi5*dist3)*(4.0*xi4*dist5 + 8.0*xi4*dist4 + 8.0*xi2*(2.0-7.0*xi2)*dist3 + 8.0*xi2*(3.0+2.0*xi2)*dist2 + (3.0-12.0*xi2+32.0*xi4)*dist - 2.0*(3.0+4.0*xi2-32.0*xi4));
			// double exppoly0 = 1.0/(512.0*pow(PI,1.5)*xi5*dist2)*(-4.0*xi4*dist4 - 8.0*xi2*(2.0-9.0*xi2)*dist2 - 3.0+36.0*xi2);
			// double erfpolyp = 1.0/(2048.0*PI*xi6*dist3)*(-8.0*xi6*dist6 - 36.0*xi4*(1.0-4.0*xi2)*dist4 + 256.0*xi6*dist3 - 18.0*xi2*(1.0-8.0*xi2)*dist2 + 3.0-36.0*xi2+256.0*xi6);
			// double erfpolym = 1.0/(2048.0*PI*xi6*dist3)*(-8.0*xi6*dist6 - 36.0*xi4*(1.0-4.0*xi2)*dist4 - 256.0*xi6*dist3 - 18.0*xi2*(1.0-8.0*xi2)*dist2 + 3.0-36.0*xi2+256.0*xi6);
			// double erfpoly0 = 1.0/(1024.0*PI*xi6*dist3)*(8.0*xi6*dist6 + 36.0*xi4*(1.0-4.0*xi2)*dist4 + 18.0*xi2*(1.0-8.0*xi2)*dist2 - 3.0+36.0*xi2);

			// Field table; I-rr component (pair potential, ai, aj)
			// Need to define ai, aj
			field_1_Irr = 1.0/(1024.0*pow(PI,1.5)*xi5*dist3*ai3*aj3)*(4.0*xi4*dist5 - 4.0*aipaj*xi4*dist4 + (16.0*xi2+8.0*(-4.0*ai2+a_i*a_j-4.0*aj2)*xi4)*dist3 + (-12.0*aipaj*xi2-8.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi4)*dist2 + (3.0-12.0*(ai2-a_i*a_j+aj2)*xi2-4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4)*dist + 3.0*aipaj+4.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi2 + 4.0*aipaj3*(ai2-4.0*a_i*a_j+aj2)*xi4);
			field_2_Irr = 1.0/(1024.0*pow(PI,1.5)*xi5*dist3*ai3*aj3)*(4.0*xi4*dist5 + 4.0*aipaj*xi4*dist4 + (16.0*xi2+8.0*(-4.0*ai2+a_i*a_j-4.0*aj2)*xi4)*dist3 + (12.0*aipaj*xi2+8.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi4)*dist2 + (3.0-12.0*(ai2-a_i*a_j+aj2)*xi2-4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4)*dist - 3.0*aipaj-4.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi2 - 4.0*aipaj3*(ai2-4.0*a_i*a_j+aj2)*xi4);
			field_3_Irr = 1.0/(1024.0*pow(PI,1.5)*xi5*dist3*ai3*aj3)*(-4.0*xi4*dist5 - 4.0*aimaj*xi4*dist4 + (-16.0*xi2+8.0*(4.0*ai2+a_i*a_j+4.0*aj2)*xi4)*dist3 + (-12.0*aimaj*xi2-8.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi4)*dist2 + (-3.0+12.0*(ai2+a_i*a_j+aj2)*xi2+4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4)*dist + 3.0*aimaj+4.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi2 + 4.0*aimaj3*(ai2+4.0*a_i*a_j+aj2)*xi4);
			field_4_Irr = 1.0/(1024.0*pow(PI,1.5)*xi5*dist3*ai3*aj3)*(-4.0*xi4*dist5 + 4.0*aimaj*xi4*dist4 + (-16.0*xi2+8.0*(4.0*ai2+a_i*a_j+4.0*aj2)*xi4)*dist3 + (12.0*aimaj*xi2+8.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi4)*dist2 + (-3.0+12.0*(ai2+a_i*a_j+aj2)*xi2+4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4)*dist - 3.0*aimaj-4.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi2 - 4.0*aimaj3*(ai2+4.0*a_i*a_j+aj2)*xi4);
			field_5_Irr = 1.0/(2048.0*PI*xi6*dist3*ai3*aj3)*(-8.0*xi6*dist6 + 36.0*xi4*dist4*(-1.0+2.0*(ai2+aj2)*xi2) + 128.0*(ai3+aj3)*dist3*xi6 + dist2*(-18.0*xi2+72.0*(ai2+aj2)*xi4+72.0*pow((ai2-aj2),2)*xi6)+3.0-18.0*(ai2+aj2)*xi2 - 36.0*(ai2-aj2)*xi4 - 8.0*aipaj4*(ai2-4*a_i*a_j+aj2)*xi6);
			field_6_Irr = 1.0/(2048.0*PI*xi6*dist3*ai3*aj3)*(-8.0*xi6*dist6 + 36.0*xi4*dist4*(-1.0+2.0*(ai2+aj2)*xi2) - 128.0*(ai3+aj3)*dist3*xi6 + dist2*(-18.0*xi2+72.0*(ai2+aj2)*xi4+72.0*pow((ai2-aj2),2)*xi6)+3.0-18.0*(ai2+aj2)*xi2 - 36.0*(ai2-aj2)*xi4 - 8.0*aipaj4*(ai2-4*a_i*a_j+aj2)*xi6);
			field_7_Irr = 1.0/(2048.0*PI*xi6*dist3*ai3*aj3)*(8.0*xi6*dist6 + 36.0*xi4*dist4*(1.0-2.0*(ai2+aj2)*xi2) + 128.0*(ai3-aj3)*dist3*xi6 + dist2*(18.0*xi2-72.0*(ai2+aj2)*xi4-72.0*pow((ai2-aj2),2)*xi6)-3.0+18.0*(ai2+aj2)*xi2 + 36.0*(ai2-aj2)*xi4 + 8.0*aimaj4*(ai2+4*a_i*a_j+aj2)*xi6);
			field_8_Irr = 1.0/(2048.0*PI*xi6*dist3*ai3*aj3)*(8.0*xi6*dist6 + 36.0*xi4*dist4*(1.0-2.0*(ai2+aj2)*xi2) - 128.0*(ai3-aj3)*dist3*xi6 + dist2*(18.0*xi2-72.0*(ai2+aj2)*xi4-72.0*pow((ai2-aj2),2)*xi6)-3.0+18.0*(ai2+aj2)*xi2 + 36.0*(ai2-aj2)*xi4 + 8.0*aimaj4*(ai2+4*a_i*a_j+aj2)*xi6);

			// // Regularization for overlapping particles
			// double regpoly;
			// if (dist < 2) {
			// 	regpoly =  -1.0/(4.0*PI*dist3) + 1.0/(4.0*PI)*(1.0 - 9.0*dist/16.0 + dist3/32.0);
			// } else {
			// 	regpoly = 0.0;
			// }

			// Regularization for overlapping particles
			double regpoly = 0.0;

			if (dist < a_i + a_j && dist >= a_i - a_j && dist >= a_j - a_i) {

				regpoly = (aipaj-dist3) / (128.0*PI*ai3*aj3*dist3) * (-dist3-3.0*aipaj*dist2 + 3.0*(ai2-4.0*a_i*a_j+aj2)*dist + ai3 - 3.0*ai2*a_j - 3.0*a_i*aj2 + aj3);

			}
			else if (dist < a_j - a_i && dist > a_i - a_j) {

				regpoly = (dist3 - aj3) / (4.0 * PI * aj3 * dist3);

			}
			else if (dist < a_i - a_j && dist > a_j - a_i) {

				regpoly = (dist3 - ai3) / (4.0 * PI * ai3 * dist3);

			}
	
			// I-rr term gets the .x field
			h_fieldtable.data[table_idx].x = Scalar(field_1_Irr*exp_1 + field_2_Irr*exp_2 + field_3_Irr*exp_3 + field_4_Irr*exp_4 + field_5_Irr*erf_5 + field_6_Irr*erf_6 + field_7_Irr*erf_7 + field_8_Irr*erf_8 + regpoly);

			// Handle the r->0 separatly
			h_fieldtable.data[pair_offset].x = Scalar(1.0/(16.0*pow(PI,1.5)*ai3*aj3*xi3)*((1.0-2.0*ai2*xi2+2.0*a_i*a_j*xi2-2.0*aj2*xi2)*exp(-aipaj2*xi2) + (-1.0+2.0*ai2*xi2+2.0*a_i*a_j*xi2+2.0*aj2*xi2)*exp(-aimaj2*xi2)) + 1.0/(8.0*PI*ai3*aj3)*((ai3-aj3)*erf(aimaj*xi) - (ai3+aj3)*erf(aipaj*xi)) + min(ai3,aj3)/(4*PI*ai3*aj3));

			// // Field table: rr component
			// exppolyp = 1.0/(512.0*pow(PI,1.5)*xi5*dist3)*(8.0*xi4*dist5 - 16.0*xi4*dist4 + 2.0*xi2*(7.0-20.0*xi2)*dist3 - 4.0*xi2*(3.0-4.0*xi2)*dist2 - (3.0-12.0*xi2+32.0*xi4)*dist - 2.0*(3.0+4.0*xi2-32.0*xi4));
			// exppolym = 1.0/(512.0*pow(PI,1.5)*xi5*dist3)*(8.0*xi4*dist5 + 16.0*xi4*dist4 + 2.0*xi2*(7.0-20.0*xi2)*dist3 + 4.0*xi2*(3.0-4.0*xi2)*dist2 - (3.0-12.0*xi2+32.0*xi4)*dist + 2.0*(3.0+4.0*xi2-32.0*xi4));
			// exppoly0 = 1.0/(256.0*pow(PI,1.5)*xi5*dist2)*(-8.0*xi4*dist4 - 2.0*xi2*(7.0-36.0*xi2)*dist2 + 3.0-36.0*xi2);
			// erfpolyp = 1.0/(1024.0*PI*xi6*dist3)*(-16.0*xi6*dist6 - 36.0*xi4*(1.0-4.0*xi2)*dist4 + 128.0*xi6*dist3 - 3.0+36.0*xi2-256.0*xi6);
			// erfpolym = 1.0/(1024.0*PI*xi6*dist3)*(-16.0*xi6*dist6 - 36.0*xi4*(1.0-4.0*xi2)*dist4 - 128.0*xi6*dist3 - 3.0+36.0*xi2-256.0*xi6);
			// erfpoly0 = 1.0/(512.0*PI*xi6*dist3)*(16.0*xi6*dist6 + 36.0*xi4*(1.0-4.0*xi2)*dist4 + 3.0-36.0*xi2);

			// Field table: rr component
			field_1_rr = 1.0/(512.0*pow(PI,1.5)*xi5*dist3*ai3*aj3)*(8.0*xi4*dist5 - 8.0*xi4*dist4*aipaj + (14.0*xi2-4.0*(7.0*ai2-4.0*a_i*a_j+7.0*aj2)*xi4)*dist3 + (-6.0*aipaj*xi2-4.0*(ai3-3.0*ai2*a_j-3*a_i*aj2+aj3)*xi4)*dist2 + (-3.0+12.0*(ai2-a_i*a_j+aj2)*xi2+4.0*aipaj2*(ai2-4*a_i*a_j+aj2)*xi4)*dist - 3.0*aipaj-4.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi2 - 4.0*aipaj3*(ai2-4.0*a_i*a_j+aj2)*xi4);
			field_2_rr = 1.0/(512.0*pow(PI,1.5)*xi5*dist3*ai3*aj3)*(8.0*xi4*dist5 + 8.0*xi4*dist4*aipaj + (14.0*xi2-4.0*(7.0*ai2-4.0*a_i*a_j+7.0*aj2)*xi4)*dist3 + (6.0*aipaj*xi2+4.0*(ai3-3.0*ai2*a_j-3*a_i*aj2+aj3)*xi4)*dist2 + (-3.0+12.0*(ai2-a_i*a_j+aj2)*xi2+4.0*aipaj2*(ai2-4*a_i*a_j+aj2)*xi4)*dist + 3.0*aipaj+4.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi2 + 4.0*aipaj3*(ai2-4.0*a_i*a_j+aj2)*xi4);
			field_3_rr = 1.0/(512.0*pow(PI,1.5)*xi5*dist3*ai3*aj3)*(-8.0*xi4*dist5 - 8.0*xi4*dist4*aimaj + (-14.0*xi2+4.0*(7.0*ai2+4.0*a_i*a_j+7.0*aj2)*xi4)*dist3 + (-6.0*aimaj*xi2-4.0*(ai3+3.0*ai2*a_j-3*a_i*aj2-aj3)*xi4)*dist2 + (3.0-12.0*(ai2+a_i*a_j+aj2)*xi2-4.0*aimaj2*(ai2+4*a_i*a_j+aj2)*xi4)*dist - 3.0*aimaj-4.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi2 - 4.0*aimaj3*(ai2+4.0*a_i*a_j+aj2)*xi4);
			field_4_rr = 1.0/(512.0*pow(PI,1.5)*xi5*dist3*ai3*aj3)*(-8.0*xi4*dist5 + 8.0*xi4*dist4*aimaj + (-14.0*xi2+4.0*(7.0*ai2+4.0*a_i*a_j+7.0*aj2)*xi4)*dist3 + (6.0*aimaj*xi2+4.0*(ai3+3.0*ai2*a_j-3*a_i*aj2-aj3)*xi4)*dist2 + (3.0-12.0*(ai2+a_i*a_j+aj2)*xi2+4.0*aimaj2*(ai2+4*a_i*a_j+aj2)*xi4)*dist + 3.0*aimaj+4.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi2 + 4.0*aimaj3*(ai2+4.0*a_i*a_j+aj2)*xi4);
			field_5_rr = 1.0/(1024.0*PI*xi6*dist3*ai3*aj3)*(-16.0*xi6*dist6 + 36.0*xi4*(-1.0+2.0*(ai2+aj2)*xi2)*dist4 + 64.0*xi6*dist3*(ai3+aj3) - 3.0+18.0*xi2*(ai2+aj2) + 36.0*pow((ai2-aj2),2)*xi4 + 8*aipaj4*(ai2-4*a_i*a_j+aj2)*xi6);
			field_6_rr = 1.0/(1024.0*PI*xi6*dist3*ai3*aj3)*(-16.0*xi6*dist6 + 36.0*xi4*(-1.0+2.0*(ai2+aj2)*xi2)*dist4 - 64.0*xi6*dist3*(ai3+aj3) - 3.0+18.0*xi2*(ai2+aj2) + 36.0*pow((ai2-aj2),2)*xi4 + 8*aipaj4*(ai2-4*a_i*a_j+aj2)*xi6);
			field_7_rr = 1.0/(1024.0*PI*xi6*dist3*ai3*aj3)*(16.0*xi6*dist6 + 36.0*xi4*(1.0-2.0*(ai2+aj2)*xi2)*dist4 + 64.0*xi6*dist3*(ai3-aj3) + 3.0-18.0*xi2*(ai2+aj2) - 36.0*pow((ai2-aj2),2)*xi4 - 8*aimaj4*(ai2+4*a_i*a_j+aj2)*xi6);
			field_8_rr = 1.0/(1024.0*PI*xi6*dist3*ai3*aj3)*(16.0*xi6*dist6 + 36.0*xi4*(1.0-2.0*(ai2+aj2)*xi2)*dist4 + 64.0*xi6*dist3*(-ai3+aj3) + 3.0-18.0*xi2*(ai2+aj2) - 36.0*pow((ai2-aj2),2)*xi4 - 8*aimaj4*(ai2+4*a_i*a_j+aj2)*xi6);

			// Regularization for overlapping particles
			// if (dist < 2) {
			// 	regpoly =  1.0/(2.0*PI*dist3) + 1.0/(4.0*PI)*(1.0 - 9.0*dist/8.0 + dist3/8.0);
			// } else {
			// 	regpoly = 0.0;
			// }

			// Regularization for overlapping particles (MATLAB version #2)
			regpoly = 0.0;

			// Case 1: (r < ai+aj) & (r >= ai-aj) & (r >= aj-ai)  <=> r < ai+aj and r >= |ai-aj|
			if (dist < a_i + a_j && dist >= a_i - a_j && dist >= a_j - a_i) {
				regpoly = -1.0 / (64.0*PI*ai3*aj3*dist3) * (aipaj4 * (ai2-4.0*a_i*a_j+aj2) - 8.0*(ai3+aj3)*dist3+9.0*(ai2+aj2)*dist4-2.0*dist6);
			}
			// Case 2: (r < aj-ai) & (r > ai-aj)
			else if (dist < a_j - a_i && dist > a_i - a_j) {
				regpoly = (dist3+2.0*aj3) / (4.0*PI*aj3*dist3);
			}
			// Case 3: (r < ai-aj) & (r > aj-ai)
			else if (dist < a_i - a_j && dist > a_j - a_i) {
				regpoly = (dist3+2.0*ai3) / (4.0*PI*ai3*dist3);
			}

			// rr term gets the .y field (combine)
			h_fieldtable.data[table_idx].y = Scalar(field_1_rr*exp_1 + field_2_rr*exp_2 + field_3_rr*exp_3 + field_4_rr*exp_4 + field_5_rr*erf_5 + field_6_rr*erf_6 + field_7_rr*erf_7 + field_8_rr*erf_8 + regpoly);

			// // Force table; -( (Si*Sj)r + (Sj*r)Si + (Si*r)Sj - 2(Si*r)(Sj*r)r ) component 
			// exppolyp = 3.0/(1024.0*pow(PI,1.5)*xi5*dist4)*(4.0*xi4*dist5 - 8.0*xi4*dist4 + 4.0*xi2*(1.0-2.0*xi2)*dist3 + 16.0*xi4*dist2 - (3.0-12.0*xi2+32.0*xi4)*dist - 2.0*(3.0+4.0*xi2-32.0*xi4));
			// exppolym = 3.0/(1024.0*pow(PI,1.5)*xi5*dist4)*(4.0*xi4*dist5 + 8.0*xi4*dist4 + 4.0*xi2*(1.0-2.0*xi2)*dist3 - 16.0*xi4*dist2 - (3.0-12.0*xi2+32.0*xi4)*dist + 2.0*(3.0+4.0*xi2-32.0*xi4));
			// exppoly0 = 3.0/(512.0*pow(PI,1.5)*xi5*dist3)*(-4.0*xi4*dist4 - 4.0*xi2*(1.0-6.0*xi2)*dist2 + 3.0-36.0*xi2);
			// erfpolyp = 3.0/(2048.0*PI*xi6*dist4)*(-8.0*xi6*dist6 - 12.0*xi4*(1.0-4.0*xi2)*dist4 + 6.0*xi2*(1.0-8.0*xi2)*dist2 - 3.0+36.0*xi2-256.0*xi6);
			// erfpolym = erfpolyp;
			// erfpoly0 = 3.0/(1024.0*PI*xi6*dist4)*(8.0*xi6*dist6 + 12.0*xi4*(1.0-4.0*xi2)*dist4 - 6.0*xi2*(1.0-8.0*xi2)*dist2 + 3.0-36.0*xi2);

			// Force table; -( (Si*Sj)r + (Sj*r)Si + (Si*r)Sj - 2(Si*r)(Sj*r)r ) component 
			force_1_Irr = 1.0/(1024.0*ai3*aj3*dist3*pow(PI,1.5)*xi5)*(3.0-12.0*(ai2-a_i*a_j+aj2)*xi2 - 4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4 - 16.0*aipaj*dist3*xi4 + 20.0*dist4*xi4 + 3*dist2*(16.0*xi2+8.0*(-4.0*ai2+a_i*a_j-4.0*aj2)*xi4) + 2*dist*(-12.0*aipaj*xi2 - 8.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi4)) - 3.0/(1024.0*ai3*aj3*dist4*pow(PI,1.5)*xi5)*(3.0*aipaj + 4.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi2 + 4.0*aipaj3*(ai2-4.0*a_i*a_j+aj2)*xi4 - 4.0*aipaj*dist4*xi4 + 4.0*dist5*xi4 + dist3*(16.0*xi2+8.0*(-4.0*ai2+a_i*a_j-4.0*aj2)*xi4) + dist*(3.0-12.0*(ai2-a_i*a_j+aj2)*xi2 - 4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4) + dist2*(-12.0*aipaj*xi2 - 8.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi4));
			force_2_Irr = 1.0/(1024.0*ai3*aj3*dist3*pow(PI,1.5)*xi5)*(3.0-12.0*(ai2-a_i*a_j+aj2)*xi2 - 4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4 + 16.0*aipaj*dist3*xi4 + 20.0*dist4*xi4 + 3*dist2*(16.0*xi2+8.0*(-4.0*ai2+a_i*a_j-4.0*aj2)*xi4) + 2*dist*(12.0*aipaj*xi2 + 8.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi4)) - 3.0/(1024.0*ai3*aj3*dist4*pow(PI,1.5)*xi5)*(-3.0*aipaj - 4.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi2 - 4.0*aipaj3*(ai2-4.0*a_i*a_j+aj2)*xi4 + 4.0*aipaj*dist4*xi4 + 4.0*dist5*xi4 + dist3*(16.0*xi2+8.0*(-4.0*ai2+a_i*a_j-4.0*aj2)*xi4) + dist*(3.0-12.0*(ai2-a_i*a_j+aj2)*xi2 - 4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4) + dist2*(12.0*aipaj*xi2 + 8.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi4));
			force_3_Irr = 1.0/(1024.0*ai3*aj3*dist3*pow(PI,1.5)*xi5)*(-3.0+12.0*(ai2+a_i*a_j+aj2)*xi2 + 4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4 - 16.0*aimaj*dist3*xi4 - 20.0*dist4*xi4 + 3*dist2*(-16.0*xi2+8.0*(4.0*ai2+a_i*a_j+4.0*aj2)*xi4) + 2*dist*(-12.0*aimaj*xi2 - 8.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi4)) - 3.0/(1024.0*ai3*aj3*dist4*pow(PI,1.5)*xi5)*(3.0*aimaj + 4.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi2 + 4.0*aimaj3*(ai2+4.0*a_i*a_j+aj2)*xi4 - 4.0*aimaj*dist4*xi4 - 4.0*dist5*xi4 + dist3*(-16.0*xi2+8.0*(4.0*ai2+a_i*a_j+4.0*aj2)*xi4) + dist*(-3.0+12.0*(ai2+a_i*a_j+aj2)*xi2 + 4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4) + dist2*(-12.0*aimaj*xi2 - 8.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi4));
			force_4_Irr = 1.0/(1024.0*ai3*aj3*dist3*pow(PI,1.5)*xi5)*(-3.0+12.0*(ai2+a_i*a_j+aj2)*xi2 + 4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4 + 16.0*aipaj*dist3*xi4 - 20.0*dist4*xi4 + 3*dist2*(-16.0*xi2+8.0*(4.0*ai2+a_i*a_j+4.0*aj2)*xi4) + 2*dist*(12.0*aimaj*xi2 + 8.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi4)) - 3.0/(1024.0*ai3*aj3*dist4*pow(PI,1.5)*xi5)*(-3.0*aimaj - 4.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi2 - 4.0*aimaj3*(ai2+4.0*a_i*a_j+aj2)*xi4 + 4.0*aimaj*dist4*xi4 - 4.0*dist5*xi4 + dist3*(-16.0*xi2+8.0*(4.0*ai2+a_i*a_j+4.0*aj2)*xi4) + dist*(-3.0+12.0*(ai2+a_i*a_j+aj2)*xi2 + 4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4) + dist2*(12.0*aimaj*xi2 + 8.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi4));
			force_5_Irr = 1.0/(2048.0*(384.0*(ai3+aj3)*dist2*xi6-48.0*dist5*xi6+144.0*dist3*xi4*(-1.0+2.0*(ai2+aj2)*xi2) + 2.0*dist*(-18.0*xi2+72.0*(ai2+aj2)*xi4 + 72.0*pow((ai2-aj2),2)*xi6))) / (ai3*aj3*dist3*PI*xi6) - 3.0/(2048.0*ai3*aj3*dist4*PI*xi6)*(3.0-18.0*(ai2+aj2)*xi2-36.0*(ai2-aj2)*xi4 - 8.0*aipaj4*(ai2-4.0*a_i*a_j+aj2)*xi6 + 128.0*(ai3+aj3)*dist3*xi6-8.0*dist6*xi6 + 36.0*dist4*xi4*(-1.0+2.0*(ai2+aj2)*xi2) + dist2*(-18.0*xi2+72.0*(ai2+aj2)*xi4 + 72.0*pow((ai2-aj2),2)*xi6));
			force_6_Irr = 1.0/(2048.0*(-384.0*(ai3+aj3)*dist2*xi6-48.0*dist5*xi6+144.0*dist3*xi4*(-1.0+2.0*(ai2+aj2)*xi2) + 2.0*dist*(-18.0*xi2+72.0*(ai2+aj2)*xi4 + 72.0*pow((ai2-aj2),2)*xi6))) / (ai3*aj3*dist3*PI*xi6) - 3.0/(2048.0*ai3*aj3*dist4*PI*xi6)*(3.0-18.0*(ai2+aj2)*xi2-36.0*(ai2-aj2)*xi4 - 8.0*aipaj4*(ai2-4.0*a_i*a_j+aj2)*xi6 - 128.0*(ai3+aj3)*dist3*xi6-8.0*dist6*xi6 + 36.0*dist4*xi4*(-1.0+2.0*(ai2+aj2)*xi2) + dist2*(-18.0*xi2+72.0*(ai2+aj2)*xi4 + 72.0*pow((ai2-aj2),2)*xi6));
			force_7_Irr = 1.0/(2048.0*(384.0*(ai3-aj3)*dist2*xi6+48.0*dist5*xi6+144.0*dist3*xi4*(1.0-2.0*(ai2+aj2)*xi2) + 2.0*dist*(18.0*xi2-72.0*(ai2+aj2)*xi4 - 72.0*pow((ai2-aj2),2)*xi6))) / (ai3*aj3*dist3*PI*xi6) - 3.0/(2048.0*ai3*aj3*dist4*PI*xi6)*(-3.0+18.0*(ai2+aj2)*xi2+36.0*(ai2-aj2)*xi4 + 8.0*aimaj4*(ai2+4.0*a_i*a_j+aj2)*xi6 + 128.0*(ai3-aj3)*dist3*xi6+8.0*dist6*xi6 + 36.0*dist4*xi4*(1.0-2.0*(ai2+aj2)*xi2) + dist2*(18.0*xi2-72.0*(ai2+aj2)*xi4 - 72.0*pow((ai2-aj2),2)*xi6));
			force_8_Irr = 1.0/(2048.0*(-384.0*(ai3-aj3)*dist2*xi6+48.0*dist5*xi6+144.0*dist3*xi4*(1.0-2.0*(ai2+aj2)*xi2) + 2.0*dist*(18.0*xi2-72.0*(ai2+aj2)*xi4 - 72.0*pow((ai2-aj2),2)*xi6))) / (ai3*aj3*dist3*PI*xi6) - 3.0/(2048.0*ai3*aj3*dist4*PI*xi6)*(-3.0+18.0*(ai2+aj2)*xi2+36.0*(ai2-aj2)*xi4 + 8.0*aimaj4*(ai2+4.0*a_i*a_j+aj2)*xi6 - 128.0*(ai3-aj3)*dist3*xi6+8.0*dist6*xi6 + 36.0*dist4*xi4*(1.0-2.0*(ai2+aj2)*xi2) + dist2*(18.0*xi2-72.0*(ai2+aj2)*xi4 - 72.0*pow((ai2-aj2),2)*xi6));

			// // Regularization for overlapping particles
			// if (dist < 2) {
			// 	regpoly =  3.0/(4.0*PI*dist4) - 3.0/(64.0*PI)*(3.0 - dist2/2.0);
			// } else {
			// 	regpoly = 0.0;
			// }

			// Regularization for overlapping particles
			regpoly = 0.0;

			if (dist < a_i + a_j && dist >= a_i - a_j && dist >= a_j - a_i) {

				regpoly = 1.0/(128.0*ai3*aj3*dist3*PI)*(3.0*(ai2-4*a_i*a_j+aj2)-6.0*aipaj*dist-3.0*dist2)*(aipaj-dist3) - 3.0*(ai3-3.0*ai2*a_j-3.0*a_i*aj2+aj3 + 3.0*(ai2-4.0*a_i*a_j+aj2)*dist - 3.0*aipaj*dist2 - dist3)/(128.0*ai3*aj3*dist*PI) - 3.0*(aipaj-dist3)*(ai3-3.0*ai2*a_j-3.0*a_i*aj2+aj3 + 3.0*(ai2-4.0*a_i*a_j+aj2)*dist - 3.0*aipaj*dist2 - dist3)/(128.0*ai3*aj3*dist4*PI);

			}
			else if (dist < a_j - a_i && dist > a_i - a_j) {

				regpoly = 3.0/(4.0*aj3*dist*PI) - 3.0/(4.0*aj3*dist4*PI)*(-aj3+dist3);

			}
			else if (dist < a_i - a_j && dist > a_j - a_i) {

				regpoly = 3.0/(4.0*ai3*dist*PI) - 3.0/(4.0*ai3*dist4*PI)*(-ai3+dist3);

			}

			// This term gets the .x field
			h_forcetable.data[table_idx].x = Scalar(field_1_Irr*dexp_1 + force_1_Irr*exp_1 + field_2_Irr*dexp_2 + force_2_Irr*exp_2 + field_3_Irr*dexp_3 + force_3_Irr*exp_3 + field_4_Irr*dexp_4 + force_4_Irr*exp_4 + field_5_Irr*derf_5 + force_5_Irr*erf_5 + field_6_Irr*derf_6 + force_6_Irr*erf_6 + field_7_Irr*derf_7 + force_7_Irr*erf_7 + field_8_Irr*derf_8 + force_8_Irr*erf_8 + regpoly);

			// // Force table; -(Si*r)(Sj*r)r component 
			// exppolyp = 9.0/(1024.0*pow(PI,1.5)*xi5*dist4)*(4.0*xi4*dist5 - 8.0*xi4*dist4 + 8.0*xi4*dist3 + 8.0*xi2*(1.0-2.0*xi2)*dist2 + (3.0-12.0*xi2+32.0*xi4)*dist + 2.0*(3.0+4.0*xi2-32.0*xi4));
			// exppolym = 9.0/(1024.0*pow(PI,1.5)*xi5*dist4)*(4.0*xi4*dist5 + 8.0*xi4*dist4 + 8.0*xi4*dist3 - 8.0*xi2*(1.0-2.0*xi2)*dist2 + (3.0-12.0*xi2+32.0*xi4)*dist - 2.0*(3.0+4.0*xi2-32.0*xi4));
			// exppoly0 = 9.0/(512.0*pow(PI,1.5)*xi5*dist3)*(-4.0*xi4*dist4 + 8.0*xi4*dist2 - 3.0+36.0*xi2);
			// erfpolyp = 9.0/(2048.0*PI*xi6*dist4)*(-8.0*xi6*dist6 - 4.0*xi4*(1.0-4.0*xi2)*dist4 - 2.0*xi2*(1.0-8.0*xi2)*dist2 + 3.0-36.0*xi2+256.0*xi6);
			// erfpolym = erfpolyp;
			// erfpoly0 = 9.0/(1024.0*PI*xi6*dist4)*(8.0*xi6*dist6 + 4.0*xi4*(1.0-4.0*xi2)*dist4 + 2.0*xi2*(1.0-8.0*xi2)*dist2 - 3.0+36.0*xi2);

			// Force table; -(Si*r)(Sj*r)r component 
			force_1_rr = 1.0/(512.0*ai3*aj3*dist3*pow(PI,1.5)*xi5)*(-3.0+12.0*(ai2-a_i*a_j+aj2)*xi2 + 4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4 - 32.0*aipaj*dist3*xi4 + 40.0*dist4*xi4 + 3*dist2*(14.0*xi2-4.0*(7.0*ai2-4.0*a_i*a_j+7.0*aj2)*xi4) + 2*dist*(-6.0*aipaj*xi2 - 4.0*(ai3-3.0*ai2*a_j-3.0*a_i*aj2+aj3)*xi4)) - 3.0/(512.0*ai3*aj3*dist4*pow(PI,1.5)*xi5)*(-3.0*aipaj - 4.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi2 - 4.0*aipaj3*(ai2-4.0*a_i*a_j+aj2)*xi4 - 8.0*aipaj*dist4*xi4 + 8.0*dist5*xi4 + dist*(-3.0+12.0*(ai2-a_i*a_j+aj2)*xi2 + 4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4) + dist3*(14.0*xi2-4.0*(7.0*ai2-4.0*a_i*a_j+7.0*aj2)*xi4) + dist2*(-6.0*aipaj*xi2 - 4.0*(ai3-3.0*ai2*a_j-3.0*a_i*aj2+aj3)*xi4));	
			force_2_rr = 1.0/(512.0*ai3*aj3*dist3*pow(PI,1.5)*xi5)*(-3.0+12.0*(ai2-a_i*a_j+aj2)*xi2 + 4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4 + 32.0*aipaj*dist3*xi4 + 40.0*dist4*xi4 + 3*dist2*(14.0*xi2-4.0*(7.0*ai2-4.0*a_i*a_j+7.0*aj2)*xi4) + 2*dist*(6.0*aipaj*xi2 + 4.0*(ai3-3.0*ai2*a_j-3.0*a_i*aj2+aj3)*xi4)) - 3.0/(512.0*ai3*aj3*dist4*pow(PI,1.5)*xi5)*(3.0*aipaj + 4.0*(4.0*ai3-3.0*ai2*a_j-3.0*a_i*aj2+4.0*aj3)*xi2 + 4.0*aipaj3*(ai2-4.0*a_i*a_j+aj2)*xi4 + 8.0*aipaj*dist4*xi4 + 8.0*dist5*xi4 + dist*(-3.0+12.0*(ai2-a_i*a_j+aj2)*xi2 + 4.0*aipaj2*(ai2-4.0*a_i*a_j+aj2)*xi4) + dist3*(14.0*xi2-4.0*(7.0*ai2-4.0*a_i*a_j+7.0*aj2)*xi4) + dist2*(6.0*aipaj*xi2 + 4.0*(ai3-3.0*ai2*a_j-3.0*a_i*aj2+aj3)*xi4));
			force_3_rr = 1.0/(512.0*ai3*aj3*dist3*pow(PI,1.5)*xi5)*(3.0-12.0*(ai2+a_i*a_j+aj2)*xi2 - 4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4 - 32.0*aimaj*dist3*xi4 - 40.0*dist4*xi4 + 3*dist2*(-14.0*xi2+4.0*(7.0*ai2+4.0*a_i*a_j+7.0*aj2)*xi4) + 2*dist*(-6.0*aimaj*xi2 - 4.0*(ai3+3.0*ai2*a_j-3.0*a_i*aj2-aj3)*xi4)) - 3.0/(512.0*ai3*aj3*dist4*pow(PI,1.5)*xi5)*(-3.0*aimaj - 4.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi2 - 4.0*aimaj3*(ai2+4.0*a_i*a_j+aj2)*xi4 - 8.0*aimaj*dist4*xi4 - 8.0*dist5*xi4 + dist*(3.0-12.0*(ai2+a_i*a_j+aj2)*xi2 - 4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4) + dist3*(-14.0*xi2+4.0*(7.0*ai2+4.0*a_i*a_j+7.0*aj2)*xi4) + dist2*(-6.0*aimaj*xi2 - 4.0*(ai3+3.0*ai2*a_j-3.0*a_i*aj2-aj3)*xi4));
			force_4_rr = 1.0/(512.0*ai3*aj3*dist3*pow(PI,1.5)*xi5)*(3.0-12.0*(ai2+a_i*a_j+aj2)*xi2 + 4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4 + 32.0*aimaj*dist3*xi4 - 40.0*dist4*xi4 + 3*dist2*(-14.0*xi2+4.0*(7.0*ai2+4.0*a_i*a_j+7.0*aj2)*xi4) + 2*dist*(6.0*aimaj*xi2 + 4.0*(ai3+3.0*ai2*a_j-3.0*a_i*aj2-aj3)*xi4)) - 3.0/(512.0*ai3*aj3*dist4*pow(PI,1.5)*xi5)*(3.0*aimaj + 4.0*(4.0*ai3+3.0*ai2*a_j-3.0*a_i*aj2-4.0*aj3)*xi2 + 4.0*aimaj3*(ai2+4.0*a_i*a_j+aj2)*xi4 + 8.0*aimaj*dist4*xi4 - 8.0*dist5*xi4 + dist*(3.0-12.0*(ai2+a_i*a_j+aj2)*xi2 + 4.0*aimaj2*(ai2+4.0*a_i*a_j+aj2)*xi4) + dist3*(-14.0*xi2+4.0*(7.0*ai2+4.0*a_i*a_j+7.0*aj2)*xi4) + dist2*(6.0*aimaj*xi2 + 4.0*(ai3+3.0*ai2*a_j-3.0*a_i*aj2-aj3)*xi4));
			force_5_rr = 1.0/(1024.0*ai3*aj3*dist3*PI*xi6)*(192.0*(ai3+aj3)*dist2*xi6 - 96.0*dist5*xi6 +144.0*dist3*xi4*(-1.0+2.0*(ai2+aj2))) - 3.0/(1024.0*ai3*aj3*dist4*PI*xi6)*(-3.0+18.0*(ai2+aj2)*xi2+36.0*pow((ai2-aj2),2)*xi4 + 8.0*aipaj4*(ai2-4*a_i*a_j+aj2)*xi6 + 64.0*(ai3+aj3)*dist3*xi6-16.0*dist6*xi6 + 36.0*dist4*xi4*(-1.0+2.0*(ai2+aj2)*xi2));
			force_6_rr = 1.0/(1024.0*ai3*aj3*dist3*PI*xi6)*(-192.0*(ai3+aj3)*dist2*xi6 - 96.0*dist5*xi6 +144.0*dist3*xi4*(-1.0+2.0*(ai2+aj2))) - 3.0/(1024.0*ai3*aj3*dist4*PI*xi6)*(-3.0+18.0*(ai2+aj2)*xi2+36.0*pow((ai2-aj2),2)*xi4 + 8.0*aipaj4*(ai2-4*a_i*a_j+aj2)*xi6 - 64.0*(ai3+aj3)*dist3*xi6-16.0*dist6*xi6 + 36.0*dist4*xi4*(-1.0+2.0*(ai2+aj2)*xi2));
			force_7_rr = 1.0/(1024.0*ai3*aj3*dist3*PI*xi6)*(192.0*(ai3-aj3)*dist2*xi6 + 96.0*dist5*xi6 +144.0*dist3*xi4*(1.0-2.0*(ai2+aj2))) - 3.0/(1024.0*ai3*aj3*dist4*PI*xi6)*(3.0-18.0*(ai2+aj2)*xi2-36.0*pow((ai2-aj2),2)*xi4 - 8.0*aimaj4*(ai2+4*a_i*a_j+aj2)*xi6 + 64.0*(ai3-aj3)*dist3*xi6+16.0*dist6*xi6 + 36.0*dist4*xi4*(1.0-2.0*(ai2+aj2)*xi2));
			force_8_rr = 1.0/(1024.0*ai3*aj3*dist3*PI*xi6)*(192.0*(-ai3+aj3)*dist2*xi6 + 96.0*dist5*xi6 +144.0*dist3*xi4*(1.0-2.0*(ai2+aj2))) - 3.0/(1024.0*ai3*aj3*dist4*PI*xi6)*(3.0-18.0*(ai2+aj2)*xi2-36.0*pow((ai2-aj2),2)*xi4 - 8.0*aimaj4*(ai2+4*a_i*a_j+aj2)*xi6 + 64.0*(-ai3+aj3)*dist3*xi6+16.0*dist6*xi6 + 36.0*dist4*xi4*(1.0-2.0*(ai2+aj2)*xi2));

			// // Regularization for overlapping particles
			// if (dist < 2) {
			// 	regpoly =  -9.0/(4.0*PI*dist4) - 9.0/(64.0*PI)*(1.0 - dist2/2.0);
			// } else {
			// 	regpoly = 0.0;
			// }

			// Case 1: (r < ai+aj) & (r >= ai-aj) & (r >= aj-ai)  <=> r < ai+aj and r >= |ai-aj|
			if (dist < a_i + a_j && dist >= a_i - a_j && dist >= a_j - a_i) {
				regpoly = -1.0/(64.0*ai3*aj3*dist3*PI)*(-24.0*(ai3+aj3)*dist2+36.0*(ai2+aj2)*dist3-12.0*dist5) + 3.0/(64.0*ai3*aj3*dist4*PI)*(aipaj4*(ai2-4.0*a_i*a_j+aj2) - 8.0*(ai3+aj3)*dist3 + 9.0*(ai2+aj2)*dist4 - 2.0*dist6);
			}
			// Case 2: (r < aj-ai) & (r > ai-aj)
			else if (dist < a_j - a_i && dist > a_i - a_j) {
				regpoly = 3.0/(4.0*aj3*dist*PI) - 3.0/(4.0*aj3*dist4*PI)*(2.0*aj3+dist3);
			}
			// Case 3: (r < ai-aj) & (r > aj-ai)
			else if (dist < a_i - a_j && dist > a_j - a_i) {
				regpoly = 3.0/(4.0*ai3*dist*PI) - 3.0/(4.0*ai3*dist4*PI)*(2.0*ai3+dist3);
			}

			// -(mi*r)(mj*r)r term gets the .y field
			h_forcetable.data[table_idx].y = Scalar(field_1_rr*dexp_1 + force_1_rr*exp_1 + field_2_rr*dexp_2 + force_2_rr*exp_2 + field_3_rr*dexp_3 + force_3_rr*exp_3 + field_4_rr*dexp_4 + force_4_rr*exp_4 + field_5_rr*derf_5 + force_5_rr*erf_5 + field_6_rr*derf_6 + force_6_rr*erf_6 + field_7_rr*derf_7 + force_7_rr*erf_7 + field_8_rr*derf_8 + force_8_rr*erf_8 + regpoly);
		}

		for (unsigned int table_i = 0; table_i < m_Ntable; table_i++)
		{
			const unsigned int table_idx = pair_offset + i;
			const unsigned int table_idx_next = pair_offset + i + 1;
			h_fieldtable.data[table_idx].z = h_fieldtable.data[table_idx_next].x;
			h_fieldtable.data[table_idx].w = h_fieldtable.data[table_idx_next].y;
			h_forcetable.data[table_idx].z = h_forcetable.data[table_idx_next].x;
			h_forcetable.data[table_idx].w = h_forcetable.data[table_idx_next].y;
		}
	}
}

	////// Initializations for needed arrays

	// Group membership list
	GPUArray<int> n_group_membership_tag(m_Ntotal, m_exec_conf);
	m_group_membership_tag.swap(n_group_membership_tag);

	// Particle dipoles
	GPUArray<Scalar3> n_dipole(m_group_size, m_exec_conf);
	m_dipole.swap(n_dipole);
	ArrayHandle<Scalar3> h_dipole(m_dipole, access_location::host, access_mode::readwrite);

	// External field at the particle's position
	GPUArray<Scalar3> n_extfield(m_group_size, m_exec_conf);
	m_extfield.swap(n_extfield);
	ArrayHandle<Scalar3> h_extfield(m_extfield, access_location::host, access_mode::readwrite);

	// Get access to particle conductivities
	ArrayHandle<Scalar> h_conductivity(m_conductivity, access_location::host, access_mode::readwrite);

	// Fill the external field and dipole arrays
	for( unsigned int ii = 0; ii < m_group_size; ++ii){

		// Use the true external field to fill the particle external field array
		h_extfield.data[ii] = m_field;
		
		// Compute beta parameter
		Scalar lambda_p = h_conductivity.data[ii];
		Scalar beta;
		if ( std::isfinite(lambda_p) ){
			beta = (lambda_p - 1.)/(lambda_p + 2.);
		} else {
			beta = 1.;
		}

		// Fill the dipole array with the isolated particle (constant dipole model) dipole
		h_dipole.data[ii] = 4.0*PI*beta*m_field;
	}
}

// Update the applied external field.  Does not require recomputing tables on the CPU.
void MutualDipole::UpdateField(std::vector<float> &field,
				std::vector<float> &gradient)
{

	// Set the new field and field gradient
	m_field = make_scalar3(field[0], field[1], field[2]);
	m_gradient = make_scalar3(gradient[0], gradient[1], gradient[2]);
	
	// Get access to the particle external field, conductivity, and dipole arrays
	ArrayHandle<Scalar3> h_extfield(m_extfield, access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar> h_conductivity(m_conductivity, access_location::host, access_mode::read);
	ArrayHandle<Scalar3> h_dipole(m_dipole, access_location::host, access_mode::readwrite);

	// Update arrays
	for( unsigned int ii = 0; ii < m_group_size; ++ii){

		// Update the particle external field array
		h_extfield.data[ii] = m_field;
		
		// Compute beta parameter
		Scalar lambda_p = h_conductivity.data[ii];
		Scalar beta;
		if ( std::isfinite(lambda_p) ){
			beta = (lambda_p - 1.)/(lambda_p + 2.);
		} else {
			beta = 1.;
		}

		// Fill the dipole array with the new isolated particle (constant dipole model) dipole
		h_dipole.data[ii] = 4.0*PI*beta*m_field;
	}
}

// Update simulation parameters.  Recomputes tables on the CPU.
void MutualDipole::UpdateParameters(std::vector<int> &group_tag,
		      		     std::vector<float> &conductivity,
						 std::vector<float> &radii,
						 std::vector<unsigned int> &radii_tag,
						 std::vector<float> &field,
				     	 std::vector<float> &gradient,
		      		     std::string fileprefix,
		      		     int period,
				     	 int constantdipoleflag,
		      		     unsigned int t0) 
{

	// Extract inputs
	m_field = make_scalar3(field[0], field[1], field[2]);
	m_gradient = make_scalar3(gradient[0], gradient[1], gradient[2]);
	m_fileprefix = fileprefix;
	m_period = period;
	m_constantdipoleflag = constantdipoleflag,
	m_t0 = t0;

	// Get access to particle external field, conductivity, and dipole arrays
	ArrayHandle<int> h_group_tag(m_group_tag, access_location::host, access_mode::read);
	ArrayHandle<Scalar3> h_extfield(m_extfield, access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar> h_conductivity(m_conductivity, access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar3> h_dipole(m_dipole, access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar> h_radii(m_radii, access_location::host, access_mode::read);
	ArrayHandle<unsigned int> h_radii_tag(m_radii_tag, access_location::host, access_mode::read);

	// Update arrays
	for (unsigned int i = 0; i < m_group_size; ++i ){

		// Update the particle external field array
		h_extfield.data[i] = m_field;
		
		// Update particle conductivities
		Scalar lambda_p = conductivity[i];
		h_conductivity.data[i] = lambda_p;

		// Compute beta parameter
		Scalar beta;
		if ( std::isfinite(lambda_p) ){
			beta = (lambda_p - 1.)/(lambda_p + 2.);
		} else {
			beta = 1.;
		}

		// Fill the dipole array with the new isolated particle (constant dipole model) dipole
		h_dipole.data[i] = 4.0*PI*beta*m_field;
	}
}

// Compute forces on particles
void MutualDipole::computeForces(unsigned int timestep) {

	// access the particle forces (associated with this plugin only; other forces are stored elsewhere)
	ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::readwrite);

	// zero the particle forces
	gpu_ZeroForce(m_Ntotal, d_force.data, 512);

	// update the neighbor list
	m_nlist->compute(timestep);

	// profile this step
	if (m_prof)
		m_prof->push(m_exec_conf, "MutualDipole2");

	////// Access all of the needed data

	// particle positions
	ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

	// particle radii
	ArrayHandle<Scalar> d_radii(m_radii, access_location::device, access_mode::read);

	// particle radii tag
	ArrayHandle<unsigned int> d_radii_tag(m_radii_tag, access_location::device, access_mode::read);

	// particle conductivities
	ArrayHandle<Scalar> d_conductivity(m_conductivity, access_location::device, access_mode::readwrite);

	// particle dipoles
	ArrayHandle<Scalar3> d_dipole(m_dipole, access_location::device, access_mode::readwrite);

	// external field at particle centers
	ArrayHandle<Scalar3> d_extfield(m_extfield, access_location::device, access_mode::readwrite);

	// active group indices
	ArrayHandle<int> d_group_membership_tag(m_group_membership_tag, access_location::device, access_mode::readwrite);

	// tag
	ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

	// particles in the active group
	ArrayHandle<unsigned int> d_group_members(m_group->getIndexArray(), access_location::device, access_mode::read);

	// group_tag
	ArrayHandle<int> d_group_tag(m_group_tag, access_location::device, access_mode::read);

	// simulation box
	BoxDim box = m_pdata->getBox();

	// wave vectors and scalings on grid
	ArrayHandle<Scalar4> d_gridk(m_gridk, access_location::device, access_mode::read);

	// x, y, and z components of the wave space grid
	ArrayHandle<CUFFTCOMPLEX> d_gridX(m_gridX, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_gridY(m_gridY, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_gridZ(m_gridZ, access_location::device, access_mode::readwrite);

	// real space field table
	ArrayHandle<Scalar4> d_fieldtable(m_fieldtable, access_location::device, access_mode::read);

	// real space force table
	ArrayHandle<Scalar4> d_forcetable(m_forcetable, access_location::device, access_mode::read);

	// neighbor list
	ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(), access_location::device, access_mode::read);

	// index in neighbor list where each particle's neighbors begin
	ArrayHandle<unsigned int> d_head_list(m_nlist->getHeadList(), access_location::device, access_mode::read);

	// number of neighbors of each particle
	ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(), access_location::device, access_mode::read);

	// set the block size of normal GPU kernels
	int block_size = 512;

	// perform the calculation on the GPU
	// NOTE: the real-space tables are now laid out by type pair. The CUDA kernel
	// still needs the matching lookup change in the .cu/.cuh files:
	// table_idx = (type_i * n_types + type_j) * (m_Ntable + 1) + r_idx.
	// Pass n_types/table_width into gpu_ComputeForce when updating that signature.
	gpu_ComputeForce(d_pos.data,
			d_radii.data,
			d_radii_tag.data,
			d_conductivity.data,
			d_dipole.data,
			d_extfield.data,
			m_field,
			m_gradient,
			d_force.data,

			m_Ntotal,
			m_group_size,
			d_group_membership_tag.data,
			d_group_members.data,
			d_group_tag.data, 
			d_tag.data, 
			box,

			block_size,

			m_xi,
			m_errortol,
			m_eta,
			m_rc,
			m_Nx,
			m_Ny,
			m_Nz,
			m_gridh,
			m_P,

			d_gridk.data,
			d_gridX.data,
			d_gridY.data,
			d_gridZ.data,
			m_plan,

			m_Ntable,
			m_drtable,
			d_fieldtable.data,
			d_forcetable.data,
			m_radii_types, 

			d_nlist.data,
			d_head_list.data,
			d_n_neigh.data,

			m_constantdipoleflag);

	if (m_exec_conf->isCUDAErrorCheckingEnabled())
		CHECK_CUDA_ERROR();

	// done profiling
	if (m_prof)
		m_prof->pop(m_exec_conf);

	// If the period is set, create a file every period timesteps.
	if ( ( m_period > 0 ) && ( (int(timestep) - m_t0) % m_period == 0 ) ) {
		OutputData(int(timestep));
	}
}

// Write quantities to file
void MutualDipole::OutputData(unsigned int timestep) {

	// Format the timestep to a string
	std::ostringstream timestep_str;
	timestep_str << std::setw(10) << std::setfill('0') << timestep;

	// Construct the filename
	std::string filename = m_fileprefix + "." + timestep_str.str() + ".txt";

	// Access needed data
	ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
	ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
	ArrayHandle<Scalar3> h_dipole(m_dipole, access_location::host, access_mode::read);
	ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
	ArrayHandle<int> h_group_membership_tag(m_group_membership_tag, access_location::host, access_mode::read);
	ArrayHandle<int> h_group_tag(m_group_tag, access_location::host, access_mode::read);

	// Open the file
	std::ofstream file;
	file.open(filename.c_str(), std::ios_base::out);

	// Check that the file opened correctly
        if (!file.good()) {
                throw std::runtime_error("Error in MutualDipole2: unable to open output file.");
        }

	////// Write the particle positions to file in global tag order

	// Header
	file << "Position_x Position_y Position_z i idx" << std::endl;

	// Loop through particle tags
	for (int i = 0; i < m_Ntotal; i++) {

		// Get the particle's global index
		unsigned int idx = h_rtag.data[i];

		// Get the particle's position
		Scalar4 postype = h_pos.data[idx];

		// Write the position to file
		file << std::setprecision(10) << postype.x << "  " << postype.y << "  " << postype.z << "  " << i << "  " << idx << "  " << std::endl;
	}

	////// Write the particle dipoles to file in global tag order
	file << "Dipole_x  Dipole_y  Dipole_z  i  group_tag" << std::endl;
	for (int i = 0; i < m_Ntotal; i++) {

		// Get the particle's global index
		unsigned int idx = h_rtag.data[i];

		// Get the particle's active group-specific index
		int group_idx = h_group_membership_tag.data[i];

		if (idx >= m_Ntotal) continue;

		// Get the particle's active group-specific index
		int group_tag = h_group_tag.data[i];
		printf("OutputData [Dipole]: i = %d, group_tag = h_group_tag.data[i] = %d \n", i, group_tag);

		// Get the particle's dipole if it is in the active group.  Else, set the dipole to 0.
		Scalar3 dipole;
		if (group_idx != -1) {
			dipole = h_dipole.data[group_tag];
		} else {
			dipole = make_scalar3(0.0, 0.0, 0.0);
		}

		// Write the dipole to file
		file << std::setprecision(10)
			<< dipole.x << "  " << dipole.y << "  " << dipole.z << "  "
			<< i << "  " << group_tag << std::endl;
	}

	////// Write the particle electric/magnetic forces to file in global tag order
	file << "Force" << std::endl;
	for (int i = 0; i < m_Ntotal; i++) {

		// Get the particle's global index
		unsigned int idx = h_rtag.data[i];

		// Get the particle's electric/magnetic force
		Scalar4 force = h_force.data[idx];

		// Write the dipole to file
		file << std::setprecision(10) << force.x << "  " << force.y << "  " << force.z << "  " << std::endl;
	}


	// Close output file
	file.close();
}

void export_MutualDipole(pybind11::module& m)
{
    pybind11::class_<MutualDipole, std::shared_ptr<MutualDipole>> (m, "MutualDipole", pybind11::base<ForceCompute>())
		.def(pybind11::init< std::shared_ptr<SystemDefinition>, 
							 std::shared_ptr<ParticleGroup>, 
							 std::shared_ptr<NeighborList>, 
							 std::vector<int>&, 	// group tag
							 std::vector<float>&, 	// conductivity
							 std::vector<float>&, 	// radii
							 std::vector<unsigned int>&, 	// radii tag
							 std::vector<float>&, 	// field
							 std::vector<float>&, 	// gradient
							 Scalar, 
							 Scalar, 
							 std::string, 
							 int, 
							 int, 
							 unsigned int >())
		.def("SetParams", &MutualDipole::SetParams)
		.def("UpdateField", &MutualDipole::UpdateField)
		.def("UpdateParameters", &MutualDipole::UpdateParameters)
		.def("computeForces", &MutualDipole::computeForces)
		.def("OutputData", &MutualDipole::OutputData)
        ;
}

#ifdef WIN32
#pragma warning( pop )
#endif
