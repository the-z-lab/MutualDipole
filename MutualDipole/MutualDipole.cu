#include <cusp/monitor.h>
#include <cusp/array1d.h>
#include <cusp/krylov/gmres.h>

#include <stdio.h>
#include "MutualDipole.cuh"
#include "PotentialWrapper.cuh"
#include "hoomd/TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define PI 3.1415926535897932f

// Declare textured memory
texture<Scalar4, 1, cudaReadModeElementType> fieldtable_tex;
texture<Scalar4, 1, cudaReadModeElementType> forcetable_tex;
texture<Scalar4, 1, cudaReadModeElementType> pos_tex;

// Zero particle forces
__global__ void zeroforce(unsigned int Ntotal, // total number of particles
			  Scalar4 *d_force) // pointer to particle forces
{
	// Linear index of current thread
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// Set the force to zero
	if (tid < Ntotal) {
		d_force[tid] = make_scalar4( 0.0, 0.0, 0.0, 0.0 );
	}
}

// Zero particle forces (called on the host)
cudaError_t gpu_ZeroForce(unsigned int Ntotal, // total number of particles
			  Scalar4 *d_force, // pointer to particle forces
			  unsigned int block_size) // number of threads per block
{
	// Use one thread per particle
	int Nthreads = block_size;
	int Nblocks = Ntotal/block_size + 1;

	// Call the GPU function to zero forces
	zeroforce<<<Nblocks,Nthreads>>>(Ntotal, d_force);

	return cudaSuccess;
}

// Zero the grid
__global__ void initialize_grid(CUFFTCOMPLEX *grid, // pointer to the grid array
				unsigned int Ngrid) // total number of grid points 
{
	// Linear index of current thread
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// Set the grid value to zero
	if (tid < Ngrid) {
		grid[tid] = make_scalar2( 0.0, 0.0 );
	}

}

// Initialize the active group membership/index list
__global__ void initialize_groupmembership_tag( int *d_group_membership_tag, // pointer to active group membership list
				      	    unsigned int Ntotal) // total number of particles
{
	// Global particle index
	unsigned int tag = threadIdx.x + blockIdx.x*blockDim.x;

	// Flag every particle as not a member of the active group
	if (tag < Ntotal) {
		d_group_membership_tag[tag] = -1;
	}
}

// Determine active group membership and group-specific index for all of the particles.  
// A particle with global index i that is not a member of the group has d_group_membership[i] = -1.
// A particle with global index i that is a member of the group has its group-specific index in d_group_membership[i].
// That is, d_group_members[d_group_membership[i]] = i.
__global__ void groupmembership_tag( int *d_group_membership_tag, // pointer to active group membership list
				 unsigned int *d_group_members, // pointer to indices of active group members
				 unsigned int group_size, // number of particles in the active group
				 unsigned int *d_tag)
{
	// Group-specific particle index
	unsigned int group_idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (group_idx < group_size) {
		
		// Global particle index
		unsigned int idx = d_group_members[group_idx]; // HOOMD index
		unsigned int tag = d_tag[idx]; // Global tag

		// Set the group-specific index at the current particle's global index position in the group membership list
		d_group_membership_tag[tag] = group_idx; // Map tag -> group_idx
	}

}

// Spread particle dipole moments to a uniform grid.  Use a P-by-P-by-P block per particle with a thread per grid node. 
__global__ void spread( Scalar4 *d_pos, // pointer to particle positions
			Scalar3 *d_dipole, // pointer to particle dipole moments
			int group_size, // number of particles belonging to the active group
			unsigned int *d_group_members, // pointer to particle indices belonging to the group
			unsigned int *d_tag, 
			int *d_group_tag, // An input from python side, no calculation needed
			BoxDim box, // simulation box
			Scalar3 eta, // spectral splitting parameter
			int Nx, // number of grid nodes in x dimension 
			int Ny, // number of grid nodes in y dimension
			int Nz, // number of grid nodes in z dimension
			Scalar3 gridh, // grid spacing
			const int P, // number of nodes to spread the particle dipole over
			CUFFTCOMPLEX *gridX, // pointer to grid on which to spread the x component of dipoles
			CUFFTCOMPLEX *gridY, // pointer to grid on which to spread the y component of dipoles
			CUFFTCOMPLEX *gridZ, // pointer to grid on which to spread the z component of dipoles
			Scalar xiterm, // precomputed term 2*xi^2
			Scalar prefac ) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)
{
	// Setup shared memory for the particle position
	__shared__ float3 shared;
	float3 *pos_shared = &shared;	

	// The block index corresponds to a particle index in the active group and the thread index corresponds to a grid node 
	unsigned int group_idx = blockIdx.x;
	unsigned int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;

	// Global ID of current particle
	unsigned int idx = d_group_members[group_idx];
	unsigned int tag = d_tag[idx];
	int group_tag = d_group_tag[tag];

	// Have the first thread fetch the particle position and store it in shared memory
	if (thread_offset == 0) {

		Scalar4 tpos = __ldg(d_pos+idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Current particle's dipole
	Scalar3 Sj = d_dipole[group_tag];

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Retrieve position from shared memory
	Scalar3 pos = *pos_shared;

	// Fractional position within box (0 to 1)
	Scalar3 pos_frac = box.makeFraction(pos);

	// Particle position in units of grid spacing
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;

	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).   
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and r^2/eta
	Scalar3 r = pos_grid - pos;
	Scalar r2eta = r.x*r.x/eta.x + r.y*r.y/eta.y + r.z*r.z/eta.z;

	// Contribution to the current grid node from the current particle dipole
	Scalar3 dipole_inp = prefac*expf( -xiterm*r2eta )*Sj;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Add dipole contribution to the grid
	atomicAdd( &(gridX[grid_idx].x), dipole_inp.x);
	atomicAdd( &(gridY[grid_idx].x), dipole_inp.y);
	atomicAdd( &(gridZ[grid_idx].x), dipole_inp.z);
}

// Scale the gridded data.  This can be thought of as converting dipole moments to fields.
__global__ void scale(  Scalar4 *gridk,      // pointer to wave vector and scaling factor associated with each grid node
			CUFFTCOMPLEX *gridX, // pointer to x-component of wave space grid
			CUFFTCOMPLEX *gridY, // pointer to y-component of wave space grid
			CUFFTCOMPLEX *gridZ, // pointer to z-component of wave space grid
			unsigned int Ngrid)  // total number of grid nodes
{
  	// Current thread's linear index
  	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  	// Ensure thread index corresponds to a node within the grid
  	if ( tid < Ngrid ) {

    		// Read the current grid value
    		Scalar2 SX = gridX[tid];
    		Scalar2 SY = gridY[tid];
    		Scalar2 SZ = gridZ[tid];

    		// Current wave-space vector and scaling 
    		Scalar4 k = gridk[tid];

    		// Dot product of wave-vector with dipoles.  Ensure that the value corresponding to the k = 0 grid node is 0.
    		Scalar2 kdotm = (tid==0) ? make_scalar2(0.0,0.0) : make_scalar2( ( k.x*SX.x+k.y*SY.x+k.z*SZ.x ), ( k.x*SX.y+k.y*SY.y+k.z*SZ.y ) );

    		// Scaled the grid values
    		gridX[tid] = make_scalar2( k.w*k.x*kdotm.x, k.w*k.x*kdotm.y );
    		gridY[tid] = make_scalar2( k.w*k.y*kdotm.x, k.w*k.y*kdotm.y );
    		gridZ[tid] = make_scalar2( k.w*k.z*kdotm.x, k.w*k.z*kdotm.y );
  	}
}

// Contract the grid values to particle centers for the field calculation. Use a P-by-P-by-P block per particle with a thread per grid node.
__global__ void contractfield(	Scalar4 *d_pos,  // pointer to particle positions
				Scalar3 *d_dipole, // pointer to particle dipole moments
				Scalar3 *d_extfield, // pointer to external field at particle centers
				int group_size, // number of particles in the active group
				unsigned int *d_group_members, // pointer to indices of particles belonging to the activer group
				unsigned int *d_tag, 
				int *d_group_tag, 
				BoxDim box, // simulation box
				Scalar xi, //  Ewald splitting parameter
				Scalar3 eta, // spectral splitting parameter
				int Nx, // number of grid nodes in x dimension
				int Ny, // number of grid nodes in y dimension
				int Nz, // number of grid nodes in z dimension
				Scalar3 gridh, // grid spacing
				const int P, // number of nodes to contract over
				CUFFTCOMPLEX *gridX, // pointer to x component of grid
				CUFFTCOMPLEX *gridY, // pointer to y component of grid
				CUFFTCOMPLEX *gridZ, // pointer to z component of grid 
				Scalar xiterm, // precomputed term 2*xi^2
				Scalar prefac) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)*gridh.x*gridh.y*gridh.z
{
	// Setup shared memory for particle fields and particle positions
	extern __shared__ float3 shared[];  // 16 kb maximum
	float3 *field = shared;
	float3 *pos_shared = &shared[P*P*P];

	// Get the block index, linear index of the thread within the block, and the number of threads in the block. The block index corresponds to a particle index in the active group and the thread index corresponds to a grid node.
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;
	int block_size = blockDim.x*blockDim.y*blockDim.z;

	// Global particle ID
    unsigned int idx = d_group_members[group_idx];
	unsigned int tag = d_tag[idx];
	int group_tag = d_group_tag[tag];

	// Initialize the shared memory for the field and have the first thread fetch the particle position and store it in shared memory
	field[thread_offset] = make_scalar3(0.0,0.0,0.0);
	if (thread_offset == 0){
		Scalar4 tpos = __ldg(d_pos+idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Fetch position from shared memory
	Scalar3 pos = pos_shared[0];

    	// Express the particle position in units of grid spacing.
    	Scalar3 pos_frac = box.makeFraction(pos);
    	pos_frac.x *= (Scalar)Nx;
    	pos_frac.y *= (Scalar)Ny;
    	pos_frac.z *= (Scalar)Nz;

    	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and "r^2/eta"
	Scalar3 r = pos - pos_grid;
	Scalar r2eta = r.x*r.x/eta.x + r.y*r.y/eta.y + r.z*r.z/eta.z;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Contribution to the current particle from the current grid node
	Scalar3 gridXYZ = make_scalar3(gridX[grid_idx].x, gridY[grid_idx].x, gridZ[grid_idx].x);
	field[thread_offset] = prefac*expf( -xiterm*r2eta )*gridXYZ;	

	// Reduction to add all of the P^3 values
	int offs = block_size;
	int oldoffs;
	while ( offs > 1){
		oldoffs = offs; // store the previous value of offs
		offs = (offs+1)/2; // the current value of offs is half of the previous (result is a rounded up integer)
		__syncthreads();
		if ( thread_offset + offs < oldoffs ){
			field[thread_offset] += field[thread_offset + offs];
		}
	}

	// Have a single thread store the current particle's field
	if (thread_offset == 0){
		d_extfield[group_tag] = field[0];
	}
}

// Contract the grid values to the particle centers for the force calculation.  Use a P-by-P-by-P block per particle with a thread per grid node.
__global__ void contractforce(	Scalar4 *d_pos,  // pointer to particle positions
				Scalar3 *d_dipole, // pointer to particle dipole moments
				Scalar4 *d_force, // pointer to particle forces
				int group_size, // number of particles in the active group
				unsigned int *d_group_members, // pointer to indices of particles belonging to the active group
				unsigned int *d_tag, 
				int *d_group_tag, 
				BoxDim box, // simulation box
				Scalar3 eta, // spectral splitting parameter
				int Nx, // number of grid nodes in x dimension
				int Ny, // number of grid nodes in y dimension
				int Nz, // number of grid nodes in z dimension
				Scalar3 gridh, // grid spacing
				const int P, // number of nodes to contract over
				CUFFTCOMPLEX *gridX, // pointer to x component of grid
				CUFFTCOMPLEX *gridY, // pointer to y component of grid
				CUFFTCOMPLEX *gridZ, // pointer to z component of grid 
				Scalar xiterm, // precomputed term 2*xi^2
				Scalar prefac ) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)*gridh.x*gridh.y*gridh.z
{
	// Setup shared memory for particle forces and particle positions
	extern __shared__ float3 shared[];  // 16 kb maximum
	float3 *force = shared;
	float3 *pos_shared = &shared[P*P*P];

	// Get the block index, linear index of the thread within the block, and the number of threads in the block. The block index corresponds to a particle index in the active group and the thread index corresponds to a grid node.
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;
	int block_size = blockDim.x*blockDim.y*blockDim.z;

	// Global particle ID
    unsigned int idx = d_group_members[group_idx];
	unsigned int tag = d_tag[idx];
	int group_tag = d_group_tag[tag];

	// Initialize the shared memory for the force and have the first thread fetch the particle position and store it in shared memory
	force[thread_offset] = make_scalar3(0.0,0.0,0.0);
	if (thread_offset == 0){
		Scalar4 tpos = __ldg(d_pos+idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Current particle's dipole
	Scalar3 Si = d_dipole[group_tag];

	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Fetch position from shared memory
	Scalar3 pos = pos_shared[0];

    // Express the particle position in units of grid spacing.
    Scalar3 pos_frac = box.makeFraction(pos);
    pos_frac.x *= (Scalar)Nx;
    pos_frac.y *= (Scalar)Ny;
    pos_frac.z *= (Scalar)Nz;

    	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and combinations with eta
	Scalar3 r = pos - pos_grid;
	Scalar3 reta = r/eta;
	Scalar r2eta = r.x*reta.x + r.y*reta.y + r.z*reta.z;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Contribution to the current grid node from the current particle dipole
	Scalar Sidotgrid = Si.x*gridX[grid_idx].x + Si.y*gridY[grid_idx].x + Si.z*gridZ[grid_idx].x;
	force[thread_offset] = 2.0*prefac*xiterm*Sidotgrid*expf( -xiterm*r2eta )*reta;

	// Reduction to add all of the P^3 values
	int offs = block_size;
	int oldoffs;
	while ( offs > 1){
		oldoffs = offs; // store the previous value of offs
		offs = (offs+1)/2; // the current value of offs is half of the previous (result is a rounded up integer)
		__syncthreads();
		if ( thread_offset + offs < oldoffs ){
			force[thread_offset] += force[thread_offset + offs];
		}
	}

	// Have a single thread store the current particle's force
	if (thread_offset == 0){
		d_force[idx] = make_scalar4(force[0].x, force[0].y, force[0].z, 0.0);
	}
}

// Add real space contribution to particle field
__global__ void real_space_field( 	Scalar4 *d_pos, // pointer to particle positions
					Scalar *d_conductivity, // pointer to particle conductivities
					Scalar3 *d_dipole, // pointer to particle dipoles
					Scalar3 *d_extfield, // pointer to particle external field
					int group_size, // number of particles in the active group
					int *d_group_membership_tag, // particle membership and index in active group
					unsigned int *d_group_members, // pointer to indices of particles in the active group
					unsigned int *d_tag, 
					int *d_group_tag,
					BoxDim box, // simulation box
					Scalar rc, // real space cutoff radius
					int Ntable, // number of entries in the real space table
					Scalar drtable, // spacing between table entries
					Scalar4 *d_fieldtable, // pointer to real space field table
					const unsigned int *d_nlist, // pointer to the neighbor list
					const unsigned int *d_head_list, // pointer to head list used to access elements of the neighbor list
					const unsigned int *d_n_neigh, // pointer to the number of neighbors of each particle 
					Scalar selfcoeff) // coefficient of the self term
{
  	// Index for current particle
  	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Ensure that current particle is within the group
  	if (group_idx < group_size) {

		// Global ID of current particle
		unsigned int idx = d_group_members[group_idx];
		unsigned int tag = d_tag[idx];
		int group_tag = d_group_tag[tag];

		// Get the wave space contribution to the field
  		Scalar3 E = d_extfield[group_tag];

		// Dipole moment and conductivity of current particle
		Scalar3 Si = d_dipole[group_tag];
		Scalar lambda_p = d_conductivity[group_tag];
		
		// Add real space self term
		E += selfcoeff*Si;

		// If the particle conductivity is finite, add an additional self term
		if ( isfinite(lambda_p) ) {
			E += 3.0/(4.0*PI*(lambda_p - 1.0))*Si;
		}


		// Number of neighbors and location of neighbors in neighbor list for current particle
		unsigned int n_neigh = d_n_neigh[idx];
		unsigned int head_i = d_head_list[idx];

		// Current particle position and type
		Scalar4 postypei = __ldg(d_pos+idx);
		Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

		// Minimum and maximum distances squared for pair calculation
		Scalar rc2 = rc*rc;
		Scalar rmin2 = drtable*drtable;

		// Loop over neighbors
    		for (int j=0; j < n_neigh; j++) {
			
			// Get neighbor global and group index
			unsigned int neigh_idx = d_nlist[head_i + j];
			unsigned int neigh_tag = d_tag[neigh_idx];
			int neigh_group_idx = d_group_membership_tag[neigh_tag];
			int neigh_group_tag = d_group_tag[neigh_tag];

			// Check if neighbor is a member of the active group
			if ( neigh_group_idx != -1 ) {

				// Position and type of neighbor particle
				Scalar4 postypej = __ldg(d_pos+neigh_idx);
				Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

				// Distance vector between current particle and neighbor
        			Scalar3 r = posi - posj;
        			r = box.minImage(r); // nearest image distance vector
        			Scalar dist2 = dot(r,r); // distance squared

				// Add neighbor contribution if it is within the real space cutoff radius
       				if ( ( dist2 < rc2 ) && ( dist2 >= rmin2 ) ) {

					Scalar dist = sqrtf(dist2); // distance
					r = r/dist; // convert r to a unit vector

					// Dipole of neighbor particle
					Scalar3 Sj = d_dipole[neigh_group_tag];

					// Dot product of neighbor dipole and r
					Scalar Sjdotr = Sj.x*r.x + Sj.y*r.y + Sj.z*r.z;

					// Read the table values closest to the current distance
					int tableind = __scalar2int_rd( Ntable * (dist-drtable)/(rc-drtable) );	
					Scalar4 entry = __ldg(d_fieldtable+tableind);

					// Linearly interpolate between the table values
					Scalar lininterp = dist/drtable - tableind - Scalar(1.0);
					Scalar C1 = entry.x + ( entry.z - entry.x )*lininterp;
					Scalar C2 = entry.y + ( entry.w - entry.y )*lininterp;  

					// Real-space contributions to the field
					E += C1*(Sj - Sjdotr*r) + C2*Sjdotr*r;
	
      				} // end neighbor contribution
			} // end membership check
		}// end neighbor loop

		// Write the result to the current particle's field
		d_extfield[group_tag] = E;

	}
}

// Add real space contribution to particle forces
__global__ void real_space_force( 	Scalar4 *d_pos, // pointer to particle positions
					Scalar3 *d_dipole, // pointer to particle dipoles
					Scalar3 field, // external field
					Scalar3 gradient, // external field gradient
					Scalar4 *d_force, // pointer to particle forces
					int group_size, // number of particles in active group
					int *d_group_membership_tag, // pointer to particle membership and index in active group
					unsigned int *d_group_members, // pointer to indices of particles in the active group
					unsigned int *d_tag, 
					int *d_group_tag, 
					BoxDim box, // simulation box
					Scalar rc, // real space cutoff radius
					int Ntable, // number of entries in the real space table
					Scalar drtable, // spacing between table entries
					Scalar4 *d_forcetable, // pointer to real space force table
					const unsigned int *d_nlist, // pointer to the neighbor list
					const unsigned int *d_head_list, // pointer to head list used to access elements of the neighbor list
					const unsigned int *d_n_neigh) // pointer to the number of neighbors of each particle 
{
  	// Index for current particle
  	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Ensure that current particle is within the group
  	if (group_idx < group_size) {

		// Global ID of current particle
		unsigned int idx = d_group_members[group_idx];
		unsigned int tag = d_tag[idx];
		int group_tag = d_group_tag[tag];

		// Get the wave spcae contribution to the force
  		Scalar4 F4 = d_force[idx];
		Scalar3 F = make_scalar3(F4.x, F4.y, F4.z);

		// Dipole moment of current particle
		Scalar3 Si = d_dipole[group_tag];

		// Add the phoretic force
		Scalar field_mag = sqrtf(field.x*field.x + field.y*field.y + field.z*field.z); // field magnitude
		if (field_mag >= 1e-6){
			F += gradient*(Si.x*field.x + Si.y*field.y + Si.z*field.z)/field_mag;
		}

		// Number of neighbors and location of neighbors in neighbor list for current particle
		unsigned int n_neigh = d_n_neigh[idx];
		unsigned int head_i = d_head_list[idx];

		// Current particle position and type
		Scalar4 postypei = __ldg(d_pos+idx);
		Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

		// Minimum and maximum distances squared for pair calculation
		Scalar rc2 = rc*rc;
		Scalar rmin2 = drtable*drtable;

		// Loop over neighbors
    		for (int j=0; j < n_neigh; j++) {

			// Get neighbor global and group index
			unsigned int neigh_idx = d_nlist[head_i + j];
			unsigned int neigh_tag = d_tag[neigh_idx];
			int neigh_group_idx = d_group_membership_tag[neigh_tag];
			int neigh_group_tag = d_group_tag[neigh_tag];

			// Check if neighbor is a member of the active group
			if ( neigh_group_idx != -1 ) {

				// Position and type of neighbor particle
				Scalar4 postypej = __ldg(d_pos+neigh_idx);
				Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

				// Distance vector between current particle and neighbor
        			Scalar3 r = posi - posj;
        			r = box.minImage(r); // nearest image distance vector
        			Scalar dist2 = dot(r,r); // distance squared

				// Add neighbor contribution if it is within the real space cutoff radius
       				if ( ( dist2 < rc2 ) && ( dist2 >= rmin2 ) ) {

					Scalar dist = sqrtf(dist2); // distance
					r = r/dist; // convert r to a unit vector

					// Dipole of neighbor particle
					Scalar3 Sj = d_dipole[neigh_group_tag];

					// Dot products involving the two dipoles
					Scalar SidotSj = Si.x*Sj.x + Si.y*Sj.y + Si.z*Sj.z;
					Scalar Sidotr = Si.x*r.x + Si.y*r.y + Si.z*r.z;
					Scalar Sjdotr = Sj.x*r.x + Sj.y*r.y + Sj.z*r.z;
	
					// Read the table values closest to the current distance
					int tableind = __scalar2int_rd( Ntable * (dist-drtable)/(rc-drtable) );	
					Scalar4 entry = __ldg(d_forcetable+tableind);

					// Linearly interpolate between the table values
					Scalar lininterp = dist/drtable - tableind - Scalar(1.0);
					Scalar C1 = entry.x + ( entry.z - entry.x )*lininterp;
					Scalar C2 = entry.y + ( entry.w - entry.y )*lininterp;  

					// Real-space contribution to the force
					F += -C1*(SidotSj*r + Sjdotr*Si + Sidotr*Sj) + (2.0*C1-C2)*Sidotr*Sjdotr*r;

      				} // end neighbor contribution
			} // end membership check
		}// end neighbor loop

		// Write the result to the current particle's force
		F4 = make_scalar4(F.x, F.y, F.z, 0.0);
		d_force[idx] = F4;
	}
}

// Compute the external field at the particle centers as determined by the particle dipoles. (called on the host)
cudaError_t ComputeField(       Scalar4 *d_pos, // pointer to particle positions
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
				const unsigned int *d_nlist, // pointer to neighbor list
				const unsigned int *d_head_list, // pointer to head list used to access entries in the neighbor list
				const unsigned int *d_n_neigh) // pointer to number of neighbors of each particle
{
	// Total number of grid nodes
 	unsigned int Ngrid = Nx*Ny*Nz;

	// For initialization and scaling, use one thread per grid node
    int Nthreads1 = ( Ngrid > block_size ) ? block_size : Ngrid;
    int Nblocks1 = ( Ngrid - 1 )/Nthreads1 + 1;

	// For spreading and contracting, use one P-by-P-by-P block of threads per particle.
	dim3 Nblocks2(group_size, 1, 1);
	dim3 Nthreads2(P, P, P);

    // For the real space calculation, use one thread per particle
    int Nblocks3 = group_size/block_size + 1;
    int Nthreads3 = block_size;

	// Precomputed quantities needed for the GPU kernels
	Scalar quadW = gridh.x*gridh.y*gridh.z; // trapezoidal rule weights
	Scalar xi2 = xi*xi;
	Scalar xi3 = xi2*xi;
	Scalar xiterm = 2.0*xi2;
	Scalar prefac = xiterm*xi/PI*sqrtf(2.0/(PI*eta.x*eta.y*eta.z));  // prefactor for the spreading and contracting exponentials
	Scalar selfterm = (-1.0+6.0*xi2)/(16.0*PI*sqrt(PI)*xi3) + (1.0 - 2.0*xi2)*exp(-4.0*xi2)/(16.0*PI*sqrt(PI)*xi3) + erfc(2.0*xi)/(4.0*PI); // self term in the potential matrix

    // Reset the grid values to zero
    initialize_grid<<<Nblocks1, Nthreads1>>>(d_gridX,Ngrid);
    initialize_grid<<<Nblocks1, Nthreads1>>>(d_gridY,Ngrid);
    initialize_grid<<<Nblocks1, Nthreads1>>>(d_gridZ,Ngrid);

	// Spread dipoles from the particles to the grid
	spread<<<Nblocks2, Nthreads2>>>(d_pos, d_dipole, group_size, d_group_members, d_tag, d_group_tag, box, eta, Nx, Ny, Nz, gridh, P, d_gridX, d_gridY, d_gridZ, xiterm, prefac);

	// Compute the Fourier transform of the gridded data
    cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_FORWARD);
    cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_FORWARD);
    cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_FORWARD);

	// Scale the grid values
    scale<<<Nblocks1, Nthreads1>>>(d_gridk, d_gridX, d_gridY, d_gridZ, Ngrid);

	// Inverse Fourier transform the gridded data
    cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_INVERSE);
    cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_INVERSE);
    cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_INVERSE);

	// Contract the gridded values to the particles to get the wave space contribution to the field
	contractfield<<<Nblocks2, Nthreads2, 3*(P*P*P+1)*sizeof(float)>>>(d_pos, d_dipole, d_extfield, group_size, d_group_members, d_tag, d_group_tag, box, xi, eta, Nx, Ny, Nz, gridh, P, d_gridX, d_gridY, d_gridZ, xiterm, quadW*prefac);

	// Compute the real space contribution to the field
    real_space_field<<<Nblocks3, Nthreads3>>>(d_pos, d_conductivity, d_dipole, d_extfield, group_size, d_group_membership_tag, d_group_members, d_tag, d_group_tag, box, rc, Ntable, drtable, d_fieldtable, d_nlist, d_head_list, d_n_neigh, selfterm); 

    gpuErrchk(cudaPeekAtLastError());
    return cudaSuccess;
}

// Compute the particle dipoles iteratively using GMRES. (called on the host)
cudaError_t ComputeDipole(	Scalar4 *d_pos, // pointer to particle posisitons
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
				Scalar errortol, // error tolerance
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
				const unsigned int *d_nlist, // pointer to neighbor list
				const unsigned int *d_head_list, // pointer to head list used to access entries in the neighbor list
				const unsigned int *d_n_neigh) // pointer to number of neighbors of each particle
{
	// Create the matrix-free potential linear operator
	cuspPotential M(d_pos, d_conductivity, group_size, d_group_membership_tag, d_group_members, d_tag, d_group_tag, box, block_size, xi, eta, rc, Nx, Ny, Nz, gridh, P, d_gridk, d_gridX, d_gridY, d_gridZ, plan, Ntable, drtable, d_fieldtable, d_nlist, d_head_list, d_n_neigh);

	// Allocate storage for the solution vector (S) and output of the matrix/vector multiply (E0) on the GPU
	cusp::array1d<float, cusp::device_memory> S(M.num_rows, 0);
	cusp::array1d<float, cusp::device_memory> E0(M.num_rows, 0);

	// Get pointers to the cusp arrays
	float *d_S = thrust::raw_pointer_cast(&S[0]);
	float *d_E0 = thrust::raw_pointer_cast(&E0[0]);

	// Use the dipoles from the previous time step as the initial guess
	cudaMemcpy(d_S, d_dipole, 3*group_size*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_E0, d_extfield, 3*group_size*sizeof(float), cudaMemcpyDeviceToDevice);

	// Solve the linear system M*S = E0 using GMRES
	cusp::default_monitor<float> monitor(E0, 100, errortol);
	int restart = 10;
	cusp::krylov::gmres(M, S, E0, restart, monitor);

	// Store the computed dipoles to the correct place in device memory
	cudaMemcpy(d_dipole, d_S, 3*group_size*sizeof(float), cudaMemcpyDeviceToDevice);

	gpuErrchk(cudaPeekAtLastError());
    return cudaSuccess;
}

// Compute particle forces. (called on the host)
cudaError_t gpu_ComputeForce(   Scalar4 *d_pos, // pointer to particle posisitons
				Scalar *d_conductivity, // pointer to particle conductivities
				Scalar3 *d_dipole, // pointer to particle dipoles
				Scalar3 *d_extfield, // pointer to external field at particle centers
				Scalar3 field, // external field
				Scalar3 gradient, // external field gradient
				Scalar4 *d_force, // pointer to particle forces
				unsigned int Ntotal, // total number of particles
				unsigned int group_size, // number of particles in active group
				int *d_group_membership_tag, // pointer to particle membership and index in active group 
				unsigned int *d_group_members, // pointer to indices of particles in active group
				int *d_group_tag, 
				unsigned int *d_tag, 
				const BoxDim& box, // simulation box
				unsigned int block_size, // number of threads to use per block
				Scalar xi, // Ewald splitting parameter
				Scalar errortol, // error tolerance
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
				Scalar4 *d_forcetable, // pointer to real space force coefficient table
				const unsigned int *d_nlist, // pointer to neighbor list
				const unsigned int *d_head_list, // pointer to head list used to access entries in the neighbor list
				const unsigned int *d_n_neigh, // pointer to number of neighbors of each particle
				int constantdipoleflag) // indicates whether or not to turn off the mutual dipole functionality  
{
	// Total number of grid nodes
 	unsigned int Ngrid = Nx*Ny*Nz;

	// Tor grid initialization and scaling, use one thread per grid node
    int Nthreads1 = ( Ngrid > block_size ) ? block_size : Ngrid;
    int Nblocks1 = ( Ngrid - 1 )/Nthreads1 + 1;

	// For spreading and contracting, use one P-by-P-by-P block of threads per active particle.
	dim3 Nblocks2(group_size, 1, 1);
	dim3 Nthreads2(P, P, P);

    // For updating group membership and the real space calculation, use one thread per active particle
    int Nblocks3 = group_size/block_size + 1;
    int Nthreads3 = block_size;

	// For initializing group membership, use one thread per total particle
	int Nblocks4 = Ntotal/block_size + 1;
	int Nthreads4 = block_size;

	// Precomputed quantities needed for the GPU kernels
	Scalar quadW = gridh.x*gridh.y*gridh.z; // trapezoidal rule weights
	Scalar xiterm = 2.0*xi*xi;
	Scalar prefac = xiterm*xi/PI*sqrtf(2.0/(PI*eta.x*eta.y*eta.z));  // prefactor for the spreading and contracting exponentials

	// Handle the real space tables and particle positions as textured memory
    fieldtable_tex.normalized = false;
    fieldtable_tex.filterMode = cudaFilterModePoint; 
    cudaBindTexture(0, fieldtable_tex, d_fieldtable, sizeof(Scalar4) * (Ntable+1));

	forcetable_tex.normalized = false;
    forcetable_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, forcetable_tex, d_forcetable, sizeof(Scalar4) * (Ntable+1));

    pos_tex.normalized = false;
    pos_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, pos_tex, d_pos, sizeof(Scalar4) * Ntotal);

	// Update the group membership list
	initialize_groupmembership_tag<<<Nblocks4, Nthreads4>>>(d_group_membership_tag, Ntotal); // one thread per total particle
	groupmembership_tag<<<Nblocks3, Nthreads3>>>(d_group_membership_tag, d_group_members, group_size, d_tag); // one thread per active particle

	// Compute the particle dipoles.  If constantdipoleflag = 1, this step is skipped and the particles keep their constant dipole model values that were precomputed on the host.
	if (constantdipoleflag != 1) {
		ComputeDipole( d_pos, d_conductivity, d_dipole, d_extfield, group_size, d_group_membership_tag, d_group_members, d_tag, d_group_tag, box, block_size, xi, errortol, eta, rc, Nx, Ny, Nz, gridh, P, d_gridk, d_gridX, d_gridY, d_gridZ, plan, Ntable, drtable, d_fieldtable, d_nlist, d_head_list, d_n_neigh);
	}

    // Reset the grid values to zero
    initialize_grid<<<Nblocks1, Nthreads1>>>(d_gridX,Ngrid);
    initialize_grid<<<Nblocks1, Nthreads1>>>(d_gridY,Ngrid);
    initialize_grid<<<Nblocks1, Nthreads1>>>(d_gridZ,Ngrid);

	// Spread dipoles from the particles to the grid
	spread<<<Nblocks2, Nthreads2>>>(d_pos, d_dipole, group_size, d_group_members, d_tag, d_group_tag, box, eta, Nx, Ny, Nz, gridh, P, d_gridX, d_gridY, d_gridZ, xiterm, prefac);

	// Compute the Fourier transform of the gridded data
    cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_FORWARD);
    cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_FORWARD);
    cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_FORWARD);

	// Scale the grid values
    scale<<<Nblocks1, Nthreads1>>>(d_gridk, d_gridX, d_gridY, d_gridZ, Ngrid);

	// Inverse Fourier transform the gridded data
    cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_INVERSE);
    cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_INVERSE);
    cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_INVERSE);

	// Contract the gridded values to the particles to get the wave space contribution to the force
	contractforce<<<Nblocks2, Nthreads2, 3*(P*P*P+1)*sizeof(float)>>>(d_pos, d_dipole, d_force, group_size, d_group_members, d_tag, d_group_tag, box, eta, Nx, Ny, Nz, gridh, P, d_gridX, d_gridY, d_gridZ, xiterm, quadW*prefac);   

	// Compute the real space contribution to the force
    real_space_force<<<Nblocks3, Nthreads3>>>(d_pos, d_dipole, field, gradient, d_force, group_size,  d_group_membership_tag, d_group_members, d_tag, d_group_tag, box, rc, Ntable, drtable, d_forcetable, d_nlist, d_head_list, d_n_neigh);

	// Unbind the textured memory
	cudaUnbindTexture(fieldtable_tex);
    cudaUnbindTexture(forcetable_tex);
    cudaUnbindTexture(pos_tex);

    gpuErrchk(cudaPeekAtLastError());
    return cudaSuccess;
}

