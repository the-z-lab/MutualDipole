# This simple python interface activates the c++ MutualDipole

# Import the C++ module.
from hoomd.MutualDipole import _MutualDipole

# MutualDipole extends a ForceCompute, so we need to bring in the base class
# force and some other parts from hoomd
import hoomd
import hoomd.md

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force
from hoomd.md.force import _force

import math

# The MutualDipole class.  Computes the dipole moments and forces on each
# particle in a dielectric/paramagnetic suspension immersed in an external
# electric/magnetic field with a small, constant gradient.  The gradient is 
# only used to impart dielecto/magnetophoric forces, while the dipolar
# interactions amoung particles are computed with a constant field. The
# calculations take into account mutual polarization among the particles. 
# This feature can be turned off so that each particle is polarized only by
# the external field.  This will speed up the computation time by a factor
# of roughly 3, but leads to qualitatively incorrect behavior.
class MutualDipole(hoomd.md.force._force):

    # Initialize the MutualDipole class
    # For group = A
    # group_tag array is of size of Ntotal: group_tag = [A1, A2, A3, A4, A5, B1, B2, B3, B4, B5]
    # conductivity array is of size of group size: conductivitiy = [A1, A2, A3, A4, A5]

    # radii[type_id] = radius of that HOOMD particle type
    # Example:
    # type id 0 -> 'small'
    # type id 1 -> 'large'
    # type id 2 -> 'wall'
    # radii = [a_small, a_large, a_wall]
    # radii = [1.0, 1.5, 2.0]
    # Even if certian type is not included in the plugin, should still list radii of all of the type

    def __init__(self, group, group_tag, conductivity, radii, field, gradient=[0., 0., 0.], xi = 0.5, errortol = 1e-3,
                 fileprefix = "", period = 0, constantdipoleflag = 0):

        hoomd.util.print_status_line();

        # initialize base class
        hoomd.md.force._force.__init__(self);

        # Set the cutoff radius for the cell and neighbor lists. This form
        # ensures the calculations are within the specified error tolerance.
        self.rcut = math.sqrt(-math.log(errortol))/xi;

        # enable the force
        self.enabled = True;

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            hoomd.context.msg.error("Sorry, we have not written CPU code for mutual dipole calculations. \n");
            raise RuntimeError('Error creating MutualDipole2');
        else:
            # Create a new neighbor list
            cl_MutualDipole = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_MutualDipole, "MutualDipole_cl");
            self.neighbor_list = _md.NeighborListGPUBinned(hoomd.context.current.system_definition, self.rcut, 0.4, cl_MutualDipole);
            self.neighbor_list.setEvery(1, True);
            hoomd.context.current.system.addCompute(self.neighbor_list, "MutualDipole_nlist");
            self.neighbor_list.countExclusions();

            # Add the new force to the system
            self.cpp_force = _MutualDipole.MutualDipole(hoomd.context.current.system_definition, group.cpp_group,
                                                        self.neighbor_list, group_tag, conductivity, radii, field, gradient, xi, errortol, fileprefix,
                                                        period, constantdipoleflag, hoomd.context.current.system.getCurrentTimeStep());
            hoomd.context.current.system.addCompute(self.cpp_force,self.force_name);

        # Set parameters for the dipole and force calculations
        self.cpp_force.SetParams();
    
        # Compute the dipoles and forces for the current time step so that the
        # electric/magnetic forces are used for the first update to particle positions.
        self.cpp_force.computeForces(hoomd.context.current.system.getCurrentTimeStep());

    # The integrator calls the update_coeffs function but there are no
    # coefficients to update, so this function does nothing
    def update_coeffs(self):
        pass

    # Update only the external field and gradient.  Faster than the update_parameters function.
    def update_field(self, field, gradient = [0, 0, 0]):
        self.cpp_force.UpdateField(field, gradient);

    # Update simulation parameters.  This is needed if any of the simulation
    # parameters change, including the volume fraction or shape of the simulation box.
    def update_parameters(self, group_tag, conductivity, field, gradient = [0., 0., 0.], fileprefix = "", period = 0, constantdipoleflag = 0):
        self.cpp_force.UpdateParameters(group_tag, field, conductivity, fileprefix, period, constantdipoleflag,
                                        hoomd.context.current.system.getCurrentTimeStep());
        self.cpp_force.SetParams();