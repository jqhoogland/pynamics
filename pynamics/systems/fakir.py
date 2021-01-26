"""
fakir bed dynamics, corresponding to
the movement of a particle in a two-dimensional complex
landscape with k unstable fixed points. More precisely,
we now consider a particle confined in a compact subset
of R2, with close to the origin a fixed number k of Gaussian
hills (corresponding to the presence of k

"""
from .double_well import BrownianMotion

class FakirBed(BrownianMotion):
    def __init__(self, n_wells, well_height_mean, well_height_var, well_width):
        """
        The phase space is the unit square with toroidal boundary conditions.

        The positions of the wells are uniformly distributed.

        The height of the wells is selected from a gaussian distribution with
        average ``well_height_mean`` and variance ``well_height_var``.

        The wells themselves are gaussians with variance ``well_width``
        """

        self.n_wells = n_wells
        self.well_height_mean = well_height_mean
        self.well_height_var = well_height_var
        self.well_width = well_width

        self.init_potential()

    def init_potential(self):
        pass
