"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ...pypicongpu import species

import picmistandard

import typeguard
import typing
import math

"""
note on rms_velocity:
---------------------
The rms_velocity is converted to a temperature in keV. This conversion requires the mass of the species to be known,
which is not the case inside the picmi density distribution.

As an abstraction, **every** PICMI density distribution implements `picongpu_get_rms_velocity_si()` which returns a
tuple (float, float, float) with the rms_velocity per axis in SI units (m/s).

In case the density profile does not have an rms_velocity, this method **MUST** return (0, 0, 0), which is translated to
"no temperature initialization" by the owning species.

note on drift:
--------------
The drift ("velocity") is represented using either directed_velocity or centroid_velocity (v, gamma*v respectively) and
for the pypicongpu representation stored in a separate object (Drift).

To accommodate that, this separate Drift object can be requested by the method get_picongpu_drift(). In case of no drift,
this method returns None.
"""


@typeguard.typechecked
class GaussianBunchDistribution(picmistandard.PICMI_GaussianBunchDistribution):
    def picongpu_get_rms_velocity_si(self) -> typing.Tuple[float, float, float]:
        return tuple(self.rms_velocity)

    def get_as_pypicongpu(self) -> species.operation.densityprofile.DensityProfile:
        # @todo respect boundaries, Brian Marre, 2023
        profile = object()
        profile.lower_bound = (-math.inf, -math.inf, -math.inf)
        profile.upper_bound = (math.inf, math.inf, math.inf)
        profile.rms_bunch_size_si = self.rms_bunch_size
        profile.centroid_position_si = tuple(self.centroid_position)

        assert 0 != self.rms_bunch_size, "rms bunch size must not be zero"

        profile.max_density_si = self.n_physical_particles / ((2 * math.pi * self.rms_bunch_size**2) ** 1.5)

        return profile

    def get_picongpu_drift(self) -> typing.Optional[species.operation.momentum.Drift]:
        """
        Get drift for pypicongpu
        :return: pypicongpu drift object or None
        """
        if [0, 0, 0] == self.centroid_velocity:
            return None

        drift = species.operation.momentum.Drift()
        drift.fill_from_gamma_velocity(tuple(self.centroid_velocity))
        return drift
