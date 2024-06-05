"""
Classes for simulating PID controllers.
"""

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx


class PIDApprox(eqx.Module):
    kp: Array | float
    ki: Array | float
    kd: Array | float
    Tf: Array | float # For approximation of derivative TF; see https://www.cds.caltech.edu/~murray/courses/cds101/fa04/caltech/am04_ch8-3nov04.pdf

    def _numerator(self):
        kp = self.kp
        ki = self.ki
        kd = self.kd
        Tf = self.Tf
        return jnp.array([kp*Tf + kd, kp + ki*Tf, ki])

    def _denominator(self):
        Tf = self.Tf
        return jnp.array([Tf, 1, 0])

    def _A(self):


    def __call__(self, x):
        pass