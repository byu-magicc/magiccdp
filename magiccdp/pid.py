"""
Classes for simulating PID controllers.
"""

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx


class PIDController(eqx.Module):
    """
    Simple PID controller for SISO system.
    """
    kp: Array | float
    ki: Array | float
    kd: Array | float
    Tf: Array | float # For approximation of derivative TF; see https://www.cds.caltech.edu/~murray/courses/cds101/fa04/caltech/am04_ch8-3nov04.pdf
    # dyn_num: list[float]
    # dyn_denom: list[float]

    def _numerator(self):
        kp = self.kp
        ki = self.ki
        kd = self.kd
        Tf = self.Tf
        return jnp.array([kp*Tf + kd, kp + ki*Tf, ki])/Tf

    def _denominator(self):
        Tf = self.Tf
        return jnp.array([1, 1/Tf, 0])

    def _statespace(self):
        """
        Returns the state-space representation of the PID controller.
        """
        denom = self._denominator()
        num = self._numerator()
        D, num = jnp.polydiv(num, denom)

        A = jnp.vstack(
            jnp.array([0.0, 1.0]),
            -jnp.flip(denom)
        )

        B = jnp.array([[0.0, 1.0]]).T
        C = jnp.flip(num)

        return A, B, C, D


    def __call__(self, x):
        pass