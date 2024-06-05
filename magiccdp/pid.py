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
    dyn_num: list[float]
    dyn_denom: list[float]

    def __init__(self,*,
        kp: Array | float,
        ki: Array | float,
        kd: Array | float,
        Tf: Array | float,
        dyn_num: list[float],
        dyn_denom: list[float]
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.Tf = Tf
        self.dyn_num = dyn_num
        self.dyn_denom = dyn_denom

    def _num_denom(self):
        kp = self.kp
        ki = self.ki
        kd = self.kd
        Tf = self.Tf

        pid_num = jnp.array([kp*Tf + kd, kp + ki*Tf, ki])
        pid_denom = jnp.array([Tf, 1.0, 0])

        denom = jnp.polymul(pid_denom, self.dyn_denom)
        lead_coeff = denom[0]
        denom = denom/lead_coeff
        num = jnp.polymul(pid_num, self.dyn_num) / lead_coeff

        return num, denom
        

    def _statespace(self):
        """
        Returns the state-space representation of the PID controller.
        """
        num, denom = self._num_denom()
        D, num = jnp.polydiv(num, denom)

        A = jnp.vstack(
            jnp.array([0.0, 1.0]),
            -jnp.flip(denom)
        )

        B = jnp.array([[0.0, 1.0]]).T
        C = jnp.flip(num)

        return A, B, C, D


    def __call__(self, x, params):
        A, B, C, D = self._statespace()

        err = params['ref'] - x

        return A @ x + B @ err