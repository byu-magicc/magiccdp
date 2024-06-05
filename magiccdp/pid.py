"""
Classes and functions for simulating PID controllers on SISO plants.

Contributors: James Usevitch (james_usevitch@byu.edu)
              Cameron Stoker
              Tanner Osburn
"""

import jax
jax.config.update('jax_platform_name', 'cpu') # Sets the default device to CPU
import jax.numpy as jnp
from jax import Array
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
import optax
import matplotlib.pyplot as plt
from argparse import ArgumentParser


class PIDSystem(eqx.Module):
    """
    Sets up a PID controller for a SISO plant.
    Automatically converts from transfer function to control canonical
    state-space representation.

    Note that this class is a frozen dataclass since it inherits from
    `eqx.Module`. This means that all fields are immutable after initialization.
    """
    kp: Array | float
    ki: Array | float
    kd: Array | float
    dyn_num: list[float]
    dyn_denom: list[float]

    def __init__(self,*,
        kp: Array | float,
        ki: Array | float,
        kd: Array | float,
        dyn_num: list[float],
        dyn_denom: list[float]
    ):
        """
        Args:

            kp:    Proportional gain. Pass in as jnp.array([kp]) to make tunable, or as a float to freeze.
            ki:    Integral gain. See kp instructions for tuning / freezing.
            kd:    Derivative gain. See kp instructions for tuning / freezing.
            dyn_num: Numerator of the plant transfer function. Pass in as list of floats.
            dyn_denom: Denominator of the plant transfer function. Pass in as list of floats.
        
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dyn_num = dyn_num
        self.dyn_denom = dyn_denom


    def _num_denom(self):
        kp = self.kp
        ki = self.ki
        kd = self.kd

        pid_num = jnp.hstack([kd, kp, ki])
        pid_denom = jnp.array([1.0, 0])

        dyn_num = jnp.array(self.dyn_num)
        dyn_denom = jnp.array(self.dyn_denom)

        denom = jnp.polyadd(jnp.polymul(dyn_denom, pid_denom), jnp.polymul(dyn_num, pid_num))
        lead_coeff = denom[0]
        denom = denom/lead_coeff
        num = jnp.polymul(dyn_num, pid_num) / lead_coeff

        return num, denom
        

    def _statespace(self):
        """
        Returns the state-space representation of the PID controller.

        Should be able to handle any proper (or strictly proper) transfer function.
        """
        num, denom = self._num_denom()
        D, num = jnp.polydiv(num, denom)

        A = jnp.vstack((
            jnp.hstack((
                jnp.zeros((len(denom)-2, 1)),
                jnp.eye(len(denom)-2)
            )),
            -jnp.flip(denom[1:])
        ))

        B = jnp.hstack((
            jnp.zeros((1, len(denom)-2)),
            jnp.array([[1]])
        )).reshape(-1)

        C = jnp.flip(jnp.hstack((jnp.zeros((len(denom) - len(num)-1)), num)))

        return A, B, C, D


    def __call__(self, t, x, params):
        A, B, _, _ = self._statespace()

        return A @ x + B * params['ref']


def solve(system: PIDSystem, x0: Array, ref: float, t1=1.0, resolution=1000):
    terms = ODETerm(system)
    sol = diffeqsolve(
        terms=terms,
        solver=Tsit5(),
        t0=0.0,
        t1=t1,
        dt0=0.01,
        y0=x0,
        args={'ref': ref},
        saveat=SaveAt(ts=jnp.linspace(0.0, t1, resolution)),
        max_steps=10000,
        # stepsize_controller=PIDController(rtol=1e-5, atol=1e-5),
    )
    return sol


def make_loss(system, t1=1.0, resolution=100):

    @eqx.filter_value_and_grad
    def loss(system, x0, ref):
        sol = solve(system, x0, ref, t1=t1, resolution=resolution)
        _, _, C, D = system._statespace()
        y = (C @ sol.ys.T).T + D * ref
        # loss = jnp.sum((y - ref)**2) + jnp.sum(jnp.maximum(y - ref, jnp.zeros_like(y)))
        loss = jnp.sum((y - ref)**2) + jnp.max(y - ref)
        return loss

    return loss
    


def make_step(opt, loss_fn):

    @eqx.filter_jit
    def step(system, opt_state, x0, ref):
        value, grads = loss_fn(system, x0, ref)
        updates, opt_state = opt.update(grads, opt_state, system)
        system = eqx.apply_updates(system, updates)
        return value, system, opt_state

    return step



def clip_gains(system: PIDSystem):
    """
    Keeps the gains within the specified ranges below.
    Useful for, e.g., preventing gains from going negative.
    """
    kp = jnp.clip(system.kp, 0.01, 10.0)
    ki = jnp.clip(system.ki, 0.0, 30.0)
    kd = jnp.clip(system.kd, 0.0, 30.0)

    return PIDSystem(kp=kp, ki=ki, kd=kd, dyn_num=system.dyn_num, dyn_denom=system.dyn_denom)



def make_PIDSystem(kp, ki, kd):
    m = 0.5
    b = 0.01
    l = 0.3
    g = 9.8

    return PIDSystem(
        kp=jnp.array([kp]).reshape(-1),
        ki=jnp.array([ki]).reshape(-1),
        kd=jnp.array([kd]).reshape(-1),
        dyn_num=[3/(m*l**2)],
        dyn_denom=[1.0, 3*b/(m*l**2), 0.0]
    )



if __name__ == "__main__":

    T1 = 35.0
    RESOLUTION = 1000

    system = make_PIDSystem(0.18, 0.0, 0.095)

    ref = 1.0
    A, B, C, D = system._statespace()
    x0 = jnp.zeros(A.shape[0])
    # x0 = x0.at[1].set(0.1)

    sol = solve(system, x0, ref, t1=T1, resolution=RESOLUTION)

    y = (C @ sol.ys.T).T + D * ref


    plt.plot(sol.ts, y)
    plt.plot(sol.ts, jnp.ones_like(sol.ts) * ref)
    plt.show()


    if True:

        lr = 1e-4
        opt = optax.sgd(learning_rate=lr, momentum=0.9, nesterov=True)
        # opt = optax.adamw(learning_rate=lr)
        opt_state = opt.init(system)

        loss = make_loss(system, t1=T1, resolution=500)
        step_fn = make_step(opt, loss)

        # Gradient descent loop
        for ii in range(100):
            value, system, opt_state = step_fn(system, opt_state, x0, ref)
            system = clip_gains(system)
            if ii % 10 == 0:
                print(f"Loss at Step {ii}: {value}")
    
        sol = solve(system, x0, ref, t1=T1, resolution=RESOLUTION)
        _, _, C, D = system._statespace()
        y = (C @ sol.ys.T).T + D * ref

        plt.plot(sol.ts, y)
        plt.plot(sol.ts, jnp.ones_like(sol.ts) * ref)
        plt.show()

        # Use the terminal command `python -i pid.py` to inspect the final gains
        kp_final = system.kp
        ki_final = system.ki
        kd_final = system.kd

