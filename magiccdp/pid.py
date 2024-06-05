"""
Classes for simulating PID controllers.
"""

import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import jit, grad
import jax.numpy as jnp
from jax import Array
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
import optax
import matplotlib.pyplot as plt

import sys
if sys.flags.debug:
    import pdb


class PIDSystem(eqx.Module):
    """
    Simple PID controller for SISO system.
    """
    kp: Array | float
    ki: Array | float
    kd: Array | float
    # Tf: Array | float # For approximation of derivative TF; see https://www.cds.caltech.edu/~murray/courses/cds101/fa04/caltech/am04_ch8-3nov04.pdf
    dyn_num: list[float]
    dyn_denom: list[float]

    def __init__(self,*,
        kp: Array | float,
        ki: Array | float,
        kd: Array | float,
        # Tf: Array | float,
        dyn_num: list[float],
        dyn_denom: list[float]
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # self.Tf = Tf
        self.dyn_num = dyn_num
        self.dyn_denom = dyn_denom

    def _num_denom(self):
        kp = self.kp
        ki = self.ki
        kd = self.kd
        # Tf = self.Tf

        # pid_num = jnp.array([kp*Tf + kd, kp + ki*Tf, ki])
        # pid_denom = jnp.array([Tf, 1.0, 0])

        pid_num = jnp.array([kd, kp, ki])
        pid_denom = jnp.array([1.0, 0])

        dyn_num = jnp.array(self.dyn_num)
        dyn_denom = jnp.array(self.dyn_denom)

        denom = jnp.polyadd(jnp.polymul(dyn_denom, pid_denom), jnp.polymul(dyn_num, pid_num))
        lead_coeff = denom[0]
        denom = denom/lead_coeff
        num = jnp.polymul(dyn_num, pid_num) / lead_coeff

        polydiv = jnp.polydiv

        return num, denom
        

    def _statespace(self):
        """
        Returns the state-space representation of the PID controller.
        """
        num, denom = self._num_denom()
        D, num = jnp.polydiv(num, denom)

        # # DEBUG
        # vstack = jnp.vstack
        # hstack = jnp.hstack
        # zeros = jnp.zeros
        # eye = jnp.eye
        # flip = jnp.flip
        # # END DEBUG


        A = jnp.vstack((
            jnp.hstack((
                jnp.zeros((len(denom)-1, 1)),
                jnp.eye(len(denom)-1)
            )),
            -jnp.flip(denom)
        ))


        B = jnp.hstack((
            jnp.zeros((1, len(denom)-1)),
            jnp.array([[1]])
        )).reshape(-1)
        C = jnp.flip(jnp.hstack((jnp.zeros((len(denom) - len(num))), num)))

        return A, B, C, D


    def __call__(self, t, x, params):
        A, B, C, D = self._statespace()

        return A @ x + B * params['ref']


def solve(system: PIDSystem, x0: Array, ref: float, t1=1.0):
    terms = ODETerm(system)
    sol = diffeqsolve(
        terms=terms,
        solver=Tsit5(),
        t0=0.0,
        t1=t1,
        dt0=0.01,
        y0=x0,
        args={'ref': ref},
        saveat=SaveAt(ts=jnp.linspace(0.0, t1, 1000)),
        max_steps=10000,
        # stepsize_controller=PIDController(rtol=1e-5, atol=1e-5),
    )
    return sol


def make_loss(system):

    A, B, C, D = system._statespace()

    @eqx.filter_value_and_grad
    def loss(system, x0, ref):
        sol = solve(system, x0, ref)
        y = (C @ sol.ys.T).T + D * ref
        return jnp.sum((y - ref)**2)

    return loss
    

def make_step(opt, loss_fn):

    @eqx.filter_jit
    def step(system, opt_state, x0, ref):
        value, grads = loss_fn(system, x0, ref)
        updates, opt_state = opt.update(grads, opt_state)
        system = eqx.apply_updates(system, updates)
        return value, system, opt_state

    return step



if __name__ == "__main__":

    m = 0.5
    b = 0.01
    l = 0.3
    g = 9.8

    single_arm = PIDSystem(
        kp=0.01,
        ki=0.01,
        # kd=0.095,
        kd=0.1,
        # Tf=1.0,
        dyn_num=[3/(m*l**2)],
        dyn_denom=[1.0, 3*b/(m*l**2), 0.0]
    )

    ref = 1.0
    A, B, C, D = single_arm._statespace()
    x0 = jnp.zeros(A.shape[0])

    pdb.set_trace() if sys.flags.debug else None

    sol = solve(single_arm, x0, ref, t1=60.0)

    y = (C @ sol.ys.T).T + D * ref


    plt.plot(sol.ts, y)
    plt.plot(sol.ts, jnp.ones_like(sol.ts) * ref)
    plt.show()


    if False:

        lr = 1e-3
        opt = optax.sgd(learning_rate=lr)
        opt_state = opt.init(single_arm)

        loss = make_loss(single_arm)
        step_fn = make_step(opt, loss)


        pdb.set_trace() if sys.flags.debug else None

        for ii in range(100):
            value, single_arm, opt_state = step_fn(single_arm, opt_state, x0, ref)
            print(f"Loss at Step {ii}: {value}")
    

