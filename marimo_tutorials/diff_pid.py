import marimo

__generated_with = "0.6.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    # Makes JAX use the CPU, not GPU.
    jax.config.update('jax_platform_name', 'cpu')

    mo.md(r"""
    # PID Autotuning Using Differentiable Programming
    """)
    return jax, jnp, mo, np, plt


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
