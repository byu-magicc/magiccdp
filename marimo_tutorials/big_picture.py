import marimo

__generated_with = "0.5.2"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    mo.md(r"""
    # Introduction to Differentiable Programming
    """)
    return jax, jnp, mo, plt


@app.cell
def __(jax):
    # Makes JAX use the CPU, not GPU.
    jax.config.update('jax_platform_name', 'cpu')

    # Sets the random seed and initializes a pseudorandom key.
    SEED = 42
    # Nasty hack to get around Marimo's define-once limitations
    # key = [jax.random.PRNGKey(SEED)]

    class JAXKey():
        def __init__(self, seed=42):
            self._key = jax.random.PRNGKey(seed)

        def split(self, size):
            _keys = jax.random.split(self._key, size+1)
            self._key = _keys[0]
            return _keys[1:]

    key = JAXKey(SEED)
    return JAXKey, SEED, key


@app.cell
def __(jax, key, mo, plt):
    mo.md(r"""
    ## First steps: Interpolating Two Points

    Suppose we have two points. We would like to interpolate a line between them. The two points are plotted below.
    """)

    # key.extend(jax.random.split(key.pop(), 3))
    fs_xkey, fs_ykey = key.split(2)
    fs_x = jax.random.uniform(fs_xkey, (2,))
    fs_y = jax.random.uniform(fs_ykey, (2,))

    plt.plot(fs_x, fs_y, "rx")
    return fs_x, fs_xkey, fs_y, fs_ykey


@app.cell
def __(mo):
    mo.md(r"""
    ---

    ## Scratch Work
    """)
    return


@app.cell
def __(jnp, mo):
    slider_k_1 = mo.ui.slider(
        start=-10.0,
        stop=10.0,
        step=1e-1,
        value=0.0,
    )

    grid = jnp.linspace(0,1)

    slider_k_1
    return grid, slider_k_1


@app.cell
def __(grid, jnp, plt, slider_k_1):
    plt.plot(grid, jnp.exp(slider_k_1.value*grid))

    # # Interactive Below
    # fig1, ax1 = plt.subplots()
    # ax1.plot(grid, jnp.exp(slider_k_1.value*grid))
    # mo.mpl.interactive(ax1)
    return


if __name__ == "__main__":
    app.run()
