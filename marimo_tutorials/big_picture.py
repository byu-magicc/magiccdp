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
    # # Nasty hack to get around Marimo's define-once limitations
    # key = [jax.random.PRNGKey(SEED)]

    # # Attempt 2
    # class JAXKey():
    #     def __init__(self, seed=42):
    #         self._key = jax.random.PRNGKey(seed)

    #     def split(self, size):
    #         _keys = jax.random.split(self._key, size+1)
    #         self._key = _keys[0]
    #         return _keys[1:]

    # key = JAXKey(SEED)

    # # Third time's the charm
    key_0 = jax.random.PRNGKey(SEED)
    return SEED, key_0


@app.cell
def __(mo):
    mo.md(r"""
    ## First steps: Interpolating Two Points

    Suppose we have two points. We would like to interpolate a line between them. The two points are plotted below.
    """)
    return


@app.cell
def __(jax, key_0, plt):
    # key.extend(jax.random.split(key.pop(), 3))
    key_1, fs_xkey, fs_ykey = jax.random.split(key_0, 3)
    fs_x = jax.random.uniform(fs_xkey, (2,))
    fs_y = jax.random.uniform(fs_ykey, (2,))

    fig_1, ax_1 = plt.subplots()
    ax_1.set_xlim((0,1))
    ax_1.set_ylim((0,1))
    ax_1.plot(fs_x, fs_y, "rx")
    return ax_1, fig_1, fs_x, fs_xkey, fs_y, fs_ykey, key_1


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    We can interpolate between the points using the equation for a line:

    $$
    f(x) = mx+b
    $$

    But what should we put as the values of $m$ and $b$? These two variables are called **parameters**. They define an entire family of lines. We specify exactly what line we want by choosing a specific value for $m$ and $b$.

    Using the sliders below, see if you can interpolate the two points by hand.
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    fs_m_slider = mo.ui.slider(
        start=-5.0,
        stop=5.0,
        step=1e-1,
        value=0.0,
        show_value=True,
        label="m Value: ",
    )

    fs_b_slider = mo.ui.slider(
        start=-5.0,
        stop=5.0,
        step=1e-1,
        value=0.0,
        show_value=True,
        label="b Value",
    )

    def fs_linear(x, m, b):
        return m*x + b

    fs_m_b_array = mo.ui.array([
        fs_m_slider,
        fs_b_slider,
    ])
    fs_m_b_array
    return fs_b_slider, fs_linear, fs_m_b_array, fs_m_slider


@app.cell(hide_code=True)
def __(fs_linear, fs_m_b_array, fs_x, fs_y, jnp, plt):
    fs_grid = jnp.linspace(0,1)

    fs_fig_2, fs_ax_2 = plt.subplots()
    fs_ax_2.set_xlim((0,1))
    fs_ax_2.set_ylim((0,1))
    fs_ax_2.plot(fs_x, fs_y, "rx")
    m, b = fs_m_b_array.value
    fs_ax_2.plot(fs_grid, fs_linear(fs_grid, m, b), "b-")
    return b, fs_ax_2, fs_fig_2, fs_grid, m


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
    fig_x_1, ax_x_1 = plt.subplots()
    ax_x_1.plot(grid, jnp.exp(slider_k_1.value*grid))

    # plt.plot(grid, jnp.exp(slider_k_1.value*grid))

    # # Interactive Below
    # fig1, ax1 = plt.subplots()
    # ax1.plot(grid, jnp.exp(slider_k_1.value*grid))
    # mo.mpl.interactive(ax1)
    return ax_x_1, fig_x_1


if __name__ == "__main__":
    app.run()
