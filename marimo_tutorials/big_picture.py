import marimo

__generated_with = "0.5.2"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    mo.md(r"""
    # Introduction to Differentiable Programming
    """)
    return jax, jnp, mo, np, plt


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


@app.cell
def __(mo):
    mo.md(r"""
    We can interpolate between the points using the equation for a line:

    $$
    f(x) = mx+b
    $$

    But what should we put as the values of $m$ and $b$? These two variables are called **parameters**. They define an entire family of lines. We specify exactly what line we want by choosing a specific value for $m$ and $b$.

    Intuitively, you can think of $m$ and $b$ as "control knobs" that change the shape of our function. Changing the control knobs morphs the function into a different form.

    Using the sliders below, see if you can interpolate the two points by hand.
    """)
    return


@app.cell
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

    # # This array is an alternate way to show both the m and b values, but vstack / hstack is cleaner.
    # fs_m_b_array = mo.ui.array([
    #     fs_m_slider,
    #     fs_b_slider,
    # ])
    # fs_m_b_array

    mo.vstack([
        fs_m_slider,
        fs_b_slider
    ])
    return fs_b_slider, fs_linear, fs_m_slider


@app.cell
def __(fs_b_slider, fs_linear, fs_m_slider, fs_x, fs_y, jnp, plt):
    fs_grid = jnp.linspace(0,1)

    fs_fig_2, fs_ax_2 = plt.subplots()
    fs_ax_2.set_xlim((0,1))
    fs_ax_2.set_ylim((0,1))
    fs_ax_2.plot(fs_x, fs_y, "rx")
    m, b = (fs_m_slider.value, fs_b_slider.value)
    fs_ax_2.plot(fs_grid, fs_linear(fs_grid, m, b), "b-")
    return b, fs_ax_2, fs_fig_2, fs_grid, m


@app.cell
def __(fs_b_slider, fs_linear, fs_m_slider, fs_x, fs_y, jnp, mo):
    def fs_error(m,b):
        return jnp.sum((fs_y - fs_linear(fs_x, m, b))**2)

    mo.vstack([
        mo.md(r"""
        Tuning this by hand is a bit tricky. We also don't have a way to quantify how "wrong" we are. Let's define an error function that tracks the error in our model.

        Given the x-coordinates $x_1$ and $x_2$ of our two points, 
        
        * The _true_ values of $y$ are $y_1$ and $y_2$. 
        * The _predicted_ values from our linear model are $f(x_1)$ and $f(x_2)$.

        Let's use the following error function:

        $$
        E(m, b, x_1, x_2, y_1, y_2) = (y_1 - f(x_1))^2 + (y_2 - f(x_2))^2.
        $$

        Try playing with the sliders again, and see how low you can make the error.
        """),
        mo.vstack([
            fs_m_slider,
            fs_b_slider,
        ]),
        mo.md(f"Error: {fs_error(fs_m_slider.value, fs_b_slider.value)}"),
        mo.md("Tip: The sliders in this cell and the sliders above the plot are linked. Use either one you'd like!").callout(kind="info")
    ])
    return fs_error,


@app.cell
def __(mo):
    mo.hstack([
        mo.md(r"""
        Tuning by hand is all fine and good, but we as researchers invented computers for a very important reason: we are lazy. ("Efficient" is probably a more elegant term.) Why waste our time on this if we can get a computer to do it for us?

        The question is, how do we get a computer to do this? Well if you look closely at our error function above, it has the important property that:

        * $E(\cdots) = 0$ if and only if our model is correct
        * $E(\cdots) > 0$ if and only if our model has some error

        If our error $E(\cdots)$ is non-zero, we simply need to make the error decrease. If we can decrease the error to exactly zero, we're done.

        The "control knobs" the computer can turn are $m$ and $b$. Notice that $E(\cdots)$ is a function of $m$ and $b$:

        $$
        E(m,b,x_1,x_2,y_1,y_2) = (y_1 - (m x_1+b)))^2 + (y_2 - (m x_2+b))^2
        $$

        So what direction should the computer turn $m$ and $b$ to make the error go down?
        """)
    ])
    return


@app.cell
def __(mo):
    mo.vstack([
        mo.md("## The Power of Derivatives"),
        mo.image("../images/vader_derivatives.jpg"),
        mo.md(r"""
        Remember derivatives from Calculus? Here's the classic definition that you all know and hate:

        $$
        \frac{d}{dt} f(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}
        $$

        But there are two problems here:

        1. What do derivatives actually mean?
        2. We have two variables $m$ and $b$, not just one ($x$).

        Let's talk about point \#1 first. A derivative is _the instantaneous slope of a function at a point_. If we approximated a nonlinear (or linear) function as a linear (straight line) model, the derivative of the original function would be the slope of the approximate model.

        If we take a very small step, the amount the function changes is approximately equal to the derivative times the step size:

        $$
        \Delta f(x) \approx \frac{df(x)}{dx} \Delta x 
        $$

        If we step in the direction of the derivative, our function value goes _up_ (if the step size is small enough).

        If we step in the direction of the _negative_ of the derivative, our function value goes _down_ (if the step size is small enough).

        You can see this by playing with the sliders below.
        """)
    ])

    return


@app.cell(hide_code=True)
def __(mo, np):
    pd_dv_x = mo.ui.slider(
        start=-np.pi,
        stop=np.pi,
        step=1e-1,
        value=0.0,
        show_value=True,
        label="Derivative Point: ",
    )

    pd_dv_step = mo.ui.slider(
        start=0.0,
        stop=4.0,
        step=1e-1,
        value=0.0,
        show_value=True,
        label="Step size: ",
    )

    pd_dv_posneg = mo.ui.radio(
        options=["Positive", "Negative"],
        value="Positive",
        label="Gradient step direction"
    )

    pd_grid = np.linspace(-np.pi,np.pi)

    mo.vstack([
        pd_dv_x,
        pd_dv_step,
        pd_dv_posneg
    ])

    return pd_dv_posneg, pd_dv_step, pd_dv_x, pd_grid


@app.cell
def __(fs_linear, np, pd_dv_posneg, pd_dv_step, pd_dv_x, pd_grid, plt):
    # Showing example for sin(x)
    pd_fig_1, pd_ax_1 = plt.subplots()

    pd_ax_1.set_xlim((-np.pi,np.pi))
    pd_ax_1.set_ylim((-1.1, 1.1))
    pd_ax_1.axhline(0, color='black')
    pd_ax_1.axvline(0, color='black')
    pd_ax_1.plot(pd_grid, np.sin(pd_grid), "b-")
    pd_ax_1.plot([pd_dv_x.value, pd_dv_x.value], [0, np.sin(pd_dv_x.value)], "ro-")
    # pd_ax_1.plot([pd_dv_x.value, pd_dv_x.value + pd_dv_step.value], [0,0], "gs-")
    # Derivative
    pd_ax_1.plot(pd_grid + pd_dv_x.value, np.sin(pd_dv_x.value) + fs_linear(pd_grid, np.cos(pd_dv_x.value), 0), "k--")

    grad_val = np.cos(pd_dv_x.value)

    if pd_dv_posneg.value.lower() == "positive":
        pd_new_x = pd_dv_x.value + pd_dv_step.value*grad_val
    else:
        pd_new_x = pd_dv_x.value - pd_dv_step.value*grad_val

    pd_x_diff = pd_new_x - pd_dv_x.value

    pd_y = np.sin(pd_dv_x.value)
    pd_new_y = np.sin(pd_dv_x.value + pd_x_diff)
    pd_pred_new_y = np.sin(pd_dv_x.value) + np.cos(pd_dv_x.value)*pd_x_diff

    # Difference Line
    pd_ax_1.plot(
        [pd_dv_x.value, pd_new_x],
        [np.sin(pd_dv_x.value), np.sin(pd_dv_x.value)],
        "k--"
    )

    # Predicted difference
    pd_ax_1.plot(
        [pd_new_x, pd_new_x],
        [pd_y, pd_pred_new_y],
        "gx--"
    )
    # Actual difference
    pd_ax_1.plot(
        [pd_new_x, pd_new_x],
        [pd_y, pd_new_y],
        "r+--"
    )
    return (
        grad_val,
        pd_ax_1,
        pd_fig_1,
        pd_new_x,
        pd_new_y,
        pd_pred_new_y,
        pd_x_diff,
        pd_y,
    )


@app.cell
def __(mo):
    mo.vstack([
        mo.md(r"""
        As you might have guessed, the derivative can be used to make our error $E(\cdots)$ converge to zero. The "negative derivative" direction is the direction we need to move our parameters to make the error converge to zero.

        But we have two parameters. How do we take the derivative with respect to them both?

        Let's define a vector containing our parameters:

        $$
        \vec{\theta} = \begin{bmatrix}m \\ b \end{bmatrix}
        $$

        **Gradients** are a generalization of derivatives to multiple dimensions.

        

        """)
    ])
    return


@app.cell(hide_code=True)
def __(mo, np):
    # Gradient function: x^T x

    pd_grad_theta = mo.ui.slider(
        start=0,
        stop=2*np.pi,
        step=1e-1,
        value=0.0,
        show_value=True,
        label="Directional derivative angle (radians)",
    )

    pd_unit = np.array([1,0])

    def pd_xyz(theta, q):
        x = q*np.cos(theta)
        y = q*np.sin(theta)
        z = np.array([x,y]) @ np.array([x,y])
        return x,y,z

    # def pd_grad

    pd_grad_x_pos = mo.ui.slider(
        start=-1,
        stop=1,
        step=1e-1,
        value=0.0,
        show_value=True,
        label="x function point",
    )

    pd_grad_y_pos = mo.ui.slider(
        start=-1,
        stop=1,
        step=1e-1,
        value=0.0,
        show_value=True,
        label="y function point",
    )

    pd_grad_grad_check = mo.ui.checkbox(
        value=False,
        label="Show gradient",
    )

    mo.vstack([
        mo.md("Position of function evaluation"),
        pd_grad_x_pos,
        pd_grad_y_pos,
        mo.md("Directional Derivative"),
        pd_grad_theta,
        pd_grad_grad_check,
    ]) 
    return (
        pd_grad_grad_check,
        pd_grad_theta,
        pd_grad_x_pos,
        pd_grad_y_pos,
        pd_unit,
        pd_xyz,
    )


@app.cell(hide_code=True)
def __(
    np,
    pd_grad_grad_check,
    pd_grad_theta,
    pd_grad_x_pos,
    pd_grad_y_pos,
    plt,
):
    pd_mesh_size = 0.05
    pd_X = np.arange(-1.1,1.1,pd_mesh_size)
    pd_Y = np.arange(-1.1,1.1,pd_mesh_size)

    pd_X, pd_Y = np.meshgrid(pd_X, pd_Y)

    pd_Z = pd_X**2 + pd_Y**2

    pd_fig_2, pd_ax_2 = plt.subplots(subplot_kw={"projection": "3d"})

    pd_ax_2.set_xlim((-1,1))
    pd_ax_2.set_ylim((-1,1))
    pd_ax_2.set_zlim((0,2))
    pd_ax_2.set_xlabel("X")
    pd_ax_2.set_ylabel("Y")

    pd_ax_2.plot_surface(pd_X, pd_Y, pd_Z, antialiased=False, alpha=0.2)

    # Plot directional derivative
    pd_ax_2.plot(
        [-np.cos(pd_grad_theta.value) + pd_grad_x_pos.value, np.cos(pd_grad_theta.value) + pd_grad_x_pos.value],
        [-np.sin(pd_grad_theta.value) + pd_grad_y_pos.value, np.sin(pd_grad_theta.value) + pd_grad_y_pos.value],
        [0,0],
        "b-"
    )

    # Plot function evaluation point
    pd_ax_2.plot(
        [pd_grad_x_pos.value,pd_grad_x_pos.value],
        [pd_grad_y_pos.value,pd_grad_y_pos.value],
        [0, pd_grad_x_pos.value**2 + pd_grad_y_pos.value**2],
        "ro--"
    )

    # Plot directional "slice"
    pd_grid_2 = np.linspace(-2,2)
    pd_X_line = np.cos(pd_grad_theta.value)*pd_grid_2 + pd_grad_x_pos.value
    pd_Y_line = np.sin(pd_grad_theta.value)*pd_grid_2 + pd_grad_y_pos.value
    pd_Z_line = pd_X_line**2 + pd_Y_line**2

    pd_ax_2.plot(
        pd_X_line,
        pd_Y_line,
        pd_Z_line,
        "r-"
    )

    # Plot directional derivative
    pd_endpoints = np.array([-0.5,0.5])
    pd_X_line_2 = np.cos(pd_grad_theta.value)*pd_endpoints + pd_grad_x_pos.value
    pd_Y_line_2 = np.sin(pd_grad_theta.value)*pd_endpoints + pd_grad_y_pos.value
    pd_grad_Z_line = (pd_grad_x_pos.value**2 + pd_grad_y_pos.value**2) + 2*pd_grad_x_pos.value*(pd_X_line_2 - pd_grad_x_pos.value) + 2*pd_grad_y_pos.value*(pd_Y_line_2 - pd_grad_y_pos.value)
    pd_ax_2.plot(
        pd_X_line_2,
        pd_Y_line_2,
        pd_grad_Z_line,
        "k--",
    )

    pd_ax_2.plot(
        [pd_X_line_2[0], pd_X_line_2[0]],
        [pd_Y_line_2[0], pd_Y_line_2[0]],
        [0, pd_grad_Z_line[0]],
        "k--",
    )
    pd_ax_2.plot(
        [pd_X_line_2[1], pd_X_line_2[1]],
        [pd_Y_line_2[1], pd_Y_line_2[1]],
        [0, pd_grad_Z_line[1]],
        "k--",
    )

    if pd_grad_grad_check.value:
        # Positive gradient
        pd_grad_angle = np.arctan2(pd_grad_y_pos.value, pd_grad_x_pos.value)
        pd_ax_2.plot(
            [pd_grad_x_pos.value, pd_grad_x_pos.value + np.cos(pd_grad_angle)],
            [pd_grad_y_pos.value, pd_grad_y_pos.value + np.sin(pd_grad_angle)],
            [0,0],
            "g-"
        )

    pd_fig_2

    return (
        pd_X,
        pd_X_line,
        pd_X_line_2,
        pd_Y,
        pd_Y_line,
        pd_Y_line_2,
        pd_Z,
        pd_Z_line,
        pd_ax_2,
        pd_endpoints,
        pd_fig_2,
        pd_grad_Z_line,
        pd_grad_angle,
        pd_grid_2,
        pd_mesh_size,
    )


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
