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
    mo.md(
        r"""
        ## First steps: Interpolating Two Points

        Suppose we have two points. We would like to interpolate a line between them. The two points are plotted below.
        """
    )
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
    mo.md(
        r"""
        We can interpolate between the points using the equation for a line:

        $$
        f(x) = mx+b
        $$

        But what should we put as the values of $m$ and $b$? These two variables are called **parameters**. They define an entire family of lines. We specify exactly what line we want by choosing a specific value for $m$ and $b$.

        Intuitively, you can think of $m$ and $b$ as "control knobs" that change the shape of our function. Changing the control knobs morphs the function into a different form.

        Using the sliders below, see if you can interpolate the two points by hand.
        """
    )
    return


@app.cell
def __(mo):
    fs_m_slider = mo.ui.slider(
        start=-5.0,
        stop=5.0,
        step=1e-2,
        value=0.0,
        show_value=True,
        label="m Value: ",
    )

    fs_b_slider = mo.ui.slider(
        start=-5.0,
        stop=5.0,
        step=1e-2,
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
        mo.md("## Derivatives, Gradients, and Jacobians"),
        mo.image("../images/vader_derivatives.jpg"),
        mo.md(r"""
        Remember derivatives from Calculus? Here's the classic definition that you all know and hate:

        $$
        \frac{d}{dx} f(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}
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

        Notice that our error function can be written $E(\vec{\theta}, \cdots)$.

        **Directional derivatives** are a generalization of derivatives to higher dimensions.

        **Gradients** are a special case of the directional derivative. Loosely speaking, the gradient at a point $\theta$ is the directional derivative with the steepest slope.

        Use the sliders below to interact with directional derivaties for the function $f(x) = x^T x$. Use the x, y sliders to change the point at which the directional derivative is evaluated. Use the angle slider to change the directional derivative.
        Use the checkbox to turn on the gradient, and compare the slopes of directional derivatives to that of the gradient.

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
    mo.md(
        r"""
        Gradients can be computed by calculating the directional derivatives of each component of a multivariable function, and then putting those directional derivatives into a matrix.

        We _could_ do this by hand. Or, we can be more pragmatic and use a programming language that supports taking gradients "automagically".

        Enter JAX. JAX is essentially Numpy with three superpowers:

        * Automatic Differentiation (it can take gradients for you)
        * GPU support (to speed up super large matrix multiplications and much more)
        * JIT compilation (so that your Python code runs fast like C++ instead of slow like....Python)

        It also has one Kryptonite: it is (largely) a functional programming language. But we'll get to that later.

        Check out the Python code below to practice taking gradients / derivatives with JAX.
        """
    )
    return


@app.cell
def __(jax, jnp):
    # JAX matrices are similar to Numpy
    x = jnp.array([1.0, 2.0])
    A = jnp.array([[3.0, 4.0],[5.0, 6.0]])
    print(f"Ax = {A @ x}")

    # Define a function
    def func1(x):
        return x @ x

    # Create a new function that calculates the gradient at any point x
    grad_func1 = jax.grad(func1)

    # Calculate the gradient at different points
    print(f"Gradient at [0,0]: {grad_func1(jnp.array([0.0, 0.0]))}")
    print(f"Gradient at [2,3]: {grad_func1(jnp.array([2.0, 3.0]))}")
    print(f"Gradient at x: {grad_func1(x)}")

    # Sanity check: We know the gradient should be 2*x. Are we getting the right values out?
    return A, func1, grad_func1, x


@app.cell
def __(func1, jax, jnp):
    # We can nest functions arbitrarily and take gradients.
    def func2(x):
        return jnp.linalg.norm(jnp.exp(-jnp.sum(jnp.sin(x)) + x**(1/3)))

    grad_func2 = jax.grad(func2)

    print(f"Gradient of func2: {grad_func2(jnp.array([10.0, 20.0, 30.0]))}")


    # We can nest our own custom functions!

    func3 = lambda x: func1(func2(x)*jnp.array([1,2,3.0]))
    grad_func3 = jax.grad(func3)

    print(f"Gradient of func3: {grad_func3(jnp.array([10.,20.0,30.0]))}")


    # If a function has multiple arguments, we can specify which arguments we want the gradient with respect to using the `argnums` keyword.

    def func4(x,y):
        return jnp.sin(x) @ jnp.cos(y)

    grad_func4_x = jax.grad(func4, argnums=0)

    print(f"Gradient of func4 w.r.t. x: {grad_func4_x(jnp.array([1,2.0]), jnp.array([3.0, 4.0]))}")

    # Try it! Can you compute the gradient of func4 with respect to y?

    # PUT YOUR CODE HERE
    return func2, func3, func4, grad_func2, grad_func3, grad_func4_x


@app.cell
def __(mo):
    mo.md(r"Try computing the gradient of the following function using the vector $[1.0, 2.0, 3.0]$. What happens?")
    return


@app.cell
def __():
    def func5(x):
        return 10*x

    # Your code here!
    return func5,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        An error occurred! JAX should have output something similar to the following:

        ```python
        TypeError: Gradient only defined for scalar-output functions. Output had shape: (3,).
        ```

        The problem is that gradients are only defined for functions with a _scalar_ output. The function above had a _vector_ output. What does it mean to take the derivative of a vector?

        Gradients are a special case of something called **Jacobian matrices**. If your function has $n$ output entries, a Jacobian essentially takes the gradient of each entry and stacks all the gradients into a matrix.

        See the following code cells for computing the Jacobian with JAX!
        """
    )
    return


@app.cell
def __(func5, jax, jnp):
    # Use jax.jacobian instead of jax.grad to compute Jacobian matrices

    jac_func5 = jax.jacobian(func5)
    print(f"Jacobian of func5: \n{jac_func5(jnp.array([1.0, 2.0, 3.0]))}")



    # Just like jax.grad, we can compute jacobians with respect to specific arguments. Just use the `argnums` keyword argument.
    # For example, let's multiply a matrix by a vector!
    def func6(W,x):
        return W @ x

    # Try it: Compute the Jacobian of func6 with respect to x!

    # Try it: Compute the Jacobian of func6 with respect to the matrix W!
    return func6, jac_func5


@app.cell
def __(mo):
    mo.vstack([
        mo.md(r"""
    JAX can handle all sorts of derivatives, gradients, Jacobians, and beyond. For example:

    * Jacobian of a scalar w.r.t. a scalar (derivative)
    * Jacobian of a vector w.r.t. a vector
    * Jacobian of a matrix w.r.t. a matrix
    * Jacobian of a tensor (multi-dimensional array) w.r.t. a tensor
    * Jacobian of a vector w.r.t. a matrix
    * Jacobian of a scalar w.r.t. a tensor
    * ...and so on.

    In the next cell, try defining some crazy functions and taking the Jacobian!
    """),
        mo.md("Tip: Marimo requires all variables to have unique names. If you run into problems, try appending your favorite superhero's name to the front of the variable name.").callout(kind="info")
    ])
    return


@app.cell
def __():
    # Go crazy :)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Back To Our Problem

        Now that we know how to compute gradients, let's compute the gradient for our error function. The function definition is given below.
        """
    )
    return


@app.cell
def __(jnp):
    def E(theta, x, y):
        m, b = theta
        return jnp.sum((y - (m*x+b))**2)

    # Try it! Define the gradient function below.
    return E,


@app.cell
def __(mo):
    mo.vstack([
        mo.md(r"""
    From above, we know that the _negative_ of the gradient is a **descent direction**. If we take small steps in that direction, our function will decrease.

    This leads us to the strategy of **gradient descent** (GD). This is an iterative algorithm that is used to find the _minimum_ of functions. The overall process is:

    1. Compute the gradient direction $p_0 = \nabla_\theta E(theta_0, \vec{x}, \vec{y}) at our starting parameter value $\theta_0$
    2. Take a small step in the direction of the negative gradient:

        $$
        \theta_1 = \theta_0 - \eta p_0
        $$

        (The value $\eta$ is explained below.)

    3. Repeat steps 1-2 by re-computing our gradient and taking another step:

        $$
        \begin{align}
        p_{k} &= \nabla_\theta E(\theta_k, \cdots),\\
        \theta_{k+1} &= \theta_k - \eta p_{k}
        \end{align}
        $$

    4. Terminate when the difference between consecutive $\theta_{k+1}$ and $\theta{k}$ values becomes "small enough"

    The value $\eta > 0$ is called the **step size**. In essence this determines how large of a step we take. Remember that we don't want to take too large of steps, or else the function might change differently than we expect! 


    """),
        mo.md(r"**Exercise**: Go back to the sin function above and try to find a situation where stepping too far in the negative gradient direction might actually _increase_ your function value!").callout(kind="info"),
        mo.md(r"""
        Below is an example of gradient descent. You can use the sliders to set the $m$ and $b$ values, and then click the `Descend!!` button to take a gradient descent step. To make it converge faster, you can set the number of steps taken every time you click the button. Watch as the line converges to interpolating the two points!
        """)
    ])
    return


@app.cell
def __(mo):
    get_gd_m_slider, set_gd_m_slider = mo.state(0.0)
    get_gd_b_slider, set_gd_b_slider = mo.state(0.0)
    get_gd_step, set_gd_step = mo.state(0.1)
    return (
        get_gd_b_slider,
        get_gd_m_slider,
        get_gd_step,
        set_gd_b_slider,
        set_gd_m_slider,
        set_gd_step,
    )


@app.cell
def __(
    get_gd_b_slider,
    get_gd_m_slider,
    get_gd_step,
    mo,
    set_gd_b_slider,
    set_gd_m_slider,
    set_gd_step,
):
    gd_step_size = mo.ui.slider(
        start=0.01,
        stop=0.3,
        step=1e-2,
        value=get_gd_step(),
        on_change=set_gd_step,
        label="Step size: ",
        show_value=True,
    )

    gd_m_slider = mo.ui.slider(
        start=-5.0,
        stop=5.0,
        step=1e-1,
        value=get_gd_m_slider(),
        on_change=set_gd_m_slider,
        label="m value (gradient descent): ",
        show_value=True,
    )

    gd_b_slider = mo.ui.slider(
        start=-5.0,
        stop=5.0,
        step=1e-1,
        value=get_gd_b_slider(),
        on_change=set_gd_b_slider,
        label="b value (gradient descent): ",
        show_value=True,
    )
    return gd_b_slider, gd_m_slider, gd_step_size


@app.cell
def __(mo):
    gd_num_steps = mo.ui.number(
        start=1,
        stop=1000,
        value=1,
        step=1.0,
        label="Number of steps to take (for each button click): "
    )
    return gd_num_steps,


@app.cell
def __(
    E,
    fs_x,
    fs_y,
    gd_b_slider,
    gd_m_slider,
    gd_num_steps,
    gd_step_size,
    get_gd_b_slider,
    get_gd_m_slider,
    get_gd_step,
    jax,
    jnp,
    mo,
    set_gd_b_slider,
    set_gd_m_slider,
):
    grad_E = jax.grad(E,argnums=0)

    def gd_update_m_b(v):
        for ii in range(gd_num_steps.value):
            _m = get_gd_m_slider()
            _b = get_gd_b_slider()
            _eta = get_gd_step()
            _theta = jnp.array([_m, _b])
            _gd_gradient = grad_E(_theta, fs_x, fs_y)
            set_gd_m_slider(_m - _eta*float(_gd_gradient[0]))
            set_gd_b_slider(_b - _eta*float(_gd_gradient[1]))

        return (1-v)

    def gd_refresh_m_b(v):
        set_gd_b_slider(get_gd_b_slider())
        set_gd_m_slider(get_gd_m_slider())



    gd_m_step_button = mo.ui.button(
        value=0,
        label="Descend!!",
        on_click=gd_update_m_b,
        on_change=gd_refresh_m_b,
    )

    # mo.ui.slider(
    #     start=-5.0,
    #     stop=5.0,
    #     step=1e-1,
    #     value=0.0,
    #     show_value=True,
    #     label="m Value: ",
    # )

    mo.vstack([
        gd_m_slider,
        gd_b_slider,
        gd_step_size,
        gd_num_steps,
        gd_m_step_button,
        mo.md(f"Error value: {float(E(jnp.array([get_gd_m_slider(), get_gd_b_slider()]), fs_x, fs_y))}")
    ])
    return gd_m_step_button, gd_refresh_m_b, gd_update_m_b, grad_E


@app.cell(hide_code=True)
def __(
    fs_grid,
    fs_linear,
    fs_x,
    fs_y,
    get_gd_b_slider,
    get_gd_m_slider,
    plt,
):
    gd_fig_1, gd_ax_1 = plt.subplots()
    gd_ax_1.set_xlim((0,1))
    gd_ax_1.set_ylim((0,1))
    gd_ax_1.plot(fs_x, fs_y, "rx")
    gd_m, gd_b = (get_gd_m_slider(), get_gd_b_slider())
    gd_ax_1.plot(fs_grid, fs_linear(fs_grid, gd_m, gd_b), "b-")
    return gd_ax_1, gd_b, gd_fig_1, gd_m


@app.cell
def __(mo):
    mo.md(
        r"""
        **Success!!!** We have taught the computer how to interpolate those two points by itself.

        But let's step back for a second. What we've just seen is a very simple example of an incredibly powerful principle: **We have programmed the computer to correct itself based on an error function.**

        This brings us to the fundamental strategy of differentiable programming:

        * Define a differentiable error function
        * Compute a descent direction for the parameters
        * Update the parameters in the descent direction
        * Repeat until the error function converges to zero

        Virtually all applications in differentiable programming boil down to this simple workflow.

        We will see more complicated examples in later workbooks. But despite the added complexity, at the end of the day the fundamental workflow is the same.
        """
    )
    return


if __name__ == "__main__":
    app.run()
