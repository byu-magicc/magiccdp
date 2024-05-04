# MAGICC Lab Differentiable Programming Tutorials

!!! warning "WIP"

    These tutorials are a work in progress. Expect bugs, rough spots, and sharp edges.

Tutorial order:

* The Big Picture
    * Modern Machine / Reinforcement Learning
    * Classical Algorithms
    * Differentiable Programming: Combining the best of both worlds
* What exactly are we doing?
    * Case study: Interpolating points with a line
    * Define an error
    * Compute the gradient
    * Make the error converge to zero
* (?) Overview of Automatic Differentiation
* Autotuning PID controllers
    * System model
    * How do we simulate a transfer function?
    * Getting gradients for P, I, D
    * Training Loop
* Autotuning a Kalman Filter
    * (Borrow from Patrick Kidger's website)
* Neural Feedback Loops
    * Use an MLP as a controller
    * Make a single integrator converge to the origin
* "Only Learn What You Need"
    * Train a classical controller + MLP controller
    * Let the NN "fill in the cracks" of our assumptions / simplifications
* Training considerations
    * Exploding / vanishing gradients
    * Bounding controllers

Other Pages:

* Crash course in JAX (copy this over from other project)