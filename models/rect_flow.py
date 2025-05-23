import math
import torch
import torch.nn as nn
import sympy
from models.cond_mlp import SimpleMLPAdaLN
from typing import Callable


class RectFlowHead(nn.Module):
    def __init__(self, token_embed_dim, decoder_embed_dim, 
                 head_width=1024, head_depth=6,
                 diff_upper_steps=25, diff_lower_steps=5, diff_sampling_strategy="linear"):
        super(RectFlowHead, self).__init__()
        self.token_embed_dim = token_embed_dim
        self.flow_net = SimpleMLPAdaLN(
            in_channels=self.token_embed_dim,
            model_channels=head_width,
            out_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            num_res_blocks=head_depth
        )
        self.num_sampling_steps = [
            step for step in range(diff_lower_steps, diff_upper_steps+1, 1)
        ]
        self.diff_sampling_strategy = diff_sampling_strategy
        self.sampler = 'euler'  # 'rk2_heun', 'euler', 'rf_solver', 'euler_maruyama'
        self.sde_sampler = SDESampler()
    
    def forward(self, target, z, mask=None):
        # Probability of the target
        x0 = torch.randn_like(target)
        x1 = target
        t = torch.rand(len(x0)).to(x0.device)
        xt = t[:, None] * x1 + (1-t[:, None]) * x0

        velocity = self.flow_net(xt, t, z)

        y = x1 - x0
        rec_loss = (velocity - y).pow(2).mean(dim=-1)
        if mask is not None:
            rec_loss = (rec_loss * mask).sum() / mask.sum()

        return rec_loss

    def sample(self, z, temperature=1.0, cfg=1.0, step=0, ar_num_iter=64):
        x_next = torch.randn(z.size(0), self.token_embed_dim, device=z.device)

        upper_step = ar_num_iter - 1
        if self.diff_sampling_strategy == "linear":
            schedule_id = int((upper_step-step) / upper_step * (len(self.num_sampling_steps)-1))
        elif self.diff_sampling_strategy == "cosine":
            schedule_id = int((math.cos(math.pi * step / upper_step) + 1) / 2 * (len(self.num_sampling_steps)-1))
        elif self.diff_sampling_strategy == "constant":
            schedule_id = 0 # Constant schedule
        else:
            raise ValueError(f"Unknown sampling strategy: {self.diff_sampling_strategy}")
        num_sampling_steps = self.num_sampling_steps[schedule_id]
        t_steps = torch.linspace(0.0, 1.0, num_sampling_steps+1, dtype=torch.float32)
        # t_steps = torch.cat([
        #     torch.linspace(0, 0.25, 7, dtype=torch.float32),
        #     torch.linspace(0.2917, 0.7083, self.num_sampling_steps-13, dtype=torch.float32),
        #     torch.linspace(0.75, 1, 7, dtype=torch.float32)
        # ])

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if self.sampler == 'rk2_heun':
                x_next = self.rk2_heun_sampler(z, t_cur, t_next, x_cur)
            elif self.sampler == 'euler':
                x_next = self.euler_sampler(z, t_cur, t_next, x_cur)
            elif self.sampler == 'rf_solver':
                x_next = self.rf_solver_sampler(z, t_cur, t_next, x_cur)
            elif self.sampler == 'euler_maruyama':
                x_next = self.euler_maruyama_sampler(z, t_cur, t_next, x_cur)
            else:
                raise NotImplementedError(f"Sampler {self.sampler} not implemented.")
            
        return x_next

    def euler_sampler(self, z, t_cur, t_next, x_cur):
        time_input = torch.ones(x_cur.size(0), device=z.device, dtype=torch.float32) * t_cur
        d_cur = self.flow_net(x_cur, time_input, z)
        x_next = x_cur + (t_next - t_cur) * d_cur
        return x_next
    
    def rk2_heun_sampler(self, z, t_cur, t_next, x_cur):
        t1 = torch.ones(x_cur.size(0), device=z.device, dtype=torch.float32) * t_cur
        t2 = torch.ones(x_cur.size(0), device=z.device, dtype=torch.float32) * t_next
        d1 = self.flow_net(x_cur, t1, z)
        x1 = x_cur + (t_next - t_cur) * d1
        d2 = self.flow_net(x1, t2, z)
        x_next = x_cur + (t_next - t_cur) * (d1 + d2) / 2

        return x_next
    
    def rf_solver_sampler(self, z, t_cur, t_next, x_cur):
        dt = t_next - t_cur
        t1 = torch.ones(x_cur.size(0), device=z.device, dtype=torch.float32) * t_cur
        d1 = self.flow_net(x_cur, t1, z)
        x_mid = x_cur + dt * d1 * 0.5
        
        t2 = torch.ones(x_cur.size(0), device=z.device, dtype=torch.float32) * (t_cur + t_next) / 2
        d2 = self.flow_net(x_mid, t2, z)

        x_next = x_cur + dt * d1 + 0.5 * dt ** 2 * ((d2 - d1) / dt * 2)

        return x_next
    
    def euler_maruyama_sampler(self, z, t_cur, t_next, x_cur):
        t = torch.ones(x_cur.size(0), device=z.device, dtype=torch.float32) * t_cur
        v_t = self.flow_net(x_cur, t, z)
        x_next = self.sde_sampler.step(t_cur, t_next, x_cur, v_t)

        return x_next

class SDESampler:
    def __init__(
        self,
        noise_scale: float | Callable = 1.0,
        noise_decay_rate: float | Callable = 1.0,
        noise_method: str = "stable",
        ode_method: str = "curved",
    ):
        self.noise_scale = self._process_coeffs(noise_scale)
        self.noise_decay_rate = self._process_coeffs(noise_decay_rate)
        self.noise_method = noise_method
        self.ode_method = ode_method
        self.interp = AffineInterp("straight")

    @staticmethod
    def _process_coeffs(coeff):
        if isinstance(coeff, (torch.Tensor, int, float)):
            return lambda t: coeff
        elif callable(coeff):
            return coeff
        else:
            raise TypeError("coeff should be a float, int, torch.Tensor, or callable.")

    def step(self, t, t_next, x_t, v_t):
        step_size = t_next - t

        # Solve for x_0 and x_1 given x_t and v_t
        interp = self.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        x_0 = interp.x_0
        x_1 = interp.x_1
        beta_t = interp.beta(t)

        # Part 1: Add noise

        # 1) Calculate beta_t_noised, the fraction of x_0 that will be noised
        beta_t_noised = (
            step_size * self.noise_scale(t) * beta_t ** self.noise_decay_rate(t)
        )
        # Clip beta_t_noised to beta_t, it's not meaningful to have beta_t_noised > beta_t
        beta_t_noised = torch.clamp(beta_t_noised, max=beta_t)

        refresh_noise = torch.randn_like(x_t)
        pi_0_mean = 0.0

        # 2) Remove beta_t_noised * x_0 and then add refreshed noise
        if self.noise_method.lower() == "stable":
            noise_std = (beta_t**2 - (beta_t - beta_t_noised) ** 2) ** 0.5
            langevin_term = -beta_t_noised * (x_0 - pi_0_mean) + noise_std * (
                refresh_noise - pi_0_mean
            )
        # this is the taylor approximation of the stable method when beta_t_noised is small, and corresponds to Euler method for the langevin dynamics
        elif self.noise_method.lower() == "euler":
            noise_std = (2 * beta_t * beta_t_noised) ** 0.5
            langevin_term = -beta_t_noised * (x_0 - pi_0_mean) + noise_std * (
                refresh_noise - pi_0_mean
            )

        else:
            raise ValueError(f"Unknown noise_method: {self.noise_method}")

        # print(f"t = {t:.5f}, coeff = {coeff:.5f}, noise_std = {noise_std:.5f}")

        x_t_noised = x_t + langevin_term

        # Advance time using the specified ODE method
        if self.ode_method.lower() == "euler":
            # standard Euler method
            x_t_next = x_t_noised + step_size * v_t

        elif self.ode_method.lower() == "curved":
            # Curved Euler method, following the underlying interpolation curve
            # a. Get x_0_noised from x_t_noised and x_1
            x_0_noised = self.interp.solve(
                t=t, x_t=x_t_noised, x_1=x_1
            ).x_0

            # b. Interpolate to get x_t_next given x_0_noised and x_1
            x_t_next = self.interp.solve(
                t=t_next, x_0=x_0_noised, x_1=x_1
            ).x_t

        else:
            raise ValueError(f"Unknown ode_method: {self.ode_method}")

        # Update the current sample
        return x_t_next


def match_dim_with_data(
    t: torch.Tensor | float | list[float],
    x_shape: tuple,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    expand_dim: bool = True,
):
    r"""
    Format the time tensor `t` to match the batch size and dimensions of the data.

    This function ensures that the time tensor `t` is properly formatted to match the batch size specified by `x_shape`.
    It handles various input types for `t`, including scalars, lists, or tensors, and converts `t` into a tensor with
    appropriate shape, device, and dtype. Optionally, it can expand `t` to match the data dimensions beyond the batch size.

    Args:
        t (`torch.Tensor`, `float`, or `list[float]`):
            The time(s) to be matched with the data dimensions. Can be a scalar, a list of floats, or a tensor.
        x_shape (`tuple`):
            The shape of the data tensor, typically `(batch_size, ...)`.
        device (`torch.device`, optional, defaults to `torch.device("cpu")`):
            The device on which to place the time tensor.
        dtype (`torch.dtype`, optional, defaults to `torch.float32`):
            The data type of the time tensor.
        expand_dim (`bool`, optional, defaults to `True`):
            Whether to expand `t` to match the dimensions after the batch dimension.

    Returns:
        t_reshaped (`torch.Tensor`):
            The time tensor `t`, formatted to match the batch size or dimensions of the data.

    Example:
        ```python
        >>> x_shape = (16, 3, 32, 32)
        >>> t_prepared = match_dim_with_data([0.5], x_shape, expand_dim=True)
        >>> t_prepared.shape
        torch.Size([16, 1, 1, 1])

        >>> x_shape = (16, 3, 32, 32)
        >>> t_prepared = match_dim_with_data([0.5], x_shape, expand_dim=False)
        >>> t_prepared.shape
        torch.Size([16])
        ```
    """
    B = x_shape[0]  # Batch size
    ndim = len(x_shape)

    if isinstance(t, float):
        # Create a tensor of shape (B,) with the scalar value
        t = torch.full((B,), t, device=device, dtype=dtype)
    elif isinstance(t, list):
        if len(t) == 1:
            # If t is a list of length 1, repeat the scalar value B times
            t = torch.full((B,), t[0], device=device, dtype=dtype)
        elif len(t) == B:
            t = torch.tensor(t, device=device, dtype=dtype)
        else:
            raise ValueError(
                f"Length of t list ({len(t)}) does not match batch size ({B}) and is not 1."
            )
    elif isinstance(t, torch.Tensor):
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            # Scalar tensor, expand to (B,)
            t = t.repeat(B)
        elif t.ndim == 1:
            if t.shape[0] == 1:
                # Tensor of shape (1,), repeat to (B,)
                t = t.repeat(B)
            elif t.shape[0] == B:
                # t is already of shape (B,)
                pass
            else:
                raise ValueError(
                    f"Batch size of t ({t.shape[0]}) does not match x ({B})."
                )
        elif t.ndim == 2:
            if t.shape == (B, 1):
                # t is of shape (B, 1), squeeze last dimension
                t = t.squeeze(1)
            elif t.shape == (1, 1):
                # t is of shape (1, 1), expand to (B,)
                t = t.squeeze().repeat(B)
            else:
                raise ValueError(
                    f"t must be of shape ({B}, 1) or (1, 1), but got {t.shape}"
                )
        else:
            raise ValueError(f"t can have at most 2 dimensions, but got {t.ndim}")
    else:
        raise TypeError(
            f"t must be a torch.Tensor, float, or a list of floats, but got {type(t)}."
        )

    # Reshape t to have singleton dimensions matching x_shape after the batch dimension
    if expand_dim:
        expanded_dims = [1] * (ndim - 1)
        t = t.view(B, *expanded_dims)

    return t


class AffineInterpSolver:
    r"""Symbolic solver for affine interpolation equations.

    This class provides a symbolic solver for the affine interpolation equations:

        x_t = a_t * x_1 + b_t * x_0,
        dot_x_t = dot_a_t * x_1 + dot_b_t * x_0.

    Given at least two known variables among `x_0, x_1, x_t, dot_x_t`, and the rest unknown,
    the solver computes the unknowns. The method precomputes symbolic solutions for all pairs
    of unknown variables and stores them as lambdified functions for efficient numerical computation.
    """

    def __init__(self):
        r"""Initialize the `AffineInterpSolver` class.

        This method sets up the symbolic equations for affine interpolation and precomputes symbolic solvers
        for all pairs of unknown variables among `x_0, x_1, x_t, dot_x_t`. The equations are:

            x_t = a_t * x_1 + b_t * x_0,
            dot_x_t = dot_a_t * x_1 + dot_b_t * x_0.

        By solving these equations symbolically for each pair of unknown variables, the method creates lambdified
        functions that can be used for efficient numerical computations during runtime.
        """
        # Define symbols
        x_0, x_1, x_t, dot_x_t = sympy.symbols("x_0 x_1 x_t dot_x_t")
        a_t, b_t, dot_a_t, dot_b_t = sympy.symbols("a_t b_t dot_a_t dot_b_t")

        # Equations
        eq1 = sympy.Eq(x_t, a_t * x_1 + b_t * x_0)
        eq2 = sympy.Eq(dot_x_t, dot_a_t * x_1 + dot_b_t * x_0)

        # Variables to solve for
        variables = [x_0, x_1, x_t, dot_x_t]
        self.symbolic_solvers = {}

        # Create symbolic solvers for all pairs of unknown variables
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                unknown1, unknown2 = variables[i], variables[j]
                # print(f"Solving for {unknown1} and {unknown2}")
                # Solve equations
                solutions = sympy.solve([eq1, eq2], (unknown1, unknown2), dict=True)
                if solutions:
                    solution = solutions[0]
                    # Create lambdified functions
                    expr1 = solution[unknown1]
                    expr2 = solution[unknown2]
                    func = sympy.lambdify(
                        [x_0, x_1, x_t, dot_x_t, a_t, b_t, dot_a_t, dot_b_t],
                        [expr1, expr2],
                        modules="numpy",
                    )
                    # Store solver function
                    var_names = (str(unknown1), str(unknown2))
                    self.symbolic_solvers[var_names] = func

    def solve(
        self,
        results,
    ):
        r"""Solve for unknown variables in the affine interpolation equations.

        This method computes the unknown variables among `x_0, x_1, x_t, dot_x_t` given the known variables
        in the `results` object. It uses the precomputed symbolic solvers to find the solutions efficiently.

        Args:
            results (`Any`): An object (e.g., a dataclass or any object with attributes) **containing the following attributes**:
            - `x_0` (`torch.Tensor` or `None`): Samples from the source distribution.
            - `x_1` (`torch.Tensor` or `None`): Samples from the target distribution.
            - `x_t` (`torch.Tensor` or `None`): Interpolated samples at time `t`.
            - `dot_x_t` (`torch.Tensor` or `None`): The time derivative of `x_t` at time `t`.
            - `a_t`, `b_t`, `dot_a_t`, `dot_b_t` (`torch.Tensor`): Interpolation coefficients and their derivatives.

            Known variables should have their values assigned; unknown variables should be set to `None`.

        Returns:
            `Any`: The input `results` object with the unknown variables computed and assigned.

        Notes:
            - If only one variable among `x_0, x_1, x_t, dot_x_t` is unknown, the method selects an additional
              known variable to form a pair for solving.
            - The method assumes that at least two variables among `x_0, x_1, x_t, dot_x_t` are known.
            - The variables `a_t`, `b_t`, `dot_a_t`, and `dot_b_t` must be provided in `results`.

        Example:
            ```python
            >>> solver = AffineInterpSolver()
            >>> class Results:
            ...     x_0 = None
            ...     x_1 = torch.tensor([...])
            ...     x_t = torch.tensor([...])
            ...     dot_x_t = torch.tensor([...])
            ...     a_t = torch.tensor([...])
            ...     b_t = torch.tensor([...])
            ...     dot_a_t = torch.tensor([...])
            ...     dot_b_t = torch.tensor([...])
            >>> results = Results()
            >>> solver.solve(results)
            >>> print(results.x_0)  # Now x_0 is computed and assigned in `results`.
            ```
        """
        known_vars = {
            k: getattr(results, k)
            for k in ["x_0", "x_1", "x_t", "dot_x_t"]
            if getattr(results, k) is not None
        }
        unknown_vars = {
            k: getattr(results, k)
            for k in ["x_0", "x_1", "x_t", "dot_x_t"]
            if getattr(results, k) is None
        }
        unknown_keys = tuple(unknown_vars.keys())

        if len(unknown_keys) > 2:
            raise ValueError(
                "At most two variables among (x_0, x_1, x_t, dot_x_t) can be unknown."
            )
        elif len(unknown_keys) == 0:
            return results
        elif len(unknown_keys) == 1:
            # Select one known variable to make up the pair
            for var in ["x_0", "x_1", "x_t", "dot_x_t"]:
                if var in known_vars:
                    unknown_keys.append(var)
                    break

        func = self.symbolic_solvers.get(unknown_keys)

        # Prepare arguments in the order [x_0, x_1, x_t, dot_x_t, a_t, b_t, dot_a_t, dot_b_t]
        args = []
        for var in ["x_0", "x_1", "x_t", "dot_x_t", "a_t", "b_t", "dot_a_t", "dot_b_t"]:
            value = getattr(results, var, None)
            if value is None:
                value = 0  # Placeholder for unknowns
            args.append(value)

        # Compute the unknown variables
        solved_values = func(*args)
        # Assign the solved values back to results
        setattr(results, unknown_keys[0], solved_values[0])
        setattr(results, unknown_keys[1], solved_values[1])

        return results


class AffineInterp(nn.Module):
    r"""Affine Interpolation Module for Rectified Flow Models.

    This class implements affine interpolation between samples `x_0` from source distribution `pi_0` and
    samples `x_1` from target distribution `pi_1` over a time interval `t` in `[0, 1]`.

    The interpolation is defined using time-dependent coefficients `alpha(t)` and `beta(t)`:

        x_t = alpha(t) * x_1 + beta(t) * x_0,
        dot_x_t = dot_alpha(t) * x_1 + dot_beta(t) * x_0,

    where `x_t` is the interpolated state at time `t`, and `dot_x_t` is its time derivative.

    The module supports several predefined interpolation schemes:

    - **Straight Line Interpolation** (`"straight"` or `"lerp"`):

        alpha(t) = t,  beta(t) = 1 - t,
        dot_alpha(t) = 1, dot_beta(t) = -1.

    - **Spherical Interpolation** (`"spherical"` or `"slerp"`):

        alpha(t) = sin(pi / 2 * t), beta(t) = cos(pi / 2  * t),
        dot_alpha(t) = pi / 2 * cos(pi / 2 * t), dot_beta(t) = -pi / 2 * sin(pi / 2 * t).

    - **DDIM/DDPM Interpolation** (`"ddim"` or `"ddpm"`):

        alpha(t) = exp(-a * (1 - t) ** 2 / 4.0 - b * (1 - t) / 2.0),
        beta(t) = sqrt(1 - alpha(t) ** 2),
        a = 19.9 and b = 0.1.

    Attributes:
        name (`str`): Name of the interpolation scheme.
        alpha (`Callable`): Function defining `alpha(t)`.
        beta (`Callable`): Function defining `beta(t)`.
        dot_alpha (`Callable` or `None`): Function defining the time derivative `dot_alpha(t)`.
        dot_beta (`Callable` or `None`): Function defining the time derivative `dot_beta(t)`.
        solver (`AffineInterpSolver`): Symbolic solver for the affine interpolation equations.
        a_t (`torch.Tensor` or `None`): Cached value of `a(t)` after computation.
        b_t (`torch.Tensor` or `None`): Cached value of `b(t)` after computation.
        dot_a_t (`torch.Tensor` or `None`): Cached value of `dot_a(t)` after computation.
        dot_b_t (`torch.Tensor` or `None`): Cached value of `dot_b(t)` after computation.
    """

    def __init__(
        self,
        name: str | None = None,
        alpha: Callable | None = None,
        beta: Callable | None = None,
        dot_alpha: Callable | None = None,
        dot_beta: Callable | None = None,
    ):
        super().__init__()

        if alpha is not None or beta is not None:
            if name and name.lower() in ["straight", "lerp", "slerp", "spherical", "ddim", "ddpm"]:
                raise ValueError(
                    f"You provided a predefined interpolation name '{name}' and also custom alpha/beta. "
                    "Only one option is allowed."
                )
            if alpha is None or beta is None:
                raise ValueError("Custom interpolation requires both alpha and beta functions.")
            
            name = name if name is not None else "custom"
            alpha = alpha
            beta = beta
            dot_alpha = dot_alpha
            dot_beta = dot_beta

        else:
            if name is None:
                raise ValueError(
                    "No interpolation scheme name provided, and no custom alpha/beta supplied."
                )

            lower_name = name.lower()

            if lower_name in ["straight", "lerp"]:
                # Straight line interpolation
                name = "straight"
                alpha = lambda t: t
                beta = lambda t: 1 - t
                dot_alpha = lambda t: torch.ones_like(t)
                dot_beta = lambda t: -torch.ones_like(t)

            elif lower_name in ["slerp", "spherical"]:
                # Spherical interpolation
                name = "spherical"
                alpha = lambda t: torch.sin(t * torch.pi / 2.0)
                beta = lambda t: torch.cos(t * torch.pi / 2.0)
                dot_alpha = lambda t: (torch.pi / 2.0) * torch.cos(t * torch.pi / 2.0)
                dot_beta = lambda t: -(torch.pi / 2.0) * torch.sin(t * torch.pi / 2.0)

            elif lower_name in ["ddim", "ddpm"]:
                # DDIM/DDPM scheme
                name = "DDIM"
                a = 19.9
                b = 0.1
                alpha = lambda t: torch.exp(-a * (1 - t) ** 2 / 4.0 - b * (1 - t) / 2.0)
                beta = lambda t: torch.sqrt(1 - self.alpha(t) ** 2)
                dot_alpha = None
                dot_beta = None

            else:
                raise ValueError(
                    f"Unknown interpolation scheme name '{name}'. Provide a known scheme name "
                    "or supply custom alpha/beta functions."
                )
            
        self.name = name
        self.alpha = lambda t: alpha(self.ensure_tensor(t))
        self.beta = lambda t: beta(self.ensure_tensor(t))
        self.dot_alpha = None if dot_alpha is None else lambda t: dot_alpha(self.ensure_tensor(t))
        self.dot_beta = None if dot_beta is None else lambda t: dot_beta(self.ensure_tensor(t))
        
        self.solver = AffineInterpSolver()
        self.a_t = None
        self.b_t = None
        self.dot_a_t = None
        self.dot_b_t = None
        self.x_0 = None
        self.x_1 = None
        self.x_t = None
        self.dot_x_t = None

    @staticmethod
    def ensure_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    @staticmethod
    def value_and_grad(f, input_tensor, detach=True):
        r"""Compute the value and gradient of a function with respect to its input tensor.

        This method computes both the function value `f(x)` and its gradient `\nabla_x f(x)` at a given input tensor `x`.

        Args:
            f (`Callable`): The function `f` to compute.
            input_tensor (`torch.Tensor`): The input tensor.
            detach (`bool`, optional, defaults to `True`): Whether to detach the computed value and gradient from the computation graph.

        Returns:
            value_and_grad (Tuple[`torch.Tensor`, `torch.Tensor`]):
                `value`: The function value `f(x)`.
                `grad`: The gradient `\nabla_x f(x)`.

        Example:
            ```python
            >>> def func(x):
            ...     return x ** 2
            >>> x = torch.tensor(3.0, requires_grad=True)
            >>> value, grad = AffineInterp.value_and_grad(func, x)
            >>> value
            tensor(9.)
            >>> grad
            tensor(6.)
            ```
        """
        x = input_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            value = f(x)
            (grad,) = torch.autograd.grad(value.sum(), x, create_graph=not detach)
        if detach:
            value = value.detach()
            grad = grad.detach()
        return value, grad

    def get_coeffs(self, t, detach=True):
        r"""Compute the interpolation coefficients `a_t`, `b_t`, and their derivatives `dot_a_t`, `dot_b_t` at time `t`.

        Args:
            t (`torch.Tensor`): Time tensor `t` at which to compute the coefficients.
            detach (`bool`, defaults to `True`): Whether to detach the computed values from the computation graph.

        Returns:
            coeff (Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`, `torch.Tensor`]):
                `(a_t, b_t, dot_a_t, dot_b_t)`: The interpolation coefficients and their derivatives at time `t`.

        Notes:
            - If `dot_alpha` or `dot_beta` are not provided, their values are computed using automatic differentiation.
            - The computed coefficients are cached in the instance attributes `a_t`, `b_t`, `dot_a_t`, and `dot_b_t`.
        """
        if self.dot_alpha is None:
            a_t, dot_a_t = self.value_and_grad(self.alpha, t, detach=detach)
        else:
            a_t = self.alpha(t)
            dot_a_t = self.dot_alpha(t)
        if self.dot_beta is None:
            b_t, dot_b_t = self.value_and_grad(self.beta, t, detach=detach)
        else:
            b_t = self.beta(t)
            dot_b_t = self.dot_beta(t)
        self.a_t = a_t
        self.b_t = b_t
        self.dot_a_t = dot_a_t
        self.dot_b_t = dot_b_t
        return a_t, b_t, dot_a_t, dot_b_t

    def forward(self, x_0, x_1, t, detach=True):
        r"""Compute the interpolated `X_t` and its time derivative `dotX_t`.

        Args:
            x_0 (`torch.Tensor`): Samples from source distribution, shape `(B, D_1, D_2, ..., D_n)`.
            x_1 (`torch.Tensor`): Samples from target distribution, same shape as `x_0`.
            t (`torch.Tensor`): Time tensor `t`
            detach (`bool`, defaults to `True`): Whether to detach computed coefficients from the computation graph.

        Returns:
            interpolation (Tuple[`torch.Tensor`, `torch.Tensor`]):
                `x_t`: Interpolated state at time `t`.
                `dot_x_t`: Time derivative of the interpolated state at time `t`.
        """
        t = match_dim_with_data(t, x_1.shape, device=x_1.device, dtype=x_1.dtype)
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t, detach=detach)
        x_t = a_t * x_1 + b_t * x_0
        dot_x_t = dot_a_t * x_1 + dot_b_t * x_0
        return x_t, dot_x_t

    def solve(self, t=None, x_0=None, x_1=None, x_t=None, dot_x_t=None, detach=True):
        r"""Solve for unknown variables in the affine interpolation equations.

        This method solves the equations:

            x_t = a_t * x_1 + b_t * x_0,
            dot_x_t = dot_a_t * x_1 + dot_b_t * x_0.

        Given any two of known variables among `x_0`, `x_1`, `x_t`, and `dot_x_t`, this method computes the unknown variables using the `AffineInterpSolver`.
        Must provide at least two known variables among `x_0`, `x_1`, `x_t`, and `dot_x_t`.

        Args:
            t (`torch.Tensor` or `None`):
                Time tensor `t`. Must be provided.
            x_0 (`torch.Tensor` or `None`, optional):
                Samples from the source distribution `pi_0`.
            x_1 (`torch.Tensor` or `None`, optional):
                Samples from the target distribution `pi_1`.
            x_t (`torch.Tensor` or `None`, optional):
                Interpolated samples at time `t`.
            dot_x_t (`torch.Tensor` or `None`, optional):
                Time derivative of the interpolated samples at time `t`.

        Returns:
            `AffineInterp`:
                The instance itself with the computed variables assigned to `x_0`, `x_1`, `x_t`, or `dot_x_t`.

        Raises:
            `ValueError`:
                - If `t` is not provided.
                - If less than two variables among `x_0`, `x_1`, `x_t`, `dot_x_t` are provided.

        Example:
            ```python
            >>> interp = AffineInterp(name='straight')
            >>> t = torch.tensor([0.5])
            >>> x_t = torch.tensor([[0.5]])
            >>> dot_x_t = torch.tensor([[1.0]])
            >>> interp.solve(t=t, x_t=x_t, dot_x_t=dot_x_t)
            >>> print(interp.x_0)  # Computed initial state x_0
            tensor([[0.]])
            >>> print(interp.x_1)  # Computed final state x_1
            tensor([[1.]])
            ```
        """
        if t is None:
            raise ValueError("t must be provided")

        self.x_0 = x_0
        self.x_1 = x_1
        self.x_t = x_t
        self.dot_x_t = dot_x_t

        non_none_values = [v for v in [x_0, x_1, x_t, dot_x_t] if v is not None]
        if len(non_none_values) < 2:
            raise ValueError("At least two of x_0, x_1, x_t, dot_x_t must not be None")

        x_not_none = non_none_values[0]
        t = match_dim_with_data(
            t, x_not_none.shape, device=x_not_none.device, dtype=x_not_none.dtype
        )
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t, detach)

        return self.solver.solve(self)