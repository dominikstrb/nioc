from typing import NamedTuple, Any, Callable
from jaxtyping import Array, Float
from jax import vmap, jacfwd, grad, jacobian, numpy as jnp

from nioc import Env


class LQRSpec(NamedTuple):
    """LQR specification"""

    Q: Float[Array, "T state state"]
    q: Float[Array, "T state"]
    Qf: Float[Array, "state state"]
    qf: Float[Array, " state"]
    P: Float[Array, "T action state"]
    R: Float[Array, "T action action"]
    r: Float[Array, "T action"]
    A: Float[Array, "T state state"]
    B: Float[Array, "T state action"]


def make_lqr_approx(p: Env, params: Any) -> Callable:
    @vmap
    def approx_timestep(x, u):
        # quadratic approximation of cost function
        P = jacfwd(grad(p._cost, argnums=1), argnums=0)(x, u, params)
        Q = jacfwd(grad(p._cost, argnums=0), argnums=0)(x, u, params)
        R = jacfwd(grad(p._cost, argnums=1), argnums=1)(x, u, params)
        q, r = grad(p._cost, argnums=(0, 1))(x, u, params)

        # linear approximation of dynamics
        A, B = jacobian(p._dynamics, argnums=(0, 1))(x, u, jnp.zeros(p.state_noise_shape), params)
        return Q, q, P, R, r, A, B

    def approx(X, U):
        assert X.shape[0] == (U.shape[0] + 1)
        Q, q, P, R, r, A, B = approx_timestep(X[:-1], U)

        # quadratic approximation of the final time step costs
        Qf = jacfwd(grad(p._final_cost, argnums=0), argnums=0)(X[-1], params)
        qf = grad(p._final_cost, argnums=0)(X[-1], params)

        return LQRSpec(Q=Q, q=q, Qf=Qf, qf=qf, P=P, R=R, r=r, A=A, B=B)

    return approx


class LQGSpec(NamedTuple):
    """ LQG specification """

    Q: Float[Array, "T state state"]
    q: Float[Array, "T state"]
    Qf: Float[Array, "state state"]
    qf: Float[Array, " state"]
    P: Float[Array, "T action state"]
    R: Float[Array, "T action action"]
    r: Float[Array, "T action"]
    A: Float[Array, "T state state"]
    B: Float[Array, "T state action"]
    V: Float[Array, "T state statenoise"]
    Cx: Float[Array, "T state statenoise state"]
    Cu: Float[Array, "T state statenoise action"]
    F: Float[Array, "T obs state"]
    W: Float[Array, "T obs obsnoise"]
    D: Float[Array, "T obs obsnoise state"]


def make_lqg_approx(p: Env, params: Any) -> Callable:
    @vmap
    def approx_noises(x, u):
        # approximate dynamics noise
        # state-independent noise
        V = jacobian(p._dynamics, argnums=2)(x, u, jnp.zeros(p.state_noise_shape), params)
        # state/action dependent noise
        Cx, Cu = jacobian(jacobian(p._dynamics, argnums=2),
                          argnums=(0, 1))(x, u, jnp.zeros(p.state_noise_shape), params)

        # approximately linear observation
        F = jacobian(p._observation, argnums=0)(x, jnp.zeros(p.obs_noise_shape), params)

        # observation noise
        W = jacobian(p._observation, argnums=1)(x, jnp.zeros(p.obs_noise_shape), params)

        # state-dependent observation noise
        D = jacobian(jacobian(p._observation, argnums=1),
                     argnums=0)(x, jnp.zeros(p.obs_noise_shape), params)

        return V, Cx, Cu, F, W, D

    def approx(X, U):
        assert X.shape[0] == (U.shape[0] + 1)

        V, Cx, Cu, F, W, D = approx_noises(X[:-1], U)

        return LQGSpec(*make_lqr_approx(p, params)(X, U), Cx=Cx, Cu=Cu, V=V, F=F, W=W, D=D)

    return approx
