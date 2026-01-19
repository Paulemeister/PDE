from dataclasses import dataclass
from typing import Callable
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.optimize as so


@dataclass
class FVM:
    f: Callable
    f1: Callable
    F: Callable
    N: int
    t_end: float
    u0: NDArray[np.floating]
    tag: str
    t_start: float = 0.0
    CFL: float = 0.8


def naive_flux(
    ul, ur, f: Callable, dx, dt, app: FVM | None = None
) -> NDArray[np.floating]:
    _f = app.f if app else f
    return 0.5 * (_f(ul) + _f(ur))


def lax_friedrichs_flux(
    ul, ur, f: Callable, dx, dt, app: FVM | None = None
) -> NDArray[np.floating]:
    _f = app.f if app else f
    return 0.5 * (_f(ul) + _f(ur)) - dx / (2 * dt) * (ur - ul)


def lax_wendroff_flux(
    ul, ur, f: Callable, dx, dt, app: FVM | None = None
) -> NDArray[np.floating]:
    _f = app.f if app else f
    # _f1 = app.f1 if app else lambda x: x * 0 + 1

    a = np.zeros_like(ul)
    mask = ul != ur
    a[mask] = (_f(ul[mask]) - _f(ur[mask])) / (ul[mask] - ur[mask])
    # a = np.where((ul - ur) < 1e-3, 0, (_f(ul) - _f(ur)) / (ul - ur))

    return 0.5 * (_f(ul) + _f(ur)) - dt / (2 * dx) * a**2 * (ur - ul)


def timestep(u, dt, dx, F: Callable, f) -> NDArray[np.floating]:
    ul = np.roll(u, 1)
    lF = F(ul, u, f, dx, dt)
    rF = np.roll(lF, -1)

    new_u = u + dt / dx * (lF - rF)
    return new_u


def calc(
    app: FVM, use_dyn_dt: bool = False
) -> tuple[list[NDArray[np.floating]], list[float], list[float]]:

    # simplify acess to vars
    u0 = app.u0
    N = app.N
    t_end = app.t_end
    F = app.F
    f = app.f
    CFL = app.CFL

    # init output lists
    us = [u0]
    ts = [0.0]
    dts = [np.nan]

    t = 0
    u = u0

    # set dx dt
    dx = 2 / (N - 1)
    dt = CFL * dx / np.max(app.f1(u))

    use_dyn_dt = True
    while t < t_end:
        if use_dyn_dt:
            dt = CFL * dx / np.max(app.f1(u))

        u = timestep(u, dt, dx, F, f)

        dts.append(dt)
        us.append(u)
        ts.append(t)
        t += dt

    return (us, ts, dts)


def lin_adv(x, t) -> NDArray[np.floating]:
    temp_x = np.fmod(x - 2 * t - 1.0, 2) + 1.0
    u = np.where(np.abs(temp_x) > 0.5, 1.0, 0.0)
    return u


def burgers(x, t) -> NDArray[np.floating]:

    u0 = lambda x: np.where(np.abs(x) > 0.5, 1.0, 0.0)
    per = lambda x: x
    per = lambda x: np.fmod(x - x1, 2) + x1
    x0 = -1.0
    x1 = 1.0
    xmin = so.fminbound(lambda x: u0(x), x0, x1)
    u0min = u0(xmin)
    xmax = so.fminbound(lambda x: -u0(x), x0, x1)
    u0max = u0(xmax)
    u_ex = lambda x, t: so.fminbound(
        lambda u: (u - u0(per(x - u * t)) ** 2), u0min, u0max
    )

    return np.vectorize(u_ex)(x, t)
