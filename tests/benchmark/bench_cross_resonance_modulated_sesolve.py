# bench_cross_resonance_modulated_sesolve.py

import dynamiqs as dq
import jax.numpy as jnp


# Model parameters from cross-resonance.py
OMEGA_1 = 4.0
OMEGA_2 = 6.0
J = 0.4
EPSILON = 0.4

NUM_TSAVE = 100


def build_problem():
    gate_time = 0.5 * jnp.pi * jnp.abs(OMEGA_2 - OMEGA_1) / (J * EPSILON)
    tsave = jnp.linspace(0.0, gate_time, NUM_TSAVE)

    sz1 = dq.tensor(dq.sigmaz(), dq.eye(2))
    sz2 = dq.tensor(dq.eye(2), dq.sigmaz())

    sp1 = dq.tensor(dq.sigmap(), dq.eye(2))
    sp2 = dq.tensor(dq.eye(2), dq.sigmap())

    sm1 = dq.tensor(dq.sigmam(), dq.eye(2))
    sm2 = dq.tensor(dq.eye(2), dq.sigmam())

    omega_d = OMEGA_2 - J**2 / (OMEGA_1 - OMEGA_2)

    H0 = (
        0.5 * OMEGA_1 * sz1
        + 0.5 * OMEGA_2 * sz2
        + J * (sp1 @ sm2 + sm1 @ sp2)
    )

    Hd = EPSILON * (sp1 + sm1)

    def fd(t):
        return jnp.cos(omega_d * t)

    H = H0 + dq.modulated(fd, Hd)

    psi0 = dq.tensor(
        dq.basis(2, 1),
        dq.basis(2, 1),
    )

    return H, psi0, tsave


def run():
    H, psi0, tsave = build_problem()

    options = dq.Options(
        progress_meter=dq.progress_meter.NoProgressMeter(),
    )

    result = dq.sesolve(
        H,
        psi0,
        tsave,
        options=options,
    )
    result.block_until_ready()

    return result


if __name__ == "__main__":
    run()