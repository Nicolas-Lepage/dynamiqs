# bench_harmonic_driven_damped_accuracy.py

import dynamiqs as dq
import jax.numpy as jnp


# Model parameters from test_harmonic_test_gate.py
TRUNC = 48
OMEGA = 1.0
KAPPA = 0.05
EPSILON = 2.0

NUM_TSAVE = 100
T_FINAL = 2.0 * jnp.pi / OMEGA


def build_problem():
    a = dq.destroy(TRUNC)

    H = OMEGA * (a.dag() @ a) + EPSILON * (a + a.dag())
    jump_ops = [jnp.sqrt(KAPPA) * a]

    rho0 = dq.fock_dm(TRUNC, 0)
    tsave = jnp.linspace(0.0, T_FINAL, NUM_TSAVE)

    # Minimal analytical observable: <a>(t) should equal alpha_ref(t).
    exp_ops = [a]

    return H, jump_ops, rho0, tsave, exp_ops


def analytical_alpha(tsave):
    s = KAPPA / 2.0 + 1j * OMEGA
    alpha_ss = -1j * EPSILON / s
    return alpha_ss * (1.0 - jnp.exp(-s * tsave))


def run():
    H, jump_ops, rho0, tsave, exp_ops = build_problem()

    options = dq.Options(
        save_states=False,
        progress_meter=dq.progress_meter.NoProgressMeter(),
    )

    result = dq.mesolve(
        H,
        jump_ops,
        rho0,
        tsave,
        exp_ops=exp_ops,
        options=options,
    )
    result.block_until_ready()

    alpha_ref = analytical_alpha(tsave)

    return result, alpha_ref


if __name__ == "__main__":
    run()