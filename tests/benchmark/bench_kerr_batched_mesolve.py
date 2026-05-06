# bench_kerr_batched_mesolve.py

import dynamiqs as dq
import jax.numpy as jnp

# Model parameters from kerr-oscillator.py
K = 1.0
KAPPA = 0.1
ALPHA = 2.0
N = 32

NUM_BATCH = 20
NUM_TSAVE = 100

EPSILONS = jnp.linspace(0.0, 0.5, NUM_BATCH)
T_FINAL = jnp.pi / K


def build_problem():
    a = dq.destroy(N)
    adag = a.dag()

    H0 = K * adag @ adag @ a @ a
    Hd = a + adag

    # Batched Hamiltonian, shape effectively (NUM_BATCH, N, N).
    Hs = H0 + EPSILONS[:, None, None] * Hd

    jump_ops = [jnp.sqrt(KAPPA) * a]

    psi0 = dq.coherent(N, ALPHA)
    tsave = jnp.linspace(0.0, T_FINAL, NUM_TSAVE)

    return Hs, jump_ops, psi0, tsave


def run():
    Hs, jump_ops, psi0, tsave = build_problem()

    options = dq.Options(
        progress_meter=dq.progress_meter.NoProgressMeter(),
    )

    result = dq.mesolve(
        Hs,
        jump_ops,
        psi0,
        tsave,
        options=options,
    )
    result.block_until_ready()

    return result


if __name__ == "__main__":
    run()