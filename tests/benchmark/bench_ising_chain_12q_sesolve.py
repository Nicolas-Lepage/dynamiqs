# bench_ising_chain_12q_sesolve.py

import dynamiqs as dq
import jax.numpy as jnp


# Model parameters from ising-model.py
NUM_MODES = 12
J = 1.0
NUM_TSAVE = 100

T_FINAL = 1.0 / J


def build_problem():
    sigmazs = [
        dq.tensor(
            *[
                dq.sigmaz() if site == mode else dq.eye(2)
                for site in range(NUM_MODES)
            ]
        )
        for mode in range(NUM_MODES)
    ]

    identity = dq.tensor(*[dq.eye(2) for _ in range(NUM_MODES)])

    H = J * sum(
        [sigmazs[mode] @ sigmazs[mode + 1] for mode in range(NUM_MODES - 1)],
        0 * identity,
    )

    plus = dq.unit(dq.basis(2, 0) + dq.basis(2, 1))
    psi0 = dq.tensor(*[plus for _ in range(NUM_MODES)])

    tsave = jnp.linspace(0.0, T_FINAL, NUM_TSAVE)

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