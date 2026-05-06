# bench_zeno_cnot_reduced_mesolve.py

import dynamiqs as dq
import jax.numpy as jnp


TWO_PI = 2.0 * jnp.pi

# Model parameters from zeno_cnot_validation.py
N_C = 24
N_T = 24
N_B = 8

ALPHA_C2 = 4.0
ALPHA_T2 = 4.0

ALPHA_C = jnp.sqrt(ALPHA_C2)
ALPHA_T = jnp.sqrt(ALPHA_T2)

G2 = 2.0 * TWO_PI
KAPPA_B = 10.0 * TWO_PI

T_GATE = 0.3
G_CNOT = jnp.pi / (4.0 * ALPHA_C * T_GATE)


def build_operators():
    Ic = dq.eye(N_C)
    Ib = dq.eye(N_B)
    It = dq.eye(N_T)

    a_c = dq.tensor(dq.destroy(N_C), Ib, It)
    b_c = dq.tensor(Ic, dq.destroy(N_B), It)
    a_t = dq.tensor(Ic, Ib, dq.destroy(N_T))

    return a_c, b_c, a_t


def build_hamiltonian_and_collapse():
    a_c, b_c, a_t = build_operators()

    pump = G2 * (a_c @ a_c - ALPHA_C2 * dq.eye_like(a_c)) @ b_c.dag()
    H_pump = pump + pump.dag()

    n_t = a_t.dag() @ a_t

    H_cnot = (
        G_CNOT
        * (a_c + a_c.dag() - 2.0 * ALPHA_C * dq.eye_like(a_c))
        @ (n_t - ALPHA_T2 * dq.eye_like(a_t))
    )

    H = H_pump + H_cnot
    jump_ops = [jnp.sqrt(KAPPA_B) * b_c]

    return H, jump_ops


def build_truth_table_states():
    plus_c = dq.coherent(N_C, ALPHA_C)
    minus_c = dq.coherent(N_C, -ALPHA_C)

    plus_t = dq.coherent(N_T, ALPHA_T)
    minus_t = dq.coherent(N_T, -ALPHA_T)

    vac_b = dq.fock(N_B, 0)

    # Three-mode input states, ordering: control cat, buffer, target cat.
    ket_00_3m = dq.tensor(plus_c, vac_b, plus_t)
    ket_01_3m = dq.tensor(plus_c, vac_b, minus_t)
    ket_10_3m = dq.tensor(minus_c, vac_b, plus_t)
    ket_11_3m = dq.tensor(minus_c, vac_b, minus_t)

    input_states = [
        dq.todm(ket_00_3m),
        dq.todm(ket_01_3m),
        dq.todm(ket_10_3m),
        dq.todm(ket_11_3m),
    ]

    # Two-mode analytical CNOT truth-table outputs, without the buffer.
    #
    # CNOT:
    #   00 -> 00
    #   01 -> 01
    #   10 -> 11
    #   11 -> 10
    ket_00_2m = dq.tensor(plus_c, plus_t)
    ket_01_2m = dq.tensor(plus_c, minus_t)
    ket_10_2m = dq.tensor(minus_c, plus_t)
    ket_11_2m = dq.tensor(minus_c, minus_t)

    expected_outputs = [
        dq.todm(ket_00_2m),
        dq.todm(ket_01_2m),
        dq.todm(ket_11_2m),
        dq.todm(ket_10_2m),
    ]

    return input_states, expected_outputs


def run_one(rho0, H, jump_ops):
    tsave = jnp.array([0.0, T_GATE])

    options = dq.Options(
        save_states=True,
        progress_meter=dq.progress_meter.NoProgressMeter(),
    )

    result = dq.mesolve(
        H,
        jump_ops,
        rho0,
        tsave,
        options=options,
    )
    result.block_until_ready()

    return result


def run():
    H, jump_ops = build_hamiltonian_and_collapse()
    input_states, expected_outputs = build_truth_table_states()

    results = [
        run_one(rho0, H, jump_ops)
        for rho0 in input_states
    ]

    return results, expected_outputs


if __name__ == "__main__":
    run()