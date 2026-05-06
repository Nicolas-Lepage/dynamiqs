# bench_two_mode_pwc_vmap_mesolve.py

import jax
import jax.numpy as jnp

import dynamiqs as dq


TWO_PI = 2.0 * jnp.pi

TRUNC_A = 32
TRUNC_B = 6

BATCH_SIZE = 64

PARAMS = {
    "delta_a": -5.0 * TWO_PI,
    "delta_b": -8.3 * TWO_PI,
    "kerr_a": -0.33 * TWO_PI,
    "kerr_b": 3.3 * TWO_PI,
    "kerr_ab": -1.0 * TWO_PI,
    "kappa_a_internal": 0.033 * TWO_PI,
    "kappa_a_external": 0.57 * TWO_PI,
    "kappa_b": 8.3 * TWO_PI,
    "transfer_function_readout_real": -3.3e-3,
    "transfer_function_readout_imaginary": -3.3e-3,
    "quantum_efficiency_readout": 1.0e-2,
    "time_of_flight_buffer": 0.25,
    "transfer_function_input_buffer": -500.0,
    "time_of_flight_memory": 0.32,
    "transfer_function_input_memory_real": -1700.0,
    "transfer_function_input_memory_imaginary": -1.0e4,
    "g2": 0.033 * TWO_PI,
    "delta_a_pump": -3.3 * TWO_PI,
    "delta_b_pump": -3.3 * TWO_PI,
    "time_of_flight_pump": 0.22,
}

TSAVE = jnp.linspace(0.0, 3.0, 1000)


def build_model(params):
    Ia = dq.eye(TRUNC_A)
    Ib = dq.eye(TRUNC_B)

    a = dq.tensor(dq.destroy(TRUNC_A), Ib)
    b = dq.tensor(Ia, dq.destroy(TRUNC_B))

    na = a.dag() @ a
    nb = b.dag() @ b

    H = params["delta_a"] * na + params["delta_b"] * nb
    H = H + 0.5 * params["kerr_a"] * (a.dag() @ a.dag() @ a @ a)
    H = H + 0.5 * params["kerr_b"] * (b.dag() @ b.dag() @ b @ b)
    H = H - params["kerr_ab"] * (na @ nb)

    readout_amp = (
        params["quantum_efficiency_readout"]
        * (
            params["transfer_function_readout_real"]
            + 1j * params["transfer_function_readout_imaginary"]
        )
        * jnp.sqrt(params["kappa_b"])
    )
    readout = readout_amp * b

    # Buffer PWC drive.
    t0_buf = params["time_of_flight_buffer"]
    t1_buf = t0_buf + 2.0
    amp_buf = (
        0.003
        * params["transfer_function_input_buffer"]
        * jnp.sqrt(params["kappa_b"])
    )
    buffer_drive = dq.pwc(
        [t0_buf, t1_buf],
        [amp_buf],
        b + b.dag(),
    )

    # Memory PWC drive.
    t0_mem = params["time_of_flight_memory"] + 0.5
    t1_mem = params["time_of_flight_memory"] + 2.5
    amp_mem = (
        0.003
        * (
            params["transfer_function_input_memory_real"]
            + 1j * params["transfer_function_input_memory_imaginary"]
        )
        * jnp.sqrt(params["kappa_a_external"])
    )
    memory_drive = dq.pwc([t0_mem, t1_mem], [amp_mem], a)
    memory_drive = memory_drive + dq.pwc(
        [t0_mem, t1_mem],
        [jnp.conj(amp_mem)],
        a.dag(),
    )

    # Pump PWC drive.
    t0_pump = 0.3 + params["time_of_flight_pump"]
    t1_pump = 2.3 + params["time_of_flight_pump"]

    pump_drive = dq.pwc(
        [t0_pump, t1_pump],
        [params["g2"]],
        a @ a @ b.dag(),
    )
    pump_drive = pump_drive + dq.pwc(
        [t0_pump, t1_pump],
        [jnp.conj(params["g2"])],
        a.dag() @ a.dag() @ b,
    )
    pump_drive = pump_drive + dq.pwc(
        [t0_pump, t1_pump],
        [params["delta_a_pump"]],
        na,
    )
    pump_drive = pump_drive + dq.pwc(
        [t0_pump, t1_pump],
        [params["delta_b_pump"]],
        nb,
    )

    H = H + buffer_drive + memory_drive + pump_drive

    jump_ops = [
        jnp.sqrt(params["kappa_b"]) * b,
        jnp.sqrt(params["kappa_a_internal"] + params["kappa_a_external"]) * a,
    ]

    return H, jump_ops, readout


def run_single_simulation(params):
    H, jump_ops, readout = build_model(params)

    rho0 = dq.tensor(
        dq.fock_dm(TRUNC_A, 0),
        dq.fock_dm(TRUNC_B, 0),
    )

    options = dq.Options(
        save_states=False,
        progress_meter=dq.progress_meter.NoProgressMeter(),
    )

    return dq.mesolve(
        H,
        jump_ops,
        rho0,
        TSAVE,
        exp_ops=[readout],
        options=options,
    )


def make_params_batch(batch_size):
    return {
        name: jnp.full((batch_size,), value)
        for name, value in PARAMS.items()
    }


def run(batch_size=BATCH_SIZE):
    params_batch = make_params_batch(batch_size)

    batched_run = jax.jit(jax.vmap(run_single_simulation))

    result = batched_run(params_batch)
    result.block_until_ready()

    return result


if __name__ == "__main__":
    run()