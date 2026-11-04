
import json
import logging
import math
import os
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["USE_NUMEXPR"] = "0"

import torch

torch.set_num_threads(1)
warnings.filterwarnings("ignore")

from tqdm import tqdm

from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from BuildGraph import build_physics_graph, evaluate_system_FoM




EXPERIMENT = {
    "n_trials": 100,
    "global_seed": 40,
    "budgets": {
        "TurBO": 300,
        "PCABO": 400,
        "REMBO": 400,
        "SAASBO": 100,
    },
}


BOUNDS = {
    "relative_half_width": 0.15,
    "zero_value_box": (1e-6, 0.1),
    "fraction_clip": (0.01, 0.99),
    "fraction_keys": (
        "ratio", "porosity", "degree", "emissivity", "fraction",
        "excess", "rate", "factor", "accuracy",
    ),
    "packing_density_clip": (0.1, 1.0),
    "geometry_keys": (
        "thickness", "area", "radius", "gap", "size", "width",
        "depth", "cross_section", "length", "volume", "pitch",
    ),
    "geometry_floor": 1e-7,
    "temp_floor": 253.15,
    "invalid_fom": -9999.0,
}


TURBO = {
    "m": 2,
    "n_init_cap": 10,
    "length": 0.8,
    "length_min": 0.5 ** 7,
    "length_max": 1.6,
    "failure_tolerance": 5,
    "success_tolerance": 3,
    "success_rel_tol": 1e-3,
    "num_restarts": 5,
    "raw_samples": 512,
    "q": 1,
}

PCABO = {
    "embedding_divisor": 2,
    "min_embedding": 2,
    "n_init_offset": 2,
    "min_n_init": 5,
    "latent_sigma": 2.0,
    "latent_pad_sigma": 0.1,
    "unit_clip": (0.0, 1.0),
}

# Mask.py run_rembo_bo. A is redrawn from N(0, I) for every trial.
REMBO = {
    "embedding_divisor": 2,
    "min_embedding": 2,
    "projection_clip": (-1.0, 1.0),
}


SAASBO = {
    "n_init": 10,
    "q": 1,
    "num_restarts": 5,
    "raw_samples": 512,
    "warmup_steps": 512,
    "num_samples": 256,
    "thinning": 16,
    "max_tree_depth": 6,
    "train_yvar": 1e-6,
}

INDEPENDENT_VARS = [
    "anode_thickness", "cathode_thickness", "separator_thickness", "electrode_gap",
    "cell_width", "cell_length", "cell_height", "tab_width", "tab_thickness",
    "jellyroll_radius", "particle_size_anode", "particle_size_cathode",
    "binder_ratio", "conductive_additive_ratio", "separator_porosity",
    "electrolyte_concentration", "pre_lithiation_degree", "packing_density",
    "solid_electrolyte_fraction", "lithium_excess", "coolant_flow",
    "chiller_temp_setpoint", "tim_thickness", "tim_thermal_conductivity",
    "base_plate_thickness", "fin_thickness", "fin_height", "fin_pitch",
    "cooling_channel_width", "cooling_channel_depth", "pump_speed", "fan_speed",
    "compressor_power", "coolant_viscosity_ref", "coolant_density_ref",
    "fan_blade_radius", "fan_blade_angle", "pump_impeller_diameter", "nozzle_diameter",
    "pipe_roughness", "wire_radius", "busbar_cross_section", "busbar_length",
    "inverter_switching_freq", "gate_resistance", "mosfet_rdson", "dc_link_cap",
    "inductor_volume", "winding_turns", "core_cross_section", "housing_thickness",
    "housing_surface_area", "casing_emissivity", "vibration_mount_stiffness",
    "vibration_mount_damping", "fastener_torque", "weld_seam_length",
    "potting_compound_thickness", "gasket_thickness", "gasket_compression",
    "calendering_pressure", "formation_current", "drying_temperature", "drying_time",
    "coating_speed", "welding_power", "assembly_pressure", "aging_temperature",
    "aging_time", "inspection_rate", "soc_target", "dod_max", "c_rate_max",
    "voltage_max", "voltage_min", "temp_limit_high", "temp_limit_low",
    "power_limit_surge", "current_limit_continuous", "derating_factor", "cell_spacing",
    "pack_volume_margin", "bms_sampling_rate", "sensor_accuracy",
    "isolation_resistance", "ambient_temp",
]

COLORS = {
    "TurBO": "red",
    "PCABO": "orange",
    "REMBO": "green",
    "SAASBO": "brown",
}


def generate_bounds(param_names, base_state):
    bounds = []
    half = BOUNDS["relative_half_width"]
    for p in param_names:
        val = base_state[p]
        if val == 0.0:
            low, high = BOUNDS["zero_value_box"]
        else:
            low, high = val * (1.0 - half), val * (1.0 + half)

        if any(key in p for key in BOUNDS["fraction_keys"]):
            lo, hi = BOUNDS["fraction_clip"]
            low, high = max(lo, low), min(hi, high)
        elif "packing_density" in p:
            lo, hi = BOUNDS["packing_density_clip"]
            low, high = max(lo, low), min(hi, high)
        elif any(key in p for key in BOUNDS["geometry_keys"]):
            low = max(BOUNDS["geometry_floor"], low)
        elif "temp" in p:
            low = max(BOUNDS["temp_floor"], low)
        bounds.append((low, high))
    return bounds


def _embedding_dim(dim, cfg):
    return max(cfg["min_embedding"], dim // cfg["embedding_divisor"])


def _sanitize(val):
    if val is None or np.isnan(val) or np.isinf(val):
        return BOUNDS["invalid_fom"]
    return float(val)


def _new_tracker():
    return {"best_fom": -np.inf, "history_y": []}


def _observe(tracker, val):
    val = _sanitize(val)
    if val > tracker["best_fom"]:
        tracker["best_fom"] = val
    tracker["history_y"].append(val)
    return val



def ax_bo_optimize(obj_func, bounds, max_evals, seed=None, initial_x_norm=None, initial_y=None):
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties

    logging.getLogger("ax").setLevel(logging.CRITICAL)
    ax_client = AxClient(random_seed=seed, verbose_logging=False)
    parameters = [
        {"name": f"x{i}", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"}
        for i in range(len(bounds))
    ]
    ax_client.create_experiment(
        name="bo_fom_experiment",
        parameters=parameters,
        objectives={"fom": ObjectiveProperties(minimize=False)},
    )

    lb = np.array([float(b[0]) for b in bounds])
    ub = np.array([float(b[1]) for b in bounds])

    if initial_x_norm is not None and initial_y is not None:
        for x_n, y_val in zip(initial_x_norm, initial_y):
            params_dict = {f"x{i}": float(np.clip(x_n[i], 0.0, 1.0)) for i in range(len(bounds))}
            _, trial_index = ax_client.attach_trial(params_dict)
            ax_client.complete_trial(trial_index=trial_index, raw_data=y_val)

    for _ in range(max_evals):
        params_dict, trial_index = ax_client.get_next_trial()
        x_norm = np.array([params_dict[f"x{i}"] for i in range(len(bounds))])
        x_real = lb + x_norm * (ub - lb)
        val = obj_func(x_real)
        ax_client.complete_trial(trial_index=trial_index, raw_data=val)




def run_turbo(full_params, full_bounds, base_state, total_iters, seed):
    tracker = _new_tracker()
    device, dtype = torch.device("cpu"), torch.double
    dim = len(full_bounds)
    lb = torch.tensor([b[0] for b in full_bounds], dtype=dtype, device=device)
    ub = torch.tensor([b[1] for b in full_bounds], dtype=dtype, device=device)

    if seed is not None:
        torch.manual_seed(seed)

    n_init = min(TURBO["n_init_cap"], max(total_iters // 2, 1))
    n_init = min(n_init, total_iters)

    def obj_full(x):
        return _observe(tracker, evaluate_system_FoM(x, full_params, base_state))

    X_norm = SobolEngine(dim, scramble=True, seed=seed).draw(n_init).to(dtype=dtype, device=device)
    X_real = lb + X_norm * (ub - lb)
    Y = torch.tensor(
        [obj_full(X_real[i].cpu().numpy()) for i in range(n_init)],
        dtype=dtype, device=device,
    ).unsqueeze(-1)

    sorted_indices = torch.argsort(Y.squeeze(-1), descending=True)
    states = []
    for i in range(min(TURBO["m"], len(Y))):
        idx = sorted_indices[i]
        states.append({
            "length": TURBO["length"],
            "length_min": TURBO["length_min"],
            "length_max": TURBO["length_max"],
            "failure_counter": 0,
            "failure_tolerance": TURBO["failure_tolerance"],
            "success_counter": 0,
            "success_tolerance": TURBO["success_tolerance"],
            "best_value": Y[idx].item(),
            "center": X_norm[idx].clone(),
        })

    eval_count = n_init
    while eval_count < total_iters:
        valid_mask = Y > -9000.0
        if valid_mask.any() and valid_mask.sum() > 1:
            y_mean = Y[valid_mask].mean()
            y_std = Y[valid_mask].std()
        else:
            y_mean = Y.mean()
            y_std = Y.std()
        if torch.isnan(y_std) or y_std < 1e-9:
            y_std = torch.tensor(1.0, dtype=dtype, device=device)

        train_Y = (Y - y_mean) / y_std
        if (~valid_mask).any() and valid_mask.any():
            train_Y[~valid_mask] = train_Y[valid_mask].min() - 1.0

        model = SingleTaskGP(X_norm, train_Y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            except Exception:
                pass

        ei = ExpectedImprovement(model, best_f=train_Y.max())
        best_acq_val, best_new_x_norm, best_tr_idx = -float("inf"), None, -1

        for idx, state in enumerate(states):
            half_len = torch.clamp(
                torch.full((dim,), state["length"], dtype=dtype, device=device),
                state["length_min"], 2.0,
            )
            tr_lb = torch.clamp(state["center"] - half_len / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(state["center"] + half_len / 2.0, 0.0, 1.0)
            try:
                new_x_norm, acq_val = optimize_acqf(
                    acq_function=ei,
                    bounds=torch.stack([tr_lb, tr_ub]),
                    q=TURBO["q"],
                    num_restarts=TURBO["num_restarts"],
                    raw_samples=TURBO["raw_samples"],
                )
                acq_v = acq_val.item() if hasattr(acq_val, "item") else float(acq_val)
            except Exception:
                new_x_norm = torch.rand(1, dim, dtype=dtype, device=device) * (tr_ub - tr_lb) + tr_lb
                acq_v = -float("inf")

            if acq_v > best_acq_val or best_new_x_norm is None:
                best_acq_val, best_new_x_norm, best_tr_idx = acq_v, new_x_norm, idx

        val = obj_full((lb + best_new_x_norm * (ub - lb))[0].cpu().numpy())
        eval_count += 1
        Y = torch.cat([Y, torch.tensor([[val]], dtype=dtype, device=device)])
        X_norm = torch.cat([X_norm, best_new_x_norm])
        state = states[best_tr_idx]

        reference = state["best_value"] if state["best_value"] != -float("inf") else 0.0
        if val > state["best_value"] + TURBO["success_rel_tol"] * math.fabs(reference):
            state["success_counter"] += 1
            state["failure_counter"] = 0
            state["best_value"] = val
            state["center"] = best_new_x_norm[0].clone()
        else:
            state["success_counter"] = 0
            state["failure_counter"] += 1

        if state["success_counter"] >= state["success_tolerance"]:
            state["length"] = min(2.0 * state["length"], state["length_max"])
            state["success_counter"] = 0
        elif state["failure_counter"] >= state["failure_tolerance"]:
            state["length"] = state["length"] / 2.0
            state["failure_counter"] = 0

        if state["length"] < state["length_min"]:
            state["length"] = TURBO["length"]
            state["failure_counter"] = 0
            state["success_counter"] = 0
            state["center"] = torch.rand(dim, dtype=dtype, device=device)
            state["best_value"] = -float("inf")

    return tracker



def run_pcabo(full_params, full_bounds, base_state, total_iters, seed):
    tracker = _new_tracker()
    dim = len(full_bounds)
    d_e = _embedding_dim(dim, PCABO)
    lb = np.array([b[0] for b in full_bounds])
    ub = np.array([b[1] for b in full_bounds])

    from sklearn.decomposition import PCA

    np.random.seed(seed)
    n_init = min(total_iters, max(PCABO["min_n_init"], d_e + PCABO["n_init_offset"]))
    X_init = np.random.uniform(lb, ub, (n_init, dim))

    initial_y = []
    for x in X_init:
        if len(tracker["history_y"]) >= total_iters:
            return tracker
        initial_y.append(_observe(tracker, evaluate_system_FoM(x, full_params, base_state)))
    if len(tracker["history_y"]) >= total_iters:
        return tracker

    X_init_norm = (X_init - lb) / (ub - lb + 1e-9)
    pca = PCA(n_components=d_e).fit(X_init_norm)
    Z_init = pca.transform(X_init_norm)

    z_mean = np.mean(Z_init, axis=0)
    z_std = np.std(Z_init, axis=0) + 1e-9
    z_lb = np.minimum(Z_init.min(axis=0) - PCABO["latent_pad_sigma"] * z_std,
                      z_mean - PCABO["latent_sigma"] * z_std)
    z_ub = np.maximum(Z_init.max(axis=0) + PCABO["latent_pad_sigma"] * z_std,
                      z_mean + PCABO["latent_sigma"] * z_std)
    Z_init_norm = np.clip((Z_init - z_lb) / (z_ub - z_lb + 1e-9), *PCABO["unit_clip"])

    def obj_pca(z):
        if len(tracker["history_y"]) >= total_iters:
            raise StopIteration()
        x_norm = np.clip(pca.inverse_transform(z.reshape(1, -1))[0], *PCABO["unit_clip"])
        x_real = lb + x_norm * (ub - lb)
        return _observe(tracker, evaluate_system_FoM(x_real, full_params, base_state))

    try:
        ax_bo_optimize(
            obj_pca,
            list(zip(z_lb, z_ub)),
            total_iters - n_init,
            seed=seed,
            initial_x_norm=Z_init_norm.tolist(),
            initial_y=initial_y,
        )
    except StopIteration:
        pass
    return tracker



def run_rembo(full_params, full_bounds, base_state, total_iters, seed):
    tracker = _new_tracker()
    dim = len(full_bounds)
    d_e = _embedding_dim(dim, REMBO)
    np.random.seed(seed)
    projection = np.random.randn(dim, d_e)
    lb = np.array([b[0] for b in full_bounds])
    ub = np.array([b[1] for b in full_bounds])
    clip_lo, clip_hi = REMBO["projection_clip"]

    def obj_rembo(y):
        if len(tracker["history_y"]) >= total_iters:
            raise StopIteration()
        projected = np.clip(projection @ y, clip_lo, clip_hi)
        x_real = lb + (projected + 1.0) / 2.0 * (ub - lb)
        return _observe(tracker, evaluate_system_FoM(x_real, full_params, base_state))

    try:
        latent_box = [(-math.sqrt(d_e), math.sqrt(d_e)) for _ in range(d_e)]
        ax_bo_optimize(obj_rembo, latent_box, total_iters, seed=seed)
    except StopIteration:
        pass
    return tracker




def _standardize_observations(Y):
    valid_mask = Y > -9000.0
    if valid_mask.any() and valid_mask.sum() > 1:
        y_mean = Y[valid_mask].mean()
        y_std = Y[valid_mask].std()
    else:
        y_mean = Y.mean()
        y_std = Y.std()
    if torch.isnan(y_std) or y_std < 1e-9:
        y_std = torch.tensor(1.0, dtype=Y.dtype, device=Y.device)
    train_Y = (Y - y_mean) / y_std
    if (~valid_mask).any() and valid_mask.any():
        train_Y = train_Y.clone()
        train_Y[~valid_mask] = train_Y[valid_mask].min() - 1.0
    return train_Y


def run_saasbo(full_params, full_bounds, base_state, total_iters, seed):
    tracker = _new_tracker()
    device, dtype = torch.device("cpu"), torch.double
    dim = len(full_bounds)
    lb = torch.tensor([b[0] for b in full_bounds], dtype=dtype, device=device)
    ub = torch.tensor([b[1] for b in full_bounds], dtype=dtype, device=device)
    unit_bounds = torch.stack([
        torch.zeros(dim, dtype=dtype, device=device),
        torch.ones(dim, dtype=dtype, device=device),
    ])

    if seed is not None:
        torch.manual_seed(seed)

    n_init = min(SAASBO["n_init"], total_iters)

    def obj_full(x):
        return _observe(tracker, evaluate_system_FoM(x, full_params, base_state))

    X_norm = SobolEngine(dim, scramble=True, seed=seed).draw(n_init).to(dtype=dtype, device=device)
    X_real = lb + X_norm * (ub - lb)
    Y = torch.tensor(
        [obj_full(X_real[i].cpu().numpy()) for i in range(n_init)],
        dtype=dtype, device=device,
    ).unsqueeze(-1)

    while Y.shape[0] < total_iters:
        train_Y = _standardize_observations(Y)
        model = SaasFullyBayesianSingleTaskGP(
            train_X=X_norm,
            train_Y=train_Y,
            train_Yvar=torch.full_like(train_Y, SAASBO["train_yvar"]),
        )
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=SAASBO["warmup_steps"],
            num_samples=SAASBO["num_samples"],
            thinning=SAASBO["thinning"],
            max_tree_depth=SAASBO["max_tree_depth"],
            disable_progbar=True,
        )
        acq = qLogExpectedImprovement(model=model, best_f=train_Y.max())
        try:
            new_x_norm, _ = optimize_acqf(
                acq_function=acq,
                bounds=unit_bounds,
                q=SAASBO["q"],
                num_restarts=SAASBO["num_restarts"],
                raw_samples=SAASBO["raw_samples"],
            )
        except Exception:
            new_x_norm = torch.rand(1, dim, dtype=dtype, device=device)

        val = obj_full((lb + new_x_norm * (ub - lb))[0].cpu().numpy())
        Y = torch.cat([Y, torch.tensor([[val]], dtype=dtype, device=device)])
        X_norm = torch.cat([X_norm, new_x_norm])

    return tracker


RUNNERS = {
    "TurBO": run_turbo,
    "PCABO": run_pcabo,
    "REMBO": run_rembo,
    "SAASBO": run_saasbo,
}


def _seed_trial(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def worker_single_trial(args):
    trial_id, budgets, full_params, full_bounds, base_state, global_seed = args
    run_seed = global_seed + trial_id
    results = {}
    for name, budget in budgets.items():
        _seed_trial(run_seed)
        results[name] = RUNNERS[name](full_params, full_bounds, base_state, budget, run_seed)
    return results


def plot_convergence(plot_data):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 7))
    for algo, rows in plot_data.items():
        if not rows:
            continue
        y_matrix = np.array(rows, dtype=float)
        cummax = np.maximum.accumulate(y_matrix, axis=1)
        mean_curve = np.mean(cummax, axis=0)
        std_curve = np.std(cummax, axis=0)
        x_axis = np.arange(1, len(mean_curve) + 1)
        color = COLORS.get(algo)
        plt.plot(x_axis, mean_curve, label=algo, linewidth=2.5, color=color)
        plt.fill_between(
            x_axis,
            np.maximum(mean_curve - std_curve, 0.0),
            mean_curve + std_curve,
            alpha=0.15,
            color=color,
        )
        print(f"{algo:8}  budget={len(mean_curve):<4}  terminal mean FoM={mean_curve[-1]:.2f}")

    plt.xlabel("Number of Evaluations", fontsize=12, fontweight="bold")
    plt.ylabel("Best FoM Found", fontsize=12, fontweight="bold")
    plt.title("Experiment 1 baselines: 86-d energy storage", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    filename = f"baseline_exp1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"convergence figure saved to {filename}")


def run_baseline_benchmark(n_trials=None, budgets=None, global_seed=None):
    n_trials = EXPERIMENT["n_trials"] if n_trials is None else n_trials
    budgets = EXPERIMENT["budgets"] if budgets is None else budgets
    global_seed = EXPERIMENT["global_seed"] if global_seed is None else global_seed

    G, base_state = build_physics_graph()
    nodes = list(G.nodes())
    full_params = [v for v in INDEPENDENT_VARS if v in nodes and G.in_degree(v) == 0]
    full_bounds = generate_bounds(full_params, base_state)
    print(f"experiment 1: {len(full_params)} independent variables, seed {global_seed}")
    print("budgets:", budgets)

    plot_data = {name: [] for name in budgets}
    worker_args = [
        (i, budgets, full_params, full_bounds, base_state, global_seed)
        for i in range(n_trials)
    ]
    n_workers = max(1, int((os.cpu_count() or 1) * 0.8))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(worker_single_trial, args) for args in worker_args]
        for future in tqdm(as_completed(futures), total=n_trials):
            result = future.result()
            for name, tracker in result.items():
                y_arr = list(tracker["history_y"])
                budget = budgets[name]
                if len(y_arr) < budget:
                    pad = y_arr[-1] if y_arr else 0.0
                    y_arr.extend([pad] * (budget - len(y_arr)))
                plot_data[name].append(y_arr[:budget])

    plot_convergence(plot_data)
    history_filename = f"baseline_exp1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_filename, "w", encoding="utf-8") as f:
        json.dump(plot_data, f)
    print(f"trajectories saved to {history_filename}")
    return plot_data


if __name__ == "__main__":
    run_baseline_benchmark()
