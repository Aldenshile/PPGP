import os
import json
import random
import math
import warnings
from datetime import datetime
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["USE_NUMEXPR"] = "0"
torch.set_num_threads(1)

warnings.filterwarnings('ignore')
from tqdm import tqdm

from BuildGraph import build_physics_graph, evaluate_system_FoM

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from torch.quasirandom import SobolEngine
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

class CausalGravityPPGP:
    def __init__(self, W, target_indices, C_target):
        self.W = np.array(W, dtype=float)
        self.N = self.W.shape[0]
        self.target_indices = target_indices
        self.C_target = C_target
        self.selected_nodes = []
        self.total_capacity = 0.0
        self.node_physics = {}
        self.history = []

    def _apply_shared_path_decay(self, W_cur, selected_node):
        successors = np.where(np.abs(W_cur[selected_node, :]) > 1e-10)[0]
        if len(successors) == 0: return

        W_orig = W_cur.copy()
        old_inflows = np.sum(np.abs(W_orig), axis=0)
        W_temp = W_cur.copy()
        W_temp[selected_node, :] = 0.0

        queue = deque(successors)
        in_queue = set(successors)
        visit_count = np.zeros(self.N)
        max_visits = 15

        while queue:
            curr = queue.popleft()
            in_queue.remove(curr)
            visit_count[curr] += 1
            if visit_count[curr] > max_visits: continue

            new_inflow = np.sum(np.abs(W_temp[:, curr]))
            old_inflow = old_inflows[curr]
            if old_inflow < 1e-10: continue

            retention_ratio = np.clip(new_inflow / old_inflow, 0.0, 1.0)
            neighbors = np.where(np.abs(W_orig[curr, :]) > 1e-10)[0]

            for nxt in neighbors:
                if nxt == selected_node: continue
                orig_w_forward = W_orig[curr, nxt]
                new_w_forward = orig_w_forward * retention_ratio
                if abs(W_temp[curr, nxt] - new_w_forward) > 1e-6:
                    W_temp[curr, nxt] = new_w_forward
                    if nxt not in in_queue:
                        queue.append(nxt)
                        in_queue.add(nxt)
        np.copyto(W_cur, W_temp)

    def _calculate_clean_total_influence(self, W_graph, target_ids, C_cands):
        in_degrees = np.sum(np.abs(W_graph) > 1e-10, axis=0)
        independent_nodes = [c for c in C_cands if in_degrees[c] <= 1e-10]
        if not independent_nodes: independent_nodes = C_cands

        reachable_from_sources = set(independent_nodes)
        queue = deque(independent_nodes)
        while queue:
            curr = queue.popleft()
            neighbors = np.where(np.abs(W_graph[curr, :]) > 1e-10)[0]
            for nxt in neighbors:
                if nxt not in reachable_from_sources:
                    reachable_from_sources.add(nxt)
                    queue.append(nxt)

        can_reach_targets = set(target_ids)
        queue = deque(target_ids)
        while queue:
            curr = queue.popleft()
            predecessors = np.where(np.abs(W_graph[:, curr]) > 1e-10)[0]
            for prev in predecessors:
                if prev not in can_reach_targets:
                    can_reach_targets.add(prev)
                    queue.append(prev)

        active_edges = set()
        for u in reachable_from_sources:
            out_neighbors = np.where(np.abs(W_graph[u, :]) > 1e-10)[0]
            for v in out_neighbors:
                if v in can_reach_targets:
                    active_edges.add((u, v))

        total_influence_pool = 0.0
        processed_pairs = set()

        for u, v in active_edges:
            pair = tuple(sorted((u, v)))
            if pair in processed_pairs: continue
            processed_pairs.add(pair)

            w_uv = W_graph[u, v] if (u, v) in active_edges else 0.0
            w_vu = W_graph[v, u] if (v, u) in active_edges else 0.0

            if w_uv * w_vu > 0:
                pair_capacity = max(abs(w_uv), abs(w_vu))
            elif w_uv * w_vu < 0:
                pair_capacity = abs(abs(w_uv) - abs(w_vu))
            else:
                pair_capacity = max(abs(w_uv), abs(w_vu))
            total_influence_pool += pair_capacity
        return total_influence_pool

    def _compute_capacity_scores(self, W_cur, candidates):
        scores = {}
        for c in candidates:
            score_c = self._calculate_clean_total_influence(W_cur, self.target_indices, [c])
            scores[c] = score_c
        return scores

    def _compute_influence_matrix(self, W_cur):
        M = np.zeros((self.N, self.N), dtype=float)
        Wk = np.eye(self.N)
        for k in range(self.N):
            Wk = Wk @ W_cur
            max_val = np.max(np.abs(Wk))
            if max_val > 1e12:
                Wk = Wk / max_val
            M += Wk
        return M

    def run_cgra(self, ratio=0.8, return_history=False):
        W_cur = self.W.copy()
        initial_scores = self._compute_capacity_scores(W_cur, self.C_target)
        valid_candidates = [c for c in self.C_target if initial_scores[c] > 1e-10]

        if not valid_candidates: return [], 0.0, [] if return_history else None

        self.total_capacity = self._calculate_clean_total_influence(W_cur, self.target_indices, self.C_target)
        target_cumulative = ratio * self.total_capacity
        current_capacity = self.total_capacity

        self.selected_nodes = []
        self.history = []
        cumulative_impact = 0.0
        iteration = 0

        while cumulative_impact < target_cumulative and valid_candidates:
            iteration += 1
            in_degree = np.sum(np.abs(W_cur) > 1e-10, axis=0)
            zero_indegree_candidates = [c for c in valid_candidates if in_degree[c] == 0]
            if not zero_indegree_candidates: break

            candidate_scores = self._compute_capacity_scores(W_cur, zero_indegree_candidates)
            best_node = max(candidate_scores, key=candidate_scores.get)

            valid_candidates.remove(best_node)
            self._apply_shared_path_decay(W_cur, best_node)
            W_cur[best_node, :] = 0.0
            W_cur[:, best_node] = 0.0

            cap_after = self._calculate_clean_total_influence(W_cur, self.target_indices, self.C_target)
            actual_node_impact = current_capacity - cap_after

            self.selected_nodes.append((best_node, actual_node_impact))
            cumulative_impact += actual_node_impact
            current_capacity = cap_after

        return self.selected_nodes, self.total_capacity, self.history if return_history else None

def plot_convergence(plot_data, max_iters):
    plt.figure(figsize=(12, 7))
    x_axis = np.arange(1, max_iters + 1)

    colors = {
        'PPGP-Acqf+TurBO': '#1f77b4',
        'PPGP-GP-Prior+TurBO': '#ff7f0e',
        'PPGP-VolumePreserved+TurBO': '#2ca02c',
        'PPGP-Final-All+TurBO': '#9467bd',
        'Pure-TurBO-m': '#d62728'
    }

    for algo, y_matrix in plot_data.items():
        if not y_matrix: continue
        y_matrix = np.array(y_matrix)
        cummax_matrix = np.maximum.accumulate(y_matrix, axis=1)

        mean_curve = np.mean(cummax_matrix, axis=0)
        std_curve = np.std(cummax_matrix, axis=0)

        c = colors.get(algo, None)
        lw = 3.5 if algo == 'PPGP-Final-All+TurBO' else 2.5
        zorder = 5 if algo == 'PPGP-Final-All+TurBO' else 3

        p = plt.plot(x_axis, mean_curve, label=algo, linewidth=lw, color=c, zorder=zorder)
        plt.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=p[0].get_color(),
                         zorder=zorder - 1)

    plt.xlabel('Number of Evaluations (Rounds)', fontsize=12, fontweight='bold')
    plt.ylabel('Best FOM Found', fontsize=12, fontweight='bold')
    plt.title('Optimization Convergence (Unified Mahalanobis Manifold)', fontsize=14, fontweight='bold')
    plt.xlim(1, max_iters)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    filename = f'convergence_cgra_unified_manifold_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300)
    plt.close()

def generate_bounds(param_names, base_state):
    bounds = []
    for p in param_names:
        val = base_state[p]
        if val == 0.0:
            low, high = 1e-6, 0.1
        else:
            low, high = val * 0.85, val * 1.15

        if any(key in p for key in
               ['ratio', 'porosity', 'degree', 'emissivity', 'fraction', 'excess', 'rate', 'factor', 'accuracy']):
            low, high = max(0.01, low), min(0.99, high)
        elif 'packing_density' in p:
            low, high = max(0.1, low), min(1.0, high)
        elif any(key in p for key in
                 ['thickness', 'area', 'radius', 'gap', 'size', 'width', 'depth', 'cross_section', 'length', 'volume',
                  'pitch']):
            low = max(1e-7, low)
        elif 'temp' in p:
            low = max(253.15, low)
        bounds.append((low, high))
    return bounds

def botorch_turbom_optimize(obj_func, bounds, max_evals, m=2, seed=None,
                            algo_mode='pure', cgra_weights=None, multi_hop_weights=None, vp_weights=None,
                            acq_inject_epochs=0, prior_inject_epochs=0, recovery_iter=0,
                            recompute_weights_callback=None):
    if max_evals <= 0: return [], []
    device, dtype = torch.device("cpu"), torch.double
    dim = len(bounds)
    lb = torch.tensor([b[0] for b in bounds], dtype=dtype, device=device)
    ub = torch.tensor([b[1] for b in bounds], dtype=dtype, device=device)

    if seed is not None: torch.manual_seed(seed)
    adaptive_fail_tol = 5
    n_init = min(10, max_evals // 2)

    X_norm = SobolEngine(dim, scramble=True, seed=seed).draw(n_init).to(dtype=dtype, device=device)
    X_real = lb + X_norm * (ub - lb)
    Y = torch.tensor([obj_func(X_real[i].cpu().numpy()) for i in range(n_init)], dtype=dtype, device=device).unsqueeze(
        -1)

    sorted_indices = torch.argsort(Y.squeeze(-1), descending=True)
    states = []
    for i in range(min(m, len(Y))):
        idx = sorted_indices[i]
        states.append({
            "length": 0.8, "length_min": 0.5 ** 7, "length_max": 1.6,
            "failure_counter": 0, "failure_tolerance": adaptive_fail_tol,
            "success_counter": 0, "success_tolerance": 3,
            "best_value": Y[idx].item(), "center": X_norm[idx].clone()
        })

    eval_count = n_init
    last_recompute_best_val = -float('inf')

    while eval_count < max_evals:
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

        bo_iter = max(0, eval_count - n_init)

        tr_weights = torch.ones(dim, dtype=dtype, device=device)
        w_norm_acq = None
        w_norm_acq_final = None

        if algo_mode == 'final' and multi_hop_weights is not None:
            U_tensor = torch.tensor(multi_hop_weights, dtype=dtype, device=device)
            U_norm = (U_tensor - U_tensor.min()) / (U_tensor.max() - U_tensor.min() + 1e-12)

            if bo_iter < prior_inject_epochs:
                U_abs = torch.abs(U_tensor)
                U_log = torch.log1p(U_abs)
                U_norm = (U_log - U_log.min()) / (U_log.max() - U_log.min() + 1e-12)
                l_min = 0.5
                l_max = math.sqrt(dim)
                alpha_val = (1.0 / l_min) - (1.0 / l_max)
                beta_val = 1.0 / l_max
                target_mean = (1.0 / (alpha_val * U_norm + beta_val)).unsqueeze(0)
                concentration = torch.full((1, dim), 3.0, dtype=dtype, device=device)
                custom_prior = GammaPrior(concentration, concentration / target_mean)
                covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_prior=custom_prior),
                    outputscale_prior=GammaPrior(2.0, 0.15)
                )
                model = SingleTaskGP(X_norm, train_Y, covar_module=covar_module)
                model.covar_module.base_kernel.lengthscale = target_mean.clone()
            else:
                model = SingleTaskGP(X_norm, train_Y)

            tr_target = 1.0 - 0.8 * U_norm
            alpha_tr = max(0.0, 1.0 - (bo_iter / (max_evals * 0.8)))
            tr_weights = tr_target * alpha_tr + 1.0 * (1.0 - alpha_tr)
            w_norm_acq_final = 1.0 - 0.75 * U_norm
        else:
            if algo_mode == 'gp_prior' and multi_hop_weights is not None and bo_iter < prior_inject_epochs:
                mh_tensor = torch.tensor(multi_hop_weights, dtype=dtype, device=device)
                mh_norm = (mh_tensor - mh_tensor.min()) / (mh_tensor.max() - mh_tensor.min() + 1e-12)
                target_mean = (6.0 - 4.0 * mh_norm).unsqueeze(0)
                concentration = torch.full((1, dim), 3.0, dtype=dtype, device=device)
                covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=dim,
                                 lengthscale_prior=GammaPrior(concentration, concentration / target_mean)),
                    outputscale_prior=GammaPrior(2.0, 0.15)
                )
                model = SingleTaskGP(X_norm, train_Y, covar_module=covar_module)
                model.covar_module.base_kernel.lengthscale = target_mean.clone()
            else:
                model = SingleTaskGP(X_norm, train_Y)

            if algo_mode == 'vp_tr' and cgra_weights is not None:
                alpha = max(0.0, 1.0 - (bo_iter / max(1, recovery_iter)))
                w_tensor = torch.tensor(cgra_weights, dtype=dtype, device=device)
                w_soft = w_tensor + 0.05
                w_scaled = w_soft / torch.exp(torch.mean(torch.log(w_soft)))
                tr_weights = w_scaled * alpha + 1.0 * (1.0 - alpha)

            if algo_mode == 'acq_inject' and cgra_weights is not None and bo_iter < acq_inject_epochs:
                w_tensor_acq = torch.tensor(cgra_weights, dtype=dtype, device=device)
                min_w = max(0.1 * (w_tensor_acq.max().item() if w_tensor_acq.max().item() > 1e-9 else 1.0), 0.2)
                w_norm_acq = torch.clamp(
                    torch.clamp(w_tensor_acq, min=min_w) / torch.clamp(w_tensor_acq, min=min_w).max(), min=min_w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            except Exception:
                pass
        EI = ExpectedImprovement(model, best_f=train_Y.max())
        best_acq_val, best_new_x_norm, best_tr_idx = -float('inf'), None, -1

        for idx, state in enumerate(states):
            anisotropic_lengths = state["length"] * tr_weights
            anisotropic_lengths = torch.clamp(anisotropic_lengths, state["length_min"], 2.0)

            tr_lb = torch.clamp(state["center"] - anisotropic_lengths / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(state["center"] + anisotropic_lengths / 2.0, 0.0, 1.0)

            try:
                if algo_mode == 'final' and w_norm_acq_final is not None:
                    num_raw_samples = 512
                    num_restarts = 5
                    normal_samples = torch.randn(num_raw_samples, 1, dim, dtype=dtype, device=device)
                    boundary_widths = tr_ub - tr_lb
                    biased_disturbances = normal_samples * 0.25 * boundary_widths * w_norm_acq_final
                    biased_raw_points = torch.clamp(state["center"] + biased_disturbances, tr_lb, tr_ub)

                    with torch.no_grad():
                        ei_vals = EI(biased_raw_points)
                    _, top_indices = torch.topk(ei_vals.squeeze(-1), num_restarts)
                    new_x_norm, acq_val = optimize_acqf(
                        acq_function=EI, bounds=torch.stack([tr_lb, tr_ub]), q=1,
                        num_restarts=num_restarts, batch_initial_conditions=biased_raw_points[top_indices]
                    )

                elif algo_mode == 'acq_inject' and w_norm_acq is not None:
                    num_raw_samples = 512
                    num_restarts = 5
                    uniform_samples = torch.rand(num_raw_samples, 1, dim, dtype=dtype, device=device)
                    biased_disturbances = (uniform_samples - 0.5) * (tr_ub - tr_lb) * w_norm_acq
                    biased_raw_points = torch.clamp(state["center"] + biased_disturbances, tr_lb, tr_ub)

                    with torch.no_grad():
                        ei_vals = EI(biased_raw_points)
                    _, top_indices = torch.topk(ei_vals.squeeze(-1), num_restarts)
                    new_x_norm, acq_val = optimize_acqf(
                        acq_function=EI, bounds=torch.stack([tr_lb, tr_ub]), q=1,
                        num_restarts=num_restarts, batch_initial_conditions=biased_raw_points[top_indices]
                    )
                else:
                    new_x_norm, acq_val = optimize_acqf(
                        acq_function=EI, bounds=torch.stack([tr_lb, tr_ub]), q=1,
                        num_restarts=5, raw_samples=512
                    )

                acq_v = acq_val.item() if hasattr(acq_val, 'item') else float(acq_val)
            except Exception as e:
                new_x_norm = torch.rand(1, dim, dtype=dtype, device=device) * (tr_ub - tr_lb) + tr_lb
                acq_v = -float('inf')

            if acq_v > best_acq_val or best_new_x_norm is None:
                best_acq_val, best_new_x_norm, best_tr_idx = acq_v, new_x_norm, idx

        val = obj_func((lb + best_new_x_norm * (ub - lb))[0].cpu().numpy())
        if np.isnan(val) or np.isinf(val): val = -9999.0

        eval_count += 1
        Y = torch.cat([Y, torch.tensor([[val]], dtype=dtype, device=device)])
        X_norm = torch.cat([X_norm, best_new_x_norm])
        state = states[best_tr_idx]

        if val > state["best_value"] + 1e-3 * math.fabs(
                state["best_value"] if state["best_value"] != -float('inf') else 0):
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
            current_global_best = Y.max().item()
            if recompute_weights_callback is not None:
                if current_global_best > last_recompute_best_val + 1e-5:
                    last_recompute_best_val = current_global_best

                    best_idx = torch.argmax(Y)
                    best_x_norm = X_norm[best_idx]
                    best_x_real = (lb + best_x_norm * (ub - lb)).cpu().numpy()

                    new_mh_weights = recompute_weights_callback(best_x_real)
                    if new_mh_weights is not None:
                        multi_hop_weights = new_mh_weights

                    state["failure_counter"] = 0
                else:
                    state["length"] = state["length"] / 2.0
                    state["failure_counter"] = 0
            else:
                state["length"] = state["length"] / 2.0
                state["failure_counter"] = 0

        if state["length"] < state["length_min"]:
            state["length"] = 0.8
            state["failure_counter"] = 0
            state["success_counter"] = 0
            state["center"] = torch.rand(dim, dtype=dtype, device=device)
            state["best_value"] = -float('inf')

    return X_norm, Y

def run_cgra_acqf_turbo(full_params, full_bounds, base_state, total_iters, seed, acq_capacity_weights,
                        acq_inject_epochs):
    tracker = {'eval_count': 0, 'best_fom': -np.inf, 'history_y': []}

    def obj_full(x):
        val = evaluate_system_FoM(x, full_params, base_state)
        if val > tracker['best_fom']: tracker['best_fom'] = val
        tracker['history_y'].append(val)
        return val

    botorch_turbom_optimize(obj_full, full_bounds, total_iters, m=2, seed=seed, algo_mode='acq_inject',
                            cgra_weights=acq_capacity_weights, acq_inject_epochs=acq_inject_epochs)
    return tracker

def run_cgra_gp_prior_turbo(full_params, full_bounds, base_state, total_iters, seed, multi_hop_weights,
                            prior_inject_epochs):
    tracker = {'eval_count': 0, 'best_fom': -np.inf, 'history_y': []}

    def obj_full(x):
        val = evaluate_system_FoM(x, full_params, base_state)
        if val > tracker['best_fom']: tracker['best_fom'] = val
        tracker['history_y'].append(val)
        return val

    botorch_turbom_optimize(obj_full, full_bounds, total_iters, m=2, seed=seed, algo_mode='gp_prior',
                            multi_hop_weights=multi_hop_weights, prior_inject_epochs=prior_inject_epochs)
    return tracker

def run_cgra_vp_turbo(full_params, full_bounds, base_state, total_iters, seed, vp_cgra_weights, recovery_iter):
    tracker = {'eval_count': 0, 'best_fom': -np.inf, 'history_y': []}

    def obj_full(x):
        val = evaluate_system_FoM(x, full_params, base_state)
        if val > tracker['best_fom']: tracker['best_fom'] = val
        tracker['history_y'].append(val)
        return val

    botorch_turbom_optimize(obj_full, full_bounds, total_iters, m=2, seed=seed, algo_mode='vp_tr',
                            cgra_weights=vp_cgra_weights, recovery_iter=recovery_iter)
    return tracker

def run_cgra_final_turbo(full_params, full_bounds, base_state, total_iters, seed, acq_capacity_weights,
                         multi_hop_weights, vp_cgra_weights, acq_inject_epochs, prior_inject_epochs, recovery_iter,
                         recompute_weights_callback=None):
    tracker = {'eval_count': 0, 'best_fom': -np.inf, 'history_y': []}

    def obj_full(x):
        val = evaluate_system_FoM(x, full_params, base_state)
        if val > tracker['best_fom']: tracker['best_fom'] = val
        tracker['history_y'].append(val)
        return val

    botorch_turbom_optimize(obj_full, full_bounds, total_iters, m=2, seed=seed, algo_mode='final',
                            cgra_weights=acq_capacity_weights, multi_hop_weights=multi_hop_weights,
                            vp_weights=vp_cgra_weights,
                            acq_inject_epochs=acq_inject_epochs, prior_inject_epochs=prior_inject_epochs,
                            recovery_iter=recovery_iter,
                            recompute_weights_callback=recompute_weights_callback)
    return tracker

def run_pure_turbo(full_params, full_bounds, base_state, total_iters, seed):
    tracker = {'eval_count': 0, 'best_fom': -np.inf, 'history_y': []}

    def obj_full(x):
        val = evaluate_system_FoM(x, full_params, base_state)
        if val > tracker['best_fom']: tracker['best_fom'] = val
        tracker['history_y'].append(val)
        return val

    botorch_turbom_optimize(obj_full, full_bounds, total_iters, m=2, seed=seed, algo_mode='pure')
    return tracker

def worker_single_trial(args):
    trial_id, max_iters, full_params, full_bounds, base_state, global_seed, \
        acq_capacity_weights, multi_hop_weights, vp_cgra_weights, \
        acq_inject_epochs, prior_inject_epochs, recovery_iter, \
        nodes, target_indices = args

    run_seed = global_seed + trial_id
    np.random.seed(run_seed)
    random.seed(run_seed)
    try:
        torch.manual_seed(run_seed)
    except:
        pass

    results = {}
    results['Pure-TurBO-m'] = run_pure_turbo(full_params, full_bounds, base_state, max_iters, run_seed)

    results['PPGP-Final-All+TurBO'] = run_cgra_final_turbo(
        full_params, full_bounds, base_state, max_iters, run_seed,
        acq_capacity_weights, multi_hop_weights, vp_cgra_weights,
        acq_inject_epochs, prior_inject_epochs, recovery_iter)

    return results

def run_ablation_benchmark(n_trials, max_iters, acq_inject_epochs=3, prior_inject_epochs=5, recovery_iter=5, global_seed=40):
    G, base_state = build_physics_graph()
    nodes = list(G.nodes())
    W = nx.to_numpy_array(G, nodelist=nodes, weight='weight')

    independent_vars = [
        'anode_thickness', 'cathode_thickness', 'separator_thickness', 'electrode_gap', 'cell_width', 'cell_length',
        'cell_height', 'tab_width', 'tab_thickness', 'jellyroll_radius', 'particle_size_anode', 'particle_size_cathode',
        'binder_ratio', 'conductive_additive_ratio', 'separator_porosity', 'electrolyte_concentration',
        'pre_lithiation_degree', 'packing_density', 'solid_electrolyte_fraction', 'lithium_excess', 'coolant_flow',
        'chiller_temp_setpoint', 'tim_thickness', 'tim_thermal_conductivity', 'base_plate_thickness', 'fin_thickness',
        'fin_height', 'fin_pitch', 'cooling_channel_width', 'cooling_channel_depth', 'pump_speed', 'fan_speed',
        'compressor_power', 'coolant_viscosity_ref', 'coolant_density_ref', 'fan_blade_radius', 'fan_blade_angle',
        'pump_impeller_diameter', 'nozzle_diameter', 'pipe_roughness', 'wire_radius', 'busbar_cross_section',
        'busbar_length', 'inverter_switching_freq', 'gate_resistance', 'mosfet_rdson', 'dc_link_cap', 'inductor_volume',
        'winding_turns', 'core_cross_section', 'housing_thickness', 'housing_surface_area', 'casing_emissivity',
        'vibration_mount_stiffness', 'vibration_mount_damping', 'fastener_torque', 'weld_seam_length',
        'potting_compound_thickness', 'gasket_thickness', 'gasket_compression', 'calendering_pressure',
        'formation_current', 'drying_temperature', 'drying_time', 'coating_speed', 'welding_power', 'assembly_pressure',
        'aging_temperature', 'aging_time', 'inspection_rate', 'soc_target', 'dod_max', 'c_rate_max', 'voltage_max',
        'voltage_min', 'temp_limit_high', 'temp_limit_low', 'power_limit_surge', 'current_limit_continuous',
        'derating_factor', 'cell_spacing', 'pack_volume_margin', 'bms_sampling_rate', 'sensor_accuracy',
        'isolation_resistance', 'ambient_temp'
    ]

    full_params = [v for v in independent_vars if v in nodes and G.in_degree(v) == 0]
    full_bounds = generate_bounds(full_params, base_state)
    target_indices = [nodes.index(name) for name in ['Global_FoM'] if name in nodes]

    cgra_analyzer = CausalGravityPPGP(W=W, target_indices=target_indices,
                                      C_target=[nodes.index(p) for p in full_params])

    initial_scores = cgra_analyzer._compute_capacity_scores(cgra_analyzer.W, cgra_analyzer.C_target)
    acq_capacity_weights = [initial_scores[nodes.index(p)] for p in full_params]

    M_influence = cgra_analyzer._compute_influence_matrix(cgra_analyzer.W)
    multi_hop_scores = [np.abs(np.mean([M_influence[nodes.index(p), t] for t in target_indices])) for p in full_params]
    max_mh = max(multi_hop_scores) if multi_hop_scores else 1.0
    multi_hop_weights = [score / (max_mh + 1e-12) for score in multi_hop_scores]

    max_score = max(initial_scores.values()) if initial_scores else 1.0
    vp_cgra_weights = [initial_scores[nodes.index(p)] / (max_score + 1e-12) for p in full_params]

    target_cores = max(1, int((os.cpu_count() or 1) * 0.8))

    algos = ['PPGP-Acqf+TurBO', 'PPGP-GP-Prior+TurBO', 'PPGP-VolumePreserved+TurBO', 'PPGP-Final-All+TurBO',
             'Pure-TurBO-m']
    plot_data = {a: [] for a in algos}

    worker_args = [(i, max_iters, full_params, full_bounds, base_state, global_seed,
                    acq_capacity_weights, multi_hop_weights, vp_cgra_weights,
                    acq_inject_epochs, prior_inject_epochs, recovery_iter,
                    nodes, target_indices) for i in range(n_trials)]

    with ProcessPoolExecutor(max_workers=target_cores) as executor:
        futures = [executor.submit(worker_single_trial, args) for args in worker_args]
        pbar = tqdm(as_completed(futures), total=n_trials)

        for future in pbar:
            res = future.result()
            for algo, data in res.items():
                y_arr = data['history_y'].copy()
                if len(y_arr) < max_iters:
                    last_val = y_arr[-1] if len(y_arr) > 0 else 0
                    y_arr.extend([last_val] * (max_iters - len(y_arr)))
                plot_data[algo].append(y_arr[:max_iters])

    plot_convergence(plot_data, max_iters)

    history_filename = f'fom_history_unified_manifold_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(history_filename, 'w', encoding='utf-8') as f:
        json.dump(plot_data, f, indent=4)

if __name__ == "__main__":
    GLOBAL_SEED = 40
    run_ablation_benchmark(
        n_trials=100,
        max_iters=200,
        global_seed=GLOBAL_SEED
    )