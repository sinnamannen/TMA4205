
from image_generation import generate_test_image
from conjugate_gradient import cg_main
from multigrid import multigrid_main_iterative
from PCG import pcg_main
import matplotlib.pyplot as plt
import numpy as np
from image_generation import generate_test_image, mycomputeColor, mycolorwheel
import time

def plot_method_solution(N=256, testcase=1, method="CG",
                         mg_s1=2, mg_s2=2, mg_max_level=3):
    """
    Visualizes optical flow for the chosen testcase and method.
    Only produces plots. Does NOT return values.
    """

    # --- Generate synthetic test images ---
    I1, I2 = generate_test_image(N, testcase=testcase)

    # --- Set regularization ---
    k = int(np.log2(N))
    reg = 4 ** (k - 4)

    # --- Solve optical flow depending on method ---
    if method == "CG":
        u, v, _, _, _ = cg_main(I1, I2, reg)

    elif method == "MG":
        u, v, _, _ = multigrid_main_iterative(
            I1, I2, reg,
            s1=mg_s1,
            s2=mg_s2,
            max_level=mg_max_level,
            tol=1e-8,
            max_cycles=50
        )

    elif method == "PCG":
        u, v, _, _, _ = pcg_main(
            I1, I2, reg,
            max_level=mg_max_level,
            s1=mg_s1,
            s2=mg_s2,
            tol=1e-8,
            maxit=500
        )

    else:
        raise ValueError("method must be 'CG', 'MG', or 'PCG'") 

    # --- Convert optical flow to a color-coded image ---
    flow_img = mycomputeColor(u, v)
    color_wheel = mycolorwheel(55)

    # --- Plot everything ---
    fig = plt.figure(figsize=(15, 5))

    fig.suptitle(
        f"Method = {method},  N = {N},  Testcase = {testcase}",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # Left: Image 1
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(I1, cmap="gray")
    ax1.set_title(f"Image 1")
    ax1.axis("off")

    # Middle: Image 2
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(I2, cmap="gray")
    ax2.set_title("Image 2")
    ax2.axis("off")

    # Right: Flow visualization
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(flow_img)
    ax3.set_title(f"Optical Flow ({method})")
    ax3.axis("off")

    # Add color wheel inset
    inset = ax3.inset_axes([0.05, 0.05, 0.25, 0.25])
    inset.imshow(color_wheel)
    inset.set_title("Color Wheel", fontsize=8)
    inset.axis("off")

    plt.tight_layout()
    plt.show()



def run_single_method(method, ks, testcase=1, mg_s1=2, mg_s2=2, mg_max_level=3):

    result = {"times": [], "residuals": []}

    for k in ks:
        reg = 4 ** (k - 4)
        N = 2 ** k

        I1, I2 = generate_test_image(N, testcase=testcase)

        # ---------------- CG ----------------
        if method == "CG":
            u, v, res, max_iter, elapsed = cg_main(I1, I2, reg)

        # ---------------- PCG ----------------
        elif method == "PCG":
            u, v, res, max_iter, elapsed = pcg_main(
                I1, I2, reg,
                max_level=mg_max_level,
                s1=mg_s1,
                s2=mg_s2,
                tol=1e-8,
                maxit=200
            )

        # ---------------- MG ----------------
        elif method == "MG":
            u, v, res, elapsed = multigrid_main_iterative(
                I1, I2, reg,
                s1=mg_s1,
                s2=mg_s2,
                max_level=mg_max_level,
                tol=1e-8,
                max_cycles=50
            )

        # Error handling
        else:
            raise ValueError(f"Unknown method: {method}")

        # normalize residuals
        res = np.array(res) / res[0]

        result["times"].append(elapsed)
        result["residuals"].append(res)

    return result


def plot_method_summary(ks, results, method="CG"):

    Ns = [2**k for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ------------------- CONVERGENCE PLOT -------------------
    ax1 = axes[0]
    for k, res in zip(ks, results["residuals"]):
        ax1.plot(res, label=f"k={k}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration / V-cycle")
    ax1.set_ylabel("Relative residual")
    ax1.set_title(f"{method} – Convergence History")
    ax1.grid(True)
    ax1.legend()

    # ------------------- TIME SCALING PLOT -------------------
    ax2 = axes[1]
    ax2.plot(Ns, results["times"], "-o", linewidth=2)

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")

    ax2.set_xlabel("Grid size N = 2^k")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title(f"{method} – Time Scaling")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


################ MULTIGRID

def run_mg_parameter_sweep(
    k=7,
    testcase=2,
    s1_values=(1, 2, 3),
    s2_values=(1, 2, 3),
    level_values=(2, 3, 4),
    tol=1e-8,
    max_cycles=50
):
    """
    Parameter sweep for the multigrid solver on a fixed grid size 2^k x 2^k.

    Parameters
    ----------
    k : int
        Exponent for grid size N = 2^k.
    testcase : int
        Synthetic testcase (usually 2 for the circling Gaussians).
    s1_values : iterable
        Values of pre-smoothing steps to test.
    s2_values : iterable
        Values of post-smoothing steps to test.
    level_values : iterable
        Values of max_level (number of levels in the V-cycle) to test.
    tol : float
        Convergence tolerance passed to multigrid_main_iterative.
    max_cycles : int
        Maximum number of V-cycles.

    Returns
    -------
    results : dict
        Nested dictionary:
        results[(max_level, s1, s2)] = {
            "time": total_solve_time,
            "cycles": num_cycles,
            "residuals": np.array(residuals),
            "conv_factor": convergence_factor
        }
    """

    N = 2**k
    reg = 4**(k - 4)

    # generate synthetic images once
    I1, I2 = generate_test_image(N, testcase=testcase)

    results = {}

    for L in level_values:
        for s1 in s1_values:
            for s2 in s2_values:

                # Run multigrid with these parameters
                u, v, res, elapsed = multigrid_main_iterative(
                    I1, I2, reg,
                    s1=s1,
                    s2=s2,
                    max_level=L,
                    tol=tol,
                    max_cycles=max_cycles
                )

                res = np.array(res)

                # number of cycles (assuming one residual per cycle)
                cycles = len(res) - 1 if len(res) > 1 else 0

                # approximate convergence factor per cycle
                if len(res) > 1 and res[0] > 0:
                    conv_factor = (res[-1] / res[0])**(1.0 / max(cycles, 1))
                else:
                    conv_factor = np.nan

                results[(L, s1, s2)] = {
                    "time": elapsed,
                    "cycles": cycles,
                    "residuals": res,
                    "conv_factor": conv_factor,
                }

                #print(f"MG sweep: L={L}, s1={s1}, s2={s2} -> "f"time={elapsed:.3f}s, cycles={cycles}, ρ≈{conv_factor:.3f}")

    return results

def summarize_mg_sweep(results, top_n=5):
    """
    Print the top_n best MG parameter combinations sorted by time.
    """

    entries = []
    for (L, s1, s2), data in results.items():
        entries.append((data["time"], data["cycles"], data["conv_factor"], L, s1, s2))

    entries.sort(key=lambda x: x[0])  # sort by time

    print("\nBest multigrid parameter combinations (sorted by time):")
    print(" time [s]  cycles   conv_factor    L   s1   s2")
    for i, (t, cyc, rho, L, s1, s2) in enumerate(entries[:top_n]):
        print(f" {t:7.3f}  {cyc:6d}   {rho:11.3f}   {L:2d}  {s1:3d}  {s2:3d}")


def results_to_matrices(results, levels=[2,3,4], svals=[1,2,3]):
    """
    Convert raw MG results into structured matrices:
    - rho_mat[L_idx, s1_idx, s2_idx]
    - time_mat[L_idx, s1_idx, s2_idx]

    results is a list of dicts with keys:
        "L", "s1", "s2", "rho", "time"
    """

    nL = len(levels)
    ns = len(svals)

    rho_mat  = np.full((nL, ns, ns), np.nan)
    time_mat = np.full((nL, ns, ns), np.nan)

    for r in results:
        L  = r["L"]
        s1 = r["s1"]
        s2 = r["s2"]

        i = levels.index(L)
        j = svals.index(s1)
        k = svals.index(s2)

        rho_mat[i, j, k]  = r["rho"]
        time_mat[i, j, k] = r["time"]

    return rho_mat, time_mat



def plot_mg_heatmaps(levels, Svals, rho_mat, time_mat):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Convergence factor heatmap ---
    im1 = axes[0].imshow(rho_mat, cmap="viridis", origin="lower")
    axes[0].set_title("MG Convergence Factor ρ")
    axes[0].set_xticks(range(len(Svals)))
    axes[0].set_xticklabels(Svals)
    axes[0].set_yticks(range(len(levels)))
    axes[0].set_yticklabels(levels)
    axes[0].set_xlabel("Total smoothing steps S = s1 + s2")
    axes[0].set_ylabel("Number of levels L")
    fig.colorbar(im1, ax=axes[0])

    # --- Runtime heatmap ---
    im2 = axes[1].imshow(time_mat, cmap="magma", origin="lower")
    axes[1].set_title("MG Runtime (seconds)")
    axes[1].set_xticks(range(len(Svals)))
    axes[1].set_xticklabels(Svals)
    axes[1].set_yticks(range(len(levels)))
    axes[1].set_yticklabels(levels)
    axes[1].set_xlabel("Total smoothing steps S = s1 + s2")
    axes[1].set_ylabel("Number of levels L")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()

def mg_parameter_sweep(k=7, testcase=2,
                       levels=[2, 3, 4],
                       smooth_vals=[1, 2, 3],
                       tol=1e-8,
                       max_cycles=200):

    N = 2**k
    reg = 4**(k - 4)

    I1, I2 = generate_test_image(N, testcase=testcase)

    results = []

    for L in levels:
        for s1 in smooth_vals:
            for s2 in smooth_vals:

                # Run MG solver
                u, v, res, elapsed = multigrid_main_iterative(
                    I1, I2, reg,
                    s1=s1,
                    s2=s2,
                    max_level=L,
                    tol=tol,
                    max_cycles=max_cycles
                )

                res = np.array(res)
                rho = (res[-1] / res[0])**(1 / len(res))   # convergence factor

                results.append({
                    "L": L,
                    "s1": s1,
                    "s2": s2,
                    "time": elapsed,
                    "cycles": len(res),
                    "rho": rho
                })

                #print(f"MG sweep: L={L}, s1={s1}, s2={s2}, "f"time={elapsed:.3f}s, cycles={len(res)}, rho={rho:.3f}")

    return results



def sweep_over_k_for_iterations(
    k_values=(6, 7, 8, 9),
    testcase=2,
    levels=(2, 3, 4),
    s1_values=(1, 2, 3),
    s2_values=(1, 2, 3),
    tol=1e-8,
    max_cycles=50
):
    """
    Runs MG parameter sweep over multiple image sizes (k = 6..9).
    Returns a dict:
    iteration_data[(L, s1, s2)] = { k : cycles }
    """

    iteration_data = {}

    for k in k_values:
        print(f"\n=== Running MG sweep for k={k} (size = {2**k}) ===")

        # Run your existing sweep for this k
        results = run_mg_parameter_sweep(
            k=k,
            testcase=testcase,
            s1_values=s1_values,
            s2_values=s2_values,
            level_values=levels,
            tol=tol,
            max_cycles=max_cycles
        )

        # Store only iteration counts
        for params, data in results.items():
            (L, s1, s2) = params
            cycles = data["cycles"]

            if (L, s1, s2) not in iteration_data:
                iteration_data[(L, s1, s2)] = {}

            iteration_data[(L, s1, s2)][k] = cycles

    return iteration_data

def rank_by_iteration_trend(iteration_data, k_values=(6,7,8,9)):
    """
    Produces a ranking score based on:
    (1) lowest total iterations
    (2) smallest variation across k
    """

    ranking = []

    for params, k_dict in iteration_data.items():
        cycles_list = [k_dict.get(k, np.inf) for k in k_values]

        total_cycles = sum(cycles_list)
        variation = max(cycles_list) - min(cycles_list)

        score = total_cycles + variation  # lower = better

        ranking.append((score, total_cycles, variation, params))

    ranking.sort(key=lambda x: x[0])
    return ranking

def pick_top_configs(ranking, top_n=3):
    return [entry[-1] for entry in ranking[:top_n]]
