
from image_generation import generate_test_image
from conjugate_gradient import cg_main
from multigrid import multigrid_main_iterative
from PCG import pcg_main
import matplotlib.pyplot as plt
import numpy as np
from image_generation import generate_test_image, mycomputeColor, mycolorwheel
import time


# FOR THEORY
def plot_gerhsgorin_disks():
    # --- Dirichlet setup ---
    center_dir = 4
    radii_dir = [2, 3, 4]
    colors_dir = ['tab:blue', 'tab:orange', 'tab:green']

    # --- Neumann setup ---
    # Centers depend on a_ii = {2, 3, 4}, radii equal to those values
    centers_neu = [2, 3, 4]
    radii_neu = [2, 3, 4]
    colors_neu = ['tab:blue', 'tab:orange', 'tab:green']

    # --- Create subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # === Dirichlet plot ===
    ax = axes[0]
    for r, c in zip(radii_dir, colors_dir):
        circle = plt.Circle((center_dir, 0), r, color=c, fill=False, lw=2, label=fr"$r_i={r}$")
        ax.add_patch(circle)

    ax.plot(center_dir, 0, 'ko', label=r"$a_{ii}=4$")
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1, 9)
    ax.set_ylim(-5, 5)
    ax.set_title("Dirichlet B.C.")
    ax.set_xlabel("Real axis")
    ax.set_ylabel("Imaginary axis")
    ax.legend()

    # === Neumann plot ===
    ax = axes[1]
    for cen, r, c in zip(centers_neu, radii_neu, colors_neu):
        circle = plt.Circle((cen, 0), r, color=c, fill=False, lw=2, label=fr"$a_{{ii}}={cen},\ r_i={r}$")
        ax.add_patch(circle)

    ax.plot(centers_neu, [0]*len(centers_neu), 'ko')
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-3, 9)
    ax.set_ylim(-5, 5)
    ax.set_title("Neumann B.C.")
    ax.set_xlabel("Real axis")
    ax.legend()

    # --- Global figure title ---
    fig.suptitle("Gershgorin Disks for $A_1^{dir}$ and $A_1^{neu}$", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_method_solution(N=256, testcase=1, method="CG",
                         mg_s1=2, mg_s2=2, mg_max_level=3, sigma=0):

    # --- Generate test images ---
    I1, I2 = generate_test_image(N, testcase=testcase)

    # --- Regularization ---
    k = int(np.log2(N))
    reg = 4 ** (k - 4) * 64

    # --- Solve optical flow ---
    if method == "CG":
        u, v, _, _, _ = cg_main(I1, I2, reg, from_file=False, sigma=sigma)
    elif method == "MG":
        u, v, _, _ = multigrid_main_iterative(
            I1, I2, reg, s1=mg_s1, s2=mg_s2,
            max_level=mg_max_level, tol=1e-8, max_cycles=50
        )
    elif method == "PCG":
        u, v, _, _, _ = pcg_main(
            I1, I2, reg, max_level=mg_max_level,
            s1=mg_s1, s2=mg_s2, tol=1e-8, maxit=500
        )
    else:
        raise ValueError("method must be 'CG', 'MG', or 'PCG'")

    # -------------------------------------------------------
    # Compute full color-coded image (2,2) — unchanged
    # -------------------------------------------------------
    flow_img = mycomputeColor(u, v)

    # -------------------------------------------------------
    # Overlay (1,0): FIX for white artifacts
    # Alpha = flow magnitude / max magnitude
    # -------------------------------------------------------
    # Compute magnitude
    mag = np.sqrt(u*u + v*v)
    M = np.max(mag) + 1e-12

    # --- boosted alpha for clearer colors ---
    gamma = 0.4             # lower = stronger color visibility
    alpha = (mag / M)**gamma

    # Scale overall opacity
    alpha = 0.8 * alpha     # base strength of overlay

    # Color image for overlay
    flow_overlay_img = mycomputeColor(u, v)

    # --- 2×2 PLOT ---
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))

    # (0,0) Image 1
    axs[0, 0].imshow(I1, cmap="gray")
    axs[0, 0].axis("off")

    # (0,1) Image 2
    axs[0, 1].imshow(I2, cmap="gray")
    axs[0, 1].axis("off")

    # (1,0) Image 1 + smooth alpha-corrected color overlay
    axs[1, 0].imshow(I1, cmap="gray")
    axs[1, 0].imshow(flow_overlay_img, alpha= alpha)
    axs[1, 0].axis("off")

    # (1,1) Full optical flow
    ax = axs[1, 1]
    ax.imshow(flow_img)
    ax.axis("off")

    # -------------------------------------------------------
    # Add color wheel to (1,1)
    # -------------------------------------------------------
    W = 128
    yy, xx = np.mgrid[-1:1:complex(0, W), -1:1:complex(0, W)]
    rad = np.sqrt(xx*xx + yy*yy)
    u_w = xx / (rad + 1e-6)
    v_w = yy / (rad + 1e-6)

    u_w[rad > 1] = 0
    v_w[rad > 1] = 0

    wheel_img = mycomputeColor(u_w, v_w)

    inset = ax.inset_axes([0.73, 0.73, 0.23, 0.23])
    inset.imshow(wheel_img)
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

def plot_all_methods_summary(ks, results_CG, results_PCG, results_MG):

    Ns = [2**k for k in ks]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ax_cg, ax_pcg, ax_mg, ax_time = axes

    # =================== CG ===================
    for k, res in zip(ks, results_CG["residuals"]):
        ax_cg.plot(res, label=f"k={k}")
    ax_cg.set_yscale("log")
    ax_cg.set_title("CG – Convergence")
    ax_cg.set_xlabel("Iteration")
    ax_cg.set_ylabel("Relative residual")
    ax_cg.grid(True)
    ax_cg.legend()

    # =================== PCG ===================
    for k, res in zip(ks, results_PCG["residuals"]):
        ax_pcg.plot(res, label=f"k={k}")
    ax_pcg.set_yscale("log")
    ax_pcg.set_title("PCG – Convergence")
    ax_pcg.set_xlabel("Iteration / Cycle")
    ax_pcg.grid(True)
    ax_pcg.legend()

    # =================== MG ===================
    for k, res in zip(ks, results_MG["residuals"]):
        ax_mg.plot(res, label=f"k={k}")
    ax_mg.set_yscale("log")
    ax_mg.set_title("MG – Convergence")
    ax_mg.set_xlabel("V-cycle")
    ax_mg.grid(True)
    ax_mg.legend()

    # =================== TIME COMPARISON ===================
    ax_time.plot(Ns, results_CG["times"], "-o", label="CG", linewidth=2)
    ax_time.plot(Ns, results_PCG["times"], "-o", label="PCG", linewidth=2)
    ax_time.plot(Ns, results_MG["times"], "-o", label="MG", linewidth=2)

    ax_time.set_xscale("log", base=2)
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Grid size N = 2^k")
    ax_time.set_ylabel("Runtime (seconds)")
    ax_time.set_title("Runtime Comparison")
    ax_time.grid(True)
    ax_time.legend()

    plt.tight_layout()
    plt.show()



################ MULTIGRID
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

def results_to_matrices(results, levels=[2,3,4], smooth_vals=[1,2,3]):
    """
    Convert MG parameter sweep results into:

        Svals     – sorted list of total smoothing steps S = s1 + s2
        rho_mat   – L × Svals matrix of convergence factors (average per cycle)
        time_mat  – L × Svals matrix of runtimes

    The heatmap assumes axis-1 corresponds to S = s1 + s2.
    """

    # --- Compute all possible total smoothings S = s1 + s2 ---
    Svals = sorted({s1 + s2 for s1 in smooth_vals for s2 in smooth_vals})
    nL = len(levels)
    nS = len(Svals)

    rho_mat  = np.full((nL, nS), np.nan)
    time_mat = np.full((nL, nS), np.nan)

    # Fill matrices
    for r in results:
        L  = r["L"]
        s1 = r["s1"]
        s2 = r["s2"]
        S  = s1 + s2

        i = levels.index(L)
        j = Svals.index(S)

        rho_mat[i, j]  = r["rho"]
        time_mat[i, j] = r["time"]

    return Svals, rho_mat, time_mat

def plot_mg_heatmaps(levels, Svals, rho_mat, time_mat,
                     method="MG", k=None, testcase=None):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Build the dynamic title ----
    if k is not None:
        N = 2**k
        title = f"Method = {method},  N = {N} (2^{k} × 2^{k})"
    else:
        title = f"Method = {method}"

    if testcase is not None:
        title += f",  Testcase = {testcase}"

    # Add the main title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)

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


def results_to_s1s2_matrices(results, smooth_vals=[1,2,3,4,5]):
    """
    Convert MG sweep results into:

        svals      – the smoothing values (same as input)
        rho_mat    – s1 × s2 matrix of convergence factors
        time_mat   – s1 × s2 matrix of runtimes
    """

    ns = len(smooth_vals)

    rho_mat  = np.full((ns, ns), np.nan)
    time_mat = np.full((ns, ns), np.nan)

    # Fill matrices using SAME logic as your original heatmap
    for r in results:
        s1 = r["s1"]
        s2 = r["s2"]

        if s1 not in smooth_vals or s2 not in smooth_vals:
            continue

        i = smooth_vals.index(s1)
        j = smooth_vals.index(s2)

        rho_mat[i, j]  = r["rho"]
        time_mat[i, j] = r["time"]

    return smooth_vals, rho_mat, time_mat

def plot_s1s2_heatmaps(svals, rho_mat, time_mat, L=None, method="MG", k=None, testcase=None):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Build the dynamic main title ----
    if k is not None:
        N = 2**k
        title = f"Method = {method},  N = {N} (2^{k} × 2^{k}),  L = {L}"
    else:
        title = f"Method = {method},  L = {L}"

    if testcase is not None:
        title += f",  Testcase = {testcase}"

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)

    # === Convergence factor heatmap ===
    im1 = axes[0].imshow(rho_mat, cmap="viridis", origin="lower")
    axes[0].set_title("MG Convergence Factor ρ")
    axes[0].set_xticks(range(len(svals)))
    axes[0].set_yticks(range(len(svals)))
    axes[0].set_xticklabels(svals)
    axes[0].set_yticklabels(svals)
    axes[0].set_xlabel("s2 (post-smoothing)")
    axes[0].set_ylabel("s1 (pre-smoothing)")
    fig.colorbar(im1, ax=axes[0])

    # === Runtime heatmap ===
    im2 = axes[1].imshow(time_mat, cmap="magma", origin="lower")
    axes[1].set_title("MG Runtime (seconds)")
    axes[1].set_xticks(range(len(svals)))
    axes[1].set_yticks(range(len(svals)))
    axes[1].set_xticklabels(svals)
    axes[1].set_yticklabels(svals)
    axes[1].set_xlabel("s2 (post-smoothing)")
    axes[1].set_ylabel("s1 (pre-smoothing)")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


##############################################################################################

def run_single_method_regs(method, regs,
                             mg_s1=2, mg_s2=2, mg_max_level=3,
                             from_file=False, sigma=0, N=480):

    result = {"times": [], "residuals": [], "regs": regs}

    # Load real images ONCE
    I1 = plt.imread("frame10.png").astype(float)
    I2 = plt.imread("frame11.png").astype(float)

    I1, I2 = generate_test_image(N, testcase=3)

    for reg in regs:
        print(reg)

        # ---------------- CG ----------------
        if method == "CG":
            u, v, res, max_iter, elapsed = cg_main(I1, I2, reg, tol=1.e-8, maxit=2000, from_file=False, sigma=0)

        # ---------------- PCG ----------------
        elif method == "PCG":
            u, v, res, max_iter, elapsed = pcg_main(
                I1, I2, reg,
                max_level=mg_max_level,
                s1=mg_s1,
                s2=mg_s2,
                tol=1e-8,
                maxit=50,
                from_file=from_file,
                sigma=sigma
            )

        # ---------------- MG ----------------
        elif method == "MG":
            u, v, res, elapsed = multigrid_main_iterative(
                I1, I2, reg,
                s1=mg_s1,
                s2=mg_s2,
                max_level=mg_max_level,
                tol=1e-8,
                max_cycles=50,
                from_file=from_file,
                sigma=sigma
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize residuals
        res = np.array(res) / res[0]

        result["times"].append(elapsed)
        result["residuals"].append(res)

    return result

def plot_method_summary_regs(regs, results, method="CG"):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ------------------- CONVERGENCE PLOT -------------------
    ax1 = axes[0]
    for lam, res in zip(regs, results["residuals"]):
        ax1.plot(res, label=f"λ={lam}")

    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration / V-cycle")
    ax1.set_ylabel("Relative residual")
    ax1.set_title(f"{method} – Convergence History")
    ax1.grid(True)
    ax1.legend()

    # ------------------- TIME VS LAMBDA -------------------
    ax2 = axes[1]
    ax2.plot(regs, results["times"], "-o", linewidth=2)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("λ (regularisation parameter)")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title(f"{method} – Time vs λ")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
