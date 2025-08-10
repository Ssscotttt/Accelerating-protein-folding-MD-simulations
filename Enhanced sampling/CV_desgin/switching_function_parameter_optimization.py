import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

history = []

def rational(r, r_0, n, m):
    ratio = r / r_0
    return (1 - ratio**n) / (1 - ratio**m)

def rational_loss(params, r_0, target_r_low, target_r_high):
    """
    Parameters:
    - params: m and n to optimize
    - r_0
    - target_r_low: The lowest distance the function is supposed to detect
    - target_r_high: The highest distance the function is supposed to detect
    """
    n, m = params
    n = int(round(n))  # Ensure n is an integer
    m = int(round(m))  # Ensure m is an integer

    r_values = np.linspace(0, 7, 1000)
    s_values = rational(r_values, r_0, n, m)

    # Ensure s(r) is near 1 for r < target_r_low
    loss_low = np.sum((s_values[r_values < target_r_low] - 1)**2)
    # Ensure s(r) is near 0 for r > target_r_high
    loss_high = np.sum((s_values[r_values > target_r_high])**2)
    # Penalize large jumps to avoid instability
    loss_smoothness = np.sum(np.diff(s_values[(r_values >= target_r_low) & (r_values <= (target_r_high / 3))])**2)
    total_loss = loss_low + loss_high + loss_smoothness

    history.append((n, m, total_loss))

    return total_loss

def rational_loss_int(params, r_0, target_r_low, target_r_high):
    n, m = params
    n = int(round(n))  # Ensure n is an integer
    m = int(round(m))  # Ensure m is an integer

    r_values = np.linspace(0, 7, 1000)
    s_values = rational(r_values, r_0, n, m)

    loss_low = np.sum((s_values[r_values < target_r_low] - 1)**2)
    loss_high = np.sum((s_values[r_values > target_r_high])**2)
    loss_smoothness = np.sum(np.diff(s_values)**2)
    total_loss = loss_low + loss_high + loss_smoothness

    history.append((n, m, total_loss))  # Store rounded values

    return total_loss

def rational_callback(xk):
    n, m = xk
    print(f"Step {len(history)}: n={n:.4f}, m={m:.4f}, Loss={history[-1][2]:.6f}")

def find_m_n_int():
    r0 = 0.45
    target_r_low = r0
    target_r_high = 6.0
    bounds = [(1, 150), (1, 150)]
    result = differential_evolution(
        rational_loss, 
        bounds, 
        args=(r0, target_r_low, target_r_high),
        strategy='rand1bin',   # Faster convergence in integer search
        mutation=(0.5, 1), 
        recombination=0.7,
        workers=1,  # Prevents multiprocessing issues with Ctrl+C
        disp=True,
        seed=9
    )
    optimal_n, optimal_m = map(int, result.x)  # Ensure output is integer
    print(f"Optimized n: {optimal_n}, m: {optimal_m}")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot([h[0] for h in history], label="n", marker="o")
    ax[0].plot([h[1] for h in history], label="m", marker="s")
    ax[0].set_ylabel("Parameter Values")
    ax[0].legend()
    ax[0].set_title("Optimization Progress")

    ax[1].plot([h[2] for h in history], label="Loss", color="red")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylim(-100, 2000)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    r_values = np.linspace(0, 7, 1000)
    s_values = rational(r_values, r0, optimal_n, optimal_m)

    plt.plot(r_values, s_values, label=f"Optimized (n={optimal_n:.2f}, m={optimal_m:.2f})")
    plt.axvline(target_r_low, color='green', linestyle='--', label=f"Start of Decay (r={target_r_low})")
    plt.axvline(target_r_high, color='red', linestyle='--', label="End of Decay (r=6.0)")
    plt.xlabel("Distance (r)")
    plt.xlim(right=7)
    plt.ylabel("s(r)")
    plt.title("Optimized Rational Switching Function")
    plt.legend()
    plt.grid()
    plt.show()

    s0 = optimal_n / optimal_m
    cmdist = (s_values - s0) ** 2
    min_index = np.argmin(cmdist)
    print(min_index)
    gradient = np.gradient(cmdist[min_index:], r_values[min_index:])
    threshold = 3*1e-3
    flat_indices = np.where(np.abs(gradient) < threshold)[0]
    if len(flat_indices) > 0:
        flat_index = flat_indices[0]  # First occurrence where function flattens
        flat_x = r_values[flat_index]
        print(f"The function starts flattening at x = {flat_x}")
    else:
        print("No clear flattening detected; consider adjusting the threshold.")
    plt.plot(r_values, cmdist)
    plt.xlim(right=7)
    plt.title("Optimized CMDIST from Rational Switching Function")
    plt.show()

def find_n_m_int_r0(ref_file, output_file):
    """
    Optimizes n and m for contact map with contact-specific r0s. 

    Parameters:
    - ref_file (str): the file that stores native contact pairs and distances
    - output_file (str): the file that stores the ns and ms
    """
    bounds = [(1, 20), (1, 20)]     # bounds for n and m
    lines = []                      # lines to be written to the output file

    ref_data = pd.read_csv(ref_file, sep='\s+', header=None, names=["Pair", "Distance"])
    for _, row in ref_data.iterrows():
        pair = row["Pair"]
        distance = row["Distance"]
        r0 = distance
        target_r_low = r0
        target_r_high = 6.0
        result = differential_evolution(
            rational_loss, 
            bounds, 
            args=(r0, target_r_low, target_r_high),
            strategy='rand1bin',   # Faster convergence in integer search
            mutation=(0.5, 1), 
            recombination=0.7,
            workers=1,  # Prevents multiprocessing issues with Ctrl+C
            disp=True
        )
        optimal_n, optimal_m = map(int, result.x)  # Ensure output is integer
        print(f"Optimized n: {optimal_n}, m: {optimal_m}")
        lines.append(f"{pair} {optimal_n} {optimal_m}")

def find_m_n_for_rational():

    r0 = 0.55
    target_r_low = r0
    target_r_high = 6.0

    initial_guess = [5, 10] # Initial guess for n and m

    result = minimize(rational_loss, initial_guess, args=(r0, target_r_low, target_r_high), 
                      bounds=[(1, 20), (1, 20)], callback=rational_callback)

    optimal_n, optimal_m = result.x
    print(f"Optimized n: {optimal_n:.4f}, m: {optimal_m:.4f}")

    # Plot optimization history
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot([h[0] for h in history], label="n", marker="o")
    ax[0].plot([h[1] for h in history], label="m", marker="s")
    ax[0].set_ylabel("Parameter Values")
    ax[0].legend()
    ax[0].set_title("Optimization Progress")

    ax[1].plot([h[2] for h in history], label="Loss", marker="d", color="red")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Iteration")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    r_values = np.linspace(0, 7, 1000)
    s_values = rational(r_values, r0, optimal_n, optimal_m)

    plt.plot(r_values, s_values, label=f"Optimized (n={optimal_n:.2f}, m={optimal_m:.2f})")
    plt.axvline(target_r_low, color='green', linestyle='--', label="Start of Decay (r=0.35)")
    plt.axvline(target_r_high, color='red', linestyle='--', label="End of Decay (r=6.0)")
    plt.xlabel("Distance (r)")
    plt.ylabel("s(r)")
    plt.title("Optimized Rational Switching Function")
    plt.legend()
    plt.grid()
    plt.show()

def smap(r, r_0, a, b):
    term1 = (2**(a/b) - 1)
    term2 = (r / r_0)**a
    return (1 + term1 * term2) ** (-b / a)

def smap_loss():
    """"""

#find_m_n_int()
#Optimized n: 11.6641, m: 14.6569
#Optimized n: 10.5579, m: 13.3263
#Optimized n: 9.3358, m: 11.8600