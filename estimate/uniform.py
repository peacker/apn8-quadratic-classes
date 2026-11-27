#!/usr/bin/env python3
"""Figure 2: classes outside the published lists found by the uniform sampler, against its draws (in order).
Reads uniform.npz, written by tools/make_numbers.py. usage: uniform.py  -> uniform.pdf (needs numpy, matplotlib)"""
import numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
z = np.load("uniform.npz")
grid, found, m, mse, WK = z["grid"], z["found"], float(z["m"]), float(z["mse"]), float(z["WK"])
D = int(z["D"])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
M = 1e6
x = grid / M
lam = grid / (WK + D)                                   # expected draws of one weight-1 class after n draws
U = 1e6                                                  # panel (a) in millions of classes
for ax, u in ((a, U), (b, 1.0)):
    ax.fill_between(x, (m - 2 * mse) / u, (m + 2 * mse) / u, color="#2a78d6", alpha=0.15, lw=0)
    ax.axhline(m / u, color="#2a78d6", ls="--", lw=0.8)
    ax.plot(x, found / u, "-", color="#eb6834", lw=1.2, label="classes found")
    ax.plot(x, D * (1 - np.exp(-lam)) / u, ":", color="black", lw=0.8, label="expected, all weights 1")
    ax.set_xlabel("draws (millions)")
a.set_ylabel("classes outside the published lists\n(millions)"); a.set_ylim(0, 1.1 * m / U)
a.text(0.02 * x[-1], m / U * 1.02, "model-free count", fontsize=8, color="#2a78d6", va="bottom")
a.legend(frameon=False, fontsize=8, loc="lower right"); a.set_title("(a) All draws", fontsize=9, loc="left")
b.set_ylim(m - 4 * mse, m + 3 * mse); b.set_xlim(x[-1] * 0.4, x[-1])
b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
b.set_title("(b) Detail", fontsize=9, loc="left")
fig.savefig("uniform.pdf")
