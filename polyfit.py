# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator
from matplotlib.colors import (ListedColormap, LinearSegmentedColormap,
                               Normalize, LogNorm, TwoSlopeNorm, to_rgb)
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy import stats, signal, optimize, integrate
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ----------------------------------------------------------------------------
# Font resolution
# ----------------------------------------------------------------------------
FONT_STACK = [
    'Helvetica Neue', 'HelveticaNeue', 'Helvetica',   # macOS
    'TeX Gyre Heros', 'Nimbus Sans', 'FreeSans',      # Helvetica metric clones
    'Arial', 'Liberation Sans',                       # Windows / Linux
    'DejaVu Sans',                                    # matplotlib's own fallback
]

def resolve_font(stack=FONT_STACK, verbose=True):
    '''Return the first font in `stack` that is actually installed.'''
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in stack:
        if name in have:
            if verbose and name != stack[0]:
                print(f'[style] "{stack[0]}" not found -> using "{name}"')
            elif verbose:
                print(f'[style] using "{name}"')
            return name
    if verbose:
        print('[style] nothing in the stack found -> DejaVu Sans')
    return 'DejaVu Sans'

ACTIVE_FONT = resolve_font()

# ----------------------------------------------------------------------------
# House style
# ----------------------------------------------------------------------------
def set_style(base=11, family=None, mathfont='cm', usetex=False,
              screen_dpi=140, save_dpi=600):
    '''
    Apply the group house style.

    base       : point size of tick labels. Axis labels are base+3.
                 Journals want >= 6 pt after reduction; 8-11 pt is the sweet spot.
    mathfont   : 'cm'   -> Computer Modern math (matches a LaTeX manuscript)
                 'stix' -> Times-like math
                 'dejavusans' / 'custom' -> sans-serif math (matches Helvetica)
    usetex     : True routes ALL text through a real LaTeX install. Prettiest
                 and most consistent, but slow and breaks on machines without
                 LaTeX. Keep False for exploration, flip True for final figures.
    save_dpi   : only affects raster elements (imshow, scatter with many points).
                 Vector output is resolution-independent.
    '''
    mpl.rcParams.update(mpl.rcParamsDefault)
    fam = family or ACTIVE_FONT
    mpl.rcParams.update({
        # --- typography ---
        'font.family'       : 'sans-serif',
        'font.sans-serif'   : [fam] + FONT_STACK,
        'mathtext.fontset'  : mathfont,
        'mathtext.default'  : 'it',
        'text.usetex'       : usetex,
        'font.size'         : base,
        'axes.labelsize'    : base + 3,
        'axes.titlesize'    : base + 1,
        'axes.titleweight'  : 'regular',
        'xtick.labelsize'   : base,
        'ytick.labelsize'   : base,
        'legend.fontsize'   : base - 1,
        'legend.title_fontsize': base - 1,

        # --- axes furniture ---
        'axes.linewidth'    : 0.8,
        'axes.labelpad'     : 4.0,
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        'axes.axisbelow'    : True,
        'axes.grid'         : False,
        'grid.linewidth'    : 0.5,
        'grid.alpha'        : 0.35,

        # --- ticks: outward, minor ticks on ---
        'xtick.direction'   : 'out',   'ytick.direction'   : 'out',
        'xtick.major.size'  : 3.5,     'ytick.major.size'  : 3.5,
        'xtick.major.width' : 0.8,     'ytick.major.width' : 0.8,
        'xtick.minor.size'  : 2.0,     'ytick.minor.size'  : 2.0,
        'xtick.minor.width' : 0.6,     'ytick.minor.width' : 0.6,
        'xtick.major.pad'   : 3.0,     'ytick.major.pad'   : 3.0,

        # --- data elements ---
        'lines.linewidth'   : 1.4,
        'lines.markersize'  : 4.5,
        'lines.markeredgewidth': 0.6,
        'lines.solid_capstyle' : 'round',
        'patch.linewidth'   : 0.6,
        'errorbar.capsize'  : 2.5,
        'image.cmap'        : 'viridis',
        'image.interpolation': 'nearest',

        # --- legend ---
        'legend.frameon'      : False,
        'legend.handlelength' : 1.5,
        'legend.handletextpad': 0.5,
        'legend.labelspacing' : 0.30,
        'legend.borderpad'    : 0.2,
        'legend.columnspacing': 1.0,

        # --- figure / export ---
        'figure.facecolor'  : 'white',
        'axes.facecolor'    : 'white',
        'figure.dpi'        : screen_dpi,   # on-screen preview only
        'savefig.dpi'       : save_dpi,     # written files
        'savefig.bbox'      : 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.facecolor' : 'white',
        'pdf.fonttype'      : 42,     # TrueType -> editable text in Illustrator
        'ps.fonttype'       : 42,
        'svg.fonttype'      : 'none', # keep <text> elements, do not outline
        'pdf.compression'   : 6,
    })

set_style()

# ----------------------------------------------------------------------------
# Journal column widths (inches). Draw at the size you will print at.
# ----------------------------------------------------------------------------
W = {
    'prl_single' : 3.375,   # Phys. Rev. Lett. / APS single column
    'prl_double' : 6.75,
    'nature_1col': 3.504,   # 89 mm
    'nature_2col': 7.205,   # 183 mm
    'pnas_1col'  : 3.42,
    'pnas_2col'  : 7.0,
    'elsevier_1col': 3.543, # 90 mm
    'elsevier_2col': 7.480, # 190 mm
    'jfm'        : 5.31,    # J. Fluid Mech. text width
    'beamer'     : 4.5,
}
print(f'[style] active font: {ACTIVE_FONT}')

# %%
fig_folder = "./figures"

# %% [markdown]
# # MODIFIED EXPERIMENT
# 
# 

# %%

import numpy as np


def generate_samples(r, N=50000, T=20):
    """
    Generate N independent AR(1) trajectories of length T.

    Each trajectory X^(n) = (x_0, ..., x_{T-1})
    is treated as one vector in R^T.
    """

    # Critical point: r_c = 0
    # a(r) -> 1 as r -> 0
    a = np.exp(-r)

    # Relaxation time
    tau = -1.0 / np.log(a)

    X = np.zeros((N, T))

    # Generate independent trajectories
    X[:, 0] = np.random.randn(N)

    for t in range(1, T):
        X[:, t] = a * X[:, t-1] + np.random.randn(N)

    return X, tau



# %%

def covariance(r, N=50000, T=20):
    X, tau = generate_samples(r, N=N, T=T)

    # Rows = independent trajectories
    # Columns = time coordinates
    Sigma = np.cov(X, rowvar=False)

    return Sigma, tau


# %%

r_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 1e-3, 5e-4, 1e-4]


# %%

for r in r_list:

    Sigma, tau = covariance(r)

    eigvals = np.linalg.eigvalsh(Sigma)

    print(
        f"r={r:.3g}, "
        f"tau={tau:.2f}, "
        f"lambda_max={eigvals[-1]:.3f}, "
        f"lambda_2={eigvals[-2]:.3f}"
    )


# %%

for r in r_list:

    Sigma, tau = covariance(r)

    eigvals, eigvecs = np.linalg.eigh(Sigma)

    lam1 = eigvals[-1]
    lam2 = eigvals[-2]

    explained = lam1 / np.sum(eigvals)

    print(
        f"r={r:.3g}, "
        f"tau={tau:.2f}, "
        f"lambda1/lambda2={lam1/lam2:.2f}, "
        f"variance fraction={explained:.3f}"
    )

Sigma_ref, tau = covariance(1e-4)

evals, evecs = np.linalg.eigh(Sigma_ref)
v_ref = evecs[:, -1]


# %%

for r in r_list:

    Sigma, tau = covariance(r)

    evals, evecs = np.linalg.eigh(Sigma)
    v = evecs[:, -1]

    overlap = abs(v @ v_ref)

    print(
        f"r={r:.3g}, "
        f"tau={tau:.2f}, "
        f"overlap={overlap:.4f}"
    )

# %% [markdown]
# # <fieldset>COMPUTATIONAL RESULTS</fieldset>

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.rcParams.update(mpl.rcParamsDefault)
# font = {
#         'weight' : 'light',
#         'size'   : 16}

# mpl.rc('font', **font)
# plt.rc("xtick", labelsize="medium")
# rc = {"mathtext.fontset" : "cm"}
# plt.rcParams.update(rc)

# =====================================================
# Configuration
# =====================================================

N_train = 50000

nu_list = [ 0.35,0.55, 0.75]

Delta_list = np.logspace(-6, -3, 20)

degree_list = [
    2,
    4,
    6,
    8,
    10,
    16
]

r_min = -2.0
r_max = 2.0

N_plot = 20000

# =====================================================
# Critical function
# =====================================================

def lam(r, nu):
    return np.abs(r) ** (-nu)

def g(r, nu):
    return np.abs(r)**nu/(1.0 + np.abs(r)**nu)

# =====================================================
# Main sweep
# =====================================================

results = []

for nu in nu_list:

    for Delta in Delta_list:

        # ---------------------------
        # Training samples
        # ---------------------------

        r_left = np.random.uniform(
            r_min,
            -Delta/2,
            N_train // 2
        )

        r_right = np.random.uniform(
            Delta/2,
            r_max,
            N_train // 2
        )

        r_train = np.concatenate(
            [r_left, r_right]
        )

        y_train = g(r_train, nu)

        # score-loss-inspired weight
        weights = 1.0 + lam(r_train, nu)

        # evaluation grid

        r_plot = np.linspace(
            r_min,
            r_max,
            N_plot
        )

        y_true = g(r_plot, nu)

        train_mask = np.abs(r_plot) > Delta/2
        crit_mask = np.abs(r_plot) < Delta/2

        for degree in degree_list:

            # -----------------------
            # Fit polynomial
            # -----------------------

            coeffs = np.polyfit(
                r_train,
                y_train,
                degree,
                w=np.sqrt(weights)
            )

            leading_coeff = coeffs[0]
            const_coeff = coeffs[-1]

            poly = np.poly1d(coeffs)

            y_pred = poly(r_plot)

            plot_weights = 1 + lam(r_plot, nu)

            # -----------------------
            # Errors
            # -----------------------

            train_mse = np.mean(
                plot_weights[train_mask] * (
                    y_true[train_mask]
                    - y_pred[train_mask]
                ) ** 2
            )

            r_crit = np.linspace(-Delta/2, Delta/2, N_plot)

            critical_mse = np.mean(
                (1 + lam(r_crit, nu)) * (
                    g(r_crit, nu)
                    - poly(r_crit)
                )**2
            )

            gap_ratio = (
                critical_mse
                / train_mse
            )

            results.append(
                {
                    "nu": nu,
                    "Delta": Delta,
                    "degree": degree,
                    "train_mse": train_mse,
                    "critical_mse": critical_mse,
                    "gap_ratio": gap_ratio,
                    "coeffs": coeffs.copy()
                }
            )
        # plt.figure()
        # plt.plot(r_plot,y_true)
        # plt.plot(r_plot, y_pred)
        # # plt.axvspan(-Delta/2,Delta/2, alpha = 0.3, label = 'excised gap')
        # # plt.legend()
        # # plt.xlim(-Delta*100, Delta * 100)
        # plt.xlabel('r')
        # plt.ylabel('g(r)')
        # plt.title(fr'$\Delta$:{Delta:.2e}, $\nu$:{nu:.2e}, K:{degree}')
        # plt.show()

# =====================================================
# Print table
# =====================================================

for r in results:

    print(
        f"nu={r['nu']:4.2f} "
        f"Delta={r['Delta']:.3e} "
        f"K={r['degree']:2d} "
        f"train={r['train_mse']:.3e} "
        f"critical={r['critical_mse']:.3e} "
        f"ratio={r['gap_ratio']:.3e}"
        f"coeff = {r['coeffs']}"
    )

# %%
#@title Error vs Delta (Error is calculated in the untrained region only)

for nu in nu_list:

    # plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=2, layout='tight')

    for degree in degree_list:

        xs = []
        ys1 = []
        ys2 = []

        for r in results:

            if (
                r["nu"] == nu
                and r["degree"] == degree
            ):
                xs.append(r["Delta"])
                ys1.append(r["train_mse"])
                ys2.append(r["critical_mse"])


        ax[0].loglog(
            xs,
            ys1,
            marker = "o",
            label = f"K={degree}"
        )

        ax[1].loglog(
            xs,
            ys2,
            marker="o",
            label=f"K={degree}"
        )

    fig.suptitle(
        fr"Error vs $\Delta$ ($\nu$={nu})"
    )
    ax[0].set_xlabel(r"$\Delta$")
    ax[1].set_xlabel(r"$\Delta$")

    ax[0].set_ylabel("Train MSE")
    ax[1].set_ylabel("Crtical MSE")
    plt.legend()
    # plt.savefig(f"{fig_folder}/critical_error_v_delta.pdf")
    plt.show()

# %%
#@title Error vs degree k (Error is calculated in the untrained region only)

for nu in nu_list:

    plt.figure(figsize= (8,5))

    for Delta in Delta_list:

        xs = []
        ys = []

        for r in results:

            if (
                r["nu"] == nu
                and r["Delta"] == Delta
            ):
                xs.append(r["degree"])
                ys.append(r["critical_mse"])

        plt.semilogy(
            xs,
            ys,
            marker="o",
            label=f"Δ={Delta}"
        )

    plt.xlabel("Polynomial degree")
    plt.ylabel("Critical MSE")
    plt.title(
        f"Degree sweep (nu={nu})"
    )
    plt.legend()
    plt.show()

# %%
#@title a_k vs Delta

for nu in nu_list:

    plt.figure(figsize=(7,5))

    for degree in degree_list:

        xs = []
        ys = []

        for r in results:

            if (
                r["nu"] == nu
                and r["degree"] == degree
            ):
                xs.append(r["Delta"])
                ys.append(r["coeffs"][0])

        order = np.argsort(xs)

        xs = np.array(xs)[order]
        ys = np.array(np.abs(ys))[order]
        a_inf = ys[0]      # smallest Delta approximation

        corr = np.abs(ys - a_inf)

        plt.loglog(
            xs[1:],
            corr[1:],
            marker="o",
            label=f"K={degree}"
        )

    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$a_K$")
    plt.title(
        f"Signed leading coefficient (nu={nu})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
nu = 0.45
degree = 8

xs = []
ys = []

for r in results:

    if (
        r["nu"] == nu
        and r["degree"] == degree
    ):
        xs.append(r["Delta"])
        ys.append(r["critical_mse"])

xs = np.array(xs)
ys = np.array(ys)

alpha, b = np.polyfit(
    np.log(xs),
    np.log(ys),
    1
)

fit_curve = np.exp(b) * xs**alpha

plt.figure()
plt.loglog(xs, ys, "o", label="data")
plt.loglog(
    xs,
    fit_curve,
    "--",
    label=f"slope={alpha:.3f}"
)
plt.xlabel("Delta")
plt.ylabel("Critical MSE")
plt.legend()
plt.show()

# %%
#@title Loss-Delta power Law exponent

loss_exp = np.zeros(
    (len(nu_list), len(degree_list))
)

for i, nu in enumerate(nu_list):

    for j, degree in enumerate(degree_list):

        xs = []
        ys = []

        for r in results:

            if (
                r["nu"] == nu
                and r["degree"] == degree
            ):
                xs.append(r["Delta"])
                ys.append(r["critical_mse"])

        xs = np.array(xs)
        ys = np.array(ys)

        mask = ys > 0

        alpha, _ = np.polyfit(
            np.log(xs[mask]),
            np.log(ys[mask]),
            1
        )

        loss_exp[i, j] = alpha

plt.figure(figsize=(8,5), dpi=300)

im = plt.imshow(
    loss_exp,
    aspect="auto",
    origin="lower"
)

plt.colorbar(
    im,
    label=r"Loss exponent $\alpha$"
)

plt.xticks(
    np.arange(len(degree_list)),
    degree_list
)

plt.yticks(
    np.arange(len(nu_list)),
    nu_list
)

plt.xlabel("Polynomial degree K")
plt.ylabel(r"$\nu$")
plt.title(
    r"$E_{crit}\sim \Delta^\alpha$"
)
# plt.savefig(f'{fig_folder}/Loss_v_Delta.pdf')
plt.show()

# %%
#@title a_{k-1} vs Delta power law exponent

coeff_exp = np.zeros(
    (len(nu_list), len(degree_list))
)

for i, nu in enumerate(nu_list):

    for j, degree in enumerate(degree_list):

        xs = []
        ys = []

        for r in results:

            if (
                r["nu"] == nu
                and r["degree"] == degree
            ):
                xs.append(r["Delta"])
                ys.append(
                    abs(abs(r["coeffs"][0]))
                )

        xs = np.array(xs)
        ys = np.array(ys)
        lead0 = ys[0]

        corr = np.abs(ys - lead0)

        # mask = ys > 0
        mask = corr > 0

        beta, _ = np.polyfit(
            np.log(xs[mask]),
            np.log(corr[mask]),
            1
        )

        coeff_exp[i, j] = beta

plt.figure(figsize=(8,5))

im = plt.imshow(
    coeff_exp,
    aspect="auto",
    origin="lower"
)

plt.colorbar(
    im,
    label=r"Coefficient exponent $\beta$"
)

plt.xticks(
    np.arange(len(degree_list)),
    degree_list
)

plt.yticks(
    np.arange(len(nu_list)),
    nu_list
)

plt.xlabel("Polynomial degree K")
plt.ylabel(r"$\nu$")
plt.title(
    r"$|a_{K-1}|\sim \Delta^\beta$"
)

plt.show()

# %%
#@title Testing scaling of inner contribution ( Clenshaw-Curtis Quadrature)

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# mpl.rcParams.update(mpl.rcParamsDefault)
# font = {
#         'weight' : 'light',
#         'size'   : 16}

# mpl.rc('font', **font)
# plt.rc("xtick", labelsize="medium")
# rc = {"mathtext.fontset" : "cm"}
# plt.rcParams.update(rc)

def clenshaw_curtis_nodes_weights(a, b, N):
    """
    Clenshaw-Curtis quadrature on [a,b].

    Returns
    -------
    x : nodes
    w : weights
    """
    if N == 1:
        return np.array([(a+b)/2]), np.array([b-a])

    k = np.arange(N)
    theta = np.pi * k / (N - 1)

    x = np.cos(theta)

    # map to [a,b]
    x = 0.5*(b-a)*x + 0.5*(a+b)

    w = np.zeros(N)

    n = N - 1
    ii = np.arange(1, n)

    v = np.ones(n-1)

    if n % 2 == 0:
        w[0] = 1/(n**2 - 1)
        w[-1] = w[0]

        for j in range(1, n//2):
            v -= 2*np.cos(2*j*theta[ii])/(4*j*j - 1)

        v -= np.cos(n*theta[ii])/(n*n - 1)

    else:
        w[0] = 1/n**2
        w[-1] = w[0]

        for j in range(1, (n+1)//2):
            v -= 2*np.cos(2*j*theta[ii])/(4*j*j - 1)

    w[ii] = 2*v/n

    w *= 0.5*(b-a)

    return x, w




# =====================================================
# Critical function
# =====================================================

def lam(r, nu):
    return np.abs(r) ** (-nu)

def g(r, nu):
    return np.abs(r)**nu/(1.0 + np.abs(r)**nu)

def rho(Delta):
    return 1/(L - Delta)

nu_list = [ 0.35,0.55, 0.75]

Delta_list = np.logspace(-9, 1, 40)

degree_list = [
    6,
    8,
    10
]

# N_quad = 100000
r_min = -1.0
r_max = 1.0

L = r_max-r_min

results = []

for nu in nu_list:
    for Delta in Delta_list:
        for degree in degree_list:



            N_cc = 2000

            rL, wL = clenshaw_curtis_nodes_weights(
                r_min,
                (-Delta/2),
                N_cc
            )

            rR, wR = clenshaw_curtis_nodes_weights(
                (Delta/2),
                r_max,
                N_cc
            )

            r = np.concatenate([rL, rR])
            quad_w = np.concatenate([wL, wR])
            # print(np.sum(quad_w * r**2))

            # dr = r[1] - r[0]


            y_train = g(r, nu)
            weights = 1.0 + lam(r, nu)

            # ---------------------------------
            # basis
            # ---------------------------------

            powers = np.arange(
                0,
                degree + 1,
                2
            )

            nbasis = len(powers)

            # ---------------------------------
            # design matrix
            # ---------------------------------

            Phi = np.column_stack(
                [
                    r**p
                    for p in powers
                ]
            )

            # ---------------------------------
            # M matrix
            # ---------------------------------

            M = (
                Phi.T
                @
                ((rho(Delta) * weights * quad_w)[:,None] * Phi)
            )

            # ---------------------------------
            # b vector
            # ---------------------------------

            b =(
                Phi.T
                @
                ((rho(Delta) * quad_w) )#* y_train)
            )

            # ---------------------------------
            # solve
            # ---------------------------------

            theta = np.linalg.solve(
                M,
                b
            )

            # coefficients
            coeffs_even = theta.copy()

            results.append(
                    {
                        "nu": nu,
                        "Delta": Delta,
                        "degree": degree,

                        "M": M.copy(),
                        "b": b.copy(),
                        "theta": theta.copy(),

                        # "train_mse": train_mse,
                        # "critical_mse": critical_mse
                    }
                          )
        # print(np.sum(quad_w))
        # print(rho(Delta) * np.sum(quad_w))

# %%
#@title Scaling of M (Clenshaw Quad)


for nu in nu_list:

    degree = 10

    k = 0
    l = 0

    xs = []
    ys = []

    for r in results:

        if (
            np.isclose(r["nu"], nu)
            and r["degree"] == degree
        ):

            xs.append(r["Delta"])
            ys.append(r["M"][k,l])

    order = np.argsort(xs)

    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    print(ys)

    # estimate asymptotic constant
    # M_inf = ys[0]   # smallest Delta
    # print(M_inf)

    # m0 =  1 + 1**(1-nu+k+l) * 2/(1-nu +k +l)* 1.0/(L - xs)  #Theoretical const coeff
    # print(m0)
    M_inf_exact = 2/L * (1/(2*k + 2*l +1) + 1/(2*k + 2*l + 1 - nu))

    diff = np.abs(ys - M_inf_exact)
    # print(diff)

    mask = np.abs(diff) > 1e-15

    alpha, intercept = np.polyfit(
        # np.log(xs[mask]),
        np.log(xs[7:27]),

        # np.log(diff[mask]),
        np.log(diff[7:27]),

        1
    )

    print(
        f"nu={nu:.2f}, "
        f"M00 exponent={alpha:.4f}"
    )

    plt.figure()

    plt.loglog(
        xs[1:],
        diff[1:],
        marker="o",
        label=fr"Fit from data: $\Delta^{{{alpha:.2f}}}$"
    )

    # plt.loglog(
    #     xs,
    #     np.exp(intercept)*xs**alpha,
    #     "--",
    #     label=fr"$\Delta^{{{alpha:.2f}}}$"
    # )

    plt.loglog(
        xs[1:],
        # np.exp(intercept)*xs**(1.0-nu),
        xs[1:] ** (1.0 - nu),
        "--",
        label=fr"Analytical soln: $\Delta^{{{1.0 - nu:.2f}}}$"
    )

    plt.axvline(1e-6, ls=':')
    plt.axvline(1e-2, ls=':')
    # plt.fill_betweenx(diff[7:27], x1=1e-6, x2=1e-2)

    plt.xlabel(r"$\Delta$")
    plt.ylabel(
        r"$|M_{00}-M_{00}(0)|$"
    )

    plt.title(
        fr"$M_{{00}}$ scaling, $\nu={nu}$"
    )
    if nu == 0.35:
        plt.savefig(f"{fig_folder}/M_v_delta.pdf")
    plt.legend()
    plt.show()

# %%
#@title Scaling of b (Clenshaw Quad)
for nu in nu_list:

    plt.figure(figsize=(7,5))

    degree = 10

    k = 1

    xs = []
    ys = []

    for r in results:

        if (
            np.isclose(r["nu"], nu)
            and r["degree"] == degree
        ):

            xs.append(r["Delta"])
            ys.append(r["b"][k])

    order = np.argsort(xs)

    xs = np.array(xs)[order]
    ys = np.array(ys)[order]

    # print(ys)

    # b_inf = ys[0]
    b10 =  2 / (2*k+1)/(L)
    # b10 = 1/(k +1)
    print(b10)

    diff = np.abs(ys - b10)


    mask = (diff > 1e-6)

    alpha, intercept = np.polyfit(
        np.log(xs[mask]),
        # np.log(xs),
        np.log(diff[mask]),
        # np.log(diff),
        1
    )

    print(
        f"nu={nu:.2f}, "
        f"b_{k} exponent={alpha:.4f}"
    )

    plt.loglog(
        xs,
        np.abs(ys - b10),
        marker="o",
        label = fr'Fit from Data: $\Delta^{{{alpha:.2f}}}$'
    )

    # plt.loglog(
    #     xs,
    #     np.exp(intercept)*xs**alpha,
    #     "--",
    #     label=fr"$\Delta^{{{alpha:.2f}}}$"
    # )

    plt.loglog(
        xs,
        xs**1.0,
        "--",
        label=fr"Analytical scaling:$\Delta^{{{1.0:.2f}}}$"
    )

    plt.xlabel(r"$\Delta$")
    plt.ylabel(rf"$|b_{k}|$")
    plt.title(
        fr"$b_{k}$ vs $\Delta$, $\nu={nu}$"
    )
    plt.legend()
    # if nu == 0.35:
        # plt.savefig(f"{fig_folder}/b_v_delta.pdf")
    plt.show()

# %%
#@title Theta vs Delta
for nu in nu_list:

    powers = np.arange(0, degree+1, 2)
    nbasis = len(powers)

    M0 = np.zeros((nbasis, nbasis))

    for i,k in enumerate(range(nbasis)):
        for j,l in enumerate(range(nbasis)):

            n = 2*k + 2*l

            M0[i,j] = (
                1/(n+1)
                +
                1/(n+1-nu)
            )

    b0 = np.array([
        1/(2*k+1)
        for k in range(nbasis)
    ])

    theta0 = np.linalg.solve(M0, b0)

    plt.figure(figsize=(7,5))

    degree = 10

    k = 0
    l = 0

    xs = []
    ys = []

    for r in results:

        if (
            np.isclose(r["nu"], nu)
            and r["degree"] == degree
        ):

            xs.append(r["Delta"])
            ys.append(r["theta"][k])


    order = np.argsort(xs)

    xs = np.array(xs)[order]
    ys = np.array(ys)[order]


    # print(theta0)

    diff = np.abs(ys - theta0[k])

    mask = (diff > 1e-6)

    alpha, intercept = np.polyfit(
        np.log(xs[mask]),

        np.log(diff[mask]),

        1
    )

    print(
        f"nu={nu:.2f}, "
        f"theta_{k} exponent={alpha:.4f}"
    )

    plt.loglog(
        xs[1:],
        diff[1:],
        marker="o",
        label = fr'Fit from Data:$\Delta^{{{alpha:.2f}}}$'
    )

    # plt.loglog(
    #     xs,
    #     np.exp(intercept)*xs**alpha,
    #     "--",
    #     label=fr"$\Delta^{{{alpha:.2f}}}$"
    # )

    plt.loglog(
        xs,
        xs**(1.0 - nu),
        "--",
        label=fr"Analytical scaling:$\Delta^{{{1.0-nu:.2f}}}$"
    )

    plt.xlabel(r"$\Delta$")
    plt.ylabel(fr"$|\theta_{k}|$")
    plt.title(
        fr"$\theta$ vs $\Delta$, $\nu={nu}$"
    )
    plt.legend()
    # if nu == 0.35:
        # plt.savefig(f"{fig_folder}/theta_v_delta.pdf")
    plt.show()


# %%
# @title a_k vs k

for nu in nu_list:

    plt.figure()
    fig, ax = plt.subplots(nrows= 4, ncols=5, figsize = (20, 12), layout = 'constrained')

    for i,delta in enumerate(Delta_list):

        for degree in degree_list:

            xs = []
            ys = []

            for r in results:
            # for degree in degree_list:


                if (
                    r["nu"] == nu
                    and r["degree"] == degree
                    and r["Delta"] == delta
                ):
                    xs.append(r["degree"])
                    ys.append(np.abs(r["theta"]))

                    xvals = np.arange(0, r['degree']+1)

                    ax.flatten()[i].semilogy(
                        np.abs(r["theta"]),
                        '-',
                        marker="o",
                        alpha = 0.7,
                        label=f"K={degree}")
                    # plt.semilogy(
                    #     xvals,
                    #     np.abs(r["coeffs"]),
                    #     marker="o",
                    #     label=f"K={degree}"
                    # )
                    # plt.xticks(xvals)

    fig.suptitle(
        fr"Coeffs vs K (nu={nu})"
    )
    fig.supxlabel(r"Coeffs")
    fig.supylabel("Amplitude")
    plt.legend()
    plt.show()

# %% [markdown]
# # FINITE SAMPLING NUMERICS

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# mpl.rcParams.update(mpl.rcParamsDefault)
# font = {
#         'weight' : 'light',
#         'size'   : 16}

# mpl.rc('font', **font)
# plt.rc("xtick", labelsize="medium")
# rc = {"mathtext.fontset" : "cm"}
# plt.rcParams.update(rc)

# =====================================================
# Configuration
# =====================================================

N = [100, 1000, 10000, int(1e+5)]

nu_list = [ 0.55]

Delta_list = np.logspace(-6, 0, 30)

degree_list = [
    6,
    8,
    10,
    20,
    32
]

r_min = -1.0
r_max = 1.0

N_plot = 20000

# =====================================================
# Critical function
# =====================================================

def lam(r, nu):
    return np.abs(r) ** (-nu)

def g(r, nu):
    return np.abs(r)**nu/(1.0 + np.abs(r)**nu)

# =====================================================
# Main sweep
# =====================================================

results = []

for nu in nu_list:

    for Delta in Delta_list:

          for N_train in N:

                # ---------------------------
                # Training samples
                # ---------------------------

                r_left = np.random.uniform(
                    r_min,
                    -Delta/2,
                    N_train // 2
                )

                r_right = np.random.uniform(
                    Delta/2,
                    r_max,
                    N_train // 2
                )

                r_train = np.concatenate(
                    [r_left, r_right]
                )

                y_train = g(r_train, nu)

                # score-loss-inspired weight
                weights = 1.0 + lam(r_train, nu)

                # evaluation grid

                r_plot = np.linspace(
                    r_min,
                    r_max,
                    N_plot
                )

                y_true = g(r_plot, nu)

                train_mask = np.abs(r_plot) > Delta/2
                crit_mask = np.abs(r_plot) < Delta/2

                for degree in degree_list:

                    # -----------------------
                    # Fit polynomial
                    # -----------------------
                    model = make_pipeline(
                        PolynomialFeatures(degree=degree),
                        Ridge(alpha=3.0)
                    )



                    model.fit(r_train[:, None], y_train, ridge__sample_weight = weights)
                    coef = model.named_steps['ridge'].coef_
                    intercept = model.named_steps['ridge'].intercept_
                    ridge = model.named_steps['ridge']

                    # predictions
                    y_pred = model.predict(
                        r_plot[:, None]
                    )

                    poly_coeffs = np.concatenate(
                        [coef[::-1], [intercept]]
                    )


                    plot_weights = 1 + lam(r_plot, nu)

                    # -----------------------
                    # Errors
                    # -----------------------

                    train_mse = np.mean(
                        plot_weights[train_mask] * (
                            y_true[train_mask]
                            - y_pred[train_mask]
                        ) ** 2
                    )

                    r_crit = np.linspace(-Delta/2, Delta/2, N_plot)
                    y_true_crit = g(r_crit, nu)
                    y_pred_crit = model.predict(r_crit[:, None]).ravel()
                    plot_weights_crit = 1.0 + lam(r_crit, nu)

                    critical_mse = np.mean( 
                    plot_weights_crit * (y_true_crit - y_pred_crit) ** 2
                    )

                    # critical_mse = np.mean(
                    #     plot_weights[crit_mask] * (
                    #         g(r_plot[crit_mask], nu)
                    #         - y_pred[crit_mask]
                    #     )**2
                    # )

                    gap_ratio = (
                        critical_mse
                        / train_mse
                    )

                    results.append(
                        {
                            "nu": nu,
                            "Delta": Delta,
                            "degree": degree,
                            "sample_size": N_train,
                            "train_mse": train_mse,
                            "critical_mse": critical_mse,
                            "gap_ratio": gap_ratio,
                            "coeffs": poly_coeffs.copy()
                        }
                    )
                # plt.figure()
                # plt.plot(r_plot,y_true)
                # plt.plot(r_plot, y_pred)
                # # plt.axvspan(-Delta/2,Delta/2, alpha = 0.3, label = 'excised gap')
                # # plt.legend()
                # # plt.xlim(-Delta*100, Delta * 100)
                # plt.xlabel('r')
                # plt.ylabel('g(r)')
                # plt.title(fr'$\Delta$:{Delta:.2e}, $\nu$:{nu:.2e}, K:{degree}')
                # plt.show()

# =====================================================
# Print table
# =====================================================

# for r in results:

#     print(
#         f"nu={r['nu']:4.2f} "
#         f"Delta={r['Delta']:.3e} "
#         f"K={r['degree']:2d} "
#         f"train={r['train_mse']:.3e} "
#         f"critical={r['critical_mse']:.3e} "
#         f"ratio={r['gap_ratio']:.3e}"
#         f"coeff = {r['coeffs']}"
#     )

# %%
coef

# %%
#@title Error vs Delta (Error is calculated in the untrained region only)

for N_train in N:

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize =(10,6), layout = 'constrained')

    for degree in degree_list:

        xs = []
        ys1 = []
        ys2 = []

        for r in results:

            if (
                r["sample_size"] == N_train
                and r["degree"] == degree
            ):
                xs.append(r["Delta"])
                ys1.append(r["train_mse"])
                ys2.append(r["critical_mse"])

        ax[0].loglog(
            xs,
            ys1,
            marker="o",
            label=f"K={degree}"
        )
        ax[1].loglog(
            xs,
            ys2,
            marker = "o",
            label = f"K={degree}"
        )

    ax[0].set_title(
        fr"Training error vs $\Delta$ (N_train={N_train})"
    )
    ax[0].set_xlabel(r"$\Delta$")
    ax[0].set_ylabel("Train MSE")

    ax[1].set_title(fr"Critical error vs $\Delta$ (N_train={N_train})")
    ax[1].set_xlabel(r"$\Delta$")
    ax[1].set_ylabel("Critical MSE")

    ax[0].legend()
    ax[1].legend()
    plt.show()

# %% [markdown]
# # $\mathcal{L} \text{ vs } N (\Delta \text{= 1e-3, K=8},\nu=0.55)$

# %%
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Select fixed parameters & Delta targets
# -----------------------------------------------------
target_nu = 0.55
target_K = 8
target_deltas = [Delta_list[10], Delta_list[15], Delta_list[20]]

# -----------------------------------------------------
# 2. Side-by-side Plotting
# -----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')

# Styling options for differentiating Delta curves
markers = ['o', 's', '^']
linestyles = ['-', '--', '-.']

for idx, target_Delta in enumerate(target_deltas):
    # Filter data for current Delta
    filtered_data = [
        item for item in results
        if item['nu'] == target_nu 
        and item['degree'] == target_K 
        and np.isclose(item['Delta'], target_Delta)
    ]

    filtered_data.sort(key=lambda x: x['sample_size'])

    n_train = [item['sample_size'] for item in filtered_data]
    train_mse = [item['train_mse'] for item in filtered_data]
    crit_mse = [item['critical_mse'] for item in filtered_data]

    label_str = fr'$\Delta = {target_Delta:.2e}$'

    # Left Subplot: Train MSE
    ax[0].plot(
        n_train, train_mse, 
        marker=markers[idx], linestyle=linestyles[idx], 
        linewidth=2, label=label_str
    )

    # Right Subplot: Critical MSE
    ax[1].plot(
        n_train, crit_mse, 
        marker=markers[idx], linestyle=linestyles[idx], 
        linewidth=2, label=label_str
    )

# -----------------------------------------------------
# 3. Axis Configuration & Formatting
# -----------------------------------------------------
# Train MSE Subplot
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel(r'Sample Size ($N_{\text{train}}$)', fontsize=12)
ax[0].set_ylabel(r'$\mathcal{L}$', fontsize=12)
ax[0].set_title(r'Train MSE vs $N_{\text{train}}$', fontsize=13)
# ax[0].grid(True, which="both", linestyle="--", alpha=0.5)
ax[0].legend(fontsize=10)

# Critical MSE Subplot
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel(r'Sample Size ($N_{\text{train}}$)', fontsize=12)
ax[1].set_ylabel(r'$\mathcal{L}$', fontsize=12)
ax[1].set_title(r'Critical MSE vs $N_{\text{train}}$', fontsize=13)
# ax[1].grid(True, which="both", linestyle="--", alpha=0.5)
ax[1].legend(fontsize=10)

# Overall Super Title
# fig.suptitle(
#     fr'Performance Comparison Across Multiple $\Delta$ ($K={target_K}$, $\nu={target_nu}$)',
#     fontsize=14
# )
# plt.savefig(f"{fig_folder}/Finite_sample_L_scaling.pdf")
plt.show()

# %%
#@title a_k vs Delta

for N_train in N:

    plt.figure(figsize=(7,5))

    for degree in degree_list:

        xs = []
        ys = []

        for r in results:

            if (
                r["sample_size"] == N_train
                and r["degree"] == degree
            ):
                xs.append(r["Delta"])
                ys.append(r["coeffs"][0])

        order = np.argsort(xs)

        xs = np.array(xs)[order]
        ys = np.array(np.abs(ys))[order]
        a_inf = ys[0]      # smallest Delta approximation

        corr = np.abs(ys - a_inf)

        plt.loglog(
            xs[1:],
            corr[1:],
            marker="o",
            label=f"K={degree}"
        )

    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$a_K$")
    plt.title(
        f"Signed leading coefficient (N_train={N_train})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
#@title Loss-Delta power Law exponent

loss_exp1 = np.zeros(
    (len(N), len(degree_list))
)
loss_exp2 = np.zeros(
    (len(N), len(degree_list))
)

for i, N_train in enumerate(N):

    for j, degree in enumerate(degree_list):

        xs = []
        ys1 = []
        ys2 = []

        for r in results:

            if (
                r["sample_size"] == N_train
                and r["degree"] == degree
            ):
                xs.append(r["Delta"])
                ys1.append(r["train_mse"])
                ys2.append(r["critical_mse"])


        xs = np.array(xs)
        ys1 = np.array(ys1)
        ys2 = np.array(ys2)

        mask1 = ys1 > 0
        mask2 = ys2 > 0

        alpha1, _ = np.polyfit(
            np.log(xs[mask1]),
            np.log(ys1[mask1]),
            1
        )

        loss_exp1[i, j] = alpha1

        alpha2, _ = np.polyfit(
            np.log(xs[mask2]),
            np.log(ys1[mask2]),
            1
        )

        loss_exp2[i, j] = alpha2


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5), layout = 'constrained')

im1 = ax[0].imshow(
    loss_exp1,
    aspect="auto",
    origin="lower"
)

im2 = ax[1].imshow(
    loss_exp2,
    aspect= 'auto',
    origin='lower')

# ax[0].colorbar(
#     im1,
#     label=r"Loss exponent $\alpha$"
# )
cbar1 = fig.colorbar(im1, ax= ax[0])
cbar2 = fig.colorbar(im2, ax= ax[1])

ax[0].set_xticks(
    np.arange(len(degree_list)),
    degree_list
)

ax[0].set_yticks(
    np.arange(len(N)),
    N
)
ax[0].set_xlabel("K")
ax[0].set_ylabel("N")
ax[0].set_title("Train Error")


ax[1].set_xticks(
    np.arange(len(degree_list)),
    degree_list
)

ax[1].set_yticks(
    np.arange(len(N)),
    N
)
ax[1].set_xlabel("K")
ax[1].set_ylabel("N")
ax[1].set_title("Critical Error")


fig.suptitle(
    r"$E_{crit}\sim \Delta^\alpha$"
)

plt.show()

# %%
#@title a_{k-1} vs Delta power law exponent

coeff_exp = np.zeros(
    (len(N), len(degree_list))
)

for i, N_train in enumerate(N):

    for j, degree in enumerate(degree_list):

        xs = []
        ys = []

        for r in results:

            if (
                r["sample_size"] == N_train
                and r["degree"] == degree
            ):
                xs.append(r["Delta"])
                ys.append(
                    abs(abs(r["coeffs"][0]))
                )

        xs = np.array(xs)
        ys = np.array(ys)
        lead0 = ys[0]

        corr = np.abs(ys - lead0)

        # mask = ys > 0
        mask = corr > 0

        beta, _ = np.polyfit(
            np.log(xs[mask]),
            np.log(corr[mask]),
            1
        )

        coeff_exp[i, j] = beta

plt.figure(figsize=(8,5))

im = plt.imshow(
    coeff_exp,
    aspect="auto",
    origin="lower"
)

plt.colorbar(
    im,
    label=r"Coefficient exponent $\beta$"
)

plt.xticks(
    np.arange(len(degree_list)),
    degree_list
)

plt.yticks(
    np.arange(len(N)),
    N
)

plt.xlabel("Polynomial degree K")
plt.ylabel(r"N")
plt.title(
    r"$|a_{K-1}|\sim \Delta^\beta$"
)

plt.show()

# %% [markdown]
# # VALIDATION OF ASSUMPTION OF THE COVARIANCE STRUCTURE

# %%
# ==============================================================================
# Figure 1: Sample Trajectories & Covariance Matrices
# ==============================================================================
set_style(base=10)

fig, axes = plt.subplots(2, 2, figsize=(W['prl_double'], 5.2), sharex='col')

# --- Panel 1: Off-critical trajectories ---
np.random.seed(42)
X_far, tau_far = generate_samples(r=0.5, N=5, T=20)
X_near, tau_near = generate_samples(r=0.01, N=5, T=20)

t_axis = np.arange(20)
for i in range(5):
    axes[0, 0].plot(t_axis, X_far[i], alpha=0.85, marker='o', ms=3)
    axes[0, 1].plot(t_axis, X_near[i], alpha=0.85, marker='o', ms=3)

axes[0, 0].set_title(r"Off-Critical ($r=0.5, \tau=2.0$)")
axes[0, 0].set_ylabel(r"Trajectory $x_t$")
axes[0, 1].set_title(r"Near-Critical ($r=0.01, \tau=100.0$)")

# --- Panel 2: Covariance Matrices ---
Sigma_far, _ = covariance(0.5, N=50000, T=20)
Sigma_near, _ = covariance(0.01, N=50000, T=20)

vmax = max(Sigma_far.max(), Sigma_near.max())
im0 = axes[1, 0].imshow(Sigma_far, cmap='magma', origin='upper', vmin=0)
fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04, label=r"Covariance $\Sigma_{t,t'}$")
axes[1, 0].set_title(r"Covariance Matrix ($r=0.5$)")
axes[1, 0].set_xlabel("Time index $t$")
axes[1, 0].set_ylabel("Time index $t'$")

im1 = axes[1, 1].imshow(Sigma_near, cmap='magma', origin='upper', vmin=0)
fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04, label=r"Covariance $\Sigma_{t,t'}$")
axes[1, 1].set_title(r"Covariance Matrix ($r=0.01$)")
axes[1, 1].set_xlabel("Time index $t$")

for ax in axes[1, :]:
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))

plt.tight_layout()
plt.savefig(f"{fig_folder}/fig1_trajectories_and_covariance.pdf")
plt.show()

# %%
# ==============================================================================
# Figure 2: Eigenspectrum Decay, Spectral Gap, & Variance Fraction
# ==============================================================================
set_style(base=10)

# Precompute diagnostics over parameter range
r_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 1e-3, 5e-4, 1e-4]
taus, lam1_list, lam2_list, gaps, var_fracs = [], [], [], [], []
evecs_dict = {}

for r in r_list:
    Sigma, tau = covariance(r, N=50000, T=20)
    evals, evecs = np.linalg.eigh(Sigma)
    
    lam1 = evals[-1]
    lam2 = evals[-2]
    
    taus.append(tau)
    lam1_list.append(lam1)
    lam2_list.append(lam2)
    gaps.append(lam1 / lam2)
    var_fracs.append(lam1 / np.sum(evals))
    evecs_dict[r] = (evals[::-1], evecs[:, ::-1])  # Sort descending

fig, axes = plt.subplots(1, 3, figsize=(W['prl_double'], 3.2))

# --- Subplot A: Eigenspectrum Decay ---
sample_r = [0.5, 0.1, 0.02, 0.001]
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(sample_r)))

for r, color in zip(sample_r, colors):
    evals_desc, _ = evecs_dict[r]
    axes[0].plot(np.arange(1, 21), evals_desc, 'o-', ms=4, lw=1.2, 
                 color=color, label=f"$r={r}$")

axes[0].set_yscale('log')
axes[0].set_xlabel("Eigenvalue rank $k$")
axes[0].set_ylabel(r"Eigenvalue $\lambda_k$")
axes[0].set_title("Eigenspectrum Decay")
axes[0].legend(frameon=False)
axes[0].xaxis.set_major_locator(MultipleLocator(5))

# --- Subplot B: Spectral Gap Ratio ---
axes[1].plot(taus, gaps, 'o-', color='#2b5c8f', lw=1.5, ms=5)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel(r"Relaxation Time $\tau = -1/\ln a$")
axes[1].set_ylabel(r"Spectral Gap $\lambda_1 / \lambda_2$")
axes[1].set_title("Gap Saturation")
axes[1].grid(True, which='both', linestyle=':', alpha=0.4)

# --- Subplot C: Explained Variance Fraction ---
axes[2].plot(taus, var_fracs, 's-', color='#d95f02', lw=1.5, ms=5)
axes[2].set_xscale('log')
axes[2].set_xlabel(r"Relaxation Time $\tau = -1/\ln a$")
axes[2].set_ylabel(r"Variance Fraction $\lambda_1 / \text{Tr}(\Sigma)$")
axes[2].set_title("Dominant Mode Content")
axes[2].set_ylim(0.15, 0.85)
axes[2].grid(True, which='both', linestyle=':', alpha=0.4)

plt.tight_layout()
# plt.savefig(f"{fig_folder}/fig2_spectral_diagnostics.pdf")
plt.show()

# %%
# ==============================================================================
# Figure 3: Eigenvector Convergence & Vector Overlap
# ==============================================================================
set_style(base=10)

# Reference vector at r = 1e-4
Sigma_ref, _ = covariance(1e-4, N=50000, T=20)
_, evecs_ref = np.linalg.eigh(Sigma_ref)
v_ref = evecs_ref[:, -1]
if v_ref[0] < 0:
    v_ref = -v_ref  # Enforce positive orientation

overlaps = []
for r in r_list:
    Sigma, _ = covariance(r, N=50000, T=20)
    _, evecs = np.linalg.eigh(Sigma)
    v1 = evecs[:, -1]
    if v1[0] < 0:
        v1 = -v1
    overlaps.append(abs(np.dot(v1, v_ref)))

fig, axes = plt.subplots(1, 2, figsize=(W['prl_double'], 3.2))

# --- Panel A: Leading Eigenvector Profile ---
plot_r = [0.5, 0.1, 0.02, 0.0001]
colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(plot_r)))

for r, color in zip(plot_r, colors):
    _, evecs_desc = evecs_dict[r]
    v1 = evecs_desc[:, 0]
    if v1[0] < 0:
        v1 = -v1
    axes[0].plot(np.arange(20), v1, 'o-', ms=4, lw=1.3, color=color, label=f"$r={r}$")

axes[0].set_xlabel("Time coordinate $t$")
axes[0].set_ylabel(r"Eigenvector Component $v_1(t)$")
axes[0].set_title("Leading Mode Shape Convergence")
axes[0].legend(frameon=False)
axes[0].xaxis.set_major_locator(MultipleLocator(5))

# --- Panel B: Alignment Overlap ---
axes[1].plot(r_list, overlaps, 'd-', color='#7570b3', lw=1.5, ms=5)
axes[1].set_xscale('log')
axes[1].set_xlabel(r"Control Parameter $r$")
axes[1].set_ylabel(r"Overlap $|\mathbf{v}_1(r) \cdot \mathbf{v}_{\text{ref}}|$")
axes[1].set_title("Alignment with Critical Mode Limit")
axes[1].set_ylim(0.88, 1.01)
axes[1].grid(True, which='both', linestyle=':', alpha=0.4)

plt.tight_layout()
# plt.savefig(f"{fig_folder}/fig3_eigenvector_alignment.pdf")
plt.show()


