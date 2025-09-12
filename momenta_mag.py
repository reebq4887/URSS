import numpy as np
import matplotlib.pyplot as plt
import uproot
import torch
from torch import optim
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.nn import nets
import copy

# ---------- Functions to retrieve data ----------
# Retrieving actual data from Uproot files as tensors
def retrieve_data(filetype, n_samples):
    mum_p = filetype['DecayTree']['mum_uncalibrated_P'].array(library='np')
    mup_p = filetype['DecayTree']['mup_uncalibrated_P'].array(library='np')
    V_M    = filetype['DecayTree']['V_M'].array(library='np')
    mum_eta = filetype['DecayTree']['mum_eta'].array(library='np')
    mup_eta = filetype['DecayTree']['mup_eta'].array(library='np')
    mum_q   = filetype['DecayTree']['mum_q'].array(library='np')
    mup_q   = filetype['DecayTree']['mup_q'].array(library='np')

    mask = (V_M > 70) & (V_M < 110)
    mum_p, mup_p = mum_p[mask], mup_p[mask]
    V_M          = V_M[mask]
    mum_eta, mup_eta = mum_eta[mask], mup_eta[mask]
    mum_q, mup_q     = mum_q[mask], mup_q[mask]

    N = len(mum_p)
    if n_samples > N:
        raise ValueError(f"Requested n_samples={n_samples} exceeds available masked events={N}")
    
    indices = np.arange(N)
    np.random.shuffle(indices)
    indices = indices[:n_samples]

    mum_p, mup_p = mum_p[indices], mup_p[indices]
    V_M          = V_M[indices]
    mum_eta, mup_eta = mum_eta[indices], mup_eta[indices]
    mum_q, mup_q     = mum_q[indices], mup_q[indices]

    # --- Build tensors ---
    momenta = (torch.tensor(np.stack([mum_p, mup_p], axis=1), dtype=torch.float32))/1000
    
    E1 = torch.sqrt(momenta[:,0]**2 + muon_mass**2)
    E2 = torch.sqrt(momenta[:,1]**2 + muon_mass**2)
    angles = torch.tensor((E1 * E2 + muon_mass**2 - 0.5 * V_M**2) / (momenta[:,0] * momenta[:,1] + 1e-12),
                          dtype=torch.float32)
    
    contexts = torch.tensor(np.stack([mum_eta, mup_eta, mum_q, mup_q], axis=1), dtype=torch.float32)
    
    return momenta, angles, contexts, indices
    

# Calculating invariant mass from (px, py, pz)
def calculate_invariant_mass(momenta, costheta):

    M2 = 2*muon_mass**2 + 2 * torch.sqrt((muon_mass**2 + momenta[:,0]**2)*(muon_mass**2 + momenta[:,1]**2)) - 2*momenta[:,0]*momenta[:,1]*costheta

    invariant_mass = torch.sqrt(torch.clamp(M2, min=0.0))
    return invariant_mass  # convert to GeV

# --------------------------------------------------------------------------------------------

# ---------- Functions for Standardising Data ------------------------------------------------
def transform_data(in_data, eps=1e-6):

    # Min-max scale to (0,1)
    min_val = torch.min(in_data, dim=0, keepdim=True).values
    max_val = torch.max(in_data, dim=0, keepdim=True).values
    denom = torch.clamp(max_val - min_val, min=eps)
    scaled = (in_data - min_val) / denom

    # Mask for valid range (0,1)
    mask = ((scaled < 1.0 - eps) & (scaled > eps)).all(dim=1)

    # Clamp scaled to avoid logit infinities
    scaled = torch.clamp(scaled, eps, 1 - eps)

    # Logit transform
    logit_data = torch.log(scaled / (1 - scaled))

    # Z-score
    mean_val = torch.mean(logit_data, dim=0, keepdim=True)
    std_val = torch.std(logit_data, dim=0, keepdim=True) + eps
    transformed = (logit_data - mean_val) / std_val

    # Return transformed data, mask, and parameters
    return transformed, mask, (min_val, max_val, mean_val, std_val)
    
def inverse_transform(transformed_data, params, eps=1e-6):

    min_val, max_val, mean_val, std_val = params

    # Undo z-score
    x = transformed_data * std_val + mean_val
    # Sigmoid inverse of logit
    scaled = torch.sigmoid(x)

    # Undo min-max
    denom = torch.clamp(max_val - min_val, min=eps)
    original = scaled * denom + min_val

    return original

# ---------------------------------------------------------------------------------------------

# ---------- Functions to create flow and calculate loss  -------------------------------------

# Creating a Conditional Normalising Flow (remove context features if not using contexts)
def make_flow(num_layers=4, hidden_features=32, features=2, context_features = 4):
    # layers = []
    # for _ in range(num_layers):
    #     layers.append(RandomPermutation(features=features))
    #     layers.append(MaskedAffineAutoregressiveTransform(features=features, hidden_features=hidden_features, context_features = context_features))
    # return CompositeTransform(layers)

    layers = []
    for _ in range(num_layers):
        layers.append(RandomPermutation(features=features))
        layers.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=torch.arange(features) % 2,  # alternate mask for coupling
                transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_blocks=2,
                    activation=torch.relu,
                    dropout_probability=0.0,
                    use_batch_norm=False,
                ),
                num_bins=8,
                tails='linear',
                tail_bound=3.0
            )
        )
    return CompositeTransform(layers)

# Calculating a custom Wasserstein Loss for Invariant Mass (using pdfs)
def wasserstein_1d_weighted(pdf_a, pdf_b, bin_width, peak_weight=1.0):
    """Wasserstein distance with extra weight on peak region"""    
    # Ensure densities are finite and normalized
    cdf_a = torch.cumsum(pdf_a, dim=0) * bin_width
    cdf_b = torch.cumsum(pdf_b, dim=0) * bin_width

    cdf_a = torch.clamp(cdf_a, 0, 1)
    cdf_b = torch.clamp(cdf_b, 0, 1)

    w1 = torch.sum(torch.abs(cdf_a - cdf_b)) * bin_width
    peak = torch.mean((pdf_a - pdf_b) ** 2)
    return w1 + peak_weight * peak  #Adding a slight weight to the peak to help the flow focus more on the peak than the tails

# Using KDE to smear histograms
def smear_histogram(
    data: torch.Tensor,
    edges: torch.Tensor,
    *,
    kernel_width: float = 0.1,
    eps: float = 1e-8
) -> torch.Tensor:

    # Make shapes compatible
    data = data[:, None]
    edges2 = edges[None, :]

    def normal_cdf(x, mu, sigma):
        return 0.5 * (1 + torch.erf((x - mu) / (sigma * (2.0 ** 0.5))))

    right = normal_cdf(edges2[:, 1:], mu=data, sigma=kernel_width)
    left  = normal_cdf(edges2[:, :-1], mu=data, sigma=kernel_width)
    counts = torch.sum(right-left, dim=0)
    pdf = counts / (counts.sum() * bin_width + 1e-8)
    return pdf

# --------------------------------------------------------------

# ---------- Seed for Reproducibility ----------
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
# ----------------------------------------------

muon_mass = 0.105658  # GeV
n_dimensions = 2 # Dimensions of space on which the flow is learning (in this case the 6D momentum vector)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MC_file = uproot.open("/storage/epp2/phrqtm/momentum_smearing_root_files/DecayTree__Z__DATA__d13TeV_2016.root")
data_file = uproot.open("/storage/epp2/phrqtm/momentum_smearing_root_files/DecayTree__Z__Z__d13TeV_2016.root")

momenta_data, angles_data, contexts_data, indices_data = retrieve_data(data_file, n_samples=100000)
momenta_MC, angles_MC, contexts_MC, indices_MC = retrieve_data(MC_file, n_samples=100000)

# Transform the source (MC) dataset and contexts for easier training, and mask to exclude any outliers
MC_transformed, mask_MC, params_MC = transform_data(momenta_MC, eps=1e-6)
muon_contexts_transformed, mask_c, params_c = transform_data(contexts_MC, eps=1e-6)

MC_transformed = MC_transformed[mask_MC] 
momenta_MC = momenta_MC[mask_MC]
muon_contexts_transformed = muon_contexts_transformed[mask_MC]
angles_MC = angles_MC[mask_MC]

print("MC kept after masking:", mask_MC.sum().item(), "/", len(mask_MC))


# Create the flow
flow = make_flow(num_layers=10, hidden_features=128)  # Increased capacity
flow = Flow(transform=flow, distribution=StandardNormal([n_dimensions])).to(device)  # 2D latent
flow = flow.to(device)
optimizer = optim.Adam(flow.parameters(), lr=5e-4)

# Grid for KDE
xmin = 80
xmax = 100
bins = 300
edges = torch.linspace(70,110, 601,device=device)
bin_width = edges[1]-edges[0]


# Training parameters
batch_size = 256
n_epochs = 600  # Increase epochs for better convergence

# Some placeholders to store the best parameters found during training (not doing much if anything at all)
best_loss = float('inf')
best_state = None

with torch.no_grad():
    target_mass = calculate_invariant_mass(momenta_data, angles_data) # Target mass calculated at the same indices as MC momenta
    target_pdf = smear_histogram(target_mass, edges=edges)
# ------------------  THE MAIN TRAINING LOOP  ------------------------------------------
for epoch in range(n_epochs):

    # Sample batch from transformed source data
    idx = torch.randint(0, MC_transformed.shape[0], (batch_size,), device=device)
    x_batch = MC_transformed[idx]
    
    c_batch = muon_contexts_transformed[idx]
    angle_batch = angles_MC[idx]
    # Transform through flow: transformed_data -> latent -> transformed_target_space
    y_batch, _ = flow._transform.forward(x_batch, context = c_batch)
    # Convert back to original space using source parameters
    smeared_momenta = inverse_transform(y_batch, params_MC)

    # Calculate mass in original space
    smeared_mass = calculate_invariant_mass(smeared_momenta, angle_batch)
    smeared_mass = smeared_mass[torch.isfinite(smeared_mass)]  # Remove NaN/Inf
    smeared_pdf = smear_histogram(smeared_mass, edges) # KDE for loss calculation
    print(smeared_mass)
    # Target PDF (precomputed on full data)

    loss = wasserstein_1d_weighted(smeared_pdf, target_pdf, bin_width)
    optimizer.zero_grad()
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in flow.parameters() if p.grad is not None)

    print("Grad norm:", grad_norm) # Checking if gradient is changing

    # Gradient clipping for stability
    # torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
    
    optimizer.step()
    
    # Finding best loss parameters
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_state = copy.deepcopy(flow.state_dict())

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}")

# Loading state with best parameters
if best_state is not None:
    flow.load_state_dict(best_state)

print("Training completed!")

# Generate final results
with torch.no_grad():
    y_all, _ = flow._transform.forward(MC_transformed,context = muon_contexts_transformed) # Intermediate variable
    smeared_momenta = inverse_transform(y_all, params_MC) # Restandardising data
    smeared_mass_all = calculate_invariant_mass(smeared_momenta, angles_MC).cpu().numpy() # Smeared Mass
    smeared_mass_all = smeared_mass_all[np.isfinite(smeared_mass_all)]  # Remove NaN/Inf

target_mass_data = calculate_invariant_mass(momenta_data, angles_data).cpu().numpy() # Target
mass_MC = calculate_invariant_mass(momenta_MC, angles_MC).cpu().numpy() # Source

# Pull Calculation

counts_data, edges_hist = np.histogram(target_mass_data, bins=bins,range=(xmin, xmax), density=False)
counts_smeared, _ = np.histogram(smeared_mass_all, bins=bins,range=(xmin, xmax),density=False)
errors = np.sqrt(counts_data)
errors[errors == 0] = 1.0
pulls = (counts_data - counts_smeared) / errors
# compare some stats
print("Target mass: mean/min/max:", target_mass_data.mean(), target_mass_data.min(), target_mass_data.max())
print("Smeared mass: mean/min/max:", smeared_mass_all.mean(), smeared_mass_all.min(), smeared_mass_all.max())

bin_centers = 0.5 * (edges_hist[1:] + edges_hist[:-1])

# Finding KDE for Target, Source and Smeared Mass for plotting
kde_mass_data = smear_histogram(target_mass_data, edges)
kde_mass_MC = smear_histogram(mass_MC, edges)
kde_y_mass_all = smear_histogram(smeared_mass_all, edges)

# Final Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10), sharex=True, gridspec_kw={'height_ratios':[3,1]})
ax1.hist(target_mass_data, range=(xmin, xmax),bins=bins,label="Data", color='tab:red', histtype='stepfilled',density=True, alpha=0.2)
ax1.hist(mass_MC, range=(xmin, xmax),bins=bins,label="MC (Unsmeared)", color='tab:blue', histtype='step',density=True)
ax1.hist(smeared_mass_all, range=(xmin, xmax),bins=bins,label="MC (smeared)", color='tab:green', histtype='step',density=True)
# plt.plot(bin_centers, kde_mass_data, label="Data", color='tab:red')
# plt.plot(bin_centers, kde_mass_MC, label="MC (Unsmeared)", color='tab:blue')
# plt.plot(bin_centers, kde_y_mass_all, label="MC (smeared)", color='tab:green')
ax1.set_title('Momentum Smearing with Normalizing Flow')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(alpha=0.2)

ax2.axhline(0, color='black', linewidth=1)
ax2.axhline(3, color='gray', linestyle='--', linewidth=1)
ax2.axhline(-3, color='gray', linestyle='--', linewidth=1)

ax2.bar(bin_centers, pulls,
        width=(edges_hist[1] - edges_hist[0]),
        color='tab:purple', alpha=0.6)

ax2.set_xlabel('Invariant Mass (GeV)')
ax2.set_ylabel('Pulls')
ax2.set_ylim(-5, 5)
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("Piecewise_Momentum_Smearing_with_Pulls_test.png", dpi=150)
plt.show()
