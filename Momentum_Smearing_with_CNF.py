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
import copy

# ---------- Functions to retrieve data ----------

# Generating toy data for code testing
def generate_toy_data(mean, std_dev, nevents):

    ptm = torch.Tensor(np.random.normal(mean, std_dev, nevents).astype(np.float32))
    etam = torch.Tensor(np.random.uniform(2.0, 4.5, nevents).astype(np.float32))
    phim = torch.Tensor(np.random.uniform(-np.pi, np.pi, nevents).astype(np.float32))

    ptp = torch.Tensor(np.random.normal(mean, std_dev, nevents).astype(np.float32))
    etap = torch.Tensor(np.random.uniform(2.0, 4.5, nevents).astype(np.float32))
    phip = torch.Tensor(np.random.uniform(-np.pi, np.pi, nevents).astype(np.float32))

    return torch.stack([ptm, etam, phim, ptp, etap, phip], dim=0)

# Retrieving actual data from Uproot files as tensors
def get_data(filetype):
    return torch.stack([torch.tensor(filetype['Z/DecayTree']['mum_PT'].array(library='np')),
                     torch.tensor(filetype['Z/DecayTree']['mum_ETA'].array(library='np')),
                     torch.tensor(filetype['Z/DecayTree']['mum_PHI'].array(library='np')),
                     torch.tensor(filetype['Z/DecayTree']['mup_PT'].array(library='np')),
                     torch.tensor(filetype['Z/DecayTree']['mup_ETA'].array(library='np')),
                     torch.tensor(filetype['Z/DecayTree']['mup_PHI'].array(library='np'))], dim=0)

# Retrieving contexts from Uproot files
def get_contexts(filetype):
    return torch.stack([
                     torch.tensor(filetype['Z/DecayTree']['mum_ETA'].array(library='np')),
                     torch.tensor(filetype['Z/DecayTree']['mup_ETA'].array(library='np')),
                     torch.tensor(filetype['Z/DecayTree']['mum_Q'].array(library='np')),
                     torch.tensor(filetype['Z/DecayTree']['mup_Q'].array(library='np'))], dim=0)

# Generating 3-momenta (px, py, pz) from (pt, eta, phi)
def get_three_momenta(muon_data):

    pxm = muon_data[0] * torch.cos(muon_data[2])
    pym = muon_data[0] * torch.sin(muon_data[2])
    pzm = muon_data[0] * torch.sinh(muon_data[1])

    pxp = muon_data[3] * torch.cos(muon_data[5])
    pyp = muon_data[3] * torch.sin(muon_data[5])
    pzp = muon_data[3] * torch.sinh(muon_data[4])

    return torch.stack([pxm, pym, pzm, pxp, pyp, pzp], dim=1)

# Calculating invariant mass from (px, py, pz)
def calculate_invariant_mass(momenta):

    Em = torch.sqrt(momenta[:, 0]**2 + momenta[:, 1]**2 + momenta[:, 2]**2 + muon_mass**2)
    Ep = torch.sqrt(momenta[:, 3]**2 + momenta[:, 4]**2 + momenta[:, 5]**2 + muon_mass**2)

    M2 = ((Em + Ep)**2
          - (momenta[:, 0] + momenta[:, 3])**2
          - (momenta[:, 1] + momenta[:, 4])**2
          - (momenta[:, 2] + momenta[:, 5])**2)

    invariant_mass = torch.sqrt(torch.clamp(M2, min=0.0))
    return invariant_mass / 1000.0  # convert to GeV

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


def retransform_data(transformed_data, params, eps=1e-6):

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
def make_flow(num_layers=4, hidden_features=32, features=6, context_features = 4):
    layers = []
    for _ in range(num_layers):
        layers.append(RandomPermutation(features=features))
        layers.append(MaskedAffineAutoregressiveTransform(features=features, hidden_features=hidden_features, context_features=context_features))
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

muon_mass = 105.658  # MeV
n_samples = 1000
n_dimensions = 6 # Dimensions of space on which the flow is learning (in this case the 6D momentum vector)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using {n_samples} samples from each dataset")

MC_file = uproot.open("/storage/epp2/phshgg/Public/ew_analyses/v25/02/13TeV_2016_Down_Z_Sim09h_42112001.root")
data_file = uproot.open("/storage/epp2/phshgg/Public/ew_analyses/v25/02/13TeV_2016_28r2_Down_EW.root")

muon_data = get_data(data_file)[:,:n_samples].to(device) #Target data
# muon_MC = generate_toy_data(30000,10,nevents).to(device) #Alternatively use toy data
muon_MC = get_data(MC_file)[:,:n_samples].to(device) #Source Data (to smear)

# Using MC file data for muon_MC does not work, i.e. when the source data to smear is MC data, 
# the model does not learn and the loss is unchanged. This issue does not occur when:
# 1) Using toy data for both, muon_data and muon_MC
# 2) Using real data for muon_data (target) and toy data for muon_MC (source)
# 3) Using real MC data for muon_data (target) and real/toy data for muon_MC (source)


# Add as contexts if using toy data
# chargem = -1 * torch.ones(n_samples)
# chargep = torch.ones(n_samples)

# Get 3-momenta and contexts
momenta_data = get_three_momenta(muon_data)
momenta_MC = get_three_momenta(muon_MC)

# muon_contexts = torch.stack([muon_data[1,:], muon_data[4,:], chargem, chargep]).T.to(device) #Add if using toy data
muon_contexts = get_contexts(MC_file)[:,:n_samples] # Remove if using toy data
muon_contexts = muon_contexts[:4, :].T.to(device) # Just getting it into the right shape (remove if using toy data)

# Transform the source (MC) dataset and contexts for easier training, and mask to exclude any outliers
MC_transformed, mask_MC, params_MC = transform_data(momenta_MC, eps=1e-6)
muon_contexts_transformed, mask_c, params_c = transform_data(muon_contexts)

MC_transformed = MC_transformed[mask_MC] 
momenta_MC = momenta_MC[mask_MC]
muon_contexts_transformed = muon_contexts_transformed[mask_MC]

print("MC kept after masking:", mask_MC.sum().item(), "/", len(mask_MC))

# Create the flow
flow = make_flow(num_layers=10, hidden_features=128)  # Increased capacity
flow = Flow(transform=flow, distribution=StandardNormal([6])).to(device)  # 6D latent
flow = flow.to(device)
optimizer = optim.Adam(flow.parameters(), lr=5e-4)

# Grid for KDE
xmin = 30
xmax = 120
edges = torch.linspace(xmin,xmax,101,device=device)
bin_width = edges[1]-edges[0]

# Training parameters
batch_size = 256
n_epochs = 200  # Increase epochs for better convergence

# Some placeholders to store the best parameters found during training (not doing much if anything at all)
best_loss = float('inf')
best_state = None

# ------------------  THE MAIN TRAINING LOOP  ------------------------------------------
for epoch in range(n_epochs):

    # Sample batch from transformed source data
    idx = torch.randint(0, MC_transformed.shape[0], (batch_size,), device=device)
    x_batch = MC_transformed[idx]
    c_batch = muon_contexts_transformed[idx]

    # Transform through flow: transformed_data -> latent -> transformed_target_space
    y_batch, _ = flow._transform.forward(x_batch, context=c_batch)

    # Convert back to original space using source parameters
    smeared_momenta = retransform_data(y_batch, params_MC)

    # Calculate mass in original space
    smeared_mass = calculate_invariant_mass(smeared_momenta)
    smeared_mass = smeared_mass[torch.isfinite(smeared_mass)]  # Remove NaN/Inf

    smeared_pdf = smear_histogram(smeared_mass, edges) # KDE for loss calculation

    # Target PDF (precomputed on full data)
    with torch.no_grad():
        target_mass = calculate_invariant_mass(momenta_data) # Target mass batched
        target_pdf = smear_histogram(target_mass, edges=edges)[:batch_size] # KDE for loss calculation

    loss = wasserstein_1d_weighted(smeared_pdf, target_pdf, bin_width)
    optimizer.zero_grad()
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in flow.parameters() if p.grad is not None)

    print("Grad norm:", grad_norm) # Checking if gradient is changing

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
    
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
    y_all, _ = flow._transform.forward(MC_transformed, context = muon_contexts_transformed) # Intermediate variable
    smeared_momenta = retransform_data(y_all, params_MC) # Restandardising data
    smeared_mass_all = calculate_invariant_mass(smeared_momenta).cpu().numpy() # Smeared Mass
    smeared_mass_all = smeared_mass_all[np.isfinite(smeared_mass_all)]  # Remove NaN/Inf

target_mass_data = calculate_invariant_mass(momenta_data[:n_samples]).cpu().numpy() # Target
mass_MC = calculate_invariant_mass(momenta_MC[:n_samples]).cpu().numpy() # Source

# compare some stats
print("Target mass: mean/min/max:", target_mass_data.mean(), target_mass_data.min(), target_mass_data.max())
print("Smeared mass: mean/min/max:", smeared_mass_all.mean(), smeared_mass_all.min(), smeared_mass_all.max())

bin_centers = 0.5 * (edges[1:] + edges[:-1])

# Finding KDE for Target, Source and Smeared Mass for plotting
kde_mass_data = smear_histogram(target_mass_data, edges)
kde_mass_MC = smear_histogram(mass_MC, edges)
kde_y_mass_all = smear_histogram(smeared_mass_all, edges)

# Final Plotting
plt.figure(figsize=(12,8))
plt.hist(target_mass_data, bins=100,label="Data", color='tab:red', histtype='stepfilled',density=True, alpha=0.2)
plt.hist(mass_MC, bins=100,label="MC (Unsmeared)", color='tab:blue', histtype='step',density=True)
plt.hist(smeared_mass_all, bins=100,label="MC (smeared)", color='tab:green', histtype='step',density=True)
plt.plot(bin_centers, kde_mass_data, label="Data", color='tab:red')
plt.plot(bin_centers, kde_mass_MC, label="MC (Unsmeared)", color='tab:blue')
plt.plot(bin_centers, kde_y_mass_all, label="MC (smeared)", color='tab:green')
plt.title('Momentum Smearing with Normalizing Flow')
plt.xlabel('Invariant Mass (GeV)')
plt.ylabel('Density')
plt.xlim((30,150))
plt.legend(); plt.grid(alpha=0.2)
plt.savefig("Momentum Smearing.png", dpi=150)
plt.show()
