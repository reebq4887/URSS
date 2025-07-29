import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform, ReversePermutation
from sklearn.model_selection import train_test_split

muon_mass = 105.658  # MeV

#Generating pt, eta and phi for both muon particles
def generate_toy_data(mean, std_dev, nevents):
    
    ptm = np.random.normal(mean, std_dev, nevents).astype(np.float32)
    etam = np.random.uniform(2.0, 4.5, nevents).astype(np.float32)
    phim = np.random.uniform(-np.pi, np.pi, nevents).astype(np.float32)

    ptp = np.random.normal(mean, std_dev, nevents).astype(np.float32)
    etap = np.random.uniform(2.0, 4.5, nevents).astype(np.float32)
    phip = np.random.uniform(-np.pi, np.pi, nevents).astype(np.float32)

    return np.stack([ptm, etam, phim, ptp, etap, phip])

muon = generate_toy_data(45000, 10, 10000)

#Finding px, py, pz for both muon particles
def get_three_momenta(muon_data):
    
    pxm = muon_data[0] * np.cos(muon_data[2])
    pym = muon_data[0] * np.sin(muon_data[2])
    pzm = muon_data[0] * np.sinh(muon_data[1])

    pxp = muon_data[3] * np.cos(muon_data[5])
    pyp = muon_data[3] * np.sin(muon_data[5])
    pzp = muon_data[3] * np.sinh(muon_data[4])

    return np.array([pxm, pym, pzm, pxp, pyp, pzp])

momenta = get_three_momenta(muon).T

#Find energies of both muon particles using 3-momenta
def get_energies(momenta):
        Em = np.sqrt(momenta[:,0]**2 + momenta[:,1]**2 + momenta[:,2]**2 + muon_mass**2)
        Ep = np.sqrt(momenta[:,3]**2 + momenta[:,4]**2 + momenta[:,5]**2 + muon_mass**2)
        return np.array([Em, Ep])

#Split and Shuffle the Data
data_train, data_test = train_test_split(
    momenta,
    test_size=0.2,
    train_size=0.8,
    random_state=42,
    shuffle=True
)

n_dimensions = 6
n_layers = 6

#Creating Normalising Flow
def create_flow():
    base_dist = StandardNormal([n_dimensions])

    transforms = []
    for i in range(0,n_layers):
        transforms.append(MaskedAffineAutoregressiveTransform(features=n_dimensions, hidden_features=128))
        transforms.append(ReversePermutation(features=n_dimensions))
    
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow

flows = create_flow()
optimiser = torch.optim.Adam(flows.parameters(), lr=1e-3)

#Defining variable transformations
def normalise_data(in_data, max_val, min_val):
  new_data = (in_data-min_val)/(max_val-min_val)
  mask = np.prod(((new_data < 1) & (new_data > 0 )), axis=1, dtype=bool)
  new_data = new_data[mask]
  return new_data, mask

def logit_data(in_data):
  new_data = np.log(in_data/(1-in_data))
  return new_data

def standardise_data(in_data, mean_val, std_val):
  new_data = (in_data - mean_val)/std_val
  return new_data


#Defining their inverse transformations
def normalise_inverse(in_data, max_val, min_val):
  new_data = in_data*(max_val-min_val) + min_val
  return new_data

def logit_inverse(in_data):
  new_data = (1+np.exp(-in_data))**(-1)
  return new_data

def standardise_inverse(in_data, mean_val, std_val):
  new_data = std_val*in_data + mean_val
  return new_data

max_values = np.max(data_train, keepdims=True,axis=0)
min_values = np.min(data_train, keepdims=True,axis=0)

# Normalise the data (only 80% dedicated for training)
transformed_momenta, mask = normalise_data(data_train, max_values, min_values)

#logit transformation
transformed_momenta = logit_data(transformed_momenta)

#standardize the data
mean_values = np.mean(transformed_momenta, keepdims=True, axis=0)
std_values = np.std(transformed_momenta,   keepdims=True, axis=0)
transformed_momenta = standardise_data(transformed_momenta, mean_values, std_values)

nepochs = 10
batch_size = 256
max_batches = int(transformed_momenta.shape[0] / batch_size)

#Training
for i in range(nepochs):
    perm = torch.randperm(transformed_momenta.shape[0])
    transformed_data_shuffle = transformed_momenta[perm]
    for i_batch in range(max_batches):

        x = transformed_data_shuffle[i_batch*batch_size:(i_batch+1)*batch_size]
        x = torch.tensor(x).float()

        optimiser.zero_grad()

        loss = -flows.log_prob(x).mean()

        if i_batch % 50 == 0:
            print(loss.item())        
        
        loss.backward()
        optimiser.step()
        
    if i % 1 == 0:
      print('Epoch: {:d}'.format(i))

#Defining what dataset to sample from
sampling_data_set = data_test 

with torch.no_grad():
    # Generated samples
    generated_momenta = flows.sample(sampling_data_set.shape[0]).cpu().numpy()

print(generated_momenta.shape)

#Retransforming the data
retransformed_samples = standardise_inverse(generated_momenta,mean_val=mean_values,std_val=std_values)
retransformed_samples = logit_inverse(retransformed_samples)
retransformed_samples = normalise_inverse(retransformed_samples,max_values,min_values)

#Function to calculate invariant mass from 4-momenta
def calculate_invariant_mass(energies, momenta):
    array = ((energies[0] + energies[1])**2 - 
                             (momenta[:,0] + momenta[:,3])**2 - 
                             (momenta[:,1] + momenta[:,4])**2 - 
                             (momenta[:,2] + momenta[:,5])**2)
    
    # Ensure non-negative values for square root
    invariant_mass = np.sqrt(np.maximum(0, array))
    return invariant_mass/1000 # Convert to GeV

#Get the energies
energies = get_energies(sampling_data_set)
generated_energies = get_energies(retransformed_samples)

#Find Invariant Mass Distribution
invariant_mass = calculate_invariant_mass(energies, sampling_data_set)
generated_invariant_mass = calculate_invariant_mass(generated_energies, retransformed_samples)

print(generated_invariant_mass.shape)

data_color = "grey"
sample_color = "darkblue"

#Plotting the Invariant Mass Distribution and Sampled Mass Distribution
fig, ax = plt.subplots(1,1,figsize=(12,8))
_, common_bins, _ = ax.hist(invariant_mass, 100, alpha=0.3, label="data", color=data_color)
ax.hist(generated_invariant_mass, common_bins, label="sampled", edgecolor=sample_color, linewidth=1.3, histtype="step")
ax.set_xlabel("Z Boson mass (GeV)")
ax.set_ylabel("events")

counts_data, bin_edges = np.histogram(invariant_mass, bins=len(common_bins))
counts_model, _ = np.histogram(generated_invariant_mass, bins=len(common_bins))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#Avoid division by zero
mask = counts_data > 0

# Pull calculation
# pulls = np.zeros_like(counts_data, dtype=float)
# pulls[mask] = (counts_data[mask] - counts_model[mask]) / np.sqrt(counts_data[mask])

#Finding Chi-squared value
sqresiduals = (counts_data[mask]-counts_model[mask])**2 / counts_data[mask]
S=np.array(sqresiduals).sum()/(len(common_bins))
print("Reduced Chi-Squared: ", S)

#Plotting the Pull
# frame2 = fig.add_axes((.1,.1,.8,.2))        
# plt.plot(bin_centers, pulls, color='gray',marker='.', linestyle='None')
# plt.ylabel('Pull')

fig.tight_layout()
fig.text(
    0.05, 0.95,
    fr'Reduced $\chi^2 = {S:.2f}$',
    transform=plt.gca().transAxes,
    ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)
plt.savefig("Binned Plots/Training Data Invariant Mass Distribution - 3.png")
plt.close()


#Plotting the 6 momenta components and their samples
fig, ax = plt.subplots(3,2, figsize=(12,8))

_, common_bins, _ = ax[0,0].hist(sampling_data_set[:,0], 100, alpha=0.3, label="data", color=data_color)
ax[0,0].hist(retransformed_samples[:,0], common_bins, label="sampled", edgecolor=sample_color, linewidth=1.3, histtype="step")
ax[0,0].set_xlabel("px_m")
ax[0,0].set_ylabel("events")

_, common_bins, _ = ax[0,1].hist(sampling_data_set[:,1], 100, alpha=0.3, label="data", color=data_color)
ax[0,1].hist(retransformed_samples[:,1], common_bins, label="sampled", edgecolor=sample_color, linewidth=1.3, histtype="step")
ax[0,1].set_xlabel("py_m")
ax[0,1].set_ylabel("events")

_, common_bins, _ = ax[1,0].hist(sampling_data_set[:,2], 100, alpha=0.3, label="data", color=data_color)
ax[1,0].hist(retransformed_samples[:,2], common_bins, label="sampled", edgecolor=sample_color, linewidth=1.3, histtype="step")
ax[1,0].set_xlabel("pz_m")
ax[1,0].set_ylabel("events")

_, common_bins, _ = ax[1,1].hist(sampling_data_set[:,3], 100, alpha=0.3, label="data", color=data_color)
ax[1,1].hist(retransformed_samples[:,3], common_bins, label="sampled", edgecolor=sample_color, linewidth=1.3, histtype="step")
ax[1,1].set_xlabel("px_p")
ax[1,1].set_ylabel("events")

_, common_bins, _ = ax[2,0].hist(sampling_data_set[:,4], 100, alpha=0.3, label="data", color=data_color)
ax[2,0].hist(retransformed_samples[:,4], common_bins, label="sampled", edgecolor=sample_color, linewidth=1.3, histtype="step")
ax[2,0].set_xlabel("py_p")
ax[2,0].set_ylabel("events")

_, common_bins, _ = ax[2,1].hist(sampling_data_set[:,5], 100, alpha=0.3, label="data", color=data_color)
ax[2,1].hist(retransformed_samples[:,5], common_bins, label="sampled", edgecolor=sample_color, linewidth=1.3, histtype="step")
ax[2,1].set_xlabel("pz_p")
ax[2,1].set_ylabel("events")

fig.tight_layout()
plt.savefig("Binned Plots/Training Data Momenta components - 3.png")
plt.close()






