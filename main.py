import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit


#Week 1: Load data and calculate invariant mass
MC_file = uproot.open("/storage/epp2/phshgg/Public/ew_analyses/v25/02/13TeV_2016_Down_Z_Sim09h_42112001.root")
data_file = uproot.open("/storage/epp2/phshgg/Public/ew_analyses/v25/02/13TeV_2016_28r2_Down_EW.root")

def get_data(filetype, muon):
    return np.stack([filetype['Z/DecayTree']['mu'+muon+'_PT'].array(library='np'),
                     filetype['Z/DecayTree']['mu'+muon+'_ETA'].array(library='np'),
                     filetype['Z/DecayTree']['mu'+muon+'_PHI'].array(library='np')])

mup_data = get_data(data_file, 'p')
mum_data = get_data(data_file, 'm')
mup_MC = get_data(MC_file, 'p')
mum_MC = get_data(MC_file, 'm')

muon_mass = 105.658 #MeV

#in GeV
Z_M_data = data_file['Z/DecayTree']['Z_M'].array()/1000
Z_M_MC = MC_file['Z/DecayTree']['Z_M'].array()/1000

def get_momentum(muon_data):
    #in MeV
    mu_px = muon_data[0] * np.cos(muon_data[2])
    mu_py = muon_data[0] * np.sin(muon_data[2])
    mu_pz = muon_data[0] * np.sinh(muon_data[1])
    return [mu_px, mu_py, mu_pz]

def calculate_invariantmass(momenta_mum, momenta_mup, muon_mass):
    mup_energy = np.sqrt(momenta_mup[0]**2 + momenta_mup[1] **2 + momenta_mup[2]**2 + muon_mass**2)
    mum_energy = np.sqrt(momenta_mum[0]**2 + momenta_mum[1]**2 + momenta_mum[2]**2 + muon_mass**2)
    invariant_mass = (mup_energy + mum_energy)**2 - (momenta_mum[0] + momenta_mup[0])**2 - (momenta_mum[1] + momenta_mup[1])**2 - (momenta_mum[2] + momenta_mup[2])**2
    return np.sqrt(invariant_mass)/1000  # Convert to GeV

#in MeV
p_mum_MC = get_momentum(mum_MC)
p_mup_MC = get_momentum(mup_MC)
p_mum_data = get_momentum(mum_data)
p_mup_data = get_momentum(mup_data)

#in GeV
invariant_mass_data = calculate_invariantmass(p_mum_data,p_mup_data,muon_mass)
invariant_mass_MC = calculate_invariantmass(p_mum_MC,p_mup_MC,muon_mass)

bins = np.arange(80,100+1,1) #GeV
bincenters = (bins[1:]+bins[:-1])/2
scale_factor = len(invariant_mass_data)/len(invariant_mass_MC)
weight = np.ones_like(invariant_mass_MC) * scale_factor

counts_data, _ = np.histogram(invariant_mass_data, bins)
counts_MC, _ = np.histogram(invariant_mass_MC, bins)


plt.hist(invariant_mass_data, bins, histtype='stepfilled')
plt.hist(invariant_mass_MC, bins, weights=weight, histtype='step')
plt.legend(['Data','MC'])
plt.xlabel('Dimuon Invariant Mass [GeV]')
plt.ylabel('Events per GeV')
plt.show()

plt.hist(invariant_mass_data, bins, histtype='step')
plt.hist(Z_M_data, bins, histtype='step')
plt.legend(['Data','Z_M'])
plt.xlabel('Dimuon Invariant Mass [GeV]')
plt.ylabel('Events per GeV')
plt.show()

plt.hist(invariant_mass_MC, bins, histtype='step')
plt.hist(Z_M_MC, bins, histtype='step')
plt.legend(['MC','Z_M'])
plt.xlabel('Dimuon Invariant Mass [GeV]')
plt.ylabel('Events per GeV')
plt.show()


#Week 2: Smearing and Chi2 evaluation
def smearing(momenta, mu, sigma):

    #give sigma in MeV
    px_smeared = momenta[0] + np.random.normal(mu,sigma, size=len(momenta[0]))
    py_smeared = momenta[1] + np.random.normal(mu,sigma,size=len(momenta[1]))
    pz_smeared = momenta[2] + np.random.normal(mu,sigma,size=len(momenta[2]))

    return [px_smeared,py_smeared,pz_smeared]

# Function to calculate chi2
def get_chi2(params):
    mu, sigma = params
    smeared_momenta_mum = smearing(p_mum_MC, mu, sigma)
    smeared_momenta_mup = smearing(p_mup_MC, mu, sigma)

    #in GeV
    smeared_invariant_mass = calculate_invariantmass(smeared_momenta_mum,smeared_momenta_mup,muon_mass)
    counts_smeared_mc, _ = np.histogram(smeared_invariant_mass, bins)
    counts_smeared_mc = counts_smeared_mc * (counts_data.sum() / counts_smeared_mc.sum())

    mask = counts_data > 0
    sqresiduals = (counts_data[mask]-counts_smeared_mc[mask])**2 / counts_data[mask]
    S=np.array(sqresiduals).sum()/(len(bins)-2)
    print('chi2: ', S)
    return S

# Function to define a Gaussian function for fitting
def gaussian(data, total, mu, sigma):
    term = -0.5 * ((data-mu)**2)/(sigma**2)
    y = total * (1 / (sigma * np.sqrt(2 * np.pi)))* np.exp(term)
    return y

#Finding a good initial guess for mu and sigma
fitParams, fitCovariances = curve_fit(gaussian, bincenters, counts_data, p0=[counts_data.sum(), 90, 4], bounds=([0, 60, 0], [np.inf,120, np.inf]))
plt.hist(invariant_mass_data, bins, range=(80,100),histtype='step', label='Data')
plt.plot(bins,gaussian(bins,fitParams[0],fitParams[1],fitParams[2]), label='Gaussian fit')
print(fitParams)
plt.show()

# Minimizing chi2 to find best mu and sigma
res = minimize(get_chi2, x0=[fitParams[1]*1000,fitParams[2]*1000], bounds=[(0, 5000),(0, 5000)], method='Powell')
best_mu, best_sigma = res.x[0], res.x[1]
print(res)
print(best_mu, best_sigma)

# Final smearing with best parameters
smeared_momenta_mum = smearing(p_mum_MC, best_mu, best_sigma)
smeared_momenta_mup = smearing(p_mup_MC, best_mu, best_sigma)

#in GeV
smeared_invariant_mass = calculate_invariantmass(smeared_momenta_mum,smeared_momenta_mup,muon_mass)

# Plotting the final result
plt.hist(invariant_mass_data, bins, range=(80,100),histtype='step', label='Data')
plt.hist(invariant_mass_MC, bins, range=(80,100),histtype='step', label='MC unsmeared', weights=weight)
plt.hist(smeared_invariant_mass, bins, range=(80,100),histtype='step', label='MC smeared', weights=weight)

plt.xlabel('Dimuon Invariant Mass [GeV]')
plt.ylabel('Events per GeV')
plt.legend()
plt.show()