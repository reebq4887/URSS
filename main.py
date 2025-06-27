import uproot
import numpy as np
import matplotlib.pyplot as plt
MC_file = uproot.open("/storage/epp2/phshgg/Public/ew_analyses/v25/02/13TeV_2016_Down_Z_Sim09h_42112001.root")
data_file = uproot.open("/storage/epp2/phshgg/Public/ew_analyses/v25/02/13TeV_2016_28r2_Down_EW.root")
mup_PT_data = data_file['Z/DecayTree']['mup_PT'].array()
mum_PT_data = data_file['Z/DecayTree']['mum_PT'].array()
mup_PT_MC = MC_file['Z/DecayTree']['mup_PT'].array()
mum_PT_MC = MC_file['Z/DecayTree']['mum_PT'].array()
mup_ETA_data = data_file['Z/DecayTree']['mup_ETA'].array()
mum_ETA_data = data_file['Z/DecayTree']['mum_ETA'].array()
mup_ETA_MC = MC_file['Z/DecayTree']['mup_ETA'].array()
mum_ETA_MC = MC_file['Z/DecayTree']['mum_ETA'].array()
mup_PHI_data = data_file['Z/DecayTree']['mup_PHI'].array()
mum_PHI_data = data_file['Z/DecayTree']['mum_PHI'].array()
mup_PHI_MC = MC_file['Z/DecayTree']['mup_PHI'].array()
mum_PHI_MC = MC_file['Z/DecayTree']['mum_PHI'].array()
muon_mass = 105.658 #MeV
Z_M_data = data_file['Z/DecayTree']['Z_M'].array()/1000
Z_M_MC = MC_file['Z/DecayTree']['Z_M'].array()/1000

def calculate_invariantmass(mup_PT,mum_PT,mup_ETA,mum_ETA,mup_PHI,mum_PHI,muon_mass):
    mup_px = mup_PT * np.cos(mup_PHI)
    mup_py = mup_PT * np.sin(mup_PHI)
    mup_pz = mup_PT * np.sinh(mup_ETA)
    mup_energy = np.sqrt(mup_px**2 + mup_py **2 + mup_pz**2 + muon_mass**2)
    mum_px = mum_PT * np.cos(mum_PHI)
    mum_py = mum_PT * np.sin(mum_PHI)
    mum_pz = mum_PT * np.sinh(mum_ETA)
    mum_energy = np.sqrt(mum_px**2 + mum_py**2 + mum_pz**2 + muon_mass**2)
    invariant_mass = (mup_energy + mum_energy)**2 - (mup_px + mum_px)**2 - (mup_py + mum_py)**2 - (mup_pz + mum_pz)**2
    return np.sqrt(invariant_mass)

invariant_mass_data = calculate_invariantmass(mup_PT_data,mum_PT_data,mup_ETA_data,mum_ETA_data,mup_PHI_data,mum_PHI_data,muon_mass)/1000
invariant_mass_MC = calculate_invariantmass(mup_PT_MC,mum_PT_MC,mup_ETA_MC,mum_ETA_MC,mup_PHI_MC,mum_PHI_MC,muon_mass)/1000
#bins = np.arange(min(invariant_mass_data),200000+1000,1000)
bins = np.arange(80,100+1,1)
scale_factor = len(invariant_mass_data)/len(invariant_mass_MC)
#plt.plot(invariant_mass_MC)
#plt.plot(invariant_mass_data)
counts_data, bin_edges_data = np.histogram(invariant_mass_data, bins=bins)
counts_MC, bin_edges_MC = np.histogram(invariant_mass_MC, bins=bins)
counts_Z_M_data, bin_edges_Z_M_data = np.histogram(Z_M_data,bins=bins)
counts_Z_M_MC, bin_edges_Z_M_MC = np.histogram(Z_M_MC,bins=bins)
print(min(invariant_mass_data),max(invariant_mass_data))
#print(invariant_mass_data)
plt.step(bin_edges_data[:-1], counts_data, where='mid')
plt.step(bin_edges_MC[:-1], counts_MC*scale_factor, where='mid')
plt.legend(['Data','MC'])
plt.xlabel('Dimuon Invariant Mass [GeV]')
plt.ylabel('Events per GeV')
plt.show()
plt.step(bin_edges_data[:-1], counts_data, where='mid')
plt.step(bin_edges_Z_M_data[:-1], counts_Z_M_data, where='mid')
plt.legend(['Data','Z_M'])
plt.xlabel('Dimuon Invariant Mass [GeV]')
plt.ylabel('Events per GeV')
plt.show()
plt.step(bin_edges_MC[:-1], counts_MC, where='mid')
plt.step(bin_edges_Z_M_MC[:-1], counts_Z_M_MC, where='mid')
plt.legend(['MC','Z_M'])
plt.xlabel('Dimuon Invariant Mass [GeV]')
plt.ylabel('Events per GeV')
plt.show()