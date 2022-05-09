import sys
sys.path.append('../')
from prs.magenpy.GWASSimulator import GWASSimulator
from prs.src.GibbsPRSSBayes import GibbsPRSSBayes

gs = GWASSimulator("../../data/1000G_EUR_Phase3_plink/1000G.EUR.QC.22.bed",
                   keep_snps="../../data/w_snplist_no_MHC.snplist.bz2",
                   pis=(0.99, 0.01),
                   h2g=0.5,
                   ld_estimator='windowed',
                   sparse_ld=True)

gs.simulate()

print("> Initializing model...")
vc = GibbsPRSSBayes(gs)
print("> Fitting model...")
vc.fit()

print(vc.get_heritability(), vc.rs_pi.mean())
#print(vc.history['ELBO'][-30:])

print("Done!")

"""
from prs.magenpy.c_utils import zarr_islice

for i, Di in enumerate(zarr_islice(gs.ld[22])):
    print(Di.shape, Di)

gs.simulate()

print("> fitting model..")
vc = vem_prs(gs) #vem_prs_cs(gs)
vc.fit(max_iter=1)
for i in range(100):
    print(vc.sigma_epsilon)
    print(vc.sigma_beta)
    print(vc.pi)
    vc.fit(max_iter=100, continued=True)

plot_history(vc)

"""