import sys
from gwasimulator.GWASDataLoader import GWASDataLoader
from gwasimulator.GWASSimulator import GWASSimulator
from vem import vem_prs

gs = GWASSimulator("../data/genotype_data/1000G.EUR.QC.22",
                   keep_snps="../data/w_snplist_no_MHC.snplist.bz2")
gs.simulate()

v = vem_prs(gs)
v.iterate()
