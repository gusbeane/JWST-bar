import arepo
import numpy as np
import os
import sys

from numba import njit

Sgr_center = np.array([80., 0., 0.])
Sgr_vel = np.array([0., 0., 80.])

NumFluidQuantities = 5
Npercell = 4
Rcut = 4.0
BoxSize = 400.0
center = np.array([BoxSize/2.0, BoxSize/2.0, BoxSize/2.0])

Nbody_snapnum = 300

@njit
def gen_mctracers(mass, ids, max_id, Npercell):
    mc_mass = np.sum(mass) / len(mass) / Npercell

    rnd = np.random.rand(len(mass))
    ntracers = np.zeros(len(mass))

    for i in range(len(mass)):
        expected = mass[i]/mc_mass
        expected_floor = np.floor(expected)
        if rnd[i] < expected - expected_floor:
            ntracers[i] = int(np.ceil(expected))
        else:
            ntracers[i] = int(expected_floor)
    
    TotNTracers = int(np.sum(ntracers))
    print(TotNTracers)
    TracerID = np.zeros(TotNTracers)
    ParentID = np.zeros(TotNTracers)
    FluidQuantities = np.zeros((TotNTracers, NumFluidQuantities))

    k = 0
    currentid = 2 * max_id
    for i in range(len(mass)):
        ntraceri = ntracers[i]

        if ntraceri > 0:
            for j in range(ntraceri):
                TracerID[k] = currentid
                ParentID[k] = ids[i]
                
                currentid += 1
                k += 1
    
    return TotNTracers, TracerID, ParentID, FluidQuantities


sn = arepo.Snapshot('./', Nbody_snapnum, combineFiles=True)
sn_gas = arepo.Snapshot('MW_ICs.dat-with-grid.hdf5')
#sn_Sgr = arepo.Snapshot('Sgr_ICs.dat')

npart = np.copy(sn.NumPart_Total)
masses = np.copy(sn.MassTable)

npart[0] = sn_gas.NumPart_Total[0]
masses[0] = sn_gas.MassTable[0]

#npart[5] = sn_Sgr.NumPart_Total[5]
#masses[5] = sn_Sgr.MassTable[5]

# now create vacuum
pos_gas = sn_gas.part0.pos - center
R = np.linalg.norm(pos_gas[:,:2], axis=1)
key = np.where(R < Rcut)[0]
gas_mass = sn_gas.part0.mass
gas_mass[key] = 0.0



#max_id = np.max([np.max(sn_gas.part0.id), np.max(sn.part1.id), np.max(sn.part2.id), np.max(sn.part3.id)])
#TotNTracers, TracerID, ParentID, FluidQuantities = gen_mctracers(gas_mass, sn_gas.part0.id, max_id, Npercell)

#npart[5] = TotNTracers

ics = arepo.ICs('ics.hdf5', npart, masses=masses)

# add MC tracer field
#ics.addField('TracerID', [0, 0, 0, 0, 0, 1], dtype='uint32')
#ics.addField('ParentID', [0, 0, 0, 0, 0, 1], dtype='uint32')
#ics.addField('FluidQuantities', [0, 0, 0, 0, 0, NumFluidQuantities], dtype='<f4')

print(npart)

ics.part0.pos[:] = sn_gas.part0.pos
ics.part0.mass[:] = gas_mass
ics.part0.vel[:] = sn_gas.part0.vel
ics.part0.id[:]  = sn_gas.part0.id
ics.part0.u[:]   = sn_gas.part0.u

ics.part1.pos[:] = sn.part1.pos + center
ics.part1.vel[:] = sn.part1.vel
ics.part1.id[:]  = sn.part1.id

ics.part2.pos[:] = sn.part2.pos + center
ics.part2.vel[:] = sn.part2.vel
ics.part2.id[:]  = sn.part2.id

ics.part3.pos[:] = sn.part3.pos + center
ics.part3.vel[:] = sn.part3.vel
ics.part3.id[:]  = sn.part3.id

#ics.part5.pos[:] = sn_Sgr.pos + center + Sgr_center
#ics.part5.vel[:] = sn_Sgr.vel + Sgr_vel

#max_id = np.max(sn_MW.part3.id)
#ics.part5.id[:] = np.arange(max_id+1, max_id + npart[5] + 1)

ics.write()

