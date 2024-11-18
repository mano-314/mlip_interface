import re 
import numpy as np 
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.visualize import view

def read_cfg(filename, mapping_atoms):
    with open(filename) as f:
        data=f.read()

    structures = []

    pattern_1 = "BEGIN_CFG"
    pattern_2 = "END_CFG"

    begin_tags = []
    for match_1 in re.finditer(pattern_1,data):
        begin_tags.append(match_1.start())
    end_tags = []
    for match_2 in re.finditer(pattern_2,data):
        end_tags.append(match_2.start())

    for block_begin, block_end in zip(begin_tags, end_tags):
        data_slice = data[block_begin:block_end]
        for match_3 in re.finditer("Size",data_slice):
            tmp_start=match_3.start() 
            size=int(data_slice[tmp_start:tmp_start+30].split()[1])
        for match_4 in re.finditer("Supercell",data_slice):
            tmp_start=match_4.start() 
            supercell=data_slice[tmp_start:tmp_start+200].split()[1:1+9]
            supercell=np.array(supercell,dtype=np.float64).reshape(3,3)
        for match_5 in re.finditer("AtomData",data_slice):
            tmp_start=match_5.start() 
            positions_block=data_slice[tmp_start:tmp_start+ 100*(size+1)].split("\n")
            positions_block = np.loadtxt(positions_block[1:1+size])
            type_index=positions_block[:,1].astype(np.int16)
            positions=positions_block[:,2:5]
            forces=positions_block[:,5:]
        for match_6 in re.finditer("Energy",data_slice):
            tmp_start=match_6.start() 
            energy=float(data_slice[tmp_start:tmp_start+30].split()[1])
        for match_7 in re.finditer("PlusStress",data_slice):
            tmp_start=match_7.start() 
            stress=data_slice[tmp_start:tmp_start+200].split("\n")[1].split()
            stress=np.loadtxt(stress)

        atoms=Atoms(positions=positions,cell=supercell)
        
        atomic_number = type_index 
        symbols = [mapping_atoms[j] for j in atomic_number]

        #atoms.set_atomic_numbers(type_index + 1)
        atoms.set_chemical_symbols(symbols)
        calc = SinglePointCalculator(atoms=atoms,forces=forces,energy=energy,stress=stress)
        atoms.calc = calc

        structures.append(atoms)

    return structures


def write_cfg(filename, atoms, mapping_atoms):

    mapping_atoms_r = dict()
    for i,j in mapping_atoms.items():
        mapping_atoms_r[j]=i

    fileobj=open(filename, "w")
    for iatoms in atoms:
        fileobj.write("BEGIN_CFG\n")
        fileobj.write(" Size\n")
        fileobj.write(" {:6d}\n".format(len(iatoms)))
        fileobj.write(" Supercell\n")
        for vec in iatoms.cell :
            fileobj.write("     {:12.6f}  {:12.6f}  {:12.6f}\n".format(*vec))
        fileobj.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz\n")

        symbols = iatoms.get_chemical_symbols()
        positions = iatoms.positions
        forces = iatoms.get_forces()

        for count, (sym,pos,force) in enumerate(zip(symbols, positions, forces),start=1):
            fileobj.write(  "      {:8d} {:4d}   {:12.6f}  {:12.6f}  {:12.6f}  {:11.6f} {:11.6f} {:11.6f}\n".format(count, mapping_atoms_r[sym], *pos, *force))

        fileobj.write(" Energy\n")
        fileobj.write("{:24.12f}\n".format(iatoms.get_potential_energy()))
        fileobj.write(" PlusStress:  xx          yy          zz          yz          xz          xy\n")
        fileobj.write("     {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f}\n".format(*iatoms.get_stress()))
        fileobj.write(" Feature   EFS_by     modASE\n")
        fileobj.write("END_CFG\n")
        fileobj.write("\n")


filein = "test.cfg"
mapping_atoms = {0:"C",1:"H",2:"N",3:"O",4:"Zn"}
structures = read_cfg(filein,mapping_atoms)
write_cfg("test_out.cfg",structures,mapping_atoms)
view(structures)

