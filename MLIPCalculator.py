import re
import os 
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes, FileIOCalculator
from ase.io import write ,read
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build.tools import sort
from ase.optimize import BFGS, QuasiNewton


class MLIPCalculator(FileIOCalculator):

    implemented_properties = ["energy","forces","stress"]
    
    def __init__(self, 
                 model, 
                 mapping_atoms, 
                 sourcepath,
                 incfg="_in.cfg",
                 outcfg="_out.cfg",
                 waterfield=None):

        self.model = model
    
        FileIOCalculator.__init__(self)

        self.sourcepath = sourcepath
        self.mapping_atoms = mapping_atoms
        self.incfg = incfg
        self.outcfg = outcfg
        self.waterfield=waterfield

        if self.waterfield is not None :
            # self.atoms_water = self.waterfield
            water_nat=self.waterfield.get_global_number_of_atoms()
            self.waterfield.set_tags([10 for i in range(water_nat)])

        self.mapping_atoms_r = dict()
        for i,j in self.mapping_atoms.items():
            self.mapping_atoms_r[j]=i

        self.command = "{} calc-efs {} {} {}".format(
            self.sourcepath, self.model, self.incfg, self.outcfg)


    def read_results(self):

        with open(self.outcfg) as f:
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

            if self.waterfield is not None :
 
                # print("total atoms in the simulation is : ", len(self.frame_index)+len(self.water_index))
                # print("total atoms printed out          : ", len(self.frame_index))
                positions = positions[self.frame_index]
                energy = energy
                forces = forces[self.frame_index]
                type_index = type_index[self.frame_index]
                
            atoms=Atoms(positions=positions,cell=supercell)

            atomic_number = type_index
            symbols = [self.mapping_atoms[j] for j in atomic_number]
            atoms.set_chemical_symbols(symbols)
            calc = SinglePointCalculator(atoms=atoms,forces=forces,energy=energy,stress=stress)
            atoms.calc = calc

            structures.append(atoms)
            self.results = atoms.calc.results

        return structures


    def write_input(self, atoms, properties=["energy","forces","stress"], system_changes=None):

        self.number_of_atoms_in_frame=len(atoms)
        
        if self.waterfield is not None :

            atoms.set_tags([1 for i in range(self.number_of_atoms_in_frame)])
            atoms += self.waterfield

            self.frame_index = [atom.index for atom in atoms if atom.tag ==1]
            self.water_index = [atom.index for atom in atoms if atom.tag ==10]

        fileobj=open(self.incfg, "w")
        
        fileobj.write("BEGIN_CFG\n")
        fileobj.write(" Size\n")
        fileobj.write(" {:6d}\n".format(len(atoms)))
        fileobj.write(" Supercell\n"
                      )
        for vec in atoms.cell :
            fileobj.write("     {:12.6f}  {:12.6f}  {:12.6f}\n".format(*vec))
        
        fileobj.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz\n")

        symbols = atoms.get_chemical_symbols()
        positions = atoms.positions
        
        try:
            forces = atoms.get_forces()
        except:
            forces = np.zeros_like(positions)
        
        try:
            stress = atoms.get_stress()
        except:
            stress = np.array([0,0,0,0,0,0])

        try:
            energy = atoms.get_potential_energy()
        except:
            energy = 0

        for count, (sym,pos,force) in enumerate(zip(symbols, positions, forces),start=1):
            fileobj.write(  "      {:8d} {:4d}   {:12.6f}  {:12.6f}  {:12.6f}  {:11.6f} {:11.6f} {:11.6f}\n".format(count, 
                                                                                self.mapping_atoms_r[sym], *pos, *force))

        fileobj.write(" Energy\n")
        fileobj.write("{:24.12f}\n".format(energy))
        fileobj.write(" PlusStress:  xx          yy          zz          yz          xz          xy\n")
        fileobj.write("     {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f}\n".format(*stress))
        fileobj.write(" Feature   EFS_by     ASE\n")
        fileobj.write("END_CFG\n")
        fileobj.write("\n")





