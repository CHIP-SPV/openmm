from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

pdb = PDBFile('protein.pdb')
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology,
        nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('output.pdb', 2))
simulation.reporters.append(StateDataReporter(stdout, 2, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(10)
