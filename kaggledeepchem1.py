!pip install torch==2.1.0 torchdata==0.7.1
!pip install dgl -f "https://data.dgl.ai/wheels/cu121/repo.html"
import torch
import dgl

print("Torch:", torch.__version__)
print("DGL:", dgl.__version__)
print("CUDA available:", torch.cuda.is_available())
!pip install rdkit-pypi
!pip install --pre deepchem
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
import dgl
import matplotlib.pyplot as plt
import networkx as nx

print(f"✅ DeepChem: {dc.__version__}")
print(f"✅ DGL: {dgl.__version__}")
print(f"✅ PyTorch: {torch.__version__}")

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {device}")
def get_methane_data(n_samples=500):
    # 1. Create Base Methane
    mol = Chem.MolFromSmiles('C')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    # Force Field for Ground Truth Energy (The "Teacher")
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    
    graphs = []
    energies = []
    
    # Store clean position for reference
    clean_conf = mol.GetConformer().GetPositions()
    
    print(f"⚗️ Generating {n_samples} training samples (Methane Perturbations)...")
    
    for _ in range(n_samples):
        # 2. Add noise to positions (Distort the molecule)
        noise = np.random.normal(0, 0.2, clean_conf.shape) # 0.2 Angstrom noise
        distorted_pos = clean_conf + noise
        
        # 3. Calculate "Real" Energy of this distorted shape
        # Update RDKit mol with new positions
        for i in range(mol.GetNumAtoms()):
            mol.GetConformer().SetAtomPosition(i, distorted_pos[i])
        
        # Get Energy (kcal/mol)
        if ff.CalcEnergy() < 1000: # Filter extreme garbage
            energies.append(ff.CalcEnergy())
            
            # 4. Convert to Graph for DeepChem/DGL
            # Create graph: 5 nodes (atoms), fully connected
            g = dgl.knn_graph(torch.tensor(distorted_pos, dtype=torch.float32), k=4) # k=4 means connect to all others
            
            # Add Node Features (Atomic Numbers: C=6, H=1)
            z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            g.ndata['z'] = torch.tensor(z, dtype=torch.float32).unsqueeze(-1)
            g.ndata['pos'] = torch.tensor(distorted_pos, dtype=torch.float32)
            
            graphs.append(g)

    return graphs, torch.tensor(energies, dtype=torch.float32).unsqueeze(-1)

# Generate Data
train_graphs, train_labels = get_methane_data()
print(f"✅ Generated {len(train_graphs)} valid samples.")
print(f"   Avg Energy: {train_labels.mean().item():.2f}")