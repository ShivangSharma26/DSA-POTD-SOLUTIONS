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
class MoleculeGNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple Graph Network that respects physics
        # Input: Atom Type (1 dim) -> Hidden (64)
        self.embedding = nn.Linear(1, 64)
        
        # Message Passing Layers (Thinking about distance)
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 64)
        
        # Output: Energy (Scalar)
        self.output = nn.Linear(64, 1)

    def forward(self, g, pos):
        # 1. Embed atomic numbers
        h = F.relu(self.embedding(g.ndata['z']))   # (N, 64)
        g.ndata['h'] = h
    
        # 2. Compute distances
        src, dst = g.edges()
        d = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)  # (E, 1)
        g.edata['d'] = d.repeat(1, 64)  # (E, 64)
    
        # 3. Message passing (FIXED)
        g.update_all(
            dgl.function.u_mul_e('h', 'd', 'm'),
            dgl.function.sum('m', 'h_new')
        )
    
        # 4. Node update
        h = h + F.relu(self.layer1(g.ndata['h_new']))
        h = F.relu(self.layer2(h))
    
        # 5. Energy output
        g.ndata['h_final'] = self.output(h)
        return dgl.sum_nodes(g, 'h_final')


model = MoleculeGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
print("✅ Model Initialized")
print("Starting Training...")
model.train()

loss_history = []

batched_graph = dgl.batch(train_graphs).to(device)
batched_labels = train_labels.to(device)

for epoch in range(200):
    optimizer.zero_grad()

    pos = batched_graph.ndata['pos']      # (N, 3)
    pred_energy = model(batched_graph, pos)

    loss = F.mse_loss(pred_energy, batched_labels)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

print("Training complete")
# --- Setup Simulation ---
print("🎬 Starting Final Simulation...")

# 1. Create a FRESH distorted methane (Unknown to model)
test_mol = Chem.AddHs(Chem.MolFromSmiles('C'))
AllChem.EmbedMolecule(test_mol, AllChem.ETKDG())
clean_pos = test_mol.GetConformer().GetPositions()

# Add lots of noise (Make it bad)
start_pos = clean_pos + np.random.normal(0, 0.4, clean_pos.shape) 

# Prepare Tensors
pos_tensor = torch.tensor(start_pos, dtype=torch.float32, device=device, requires_grad=True)
z_tensor = torch.tensor([a.GetAtomicNum() for a in test_mol.GetAtoms()], dtype=torch.float32, device=device).unsqueeze(-1)

# Graph Structure (Fixed topology, positions change)
g_sim = dgl.knn_graph(pos_tensor.detach(), k=4).to(device)
g_sim.ndata['z'] = z_tensor

# Optimizer for Positions (Simulation)
sim_optimizer = torch.optim.Adam([pos_tensor], lr=0.01)

history = []

# --- Minimization Loop ---
for step in range(100):
    sim_optimizer.zero_grad()
    
    # Predict Energy using TRAINED model
    # Note: We pass the CURRENT pos_tensor which has gradients
    energy = model(g_sim, pos_tensor)
    
    # Calculate Force (Gradient of Energy)
    energy.backward()
    
    # Move Atoms
    sim_optimizer.step()
    
    # Log
    # Calculate deviation from perfect tetrahedral shape (Bond length variance)
    # Carbon is at index 0
    bonds = torch.norm(pos_tensor[1:] - pos_tensor[0], dim=1)
    avg_bond = bonds.mean().item()
    
    history.append(energy.item())
    
    if step % 20 == 0:
        print(f"Step {step}: Energy = {energy.item():.2f}, Bond Len = {avg_bond:.2f}")

# --- Plotting ---
plt.figure(figsize=(8,4))
plt.plot(history, 'g-', linewidth=2)
plt.title("Energy Minimization using Trained Neural Network")
plt.xlabel("Simulation Step")
plt.ylabel("Predicted Energy")
plt.grid(True, alpha=0.3)
plt.show()

print("🎉 Simulation Finished. The model successfully relaxed the molecule!")
print("🎥 Re-running Simulation to capture video frames...")

# Wapas distorted molecule banate hain
test_mol = Chem.AddHs(Chem.MolFromSmiles('C'))
AllChem.EmbedMolecule(test_mol, AllChem.ETKDG())
clean_pos = test_mol.GetConformer().GetPositions()

# Noise add karte hain
start_pos = clean_pos + np.random.normal(0, 0.4, clean_pos.shape) 

# Tensors setup
pos_tensor = torch.tensor(start_pos, dtype=torch.float32, device=device, requires_grad=True)
z_indices = [a.GetAtomicNum() for a in test_mol.GetAtoms()] # [6, 1, 1, 1, 1]

# Graph setup
g_sim = dgl.knn_graph(pos_tensor.detach(), k=4).to(device)
z_tensor = torch.tensor(z_indices, dtype=torch.float32, device=device).unsqueeze(-1)
g_sim.ndata['z'] = z_tensor

sim_optimizer = torch.optim.Adam([pos_tensor], lr=0.01)

# Frame store karne ke liye list
frames = []

for step in range(100):
    sim_optimizer.zero_grad()
    
    # Model se Energy pucho
    energy = model(g_sim, pos_tensor)
    energy.backward()
    sim_optimizer.step()
    
    # Har step ki position copy karke save karo
    current_pos = pos_tensor.detach().cpu().numpy().copy()
    frames.append(current_pos)

print(f"✅ Captured {len(frames)} frames. Ready to animate!")