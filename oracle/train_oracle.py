import sys
sys.path.append('e3_diffusion_for_molecules')
import torch
import argparse
import torch.nn.functional as F
import torch.optim as optim
from qm9.dataset import retrieve_dataloaders
from configs.datasets_config import get_dataset_info
from oracle.oracle_model import PropertyOracle
from qm9.utils import compute_mean_mad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--properties', nargs='+', default=['alpha', 'mu', 'gap'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--datadir', type=str, default='e3_diffusion_for_molecules/qm9/temp')
    parser.add_argument('--dataset', type=str, default='qm9')
    parser.add_argument('--remove_h', action='store_true')
    parser.add_argument('--include_charges', type=bool, default=True)
    parser.add_argument('--filter_n_atoms', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataloaders, charge_scale = retrieve_dataloaders(args)
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    model = PropertyOracle(in_node_nf=in_node_nf, n_properties=len(args.properties), device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    property_norms = compute_mean_mad(dataloaders, args.properties, args.dataset)
    
    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0
        for i, data in enumerate(dataloaders['train']):
            x = data['positions'].to(device)
            node_mask = data['atom_mask'].to(device).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device)
            one_hot = data['one_hot'].to(device)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device)
            
            h = torch.cat([one_hot, charges], dim=2)
            
            target_list = []
            for prop in args.properties:
                val = (data[prop] - property_norms[prop]['mean']) / property_norms[prop]['mad']
                target_list.append(val.unsqueeze(1))
            targets = torch.cat(target_list, dim=1).to(device)
            
            optimizer.zero_grad()
            preds = model(x, h, node_mask, edge_mask)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
        print(f"Epoch {epoch} finished. Average Train Loss: {train_loss / len(dataloaders['train']):.4f}")
        
        torch.save(model.state_dict(), 'oracle/oracle_model.pth')

if __name__ == '__main__':
    main()
