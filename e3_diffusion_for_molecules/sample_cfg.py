import argparse
import pickle
import torch
from os.path import join
import sys

# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import utils
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.models import get_model
from qm9.utils import compute_mean_mad
from qm9.sampling import sample_cfg_properties
import qm9.visualizer as vis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/cfg_multi_property", help='Specify model path')
    parser.add_argument('--target_alpha', type=float, default=75.0, help='Target polarizability')
    parser.add_argument('--target_mu', type=float, default=2.5, help='Target dipole moment')
    parser.add_argument('--target_gap', type=float, default=0.25, help='Target HOMO-LUMO gap')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='CFG guidance scale w')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of molecules to generate')
    parser.add_argument('--output_dir', type=str, default="eval/cfg_molecules/", help='Output directory')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # Missing arguments from older runs
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not eval_args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    print(f"Loading model from {eval_args.model_path}")
    print(f"Guidance scale: {eval_args.guidance_scale}")

    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    flow, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    else:
        print("Error: Model was not trained with conditioning! Cannot run CFG.")
        sys.exit(1)

    flow.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    flow.load_state_dict(flow_state_dict)
    flow.eval()

    target_properties = {
        'alpha': eval_args.target_alpha,
        'mu': eval_args.target_mu,
        'gap': eval_args.target_gap
    }
    
    print(f"Target properties: {target_properties}")
    print(f"Generating {eval_args.n_samples} molecules...")

    one_hot, charges, x, node_mask = sample_cfg_properties(
        args, device, flow, dataset_info, target_properties, property_norms, 
        guidance_scale=eval_args.guidance_scale, n_samples=eval_args.n_samples
    )

    out_path = join(eval_args.model_path, eval_args.output_dir)
    print(f"Saving to {out_path}")
    
    vis.save_xyz_file(
        out_path, one_hot, charges, x,
        id_from=0, name='cfg_molecule', dataset_info=dataset_info,
        node_mask=node_mask
    )

    print('Visualizing molecules.')
    vis.visualize(
        out_path, dataset_info,
        max_num=eval_args.n_samples, spheres_3d=False
    )

if __name__ == "__main__":
    main()
