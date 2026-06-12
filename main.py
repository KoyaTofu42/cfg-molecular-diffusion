import os
import sys

def main():
    """
    Entry point for running the diffusion model from either Docker or a Jupyter Notebook.
    """
    # Detect if we are running inside an IPython/Jupyter notebook
    is_notebook = 'ipykernel' in sys.modules

    default_args = [
        'main.py',
        '--n_epochs', '3000',
        '--exp_name', 'cfg_multi_property',
        '--conditioning', 'alpha', 'mu', 'gap',
        '--cfg_dropout_prob', '0.15',
        '--cfg_dropout_mode', 'joint',
        '--n_stability_samples', '100',
        '--diffusion_noise_schedule', 'polynomial_2',
        '--diffusion_noise_precision', '1e-5',
        '--diffusion_steps', '500',
        '--diffusion_loss_type', 'l2',
        '--batch_size', '64',
        '--num_workers', '2',
        '--nf', '128',
        '--n_layers', '6',
        '--lr', '2e-4',
        '--normalize_factors', '[1,4,1]',
        '--test_epochs', '20',
        '--ema_decay', '0.9999',
        '--save_model', 'True',
        
        # Uncomment and replace if you want to log to W&B
        # '--wandb_usr', 'YOUR_WANDB_USERNAME',
        
        # Uncomment and set path if you are resuming a run
        # '--resume', 'path/to/checkpoint_dir'
    ]

    if is_notebook:
        print("Detected Jupyter Notebook environment. Overriding kernel arguments.")
        sys.argv = default_args
    else:
        user_args = sys.argv[1:]
        if len(user_args) == 0:
            print("No command-line arguments provided. Using defaults.")
        else:
            print(f"Command-line arguments detected. Overriding defaults with: {user_args}")
        sys.argv = default_args + user_args
    
    # Get the directory of this script (project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, 'e3_diffusion_for_molecules')
    
    # Change the working directory so relative paths (like 'outputs/' or 'qm9/temp') 
    # resolve correctly just like they did originally
    if os.path.abspath(os.getcwd()) != os.path.abspath(target_dir):
        os.chdir(target_dir)
        
    # Add the target directory to sys.path so 'import main_qm9' works
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
        
    print(f"Running main_qm9.py with arguments: {sys.argv[1:]}")
    
    # Importing main_qm9 triggers its module-level parser.parse_args()
    import main_qm9
    
    # Run the main loop
    main_qm9.main()

if __name__ == '__main__':
    main()
