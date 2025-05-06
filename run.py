import argparse
import yaml
from pathlib import Path
import datetime
from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots, get_prediction_timesteps, get_metadata
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from sindy import SINDy

def main(config_path):
    """
    Main function to run the SINDy model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    Args:
        config_path (str): Path to the configuration file.
    """
        
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract dataset name and parse pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    # Define model name
    model_name = f"{config['model']['name']}"

    # Generate a unique batch_id for this run
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize visualization object
    viz = Visualization()
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:

        # Load sub-dataset
        train_data, init_data = load_dataset(dataset_name, pair_id)
        
        # Load metadata
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
        delta_t = get_metadata(dataset_name)['delta_t']
        
        # Initialize model
        model = SINDy(pair_id, config, train_data, delta_t, init_data, prediction_timesteps)

        # Generate predictions
        predictions = model.predict()

        # Evaluate predictions
        results = evaluate(dataset_name, pair_id, predictions)
        
        # Save results and get directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, predictions, results)

        # Append metrics to batch results
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations
        for plot_type in applicable_plots:
            try:
                fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
                viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type)
            except:
                print(f"Error generating plot for pair_id {pair_id}")

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)