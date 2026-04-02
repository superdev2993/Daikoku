"""
Evaluation Script - Entry point for model evaluation

Usage:
    python evaluate.py                              # Use config defaults
    python evaluate.py --checkpoint models/best_model.pt
    python evaluate.py --split all
    python evaluate.py --inference-only             # No metrics, just predictions
"""

import sys
import os
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.evaluation.evaluator import ModelEvaluator
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Daikoku - Model Evaluation"
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=config.EVAL_CHECKPOINT,
        help=f'Path to checkpoint file (default: {config.EVAL_CHECKPOINT})'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=config.EVAL_DATA_FILE,
        help=f'Path to data CSV file (default: {config.EVAL_DATA_FILE})'
    )

    parser.add_argument(
        '--split',
        type=str,
        default=config.EVAL_SPLIT,
        choices=['train', 'test', 'all'],
        help=f'Dataset split to evaluate (default: {config.EVAL_SPLIT})'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: auto-generated in evaluation/)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )

    parser.add_argument(
        '--inference-only',
        action='store_true',
        default=config.EVAL_INFERENCE_ONLY,
        help='Inference only mode (no labels, no metrics)'
    )

    return parser.parse_args()


def main():
    """
    Main evaluation pipeline.
    """
    # Parse arguments
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("DAIKOKU - MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Inference only: {args.inference_only}")
    logger.info("=" * 80)

    try:
        # Create evaluator
        evaluator = ModelEvaluator(
            checkpoint_path=args.checkpoint,
            device=args.device
        )

        # Run evaluation
        results = evaluator.evaluate(
            data_path=args.data,
            split=args.split,
            output_dir=args.output,
            inference_only=args.inference_only
        )

        # Copy checkpoint to output directory
        try:
            checkpoint_filename = os.path.basename(args.checkpoint)
            checkpoint_copy_path = os.path.join(results['output_dir'], checkpoint_filename)
            shutil.copy2(args.checkpoint, checkpoint_copy_path)
            logger.info(f"Checkpoint copied to: {checkpoint_copy_path}")
        except Exception as e:
            logger.warning(f"Failed to copy checkpoint: {e}")

        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output directory: {results['output_dir']}")
        logger.info(f"Total predictions: {len(results['predictions'])}")

        if not args.inference_only:
            logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            logger.info(f"Balanced Accuracy: {results['metrics']['balanced_accuracy']:.4f}")
            logger.info(f"F1 Macro: {results['metrics']['f1_macro']:.4f}")

        logger.info("=" * 80)

        # Generate interactive HTML report
        if config.EVAL_GENERATE_PLOTS:
            try:
                from modules.evaluation.visualizer import PredictionVisualizer
                from modules.data.loader import get_raw_data

                df_raw = get_raw_data(args.data)
                visualizer = PredictionVisualizer(
                    df_raw=df_raw,
                    predictions=results['predictions'],
                    labels=results['labels'],
                    probabilities=results['probabilities'],
                    timestamps=results['timestamps'],
                    metrics=results['metrics'],
                    config_params={
                        'raw_start_idx': results['split_info']['raw_start_idx'],
                        'EVAL_ACCURACY_WINDOW': config.EVAL_ACCURACY_WINDOW,
                        'ATR_PERIOD': evaluator.config_params['ATR_PERIOD'],
                        'ATR_MULTIPLIER_TP': evaluator.config_params['ATR_MULTIPLIER_TP'],
                        'ATR_MULTIPLIER_SL': evaluator.config_params['ATR_MULTIPLIER_SL'],
                        'MAX_HORIZON': evaluator.config_params['MAX_HORIZON'],
                        'PREDICTION_TARGET': evaluator.prediction_target,
                    },
                    num_classes=evaluator.num_classes,
                )
                report_path = os.path.join(results['output_dir'], 'interactive_report.html')
                visualizer.generate_html_report(report_path)
                logger.info(f"Interactive report: {report_path}")
            except Exception as e:
                logger.error(f"Failed to generate visualizations: {e}")

            # Generate input features visualization
            try:
                from modules.evaluation.visualize_interactive import (
                    prepare_data, build_json_data, generate_html
                )

                viz_data = prepare_data(args.data, checkpoint_params=evaluator.config_params, split=args.split)
                viz_json = build_json_data(viz_data)
                input_features_path = os.path.join(
                    results['output_dir'], 'input_features.html'
                )
                generate_html(viz_json, input_features_path)
                logger.info(f"Input features report: {input_features_path}")
            except Exception as e:
                logger.error(f"Failed to generate input features visualization: {e}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
