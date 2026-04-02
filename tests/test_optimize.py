"""
Unit tests for optimize.py — search space engine and config mutation/restoration.
"""

import os
import sys
from contextlib import contextmanager

import optuna
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from optimize import suggest_params, restore_config


@contextmanager
def _search_space(space):
    """Temporarily override config.OPTUNA_SEARCH_SPACE, restore on exit."""
    saved = config.OPTUNA_SEARCH_SPACE
    config.OPTUNA_SEARCH_SPACE = space
    try:
        yield
    finally:
        config.OPTUNA_SEARCH_SPACE = saved


def _make_trial():
    """Create a fresh Optuna trial."""
    study = optuna.create_study()
    return study.ask()


class TestSuggestParams:
    """Test the config mutation/restoration mechanism."""

    def test_categorical_mutates_config(self):
        """Categorical spec mutates config and returns searched params."""
        original = config.D_MODEL
        with _search_space({'d_model': [16, 32, 64]}):
            trial = _make_trial()
            searched, saved = suggest_params(trial)

            assert 'D_MODEL' in saved
            assert saved['D_MODEL'] == original
            assert 'd_model' in searched
            assert searched['d_model'] in [16, 32, 64]
            assert config.D_MODEL == searched['d_model']

            restore_config(saved)
        assert config.D_MODEL == original

    def test_int_range_mutates_config(self):
        """Integer range spec mutates config correctly."""
        original = config.N_LAYERS
        with _search_space({'n_layers': (2, 10)}):
            trial = _make_trial()
            searched, saved = suggest_params(trial)

            assert 'n_layers' in searched
            assert 2 <= searched['n_layers'] <= 10
            assert config.N_LAYERS == searched['n_layers']

            restore_config(saved)
        assert config.N_LAYERS == original

    def test_float_range_mutates_config(self):
        """Float range spec mutates config correctly (log scale)."""
        original = config.LEARNING_RATE
        with _search_space({'learning_rate': (1e-5, 1e-2)}):
            trial = _make_trial()
            searched, saved = suggest_params(trial)

            assert 'learning_rate' in searched
            assert 1e-5 <= searched['learning_rate'] <= 1e-2
            assert config.LEARNING_RATE == searched['learning_rate']

            restore_config(saved)
        assert config.LEARNING_RATE == original

    def test_empty_search_space(self):
        """Empty search space returns empty dicts, config unchanged."""
        original_lr = config.LEARNING_RATE
        with _search_space({}):
            trial = _make_trial()
            searched, saved = suggest_params(trial)

            assert searched == {}
            assert saved == {}
        assert config.LEARNING_RATE == original_lr

    def test_unknown_param_raises(self):
        """Unknown param name raises ValueError."""
        with _search_space({'nonexistent_param_xyz': [1, 2, 3]}):
            trial = _make_trial()
            with pytest.raises(ValueError, match="Unknown param"):
                suggest_params(trial)

    def test_invalid_spec_raises(self):
        """Invalid spec format raises ValueError."""
        with _search_space({'d_model': "not_a_valid_spec"}):
            trial = _make_trial()
            with pytest.raises(ValueError, match="Invalid spec"):
                suggest_params(trial)

    def test_multi_param_search(self):
        """Multiple params searched simultaneously, all restored."""
        originals = {
            'D_MODEL': config.D_MODEL,
            'N_LAYERS': config.N_LAYERS,
            'DROPOUT': config.DROPOUT,
        }
        with _search_space({
            'd_model': [16, 32, 64],
            'n_layers': (2, 8),
            'dropout': [0.0, 0.1, 0.2],
        }):
            trial = _make_trial()
            searched, saved = suggest_params(trial)

            assert len(searched) == 3
            assert len(saved) == 3

            restore_config(saved)
        for attr, original in originals.items():
            assert getattr(config, attr) == original

    def test_restore_after_exception(self):
        """Config is restorable even if suggest_params partially fails."""
        original = config.D_MODEL
        # First valid param, then invalid — partial mutation
        with _search_space({'d_model': [16, 32], 'bad_param_xyz': [1]}):
            trial = _make_trial()
            try:
                searched, saved = suggest_params(trial)
            except ValueError:
                pass
        # D_MODEL may have been mutated before the error on bad_param_xyz
        # But saved_config was not returned — manual restore needed
        config.D_MODEL = original


class TestRestoreConfig:
    """Test restore_config edge cases."""

    def test_empty_dict(self):
        """Restoring empty dict is a no-op."""
        original_lr = config.LEARNING_RATE
        restore_config({})
        assert config.LEARNING_RATE == original_lr

    def test_restore_is_exact(self):
        """Restored value is identical to original."""
        original = config.LEARNING_RATE
        config.LEARNING_RATE = 0.999
        restore_config({'LEARNING_RATE': original})
        assert config.LEARNING_RATE == original


class TestOptimizeImports:
    """Verify optimize.py module integrity."""

    def test_import(self):
        """Module imports without error."""
        import optimize
        assert hasattr(optimize, 'objective')
        assert hasattr(optimize, 'suggest_params')
        assert hasattr(optimize, 'restore_config')
        assert hasattr(optimize, 'print_pareto_front')
        assert hasattr(optimize, 'save_results')
        assert hasattr(optimize, 'main')
