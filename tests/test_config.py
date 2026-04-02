"""
Test config.py - Validation of configuration parameters
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def test_data_parameters():
    """Test DATA parameters exist and have correct types"""
    assert isinstance(config.INPUT_FILES, list)
    assert len(config.INPUT_FILES) > 0
    for f in config.INPUT_FILES:
        assert isinstance(f, str)
    assert isinstance(config.WINDOW_SIZE, int)
    assert isinstance(config.TRAIN_TEST_SPLIT, float)

    assert config.WINDOW_SIZE > 0
    assert 0.0 < config.TRAIN_TEST_SPLIT < 1.0
    print("✓ DATA parameters OK")


def test_normalization_parameters():
    """Test NORMALIZATION parameters"""
    assert isinstance(config.NORMALIZE, bool)
    print("✓ NORMALIZATION parameters OK")


def test_labeling_parameters():
    """Test TRIPLE BARRIER parameters"""
    assert isinstance(config.ATR_PERIOD, int)
    assert isinstance(config.ATR_MULTIPLIER_TP, (int, float))
    assert isinstance(config.ATR_MULTIPLIER_SL, (int, float))
    assert isinstance(config.MAX_HORIZON, int)

    assert config.ATR_PERIOD > 0
    assert config.ATR_MULTIPLIER_TP > 0
    assert config.ATR_MULTIPLIER_SL > 0
    assert config.ATR_MULTIPLIER_TP >= config.ATR_MULTIPLIER_SL
    assert config.MAX_HORIZON > 0
    print("✓ TRIPLE BARRIER parameters OK")


def test_model_parameters():
    """Test MODEL parameters"""
    assert isinstance(config.D_MODEL, int)
    assert isinstance(config.D_STATE, int)
    assert isinstance(config.D_CONV, int)
    assert isinstance(config.EXPAND, int)
    assert isinstance(config.N_LAYERS, int)

    assert config.D_MODEL > 0
    assert config.D_STATE > 0
    assert config.D_CONV > 0
    assert config.EXPAND >= 1
    assert config.N_LAYERS >= 1

    # Attention & Gate
    assert config.ATTENTION_POSITION in ("off", "pre_gate", "post_gate", "aligned")
    assert isinstance(config.ATTENTION_NUM_HEADS, int)
    assert config.ATTENTION_NUM_HEADS > 0
    assert config.GATE_VERSION in ("v2", "v3")
    print("✓ MODEL parameters OK")


def test_training_parameters():
    """Test TRAINING parameters"""
    assert isinstance(config.DEVICE, str)
    assert isinstance(config.DROPOUT, (int, float))
    assert isinstance(config.LEARNING_RATE, float)
    assert isinstance(config.BATCH_SIZE, int)
    assert isinstance(config.EPOCHS, int)
    assert isinstance(config.WEIGHT_DECAY, (int, float))
    assert isinstance(config.SEED, int)

    assert config.DEVICE in ("auto", "cuda", "cpu")
    assert 0.0 <= config.DROPOUT < 1.0
    assert config.LEARNING_RATE > 0
    assert config.BATCH_SIZE > 0
    assert config.EPOCHS > 0
    assert config.WEIGHT_DECAY >= 0
    print("✓ TRAINING parameters OK")


def test_checkpoint_parameters():
    """Test CHECKPOINT parameters"""
    assert isinstance(config.CHECKPOINT_DIR, str)
    assert isinstance(config.SAVE_EVERY_N_EPOCHS, int)
    assert isinstance(config.KEEP_LATEST, bool)
    assert isinstance(config.SAVE_ON_INTERRUPT, bool)

    assert config.CHECKPOINT_DIR == "models/"
    assert config.SAVE_EVERY_N_EPOCHS == 1
    assert config.KEEP_LATEST is True
    assert config.SAVE_ON_INTERRUPT is True
    print("✓ CHECKPOINT parameters OK")


def test_logging_parameters():
    """Test LOGGING parameters"""
    assert isinstance(config.LOG_DIR, str)
    assert isinstance(config.LOG_LEVEL, str)
    assert isinstance(config.TENSORBOARD_DIR, str)

    assert config.LOG_DIR == "logs/"
    assert config.LOG_LEVEL == "INFO"
    assert config.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert config.TENSORBOARD_DIR == "runs/"
    print("✓ LOGGING parameters OK")


if __name__ == "__main__":
    print("\n=== Testing config.py ===\n")

    test_data_parameters()
    test_normalization_parameters()
    test_labeling_parameters()
    test_model_parameters()
    test_training_parameters()
    test_checkpoint_parameters()
    test_logging_parameters()

    print("\n✅ All config tests passed!\n")
