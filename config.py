# ============================================
# FEATURES (18 or 23 total = 12 global + 6 per-window + 5 VP if enabled)
# ============================================
# Global (transform_pipeline → 12 features):
#   Normalized: [0] log_price [1] log_open [2] log_wick_high [3] log_wick_low
#               [4] log_volume [5] log_atr
#   Time:       [6] sin_hour [7] cos_hour [8] sin_weekday [9] cos_weekday
#   Regime:     [10] regime_trend [11] regime_voltrend
#
# Per-window (Dataset.__getitem__ → 6 features, prevents zigzag look-ahead):
#   Normalized: log_zz (inserted at col 5 during assembly)
#   Structural: struct_rank, dist_res_1/2, dist_sup_1/2
#
# Final order after assembly (VP off):  [5 OHLCV | log_zz | log_atr | 4 time | 5 struct | 2 regime] = 18
# Final order after assembly (VP on):   [5 OHLCV | log_zz | log_atr | 4 time | 5 struct | 5 VP | 2 regime] = 23

# ============================================
# DATA PARAMETERS
# ============================================

INPUT_FILES = [       # Multiple files supported
    "data/BTC_1h_agg.csv",
    #"data/ETH_1h_bin.csv",
]

PREDICTION_TARGET = "triple"  # "triple" (3-class) / "bull" / "bear" / "uncertain" / "close1" / "close2" / "close3" (binary next-close)

WINDOW_SIZE = 100       # Number of candles per input window
TRAIN_TEST_SPLIT = 0.9  # Chronological split ratio (no shuffle)
NORMALIZE = True        # Apply per-window median/IQR normalization to first N features

# ============================================
# TRIPLE BARRIER LABELING
# ============================================
ATR_PERIOD = 24        # ATR smoothing period (Wilder)
ATR_MULTIPLIER_TP = 2  # Take profit distance = ATR_MULTIPLIER_TP × ATR
ATR_MULTIPLIER_SL = 1  # Stop loss distance = ATR_MULTIPLIER_SL × ATR (R:R = TP/SL)
MAX_HORIZON = 12       # Maximum candles to wait

# ============================================
# MODEL ARCHITECTURE
# ============================================
D_MODEL = 48            # Hidden dimension (embedding size)

N_LAYERS = 5            # Mamba blocks in primary branch
MULTI_TF_N_LAYERS = 4   # Mamba blocks in secondary branch

D_STATE = 32            # SSM state dimension (primary)
MULTI_TF_D_STATE = 24   # SSM state dimension (secondary)

D_CONV = 4              # Causal convolution kernel size in Mamba
EXPAND = 2              # Inner dimension expansion factor (d_inner = d_model * expand)

MAMBA_LAYERNORM = True # required: LayerNorm in MambaBlocks (pre-norm + final_norm). False = identity.
CNN_LAYERNORM = False  # LayerNorm after CNN projection. False = identity.

# ============================================
# MULTI-TIMEFRAME
# d_model, d_conv, expand, dropout : shared with main branch
# ============================================
MULTI_TF_MODE = "dual"              # "off" / "calibration" / "dual"
MULTI_TF_DIVISOR = 4                # aggregation factor (primary TF × divisor = secondary TF)
MULTI_TF_ALIGN = "unstructured"     # "structured" (hourly boundaries) / "unstructured" (consecutive blocks)
CROSS_TF_NORMALIZE = True           # Normalize secondary branch using primary branch's median/IQR stats


# ============================================
# CNN INPUT PROJECTION
# ============================================
CNN_ENABLED = True              # Enable CNN branch in input projection (False = linear only)
CNN_KERNEL_SIZE = 10            # Number of candles the CNN looks at (3, 5, 7...)
CNN_FUSION = "gated_residual"   # "add" (simple sum) / "gated_residual" (learned gate filters CNN)


# ============================================
# ATTENTION & GATE
# ============================================
ATTENTION_POSITION = "aligned"    # "off" / "pre_gate" (intra-TF SelfAttn) / "post_gate" (cross-TF attention) / "aligned" (cross-TF temporal alignment)
                                  # NOTE: "aligned" uses AlignedFusion (gated residual) to inject secondary context
                                  # into primary at token level, then a single-path MLP head (no Gate, no UnifiedHead).
                                  # Secondary info is already fused into the enriched primary — a separate secondary
                                  # path in the head would be redundant and leads to gate collapse.
ATTENTION_NUM_HEADS = 4           # Number of attention heads (must divide D_MODEL)
GATE_VERSION = "v3"               # "v2" (scalar gate) / "v3" (MLP per-feature) — ignored when ATTENTION_POSITION="aligned"
GATE_BOTTLENECK = 64              # Bottleneck size for V3 gate MLP — ignored when ATTENTION_POSITION="aligned"


# ============================================
# TRAINING
# ============================================

EPOCHS = 50

DEVICE = "auto"             # auto, cuda, cpu (very slower)
DROPOUT = 0.1               # Dropout rate in Mamba blocks
DROPOUT_LAST_N_LAYERS = 8   # Apply dropout only to last N layers (0 = all layers)
CNN_DROPOUT = 0.1
DROPOUT_HEAD = 0.1
BATCH_SIZE = 128
SHUFFLE_TRAIN = True        # Shuffle training windows (not test) for better generalization
SHUFFLE_CHUNKS = 8          # 1 = full shuffle (default), N>1 = split train into N chrono blocks, shuffle within each

# CASCADE TRAINING (progressive layer growing + freezing)
CASCADE_TRAINING = False  # True = grow 1 layer/epoch + freeze, False = standard training

WEIGHT_DECAY = 0.03         # AdamW L2 regularization
SEED = 42

# LR SCHEDULING
LEARNING_RATE = 0.00005       #default 0.00005
LR_SCHEDULER = "cosine"       # "none" / "cosine" / "wsd"
LR_MIN = 0.000005             # default 0.000005 End LR (cosine: target, wsd: end of decay)
LR_WARMUP_PCT = 0.4           # WSD only: % of epochs for warmup (0 → LR)
LR_STABLE_PCT = 0.2           # WSD only: % of epochs at constant LR


# ============================================
# VOLUME PROFILE (sliding VP features)
# ============================================
VP_LOOKBACK = 100      # Number of candles for VP calculation
VP_BINS = 50           # Number of horizontal price levels
VP_VA_PCT = 0.70       # Value Area percentage (70% standard)

# ============================================
# ZIGZAG STRUCTURAL FEATURES
# ============================================
ZIGZAG_LENGTH = 3  # Fractal radius (pivot = extremum in 2*length+1 window)

# ============================================
# REGIME FILTER (HMA-based market regime)
# ============================================
REGIME_LENGTH = 20  # Single param: HMA period + sign accumulation window


# ============================================
# LABEL CONFIDENCE WEIGHTING
# ============================================
# Weight loss per sample by Triple Barrier hit time confidence.
# Fast barrier hits (early) → high confidence, slow hits → low confidence.
# Uncertain labels get a fixed confidence value.
LABEL_CONFIDENCE_ENABLED = False
LABEL_CONFIDENCE_CURVE = "sqrt"   # "linear" (1-t/H) or "sqrt" (sqrt(1-t/H)) — sqrt is gentler

# ============================================
# UNIFIED LOSS (Focal + Class Weighting)
# ============================================

# Focal: down-weights easy examples (high confidence predictions)
# gamma=0 → standard CE, gamma=2 → strong focal effect
FOCAL_GAMMA = 2

# Class weighting: compensates class imbalance (computed dynamically from label distribution)
# alpha=0 → no weighting, alpha=0.5 → sqrt inverse freq (soft), alpha=1.0 → full inverse freq
CLASS_WEIGHT_ALPHA = 0.5


# ============================================
# PERFORMANCE (GPU optimization)
# ============================================
MIXED_PRECISION = True         # AMP float16 on forward pass
CUDNN_BENCHMARK = True         # Auto-tune convolution algorithms 
NUM_WORKERS = "auto"           # "auto" (adapts to CPU cores and device) or int (0 = main thread only)
PERSISTENT_WORKERS = True      # Keep workers alive between epochs
NON_BLOCKING_TRANSFER = True   # Overlap CPU/GPU data transfers

# ============================================
# CHECKPOINTS
# ============================================
CHECKPOINT_DIR = "models/"
SAVE_EVERY_N_EPOCHS = 1
KEEP_LATEST = True             # Always save latest.pt (in addition to best_model.pt)
SAVE_ON_INTERRUPT = True

# ============================================
# LOGGING
# ============================================
LOG_DIR = "logs/"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
TENSORBOARD_DIR = "runs/"

# ============================================
# ACTIVATION MONITORING
# ============================================
MONITOR_ACTIVATIONS = True
MONITOR_EVERY_N_EPOCHS = 1
MONITOR_DEAD_THRESHOLD = 1e-6
MONITOR_DEAD_ALERT_PCT = 30.0
MONITOR_GRAD_VANISH_THRESHOLD = 1e-7
MONITOR_GRAD_EXPLODE_THRESHOLD = 1000.0

# ============================================
# MULTI-TF BRANCH DIAGNOSTICS
# ============================================
MONITOR_BRANCH_BALANCE = True  # Enable primary/secondary branch diagnostics (first & last epoch)


# ============================================
# DATA AUGMENTATION
# ============================================
AUGMENT_VARIANTS = 4    # Number of augmented datasets per original
AUGMENT_PCT = 50        # Base %: open/close = OU(max(PCT% body, 0.1% price)), wicks/volume = ±PCT%

# ============================================
# EVALUATION
# ============================================
EVAL_CHECKPOINT = "models/latest.pt"          # Checkpoint to evaluate
EVAL_DATA_FILE = "data/BTC_1h_3y.csv"        # Dataset (can differ from training data)
EVAL_SPLIT = "test"                           # "train", "test", "all"
EVAL_BATCH_SIZE = 512                         # Batch size for evaluation inference
EVAL_GENERATE_PLOTS = True                    # Generate visualizations
EVAL_ACCURACY_WINDOW = 100                    # Window size for accuracy timeline
EVAL_INFERENCE_ONLY = False                   # Inference-only mode (no labels, no metrics)
EVAL_OUTPUT_DIR = "evaluation/"               # Output directory for evaluation results
EVAL_MIXED_PRECISION = False                  # True = fp16, False = fp32 (full precision)

# ============================================
# HTTP PROXY (ccxt: download_candles.py, live feed)
# Host/port/credentials: project-root .env or shell exports
#   cp .env.example .env   # then edit .env
# Optional in .env or shell: PROXY_ENABLED, PROXY_TYPE
# Requires: pip install PySocks
# ============================================
PROXY_ENABLED = True
PROXY_TYPE = "socks5"

# ============================================
# LIVE INFERENCE
# ============================================
LIVE_EXCHANGE = "binance"      # "binance" / "bybit" / "bitget"
LIVE_SYMBOL = "BTC/USDT"
LIVE_TIMEFRAME = "1h"          # Supported: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
LIVE_BUFFER_SIZE = 2000        # Raw candles to maintain (min ~1000 for dual TF)
LIVE_CHECKPOINT = "models/latest.pt"
LIVE_HTTP_PORT = 7777
LIVE_RETRY_TIMEOUT = 10        # Alert if candle fetch > N seconds
LIVE_RETRY_MAX = 6             # Max consecutive fetch retries before alert
LIVE_REFRESH_INTERVAL = 10     # Seconds between open candle refreshes (display + TP/SL check)
LIVE_COMPARE_EVERY = 1         # Run evaluate-style verification every N closed candles
LIVE_OUTPUT_DIR = "live/"
LIVE_MIXED_PRECISION = False                       # True = fp16 (like training), False = fp32 (full precision)


# ============================================
# OPTUNA HYPERPARAMETER SEARCH
# ============================================
# Only params listed here are searched by Optuna.
# Everything else uses config.py defaults as-is.
#
# Formats:
#   [v1, v2, ...]    → suggest_categorical
#   (min, max)        → suggest_int (if both int) or suggest_float with log scale (if float)
#
# Key = lowercase version of the config param (e.g. 'd_model' → config.D_MODEL)
#
# Any parameter in config.py can be searched: just add its lowercase name as key.
# The engine does config.{KEY.upper()} = suggested_value at each trial.
#
# Common choices:
#   Model:    d_model, n_layers, d_state, d_conv, expand
#   Training: learning_rate, batch_size, weight_decay, dropout, dropout_last_n_layers
#   Data:     window_size, atr_period, atr_multiplier_tp, atr_multiplier_sl, max_horizon
#   Loss:     focal_gamma, class_weight_alpha
#   CNN:      cnn_kernel_size
#   Attn:     attention_position, gate_version, gate_bottleneck
#   MTF:      multi_tf_divisor, multi_tf_n_layers, multi_tf_d_state
#   Label:    label_confidence_enabled, label_confidence_curve

OPTUNA_SEARCH_SPACE = {
    'learning_rate': (0.00003, 0.001),
    'weight_decay': [0.01, 0.03, 0.05, 0.1],
    'd_model': [32, 48, 64],
    'n_layers': (3, 8),
    'dropout': [0.05, 0.1, 0.15, 0.2],
    'window_size': [80, 100, 120, 150],
}
