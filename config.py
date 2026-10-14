import torch

DATA_DIR = '/kaggle/input/brain-tumor-mri-dataset'
OUTPUT_DIR = './outputs'
MODEL_PATH = ''

CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

FEATURE_DIM = 1280
NUM_HEADS = 8
ATTENTION_DROPOUT = 0.1
HEAD_DROPOUT = 0.4
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-5

PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 15
PHASE3_EPOCHS = 10

PHASE1_LR = 1e-3
PHASE2_LR_BACKBONE = 5e-5
PHASE2_LR_HEAD = 5e-4
PHASE3_LR_BACKBONE = 1e-5
PHASE3_LR_HEAD = 2e-4

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')