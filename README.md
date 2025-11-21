# GLR-SSM-EAD-A-Memory-Mechanism-for-Explainable-Medical-Diagnosis

📋 Table of Contents

Overview
Repository Structure
Installation
Quick Start
Code Architecture
Key Components
Usage Examples
Configuration
Output Files
Citation


📖 Overview
This repository contains the complete implementation of GLR-SSM-EAD (Gated Low-Rank State Space Model with Evidence-Adaptive Decay), a novel memory mechanism for interpretable medical diagnosis from clinical narratives.
Key Innovations

Evidence-Adaptive Decay: Per-token evidence scores (Imaging, Symptoms, Risk Factors) modulate channel-specific decay parameters
Low-Rank Input Coupling: Efficient B(x) = U(V(x)) factorization for CPU-optimized state transitions
Gated Output Fusion: Lightweight gating to blend memory output with attention stream

Research Context
This code was developed as part of a Master's thesis investigating explainable hybrid architectures for medical diagnosis. The GLR-SSM-EAD memory mechanism replaces baseline diagonal SSM to provide:

Interpretable evidence categorization (IMG/SYM/RF)
Adaptive temporal dynamics based on evidence type
CPU-efficient training and inference
Comprehensive explainability metrics

Performance Highlights

Accuracy: 97-98% on lung cancer classification
ROC-AUC: 0.99+
False Negative Rate: 2-3% (critical for cancer screening)
Model Size: ~11M parameters
Training Time: 60-90 minutes on CPU
Explanation Quality:

Deletion AUC < 0.01 (faithful)
Groundedness > 0.60 (traceable)
Jaccard > 0.84 (stable)




📁 Repository Structure
GLR-SSM-EAD-Archive/
│
├── README.md                           # This file
├── REPRODUCIBILITY_REPORT.md           # Detailed reproduction instructions
├── CODE_DOCUMENTATION.md               # In-depth code documentation
├── requirements.txt                    # Python dependencies
│
├── src/                                # Source code
│   ├── mini_Hymba_Medical_GLR_SSM_EAD_with_explanation_metrics_v1.py
│   └── __init__.py
│
├── scripts/                            # Utility scripts
│   ├── extract_glr_ssm_results.py     # Results extraction
│   ├── plot_training_curves.py         # Visualization
│   └── verify_installation.py          # Environment verification
│
├── data/                               # Dataset directory (empty - add your data here)
│   └── README.md                       # Dataset format instructions
│
├── models/                             # Saved models directory
│   └── .gitkeep
│
├── results/                            # Results output directory
│   └── .gitkeep
│
├── docs/                               # Additional documentation
│   ├── ARCHITECTURE.md                 # Architecture details
│   ├── EXPERIMENTS.md                  # Experimental setup
│   └── TROUBLESHOOTING.md              # Common issues and solutions
│
└── tests/                              # Unit tests
    ├── test_glr_ssm_memory.py
    ├── test_data_loading.py
    └── test_explanation_metrics.py

🔧 Installation
Prerequisites

Python: 3.8, 3.9, 3.10, or 3.11
Operating System: Linux (Ubuntu 22.04+), macOS (12+), or Windows (with WSL2)
RAM: 8 GB minimum, 16 GB recommended
Disk Space: 5 GB

Step 1: Clone Repository
bashgit clone <repository-url>
cd GLR-SSM-EAD-Archive
Step 2: Create Virtual Environment
bashpython3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Step 3: Install Dependencies
bashpip install --upgrade pip
pip install -r requirements.txt
Step 4: Verify Installation
bashpython scripts/verify_installation.py
Expected output:
✓ PyTorch 2.1.0+cpu installed
✓ Transformers 4.35.0 installed
✓ All dependencies satisfied
✓ BERT tokenizer downloadable
✓ Environment ready for training

🚀 Quick Start
Basic Training
bash# Ensure dataset is in data/ directory
cp your_dataset.csv data/cases_lung_cancer_filtered.csv

# Run training
cd src
python mini_Hymba_Medical_GLR_SSM_EAD_with_explanation_metrics_v1.py
Training with Logging
bashpython mini_Hymba_Medical_GLR_SSM_EAD_with_explanation_metrics_v1.py > ../results/training_log.txt 2>&1
Extract Results
bashcd scripts
python extract_glr_ssm_results.py ../results/training_log.txt
Interactive Prediction
After training completes, the script enters interactive mode:
Paste a clinical note (empty line to quit):
> 65-year-old male with 40-pack-year smoking history presented with persistent cough and hemoptysis. CT chest revealed 3.5 cm spiculated right upper lobe nodule with ground-glass opacity.

Yes, this patient has lung cancer. [P=96.8%]
Reason → Strong evidence supports a diagnosis of lung cancer. **Imaging:** pulmonary nodule; spiculated lesion; ground-glass opacity **Symptoms:** persistent cough; hemoptysis **Risk factors:** smoking history; advanced age
Key evidence: nodule, spiculated, hemoptysis, smoking, cough

🏗️ Code Architecture
High-Level Structure
Input Text → Tokenization → Embedding → Hybrid Blocks → Classification Head → Output
                                            ↓
                              [Attention + GLR-SSM-EAD Memory]
Hybrid Block Architecture
Each of 4 hybrid blocks contains:

Layer Normalization
Multi-Head Attention (8 heads, RoPE positioning)

LoRA adapters (rank=8) for parameter efficiency


GLR-SSM-EAD Memory

Evidence head: x → α ∈ [0,1]³ (IMG/SYM/RF)
Adaptive decay: λ_t = exp(-softplus(-a_eff_t))
Low-rank coupling: B_lr(x) = U(V(x))
Gated output: y = GELU(C(s)) ⊙ σ(G(x))


Feed-Forward MLP (4× expansion)

LoRA adapters



GLR-SSM-EAD Memory Mechanism
Mathematical Formulation:
python# Evidence scores (per token)
α_t = σ(MLP(x_t)) ∈ [0,1]³  # IMG, SYM, RF

# Adaptive decay (per channel group)
a_eff_t[IMG] = a_base[IMG] + α_t[0] * tanh(a_delta[IMG])
a_eff_t[SYM] = a_base[SYM] + α_t[1] * tanh(a_delta[SYM])
a_eff_t[RF]  = a_base[RF]  + α_t[2] * tanh(a_delta[RF])

λ_t = exp(-softplus(-a_eff_t))  # Decay factor ∈ (0,1]

# State update with low-rank input
s_t = λ_t ⊙ s_{t-1} + U(V(x_t))

# Gated output
y_t = GELU(C(s_t)) ⊙ σ(G(x_t))
Channel Partitioning (D=256):

Imaging channels: 85 (indices 0-84)
Symptoms channels: 85 (indices 85-169)
Risk factors channels: 86 (indices 170-255)


🔑 Key Components
1. Data Loading (load_csv)
Location: Lines 100-109
Purpose: Load and validate CSV dataset
pythondf = load_csv("data/cases_lung_cancer_filtered.csv")
# Returns: DataFrame with required columns and filtered by text length
Required CSV Columns:

lung_cancer_label: "lung cancer" or "no lung cancer"
case_text: Clinical narrative text
age (optional): Patient age
gender (optional): Patient gender

2. GLR-SSM-EAD Memory (GLR_SSM_EAD_Memory)
Location: Lines 145-237
Purpose: Novel memory mechanism with evidence-adaptive decay
Key Methods:

__init__(d, rank, p_drop): Initialize with dimension d, low-rank r
_build_a_eff(alpha_t): Compute effective decay from evidence scores
forward(x): Process sequence with adaptive state transitions

Usage:
pythonmemory = GLR_SSM_EAD_Memory(d=256, rank=16, p_drop=0.10)
output = memory(x)  # x: (batch, time, dim)
3. Hybrid Block (HybridHymbaBlockCPU)
Location: Lines 239-275
Purpose: Combine attention and GLR-SSM-EAD memory
Components:

RoPE attention with 8 heads
GLR-SSM-EAD memory
LoRA adapters (rank=8)
Feed-forward MLP

4. Explanation System (ReasonerV4)
Location: Lines 328-463
Purpose: Generate interpretable explanations with evidence categorization
Key Features:

Extracts imaging findings, symptoms, risk factors
Detects negations ("no", "without", "absence of")
Regex-based augmentation for common patterns
Clinical scoring system

Usage:
pythonreasoner = ReasonerV4(tokenizer)
rationale, buckets = reasoner.explain(
    positive=True,
    token_ids=ids,
    salient_indices=top_k_indices
)
5. Explanation Metrics
Location: Lines 465-550
Purpose: Comprehensive explainability evaluation
Metrics:

Faithfulness: deletion_insertion_auc(), sufficiency_comprehensiveness()
Plausibility: explanation_groundedness()
Stability: stability_checks(), jaccard()
Coherence: negation_mismatch_rate()
Readability: readability_proxy()


📚 Usage Examples
Example 1: Training from Scratch
pythonfrom src.mini_Hymba_Medical_GLR_SSM_EAD_with_explanation_metrics_v1 import *

# Load data
df = load_csv("data/cases_lung_cancer_filtered.csv")
df["input_text"] = df.apply(build_input, axis=1)

# Balance classes
pos = df[df.has_lung_cancer == 1]
neg = df[df.has_lung_cancer == 0]
df = pd.concat([neg, pos.sample(len(neg), replace=True, random_state=42)])

# Split
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["has_lung_cancer"], random_state=42)

# Initialize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = MiniHymbaCPUEncoder(tokenizer.vocab_size)

# Train
train(model, tokenizer, df_train, df_val, epochs=20, patience=3)

# Evaluate
evaluate(model, tokenizer, df_val)
Example 2: Load Trained Model and Predict
pythonimport torch
from transformers import AutoTokenizer
from src.mini_Hymba_Medical_GLR_SSM_EAD_with_explanation_metrics_v1 import *

# Load model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = MiniHymbaCPUEncoder(tokenizer.vocab_size)
model.load_state_dict(torch.load("models/best_lung_cls_glr_ssm_ead_cpu.pth", map_location='cpu'))
model.eval()

# Initialize reasoner
reasoner = ReasonerV4(tokenizer)

# Predict
text = "55-year-old male with persistent cough and pulmonary nodule on CT."
answer, prob, tokens, rationale = predict(model, tokenizer, reasoner, text)

print(f"Prediction: {answer}")
print(f"Probability: {prob:.2%}")
print(f"Explanation: {rationale}")
Example 3: Compute Explanation Metrics
pythonfrom src.mini_Hymba_Medical_GLR_SSM_EAD_with_explanation_metrics_v1 import *

# Get salient tokens
tokens, indices, encoding = salient_tokens(model, tokenizer, text, top_k=25)

# Faithfulness
deletion_insertion_result, base_prob = deletion_insertion_auc(
    model, tokenizer, text, indices, steps=(5, 10, 20)
)
print(f"Deletion AUC: {deletion_insertion_result['deletion_curve_mean']:.4f}")

suff_comp = sufficiency_comprehensiveness(model, tokenizer, text, indices)
print(f"Sufficiency: {suff_comp['suff']:.4f}")
print(f"Comprehensiveness: {suff_comp['comp']:.4f}")

# Stability
stability = stability_checks(model, tokenizer, reasoner, text, k=25)
print(f"Explanation Jaccard: {stability['exp_jacc_mean']:.4f}")
print(f"Prediction Invariance: {stability['pred_invariance']:.4f}")
Example 4: Custom Configuration
python# Modify hyperparameters at top of script
PATIENCE = 5        # Increase early stopping patience
MAX_LEN = 384       # Reduce sequence length for speed
BATCH_SIZE = 32     # Increase batch size if RAM allows
LR = 1e-4           # Lower learning rate
LORA_RANK = 4       # Reduce LoRA rank for efficiency

# Model with custom dimensions
model = MiniHymbaCPUEncoder(
    vocab_size=tokenizer.vocab_size,
    d=384,           # Larger hidden dimension
    n_layers=6,      # More layers
    n_heads=12       # More attention heads
)

# GLR-SSM-EAD with custom rank
memory = GLR_SSM_EAD_Memory(d=384, rank=32, p_drop=0.05)

⚙️ Configuration
Global Hyperparameters
File: src/mini_Hymba_Medical_GLR_SSM_EAD_with_explanation_metrics_v1.py
Lines: 30-39
pythonPATIENCE   = 3      # Early stopping patience (epochs)
THRESHOLD  = 0.5    # Classification threshold
MAX_LEN    = 512    # Maximum sequence length
BATCH_SIZE = 16     # Training batch size
LR         = 3e-4   # Learning rate
LORA_RANK  = 8      # LoRA adapter rank
SEED       = 42     # Random seed for reproducibility
Model Architecture
Default Configuration:

Hidden dimension (d): 256
Number of layers: 4
Attention heads: 8
MLP expansion: 4×
GLR-SSM-EAD rank: 16

Evidence-Adaptive Decay Initialization
python# Imaging: slowest decay (most persistent)
a_base_img = -0.6
a_delta_img = 0.0

# Symptoms: medium decay
a_base_sym = -1.0
a_delta_sym = 0.0

# Risk Factors: fastest decay (most transient)
a_base_rf = -1.6
a_delta_rf = 0.0
Training Configuration
python# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# Loss function
criterion = CrossEntropyLoss(weight=class_weights)

# Class balancing
# Positive class oversampled to match negative class size

# Gradient clipping
clip_grad_norm_(model.parameters(), 1.0)

📊 Output Files
Training Outputs
FileDescriptionLocationbest_lung_cls_glr_ssm_ead_cpu.pthTrained model weightsmodels/training_log.txtComplete execution logresults/
Results Outputs (from extraction script)
FileDescriptionContentglr_ssm_results.jsonStructured resultsAll metrics in JSON formattable1_classification_performance.csvClassification metricsPrecision, recall, F1 per classtable2_detailed_metrics.csvDetailed performanceROC-AUC, confusion matrix, etc.table3_explanation_metrics.csvExplanation qualityFaithfulness, plausibility, stabilitytable4_training_progress.csvTraining historyLoss and accuracy per epochresults_summary.mdSummary reportMarkdown format summary

🧪 Testing
Run Unit Tests
bashcd tests
python -m pytest test_glr_ssm_memory.py -v
python -m pytest test_data_loading.py -v
python -m pytest test_explanation_metrics.py -v
Test Coverage
bashpython -m pytest --cov=src tests/
Manual Testing
python# Test GLR-SSM-EAD memory
from src.mini_Hymba_Medical_GLR_SSM_EAD_with_explanation_metrics_v1 import GLR_SSM_EAD_Memory
import torch

mem = GLR_SSM_EAD_Memory(d=256, rank=16)
x = torch.randn(2, 100, 256)
output = mem(x)

assert output.shape == x.shape, "Shape mismatch"
print("✓ GLR-SSM-EAD memory test passed")

# Test evidence head
alpha = mem.evidence(x)
assert alpha.shape == (2, 100, 3), "Evidence shape mismatch"
assert alpha.min() >= 0 and alpha.max() <= 1, "Evidence scores out of range"
print("✓ Evidence head test passed")

📈 Performance Benchmarks
Training Performance
ConfigurationTime (min)AccuracyROC-AUCRAM (GB)Default (4 layers, d=256)75-9097.8%0.9923.5Small (3 layers, d=192)45-6096.5%0.9852.5Large (6 layers, d=384)120-15098.5%0.9955.5
Inference Performance
Batch SizeTime per SampleThroughput10.85 sec1.18 samples/sec160.12 sec8.33 samples/sec320.08 sec12.5 samples/sec

🔍 Troubleshooting
Common Issues
Issue: Out of memory during training
Solution: Reduce BATCH_SIZE to 8 or 4, or reduce MAX_LEN to 256
Issue: Slow training speed
Solution: Increase BATCH_SIZE if RAM allows, or reduce model size
Issue: Dataset file not found
Solution: Ensure CSV is in data/ directory with correct filename
Issue: BERT tokenizer download fails
Solution: Pre-download with python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"
See docs/TROUBLESHOOTING.md for comprehensive troubleshooting guide.

📝 Citation
If you use this code in your research, please cite:
bibtex@mastersthesis{caleb2025glr,
  title={Explainable Hybrid Architectures for Medical Diagnosis: A Hymba-Inspired State Space Framework with Evidence-Adaptive Memory},
  author={Caleb},
  year={2025},
  school={University of Windsor},
  type={Master's Thesis},
  supervisor={Hon Keung Kwan}
}
