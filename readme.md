# H. Pylori Classification Benchmarks

This repository contains benchmarks for H. Pylori detection and classification using various pathology foundation models.

## 🔨 1. Installation

### Clone the Repository
Since this repository points to other repositories (submodules), perform a recursive clone:
```bash
git clone --recursive git@github.com:laneegale/hp_detection.git
cd hp_detection

```

### Environment Setup

Create and activate a clean Conda environment:

```bash
conda create -n "trident" python=3.10
conda activate trident

```

### Install Dependencies

Install the required packages. This setup assumes you are using the `trident` submodule or repository structure.

```bash
# Clone trident if not already present via recursive clone
# git clone [https://github.com/mahmoodlab/trident.git](https://github.com/mahmoodlab/trident.git) && cd trident

# Install local package in editable mode
pip install -e .

# Install additional requirements
pip install h5py tqdm

```

> **Note:** Additional packages may be required to load specific pretrained models. Please follow the error messages for any missing model-specific dependencies.

### Hugging Face Authentication

Ensure you are logged into Hugging Face and have acquired the necessary permissions (Gated Repos) for the models you intend to use (e.g., Virchow2, CONCH, etc.).

```python
from huggingface_hub import login
login(token="<YOUR_HUGGINGFACE_WRITE_TOKEN>")

```

---

## 🧬 2. Feature Extraction

Extract features from pathology images using various foundation models.

### Standard Models

**Usage:**

```bash
python get_feats.py <models> <data_dir> <save_path>

```

**Arguments:**

* `<models>`: Comma-separated list of models to use (e.g., `virchow2,hoptimus0,musk`).
* `<data_dir>`: Path to the directory containing input images (can be nested).
* `<save_path>`: Path where the extracted `.h5` features will be saved.

**Supported Models:**

* `virchow2`
* `ctranspath`
* `hoptimus0`
* `hoptimus1`
* `uni_v2`
* `musk`
* `conch_v15`
* `hibou-l` (Transformers: `histai/hibou-L`)

### CHIEF Model

**Note:** The CHIEF model requires a specific environment setup. Please refer to the [CHIEF Repository](https://github.com/hms-dbmi/CHIEF) for installation instructions.

**Usage:**
Once the CHIEF environment is active:

```bash
python get_CHIEF_feats.py <data_dir> <save_path>

```

---

## 🎯 3. Classification

Train and evaluate a logistic regression classifier on the extracted features. This implementation is adapted from the [UNI repository](https://github.com/mahmoodlab/UNI).

**Usage:**

```bash
python classification.py <models> <h5_dir> <save_path>

```

**Arguments:**

* `<models>`: Comma-separated list of models to classify (e.g., `virchow2,hoptimus0,musk`).
* `<h5_dir>`: Directory containing the `.h5` features extracted in the previous step.
* `<save_path>`: Location to save the results (saved as a pickle file).

---

## 🎨 4. Visualization (Attention Masks)

Generate attention masks to visualize model focus on specific images or folders.

**Usage:**

```bash
python draw_attn.py [-i IMAGE] [-f FOLDER] -o OUTPUT -m MODEL

```

**Arguments:**

| Argument | Flag | Required | Description |
| --- | --- | --- | --- |
| **Image** | `-i`, `--image` | No* | Path to a single input image. |
| **Folder** | `-f`, `--folder` | No* | Path to a folder of input images. |
| **Output** | `-o`, `--output` | **Yes** | Path/Location for the output attention masks. |
| **Model** | `-m`, `--model` | **Yes** | The model used to generate attention maps. |

*\*Note: You must provide either an image (`-i`) or a folder (`-f`).*

**Supported Models for Visualization:**

* `conch_v15`
* `virchow2`

### Example

```bash
python draw_attn.py --image data/slide_1.tif --output results/attn_masks/ --model virchow2

```
