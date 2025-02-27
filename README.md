## `torchtitan` for masked language modeling

This is a copy of the [`torchtitan`](https://github.com/pytorch/torchtitan) library that I use to run masked LLM training experiments on Frontier. 

### Prerequisites
* Install PyTorch nightly with ROCm 6.3:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
```
My PyTorch-ROCm version is nightly `2.7.0.dev20250221+rocm6.3` and I think a reasonably recent nightly version is necessary to reproduce the results below.

* Clone this repo and install the following packages:
```bash
pip install datasets torchdata tomli tensorboard sentencepiece tiktoken blobfile tabulate ninja
``` 

* Download the Llama-3.1-8B tokenizer:

```python 
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=...
```

where `hf_token` is your Hugging Face Hub token.

* Unlike for CUDA, you will need to install FlashAttention-2 for ROCm separately. [This page](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html) provides the instructions for that. Basically, to install from source:

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention/
GPU_ARCHS=gfx90a python setup.py install  # MI200 series
```
Here, `gfx90a` is the correct GPU architecture choice for MI250X. In the last step, make sure to build with `ninja` (`pip install ninja` if it's not already installed), otherwise it might take forever. Before running the above, make sure to set your ROCm home directory correctly for the installation to proceed: *e.g.* `export ROCM_HOME=/opt/rocm-6.3.1` for ROCm 6.3; also set `export MAX_JOBS=64` or something large like that to speed up the installation.

* Install the `aws-ofi-rccl` plugin, which enables `rccl` (AMD ROCm's version of `nccl`) to use `libfabric` for a more performant interconnect. I provide a shell script here ([`aws_ofi_rccl.sh`](https://github.com/eminorhan/frontier-torchtitan/blob/master/aws_ofi_rccl.sh)) to install this plugin. Simply run this script (*e.g.* `sh aws_ofi_rccl.sh`) to install the plugin (the script assumes that your ROCm version is 6.3.1 and the `libfabric` version is 1.22.0; if you're using different versions, change it accordingly).

### Pretraining data
Currently, the pretraining data consist of a combination of the following datasets:

* [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2), which is itself a cross-deduplicated and filtered combination of DCLM (3.3T), FineWeb-Edu (1.3T), Dolma (0.2T), Zyda (0.2T).

* Stack-2: the [`the-stack-v2-train-smol-ids`](https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids) subset (525B).

* [`FineMath`](https://huggingface.co/datasets/HuggingFaceTB/finemath): the `finemath-3plus` subset (34B).

The numbers in parentheses represent the approximate token counts (the full dataset has ~5.56T tokens). The subdirectory [`download_scripts`](https://github.com/eminorhan/frontier-torchtitan/tree/master/download_scripts) contains basic Python scripts to download these datasets. The mixture weights for these components are currently as follows (in terms of data rows, not tokens): DCLM (40%), FineWeb-Edu (44%), Dolma (3%), Zyda (2%), Stack-2 (10%), FineMath (1%).

### Data loading strategy
The data loading strategy is currently as follows (implemented [here](https://github.com/eminorhan/frontier-torchtitan/blob/master/torchtitan/datasets/hf_datasets.py)):

* load individual component datasets in streaming mode (as iterable datasets)
* interleave the component datasets using `ds.interleave_datasets()`
* shuffle the combined dataset with a large buffer size (`buffer_size=100000`) and a globally shared random seed
* split the dataset across `dp` (data-parallel) ranks using `ds.split_dataset_by_node()`

The shuffle is performed once at the beginning of each training session with a fresh global random shuffling seed (due to job runtime limits on Frontier, each session takes 24 hours at most after which we checkpoint and restart again). The shuffle operation shuffles the dataset shards as well as the rows in the buffer and the large buffer size ensures that all data rows in the shard get a chance to be consumed during a ~24 hour training session.

This data loading pipeline is preferred over the one implemented in the torchtitan library ([here](https://github.com/pytorch/torchtitan/blob/main/torchtitan/datasets/hf_datasets.py)), which checkpoints a `_sample_idx` variable and attempts to skip to that idx at the beginning of the next training session, since I couldn't verify that this implementation works correctly (I observed that after resuming the checkpoint, the data loader would keep sampling some of the same data rows from the previous sessions, which should have been skipped).

### Training
The SLURM batch script in [`train_1B_n64.sh`](https://github.com/eminorhan/torchtitan-mlm/blob/master/train_1B_n64.sh) can be used to train a Llama-3.2-1B model with a context size of 16384 tokens over 64 Frontier nodes (with a global batch size of 12.6M tokens per training step). This script uses the training config file in [`train_configs/llama3_1b_n64.toml`](https://github.com/eminorhan/torchtitan-mlm/blob/master/train_configs/llama3_1b_n64.toml). Feel free to modify the config according to your needs.

#### Training throughput
In the training setup above, we can go through ~300 training steps per hour (3.8B tokens per hour). Going through the entire dataset once, *i.e.* 1 epoch over 5.56T tokens would take about 1500 hours or roughly 2 months.

### A note on IP network interfaces
For loading and saving distributed checkpoints, the code uses the `torch.distributed.checkpoint` (DCP) library. A new process group with the `gloo` backend is created for this purpose (separate from the process group used by `nccl` for training). In my experience, the IP network interface to be used by both `gloo` and `nccl` needs to be explicitly set to `hsn0`, *i.e.*:
```bash
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
```
Otherwise, it becomes impossible to run on more than ~300 nodes due to communication failures.

### Checkpoint conversions
Two utility scripts to convert checkpoints between `DCP` and `torch.save` formats are provided here. [`llama_to_dcp.py`](https://github.com/eminorhan/torchtitan-mlm/blob/master/llama_to_dcp.py) converts a checkpoint saved with `torch.save` to `DCP` format. This is useful when initially converting the original Llama-3 checkpoints into `DCP` format to continue pretraining them with the code in this repository (you will most likely need to use this only once before starting continued pretaining). You can do this as follows:
```bash
python llama_to_dcp.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```
where `INPUT_DIR` is the directory where the original checkpoint is saved (downloaded from [here](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main/original) for the Llama-3.2-1B model) and `OUTPUT_DIR` is the directory where the `DCP` checkpoint will be saved. The bulk of this script was copied from [this PR](https://github.com/pytorch/torchtitan/commit/3247841423429faf37bdf6918204350db293e482) by [`rlsl (Rasmus)`](https://github.com/rlrs). 

For the conversion in the other direction (`DCP --> torch.save`), you can use the [`dcp_to_llama.py`](https://github.com/eminorhan/torchtitan-mlm/blob/master/dcp_to_llama.py) script like so:
```bash
python dcp_to_llama.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```
where `INPUT_DIR` now holds the `DCP` checkpoint and the `.pth` checkpoint will be saved in `OUTPUT_DIR`. You will need to do this conversion to evaluate the intermediate checkpoints. Optionally, you can also push the intermediate checkpoints (converted into `.pth` format) to huggingface by passing the argument `--push_to_hub`.
