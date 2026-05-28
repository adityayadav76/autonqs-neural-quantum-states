# Installation

autonqs requires Python 3.10 or newer and PyTorch. The existing workspace was
tested with the conda environment named `pytorch`.

## Existing Conda Environment

```cmd
conda run -n pytorch python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected result on a CUDA machine:

```text
True
<your GPU name>
```

## Pip Requirements

```cmd
pip install -r requirements.txt
```

For CUDA builds of PyTorch, install the wheel appropriate for your driver and
CUDA runtime from the official PyTorch installation selector:

```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining dependencies:

```cmd
pip install numpy pytest
```

## Editable Install

From the repository root:

```cmd
pip install -e .
```

This makes `python -m autonqs.cli ...` available from any working directory in
the same environment.
