set -x

pip3 install torch torchvision

# install flash attention (optional)
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
pip install -v .
