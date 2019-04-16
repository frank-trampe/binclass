# binclass

This is a binary image classifier. It uses slicing rules, a neural network, and post-processing rules to return a true or false value. See `binclass.py` for specific options and `examples/handwriting/README.md` for invocation examples.

## Setup

Run `git submodule update --init` to get img2vec and `pip install torch torchvision sklearn` to get other Python dependencies. On first run, torch will try to fetch the neural networks.

