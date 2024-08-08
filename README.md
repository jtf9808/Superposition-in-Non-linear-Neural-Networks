# Superposition in Non-linear Neural Networks #

This repository contains reproductions of some of the graphs presented in [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) showing how smaller non-linear neural networks can use super position to represent/compress larger neural networks.

I was interested to see how the representations form, especially after seeing [Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), so have extended these graphs to show how they change during training.

I have only tested the code with python3.11.

Non-linear neural network layers can represent more features than they have neurons through a process called superposition provided these features are sparse. 
The following graphs show how 5 dimensional features are mapped into a 2-dimensional nn layer for different sparsities. 

For more explanation on the following graphs see [here](https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating).

<img src="gifs/linear_network_sparsity_0.00_plots.gif" alt="linear_network_sparsity_0.00_plots.gif" width="40%"/>
<img src="gifs/non_linear_network_sparsity_0.00_plots.gif" alt="non_linear_network_sparsity_0.00_plots.gif" width="40%"/>
<img src="gifs/linear_network_sparsity_0.25_plots.gif" alt="linear_network_sparsity_0.25_plots.gif" width="40%"/>
<img src="gifs/non_linear_network_sparsity_0.25_plots.gif" alt="non_linear_network_sparsity_0.25_plots.gif" width="40%"/>
<img src="gifs/linear_network_sparsity_0.50_plots.gif" alt="linear_network_sparsity_0.50_plots.gif" width="40%"/>
<img src="gifs/non_linear_network_sparsity_0.50_plots.gif" alt="non_linear_network_sparsity_0.50_plots.gif" width="40%"/>
<img src="gifs/linear_network_sparsity_0.80_plots.gif" alt="linear_network_sparsity_0.80_plots.gif" width="40%"/>
<img src="gifs/non_linear_network_sparsity_0.80_plots.gif" alt="non_linear_network_sparsity_0.80_plots.gif" width="40%"/>
<img src="gifs/linear_network_sparsity_0.90_plots.gif" alt="linear_network_sparsity_0.90_plots.gif" width="40%"/>
<img src="gifs/non_linear_network_sparsity_0.90_plots.gif" alt="non_linear_network_sparsity_0.90_plots.gif" width="40%"/>
<img src="gifs/linear_network_sparsity_0.95_plots.gif" alt="linear_network_sparsity_0.95_plots.gif" width="40%"/>
<img src="gifs/non_linear_network_sparsity_0.95_plots.gif" alt="non_linear_network_sparsity_0.95_plots.gif" width="40%"/>