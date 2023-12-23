# Path Integrals in DU circuits by Tile-Sum decomposition

### Problem Setting

This repository is built for computing infinite temperature correlations in the Heisenberg picture between a local charge operators evolved from the origin $(x_0,t=0)$ and an arbitrary spatial location $(x_b,t=T)$ at some later time. This is computed under a dual-unitary framework (Bertini, Kos and Prosen), where under perfect behaviour we expect infinite temperature correlations to be non-zero solely at the light cone front, $x_b = T$. As we perturb away from dual-unitarity the solution becomes non-trival as non-zero correlation propagate deeper into the light-cone.

To quantify the effect of a uniform pertubation on the framework's ability to propagate information as a soliton, measurements must be made at ever increasing time and depth. Consequently, the number of contracted tensors in the causal block - a region that sees both local operators - scales quadratically and so simple complete contraction methods are not effective for larger times and depths. Bertini, Kos and Prosen suggest the use of a path-integral method where the two-point correlation function is approximated by a sum of skeleton paths between the $(x_0,t=0)$ and $(x_b,t=T)$. Our work implements this method and extends the idea to use paths of arbitrary thickness - i.e. to encompass paths with looping & branching supported within some maximum width.

### The Code

Prior to computing any infinite temperature correlations, we require a perturbed dual-unitary framework with which we can can evolve our orgin charge. This requires a) a dual unitary matrix, $U$ of dimension $\mathcal{R}^{2\times2}$ and b) A perturbing matrix, $P$, to apply to $U$. Both of these are given in the files DU_generator.py and P_generator.py.
