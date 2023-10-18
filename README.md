# Path Integrals in DU circuits by Tile-Sum decomposition

### Problem Setting

This repository is built for computing infinite temperature correlations in the Heisenberg picture between a time-evolved origin charge, $(x_0,t=0)$, and a local operator at an arbitrary spatial location $(x_b,t=T)$. This is computed under a dual-unitary framework (Bertini, Kos and Prosen), where under perfect behaviour we expect infinite temperature correlations to be zero solely at the light cone front, $x_b = T$. As we peterb away from dual-unitarity the solution becomes not so trival as non-zero correlation propogate deeper into the light-cone.

To quantify the effect of a uniform pertubation on the frameworks ability to propagate information as a soliton measurements must be made at ever increasing time and depth. Consequently, the number of contracted tensors in the causal block - a region that sees both local operators - scales quadratically and so simple complete contraction methods are not effective for larger times and depths. Bertini, Kos and Prosen suggest the use of a path-integral method where the two-point correlation function is approximated by a sum of skeleton paths between the $(x_0,t=0)$ and $(x_b,t=T)$. This work implements this method and extends the idea to arbitrary $d$ dimensional Hilbert space.

### The Code


