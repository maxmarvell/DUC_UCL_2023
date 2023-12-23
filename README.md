# Generalized Path/Tile-Sum Algorithm for Computing Operator Correlations 

### Problem Setting

We consider a tensor product of single-site $q$-dimensional Hilbert spaces $\mathcal{H} \simeq \mathbb{C}^q$ over a spatial lattice of length $2L$ \($x \in \{-\frac{(L-1)}{2},...,-\frac{1}{2},0,\frac{1}{2},...,\frac{L}{2}\}$\). By convention, time evolution is given by layering rows of local unitary gates along the vertical axis of the tensor network. Each local gate $U$ evolves a combined 2-site Hilbert space $\mathbb{C}^q \otimes \mathbb{C}^q$ for half a full time step (N.B. another half step is needed to couple sites to both their neighbours within the full step). At $t=0$ we place a local charge operator on the central integer site $x=0$. We compute infinite-temperature correlation functions in the Heisenberg picture, which are nothing more than the Hilbert-Schmidt inner product $\frac{1}{q^{2L}}$ $Tr\[ \mathbb{U}^{\dagger}_{t} Z_{0} \mathbb{U}_{t} Z_{x} \]$ between our time-evolved origin charge and each local charge operator throughout space.

### The Code

