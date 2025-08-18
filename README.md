# Implementation of Separable Sim(2) Group Convolutional Networks

## 1. Project Overview

This document provides a comprehensive theoretical and practical guide to implementing a Group Convolutional Neural Network (G-CNN) that is equivariant to the action of the 2D Similarity group, $Sim(2)$. The $Sim(2)$ group is the set of transformations on $\mathbb{R}^2$ comprising translations, rotations, and isotropic scaling. This  repository is concerned mainly on the application to medical images (in particular histopathological images). 

To this end an image, is conceptualized as a function $f: \mathbb{R}^2 \to \mathbb{R}^C$ (usually $C = 3$ for RGB images), so it can be thought as the coordinates of the $xy$-plane mapped to individual pixel values. In particular for medical images, these images can be transformed under the action of an element $g \in Sim(2)$ and conserve their properties (a cancerous cell is the same cancerous cell regardless of translation, zoom or rotation). A network is said to be equivariant if a transformation of the input (image) results in an equivalent transformation of the network's feature representations (e.g. rotation of the input rotates the same way the feature representations).

The motivation for enforcing $Sim(2)$-equivariance is to build a strong geometric inductive bias into the model. This enables the network to generalize from a single view of an object to recognize it under different positions, orientations, and scales, thereby improving data efficiency and robustness.

The implementation is based on two key principles from recent literature:
1.  **Continuous Kernel Parameterization**: The convolution kernels are not defined as discrete grids of weights but as continuous functions parameterized by a Sinusoidal Representation Network (SIREN). This allows for the evaluation of the kernel at any point in the continuous domain of the group, a necessity for handling Lie groups.
2.  **Separable Group Convolutions**: To mitigate the computational cost associated with convolutions over the higher-dimensional $Sim(2)$ group, we employ a separable formulation. The convolution is factorized into a sequence of operations over the constituent subgroups, significantly improving efficiency while preserving the equivariance property.

This document is structured as a step-by-step recipe, detailing the mathematical foundations and their translation into a concrete implementation.

---

## 2. Theoretical Framework

### 2.1. Group Convolutions

Let $G$ be a Lie group. A feature map in a G-CNN is a function $f: G \to \mathbb{R}^{C_{in}}$. The group convolution of $f$ with a kernel $k: G \to \mathbb{R}^{C_{out} \times C_{in}}$ is defined as:

$$(f \star_G k)(g) = \int_G k(g^{-1}\tilde{g}) f(\tilde{g}) \, d\mu(\tilde{g})$$

where $$g, \tilde{g} \in G$$ and $$\mu$$ is the left-invariant Haar measure on $$G$$. This operation is equivariant to the left-regular representation of the group, i.e., $$\begin{aligned}(\mathcal{L}_{g'}f) \star_G k = \mathcal{L}_{g'}(f \star_G k)\end{aligned}$$, where $$begin{aligned}(\mathcal{L}_{g'}f)(g) = f(g'^{-1}g)\end{aligned}$$.

### 2.2. The $Sim(2)$ Group

The group $Sim(2)$ is an affine Lie group that can be expressed as the semidirect product $G = \mathbb{R}^2 \rtimes H$, where $H = SO(2) \times \mathbb{R}^+$ is the subgroup of rotations and scalings. An element $g \in G$ is a pair $(x, h)$ where $x \in \mathbb{R}^2$ is a translation vector and $h = (R_\theta, s) \in H$ is a rotation-scaling representation ($s \in \mathbb{R}^+, R_\theta \in SO(2)$).

The group product is $(x, h) \cdot (\tilde{x}, \tilde{h}) = (x + h\tilde{x}, h\tilde{h})$, and the inverse is $(x, h)^{-1} = (-h^{-1}x, h^{-1})$.

### 2.3. Lifting and Group Convolution Layers

**Lifting Convolution (Layer 1):**
The initial layer must "lift" a standard 2D image, $f_{\text{in}}: \mathbb{R}^2 \to \mathbb{R}^{C_{in}}$, to a feature map on the group, $f_{\text{out}}: G \to \mathbb{R}^{C_{out}}$. This is achieved by convolving the input with a bank of kernels, where each kernel is a transformed version of a base spatial kernel $k_{\mathbb{R}^2}: \mathbb{R}^2 \to \mathbb{R}^{C_{out} \times C_{in}}$.

$$f_{\text{out}}(x, h) = (f_{\text{in}} \star_{\text{lift}} k_{\mathbb{R}^2})(x, h) = \int_{\mathbb{R}^2} f_{\text{in}}(\tilde{x}) \cdot (\mathcal{L}_h[k_{\mathbb{R}^2}])(\tilde{x}-x) \, d\tilde{x}$$

The group action on the kernel is defined as $(\mathcal{L}_h[k])(y) = \frac{1}{|\det(h)|} k(h^{-1}y)$. For $h=(R_\theta, s)$, $|\det(h)| = s^2$.

**Separable Group Convolution (Subsequent Layers):**
For a feature map $f: G \to \mathbb{R}^{C_{in}}$, a full group convolution is computationally expensive. So we assume the kernel $k: G \to \mathbb{R}$ is separable, i.e., $k(x, h) = k_{\mathbb{R}^2}(x) \cdot k_H(h)$. The group convolution integral can then be factorized:

$$(f \star_G k)(x, h) = \int_{\mathbb{R}^2} \underbrace{\left( \int_H f(\tilde{x}, \tilde{h}) k_H(h^{-1}\tilde{h}) \, d\mu_H(\tilde{h}) \right)}_{\text{Step 1: Subgroup Convolution}} \cdot \underbrace{(\mathcal{L}_h[k_{\mathbb{R}^2}])(\tilde{x}-x)}_{\text{Step 2: Spatial Convolution}} \, d\tilde{x}$$

This factorization allows for a more efficient two-step implementation.

---

## 3. Implementation Recipe

### Step 1: Abstract `Group` Interface

Define an abstract base class `Group` inheriting from `torch.nn.Module` to provide a standardized interface for all Lie group implementations.

**Required Methods:**
* `product(g1, g2)`: Implements the group multiplication $g_1 \cdot g_2$.
* `inverse(g)`: Computes the group inverse $g^{-1}$.
* `exponential_map(h)`: The map $\exp: \mathfrak{g} \to G$ from the Lie algebra to the group.
* `logarithmic_map(g)`: The map $\log: G \to \mathfrak{g}$ from the group to the Lie algebra.
* `representation(g)`: Returns the matrix representation of $g$ for its action on $\mathbb{R}^d$.
* `left_action_on_Rd(g, x)`: Computes the action of $g$ on a vector $x \in \mathbb{R}^d$.
* `determinant(g)`: Computes $|\det(\rho(g))|$ for the representation $\rho$.
* `normalize(h)`: Maps coordinates of a Lie algebra element $h \in \mathfrak{g}$ to a canonical domain (e.g., $[-1, 1]$) for stable input to the SIREN.
* `sample(...)`: Generates a discrete set of group elements from $G$.

### Step 2: Implement the `Sim(2)` Group and its Subgroups

Implement concrete `Group` classes for `SO(2)`, `R⁺`, and their combination `H = SO(2) \times R⁺`.

* **`SO(2)` (Rotations):**
    * **Manifold & Parameterization**: $S^1$, parameterized by an angle $\theta \in [0, 2\pi)$.
    * **Lie Algebra $\mathfrak{so}(2)$**: Isomorphic to $\mathbb{R}$. An element is identified with $\theta$. The generator is the skew-symmetric matrix $J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$.
    * **Maps**: $\exp(\theta) = e^{\theta J} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$. The `log` map is the matrix logarithm, which extracts $\theta$.

* **`Rplus` ($\mathbb{R}^+$, Scaling):**
    * **Manifold & Parameterization**: $(0, \infty)$, parameterized by a scale $s \in \mathbb{R}^+$.
    * **Lie Algebra $\mathfrak{r}^+$**: Isomorphic to $\mathbb{R}$. An element is identified with $\alpha = \ln s$.
    * **Maps**: $\exp(\alpha) = e^\alpha = s$. The `log` map is the natural logarithm, $\log(s) = \alpha$.

* **`Sim2Group` (Represents $H = SO(2) \times \mathbb{R}^+$):**
    * **Structure**: A direct product of `SO(2)` and `R⁺`.
    * **Lie Algebra $\mathfrak{h}$**: $\mathfrak{so}(2) \oplus \mathfrak{r}^+ \cong \mathbb{R}^2$. An element is a pair $(\theta, \ln s)$.
    * **Operations**: All group operations are performed component-wise on the `SO(2)` and `R⁺` parts.
    * **Representation**: The matrix representation of an element $(R_\theta, s)$ acting on $\mathbb{R}^2$ is $s R_\theta$.

### Step 3: Implement the Continuous Kernel Network (SIREN)

The convolution kernels are generated by a neural network, $k_\phi: \mathfrak{g} \to \mathbb{R}^{\dots}$, parameterized by weights $\phi$. We use a SIREN for this purpose due to its proven efficacy in representing complex, continuous functions and their derivatives.

* **Spatial Kernel SIREN ($k_{\mathbb{R}^2}$)**:
    * A SIREN, $k_{\phi_1}$, that maps a 2D spatial coordinate vector $v \in \mathbb{R}^2$ to the kernel value at that location.
    * Input Dimension: 2.
    * Output Dimension: $C_{out} \times C_{in}$.

* **Group Kernel SIREN ($k_H$)**:
    * A SIREN, $k_{\phi_2}$, that maps a 2D Lie algebra vector $h = (\theta, \ln s) \in \mathfrak{h}$ to a kernel value.
    * Input Dimension: 2.
    * Output Dimension: $C_{out} \times C_{in}$.

### Step 4: Implement the Separable Group Convolution Layer

This involves creating `torch.nn.Module`s for the lifting and group convolution layers.

* **Discretization and Sampling Strategy**:
    The continuous integrals are approximated by finite sums. This requires a discrete grid of group elements $\mathcal{H} = \{h_1, \dots, h_{N_h}\} \subset H$.
    * **`SO(2)` (Compact Subgroup)**: During training, use **random sampling**. A new set of $N_\theta$ rotations is drawn uniformly from $[0, 2\pi)$ for each forward pass. This provides an unbiased Monte Carlo estimate of the integral and acts as a powerful regularizer.
    * **`R⁺` (Non-Compact Subgroup)**: Use a fixed **discretization**. A grid of $N_s$ scales is generated by sampling uniformly in the Lie algebra (log-space) over a truncated range $[\ln(s_{min}), \ln(s_{max})]$ and then applying the exponential map. This ensures stability.

* **`LiftingConvolution` Module**:
    * **Input**: An image tensor of shape $(B, C_{in}, H, W)$.
    * **Output**: A group feature map of shape $(B, C_{out}, N_h, H', W')$, where $N_h = N_\theta \times N_s$.
    * **Forward Pass**:
        1.  Generate a spatial coordinate grid $\mathcal{X}_{\text{ker}}$ of size $(k_s, k_s)$.
        2.  Sample a grid of group elements $\mathcal{H} = \{h_j\}_{j=1}^{N_h}$ using the strategy above.
        3.  For each $h_j \in \mathcal{H}$:
            a.  Transform the kernel coordinates: $\mathcal{X}'_j = \{h_j^{-1}x \mid x \in \mathcal{X}_{\text{ker}}\}$.
            b.  Evaluate the spatial SIREN $k_{\phi_1}$ on $\mathcal{X}'_j$ to get the transformed kernel weights $W_j$.
            c.  Apply the Jacobian correction: $W'_j = \frac{1}{|\det(h_j)|} W_j$.
            d.  Apply a standard 2D convolution (`torch.nn.functional.conv2d`) to the input using $W'_j$ to produce the $j$-th slice of the output feature map.
        4.  Stack the $N_h$ output slices along the new group dimension.

* **`SeparableGroupConvolution` Module**:
    * **Input**: A group feature map of shape $(B, C_{in}, N_h, H, W)$.
    * **Output**: A group feature map of shape $(B, C_{out}, N_h, H', W')$.
    * **Forward Pass**:
        1.  **Step A: Subgroup Convolution**:
            a.  Sample input and output group grids, $\mathcal{H}_{\text{in}} = \{h_i\}_{i=1}^{N_h}$ and $\mathcal{H}_{\text{out}} = \{h_j\}_{j=1}^{N_h}$.
            b.  Compute all relative transformations in the Lie algebra: $v_{ij} = \log(h_j^{-1} h_i) \in \mathfrak{h}$.
            c.  Evaluate the group SIREN $k_{\phi_2}$ on the set of $\{v_{ij}\}$ to obtain the group convolution weights $W_H \in \mathbb{R}^{N_h \times N_h \times C_{out} \times C_{in}}$.
            d.  Perform the convolution over the group dimension. This is equivalent to a matrix multiplication at each spatial location, which can be implemented efficiently using `torch.einsum`: `bcij,bchw->bohw`. The output is an intermediate feature map $f_{\text{int}}$ of shape $(B, C_{out}, N_h, H, W)$.
        2.  **Step B: Spatial Convolution**:
            a.  This step is analogous to the `LiftingConvolution`. For each output group element $h_j \in \mathcal{H}_{\text{out}}$, generate a transformed spatial kernel $W'_j = \mathcal{L}_{h_j}[k_{\mathbb{R}^2}]$.
            b.  Convolve each slice $f_{\text{int}}(\cdot, \cdot, j, \cdot, \cdot)$ with its corresponding spatial kernel $W'_j$.
            c.  The results are the final output slices, which are already arranged correctly.

### Step 5: Assemble the Full Network Architecture

Combine the custom layers into a cohesive model, such as a ResNet.

1.  **Lifting Layer**: The first layer of the network is the `LiftingConvolution` module.
2.  **Equivariant Body**: The main body consists of a series of blocks, each containing one or more `SeparableGroupConvolution` layers and non-linear activations.
3.  **Invariant Pooling**: After the final equivariant block, the feature map has shape $(B, C_{final}, N_h, H_{out}, W_{out})$. To achieve a final prediction that is invariant to the group action, apply a pooling operation (e.g., max or mean) across the group dimension (dim=2). The result is an invariant feature map of shape $(B, C_{final}, H_{out}, W_{out})$.
4.  **VAE Head**: Flatten or apply global average pooling over the spatial dimensions, followed by one or more `torch.nn.Linear` layers to produce the final output logits.