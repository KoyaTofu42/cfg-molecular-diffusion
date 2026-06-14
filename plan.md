Here is a complete, structured Markdown document outlining the research idea. You can share this directly with collaborators, students, or an engineering team to guide the implementation.

***

# Research Project: Multi-Property Guided Molecular Diffusion via Classifier-Free Guidance (CFG)

## 1. Motivation and Background
A recent comprehensive survey on molecular diffusion models identified a critical gap: **existing research heavily focuses on unconditional or single-property conditional generation.** However, real-world drug and material design is an inherently multi-objective problem. A useful molecule must simultaneously satisfy multiple criteria (e.g., high binding affinity, low toxicity, and high solubility).

This project aims to implement a **multi-conditional 3D molecular diffusion model** with a medium resource footprint. To achieve this without inventing complex new mathematical frameworks, we will adapt **Classifier-Free Guidance (CFG)**—the exact mechanism that allows models like Stable Diffusion to follow complex text prompts—to molecular property conditioning.

## 2. Project Scope & Resource Requirements
*   **Target Modality:** 3D Geometric Space (Atom types + 3D coordinates).
*   **Baseline Architecture:** [Equivariant Diffusion Model (EDM)](https://github.com/ehoogeboom/e3_diffusion_for_molecules) (Hoogeboom et al., 2022). It is open-source, mathematically continuous (DDPM), and relies on standard Equivariant Graph Neural Networks (EGNNs).
*   **Dataset:** **QM9** (~130,000 small molecules). It is lightweight and includes multiple continuous quantum properties.
*   **Hardware:** 1x GPU with 16GB–24GB VRAM (e.g., RTX 3090, RTX 4090, or Google Colab Pro A100).
*   **Estimated Timeline:** 3–4 weeks for a single developer.

## 3. Methodology & Technical Approach

### 3.1. Data Preparation
Select 3 continuous properties from the QM9 dataset to act as our multi-condition vector $C$. Good candidates are:
1.  Polarizability ($\alpha$)
2.  Dipole Moment ($\mu$)
3.  HOMO-LUMO Gap ($\Delta \epsilon$)

**Preprocessing:** Normalize each property across the dataset to have a mean of $0$ and a standard deviation of $1$.

### 3.2. Architectural Modifications
In the baseline EDM code, the model ingests conditioning information alongside the diffusion time step $t$.
1.  Modify the input layer to accept a condition vector $C = [p_1, p_2, p_3]$.
2.  Pass $C$ through a small Multi-Layer Perceptron (MLP) to project it into a condition embedding.
3.  Concatenate or add this condition embedding to the diffusion time embedding before passing it into the EGNN layers.

### 3.3. Training with Classifier-Free Guidance
To enable CFG, the model must learn both conditional and unconditional generation simultaneously. 
During the training loop:
1.  Define a dropout probability, typically $p_{uncond} = 0.15$ (15%).
2.  For each batch, generate a random mask based on $p_{uncond}$.
3.  Where the mask is `True`, replace the ground-truth property vector $C$ with a learned dummy vector `[NULL]` (e.g., a vector of zeros or a specific initialized tensor).
4.  Train the model to predict the noise $\epsilon$ using standard DDPM loss.

*Pseudocode for Training:*
```python
# p_uncond = 0.15
if random.random() < p_uncond:
    condition_vector = learnable_null_vector
else:
    condition_vector = real_property_vector

loss = diffusion_loss(noisy_molecule, time_step, condition_vector)
loss.backward()
```

### 3.4. Inference (Sampling) with Guidance
During the reverse diffusion (generation) process, we generate molecules by guiding the unconditional baseline towards our desired property conditions.
For every diffusion time step $t$, run the EGNN model **twice**:
1.  $\epsilon_{cond}$: Forward pass with the target properties $C$.
2.  $\epsilon_{uncond}$: Forward pass with the `[NULL]` condition vector.

Combine the predictions using a **Guidance Scale ($w$)**:
$$ \epsilon_{final} = \epsilon_{uncond} + w \cdot (\epsilon_{cond} - \epsilon_{uncond}) $$

*   If $w = 0.0$, the model generates random molecules (unconditional).
*   If $w = 1.0$, it is standard conditional generation.
*   If $w > 1.0$ (e.g., 2.0 to 4.0), the model strongly forces the molecule to adhere to the requested properties.

## 4. Implementation Milestones

### Phase 1: Environment & Baseline Setup
*   Clone the EDM repository.
*   Set up the Python/PyTorch environment.
*   Run a standard *unconditional* training loop on QM9 for 10-20 epochs to verify that the baseline works and generates valid 3D point clouds.

### Phase 2: Conditioning Pipeline
*   Extract the selected properties ($\alpha, \mu, \Delta \epsilon$) from the QM9 dataloader.
*   Implement the condition MLP and integrate it into the EGNN time-embedding block.
*   Implement the `[NULL]` masking logic for the training loop.

### Phase 3: Training
*   Train the modified model on QM9. Monitor the standard variational lower bound (VLB) loss.
*   Save model checkpoints.

### Phase 4: Inference Engine & Classifier-Free Guidance
*   Write a custom sampling script that implements the CFG equation.
*   Allow the user to input desired property values (e.g., "Generate a molecule with high polarizability and a low HOMO-LUMO gap").

## 5. Evaluation Metrics

To prove the project is successful, evaluate the generated molecules on two fronts:

### A. Chemical Viability (Standard Metrics)
Use `RDKit` (or the evaluation scripts provided in EDM) to parse the generated 3D point clouds into graphs and calculate:
*   **Validity:** % of generated point clouds that form chemically valid molecules.
*   **Uniqueness:** % of generated valid molecules that are unique.
*   **Novelty:** % of generated molecules that do not exist in the QM9 training set.

### B. Condition Adherence (Target vs. Actual)
Because QM9 properties are based on quantum chemistry (DFT), calculating them from scratch for new molecules is computationally expensive.
*   **Solution:** Train a simple, lightweight predictive EGNN (an "Oracle") on the QM9 dataset that takes a 3D molecule and predicts its $\alpha, \mu, \Delta \epsilon$. 
*   **Metric:** Generate 1,000 molecules with specific requested properties. Pass them through your Oracle. Calculate the **Mean Absolute Error (MAE)** between the *requested* properties and the *Oracle-predicted* properties.
*   Plot the MAE across different Guidance Scales ($w = [1.0, 2.0, 3.0, 5.0]$) to demonstrate the effectiveness of CFG.

## 6. Potential Challenges & Mitigations
*   **Conflicting Conditions:** Requesting a physical impossibility (e.g., properties that are inversely correlated in nature) will cause the model to generate fragmented/invalid molecules. *Mitigation:* Ensure test conditions are sampled from the joint distribution of the training data.
*   **Guidance Collapse:** A guidance scale $w$ that is too high (e.g., $w > 10$) usually destroys the structural integrity of the generation in diffusion models. *Mitigation:* Perform a grid search to find the "sweet spot" for $w$ (usually between 1.5 and 4.0).
