# STAGE ONE

## Single Cell Setup

**What remains the same?**

- Single cell
- $N$ Users uniformly sampled from a grid on a plane at 1.5m height.
- Post process channels and filter-out blocked channels, resulting in valid users.
- Different environments with varying density/height of scatterers.

**What is new?**

- BS is located on top of a building/tower with a realistic height.
- BS array size, carrier frequency, tilt angle varies.
- UE array size varies.
- Simple mobility modeling: For each user, sample a speed from some distribution and a random direction. Mixture of static users, pedestrians, mobile users.
- Codebook-based precoding so that each valid user has a decent channel.
- Simulate time-frequency channel for each antenna pair for each user. Save $H \in C^{n_t \times n_r \times n_s \times n_f}$ where $n_f$ is the number of subcarriers and $n_s$ the number of OFDM symbols per block.

**Why?**

This dataset enables studying two fundamental aspects:

- Modeling mobility
- Pretraining over multiple array TX/RX antenna array sizes

## Simple Multi-Cell Pretraining

Multi-cell scenarios enable realistic inter-cell interference modeling. The single cell case described above can be viewed as a special case of the multi-cell setting.

**Data Generation Procedure:**

For each user, the following steps are performed:

1. Select precoders from the codebooks for all base stations.
2. Obtain multi-cell channel measurements to model interference given the selected precoders.
3. Extract the channel from each BS given the corresponding precoder

**Why?**

This dataset enables studying:

- Inter-cell interference characterization and modeling
- Base station assignment policies and their impact on channel statistics
- Resource allocation in multi-cell deployments
- Realistic interference patterns that arise in operational networks