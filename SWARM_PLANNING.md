# Next-Generation PSI Swarm (Mixture of Experts) Architecture Plan

To move beyond the basic "soft-average" swarm to a true state-of-the-art Mixture of Experts (MoE), we must improve how the experts are selected and how they interconnect. 

## 1. Top-K Sparse Routing (The Mixtral Approach)
**The Problem:** In the naive swarm, every expert processes every reaction. If we scale to 8 experts, it takes 8x the compute. 
**The Solution:** Implement a **Top-2 Router**. The router evaluates the molecule and selects only the top 2 most qualified experts for that specific reaction. The other 6 experts are turned off for that batch. 
* *Benefit:* We can massively increase the number of experts (more capacity to memorize complex physics) without slowing down the GPU.

## 2. Decoupled Swarms (Geometry vs. Energy)
**The Problem:** The current script routes one expert to handle *both* the EGNN geometry refinement and the Ea energy prediction. However, an expert that is brilliant at identifying hydrogen transfers (Geometry) might not be the best at calculating high-barrier thermodynamics (Energy).
**The Solution:** Two separate routers and two separate swarms.
* **Geometry Swarm:** 4 EGNN experts managed by a Geometry Router.
* **Thermodynamic Swarm:** 4 Ea Head experts managed by an Energy Router.

## 3. Expert Cross-Talk (Latent Interconnection)
**The Problem:** The experts work in total isolation. They don't know what the other experts are thinking until the very end when their answers are averaged.
**The Solution:** Add a **Swarm Consensus Layer** (Self-Attention across experts). 
* Before the final Activation Energy is predicted, the Top-2 chosen experts share their internal feature vectors. 
* Expert A can "attend" to Expert B's representation. If Expert A is uncertain about a bond angle, it can dynamically borrow information from Expert B before making its final Ea prediction.

## Implementation Roadmap
1. **Refactor `psi_swarm_architecture.py`:** Update the mock script to implement Top-2 routing and Expert Cross-Talk to validate the tensor dimensions.
2. **Loss Function Updates:** Add a "Load Balancing Loss" to ensure the Router doesn't become biased and just pick Expert 1 every single time.
3. **Integration:** Transplant the finalized architecture into `psi_full_pipeline.py` when the current baseline run finishes.
