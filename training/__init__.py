"""WorldDistill Training Module.

Provides distillation training pipelines for video generation and world models.
Supports multiple distillation methods:
- Step Distillation (fixed N-step)
- Stream Distillation (Diffusion Forcing)
- Progressive Distillation
- Consistency Distillation (TCD/LCD)
- Context Forcing Distillation

References:
- HY-WorldPlay: Flow Matching training + Context Forcing + Muon optimizer
- Open-Sora 2.0: ColossalAI Booster + bucket sampling + Rectified Flow
"""
