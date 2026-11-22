---
title: gmpe-bench explorer
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Leakage-audited NGA-West2 ground-motion model benchmark
---

# gmpe-bench explorer

Interactive view of the leakage-audited NGA-West2 benchmark of ground-motion models and of HSCP calibrated
uncertainty. Source code, data instructions and the full pipeline:
<https://github.com/PavanMohanN/gmpe-benchmarking>.

This Space shows the published benchmark results. It does not hold PEER ground-motion data: uploaded
prediction files are audited against the frozen splits, and the app gives the code to run the full
evaluation locally after building the NGA-West2 profile from the PEER flatfiles.
