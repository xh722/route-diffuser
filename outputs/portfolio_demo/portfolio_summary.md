# RouteDiffuser

> Autonomous driving trajectory planning with a route-conditioned diffusion policy

A planner-core autonomous driving project that demonstrates canonical scene modeling, route-conditioned diffusion inference, and end-to-end debug artifacts without relying on proprietary logs.

## Highlights
- Canonical scene schema for ego, neighbors, lanes, route polylines, and masks.
- Route-prior residual diffusion with a conditional 1D U-Net decoder.
- Structured synthetic driving scenarios spanning keep-lane, lane changes, and curves.
- End-to-end scripts for training, inference, evaluation, and portfolio artifact generation.

## Scenario Coverage
- `keep_lane`: Stable forward planning on a straight lane centerline.
- `lane_change_left`: Lateral transition into the left lane while preserving forward progress.
- `lane_change_right`: Lateral transition into the right lane while preserving forward progress.
- `gentle_curve`: Route tracking on a gradually curving road segment.

## Open-Loop Metrics
- ADE: 0.877
- FDE: 1.474
- Route Error: 0.672

## Resume Bullets
- Built a route-conditioned autonomous driving planner around a conditional diffusion policy.
- Implemented route-prior residual diffusion with a conditional 1D U-Net decoder and iterative denoising sampler.
- Designed structured synthetic scenarios and open-loop metrics to demonstrate planning behavior without proprietary data.

## Artifacts
- Checkpoint: `outputs/portfolio_demo/demo_checkpoint.pt`
- Predictions: `outputs/portfolio_demo/predictions.pt`
- Plot: `outputs/portfolio_demo/prediction_plot.png`
- JSON Summary: `outputs/portfolio_demo/portfolio_summary.json`
- Markdown Summary: `outputs/portfolio_demo/portfolio_summary.md`
