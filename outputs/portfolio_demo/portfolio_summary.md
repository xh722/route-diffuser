# RouteDiffuser

> Autonomous driving trajectory planning with a route-conditioned diffusion policy

A planner-core autonomous driving project that demonstrates canonical scene modeling, route-conditioned diffusion inference, and end-to-end debug artifacts without relying on proprietary logs.

## Highlights
- Canonical scene schema for ego, neighbors, lanes, route polylines, and masks.
- Route-prior residual diffusion with a conditional 1D U-Net decoder.
- Optional multi-resolution pyramid noise inspired by a larger reference diffusion planner.
- Heuristic candidate ranking over route adherence, clearance, and comfort instead of defaulting to the first sample.
- Structured synthetic driving scenarios spanning keep-lane, lane changes, and curves.
- End-to-end scripts for training, inference, scenario-level evaluation, and portfolio artifact generation.

## Scenario Coverage
- `keep_lane`: Stable forward planning on a straight lane centerline.
- `lane_change_left`: Lateral transition into the left lane while preserving forward progress.
- `lane_change_right`: Lateral transition into the right lane while preserving forward progress.
- `gentle_curve`: Route tracking on a gradually curving road segment.

## Open-Loop Metrics
- ADE: 1.305
- FDE: 1.724
- Route Error: 1.061
- Progress: 37.403
- Min Clearance: 2.263
- Collision Rate: 0.375
- Comfort Violation Rate: 1.0

## Candidate Set Metrics
- Oracle ADE: 1.201
- Oracle FDE: 1.036
- Oracle Route Error: 0.942
- Final-State Diversity: 1.908

## Scenario Breakdown
- `gentle_curve`: ADE 1.52, FDE 2.815, Route Error 1.045
- `keep_lane`: ADE 1.211, FDE 1.306, Route Error 1.085
- `lane_change_left`: ADE 1.247, FDE 1.36, Route Error 1.076
- `lane_change_right`: ADE 1.243, FDE 1.415, Route Error 1.039

## Selection Strategy
- Strategy: `heuristic_route_clearance_comfort_scoring`
- Candidate Samples: 3
- Mean Selected Index: 1.052
- Mean Selected Score: 27.579

## Resume Bullets
- Built a route-conditioned autonomous driving planner around a conditional diffusion policy.
- Implemented route-prior residual diffusion with a conditional 1D U-Net decoder and iterative denoising sampler.
- Added multi-sample candidate scoring, scenario-level evaluation, and multi-resolution diffusion noise inspired by a larger reference planner stack.

## Artifacts
- Checkpoint: `outputs/portfolio_demo/demo_checkpoint.pt`
- Predictions: `outputs/portfolio_demo/predictions.pt`
- Plot: `outputs/portfolio_demo/prediction_plot.png`
- Candidate Plot: `outputs/portfolio_demo/candidate_trajectories.png`
- Scenario Gallery: `outputs/portfolio_demo/scenario_gallery.png`
- JSON Summary: `outputs/portfolio_demo/portfolio_summary.json`
- Markdown Summary: `outputs/portfolio_demo/portfolio_summary.md`
