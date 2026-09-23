# Configuration Reference

Task files recursively override `defaults.yaml`, followed by the selected
`schemes` override. Explicit command-line options take precedence. Each run
stores the fully resolved configuration. Unknown fields, duplicate YAML keys,
and incompatible settings are rejected.

## Configuration Groups

| Group | Purpose |
| --- | --- |
| `model` | Proximal weights, architecture, and inference precision |
| `operator` | Observation noise, task geometry, kernels, and assets |
| `sampler` | Discretization, initialization, annealing, guidance, and readout |
| `consistency` | Amplitude projection, observed-pixel restoration, and in-chain fitting |
| `refinement` | Regularized optimization of the final image |
| `seed` / `seed_step` | Sampling seed and per-image offset |
| `observation_seed` / `observation_seed_step` | Independent observation seed and offset |

## Sampling

`schedule` generates time points and integer inner-step counts from the budget,
endpoints, stage count, and allocation rule. `budget` includes the final proximal
readout when enabled. `spacing` controls time spacing. `allocation` supports
uniform, head, tail, middle, ends, and focused allocation; `power`, `center`, and
`width` control its shape. `warmup_end` and `warmup_fraction` define a piecewise
time schedule.

`likelihood` accepts a constant or an endpoint curve with optional peak and
progress controls. `likelihood_mode` supports direct, inverse-lambda, and
displacement-based coefficients. `delta_mode` parameterizes the step through
lambda, squared lambda, an absolute value, or the linear drift scale.
`delta_scale` and `delta_cap` set its scale and upper bound.

`init` selects backprojection, a zero image, a Gaussian image, or a perturbed
observation. `init_filter_sigma`, `init_filter_gain`, `init_gain`, and
`init_noise` control filtering, scaling, and matched noise. Additional
initialization noise uses a separate random stream from the sampling chain.

`data_direction` chooses the raw gradient or an adaptive direction.
`data_location` selects the chain state, the proximal output, or a composite
likelihood gradient. `preconditioner` controls the fixed spectral metric.
`guidance_rms_cap` and `guidance_pixel_cap` bound data displacements.
`temperature`, `temperature_mode`, and `temperature_power` control stochastic
forcing. `confinement` controls the additional Partial drift term. The
configuration validator checks compatibility between these options.

`readout` selects a final proximal map, the chain state, or an average of recent
proximal estimates. `final_t` sets the final evaluation time. `clamp` controls
chain-state clipping.

## Image Refinement

`refinement.optimizer` selects Adam or L-BFGS. `steps` controls outer iterations;
disabling refinement returns the sampling estimate directly. `lr` sets the
learning rate. `data`, `lpips`, and `tv` weight the objective terms.
`tv_epsilon` controls TV smoothing, `history_size` controls L-BFGS memory, and
`max_evaluations` bounds objective evaluations. Refinement never calls the
proximal network.

## Resource Paths

`model.checkpoint` selects the proximal weights. `model.architecture` specifies
the architecture for a legacy state dictionary. Training exports embed their
architecture. Legacy weights default to the FFHQ architecture when no explicit
architecture is supplied. Conflicting embedded and explicit architectures are
rejected. `models` contains FFHQ and ImageNet definitions.

`operator.options` selects the nonlinear deblurring configuration, whose
`KernelWizard.pretrained` field specifies its weights. Relative resource paths
are resolved from the project root.

Nonlinear deblurring accepts an independent asset directory through
`operator.engine_dir`, `--nl-engine`, `PGM_NDB_ENGINE`, or `ndb_engine` in
`local.yaml`. The selected directory is searched first for the operator
configuration, with the project configuration as fallback. Relative weight
paths are resolved from the selected asset directory. KernelWizard code is
included in the project.

Use `--image-offset` when replaying a subset to preserve original image indices,
seeds, and masks. Training settings are stored in the `train` subdirectory.
