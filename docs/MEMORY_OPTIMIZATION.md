# Memory use on long clips

ExMR evaluates long-sequence attention in batches of 256 queries. Each batch
uses every key allowed by the model's temporal window and retains the absolute
frame position. The complete sequence still runs through each layer. The
model shares canonical FP32 rotary tables within one model lifetime.

For moving-camera ExMR, SimpleVO decodes sampled frames as it uses them. It
keeps tracking on the canonical video with the existing pixel conversion,
frame sampling, matching and interpolation. The frame reader closes when the
operation ends.

The retarget solver similarly reduces intermediate contact surfaces in
bounded frame batches. HTD keeps gradients during optimization and runs the
subsequent IK postprocessing without recording gradients.

The archived ExMR source and backend identities include these changes.
Projects with older receipts may need to rebuild the affected ExMR stages.
Existing project media and result files stay in place. Long clips still
require memory for their features, model activations, trajectories and
exports.

The source adaptation and its component measurements are documented in
[jimmyshuixin's fork](https://github.com/jimmyshuixin/toyxyz_RGBMocap/blob/9cbdfef/docs/MEMORY_OPTIMIZATION.md).
The [validation record](VALIDATION.md) describes checks on this integration.
For resource limits and part sizing, see [Resource Safety](PERFORMANCE.md#resource-safety).
