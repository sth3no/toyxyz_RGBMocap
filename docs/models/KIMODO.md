# Optional Kimodo automatic fill

Kimodo is the generator behind **Automatic** in **Hidden ranges**. It makes up
the motion of a person during a range in which they were not tracked, from the
captured poses just before and just after it. It is not a motion model for
reconstruction: ExMR or GVHMR still reconstruct the takes, and Kimodo only
fills the gap between two of them. The rest of the workflow is described in
[Hidden ranges](../USER_GUIDE.md#hidden-ranges).

## What to expect

**Automatic produces plausible motion, not what the person did.** The range is
generated; nothing of it was seen. In the one comparison made so far, a group
dance with fully seen dancers and ranges of 46, 90 and 150 frames hidden on
purpose, the generated motion was further from the real motion than a plain
blend of the two end poses (mean pose error 10.6 cm against 9.0 cm). It was a
person shifting weight and moving the arms, not a dance. Automatic is meant for
ranges in which a person keeps moving or travels, for example while walking
behind an object. That use has not been measured with real footage yet, so
treat the result as a draft to review. For dancers who do the same moves,
**Copy from person** is usually closer to the truth.

The captured poses on both sides are kept: the generated frames meet them to
about 2 to 3 cm on average in the tests made, and the Console reports the
figure for each range. The joined take marks the generated frames on the
timeline, and export counts them as generated.

## Setup

1. Run `install.cmd` in the app folder. Close RGB MoCap, then run this command
   from the same folder and review the upstream terms when prompted.

   ```cmd
   Setup_Kimodo.cmd
   ```

   Setup needs Git and internet access. It reuses the installed ExMR Python
   and Torch and adds a separate `runtime_kimodo/` overlay of about 15 MB. It
   fetches Kimodo's code from GitHub at a pinned revision
   (`e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954`) and six small packages from
   PyPI with pinned hashes, then checks that Kimodo's skeleton matches the
   conversion and that CUDA works. Robot meshes and demo files are left out.

2. Reopen RGB MoCap, select **GVHMR** or **ExMR**, and open **Model Manager**.
   Click **Auto Download** on the **Kimodo** rows: eight files of
   `nvidia/Kimodo-SOMA-RP-v1.2` at a pinned revision, **1.13 GB** in total
   (the model itself is 1.13 GB; the other seven are a few kilobytes). The rows
   appear on the **GVHMR** and **ExMR** tabs and are checked by SHA-256.
   **Download All** skips them. **Import...** works on each row for files you
   downloaded yourself; the default location is `models/kimodo/Kimodo-SOMA-RP-v1.2/`.

3. Open a project with hidden ranges and choose **Automatic** for a range.
   Until both parts are in place, **Automatic** is greyed out and its tooltip
   says which one is missing.

Re-running `Setup_Kimodo.cmd` keeps a runtime whose receipt is valid. To
rebuild a damaged one, close the app, delete the `runtime_kimodo` folder and run
the setup again. Each reinstall over an existing folder leaves a backup folder
under `.setup`, which you may delete.

## Fill a hidden range

1. Choose **Automatic** in the row of the range. **Variation** is a number from
   0 to 999. Another number gives another motion for the range; the same
   number gives the same motion again, because generated ranges are kept in the
   folder `auto_fill` inside the person's preview folder and served from there.
2. Click **Fill hidden ranges**. The first range of a run takes about 10 to 15
   seconds on an RTX 5060 Ti including starting the worker (12.6 seconds for 70
   frames, measured once); a range that was generated before comes from the
   cache without starting the worker.
3. Click **Apply Correction** for the joined take, as for any joined take.
   Other people of the scene ask for **Apply Correction** again, as after the
   other methods.

Changing **Variation** and filling again generates the range again and writes
a new joined take. A new Kimodo runtime or model version does not make an
existing joined take out of date; fill with another **Variation** to regenerate.

## Limits of this version

- GVHMR, and ExMR with **Static Camera**; the `initial` and `footmr_initial`
  results and their corrected variants. FlowHMR, PromptHMR, MAMMA and GEM-X
  projects are not supported.
- 30 fps takes only, and at most **290 frames** (9.7 seconds) per range. With
  five known frames on each side that is the generator's 300-frame limit.
- Fingers are not generated. They blend between the two sides.
- There is no text prompt in this version. Kimodo is given the poses on both
  sides and nothing else.
- It needs an NVIDIA GPU with about **1.2 GB** of free video memory on top of
  what is in use. If the memory is not there, the fill stops with a message and
  nothing is written; close other GPU work and try again.
- Kimodo's own post-processing is not used.

## How it works

Five known frames on each side of the range (fewer when a take is shorter) are
converted from SMPL-X to Kimodo's SOMA skeleton and given to Kimodo as
constraints, together with a smoothed root path read from up to 30 frames next
to the range. The floor is read from up to 45 captured frames on each side and
joined by a straight line across the range, so a drifting capture height does
not reach Kimodo. Kimodo denoises for 100 steps with the seed from
**Variation**. The result is converted back to SMPL-X, and its foot contacts
become the contact probabilities of the range. The conversion code is in
`mocap/soma_convert.py`, the worker in `mocap/kimodo_fill.py` and the cache in
`mocap/auto_fill.py`. The generator runs in its own process
(`kimodo_worker_boot.py`) in the ExMR Python with the `runtime_kimodo/`
overlay; the app itself imports none of it.

## Licences

- Kimodo code is Apache-2.0 (NVIDIA). The app does not bundle it; the setup
  fetches it at the pinned revision. The fetched package also contains
  MIT-licensed LLM2Vec files for Kimodo's text encoder. They are not loaded,
  because this version sends no text, and the setup keeps Kimodo's licence and
  attributions in `runtime_kimodo/` as `KIMODO_LICENSE` and
  `KIMODO_ATTRIBUTIONS.MD`.
  [Kimodo licence](https://github.com/nv-tlabs/kimodo/blob/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954/LICENSE)
- The Kimodo-SOMA-RP model is under the NVIDIA Open Model License. You download
  it from Hugging Face in Model Manager; the app does not redistribute it.
  [Model terms](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/blob/0e3c89024d896064b6851201373f7834784a386b/LICENSE)
- The vendored `third_party/KimodoMotionCorrection` contact-correction backend
  is a separate use of the same project and is unchanged.

## Troubleshooting

- If the tooltip of **Automatic** says `Close RGB MoCap and run
  Setup_Kimodo.cmd to prepare automatic fill.`, close the app and run the setup.
  If it says `Get the Kimodo model in Model Manager to use automatic fill.`,
  download or import the eight rows in Model Manager. A row marked **Missing** after a copy should
  be imported again.
- `Automatic fill requires a CUDA GPU.` means the worker could not use an
  NVIDIA GPU. `Not enough free video memory for automatic fill.` means other
  programs hold the memory.
- Videos and poses are processed locally; nothing is sent anywhere. Only the
  setup and **Auto Download** use the internet.
