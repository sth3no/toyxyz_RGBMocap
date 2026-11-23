# macOS on Apple Silicon (experimental)

The macOS port installs and launches the app natively on Apple Silicon Macs.
It installs the GUI runtime, the ExMR runtime, the native motion correction
module and Blender 4.5 for export and character retargeting. **ExMR and person
tracking run on the Apple GPU (PyTorch MPS).** The other motion models need an
NVIDIA CUDA GPU and do not run on macOS. Keep the macOS checkout separate from
any Windows or Linux checkout; each platform has its own runtimes.

## What works

| Verified on the test Mac | Not yet |
| --- | --- |
| Installing the pinned shared, GUI and ExMR runtimes (Python 3.10.22 and 3.12.10, PySide6 6.8.3, PyTorch 2.7.1 and 2.8.0 CPU/MPS builds) | GVHMR with HTD-Refine, FootMR and HandFlow, FlowHMR, MAMMA, PromptHMR, GEM-X, SAM 3.1 Find person and PEAR need CUDA. Their runtimes are not installed; starting them reports which models are CUDA-only. |
| Launching the GUI on the managed runtime | ExMR's SAM 3D Body body features and the final ExMR motion stage have not run on MPS yet: their registered and gated files were not available on the test Mac (see [Model files](#model-files-for-exmr)). |
| **Find person** (YOLOv8x, BoT-SORT, OSNet) and single-actor tracking on MPS | Track Face, the Unreal Live Link sender and the other optional setups use Windows tooling. |
| ExMR keypoints (Sapiens2-1B, FP16) on MPS through the worker; HaMeR hand features (FP16 autocast) on MPS through the app's HaMeR adapter, with the registered MANO layer stubbed because `MANO_RIGHT.pkl` was absent (the compared token is read before MANO is used) | A complete ExMR reconstruction, correction, preview and export on macOS have not been tested end to end. |
| Native motion correction, built for arm64 during installation and checked against an x86 build on synthetic motion | |
| Installing Blender 4.5 (`--with-blender`) and finding it for retargeting | |

Auto Setup shows the Mac's memory and says that ExMR runs on the Apple GPU.
It does not recommend a mode because the other models need CUDA.

## ExMR on the Apple GPU

`mocap/compute_device.py` chooses CUDA when it is available, else MPS, else
the CPU. ExMR and person tracking accept MPS; on CUDA they run exactly as before.
MPS results have their own cache identity (device `Apple MPS <chip>`), so a
project moved between a Mac and a CUDA PC recomputes rather than mixing results.
Precision is the same as on CUDA: Sapiens2 FP16, HaMeR FP16 autocast, SAM 3D Body
BF16. MPS has no float64, so the two small float64 steps (Sapiens2's heatmap
refinement and the rotation smoothing) run on the CPU. Workers set
`PYTORCH_ENABLE_MPS_FALLBACK=1`, so an operator that MPS lacks runs on the CPU
instead of failing. SimpleVO (moving camera) uses OpenCV on the CPU, as on Windows.

### Measured speed and agreement

Test Mac: Apple M5, 24 GiB unified memory, macOS 27, PyTorch 2.8.0 (ExMR) and
2.7.1 (tracking). Other applications were open, so only about 5 GiB of RAM was
available and macOS was swapping; Automatic therefore selected Low Memory for
Sapiens2. These are single runs, not guarantees. The sample is
`docs/samples/motion-capture.mp4` (1920x1080, 690 frames at the 30 fps working rate).

| Stage | Apple GPU (MPS) | CPU (same Mac) | MPS vs CPU difference |
| --- | --- | --- | --- |
| Find person, 690 frames, two people, worker | 140.9 s (0.20 s/frame, including model loading) | 401 s | Same two people; boxes differ by at most 0.001 px (minimum IoU 0.999996) |
| Single-actor tracking, 690 frames, worker | 68.2 s | not run | |
| Sapiens2 keypoints, 60-frame take, worker, Low Memory, 5.0 GiB cap | 89.6 s stage, of which 48 s inference (0.81 s/frame) | 72.7 s/frame (4 frames) | Confident keypoints at most 0.057 px apart (mean 0.010 px; all keypoints at most 0.30 px; person about 370 px tall); scores at most 0.002 |
| HaMeR hand features, 8 frames with both hands | 0.155 s/frame | 9.2 s/frame | Token differences at most 0.009 (mean token magnitude 0.33); cosine similarity at least 0.99999 |

For comparison, the RTX 5090 in [the ExMR guide](models/EXMR.md#measured-effects)
needs about 1.5 s of warm Sapiens2 inference for 60 frames, so this Mac is roughly
30 times slower for that stage. Expect a full 60-frame ExMR reconstruction to take
several minutes; it has not been measured yet.

### Memory on unified memory

CPU and GPU share one memory pool, so the Apple GPU budget is Metal's
recommended working set (17.8 GiB on the 24 GiB test Mac) minus what the GPU
already holds, limited by the RAM that is actually available above the usual RAM
reserve. No separate GPU reserve is subtracted. With **Resource Safety** on, the
worker sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` and
`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8` and caps PyTorch at that budget, so a job
that does not fit stops with an out-of-memory error instead of swapping.
Automatic then retries once in Low Memory, as on CUDA. With Resource Safety off,
the app sets neither the watermarks nor a cap and PyTorch's own limits apply
(measured MPS driver allocation after Sapiens2 on the 60-frame take: 6.5 GiB instead of 4.4 GiB).
Close other applications before long ExMR runs: free RAM is the limit on most Macs.

## Model files for ExMR

Use Model Manager as on Windows; the files go to the same places under `models/`.

- **Downloaded by the app** (public): ExMR weights `exmr/exmr.safetensors`
  (225 MiB) and `exmr/config.json`, Sapiens2 `exmr/sapiens2_1b_pose.safetensors`
  (5.7 GiB), HaMeR `handflow/hamer.ckpt` (2.5 GiB) and `handflow/mano_mean_params.npz`,
  YOLOv8x `yolo/yolov8x.pt` (131 MiB) and OSNet `reid/osnet_x0_25_msmt17.pth`
  (8.9 MiB). About 8.5 GiB in total. YOLOv8x and OSNet come from Google Drive,
  which sometimes refuses with "Too many users"; then use **Manual Download** and
  **Import**. On the test Mac the official `yolov8x.pt` from the Ultralytics
  `v8.1.0` release had the pinned SHA-256 and imported.
- **Registration required, obtain yourself and Import**: SMPL-X v1.1
  `SMPLX_NEUTRAL.npz` (smpl-x.is.tue.mpg.de), MANO `MANO_RIGHT.pkl`
  (mano.is.tue.mpg.de) and the gated `facebook/sam-3d-body-dinov3` files
  `model.ckpt`, `model_config.yaml` and `assets/mhr_model.pt` (accept the terms on
  Hugging Face). Without them ExMR stops before body features with the list of
  missing files.

## Requirements

- An Apple Silicon Mac (M1 or newer). Intel Macs are refused by the installer.
- macOS 14 Sonoma or newer. The SciPy 1.14.1 wheel needs macOS 14. Tested on macOS 27 with an M5.
- The Xcode Command Line Tools, for Git, clang and the motion correction build:
  `xcode-select --install`.
- [Homebrew](https://brew.sh) with CMake 3.24 or newer and FFmpeg:
  `brew install cmake ffmpeg`.
- About 7 GiB of free space during installation (the installer checks 6.8 GiB).
  The runtimes download 0.89 GiB and Blender 0.3 GiB. On the test Mac the
  result was `runtime/` 1.0 GiB, `runtime_gui/` 1.1 GiB, `runtime_exmr/` 1.0 GiB,
  Blender 0.8 GiB and the reusable download cache in `.setup/`. ExMR's public
  model files need about 8.5 GiB more, plus the registered files; projects need
  additional space.
- A fresh `./install.sh` (shared, GUI and ExMR runtimes, without Blender) took
  4 min 49 s on the test Mac, including all downloads and the native build.
  Replacing only the ExMR runtime after a lock change took 1 min 9 s.

## Install, launch and update

```sh
git clone --branch macos-port --single-branch https://github.com/toyxyz/toyxyz_RGBMocap.git
cd toyxyz_RGBMocap
./install.sh --with-blender
./RGBMoCap.sh
./update.sh
```

In Finder you can instead double-click `install.command`, `RGBMoCap.command`
and `update.command`. Each one opens Terminal and runs the matching `.sh`
script. The install and update windows wait for Return before they close.
The launcher window waits only after an error. Keep the launcher window open
while the app runs: it shows launch errors and the app's console output.

The installer downloads the pinned uv for `aarch64-apple-darwin` and checks its
hash. It then installs private CPython 3.10.22 under `.setup/python/` and the
hash-pinned shared and GUI wheels into `runtime/` and `runtime_gui/`. It builds
native motion correction with the private Python and verifies everything before
it promotes the new runtimes. `--with-blender` copies the official
`blender-4.5.0-macos-arm64.dmg` (size and SHA-256 pinned in
`installer/macos-blender-lock.json`) to `runtime_character/blender-macos/Blender.app`.
It mounts the image read-only, with no Finder window, and runs
`Blender --version` to verify the copy.

`update.sh` fast-forwards a clean local `macos-port` branch that tracks the
remote `macos-port` and then repairs the runtimes. Use `./install.sh --verify-all`
for a full runtime check. Installation, repair and update preserve `models/`,
`projects/`, `cache/` and `config/`.

To use another Blender 4.5 LTS, for example one in `/Applications`, set
`RGBMOCAP_BLENDER` to its `Blender.app` or to `Blender.app/Contents/MacOS/Blender`:

```sh
RGBMOCAP_BLENDER=/Applications/Blender.app ./RGBMoCap.sh
```

## Gatekeeper and quarantine

A Git clone is not quarantined, so the `.command` files and the private Python
run without prompts. The installer fetches uv, Python, the wheels and Blender
with `curl`, uv and Python, which do not set the quarantine flag. The Blender
build is Apple-notarized.

If you downloaded the source as a ZIP in a browser, macOS quarantines every
file. Double-clicking a `.command` file is then blocked as coming from an
unidentified developer. Either Control-click it and choose **Open** once, or
remove the flag from the folder you unpacked:

```sh
xattr -dr com.apple.quarantine /path/to/toyxyz_RGBMocap
```

## Troubleshooting

**`Install FFmpeg first (for example: brew install ffmpeg)`**, or **Video tools
are missing**. Install FFmpeg with Homebrew. The launcher adds `/opt/homebrew/bin`
to `PATH` itself, so this also works when you open the app from Finder. A pinned
FFmpeg build may replace the Homebrew requirement later.

**`Motion correction build failed; see .../build.log`**. Check `cmake --version`
(3.24 or newer) and `clang++ --version`, then rerun `./install.sh`. The first
build fetches pybind11, Eigen and the pinned `sse2neon.h` from GitHub/GitLab.
For an offline build, place them under `.build-inputs/native-deps/`
(`pybind11/`, `eigen-3.4.0/`, `sse2neon/sse2neon.h`).

**`Could not find the Qt platform plugin "cocoa"`**. Qt does not list files
that carry the macOS *hidden* flag (`UF_HIDDEN`, shown by `ls -lO`). Some sync
or backup tools set this flag on everything inside dot-folders. One example is
a checkout under `~/Documents/.../.something/`. The launcher clears the flag on
the PySide6 plugin files before Qt starts. If the error persists, move the
checkout out of hidden folders, or run
`chflags -R nohidden runtime_gui/PySide6/Qt/plugins`.

**`This folder contains a Linux RGB MoCap installation`**. Use a separate
checkout for each platform. Do not copy `runtime*/` or `.setup/` between machines.

**Blender verification failed**. Run
`runtime_character/blender-macos/Blender.app/Contents/MacOS/Blender --version`.
If an older unmanaged folder was preserved, move it away and rerun
`./install.sh --with-blender`, or point `RGBMOCAP_BLENDER` at your own Blender 4.5.

## Maintainer notes

`installer/runtime-lock-macos.json` is generated by `tools/macos_lock.py` from
the Windows lock: the same versions, with Apple Silicon wheels. CUDA local
versions such as `torch 2.7.1+cu128` become the public PyPI build `2.7.1`.
`mocap/runtime_registry.py` accepts that build on macOS only. Regenerate the
lock after changing the Windows lock or the registry:

```sh
uv run --no-project --python 3.10 --with packaging python tools/macos_lock.py --uv-sha256 <sha256 of the extracted aarch64-apple-darwin uv>
```

The lock converts the `shared`, `gui` and `exmr` profiles. `UNLOADABLE` in
`tools/macos_lock.py` lists wheels that install but do not load on current macOS;
the next best wheel of the same version is used instead. SciPy 1.15.2's
`macosx_14_0_arm64` wheel is listed: dyld on macOS 27 rejects its gfortran-built
`_spropack` ("`__DATA/__thread_bss` has a zero-fill section type"), which breaks
`import scipy.spatial`. Its `macosx_12_0_arm64` wheel loads.

The motion correction source compiles its SSE code on ARM64 through
[sse2neon](https://github.com/DLTcollab/sse2neon) v1.8.0, pinned in
`third_party/KimodoMotionCorrection/CMakeLists.txt`. x86 builds are unchanged.
