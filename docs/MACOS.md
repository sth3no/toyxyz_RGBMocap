# macOS on Apple Silicon (experimental, phase 1)

The macOS port installs and launches the app natively on Apple Silicon Macs.
This first phase installs the GUI runtime, the native motion correction module
and Blender 4.5 for export and character retargeting. **AI motion inference
does not run on macOS yet.** Running the models on Apple GPUs (MPS)
is the next phase of the port. Keep the macOS checkout separate from any
Windows or Linux checkout; each platform has its own runtimes.

## What works in phase 1

| Verified on the test Mac | Not yet |
| --- | --- |
| Installing the pinned shared and GUI runtimes (Python 3.10.22, PySide6 6.8.3, PyTorch 2.7.1 CPU/MPS build) | ExMR, FlowHMR, GVHMR, HaMeR and the other model runtimes (`flow`, `hand`, `htd`, `exmr`) are not installed, so the app cannot reconstruct motion. |
| Launching the GUI on the managed runtime | MAMMA, PromptHMR, GEM-X, SAM 3.1 and PEAR need CUDA and have no macOS path. |
| Native motion correction, built for arm64 during installation and checked against an x86 build on synthetic motion | Track Face, the Unreal Live Link sender and the other optional setups use Windows tooling. |
| Installing Blender 4.5 (`--with-blender`) and finding it for retargeting | The 3D viewport is being ported separately. Until that work lands, meshes and skeletons may not draw. |
| | Project, video import, export and retargeting workflows on macOS have not been tested end to end. Video import uses FFmpeg from Homebrew. |

Auto Setup shows the Mac's memory and says that inference is a later phase.
It does not recommend a mode because there is no CUDA GPU.

## Requirements

- An Apple Silicon Mac (M1 or newer). Intel Macs are refused by the installer.
- macOS 14 Sonoma or newer. The SciPy 1.14.1 wheel needs macOS 14. Tested on macOS 27 with an M5.
- The Xcode Command Line Tools, for Git, clang and the motion correction build:
  `xcode-select --install`.
- [Homebrew](https://brew.sh) with CMake 3.24 or newer and FFmpeg:
  `brew install cmake ffmpeg`.
- About 6 GiB of free space during installation (the installer checks 5.7 GiB).
  The runtimes download 0.64 GiB and Blender 0.3 GiB. On the test Mac the
  result was `runtime/` 1.0 GiB, `runtime_gui/` 1.1 GiB, Blender 0.8 GiB and the
  reusable download cache in `.setup/` 1.3 GiB. Models and projects need
  additional space.
- A fresh `./install.sh --with-blender` took 3 min 40 s on the test Mac,
  including all downloads and the native build. A rerun with nothing to
  repair took 27 s.

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

The motion correction source compiles its SSE code on ARM64 through
[sse2neon](https://github.com/DLTcollab/sse2neon) v1.8.0, pinned in
`third_party/KimodoMotionCorrection/CMakeLists.txt`. x86 builds are unchanged.
