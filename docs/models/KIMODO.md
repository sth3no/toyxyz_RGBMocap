# Optional Kimodo automatic fill

**Experimental support. Generated motion is plausible, not what the person did.**

Kimodo is the generator behind **Automatic** in **Hidden ranges**. It makes up
the motion of a person during a range in which they were not tracked, from the
captured poses just before and just after it. It is not a motion model for
reconstruction: ExMR or GVHMR still reconstruct the takes, and Kimodo only
fills the gap between two of them. In the app the method is called **Automatic
(Experimental)**. It can be given a short **Description** of what the person
does. The rest of the workflow is described in
[Hidden ranges](../USER_GUIDE.md#hidden-ranges).

## What to expect

Nobody has looked at a described fill by eye yet; what is known about
descriptions is measured and listed under [Describe the range](#describe-the-range).

**Automatic produces plausible motion, not what the person did.** The range is
generated; nothing of it was seen. In the one comparison made so far, a group
dance with fully seen dancers and ranges of 46, 90 and 150 frames hidden on
purpose, the generated motion was further from the real motion than a plain
blend of the two end poses (mean pose error 10.6 cm against 9.0 cm). It was a
person shifting weight and moving the arms, not a dance. Automatic is meant for
ranges in which a person keeps moving or travels, for example while walking
behind an object. That use was measured on one clip: a walker covered for 91
frames (3 seconds) while he came 1.6 m closer. The two variations made walked
through the range with 5 and 3 foot strikes where the real walk has 5, and the
toes slid at 13 to 30 cm/s while low, where a blend of the two end poses slides
one frozen pose across the floor at 58 cm/s. The poses themselves were no
closer to the real walk than the blend's (7.1 and 7.8 cm against 7.8 cm), and
where the pelvis was in the room could not be checked on that clip. Treat the
result as a draft to review. For dancers who do the same moves, **Copy from
person** is usually closer to the truth.

The captured poses on both sides are kept, but the generated motion does not
meet them exactly. In the product runs made it missed the captured pose at the
frame next to the range by 1 to 8 cm and the five known frames on each side by
2 to 4 cm on average (a 46-frame range of a dance: 4 to 8 cm next to the range;
the 91-frame walking range: 3 cm before it and 1 to 2 cm after it);
the join then blends the difference out over the first and last frames of the
range. The Console reports the two figures next to the range for each range,
with a line such as `Frames 101–120 generated: the captured poses next to the
range were missed by 7.8 cm before it and 7.1 cm after it.` The joined take marks the
generated frames on the timeline, and export counts them as generated.

When the generated motion misses a captured pose next to the range by more
than 25 cm, the takes on the two sides do not fit together, for example because
one of them ends with poses reconstructed from a covered view. The fill is then
refused and nothing is written; an earlier joined take stays as it was. See
[Troubleshooting](#troubleshooting).

## Setup

**Automatic fill is Experimental.** The whole setup happens inside the app; RGB
MoCap stays open. A description is optional and needs two more groups of files
(step 3); without one, only the runtime and the motion model are needed.

1. Select **GVHMR** or **ExMR** and open **Model Manager** (or click **Set up…**
   under the hidden ranges of a person once **Automatic (Experimental)** is
   chosen, which opens it). Above the rows, click **Install Runtime** and
   confirm the question. It downloads Kimodo's code from GitHub at a pinned
   revision (`e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954`), twelve small
   packages from PyPI with pinned hashes and `antlr4-python3-runtime`, built
   from source with the runtime's own `setuptools` (among them `transformers` and `peft`,
   Apache-2.0, for descriptions) into a separate `runtime_kimodo/` folder of
   about 140 MB on top of the installed ExMR Python and Torch, then checks that
   Kimodo's skeleton matches the conversion, that CUDA works and that the text
   encoder's code imports. Robot meshes and demo files are left out. It needs
   Git, internet access and 1 GiB of free disk space, and runs as a job with
   progress in the status line (63 seconds in the measured run). The button
   then reads **Reinstall Runtime**: it checks the installed runtime again and
   keeps it when the check passes (the status line says `Automatic fill runtime
   verified.`); when the check fails it installs the runtime again (`Automatic
   fill runtime installed.`). A runtime installed by an earlier version of
   `Setup_Kimodo.cmd` lacks the text packages and counts as not installed:
   click **Install Runtime** again. The old folder is replaced only once the
   new runtime has been checked, and then removed.
2. The motion model. Kimodo needs eight files of `nvidia/Kimodo-SOMA-RP-v1.2`
   at a pinned revision, **1.13 GB** in total (the model itself is 1.13 GB; the
   other seven are a few kilobytes). Their rows appear on the **GVHMR** and
   **ExMR** tabs as one row, **Kimodo Motion Generation (Experimental,
   optional)**. **Auto Download** on it fetches all eight files, each checked
   by SHA-256. **Download All** skips it. If the download fails, download the
   eight files yourself into one folder, keeping their folders
   (`model.safetensors`, `config.yaml` and
   `stats/motion/<body|global_root|local_root>/<mean|std>.npy`), click
   **Import...** on the row and pick `model.safetensors`: the app imports all
   eight from that folder as it does the Llama files below. They go to
   `models/kimodo/Kimodo-SOMA-RP-v1.2/`.
3. For a description, two more rows:
   - **Kimodo Text Adapters (Experimental, optional)**, **345 MB**, eight
     files: the two LLM2Vec adapters `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`
     and `...-mntp-supervised` (MIT, public) with their configurations and the
     tokenizer, at pinned revisions. **Auto Download** fetches them (about 10
     seconds in the measured run). They live in `models/kimodo_text/mntp/` and
     `models/kimodo_text/supervised/`. If the download fails, download the six
     files of the `...-mntp` repository into one folder and the two of
     `...-mntp-supervised` into a folder named
     `LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised` (or `supervised`) next
     to it, click **Import...** and pick `adapter_model.safetensors` of the
     first folder: the app imports all eight.
   - **Llama 3 8B Instruct Weights for Descriptions (Experimental, optional)**,
     **16.06 GB**, four files `model-00001-of-00004.safetensors` to
     `model-00004-of-00004.safetensors` of `meta-llama/Meta-Llama-3-8B-Instruct`.
     The text encoder is Llama 3 with the two adapters, so it needs these
     weights. **Meta gates them**: they are under the Llama 3 Community
     License, which you accept on Hugging Face with your own account. The app
     cannot download them and never asks for or stores a token or password.
     Sign in at Hugging Face, open the model page, accept the licence, wait for
     access, download the four files into one folder, then click **Import...**
     on the row and pick **any one of them**: the app imports all four from
     that folder (21 seconds from a local folder in the measured run), checks
     each one's size and SHA-256 and publishes none of them unless all pass. It
     needs free disk space for the files still missing plus 256 MiB (about
     16.3 GB when all four are missing). Other Llama models, Llama 3.1 and
     the original `consolidated.00.pth` are not accepted. They go to
     `models/kimodo_text/llama/`.
4. Open a project with hidden ranges and choose **Automatic (Experimental)**
   for a range. If something is missing, the row says what and **Set up…**
   opens Model Manager. Before the runtime and the motion model are in place
   the range cannot be filled automatically; a description additionally needs
   the two rows of step 3.

`Setup_Kimodo.cmd` still installs the same runtime without the app: close RGB
MoCap, run it from the app folder and review the upstream terms when prompted.
It is the way to set up while the app is closed; the model rows are then
obtained in Model Manager as above.

   ```cmd
   Setup_Kimodo.cmd
   ```

Re-running the setup (**Reinstall Runtime**, or `Setup_Kimodo.cmd`) keeps a
runtime whose receipt is valid and that passes its check; a runtime that fails
the check is installed again. Staging folders of a cancelled setup are removed
by the next setup that installs a runtime.

## Fill a hidden range

1. Choose **Automatic (Experimental)** in the row of the range. **Variation** is
   a number from 0 to 999. Another number gives another motion for the range.
   Generated ranges are kept in the folder `auto_fill` inside the person's
   preview folder and served from there, so the same number gives the same
   motion again while the takes joined before the range stay the same. Filling
   or unfilling an earlier range of the same person changes the chain of
   joined takes: the later **Automatic** ranges are then generated again and
   may differ. Ranges generated before descriptions existed are generated
   again once, because the generator's code changed.
2. Optionally type a **Description** (see below).
3. Click **Fill hidden ranges**. On an RTX 5060 Ti a range of 46 frames took
   13 to 18 seconds and a range of 91 frames 21 to 24 seconds in the product
   joins, including starting the worker; a range of 290 frames, the longest,
   took about 40 seconds while other programs were using the GPU. A range that
   was generated before comes from the cache without starting the worker.
4. Click **Apply Correction** for the joined take, as for any joined take.
   Other people of the scene ask for **Apply Correction** again, as after the
   other methods.

Changing **Variation** or the **Description** and filling again generates the
range again and writes a new joined take. A new Kimodo runtime or model version
does not make an existing joined take out of date; fill with another
**Variation** to regenerate.

## Describe the range

The **Description** field under **Variation** says what the person does in the
range. Leave it empty and nothing changes: the range is generated exactly as
before descriptions existed, and none of the description files are needed.

**What to write.** One short action, in English, in the third person, as in
the field's placeholder `Optional, in English: A person waves with the right
hand.` Other examples: `A person raises both arms above the head.`, `A person
covers their face with one hand.` At most 200 characters, one line (line
breaks and repeated spaces become one space, control characters are dropped,
anything above 200 characters is cut). Other languages were not tried. Before
it is read, the description is tidied the way Kimodo tidies its own prompts:
the first letter is made a capital and every other letter small, and a final
period is added. Capitals inside the sentence therefore do not survive, and
neither do decimal numbers: `1.5` is read as `1. 5`, a sentence break. Write
plain words, such as "one and a half metres". When the tidied sentence differs
from what you typed, the Console line says what was read, for example
`Description read as "A person covers their face with one hand." in N
seconds; peak video memory X GiB.`

**What it costs.** The first time a description is used, a separate worker
reads it with the text encoder: 14 to 25 seconds, a peak of 1.7 GiB of video
memory, about 1.5 GiB of RAM and 3.65 GiB of committed memory (backed by the
page file) on an RTX 5060 Ti. It reads the 16 GB of Llama files from disk for
every new description; 14 to 25 seconds is the time from an NVMe drive, and a
slower disk takes longer. The encoder is Llama 3 8B Instruct with the two
LLM2Vec adapters. It does not fit this GPU whole, so the worker streams its 32
layers through the GPU one at a time, reading each from the four Llama files
and adding the adapters on the way. The result is a 4096-number vector for the
text, kept in `<data folder>/cache/kimodo_text/` for every project; the same
description is read again only when the text runtime, the text files or the
worker's code change (a description read before the tidying was added is read
again once). What was checked of that
vector: on the encoder cut to 4 of its 32 layers it matches what LLM2Vec's own
code computes (cosine 0.99997); at full depth it was checked only through its
effect on the generated motion. The worker computes in float32 from weights
rounded once to bfloat16, where Kimodo's own encoder computes in bfloat16. The
Console shows `Description read in N seconds; peak video memory X GiB.` Then
the range is generated with the text guided at weight 4 (the constraints keep
Kimodo's 2). The line for the range reads `Frames 101–120 generated from the
description "A person raises both arms above the head.": the captured poses
next to the range were missed by ... cm before it and ... cm after it.` The
joined take's `fill.json` records the description, the identity of the text
files used and the text worker's version.

**What it did.** Two kinds of runs were made. The worker's own test, on a
synthetic walking request of 60 frames: with `A person raises both arms above
the head.` the highest wrist rose 23 cm above the head; without a description
it stayed 59 cm below it. A run through the app's window, on a 91-frame range
of a real clip (a man walking towards a fixed camera, covered for 3 seconds),
with the descriptions typed into the field:

| Description typed | In the 91 frames | Seams before / after | Fill took |
| --- | --- | --- | --- |
| `A person raises both arms above the head.` | both wrists above the head in 55 frames, the highest 36 cm above it | 2.4 / 1.5 cm | 36 s |
| `a person covers their face with one hand` (read as `A person covers their face with one hand.`) | a wrist within 25 cm of the head in 48 frames, nearest 22 cm | 2.5 / 1.7 cm | 36 s |
| (none) | nearest wrist 77 cm from the head | 2.9 / 1.2 cm | 13 s |

The described fills include reading the description (14 seconds). These times
are with the GPU otherwise idle; while another program used it the same fills
took 57 to 60 seconds (20 seconds of it reading the description) and the plain
one 27 seconds. The
real man walks with his arms down, so the described fills are further from what
he did, as they must be: a mean pose error of 16.6 cm (arms raised) and 10.6 cm
(face covered) against 7.7 cm for the fill without a description. On this one
range the described fills also slid the toes 30 to 45 % more than the plain
fill.

In the trials made before (`experiments/hidden-person-fill/step-5-gate.md`), a
91-frame range of the walking clip with two seeds each, at weight 4:

| Description | Measure | No description | Weight 4 |
| --- | --- | --- | --- |
| A person covers their face with one hand. | nearest wrist to head | 77 cm | 20 to 22 cm |
| A person raises both arms above the head. | highest wrist above head | below | 27 to 36 cm above |
| A person walks while waving with the right hand. | highest wrist above head | below | 12 to 14 cm above |

The seams stayed at 2 to 3 cm before and 1 to 3 cm after the range at every
weight tried, and the pelvis speed stayed even across the range. Weights 6 and
8 gained little and moved joints faster, so 4 is used.

**What it did not do.** Single actions show; "first this, then that" mostly does
not. At weight 2 the compound `A person walks slowly and then covers their face
with one hand.` did nothing visible in a 3-second range (wrist 76 to 77 cm from
the head), and `A person sits down on the ground.` lowered the pelvis in one
seed of three; the pelvis is carried straight (see below), so an action that
fights that path or does not fit the range mostly loses. A description does not
make the motion closer to what the person did. Treat it as a way to ask for one
visible action in a gap, and review the result.

If the description files are missing, the row shows `A description needs the
Kimodo text adapters and the Llama 3 weights. Use Set up… to get them in
Model Manager.` and **Fill hidden ranges** is disabled until you do that or
clear the description.

## Limits of this version

- GVHMR, and ExMR with **Static Camera**; the `initial` and `footmr_initial`
  results and their corrected variants. FlowHMR, PromptHMR, MAMMA and GEM-X
  projects are not supported.
- 30 fps takes only, and at most **290 frames** (9.7 seconds) per range. With
  five known frames on each side that is the generator's 300-frame limit.
- Fingers are not generated. They blend between the two sides.
- A description is one short action, in English, third person, up to 200
  characters. Compound descriptions and actions that fight the straight path
  mostly did not show in a 3-second range, and nobody has looked at a described
  fill by eye yet. Capitals inside it and decimal numbers do not survive the
  tidying (see [Describe the range](#describe-the-range)).
- A range with a description cannot be filled again, for example with another
  **Variation**, once the Llama files or the text adapters were removed, even
  though its description was read before. Remove model files with **Model
  Manager** (**Delete Selected...** on the installed models): the app also keeps
  hard links of model files in other places, so deleting a file by hand may free
  no disk space.
- Without a description, Kimodo is given the poses on both sides and nothing
  else.
- It needs an NVIDIA GPU with about **2 GB** of free video memory on top of
  what is in use (reading a new description peaked at 1.7 GiB, and runs before
  and apart from the generator). Measured on an RTX 5060 Ti, the generator's peak was 1.17 GiB
  for a 290-frame range and 1.12 GiB for a 60-frame range; with **Resource
  Safety** on, another 600 to 700 MiB is kept free for other programs. If the
  memory is not there, the fill stops with a message and nothing is written;
  close other GPU work and try again. The worker's Console line gives the peak
  of each run.
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
overlay; the app itself imports none of it. A description is read by the same
boot script with the first argument `text` (`mocap/kimodo_text.py`) into a
vector that the generation worker takes as its text feature.

Kimodo is also asked to carry the pelvis in a straight line at constant speed
across the range, from where the person was last seen to where they are seen
again. A person who stopped, turned a corner or went around something behind
the cover is still moved straight. Without this path the generated person
tended to stand and then hurry. That was measured on one walking clip, a walker
hidden for 91 frames (3 seconds) while covering 1.6 m, with three variations:
without the path the median pelvis speed was 0.06 to 0.20 m/s in the first
third of the range and 0.60 to 0.79 m/s in the last, with 3 foot strikes where
the real walk has 5; with it the median speed stayed between 0.38 and 0.67 m/s
in every third, with 3 to 5 foot strikes, and the mean pose error against the
real walk was 7.2 to 7.9 cm (7.1 to 9.3 cm without it). The toes slid more
while low (19 to 22 cm/s against 5 to 16 cm/s); **Apply Correction** works on
the joined take afterwards. On a dance range with 0.18 m of travel the path changed
nothing measurable.

## Licences

- Kimodo code is Apache-2.0 (NVIDIA). The app does not bundle it; the setup
  fetches it at the pinned revision and keeps Kimodo's licence and attributions
  in `runtime_kimodo/` as `KIMODO_LICENSE` and `KIMODO_ATTRIBUTIONS.MD`.
  [Kimodo licence](https://github.com/nv-tlabs/kimodo/blob/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954/LICENSE)
- The Kimodo-SOMA-RP model is under the NVIDIA Open Model License. You download
  it from Hugging Face in Model Manager; the app does not redistribute it.
  [Model terms](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/blob/0e3c89024d896064b6851201373f7834784a386b/LICENSE)
- Descriptions use LLM2Vec (BehnamGhader et al., 2024; see
  [Citations](../CITATIONS.md#llm2vec-description-for-automatic-fill)): the
  fetched Kimodo package contains its MIT-licensed code (copyright McGill NLP),
  and the two adapters `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` and
  `...-mntp-supervised` are MIT and downloaded by Model Manager.
  `transformers` and `peft`, fetched by the setup with pinned hashes, are
  Apache-2.0.
- The Llama 3 8B Instruct weights are Meta's, under the
  [Meta Llama 3 Community License](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct),
  and gated on Hugging Face. You accept the licence and obtain the files
  yourself and import them; the app does not download or redistribute them, and
  it never reads, stores or sends a Hugging Face token or password.
- The vendored `third_party/KimodoMotionCorrection` contact-correction backend
  is a separate use of the same project and is unchanged.

## Troubleshooting

- If the row says `Automatic fill is not set up. Use Set up… and install the
  runtime in Model Manager.`, click **Set up…** and then **Install Runtime**
  (or run `Setup_Kimodo.cmd` with the app closed). If it says `Get the Kimodo
  model in Model Manager to use automatic fill.`, click **Auto Download** on the
  **Kimodo Motion Generation (Experimental, optional)** row, which fetches all
  eight files, or import them. A row marked **Missing** after a copy should be
  imported again.
- Installing the runtime failed with `Automatic fill setup could not download
  its files. Check the internet connection and try again.` (followed by the last
  line of the tool's output), `Git was not found. Install Git for Windows or run
  install.cmd again.` or `Not enough free disk space to set up automatic fill (1
  GiB needed).`: fix that and click **Install Runtime** (or **Reinstall
  Runtime**) again. Nothing of the old runtime is replaced until the new one
  has been checked. If an installed runtime is damaged, **Reinstall Runtime**
  finds it in its check and installs it again; nothing needs deleting by hand.
- Installing the runtime failed with `Automatic fill setup could not unpack the
  Kimodo source.`, `Automatic fill setup could not get or build
  antlr4-python3-runtime.` or `Automatic fill setup failed.`: these are not
  connection problems. The last line of the tool's output follows the sentence
  and says what went wrong; for the second, running `install.cmd` repairs the
  ExMR runtime that builds the package.
- `A description needs the Kimodo text adapters and the Llama 3 weights. Use
  Set up… to get them in Model Manager.`: **Auto Download** the **Kimodo Text
  Adapters** row and import the Llama weights. Importing the Llama weights can
  say `Choose one of the four model-0000N-of-00004.safetensors files.`, `... is
  missing next to the file you chose.`, `... is not complete (expected N bytes).`,
  `... is not the expected Llama 3 8B Instruct file.` or `Not enough free disk
  space for ... (X GiB needed).`; nothing new is kept after a refusal. Import
  on the other two rows refuses the same way (`... is not the expected
  Kimodo-SOMA-RP-v1.2 file.`, `... is not the expected LLM2Vec adapter
  file.`), names a missing file by its place relative to the file you chose
  (for example `stats/motion/body/mean.npy is missing next to the file you
  chose.`) and asks for the right file to pick (`Choose model.safetensors of a
  downloaded copy of nvidia/Kimodo-SOMA-RP-v1.2.`, `Choose
  adapter_model.safetensors in the folder of
  McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp.`).
- `The text encoder returned an empty result.` means the text worker produced a
  vector of zeros, which the generator cannot use; nothing is kept. Try again,
  and if it repeats, get the description files again in Model Manager.
- `The description files are not the expected version. Get them again in Model
  Manager.` means a Llama or adapter file does not read as the pinned one.
  `A description requires a CUDA GPU.` and `Not enough free video memory to read
  the description. Close other GPU work and try again.` are the text worker's
  versions of the two messages below. `The description is empty.` does not occur
  from the app: an empty field means no description.
- `Automatic fill could not connect the motion on both sides of frames
  137–166: the generated motion misses the captured poses next to the range by
  103 cm. The takes before and after it do not fit together. Check the frames
  next to the range, reconstruct those takes again, or choose Do not fill.`
  means that the generated motion missed a captured pose next to the range by
  more than 25 cm: the two takes do not fit together. In the case seen so far
  they placed the person 8.4 m apart and turned by about 135 degrees across a
  one-second range, because the take before it ended with frames reconstructed
  from a partly covered view. Look at those frames, reconstruct those takes
  again, or choose **Do not fill** for the range. Nothing is written, and an
  earlier joined take stays as it was. The generated range is kept in the
  cache, so filling again with the same takes refuses at once.
- `Automatic fill requires a CUDA GPU.` means the worker could not use an
  NVIDIA GPU. `Not enough free video memory for automatic fill.` means other
  programs hold the memory.
- Videos and poses are processed locally; nothing is sent anywhere. Only the
  setup and **Auto Download** use the internet.
