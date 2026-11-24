# Stream VMC output (experimental)

## Controls

Open the **Stream** tab and choose **Target > VMC**. Enter the receiving IP/UDP
port (default `127.0.0.1:39539`) and click **Start VMC Output**. Click **Stop VMC
Output** to stop. IP/port are editable when output is stopped.

For ExMR, FlowHMR, GVHMR, MAMMA and PromptHMR, prepare the chosen motion preview
and select one person in the Stream list. VMC follows the current timeline,
including pause/scrub, selected result and scene placement. GEM-X uses its prepared
Native77 motion with a dedicated mapping. Prepare an offline take before starting output.

PEAR uses the same Stream controls; Start Live Tracking remains separate. The
PEAR makes the Stream tab available for live output. It uses accepted live preview packets after
filtering/position lock/foot lock. Entering PEAR stops the previous stream; select
VMC and Start to begin the new source. Leaving PEAR or stopping live tracking
stops output. Blender streaming stays restricted to prepared offline motion.

VMC sends one actor per destination.
Checking a different actor replaces the previous selection. A missing take or
person sends unavailable status. Offline pause
keeps refreshing the current pose four times per second; playback
updates sample the current playhead. The send thread retains the latest small
pose and caches one rig. Settings are session-only.

## Skeleton contract

52 body/finger names, parent-local metre positions and XYZW quaternions in Unity
coordinates. Root is identity; Hips carries translation once. Prepared motion
preserves the original source bone axes and positions, including any source
bind-pose convention. Receivers need bind/retarget correction where their avatar
differs. Finger motion is present only to the extent the source model estimated
it; e.g. PromptHMR's neutral fingers do not become tracked fingers.

SOMA uses LeftLeg/LeftShin rather than Mixamo LeftUpLeg/LeftLeg, a three-bone
spine and extra neck/finger/end joints. The adapter remaps names and collapses
unmapped helpers through world transforms, preserving the mapped pivots.
Face/eye/jaw/terminal helpers are not sent. No source matrices are overwritten.

Each OSC bundle is at most 1400 bytes; a pose spans multiple datagrams. Delivery
is unacknowledged and partial UDP loss can leave a partial pose. `/VMC/Ext/OK 0`
is sent on loss/stop; receivers that ignore it may retain their last pose. This
feature does not guarantee accurate avatar retargeting or end-to-end latency.

## Verification

The feature comes from [hysaint3d's VMC implementation](https://github.com/hysaint3d/toyxyz_RGBMocap/commit/184a30a).
Its [PEAR proposal](https://github.com/toyxyz/toyxyz_RGBMocap/pull/88) records
reception in Operator V6. Tests in this distribution cover OSC and local UDP,
SMPL-family and SOMA mappings, scene transforms, and invalid inputs.
See the [VMC protocol](https://protocol.vmc.info/english.html).
