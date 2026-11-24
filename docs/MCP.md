# RGB MoCap MCP toolkit

The toolkit exposes the managed Windows CLI and can open a desktop window for
interactive work. It communicates with the connected MCP client over local
standard input and output.

Install RGB MoCap normally. In a client that supports local stdio servers, set
the executable to the installation's `runtime\python.exe` and pass
`-I -X utf8 -u -B <installation>\tools\rgbmocap_mcp.py` as separate arguments.
The included `RGBMoCapMCP.cmd` provides the same entry point in CMD.
Restart the MCP session after updating the application source.

This example registers the toolkit through Codex CLI.

```powershell
$rgbMoCapInstall = (Get-Location).Path
codex mcp add rgbmocap -- (Join-Path $rgbMoCapInstall 'runtime\python.exe') -I -X utf8 -u -B (Join-Path $rgbMoCapInstall 'tools\rgbmocap_mcp.py')
```

## Run workflows

Use the tool names returned by tools/list. CLI tools derive their arguments
from the [command line](CLI.md) and return a job ID. `job_list` shows active
and recent jobs from the desktop, CLI and MCP in the same installation.
`job_status` reads the latest phase, result or error by ID, including after
reconnecting. `job_cancel` requests cancellation of a desktop or MCP job.

MCP commands that write are run in submission order and wait for the shared
installation lock if the desktop or another process holds it. Read-only
checks can run while a write is waiting. Closing the MCP connection cancels
its pending and active CLI commands and preserves their records. An owner
that exits unexpectedly leaves records marked interrupted when read again.

Examples include `models_list`, `project_inspect`, `reconstruct`, `correct`,
`contacts_self_set`, `character`, `preview`, `export`, `stream`, and `pear`.
For a generic native FBX, pass `format: "fbx"` and `fbx_profile: "generic"` to `export`.
Use an explicit project directory and a new destination file.

Use `unreal_profile` to save a local target profile, and `unreal_import` to
check or import a list of animation FBX files for that profile. The `export`
tool also accepts `unreal_profile` as a path to import the outputs. See
[Unreal animation import](CLI.md#import-animations-into-unreal).

## Operate the desktop

Save and close any existing RGB MoCap window before `desktop_start`. The
toolkit launches a window with the same application lock used by the desktop
and mutations through the CLI. Pass an optional source video or project.

1. Read `desktop_state` and `desktop_controls` to find the active project, frame,
   menu commands, editable controls and modal dialogs.
2. Use `desktop_invoke` with the returned control ID. For a choice, tab or list
   use the zero-based index shown by the control.
3. Use `desktop_seek` for a one-based timeline frame. Use `desktop_capture`,
   `desktop_pointer`, `desktop_drag` and `desktop_key` for the viewport and
   timeline gestures. Refresh the control list after opening a dialog or changing pages.
4. Use `desktop_stop` to close the window. Any normal save prompt remains active.

The window handles sequence review, character jobs, contact editing, setup dialogs,
PEAR and the Stream page through their normal controls. Model downloads,
installs, replacement and removal still follow the application's workflows.
CLI commands that change project data use the installation lock while running.
