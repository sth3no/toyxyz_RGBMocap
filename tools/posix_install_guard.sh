# Sourced by the Linux installer and updater before they modify setup state.
rgbmocap_require_posix_install() {
    for marker in runtime/python.exe runtime_exmr/python.exe .setup/bootstrap-git/uv.exe; do
        if [ -e "$1/$marker" ] || [ -L "$1/$marker" ]; then
            echo "This folder contains a Windows RGB MoCap installation. Use a separate Linux checkout with its own runtimes." >&2
            return 1
        fi
    done
    # Each POSIX platform needs its own checkout: the private CPython and wheels are platform builds.
    case "$(uname -s)" in
        Darwin) other=linux-x86_64-gnu name=Linux ;;
        *) other=macos-aarch64-none name=macOS ;;
    esac
    for marker in "$1"/.setup/python/cpython-*-"$other"; do
        if [ -e "$marker" ]; then
            echo "This folder contains a $name RGB MoCap installation. Use a separate checkout with its own runtimes." >&2
            return 1
        fi
    done
    return 0
}
