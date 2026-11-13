#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
entry_dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$entry_dir"
cat > "$entry_dir/amsrr-task-console.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=A-MSRR Task Workspace
Comment=Structure generation and robot simulation
Exec="$repo_dir/scripts/task_desktop.sh"
Path=$repo_dir
Icon=applications-engineering
Terminal=false
Categories=Education;Science;Robotics;
EOF
printf '%s\n' "Installed: $entry_dir/amsrr-task-console.desktop"
