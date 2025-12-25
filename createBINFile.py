from __future__ import annotations

import subprocess
from pathlib import Path


class MissionResaverError(RuntimeError):
    pass


def find_mission_resaver(sim_root: str | Path) -> Path:
    """
    Matches the Java logic:
      sim_root\\bin\\resaver\\MissionResaver.exe
    """
    sim_root = Path(sim_root)
    resaver_exe = sim_root / "bin" / "resaver" / "MissionResaver.exe"
    if not resaver_exe.exists():
        raise MissionResaverError(f"MissionResaver.exe not found at: {resaver_exe}")
    return resaver_exe


def build_binary_mission(
    sim_root: str | Path,
    mission_file: str | Path,
    timeout_minutes: int = 5,
) -> Path:
    """
    Calls MissionResaver.exe to generate the binary mission file (.msnbin).

    Java equivalent command shape:
      "MissionResaver.exe"  -t  -d "SIM_ROOT\\data\\"  -f "....\\name.mission"

    Output is typically created next to the .mission file with the same base name:
      name.msnbin
    """
    sim_root = Path(sim_root)
    mission_file = Path(mission_file)

    if not mission_file.exists():
        raise FileNotFoundError(f".mission file not found: {mission_file}")

    resaver_exe = find_mission_resaver(sim_root)
    working_dir = resaver_exe.parent
    data_dir = sim_root / "data"

    cmd = [
        str(resaver_exe),
        "-t",
        "-d", str(data_dir) + "\\",  # keep trailing slash like the Java version (Windows-style)
        "-f", str(mission_file),
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise MissionResaverError(f"Timed out after {timeout_minutes} minutes running: {cmd}") from e

    if completed.returncode != 0:
        raise MissionResaverError(
            "MissionResaver.exe failed.\n"
            f"Command: {cmd}\n"
            f"Return code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}\n"
        )

    msnbin_path = mission_file.with_suffix(".msnbin")
    if not msnbin_path.exists():
        # Some resaver versions can place output differently; this is the common case.
        raise MissionResaverError(
            "MissionResaver.exe reported success but .msnbin was not found where expected:\n"
            f"Expected: {msnbin_path}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}\n"
        )

    return msnbin_path


# Example:
# sim_root = r"C:\Program Files\IL-2 Sturmovik Great Battles"
# mission_path = r"C:\Program Files\IL-2 Sturmovik Great Battles\data\Missions\MyMission.mission"
# msnbin = build_binary_mission(sim_root, mission_path, timeout_minutes=10)
# print("Created:", msnbin)
