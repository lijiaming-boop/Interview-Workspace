import glob
import os
import subprocess
import sys

BASH = r"C:\Users\27013\.workbuddy\binaries\PortableGit\versions\1.2.0\bin\bash.exe"
if not os.path.exists(BASH):
    for cand in glob.glob(r"C:\Users\27013\.workbuddy\binaries\PortableGit\**\bash.exe", recursive=True):
        BASH = cand
        break

print("bash:", BASH)
print("=" * 70)

os.chdir(r"C:\Users\27013\Desktop\oft\10-INT8量化与并行解码教材")

env = dict(os.environ)
env["PATH"] = r"C:\Users\27013\.workbuddy\binaries\PortableGit\versions\1.2.0\usr\bin;" + env.get("PATH", "")

for f in sorted(glob.glob("*.sh")):
    p = subprocess.run(
        [BASH, "-n", f],
        capture_output=True, env=env,
    )
    out = (p.stdout + p.stderr).decode("utf-8", "replace").strip()
    status = "OK" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
    print(f"{f:32} {status}")
    if out:
        for line in out.splitlines():
            print(f"      {line}")
