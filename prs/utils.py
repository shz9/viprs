import os
import glob
import subprocess


def delete_temp_files(prefix):
    for f in glob.glob(f"{prefix}*"):
        try:
            os.remove(f)
        except Exception as e:
            continue


def run_shell_script(cmd):

    result = subprocess.run(cmd, shell=True, capture_output=True)

    if result.stderr:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=result.args,
            stderr=result.stderr
        )

    return result

