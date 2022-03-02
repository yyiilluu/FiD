#!/usr/bin/env python
"""
python scripts/sync.py -u yilu -o dpr-training-2.us-west1-b.training-pipeline-274108
"""
import argparse
import subprocess
from typing import Dict


def get_remote_dir(username) -> str:
    """Returns the copy destination for the specified cluster.
    """
    return f"/home/{username}/FiD"


def get_upload_command(command_args: Dict[str, str]) -> str:
    """Returns the rsync command that copies your local code to the task host.
    """
    return (
        'rsync -Pauv -e "ssh -F /Users/{username}/.ssh/config" --exclude=".git/" '
        '--exclude=".git/" --exclude-from="$(git -C ./ ls-files '
        '--exclude-standard -oi --directory > /tmp/excludes; echo /tmp/excludes)" '
        "./* {username}@{host}:{remote_dir}"
    ).format(**command_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username")
    parser.add_argument("-o", "--host")

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run sync once and stop. "
             "If set, the usual upload will complete and stop.",
    )
    args = parser.parse_args()

    command_args = {
        "username": args.username,
        "host": args.host,
        "remote_dir": get_remote_dir(args.username),
    }
    upload_command = get_upload_command(command_args)

    print("Watching for changes. Running:\n" + upload_command)
    subprocess.call(upload_command, shell=True)

    # Start watching
    if not args.once:
        try:
            watch_command = (
                'fswatch -or0 -l 0.2 --exclude ".\*.git.\*" ./ | '
                'xargs -0 -n 1 bash -c "{upload_command}";'
            ).format(
                upload_command=upload_command.replace('"', '\\"'),
            )
            print(watch_command)
            subprocess.Popen(watch_command, shell=True).wait()
        except:
            # End quietly.
            pass
