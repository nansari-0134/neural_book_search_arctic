"""Download the UCSD Goodreads book metadata dump used by prepare_data.py."""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
    "goodreads_books.json.gz"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "goodreads_books.json.gz"
CHUNK_SIZE = 1024 * 1024


def download(output: Path, force: bool = False) -> None:
    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists; pass --force to replace it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(URL, headers=headers)

    try:
        response = urlopen(request, timeout=60)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Could not download Goodreads data: {exc}") from exc

    with response:
        status = response.getcode()
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)

        if offset and status == 206 and match and int(match.group(1)) == offset:
            mode = "ab"
            total_size = int(match.group(3))
            downloaded = offset
        elif status == 200:
            mode = "wb"
            total_size = int(response.headers.get("Content-Length", "0")) or None
            downloaded = 0
        else:
            raise RuntimeError(
                f"Unexpected HTTP response {status} while downloading "
                f"(Content-Range: {content_range or 'none'})."
            )

        started = last_report = time.monotonic()
        with partial.open(mode) as destination:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                destination.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 2:
                    if total_size:
                        progress = f"{downloaded / total_size:.1%}"
                    else:
                        progress = f"{downloaded:,} bytes"
                    print(f"\rDownloaded {progress}", end="", flush=True)
                    last_report = now

        if total_size is not None and downloaded != total_size:
            raise RuntimeError(
                f"Incomplete download: received {downloaded:,} of "
                f"{total_size:,} bytes. Run the command again to resume."
            )

    partial.replace(output)
    elapsed = time.monotonic() - started
    print(
        f"\rSaved {downloaded:,} bytes to {output} "
        f"in {elapsed / 60:.1f} minutes."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing completed download",
    )
    args = parser.parse_args()

    try:
        download(args.output, force=args.force)
    except (FileExistsError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
