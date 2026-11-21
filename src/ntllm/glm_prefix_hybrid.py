"""Graft the public 0.716 GLM stage into Areeb's 0.711 v21 notebook."""

import argparse
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


PREFIX_ASSET = "estopadilla/ntl-glm-prefix-ab"


def source(cell):
    return "".join(cell["source"])


def cell_index(notebook, marker):
    matches = [i for i, cell in enumerate(notebook["cells"]) if source(cell).startswith(marker)]
    if len(matches) != 1:
        raise ValueError(f"Expected one cell starting with {marker!r}; found {len(matches)}")
    return matches[0]


def replace_once(text, original, replacement):
    if text.count(original) != 1:
        raise ValueError(f"Unexpected source around {original[:100]!r}")
    return text.replace(original, replacement, 1)


def add_apex_logits(setup):
    payload = re.search(r'^V1_PAYLOAD = "([A-Za-z0-9+/=]+)"$', setup, re.MULTILINE)
    if payload is None:
        raise ValueError("Missing the v5 GLM source archive")
    original = base64.b64decode(payload[1], validate=True)
    original_hash = hashlib.sha256(original).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(original), mode="r:gz") as package:
        files = {entry.name: package.extractfile(entry).read() for entry in package.getmembers()}

    stage = files["v1_stage.py"].decode("utf-8")
    stage = replace_once(
        stage,
        "    after = before.copy()\n    after[sel] = (zl[sel] + w * ll).argmax(1)\n",
        "    post_ll = zl.copy()\n    post_ll[sel] += w * ll\n    after = post_ll.argmax(1)\n",
    )
    stage = replace_once(
        stage,
        "    assert pd.read_parquet(a.out).equals(sub)\n",
        "    assert pd.read_parquet(a.out).equals(sub)\n"
        "    np.savez(Path(a.outdir) / 'post-glm.npz', ids=np.asarray(ids),\n"
        "             labels=np.asarray(labels), logits=post_ll)\n",
    )
    files["v1_stage.py"] = stage.encode("utf-8")
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as package:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            package.addfile(entry, io.BytesIO(data))
    updated = blob.getvalue()
    setup = replace_once(setup, payload[1], base64.b64encode(updated).decode("ascii"))
    return replace_once(setup, original_hash, hashlib.sha256(updated).hexdigest())


def hybrid_notebook(base, donor):
    """Return a notebook document, with mounts provided by Kaggle kernel metadata."""
    result = deepcopy(base)
    start = cell_index(result, "## Part D · The GLM likelihood test")
    end = cell_index(result, "## Part E · Soft typography and the final gate")
    config = deepcopy(donor["cells"][cell_index(donor, "# ---- ours, V1: likelihood under GLM-5.3-Flash")])
    setup = deepcopy(donor["cells"][cell_index(donor, "# ---- ours, V1 setup and checks")])
    run = deepcopy(donor["cells"][cell_index(donor, "# ---- ours, V1 stage: likelihood of the test")])
    old_setup = source(result["cells"][cell_index(result, "from pathlib import Path\nV1_TEST_CSV=")])
    pinned_segments = old_setup[old_setup.index("GLM_SEGMENTS = "):old_setup.index("artifacts = Path(")]

    config["source"] = source(config) + "\nV1_REHEARSAL = False\n"
    setup_text = add_apex_logits(source(setup))
    binary_start = setup_text.index("_cands = ")
    binary_end = setup_text.index("V1_BIN = ", binary_start)
    setup_text = setup_text[:binary_start] + '''_roots = [Path('/kaggle/input/notebooks/estopadilla/ntl-glm-prefix-ab/ntl-bin'),
          Path('/kaggle/input/ntl-glm-prefix-ab/ntl-bin')]
_libs = ('libcublas.so.13', 'libcublasLt.so.13', 'libcudart.so.13')
_ok = [root / 'ntl-segrun' for root in _roots
       if (root / 'ntl-segrun').is_file() and all((root / 'lib' / lib).is_file() for lib in _libs)
       and _sha(root / 'ntl-segrun') == 'a9e071e1529b46f2c79b11a1c0466a83e897bfd01bd0d3fc28f3a859626f25b0']
assert _ok, f'V1: missing the expected ntl-segrun or CUDA 13 libraries under {_roots}'
''' + setup_text[binary_end:]
    segments_start = setup_text.index("# segments of the two models")
    segments_end = setup_text.index("# checks before S10 runs", segments_start)
    setup["source"] = setup_text[:segments_start] + pinned_segments + '''def _link_segments(paths, name):
    dst = Path(f'/kaggle/temp/v1-segs-{name}')
    dst.mkdir(exist_ok=True)
    for path in map(Path, paths):
        link = dst / path.name
        if not link.exists():
            link.symlink_to(path)
    return dst


V1_FLASH = _link_segments(V1_FLASH, 'flash')
V1_GLM53 = _link_segments(V1_GLM53, 'glm53')

''' + setup_text[segments_end:]
    run["source"] = replace_once(
        source(run),
        "if V1_ENABLED and not FAST_COMMIT:\n",
        "if not FAST_COMMIT:\n"
        "    # Apex reads these logits for its Fable return rule, including on timeout.\n"
        "    shutil.copyfile(FINAL_LOGITS, V1_OUT / 'post-glm.npz')\n"
        "if V1_ENABLED and not FAST_COMMIT:\n",
    )
    intro = deepcopy(result["cells"][start])
    intro["source"] = (
        "## Part D · Deadline-aware GLM likelihood\n\n"
        "The public v5 GLM stage scores up to 160 answer tokens after a user request "
        "for a single paragraph of approximately the observed word count. It ranks passages by "
        "the combined GLM probability and can shrink that ranked queue between segments. "
        "It shares the original notebook's clock and finishes by 8.8 hours. "
        "The resulting aligned logits feed the Apex Fable return rule.\n"
    )
    result["cells"][start:end] = [intro, config, setup, run]
    for cell in result["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=f"Add {PREFIX_ASSET} to the original Kaggle kernel_sources.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    donor = json.loads(args.donor.read_text(encoding="utf-8"))
    result = hybrid_notebook(base, donor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
