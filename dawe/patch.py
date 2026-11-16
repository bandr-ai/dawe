"""Sparse, checkpoint-bound weight patches.

A patch lists, per tensor, the flat indices that changed with their old and new values.
It names the exact checkpoint it was made for (content hash of every tensor, plus config and
tokenizer hashes), so applying it to any other checkpoint is refused, and it can be undone exactly.

Pure standard library: no torch, no numpy. Weights are treated as raw bytes, so any dtype works.
Format description: docs/PATCH_FORMAT.md.

    dawe patch create    --source CKPT --edited CKPT -o x.gpatch
    dawe patch from-dense --source CKPT --dense old_patch.safetensors -o x.gpatch
    dawe patch inspect   x.gpatch
    dawe patch apply     x.gpatch --source CKPT --out NEW_DIR
    dawe patch restore   x.gpatch --patched NEW_DIR --out ORIGINAL_DIR
    dawe patch verify    x.gpatch DIR --expect source|result
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import mmap
import os
import shutil
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Keep v1 wire identifiers stable so published patches remain readable after the rename.
FORMAT = "geodesic-patch"
VERSION = 1
PROVENANCE_FILE = "geodesic_applied.json"
SUFFIX = ".gpatch"

DTYPE_SIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
}
_UNSIGNED = {1: "B", 2: "H", 4: "I", 8: "Q"}
# Files whose content decides how text becomes token ids. They must match exactly.
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
    "added_tokens.json", "chat_template.jinja", "chat_template.json",
)
CONFIG_FILES = ("config.json", "generation_config.json")
HASH_CHUNK = 16 << 20
DIFF_BLOCK = 1 << 20

Log = Callable[[str], None]


class PatchError(Exception):
    """The patch file or the operation is invalid."""


class IncompatibleCheckpoint(PatchError):
    """The checkpoint is not the one this patch was made for (or, when restoring, not the patched result)."""


# --- safetensors, minimal ----------------------------------------------------------------------


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int  # byte offsets inside the data section
    end: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def itemsize(self) -> int:
        return DTYPE_SIZE[self.dtype]


@dataclass(frozen=True)
class Shard:
    path: Path
    data_start: int  # absolute file offset of the data section
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]


def read_header(path: Path) -> Shard:
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise PatchError(f"{path.name}: not a safetensors file")
        (n,) = struct.unpack("<Q", head)
        if n > size - 8 or n > 100 << 20:
            raise PatchError(f"{path.name}: implausible safetensors header length {n}")
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {}) or {}
    data_start = 8 + n
    out = {}
    for name, d in header.items():
        info = TensorInfo(d["dtype"], tuple(d["shape"]), *d["data_offsets"])
        if info.dtype not in DTYPE_SIZE:
            raise PatchError(f"{path.name}: tensor {name} has unsupported dtype {info.dtype}")
        if info.end - info.start != info.numel * info.itemsize or data_start + info.end > size:
            raise PatchError(f"{path.name}: tensor {name} has inconsistent offsets")
        out[name] = info
    return Shard(path, data_start, out, meta)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, Sequence[int], bytes]], metadata: dict[str, str] | None = None) -> None:
    """tensors: name -> (dtype, shape, raw little-endian bytes). Deterministic: written in the order given."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        if len(raw) != DTYPE_SIZE[dtype] * _prod(shape):
            raise PatchError(f"{name}: {len(raw)} bytes do not fit dtype {dtype} shape {list(shape)}")
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, raw in tensors.values():
            f.write(raw)


def _prod(shape: Iterable[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# --- checkpoints ---------------------------------------------------------------------------------


class Checkpoint:
    """A model directory: every top-level *.safetensors file, indexed by tensor name."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        if not self.root.is_dir():
            raise PatchError(f"{self.root}: not a directory")
        self.shards = [read_header(p) for p in sorted(self.root.glob("*.safetensors"))]
        if not self.shards:
            raise PatchError(f"{self.root}: no .safetensors files")
        self.where: dict[str, tuple[Shard, TensorInfo]] = {}
        for sh in self.shards:
            for name, info in sh.tensors.items():
                if name in self.where:
                    raise PatchError(f"tensor {name} appears in two shards of {self.root}")
                self.where[name] = (sh, info)

    def names(self) -> list[str]:
        return sorted(self.where)

    def info(self, name: str) -> TensorInfo:
        return self.where[name][1]

    def span(self, name: str) -> tuple[Path, int, int]:
        """(file, absolute start, absolute end) of a tensor's bytes."""
        sh, info = self.where[name]
        return sh.path, sh.data_start + info.start, sh.data_start + info.end

    def resolve(self, name: str) -> str:
        """Map a framework parameter name to this checkpoint's key (HF modules vs. saved names differ by prefix)."""
        cands = [name, name.removeprefix("model."), "model." + name]
        hits = [c for c in dict.fromkeys(cands) if c in self.where]
        if len(hits) != 1:
            raise PatchError(f"cannot match tensor {name!r} to the checkpoint ({len(hits)} candidates among {cands})")
        return hits[0]


def _shape_str(shape: Sequence[int]) -> str:
    return ",".join(str(d) for d in shape)


def sha256_range(path: Path, start: int, end: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(start)
        left = end - start
        while left:
            b = f.read(min(HASH_CHUNK, left))
            if not b:
                raise PatchError(f"{path.name}: file is shorter than its header says")
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_range(path, 0, path.stat().st_size)


def _digest_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def layout_digest(ck: Checkpoint) -> str:
    """Hash of every tensor's name, dtype and shape (no weight values)."""
    return _digest_lines([f"{n}\t{ck.info(n).dtype}\t{_shape_str(ck.info(n).shape)}" for n in ck.names()])


def fingerprint_of(ck: Checkpoint, hashes: dict[str, str]) -> str:
    """Content identity: hash of every tensor's name, dtype, shape and sha256 of its bytes."""
    return _digest_lines([f"{n}\t{ck.info(n).dtype}\t{_shape_str(ck.info(n).shape)}\t{hashes[n]}" for n in ck.names()])


def tensor_hashes(ck: Checkpoint, log: Log | None = None) -> dict[str, str]:
    out = {}
    names = ck.names()
    for i, n in enumerate(names):
        out[n] = sha256_range(*ck.span(n))
        if log and (i + 1) % 100 == 0:
            log(f"  hashed {i + 1}/{len(names)} tensors")
    return out


def side_files(root: Path) -> dict[str, dict[str, str]]:
    """sha256 of the config and tokenizer files that are present."""
    return {
        "config": {f: sha256_file(root / f) for f in CONFIG_FILES if (root / f).is_file()},
        "tokenizer": {f: sha256_file(root / f) for f in TOKENIZER_FILES if (root / f).is_file()},
    }


@dataclass
class Identity:
    layout_sha256: str
    fingerprint: str
    n_tensors: int
    config: dict[str, str]
    tokenizer: dict[str, str]
    hashes: dict[str, str]  # per tensor (not stored in patches; too big)


def identify(root: Path | str, log: Log | None = None) -> Identity:
    ck = Checkpoint(root)
    hashes = tensor_hashes(ck, log)
    sf = side_files(ck.root)
    return Identity(layout_digest(ck), fingerprint_of(ck, hashes), len(hashes), sf["config"], sf["tokenizer"], hashes)


# --- the patch object -----------------------------------------------------------------------------


@dataclass
class TensorEdit:
    name: str
    dtype: str
    shape: tuple[int, ...]
    indices: list[int]  # strictly increasing flat (row-major) positions
    old: bytes  # len(indices) * itemsize
    new: bytes
    source_sha256: str = ""  # whole tensor before / after
    result_sha256: str = ""

    @property
    def itemsize(self) -> int:
        return DTYPE_SIZE[self.dtype]

    @property
    def numel(self) -> int:
        return _prod(self.shape)


@dataclass
class Patch:
    manifest: dict
    edits: list[TensorEdit]
    size_bytes: int = 0

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def patch_id(self) -> str:
        return self.manifest["patch_id"]

    def changed(self) -> int:
        return sum(len(e.indices) for e in self.edits)


def _idx_dtype(numel: int) -> str:
    return "U32" if numel <= 1 << 32 else "U64"


def _pack_idx(indices: Sequence[int], dtype: str) -> bytes:
    return struct.pack(f"<{len(indices)}{'I' if dtype == 'U32' else 'Q'}", *indices)


def _unpack_idx(raw: bytes, dtype: str) -> list[int]:
    n = len(raw) // DTYPE_SIZE[dtype]
    return list(struct.unpack(f"<{n}{'I' if dtype == 'U32' else 'Q'}", raw))


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _payload_digest(edits: list[TensorEdit]) -> str:
    h = hashlib.sha256()
    for e in edits:
        h.update(_pack_idx(e.indices, _idx_dtype(e.numel)))
        h.update(e.old)
        h.update(e.new)
    return h.hexdigest()


def _validate_edit(e: TensorEdit) -> None:
    k = len(e.indices)
    if len(e.old) != k * e.itemsize or len(e.new) != k * e.itemsize:
        raise PatchError(f"{e.name}: value bytes do not match {k} indices of dtype {e.dtype}")
    if k and (e.indices[0] < 0 or e.indices[-1] >= e.numel):
        raise PatchError(f"{e.name}: index out of range for {e.numel} elements")
    if any(b <= a for a, b in zip(e.indices, e.indices[1:])):
        raise PatchError(f"{e.name}: indices must be strictly increasing")


def write_patch(path: Path | str, manifest: dict, edits: list[TensorEdit]) -> Patch:
    """Fill in the derived manifest fields, then write. The bytes depend only on the inputs."""
    for e in edits:
        _validate_edit(e)
    manifest = dict(manifest)
    manifest["format"], manifest["version"] = FORMAT, VERSION
    manifest["tensors"] = [
        {
            "name": e.name, "dtype": e.dtype, "shape": list(e.shape), "numel": e.numel, "changed": len(e.indices),
            "source_sha256": e.source_sha256, "result_sha256": e.result_sha256,
        }
        for e in edits
    ]
    manifest["changed_weights"] = sum(len(e.indices) for e in edits)
    manifest["payload_sha256"] = _payload_digest(edits)
    manifest.pop("patch_id", None)
    manifest["patch_id"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    tensors: dict = {}
    for e in edits:
        it = _idx_dtype(e.numel)
        tensors[f"index/{e.name}"] = (it, [len(e.indices)], _pack_idx(e.indices, it))
        tensors[f"old/{e.name}"] = (e.dtype, [len(e.indices)], e.old)
        tensors[f"new/{e.name}"] = (e.dtype, [len(e.indices)], e.new)
    path = Path(path)
    write_safetensors(path, tensors, {"geodesic_patch": json.dumps(manifest, sort_keys=True), "format": f"{FORMAT}/{VERSION}"})
    return Patch(manifest, edits, path.stat().st_size)


def load_patch(path: Path | str) -> Patch:
    path = Path(path)
    sh = read_header(path)
    raw = sh.metadata.get("geodesic_patch")
    if raw is None:
        raise PatchError(f"{path.name}: not a dawe patch (no manifest)")
    manifest = json.loads(raw)
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise PatchError(f"{path.name}: unsupported patch format {manifest.get('format')}/{manifest.get('version')}")
    expect_id = manifest.pop("patch_id")
    if hashlib.sha256(_canonical(manifest)).hexdigest() != expect_id:
        raise PatchError(f"{path.name}: manifest was modified (patch_id mismatch)")
    manifest["patch_id"] = expect_id
    edits = []
    with path.open("rb") as f:
        def get(key: str) -> bytes:
            info = sh.tensors[key]
            f.seek(sh.data_start + info.start)
            return f.read(info.end - info.start)

        for t in manifest["tensors"]:
            n = t["name"]
            e = TensorEdit(n, t["dtype"], tuple(t["shape"]), _unpack_idx(get(f"index/{n}"), _idx_dtype(t["numel"])),
                           get(f"old/{n}"), get(f"new/{n}"), t["source_sha256"], t["result_sha256"])
            if len(e.indices) != t["changed"]:
                raise PatchError(f"{path.name}: {n} has {len(e.indices)} indices, manifest says {t['changed']}")
            _validate_edit(e)
            edits.append(e)
    if _payload_digest(edits) != manifest["payload_sha256"]:
        raise PatchError(f"{path.name}: payload hash mismatch (file is corrupted or was edited)")
    return Patch(manifest, edits, path.stat().st_size)


# --- finding and describing differences ------------------------------------------------------------


def _typed(buf: bytes, size: int) -> array:
    a = array(_UNSIGNED[size])
    if a.itemsize != size:  # exotic platform: fall back to bytes-level comparison
        raise PatchError("unsupported platform integer sizes")
    a.frombytes(buf)
    return a


def diff_regions(a: tuple[Path, int, int], b: tuple[Path, int, int], itemsize: int) -> list[int]:
    """Flat indices whose raw bytes differ between two same-sized byte regions (compared bit for bit)."""
    (pa, sa, ea), (pb, sb, eb) = a, b
    if ea - sa != eb - sb:
        raise PatchError("regions differ in size")
    out: list[int] = []
    block = max(1, DIFF_BLOCK // itemsize) * itemsize
    with pa.open("rb") as fa, pb.open("rb") as fb:
        off = 0
        total = ea - sa
        while off < total:
            n = min(block, total - off)
            fa.seek(sa + off)
            fb.seek(sb + off)
            xa, xb = fa.read(n), fb.read(n)
            if xa != xb:
                ta, tb = _typed(xa, itemsize), _typed(xb, itemsize)
                base = off // itemsize
                out.extend(base + i for i, (p, q) in enumerate(zip(ta, tb)) if p != q)
            off += n
    return out


def gather(region: tuple[Path, int, int], itemsize: int, indices: Sequence[int]) -> bytes:
    path, start, _ = region
    parts = []
    with path.open("rb") as f:
        for lo, hi, _ in _runs(indices):
            f.seek(start + lo * itemsize)
            parts.append(f.read((hi - lo) * itemsize))
    return b"".join(parts)


def sha256_with_edits(region: tuple[Path, int, int], itemsize: int, indices: Sequence[int], new: bytes) -> str:
    """sha256 of a tensor as it would be after the edits, without writing it anywhere."""
    path, start, end = region
    h = hashlib.sha256()
    chunk = max(1, HASH_CHUNK // itemsize) * itemsize
    with path.open("rb") as f:
        off = 0
        while off < end - start:
            n = min(chunk, end - start - off)
            f.seek(start + off)
            buf = bytearray(f.read(n))
            lo = bisect.bisect_left(indices, off // itemsize)
            hi = bisect.bisect_left(indices, (off + n) // itemsize)
            for k in range(lo, hi):
                p = indices[k] * itemsize - off
                buf[p:p + itemsize] = new[k * itemsize:(k + 1) * itemsize]
            h.update(buf)
            off += n
    return h.hexdigest()


def _runs(indices: Sequence[int]) -> Iterable[tuple[int, int, int]]:
    """(first element, last element + 1, position in the value list) for each run of consecutive indices."""
    i = 0
    while i < len(indices):
        j = i
        while j + 1 < len(indices) and indices[j + 1] == indices[j] + 1:
            j += 1
        yield indices[i], indices[j] + 1, i
        i = j + 1


def touched_rows(e: TensorEdit) -> list[int] | None:
    """Which rows (first axis) contain changes, for tensors with 2+ axes."""
    if len(e.shape) < 2:
        return None
    width = _prod(e.shape[1:])
    return sorted({i // width for i in e.indices})


# --- building patches ------------------------------------------------------------------------------


def _build(
    ck: Checkpoint,
    ident: Identity,
    diffs: dict[str, tuple[list[int], bytes]],
    *,
    name: str,
    description: str = "",
    lineage: dict | None = None,
    max_changed_fraction: float = 0.25,
    out: Path | str,
) -> Patch:
    edits = []
    result_hashes = dict(ident.hashes)
    for key in sorted(diffs):
        idx, new = diffs[key]
        info = ck.info(key)
        region = ck.span(key)
        old = gather(region, info.itemsize, idx)
        keep = [k for k in range(len(idx)) if old[k * info.itemsize:(k + 1) * info.itemsize] != new[k * info.itemsize:(k + 1) * info.itemsize]]
        if len(keep) != len(idx):  # drop entries that rewrite a value with itself
            idx = [idx[k] for k in keep]
            old = b"".join(old[k * info.itemsize:(k + 1) * info.itemsize] for k in keep)
            new = b"".join(new[k * info.itemsize:(k + 1) * info.itemsize] for k in keep)
        if not idx:
            continue
        if len(idx) > max_changed_fraction * info.numel:
            raise PatchError(
                f"{key}: {len(idx)} of {info.numel} weights changed ({len(idx) / info.numel:.0%}); "
                f"this is not a sparse edit (raise max_changed_fraction to force)"
            )
        e = TensorEdit(key, info.dtype, info.shape, list(idx), old, new, ident.hashes[key],
                       sha256_with_edits(region, info.itemsize, idx, new))
        result_hashes[key] = e.result_sha256
        edits.append(e)
    if not edits:
        raise PatchError("no weights differ; nothing to patch")
    manifest = {
        "name": name,
        "description": description,
        "source": {
            "fingerprint": ident.fingerprint,
            "layout_sha256": ident.layout_sha256,
            "n_tensors": ident.n_tensors,
            "config": ident.config,
            "tokenizer": ident.tokenizer,
            "lineage": lineage or {},
        },
        "result": {"fingerprint": fingerprint_of(ck, result_hashes)},
    }
    return write_patch(out, manifest, edits)


def create_from_dirs(source: Path | str, edited: Path | str, out: Path | str, *, name: str, log: Log | None = None, **kw) -> Patch:
    """Diff two checkpoints with identical layout."""
    ck, ek = Checkpoint(source), Checkpoint(edited)
    if layout_digest(ck) != layout_digest(ek):
        raise PatchError("source and edited checkpoints differ in tensor names, dtypes or shapes")
    diffs = {}
    for n in ck.names():
        info = ck.info(n)
        idx = diff_regions(ck.span(n), ek.span(n), info.itemsize)
        if idx:
            diffs[n] = (idx, gather(ek.span(n), info.itemsize, idx))
    if log:
        log(f"{len(diffs)} tensors differ; fingerprinting source")
    return _build(ck, identify(source, log), diffs, name=name, out=out, **kw)


def create_from_dense(source: Path | str, dense: Path | str, out: Path | str, *, name: str, log: Log | None = None,
                      ident: Identity | None = None, **kw) -> Patch:
    """Convert an old-style patch that stores whole changed tensors (a safetensors file of replacement tensors)."""
    ck = Checkpoint(source)
    dk = read_header(Path(dense))
    diffs = {}
    for dn, di in dk.tensors.items():
        key = ck.resolve(dn)
        info = ck.info(key)
        if (di.dtype, di.shape) != (info.dtype, info.shape):
            raise PatchError(f"{dn}: dense patch has {di.dtype}{list(di.shape)}, checkpoint has {info.dtype}{list(info.shape)}")
        region = (dk.path, dk.data_start + di.start, dk.data_start + di.end)
        idx = diff_regions(ck.span(key), region, info.itemsize)
        diffs[key] = (idx, gather(region, info.itemsize, idx))
    if log:
        log(f"{len(diffs)} tensors in the dense patch; fingerprinting source")
    return _build(ck, ident or identify(source, log), diffs, name=name, out=out, **kw)


def create_from_edits(source: Path | str, edits: dict[str, tuple[Sequence[int], bytes]], out: Path | str, *, name: str,
                      ident: Identity | None = None, **kw) -> Patch:
    """edits: parameter name -> (flat indices, new values as raw bytes). Used by the lab, which has indices in hand."""
    ck = Checkpoint(source)
    diffs = {}
    for pn, (idx, new) in edits.items():
        order = sorted(range(len(idx)), key=idx.__getitem__)
        size = ck.info(ck.resolve(pn)).itemsize
        diffs[ck.resolve(pn)] = ([idx[k] for k in order], b"".join(new[k * size:(k + 1) * size] for k in order))
    return _build(ck, ident or identify(source), diffs, name=name, out=out, **kw)


# --- checking a checkpoint against a patch ----------------------------------------------------------


def _expected(patch: Patch, side: str) -> tuple[str, Callable[[TensorEdit], str]]:
    if side == "source":
        return patch.manifest["source"]["fingerprint"], lambda e: e.source_sha256
    if side == "result":
        return patch.manifest["result"]["fingerprint"], lambda e: e.result_sha256
    raise ValueError(side)


def check_checkpoint(patch: Patch, root: Path | str, side: str, *, full: bool = True, log: Log | None = None) -> tuple[list[str], Identity | None]:
    """Problems that stop this directory from being the patch's `source` (or `result`). Empty list = compatible.

    full=True hashes every tensor, which is the only check that tells apart two checkpoints that
    agree on names, shapes, config, tokenizer and the patched tensors (e.g. a fine-tune and its base,
    when the patch edits tied embeddings that fine-tuning left alone). full=False skips that.
    """
    src = patch.manifest["source"]
    want_fp, want_t = _expected(patch, side)
    problems = []
    ck = Checkpoint(root)
    if layout_digest(ck) != src["layout_sha256"]:
        problems.append(f"tensor names/dtypes/shapes differ from the patch's checkpoint ({len(ck.where)} tensors here, {src['n_tensors']} expected)")
    sf = side_files(ck.root)
    for kind in ("config", "tokenizer"):
        want, got = src[kind], sf[kind]
        if want != got:
            diff = sorted({k for k in set(want) | set(got) if want.get(k) != got.get(k)})
            problems.append(f"{kind} files differ: {', '.join(diff)}")
    for e in patch.edits:
        if e.name not in ck.where:
            problems.append(f"missing tensor {e.name}")
            continue
        info = ck.info(e.name)
        if (info.dtype, info.shape) != (e.dtype, e.shape):
            problems.append(f"{e.name}: is {info.dtype}{list(info.shape)}, patch expects {e.dtype}{list(e.shape)}")
            continue
        if sha256_range(*ck.span(e.name)) != want_t(e):
            problems.append(f"{e.name}: tensor content differs from the {side} checkpoint")
    ident = None
    if full and not problems:
        ident = identify(root, log)
        if ident.fingerprint != want_fp:
            problems.append("full-checkpoint fingerprint differs: some tensor has different values than the patch's checkpoint")
    return problems, ident


def verify_checkpoint(patch_path: Path | str, root: Path | str, expect: str = "source", *, full: bool = True, log: Log | None = None) -> list[str]:
    return check_checkpoint(load_patch(patch_path), root, expect, full=full, log=log)[0]


def _explain(patch: Patch, root: Path, side: str, problems: list[str]) -> str:
    other = "result" if side == "source" else "source"
    hint = ""
    if not check_checkpoint(patch, root, other, full=False)[0]:
        hint = (f"\nThis directory already matches the patch's {other} checkpoint; "
                + ("it is already patched. Use `restore` to undo." if side == "source" else "it is not patched yet. Use `apply`."))
    return (f"{root} is not the patch's {side} checkpoint ({patch.name}, id {patch.patch_id[:12]}):\n  - "
            + "\n  - ".join(problems) + hint)


# --- apply / restore ---------------------------------------------------------------------------------


def _prepare_out(src: Path, out: Path) -> Path:
    src, out = src.resolve(), out.resolve()
    if out == src or src in out.parents or out in src.parents:
        raise PatchError("output directory must be separate from the input directory (patches never edit in place)")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise PatchError(f"{out} already exists and is not empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.dawe-tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    return tmp


def _place(src: Path, dst: Path, link: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link == "copy":
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | 0o200)  # the source may be read-only (e.g. a Hugging Face cache blob)
    elif link == "symlink":
        os.symlink(src.resolve(), dst)
    elif link == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"link must be copy, hardlink or symlink, not {link!r}")


def _transform(patch: Patch, src_root: Path, out_root: Path, direction: str, *, full: bool, link: str, log: Log | None) -> dict:
    side, after = ("source", "result") if direction == "apply" else ("result", "source")
    problems, ident = check_checkpoint(patch, src_root, side, full=full, log=log)
    if problems:
        raise IncompatibleCheckpoint(_explain(patch, src_root, side, problems))
    ck = Checkpoint(src_root)
    tmp = _prepare_out(src_root, out_root)
    try:
        touched = {e.name: e for e in patch.edits}
        shards_touched = {ck.where[n][0].path for n in touched}
        for dirpath, _, files in os.walk(ck.root):
            for fn in files:
                p = Path(dirpath) / fn
                rel = p.relative_to(ck.root)
                if rel.name == PROVENANCE_FILE:
                    continue
                _place(p, tmp / rel, "copy" if p in shards_touched else link)
        want_after = _expected(patch, after)[1]
        for e in patch.edits:
            sh, info = ck.where[e.name]
            values = e.new if direction == "apply" else e.old
            with (tmp / sh.path.relative_to(ck.root)).open("r+b") as f:
                base = sh.data_start + info.start
                for lo, hi, pos in _runs(e.indices):
                    f.seek(base + lo * info.itemsize)
                    f.write(values[pos * info.itemsize:(pos + hi - lo) * info.itemsize])
            got = sha256_range(tmp / sh.path.relative_to(ck.root), base, base + (info.end - info.start))
            if got != want_after(e):
                raise PatchError(f"{e.name}: written tensor does not match the expected {after} hash (bug or disk error)")
        fp = _expected(patch, after)[0]
        (tmp / PROVENANCE_FILE).write_text(json.dumps({
            "tool": f"{FORMAT}/{VERSION}", "operation": direction, "patch_name": patch.name, "patch_id": patch.patch_id,
            "input_fingerprint": _expected(patch, side)[0], "output_fingerprint": fp,
        }, indent=1) + "\n")
        if out_root.exists():
            out_root.rmdir()
        tmp.rename(out_root)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"operation": direction, "out": str(out_root), "changed_weights": patch.changed(), "tensors": [e.name for e in patch.edits],
            "output_fingerprint": fp, "checked": "full" if full else "fast"}


def apply_patch(patch_path: Path | str, source: Path | str, out: Path | str, *, full: bool = True, link: str = "copy", log: Log | None = None) -> dict:
    """Write `out` = `source` + patch. Refuses any checkpoint that is not the patch's exact source."""
    return _transform(load_patch(patch_path), Path(source), Path(out), "apply", full=full, link=link, log=log)


def restore_patch(patch_path: Path | str, patched: Path | str, out: Path | str, *, full: bool = True, link: str = "copy", log: Log | None = None) -> dict:
    """Write `out` = the original checkpoint, from a directory produced by apply (or any exact match of the result)."""
    return _transform(load_patch(patch_path), Path(patched), Path(out), "restore", full=full, link=link, log=log)


# --- reporting ---------------------------------------------------------------------------------------


def describe(patch: Patch) -> dict:
    m = patch.manifest
    dense = sum(e.numel * e.itemsize for e in patch.edits)
    return {
        "name": m["name"], "description": m.get("description", ""), "patch_id": m["patch_id"],
        "file_bytes": patch.size_bytes, "dense_bytes_if_whole_tensors": dense,
        "changed_weights": patch.changed(),
        "source_fingerprint": m["source"]["fingerprint"], "result_fingerprint": m["result"]["fingerprint"],
        "source_layout_sha256": m["source"]["layout_sha256"], "n_source_tensors": m["source"]["n_tensors"],
        "config": m["source"]["config"], "tokenizer": m["source"]["tokenizer"], "lineage": m["source"].get("lineage", {}),
        "tensors": [
            {"name": e.name, "dtype": e.dtype, "shape": list(e.shape), "changed": len(e.indices), "of": e.numel,
             "rows": touched_rows(e), "source_sha256": e.source_sha256, "result_sha256": e.result_sha256}
            for e in patch.edits
        ],
    }


def format_description(d: dict) -> str:
    lines = [f"{d['name']}  (id {d['patch_id'][:16]})", d["description"] or "", ""]
    lines += [f"file size      {d['file_bytes']:,} bytes  (whole changed tensors would be {d['dense_bytes_if_whole_tensors']:,})",
              f"weights changed {d['changed_weights']:,}",
              f"applies to     checkpoint {d['source_fingerprint'][:16]}...  ({d['n_source_tensors']} tensors)",
              f"produces       checkpoint {d['result_fingerprint'][:16]}...", ""]
    for t in d["tensors"]:
        rows = t["rows"]
        where = "" if rows is None else f"  rows {rows[:8]}{' ...' if len(rows) > 8 else ''}"
        lines.append(f"  {t['name']}  {t['dtype']}{t['shape']}: {t['changed']:,} of {t['of']:,}{where}")
    lines += ["", "config files    " + ", ".join(f"{k} {v[:12]}" for k, v in d["config"].items()),
              "tokenizer files " + ", ".join(f"{k} {v[:12]}" for k, v in d["tokenizer"].items())]
    if d["lineage"]:
        lines += ["", "lineage:", json.dumps(d["lineage"], indent=2)]
    return "\n".join(lines)


# --- command line ------------------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _lineage(a: argparse.Namespace) -> dict | None:
    return json.loads(Path(a.lineage_file).read_text()) if a.lineage_file else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dawe patch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_create(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", required=True, type=Path, help="checkpoint directory the patch applies to")
        p.add_argument("-o", "--out", required=True, type=Path)
        p.add_argument("--name", default=None, help="default: output file name")
        p.add_argument("--description", default="")
        p.add_argument("--lineage-file", type=Path, help="JSON describing how the source checkpoint was made")
        p.add_argument("--max-changed-fraction", type=float, default=0.25)

    c = sub.add_parser("create", help="diff two checkpoint directories")
    common_create(c)
    c.add_argument("--edited", required=True, type=Path)
    d = sub.add_parser("from-dense", help="convert an old patch file that stores whole tensors")
    common_create(d)
    d.add_argument("--dense", required=True, type=Path)
    d.add_argument("--expect-changed", type=int, default=None, help="fail unless exactly this many weights change")
    i = sub.add_parser("inspect")
    i.add_argument("patch", type=Path)
    i.add_argument("--json", action="store_true")
    for nm, helptext in (("apply", "write SOURCE + patch to a new directory"), ("restore", "write the original from a patched directory")):
        p = sub.add_parser(nm, help=helptext)
        p.add_argument("patch", type=Path)
        p.add_argument("--source" if nm == "apply" else "--patched", required=True, type=Path, dest="input")
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--fast", action="store_true", help="skip hashing untouched tensors (weaker: see docs/PATCH_FORMAT.md)")
        p.add_argument("--link", choices=["copy", "hardlink", "symlink"], default="copy", help="how untouched files get into the new directory")
    v = sub.add_parser("verify")
    v.add_argument("patch", type=Path)
    v.add_argument("dir", type=Path)
    v.add_argument("--expect", choices=["source", "result"], default="source")
    v.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)

    try:
        if a.cmd in ("create", "from-dense"):
            kw = dict(name=a.name or a.out.stem, description=a.description, lineage=_lineage(a),
                      max_changed_fraction=a.max_changed_fraction, log=_log)
            if a.cmd == "create":
                p = create_from_dirs(a.source, a.edited, a.out, **kw)
            else:
                p = create_from_dense(a.source, a.dense, a.out, **kw)
                if a.expect_changed is not None and p.changed() != a.expect_changed:
                    a.out.unlink()
                    raise PatchError(f"{p.changed()} weights changed, expected {a.expect_changed}; patch not written")
            print(format_description(describe(p)))
        elif a.cmd == "inspect":
            d_ = describe(load_patch(a.patch))
            print(json.dumps(d_, indent=1) if a.json else format_description(d_))
        elif a.cmd in ("apply", "restore"):
            fn = apply_patch if a.cmd == "apply" else restore_patch
            if a.fast:
                _log("warning: --fast does not hash untouched tensors; it cannot tell apart checkpoints that agree on config, tokenizer, shapes and the patched tensors")
            r = fn(a.patch, a.input, a.out, full=not a.fast, link=a.link, log=_log)
            print(json.dumps(r, indent=1))
        elif a.cmd == "verify":
            problems = verify_checkpoint(a.patch, a.dir, a.expect, full=not a.fast, log=_log)
            print(f"{a.dir} is the patch's {a.expect} checkpoint" if not problems else "\n".join(problems))
            return 1 if problems else 0
    except PatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
