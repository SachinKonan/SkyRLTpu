"""Validated tmpfs caches and additive GCS compilation-cache writeback."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
import zlib

from .events import emit
from .process import Process

GIB = 1024**3


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in ("..", ".") for p in name.split("/")):
        raise ValueError(f"unsafe cache object path: {name!r}")
    if any(c in name for c in ("\n", "\r", "\\")):
        raise ValueError("invalid cache object name")
    return path


def completed(name):
    return not any(part.endswith((".tmp", ".partial", ".gstmp"))
                   for part in PurePosixPath(name).parts)


@dataclass(frozen=True)
class Object:
    uri: str
    relative: str
    size: int
    generation: str
    md5: str | None
    crc32c: str | None

    def identity(self):
        return [self.uri, self.generation, self.size, self.md5, self.crc32c]


def objects_from_listing(listing, prefix):
    prefix = prefix.rstrip("/") + "/"
    result = []
    for entry in listing:
        if entry.get("type") != "cloud_object":
            continue
        meta = entry["metadata"]
        uri = "gs://" + meta["bucket"] + "/" + meta["name"]
        if not uri.startswith(prefix):
            raise ValueError("GCS returned an object outside the requested cache")
        relative = uri[len(prefix):]
        if not relative or relative.endswith("/"):
            continue
        safe_relative(relative)
        size = int(meta["size"])
        if size < 0 or not (meta.get("md5Hash") or meta.get("crc32c")):
            raise ValueError("cache object has no usable integrity metadata")
        result.append(Object(uri, relative, size, str(meta["generation"]),
                             meta.get("md5Hash"), meta.get("crc32c")))
    if len({o.relative for o in result}) != len(result):
        raise ValueError("duplicate cache object names")
    return sorted(result, key=lambda obj: obj.relative)


def valid_file(path, obj):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != obj.size:
        return False
    if obj.md5:
        digest = hashlib.md5()
        expected = obj.md5
    else:
        import google_crc32c
        digest = google_crc32c.Checksum()
        expected = obj.crc32c
    with path.open("rb") as data:
        while chunk := data.read(8 * 1024**2):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode() == expected


def fingerprint(objects):
    return hashlib.sha256(json.dumps([o.identity() for o in objects]).encode()).hexdigest()


def complete_compile_file(path):
    """Check the compressed JAX frame, not just its final-looking filename.

    JAX can write directly to *-cache. A stopped writer may leave that name
    containing a truncated frame; never publish it as an immutable GCS entry.
    """
    import zstandard

    try:
        with path.open("rb") as stream:
            magic = stream.read(4)
            stream.seek(0)
            decoder = (zstandard.ZstdDecompressor().decompressobj()
                       if magic == b"\x28\xb5\x2f\xfd" else zlib.decompressobj())
            size = 0
            while chunk := stream.read(4096):
                if decoder.eof:
                    return False
                size += len(decoder.decompress(chunk))
                if decoder.unused_data:
                    return False
            # Four bytes encode compile time; a serialized executable follows.
            return decoder.eof and size > 4
    except (OSError, zlib.error, zstandard.ZstdError):
        return False


def mount_cache(root, cap_gib, reserve_gib):
    if cap_gib < 64 or reserve_gib < 128:
        raise ValueError("cache budget would leave insufficient runtime memory")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("cache mount cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    mem = {s.split(":")[0]: int(s.split()[1]) * 1024
           for s in Path("/proc/meminfo").read_text().splitlines()}
    if mem["SwapTotal"]:
        raise RuntimeError("tmpfs cache requires swap disabled")
    if not os.path.ismount(root):
        if any(root.iterdir()):
            raise RuntimeError("refusing to mount over existing files")
        if mem["MemAvailable"] < (cap_gib + reserve_gib) * GIB:
            raise RuntimeError("not enough available memory for RAM cache plus runtime reserve")
        options = f"size={cap_gib}G,mode=0700,uid={os.getuid()},gid={os.getgid()},nosuid,nodev"
        subprocess.run(["sudo", "-n", "mount", "-t", "tmpfs", "-o", options,
                        "tmpfs", str(root)], check=True, timeout=30)
    info = json.loads(subprocess.check_output([
        "findmnt", "--json", "--mountpoint", str(root), "--output", "FSTYPE,TARGET"]))
    if info["filesystems"][0]["fstype"] != "tmpfs":
        raise RuntimeError("unexpected filesystem type for RAM cache")
    stat = os.statvfs(root)
    if stat.f_blocks * stat.f_frsize < cap_gib * GIB:
        # A mount left by an earlier run keeps that run's capacity; tmpfs
        # resizes in place, so grow it to this profile's budget (data is kept).
        if mem["MemAvailable"] < (cap_gib + reserve_gib) * GIB - stat.f_bfree * stat.f_frsize:
            raise RuntimeError("not enough available memory to grow the RAM cache")
        subprocess.run(["sudo", "-n", "mount", "-o", f"remount,size={cap_gib}G", str(root)],
                       check=True, timeout=30)
        stat = os.statvfs(root)
    if stat.f_blocks * stat.f_frsize > cap_gib * GIB:
        raise RuntimeError("unexpected filesystem type or cache capacity")
    if mem["MemAvailable"] < reserve_gib * GIB:
        raise RuntimeError("runtime memory reserve is unavailable")
    return root


class GCS:
    def __init__(self, log_dir, config):
        self.logs = Path(log_dir)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.events = self.logs / "cache-events.jsonl"
        self.env = dict(os.environ, CLOUDSDK_STORAGE_PROCESS_COUNT=str(config.process_count),
                        CLOUDSDK_STORAGE_THREAD_COUNT=str(config.thread_count),
                        CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=config.slice_threshold)
        self.running = None

    def metadata(self, *args, allow_empty=False):
        result = subprocess.run(["gcloud", "storage", *args], env=self.env,
                                capture_output=True, text=True, timeout=300)
        if result.returncode:
            if allow_empty and "matched no objects" in result.stderr.lower():
                return "[]"
            raise RuntimeError(f"GCS metadata request failed: {result.stderr[-1500:]}")
        return result.stdout

    def list(self, prefix, allow_empty=False):
        raw = self.metadata("ls", "--json", prefix.rstrip("/") + "/**", allow_empty=allow_empty)
        return objects_from_listing(json.loads(raw), prefix)

    def transfer(self, args, label, destination=None, timeout=3600):
        log = self.logs / f"{label}.log"
        self.running = Process(["gcloud", "storage", *args], log, env=self.env)
        start = time.monotonic()
        next_report = 0
        try:
            while self.running.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    raise TimeoutError(f"{label} exceeded {timeout}s")
                if elapsed >= next_report:
                    fields = {"transfer": label, "elapsed_seconds": elapsed}
                    if destination and Path(destination).exists():
                        stat = os.statvfs(destination)
                        fields["filesystem_free_bytes"] = stat.f_bavail * stat.f_frsize
                    emit(self.events, "transfer_progress", **fields)
                    next_report = elapsed + 30
                time.sleep(0.5)
            if self.running.poll() != 0:
                raise RuntimeError(f"{label} failed; see {log}")
            emit(self.events, "transfer_complete", transfer=label, seconds=time.monotonic()-start)
        finally:
            self.running.stop()
            self.running = None

    def stop(self):
        if self.running:
            self.running.stop()


class CacheStore:
    def __init__(self, root, gcs):
        self.root = Path(root).resolve()
        self.gcs = gcs

    def owned(self, name):
        safe_relative(name)
        path = self.root / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("cache path escapes private root")
        return path

    def clear(self, name):
        path = self.owned(name)
        if path == self.root:
            raise ValueError("cannot clear cache root")
        if path.exists():
            shutil.rmtree(path)

    def reconcile_role(self, role):
        if role not in ("trainer", "inference"):
            raise ValueError("unknown host role")
        previous = self.root / "role.json"
        if previous.exists() and json.loads(previous.read_text())["role"] != role:
            # This namespace belongs exclusively to this controller, not jobman.
            for name in ("hf", "orbax", "compile", "download", "metadata"):
                self.clear(name)
        previous.write_text(json.dumps({"role": role}))

    def prune_siblings(self, name):
        """Evict other models' trees that share this tree's parent directory.

        The tmpfs budget is per host and a host that keeps its role across runs
        keeps its private cache too; a Qwen orbax tree left behind by an earlier
        run would otherwise be charged against the next model's restore.
        """
        target = self.owned(name)
        parent = target.parent
        if parent == self.root or not parent.exists():
            return
        keep = {target.name, target.name + "-partial"}
        for child in parent.iterdir():
            if child.name not in keep:
                self.clear(str(Path(name).parent / child.name))
                emit(self.gcs.events, "tree_evicted", tree=str(Path(name).parent / child.name), kept=name)

    def scope_compile(self, prefix, name="compile"):
        """Bind the private compile cache to one publish prefix.

        Entries compiled for another model must never be published into this
        prefix, so a prefix change discards the cache before it is restored.
        """
        marker = self.owned(name + ".prefix")
        previous = marker.read_text().strip() if marker.exists() else None
        if previous != prefix:
            if previous is not None:
                emit(self.gcs.events, "compile_cache_rescoped", previous=previous, prefix=prefix)
            self.clear(name)
            self.clear(name + "-upload")
        marker.write_text(prefix)

    def restore_tree(self, prefix, name):
        """Reuse fully verified immutable trees; restart incomplete copies cleanly."""
        objects = self.gcs.list(prefix)
        if not objects:
            raise RuntimeError(f"required model cache is absent: {prefix}")
        if any(not completed(o.relative) for o in objects):
            # An Orbax source must itself be complete, never an interrupted mirror.
            raise RuntimeError(f"model cache contains temporary objects: {prefix}")
        destination = self.owned(name)
        self.prune_siblings(name)
        marker = destination / ".complete.json"
        identity = fingerprint(objects)
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous.get("identity") == identity and all(valid_file(destination / o.relative, o) for o in objects):
                emit(self.gcs.events, "tree_reused", source=prefix, objects=len(objects))
                return destination
        stage_name = name + "-partial"
        for attempt in range(3):
            self.clear(name)
            self.clear(stage_name)
            stage = self.owned(stage_name)
            stage.mkdir(parents=True)
            stat = os.statvfs(stage)
            required = sum(o.size for o in objects)
            if required + 2 * GIB > stat.f_bavail * stat.f_frsize:
                raise RuntimeError("model cache exceeds remaining tmpfs budget")
            try:
                # Recursive cp keeps Orbax's nested names; one native gcloud
                # process manages all concurrency, even on images without rsync.
                self.gcs.transfer(["cp", "--recursive", prefix.rstrip("/"), str(stage)],
                                  "model-tree", stage)
                copied = stage / prefix.rstrip("/").rsplit("/", 1)[1]
                if not all(valid_file(copied / o.relative, o) for o in objects):
                    raise RuntimeError("model tree checksum validation failed")
                destination.parent.mkdir(parents=True, exist_ok=True)
                copied.rename(destination)
                marker.write_text(json.dumps({"identity": identity, "source": prefix}))
                self.clear(stage_name)
                return destination
            except (RuntimeError, TimeoutError):
                self.clear(stage_name)
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def restore_compile(self, prefix, name="compile"):
        cache = self.owned(name)
        cache.mkdir(parents=True, exist_ok=True)
        objects = [o for o in self.gcs.list(prefix, allow_empty=True)
                   if completed(o.relative) and o.relative.endswith("-cache")]
        if any("/" in o.relative for o in objects):
            raise ValueError("compilation cache must use a flat JAX cache prefix")
        missing = [o for o in objects if not valid_file(cache / o.relative, o)]
        if missing:
            # Bound command-line length while leaving cross-object concurrency
            # entirely to one native gcloud invocation at a time.
            for offset in range(0, len(missing), 64):
                batch = missing[offset:offset+64]
                self.gcs.transfer(["cp", *[o.uri + "#" + o.generation for o in batch], str(cache)],
                                  "compile-restore", cache)
            if not all(valid_file(cache / o.relative, o) for o in missing):
                raise RuntimeError("compilation cache checksum validation failed")
        emit(self.gcs.events, "compile_cache_ready", source=prefix, restored=len(missing),
             reused=len(objects)-len(missing), cold=not objects)
        return cache

    def restore_hf(self, prefix, model_name, weights):
        model_key = "models--" + model_name.replace("/", "--")
        source = prefix.rstrip("/") + "/" + model_key
        revision = self.gcs.metadata("cat", source + "/refs/main").strip()
        if not re.fullmatch(r"[a-f0-9]{40,64}", revision):
            raise ValueError("HF cache revision is not a pinned commit")
        manifest = json.loads(self.gcs.metadata("cat", source + f"/trees/{revision}.json"))
        selected = {}
        for name, meta in manifest["files"].items():
            safe_relative(name)
            if not weights and name.endswith((".safetensors", ".bin", ".pt", ".pth")):
                continue
            blob = meta.get("lfs_sha256") or meta.get("blob_id", "")
            if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", blob):
                raise ValueError("invalid HF blob identifier")
            selected[name] = (blob, int(meta["size"]))
        if not {"config.json", "tokenizer.json"}.issubset(selected):
            raise ValueError("HF cache lacks required model/tokenizer metadata")
        cloud = {obj.relative: obj for obj in self.gcs.list(source + "/blobs")}
        blobs = {}
        for blob, size in selected.values():
            obj = cloud.get(blob)
            if obj is None or obj.size != size:
                raise ValueError(f"HF manifest references an absent or wrong-sized blob: {blob}")
            blobs[blob] = obj
        destination = self.owned("hf") / "hub" / model_key
        identity = hashlib.sha256(json.dumps([revision, weights, selected,
                                             [o.identity() for o in blobs.values()]], sort_keys=True).encode()).hexdigest()
        marker = self.owned("hf") / ".complete.json"
        complete = marker.exists() and json.loads(marker.read_text()).get("identity") == identity
        if not complete or not all(valid_file(destination / "blobs" / blob, obj)
                                   for blob, obj in blobs.items()):
            self.clear("hf")
            folder = destination / "blobs"
            folder.mkdir(parents=True)
            stat = os.statvfs(folder)
            if sum(o.size for o in blobs.values()) + 2 * GIB > stat.f_bavail * stat.f_frsize:
                raise RuntimeError("HF cache exceeds remaining tmpfs budget")
            for attempt in range(3):
                missing = [o for o in blobs.values() if not valid_file(folder / o.relative, o)]
                try:
                    for offset in range(0, len(missing), 64):
                        batch = missing[offset:offset+64]
                        for obj in batch:
                            (folder / obj.relative).unlink(missing_ok=True)
                            (folder / (obj.relative + "_.gstmp")).unlink(missing_ok=True)
                        self.gcs.transfer(["cp", *[o.uri + "#" + o.generation for o in batch], str(folder)],
                                          "hf-restore", folder)
                    if not all(valid_file(folder / blob, obj) for blob, obj in blobs.items()):
                        raise RuntimeError("HF cache checksum validation failed")
                    break
                except (RuntimeError, TimeoutError):
                    if attempt == 2:
                        self.clear("hf")
                        raise
        snapshot = destination / "snapshots" / revision
        for name, (blob, _) in selected.items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            target.symlink_to(os.path.relpath(destination / "blobs" / blob, target.parent))
        refs = destination / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(revision)
        if weights:
            index = snapshot / "model.safetensors.index.json"
            if not index.exists():
                raise RuntimeError("expected a sharded HF model index")
            for filename in set(json.loads(index.read_text())["weight_map"].values()):
                safe_relative(filename)
                if filename not in selected or not (snapshot / filename).is_file():
                    raise RuntimeError("HF index references an absent weight shard")
        marker.write_text(json.dumps({"identity": identity}))
        emit(self.gcs.events, "hf_cache_ready", revision=revision, weights=weights,
             files=len(selected), reused=bool(complete))
        return snapshot

    def publish_compile(self, prefix, name="compile"):
        """Publish new immutable JAX entries only; never remote-delete or sync temp files."""
        cache = self.owned(name)
        if not cache.exists():
            return 0
        remote = {o.relative: o for o in self.gcs.list(prefix, allow_empty=True)}
        files = [p for p in cache.iterdir() if p.is_file() and not p.is_symlink()
                 and completed(p.name) and p.name.endswith("-cache") and p.name not in remote]
        uploaded, deferred = 0, 0
        # The host serializes writeback calls. Staging belongs only to this
        # private cache and is replaceable, including after an interrupted sync.
        stage_name = name + "-upload"
        self.clear(stage_name)
        stage = self.owned(stage_name)
        try:
            for offset in range(0, len(files), 64):
                stage.mkdir(parents=True, exist_ok=True)
                batch = []
                for path in files[offset:offset+64]:
                    snapshot = stage / path.name
                    try:
                        before = path.stat()
                        if shutil.disk_usage(stage).free < before.st_size + 512 * 1024**2:
                            raise RuntimeError("insufficient tmpfs space for compilation upload snapshot")
                        shutil.copyfile(path, snapshot)
                        after = path.stat()
                        unchanged = ((before.st_ino, before.st_size, before.st_mtime_ns) ==
                                     (after.st_ino, after.st_size, after.st_mtime_ns))
                    except FileNotFoundError:
                        unchanged = False
                    if not unchanged or not complete_compile_file(snapshot):
                        snapshot.unlink(missing_ok=True)
                        deferred += 1
                        continue
                    batch.append(snapshot)
                if batch:
                    transfer_error = None
                    try:
                        self.gcs.transfer(["cp", "--no-clobber", *map(str, batch), prefix.rstrip("/") + "/"],
                                          "compile-publish", timeout=300)
                    except (RuntimeError, TimeoutError) as exc:
                        transfer_error = exc
                    # Concurrent hosts may create identical keys after our
                    # listing, causing gcloud's create-only upload to return
                    # 412. Accept that race only if every object verifies.
                    remote = {o.relative: o for o in self.gcs.list(prefix, allow_empty=True)}
                    for path in batch:
                        if path.name not in remote or not valid_file(path, remote[path.name]):
                            if transfer_error:
                                raise transfer_error
                            raise RuntimeError(f"compilation-cache upload verification failed: {path.name}")
                    if transfer_error:
                        emit(self.gcs.events, "compile_upload_race_verified", objects=len(batch), destination=prefix)
                    uploaded += len(batch)
                self.clear(stage_name)
        finally:
            self.clear(stage_name)
        emit(self.gcs.events, "compile_cache_published", destination=prefix,
             objects=uploaded, deferred_incomplete=deferred)
        return uploaded
