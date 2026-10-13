"""Command-line interface: ``vap <command> ...``.

Commands work on local files and the local SQLite state; ``serve`` starts the
REST API. Run ``vap <command> --help`` for options.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vap import __version__
from vap.config import Settings, load_settings
from vap.detection.specs import build_detector
from vap.errors import ConfigError, InvalidInputError, VapError, validation_details
from vap.evaluation.runner import EvaluationRequest, evaluate_files, run_evaluation
from vap.events.model import Event
from vap.exporting import write_export
from vap.extraction import ExtractionSpec, extract_frames
from vap.imaging.backend import get_backend
from vap.imaging.chain import OpChain, apply_chain
from vap.imaging.codec import load_image, save_image
from vap.observability.logs import configure_logging
from vap.pipeline.config import PipelineConfig
from vap.pipeline.pipeline import CameraPipeline
from vap.registry.models import ModelRegistry
from vap.sources.base import FrameSource
from vap.sources.image_sequence import ImageSequenceSource
from vap.sources.specs import SyntheticSourceSpec
from vap.sources.synthetic import SyntheticScene, SyntheticSource
from vap.sources.video_file import VideoFileSource, write_video
from vap.storage.database import Database
from vap.storage.repositories import Repositories
from vap.tracking.trackers import build_tracker


def _print(data: Any) -> None:
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidInputError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidInputError(f"{path} must contain a JSON object")
    return data


def parse_synthetic(arg: str) -> SyntheticSourceSpec:
    """``synthetic`` or ``synthetic:seed=3,objects=4,frames=200,fps=15,width=320,height=240``."""
    params: dict[str, Any] = {}
    _, _, rest = arg.partition(":")
    aliases = {
        "objects": "num_objects",
        "frames": "num_frames",
        "noise": "noise_std",
        "speed": "max_speed",
    }
    for item in filter(None, rest.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise InvalidInputError(f"bad synthetic parameter {item!r} (expected key=value)")
        params[aliases.get(key, key)] = float(value) if "." in value else int(value)
    try:
        return SyntheticSourceSpec.model_validate(params)
    except ValueError as exc:
        raise InvalidInputError(f"invalid synthetic parameters: {exc}") from exc


def source_from_arg(arg: str, fps: float = 10.0) -> FrameSource:
    if arg == "synthetic" or arg.startswith("synthetic:"):
        spec = parse_synthetic(arg)
        return SyntheticSource("cli", spec.scene_spec(), loop=False)
    path = Path(arg)
    if path.is_dir():
        return ImageSequenceSource("cli", path, fps=fps)
    if path.is_file():
        return VideoFileSource("cli", path)
    raise InvalidInputError(f"source not found: {arg}")


# -- commands ---------------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    spec = parse_synthetic(f"synthetic:{args.params}" if args.params else "synthetic")
    scene = SyntheticScene(spec.scene_spec())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = scene.frames("generate")
    if args.format == "avi":
        written = write_video(out / "video.avi", (f.image for f in frames), scene.spec.fps)
    else:
        for f in frames:
            save_image(out / f"frame_{f.index:05d}.png", f.image)
        written = len(frames)
    with (out / "ground_truth.jsonl").open("w", encoding="utf-8") as fh:
        for f in frames:
            for g in scene.ground_truth(f.index):
                rec = {"frame": g.frame_index, "id": g.object_id, "label": g.label}
                rec["bbox"] = [round(v, 2) for v in g.bbox.to_list()]
                fh.write(json.dumps(rec) + "\n")
    _print({"frames": written, "directory": str(out), "fps": scene.spec.fps})
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    spec = ExtractionSpec(
        mode=args.mode,
        every_n=args.every_n,
        interval_s=args.interval,
        scene_threshold=args.threshold,
        max_frames=args.max_frames,
        image_format=args.image_format,
    )
    with source_from_arg(args.source, args.fps) as source:
        result = extract_frames(source, spec, args.out, backend=args.backend)
    result.pop("frames")
    _print(result)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run a full pipeline offline and write predictions/events as JSONL."""
    config = PipelineConfig.model_validate({**_read_json(args.config), "realtime": False})
    source = source_from_arg(args.source, args.fps)
    pipe = CameraPipeline(
        "cli",
        "cli",
        source,
        build_detector(config.detector, backend=args.backend),
        build_tracker(config.tracker),
        config,
    )
    events: list[Event] = []
    preds = Path(args.out)
    preds.parent.mkdir(parents=True, exist_ok=True)
    with source, preds.open("w", encoding="utf-8") as fh:
        for frame in source:
            events.extend(pipe.process_frame(frame))
            for t in pipe.tracker.confirmed_tracks():
                if t.misses == 0:
                    rec = {"frame": frame.index, "id": t.track_id, "score": round(t.score, 4)}
                    rec["bbox"] = [round(v, 2) for v in t.bbox.to_list()]
                    rec["label"] = t.label
                    fh.write(json.dumps(rec) + "\n")
            if args.max_frames and pipe.frames_processed >= args.max_frames:
                break
    if args.events:
        write_export(args.events, (e.to_dict() for e in events), "events", "jsonl")
    summary = pipe.engine.snapshot()
    summary["frames"] = pipe.frames_processed
    summary["events"] = len(events)
    _print(summary)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.gt or args.pred:
        if not (args.gt and args.pred):
            raise InvalidInputError("--gt and --pred must be given together")
        thresholds = [float(t) for t in args.iou.split(",")]
        _print(evaluate_files(Path(args.gt), Path(args.pred), thresholds))
        return 0
    request = EvaluationRequest.model_validate(_read_json(args.request))
    report = run_evaluation(request)
    _print(report)
    if args.min_mota is not None and report["tracking"]["mota"] < args.min_mota:
        sys.stderr.write(f"MOTA {report['tracking']['mota']:.4f} below gate {args.min_mota}\n")
        return 2
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    chain = OpChain.model_validate({"ops": json.loads(args.ops)})
    image = load_image(args.image)
    result, steps = apply_chain(image, chain, get_backend(args.backend))
    save_image(args.out, result)
    _print({"output": args.out, "steps": steps})
    return 0


def _repos(settings: Settings) -> Repositories:
    settings.ensure_dirs()
    return Repositories(Database(settings.storage.database_path))


def cmd_models(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    repos = _repos(settings)
    registry = ModelRegistry(repos.db, settings.storage.models_dir)
    try:
        if args.action == "list":
            _print([m.to_dict() for m in registry.list(args.name)])
        elif args.action == "register":
            meta = json.loads(args.metadata) if args.metadata else {}
            model = registry.register_file(
                args.name,
                args.version,
                args.file,
                fmt=args.format,
                metadata=meta,
                activate=args.activate,
            )
            _print(model.to_dict())
        elif args.action == "activate":
            _print(registry.activate(args.name, args.version).to_dict())
        elif args.action == "verify":
            _print(registry.verify(args.name, args.version).to_dict())
    finally:
        repos.db.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    repos = _repos(settings)
    try:
        if args.kind == "events":
            rows = repos.events.list(pipeline_id=args.pipeline, limit=args.limit, ascending=True)
        elif args.kind == "tracks":
            rows = repos.tracks.list(args.pipeline, None, args.limit)
        else:
            if not args.pipeline:
                raise InvalidInputError("--pipeline is required for metrics exports")
            rows = repos.metrics.list(args.pipeline, None, args.limit)
        count = write_export(args.out, rows, args.kind, args.format)
    finally:
        repos.db.close()
    _print({"rows": count, "output": args.out})
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    data = settings.model_dump(mode="json")
    roles = sorted(data["auth"]["api_keys"].values())
    data["auth"]["api_keys"] = {"count": len(roles), "roles": roles}
    if data["webhooks"]["secret"]:
        data["webhooks"]["secret"] = "***"
    _print(data)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from vap.api.app import create_app

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.format)
    host = args.host or settings.server.host
    port = args.port or settings.server.port
    uvicorn.run(
        create_app(settings),
        host=host,
        port=port,
        log_level=settings.logging.level,
        access_log=False,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vap", description="Vision analytics platform CLI")
    p.add_argument("--version", action="version", version=f"vap {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the REST API")
    s.add_argument("--config")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(func=cmd_serve)

    g = sub.add_parser("generate", help="render a synthetic video with ground truth")
    g.add_argument("--out", required=True)
    g.add_argument("--params", default="", help="e.g. seed=3,objects=4,frames=200")
    g.add_argument("--format", choices=["png", "avi"], default="png")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("extract", help="extract frames from a source")
    e.add_argument("source", help="video file, image directory or synthetic[:k=v,...]")
    e.add_argument("--out", required=True)
    e.add_argument("--mode", choices=["every_n", "interval", "scene_change"], default="every_n")
    e.add_argument("--every-n", type=int, default=10)
    e.add_argument("--interval", type=float, default=1.0)
    e.add_argument("--threshold", type=float, default=0.3)
    e.add_argument("--max-frames", type=int, default=100)
    e.add_argument("--image-format", choices=["png", "jpeg"], default="jpeg")
    e.add_argument("--fps", type=float, default=10.0, help="fps for image directories")
    e.add_argument("--backend", default="auto")
    e.set_defaults(func=cmd_extract)

    a = sub.add_parser("analyze", help="run detection + tracking + rules offline")
    a.add_argument("source")
    a.add_argument("--config", help="pipeline config JSON")
    a.add_argument("--out", required=True, help="predictions JSONL")
    a.add_argument("--events", help="events JSONL")
    a.add_argument("--max-frames", type=int)
    a.add_argument("--fps", type=float, default=10.0)
    a.add_argument("--backend", default="auto")
    a.set_defaults(func=cmd_analyze)

    v = sub.add_parser("evaluate", help="score detection/tracking")
    v.add_argument("--request", help="EvaluationRequest JSON (synthetic benchmark)")
    v.add_argument("--gt", help="ground truth JSONL")
    v.add_argument("--pred", help="predictions JSONL")
    v.add_argument("--iou", default="0.5", help="comma-separated IoU thresholds for file mode")
    v.add_argument("--min-mota", type=float, help="exit 2 if MOTA is below this gate")
    v.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("process", help="apply an image op chain")
    pr.add_argument("image")
    pr.add_argument("--ops", required=True, help='JSON list, e.g. [{"op":"gray"},{"op":"canny"}]')
    pr.add_argument("--out", required=True)
    pr.add_argument("--backend", default="auto")
    pr.set_defaults(func=cmd_process)

    m = sub.add_parser("models", help="manage the model registry")
    m.add_argument("action", choices=["list", "register", "activate", "verify"])
    m.add_argument("--name")
    m.add_argument("--version")
    m.add_argument("--file")
    m.add_argument("--format", default="onnx")
    m.add_argument("--metadata", help="JSON object")
    m.add_argument("--activate", action="store_true")
    m.add_argument("--config")
    m.set_defaults(func=cmd_models)

    x = sub.add_parser("export", help="export events/tracks/metrics from the local database")
    x.add_argument("kind", choices=["events", "tracks", "metrics"])
    x.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    x.add_argument("--out", required=True)
    x.add_argument("--pipeline")
    x.add_argument("--limit", type=int, default=1000)
    x.add_argument("--config")
    x.set_defaults(func=cmd_export)

    c = sub.add_parser("config", help="validate and print the effective configuration")
    c.add_argument("--config")
    c.set_defaults(func=cmd_config)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "models" and args.action != "list" and not (args.name and args.version):
        parser.error("models register/activate/verify need --name and --version")
    if args.command == "models" and args.action == "register" and not args.file:
        parser.error("models register needs --file")
    try:
        return int(args.func(args))
    except ConfigError as exc:
        sys.stderr.write(f"configuration error: {exc.message}\n{json.dumps(exc.details)}\n")
        return 3
    except VapError as exc:
        sys.stderr.write(f"error ({exc.code}): {exc.message}\n")
        return 1
    except ValidationError as exc:
        details = validation_details(list(exc.errors()))
        sys.stderr.write(f"error (invalid_input): {json.dumps(details)}\n")
        return 1
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"error (invalid_input): invalid JSON argument: {exc}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
