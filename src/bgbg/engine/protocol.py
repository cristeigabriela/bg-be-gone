"""The compute protocol, declared once. Stdlib only.

The engine never does pixels or models. It asks a *compute port* for work and
gets events back, over this wire format: 12 jobs out, 20 events in.

Today the port is a venv subprocess speaking line-delimited JSON on a pipe. In
the browser it will be a Web Worker speaking the same JSON over `postMessage`,
and ORT-Web on the other side. The point of declaring the format here — rather
than leaving it as a docstring next to the sender and a chain of `elif`s next to
the receiver — is that both implementations get checked against one catalogue.

That is not hypothetical: when this file was written, the worker's own docstring
had already drifted from the code (it was missing the `unload` job, and the
`notice` and `unloaded` events). `tests/test_engine_protocol.py` now fails if the
code sends anything this file does not declare.

**No filesystem in the contract.** An image field is an `ImagePayload`, which is
a path *today* and can be bytes or an opaque handle tomorrow — the browser has no
paths. A bare string decodes as a path, so the desktop's existing wire format is
unchanged and the web can start sending `{"bytes": ...}` without a flag day.
"""

VERSION = 1


# ---------------------------------------------------------------- payloads ---
PATH = "path"
BYTES = "bytes"
HANDLE = "handle"


class ImagePayload:
    """An image on the wire: a path, some bytes, or an opaque handle.

    Disk is the desktop's *cache*, not the protocol. A bare JSON string means a
    path, which is what keeps today's wire format byte-compatible.
    """

    __slots__ = ("kind", "value", "mime")

    def __init__(self, kind, value, mime=None):
        self.kind = kind
        self.value = value
        self.mime = mime

    @staticmethod
    def path(p):
        return ImagePayload(PATH, str(p))

    @staticmethod
    def data(b, mime="image/png"):
        return ImagePayload(BYTES, b, mime)

    @staticmethod
    def handle(h):
        return ImagePayload(HANDLE, h)

    def encode(self):
        # A path encodes as a bare string: that IS the current wire format, so
        # nothing has to change on either side today.
        if self.kind == PATH:
            return self.value
        d = {"kind": self.kind, "value": self.value}
        if self.mime:
            d["mime"] = self.mime
        return d

    @staticmethod
    def decode(v):
        if v is None:
            return None
        if isinstance(v, str):
            return ImagePayload.path(v)
        if isinstance(v, dict) and "kind" in v:
            return ImagePayload(v["kind"], v.get("value"), v.get("mime"))
        raise ValueError("not an image payload: %r" % (v,))

    def __eq__(self, other):
        return (isinstance(other, ImagePayload) and self.kind == other.kind
                and self.value == other.value and self.mime == other.mime)

    def __repr__(self):
        return "ImagePayload(%s, %r)" % (self.kind, self.value)


# ------------------------------------------------------------------ fields ---
class Field:
    __slots__ = ("name", "required")

    def __init__(self, name, required=False):
        self.name = name
        self.required = required


def req(name):
    return Field(name, True)


def opt(name):
    return Field(name, False)


class Message:
    __slots__ = ("name", "fields", "doc")

    def __init__(self, name, fields=(), doc=""):
        self.name = name
        self.fields = list(fields)
        self.doc = doc

    def required(self):
        return [f.name for f in self.fields if f.required]

    def known(self):
        return {f.name for f in self.fields} | {"id"}


def _index(msgs):
    return {m.name: m for m in msgs}


# -------------------------------------------------------------------- jobs ---
#: `bg` is "transparent", "blur", or an "#rrggbb" to flatten onto.
JOBS = _index([
    Message("single", [req("input"), req("output"), req("model"),
                       opt("alpha"), opt("bg"), opt("blur")],
            "Remove the background from one image."),
    Message("batch", [req("input_dir"), req("output_dir"), req("model"),
                      opt("alpha"), opt("bg"), opt("blur"), opt("pattern")],
            "A whole folder."),
    Message("gif", [req("input"), req("output"), req("model"),
                    opt("alpha"), opt("bg"), opt("blur"),
                    opt("rot"), opt("fh"), opt("fv")],
            "Per-frame background removal, re-encoded as a GIF."),
    Message("cancel", [], "Stop a single/batch/gif in progress."),
    Message("models", [], "What models are installed."),
    Message("unload", [opt("scope")], "Release a model from the GPU."),

    Message("seg_load", [req("input"), opt("model")],
            "Encode an image for SAM. model: auto|large|base_plus|small|tiny|mobile."),
    Message("seg_everything", [opt("points_per_side")],
            "Find every object (the AMG pass)."),
    Message("seg_point", [opt("points"), opt("use_prev")],
            "Refine one object from click prompts: [[x, y, label], ...]."),
    Message("seg_extract", [req("output"), opt("ids"), opt("mask"),
                            opt("bg"), opt("blur"),
                            opt("rot"), opt("fh"), opt("fv")],
            "Composite the selected objects (or one mask) onto a background. "
            "The masks live in un-rotated source space, so rot/fh/fv carry the "
            "view transform the user was looking at and get baked into the "
            "output — what you save is what you saw."),
    Message("seg_cancel", [], "Stop a segmentation pass in progress."),
    Message("shutdown", [], "Exit the worker."),
])

#: Which image fields of a job carry an ImagePayload rather than a plain value.
JOB_IMAGES = {
    "single": ("input", "output"),
    "gif": ("input", "output"),
    "seg_load": ("input",),
    "seg_extract": ("output", "mask"),
}


# ------------------------------------------------------------------ events ---
EVENTS = _index([
    Message("ready", [opt("models"), opt("providers"), opt("seg"),
                      opt("bgremove"), opt("seg_models")],
            "Sent once at startup: what this worker can actually do."),
    Message("loading", [req("model")], "A model is being loaded."),
    Message("device", [req("provider"), opt("label"), opt("gpu")],
            "Which execution provider won."),
    Message("notice", [req("message")],
            "Something the user should know but that is not a failure — e.g. "
            "the GPU could not run this and we fell back to the CPU."),
    Message("progress", [req("done"), req("total"), opt("name")], "Batch progress."),
    Message("done_single", [req("output"), opt("preview"), opt("seconds")], ""),
    Message("done_batch", [req("count"), opt("seconds"), opt("outdir")], ""),
    Message("gif_progress", [req("done"), req("total")], ""),
    Message("gif_done", [req("output"), opt("frames"), opt("seconds")], ""),
    Message("canceled", [req("scope"), opt("done"), opt("total")],
            "scope: single|batch|gif."),
    Message("unloaded", [opt("scope")], ""),

    Message("seg_download", [req("done"), req("total")],
            "Fetching model weights."),
    Message("seg_step", [opt("rung"), req("message")],
            "Progress through the model ladder."),
    Message("seg_ready", [opt("rung"), opt("model"), opt("provider"),
                          opt("label"), opt("gpu"), opt("mode"), opt("family")],
            "The image is encoded and SAM is resident. mode: auto|manual|fallback."),
    Message("seg_progress", [req("done"), req("total")], "The AMG grid pass."),
    Message("seg_objects", [req("label_map"), req("count"), req("objects"),
                            opt("general_map"), opt("depth_map"), opt("seconds")],
            "Every object found, with its mask, colour, contour and bbox, plus "
            "the three per-pixel lookup maps (specific / general / depth)."),
    Message("seg_mask", [req("mask"), opt("score"), opt("bbox")],
            "The point-mode object."),
    Message("seg_extracted", [req("output"), opt("seconds")], ""),
    Message("seg_canceled", [], ""),
    Message("error", [req("message")], ""),
])

EVENT_IMAGES = {
    "done_single": ("output", "preview"),
    "gif_done": ("output",),
    "seg_objects": ("label_map", "general_map", "depth_map"),
    "seg_mask": ("mask",),
    "seg_extracted": ("output",),
}


# ------------------------------------------------------------------ codec ----
class ProtocolError(ValueError):
    pass


def _check(catalogue, key, d, kind, strict):
    name = d.get(key)
    if name not in catalogue:
        raise ProtocolError("unknown %s %r" % (kind, name))
    m = catalogue[name]
    for f in m.required():
        if f not in d:
            raise ProtocolError("%s %r is missing required field %r"
                                % (kind, name, f))
    if strict:
        extra = set(d) - m.known() - {key}
        if extra:
            raise ProtocolError("%s %r has undeclared field(s) %s"
                                % (kind, name, ", ".join(sorted(extra))))
    return m


def validate_job(d, strict=False):
    """Raises ProtocolError if `d` is not a job this protocol declares."""
    return _check(JOBS, "op", d, "job", strict)


def validate_event(d, strict=False):
    return _check(EVENTS, "type", d, "event", strict)


def job(op, **fields):
    """Build a job, checked against the catalogue.

    ImagePayloads encode themselves; a plain path string still works, which is
    what keeps the desktop's wire format unchanged.
    """
    d = {"op": op}
    for k, v in fields.items():
        if v is None:
            continue
        d[k] = v.encode() if isinstance(v, ImagePayload) else v
    validate_job(d)
    return d


def event(type, **fields):
    d = {"type": type}
    for k, v in fields.items():
        if v is None:
            continue
        d[k] = v.encode() if isinstance(v, ImagePayload) else v
    validate_event(d)
    return d


def images_of(d):
    """The ImagePayloads carried by a job or event, by field name."""
    if "op" in d:
        names = JOB_IMAGES.get(d["op"], ())
    else:
        names = EVENT_IMAGES.get(d.get("type"), ())
    out = {}
    for n in names:
        if d.get(n) is not None:
            out[n] = ImagePayload.decode(d[n])
    return out
