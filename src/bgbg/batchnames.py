"""Output names for a batch run, and putting each finished result under one.

Standard library only: the worker names and publishes the outputs, and the
frontend can check a rename pattern with the same rules while it is typed.
"""
import os
import secrets
import string

# The tokens a rename pattern may use, and what each one is.
TOKENS = {
    "name": "the source file's name, without its extension",
    "n": "the file's number in the batch, from 1",
}
# Characters Windows refuses in a file name, refused everywhere so a pattern
# that works on one system works on the other.
_BAD_CHARS = '<>:"/\\|?*'
# What a name may not be, whatever made it: these would leave the output folder.
_SEPARATORS = {"/", "\\", os.sep}
_NOT_FILE_NAMES = {"", ".", ".."}


class BatchPatternError(ValueError):
    """A rename pattern that can't name files.

    `user_message` is the sentence the window shows.
    """

    def __init__(self, pattern, reason, user_message):
        super().__init__(f"bad rename pattern {pattern!r}: {reason}")
        self.pattern = pattern
        self.reason = reason
        self.user_message = user_message


def _check_tokens(pattern):
    """Only bare {name} and {n}: a format spec, a conversion or a nested field
    could make one file's name leave the folder ("{n:c}" makes item 47 "/")."""
    try:
        parsed = list(string.Formatter().parse(pattern))
    except ValueError as err:
        raise BatchPatternError(pattern, str(err),
                                "The rename pattern has a { or } without its pair.") from err
    for _, field, spec, conversion in parsed:
        if field is None:
            continue
        if field not in TOKENS:
            raise BatchPatternError(
                pattern, f"unknown token {{{field}}}",
                "The rename pattern uses {%s}, which isn't a token. Use {name} or {n}."
                % field)
        if spec or conversion:
            raise BatchPatternError(
                pattern, f"{{{field}}} has a format spec or conversion",
                "The rename pattern can only use {name} and {n}, with nothing else in "
                "the braces.")


def stem_for(pattern, source, n):
    """The output name (without extension) `pattern` gives `source`, the n-th file.

    Raises:
        BatchPatternError: the pattern uses more than bare tokens, or the name
            it gives isn't a plain file name (empty, "..", or with a separator).
    """
    _check_tokens(pattern)
    name = os.path.splitext(os.path.basename(source))[0]
    stem = pattern.format(name=name, n=n)
    if stem.strip() in _NOT_FILE_NAMES or set(stem) & _SEPARATORS:
        raise BatchPatternError(pattern, f"{stem!r} is not a file name",
                                "The rename pattern gives %r, which can't be a file name."
                                % stem)
    return stem


def check_pattern(pattern):
    """Raise BatchPatternError if `pattern` can't name files.

    On top of stem_for's checks, the pattern's own text may not hold a folder
    separator or a character Windows refuses: outputs stay in the output folder.
    A source's own name is not checked here, since it is already a valid name.
    """
    _check_tokens(pattern)
    bad = sorted(set(pattern.format(name="example", n=1)) & set(_BAD_CHARS))
    if bad:
        raise BatchPatternError(
            pattern, f"name has {''.join(bad)}",
            "File names can't contain %s, so the rename pattern can't either." % " ".join(bad))
    stem_for(pattern, "example.png", 1)


def _path_key(path):
    """A path as a case-insensitive disk compares it.

    Lowercased on every system: Windows and macOS disks ignore case, so two
    names that differ only in case must never be handed out as two outputs.
    """
    return os.path.normcase(os.path.abspath(path)).casefold()


def new_part(outdir, stem, suffix):
    """A new, empty, hidden file in `outdir` for one result to be written into.

    It is the batch's own: nothing else has its name, so writing it, or
    deleting it after a failure, can't touch anyone's file. publish() then
    gives it its real name. It ends in `suffix` (".png", ".webp", ".jpg"),
    since the worker picks the format to write from the name. Created like
    any file (mode 0666 less the umask), since it becomes the result.
    """
    while True:
        part = os.path.join(outdir, ".%s.%s.part%s" % (stem, secrets.token_hex(4), suffix))
        try:
            os.close(os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
            return part
        except FileExistsError:
            continue


def _candidates(outdir, stem, suffix):
    """The names a result may take, in order: ``stem.png``, ``stem (2).png``..."""
    yield os.path.join(outdir, stem + suffix)
    n = 2
    while True:
        yield os.path.join(outdir, "%s (%d)%s" % (stem, n, suffix))
        n += 1


def publish(part, outdir, stem, suffix, taken):
    """Give the finished `part` the first free name, ``stem.png``, ``stem (2).png``,
    ``stem (3).png``... (in `suffix`), and return that path.

    A name is taken only by an operation that fails when the name exists, so
    a file that appears at any moment, even while its result was rendered, is
    never written over. A name in `taken` (the batch's sources and earlier
    outputs, as _path_key keys) is skipped too. The published path is added
    to `taken`; `part` is gone afterwards.
    """
    for path in _candidates(outdir, stem, suffix):
        if _path_key(path) in taken:
            continue
        try:
            _move_no_replace(part, path)
        except FileExistsError:
            continue
        taken.add(_path_key(path))
        return path


def first_free(outdir, stem, suffix, taken):
    """The name publish() would give a result if it ran now; it claims nothing.

    For showing a run's names ahead of it: a file that appears before the
    result is written still moves it to the next number.
    """
    for path in _candidates(outdir, stem, suffix):
        if _path_key(path) not in taken and not os.path.lexists(path):
            return path


def _move_no_replace(src, dst):
    """Move `src` to `dst`, raising FileExistsError instead of replacing a file.

    A hard link fails when `dst` exists, so link-then-unlink never replaces.
    """
    try:
        os.link(src, dst)
    except FileExistsError:
        raise
    except OSError:
        # No hard links on this disk (FAT, exFAT, some network shares).
        # NOTE: claim-then-replace has a race the link doesn't: a file put at
        # `dst` between the two calls would be replaced.
        os.close(os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
        os.replace(src, dst)
        return
    os.remove(src)


def taken_by(files):
    """The _path_key keys of `files`, for publish to keep clear of."""
    return {_path_key(f) for f in files}
