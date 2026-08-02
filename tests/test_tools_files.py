from __future__ import annotations

from turnloop.tools.edit import EditArgs, EditOp, EditTool
from turnloop.tools.glob import GlobArgs, GlobTool
from turnloop.tools.grep import GrepArgs, GrepTool
from turnloop.tools.read import ReadArgs, ReadTool
from turnloop.tools.write import WriteArgs, WriteTool


async def read(ctx, path, **kw):
    return await ReadTool().run(ReadArgs(file_path=path, **kw), ctx)


# --------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------


async def test_read_numbers_lines_from_one(ctx, project):
    (project / "a.py").write_text("first\nsecond\n", encoding="utf-8")
    out = await read(ctx, "a.py")
    assert not out.is_error
    assert out.content.splitlines()[0].endswith("\tfirst")
    assert "     2\tsecond" in out.content


async def test_read_paging_reports_how_to_continue(ctx, project):
    (project / "big.txt").write_text("\n".join(str(i) for i in range(100)), encoding="utf-8")
    out = await read(ctx, "big.txt", offset=1, limit=10)
    assert "offset=11" in out.content


async def test_read_refuses_binary(ctx, project):
    (project / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    out = await read(ctx, "blob.bin")
    assert out.is_error and "binary" in out.content


async def test_read_refuses_outside_project(ctx, project):
    out = await read(ctx, str(project.parent / "outside.txt"))
    assert out.is_error and "outside the project root" in out.content


async def test_read_suggests_close_names(ctx, project):
    (project / "config.py").write_text("x", encoding="utf-8")
    out = await read(ctx, "cofnig.py")
    assert out.is_error and "config.py" in out.content


async def test_read_returns_an_image_on_a_vision_provider(ctx, project):
    ctx.session.provider = "anthropic"  # configured with supports_vision=True
    (project / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    out = await read(ctx, "pic.png")
    assert not out.is_error
    assert out.image is not None
    assert out.image.media_type == "image/png"


async def test_read_refuses_an_image_on_a_non_vision_provider(ctx, project):
    """GLM is text-only; sending it an image payload would just get rejected."""
    ctx.session.provider = "glm"
    (project / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    out = await read(ctx, "pic.png")
    assert out.is_error
    assert "glm" in out.content and "vision" in out.content
    assert out.image is None


async def test_image_detection_goes_by_magic_bytes_not_a_lying_extension(ctx, project):
    ctx.session.provider = "anthropic"
    (project / "not_really.txt").write_bytes(b"\xff\xd8\xff" + b"\x00" * 16)  # JPEG signature
    out = await read(ctx, "not_really.txt")
    assert not out.is_error
    assert out.image.media_type == "image/jpeg"


async def test_partial_read_does_not_authorize_an_edit(ctx, project):
    """A read of lines 1-2 must not license an edit against unseen lines."""
    (project / "a.py").write_text("\n".join(f"line{i}" for i in range(50)), encoding="utf-8")
    await read(ctx, "a.py", offset=1, limit=2)
    out = await EditTool().run(
        EditArgs(file_path="a.py", edits=[EditOp(old_string="line40", new_string="x")]), ctx
    )
    assert out.is_error and "has not been read" in out.content


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------


async def test_write_creates_a_new_file(ctx, project):
    out = await WriteTool().run(WriteArgs(file_path="new.txt", content="hello\n"), ctx)
    assert not out.is_error
    assert (project / "new.txt").read_text(encoding="utf-8") == "hello\n"


async def test_write_refuses_to_overwrite_an_unread_file(ctx, project):
    (project / "exists.txt").write_text("important\n", encoding="utf-8")
    out = await WriteTool().run(WriteArgs(file_path="exists.txt", content="gone"), ctx)
    assert out.is_error and "has not been read" in out.content
    assert (project / "exists.txt").read_text(encoding="utf-8") == "important\n"


async def test_write_refuses_when_the_file_changed_underneath(ctx, project):
    path = project / "f.txt"
    path.write_text("v1\n", encoding="utf-8")
    await read(ctx, "f.txt")
    import os
    import time

    time.sleep(0.01)  # noqa: ASYNC251 - a real mtime change is the point of the test
    path.write_text("v2 from elsewhere\n", encoding="utf-8")
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))

    out = await WriteTool().run(WriteArgs(file_path="f.txt", content="v3"), ctx)
    assert out.is_error and "changed on disk" in out.content


async def test_write_preserves_crlf(ctx, project):
    path = project / "crlf.txt"
    path.write_bytes(b"a\r\nb\r\n")
    await read(ctx, "crlf.txt")
    await WriteTool().run(WriteArgs(file_path="crlf.txt", content="x\ny\n"), ctx)
    assert path.read_bytes() == b"x\r\ny\r\n"


async def test_a_bom_is_hidden_from_the_model_but_preserved_on_disk(ctx, project):
    """A U+FEFF on line 1 breaks ast.parse and makes exact-match edits fail invisibly."""
    path = project / "bom.py"
    path.write_bytes(b"\xef\xbb\xbfimport os\n")

    out = await read(ctx, "bom.py")
    assert "﻿" not in out.content, "the BOM must not reach the model"
    assert "import os" in out.content

    from turnloop.tools.edit import EditArgs, EditOp, EditTool

    result = await EditTool().run(
        EditArgs(file_path="bom.py", edits=[EditOp(old_string="import os", new_string="import sys")]),
        ctx,
    )
    assert not result.is_error
    assert path.read_bytes() == b"\xef\xbb\xbfimport sys\n", "the file's encoding is preserved"


async def test_plan_mode_blocks_write_even_if_dispatch_were_bypassed(ctx, project):
    """Belt and braces: the tool itself refuses when the context is read-only."""
    ctx.readonly = True
    out = await WriteTool().run(WriteArgs(file_path="new.txt", content="x"), ctx)
    assert out.is_error and "read-only" in out.content
    assert not (project / "new.txt").exists()


# --------------------------------------------------------------------------
# Edit
# --------------------------------------------------------------------------


async def test_edit_applies_multiple_edits_in_order(ctx, project):
    (project / "m.py").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    await read(ctx, "m.py")
    out = await EditTool().run(
        EditArgs(
            file_path="m.py",
            edits=[
                EditOp(old_string="alpha", new_string="one"),
                EditOp(old_string="gamma", new_string="three"),
            ],
        ),
        ctx,
    )
    assert not out.is_error
    assert (project / "m.py").read_text(encoding="utf-8") == "one\nbeta\nthree\n"


async def test_edit_is_atomic_across_the_batch(ctx, project):
    """A failing later edit must leave the file completely untouched."""
    original = "alpha\nbeta\n"
    (project / "m.py").write_text(original, encoding="utf-8")
    await read(ctx, "m.py")
    out = await EditTool().run(
        EditArgs(
            file_path="m.py",
            edits=[
                EditOp(old_string="alpha", new_string="one"),
                EditOp(old_string="does-not-exist", new_string="x"),
            ],
        ),
        ctx,
    )
    assert out.is_error
    assert (project / "m.py").read_text(encoding="utf-8") == original


async def test_edit_rejects_ambiguous_match_with_a_count(ctx, project):
    (project / "d.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    await read(ctx, "d.py")
    out = await EditTool().run(
        EditArgs(file_path="d.py", edits=[EditOp(old_string="x = 1", new_string="x = 2")]), ctx
    )
    assert out.is_error and "appears 2 times" in out.content


async def test_edit_replace_all(ctx, project):
    (project / "d.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    await read(ctx, "d.py")
    out = await EditTool().run(
        EditArgs(
            file_path="d.py",
            edits=[EditOp(old_string="x = 1", new_string="x = 2", replace_all=True)],
        ),
        ctx,
    )
    assert not out.is_error
    assert (project / "d.py").read_text(encoding="utf-8") == "x = 2\nx = 2\n"


async def test_edit_diagnoses_a_whitespace_mismatch(ctx, project):
    (project / "w.py").write_text("    indented = 1\n", encoding="utf-8")
    await read(ctx, "w.py")
    out = await EditTool().run(
        EditArgs(file_path="w.py", edits=[EditOp(old_string="indented = 1\n", new_string="x")]),
        ctx,
    )
    assert out.is_error and "whitespace" in out.content


async def test_edit_preserves_trailing_newline_convention(ctx, project):
    path = project / "n.py"
    path.write_bytes(b"a = 1")  # no trailing newline
    await read(ctx, "n.py")
    await EditTool().run(
        EditArgs(file_path="n.py", edits=[EditOp(old_string="a = 1", new_string="a = 2\n")]), ctx
    )
    assert path.read_bytes() == b"a = 2"


# --------------------------------------------------------------------------
# Glob / Grep
# --------------------------------------------------------------------------


async def test_glob_implies_recursive_for_a_bare_pattern(ctx, project):
    (project / "src").mkdir()
    (project / "src" / "deep.py").write_text("x", encoding="utf-8")
    (project / "top.py").write_text("x", encoding="utf-8")
    out = await GlobTool().run(GlobArgs(pattern="*.py"), ctx)
    assert "src/deep.py" in out.content and "top.py" in out.content


async def test_glob_skips_dependency_directories(ctx, project):
    (project / "node_modules" / "pkg").mkdir(parents=True)
    (project / "node_modules" / "pkg" / "index.js").write_text("x", encoding="utf-8")
    (project / "app.js").write_text("x", encoding="utf-8")
    out = await GlobTool().run(GlobArgs(pattern="**/*.js"), ctx)
    assert "app.js" in out.content
    assert "node_modules" not in out.content


async def test_grep_default_mode_returns_paths_only(ctx, project):
    (project / "a.py").write_text("needle here\nother\n", encoding="utf-8")
    (project / "b.py").write_text("nothing\n", encoding="utf-8")
    out = await GrepTool().run(GrepArgs(pattern="needle"), ctx)
    assert "a.py" in out.content
    assert "needle here" not in out.content, "default mode must not spend context on lines"


async def test_grep_content_mode_includes_line_numbers(ctx, project):
    (project / "a.py").write_text("one\nneedle\n", encoding="utf-8")
    out = await GrepTool().run(GrepArgs(pattern="needle", output_mode="content"), ctx)
    assert ":2:" in out.content


async def test_grep_reports_an_invalid_regex_instead_of_raising(ctx, project):
    out = await GrepTool().run(GrepArgs(pattern="a[b"), ctx)
    assert out.is_error and "invalid regular expression" in out.content


def _init_repo(root):
    """A minimal real git repo, since the bug only reproduces against real
    `git ls-files` output — the mock-free tools have no other way to observe it."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)


def _git_add(root, *paths):
    import subprocess

    subprocess.run(["git", "add", *paths], cwd=root, check=True)


async def test_glob_finds_files_inside_a_git_ignored_root(ctx, project):
    """Regression: rooting Glob inside an ignored directory (e.g. any
    .turnloop/experiments/* workspace) must not silently return zero matches.
    `git ls-files` reports nothing for an ignored subtree, and that used to be
    taken at face value as "no files exist here"."""
    _init_repo(project)
    (project / ".gitignore").write_text("ignored_dir/\n", encoding="utf-8")
    ignored = project / "ignored_dir"
    ignored.mkdir()
    (ignored / "a.py").write_text("x", encoding="utf-8")
    (ignored / "b.py").write_text("x", encoding="utf-8")

    out = await GlobTool().run(GlobArgs(pattern="**/*.py", path="ignored_dir"), ctx)
    assert "a.py" in out.content and "b.py" in out.content


async def test_glob_still_excludes_ignored_files_when_the_root_is_not_ignored(ctx, project):
    """The fallback above must not regress the ordinary case: a git-ignored
    file inside a repo whose root is NOT ignored still has to be excluded."""
    _init_repo(project)
    (project / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (project / "secret.py").write_text("x", encoding="utf-8")
    (project / "tracked.py").write_text("x", encoding="utf-8")
    _git_add(project, "tracked.py")

    out = await GlobTool().run(GlobArgs(pattern="**/*.py"), ctx)
    assert "tracked.py" in out.content
    assert "secret.py" not in out.content
