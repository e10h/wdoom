# Agent instructions

# Memory Safety & Debugging

## Lessons from C → Rust Porting

- **Null-terminated strings for C FFI**: Pass `&[u8] = b"/\0"` instead of `&str = "/"` when C code will call `strlen` on the pointer. `&str` is not null-terminated and causes `global-buffer-overflow`.
- **Avoid `ptr::write_bytes` for precise zeroing**: `ptr::write_bytes` compiles to `memset`, which under ASan may use SIMD writes that overshoot non-aligned sizes and corrupt adjacent allocator metadata. Use a byte-by-byte loop instead.
- **Have allocators zero internally**: Rather than relying on callers to `memset`, zero user data inside `Z_Malloc` before returning.
- **Struct padding fields and zero-initialization**: When porting C structs that contain private `_pad` fields, you cannot use struct literal syntax. Use `std::mem::zeroed()` or `MaybeUninit::zeroed().assume_init()` instead.
- **Moving globals between ported modules**: When a global was previously accessed via `extern "C"` in one Rust module and the C source gets ported, move the `#[no_mangle] pub static mut` definition to the new module and update the consumer to access it directly (e.g., `crate::doom::r_things::spryscale`).
- **Unsigned angle arithmetic wrapping**: C `angle_t` is `u32`, and subtraction/addition can wrap around zero. Rust's default `-` and `+` on `u32` panic in debug mode. Always use `wrapping_sub`, `wrapping_add`, `wrapping_neg` for angle arithmetic.
- **Opaque `state_t` vs concrete `State` struct**: `d_player.rs` declares `state_t` as an empty enum for FFI, but `info.rs` defines the real `State` struct. When accessing state fields from ported code, cast the pointer to `*mut State`.

## AddressSanitizer

ASan gives exact line numbers for memory corruption across the Rust/C boundary. It is especially useful for `unsafe`, FFI, raw pointers, manual buffers, and ownership mistakes.

### Quick Start

```bash
# Via Taskfile
task asan:test -- test_name -- --nocapture

# Manual
ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:abort_on_error=1:symbolize=1" \
RUST_BACKTRACE=1 \
RUSTFLAGS="-Zsanitizer=address" \
cargo +nightly test -Zbuild-std --target x86_64-unknown-linux-gnu
```

Use `-Zbuild-std` so `std` is also instrumented. Compile the C side with ASan too by setting `ASAN=1` (gated in `doomgeneric-sys/build.rs`).

### Useful `ASAN_OPTIONS` Flags

| Flag | Meaning |
|------|---------|
| `detect_leaks=1` | Also report leaks |
| `halt_on_error=1` | Stop at first error |
| `abort_on_error=1` | Generate a hard crash for debuggers/agents |
| `symbolize=1` | Print readable stack traces |

### What ASan Reports Look Like

```text
ERROR: AddressSanitizer: heap-use-after-free
READ of size 8 at 0x...
    #0 my_crate::module::function src/foo.rs:123
    #1 my_crate::ffi_wrapper::call src/ffi.rs:45

freed by thread T0 here:
    #0 free
    #1 native_destroy src/native/foo.c:88

previously allocated by thread T0 here:
    #0 malloc
    #1 native_create src/native/foo.c:42
```

ASan reports three locations: the bad access, the free, and the allocation.

### Limitations

- Needs nightly Rust.
- Slows execution and increases memory use.
- May conflict with proc macros or dynamic libraries.
- Best on Linux/macOS x86_64/aarch64.
- Does not replace Miri or prove memory safety.

## c2rust Intermediate Reference

A fully-automated `c2rust transpile` output lives in `c2rust-intermediate/`. It is **not** linked into the main binary and is excluded from the default workspace build (`default-members`). Its sole purpose is as a behavioural reference when porting or debugging C modules.

### Regenerating

```bash
./tools/c2rust-transpile.sh
```

This script:
1. Scans `vendor/doomgeneric/*.c` and filters out non-transpilable files.
2. Emits `compile_commands.json` with the exact flags from `doomgeneric-sys/build.rs`.
3. Runs `c2rust transpile --emit-build-files --overwrite-existing`.
4. Fixes up `Cargo.toml` and `src/lib.rs` so the crate is usable.

### Checking the reference crate

The transpiled code requires nightly because c2rust emits `extern type` declarations (still unstable):

```bash
cargo +nightly check -p c2rust-intermediate
```

### Excluded files

| File | Reason |
|------|--------|
| `layout_probe.c` | Explicitly excluded from upstream Makefile |
| `gusconf.c` | Requires `FEATURE_SOUND` |
| `m_misc.c` | Contains variadic macros (`M_StringJoin`, `M_vsnprintf`) that crash c2rust |
| `dummy.c` | Empty stub |
| `doomdef.c` | Header-only in practice, no symbols |

### How to use it

- **Type layout validation**: Compare `#[repr(C)]` struct definitions against `room/src/doom/c_ffi.rs`.
- **Behavioural comparison**: When a hand-ported module behaves differently, compare its logic to the transpiled version (which faithfully reproduces C semantics).
- **Symbol inventory**: See exactly which functions, globals, and types a given C module exports before porting it.

### Caveats

- All code is `unsafe` and non-idiomatic — do not copy-paste into the hand-ported codebase.
- Duplicate type definitions exist across modules (e.g. `mobj_t` appears in many files). This is expected because each transpiled file is standalone.
- `#define` values are baked in at transpile time.
- Cross-module calls remain `extern "C"` FFI; there are no Rust `use` imports between modules.

## dhat Heap Profiler

For callsite-level allocation tracking (volume and ownership paths), use `dhat` separately from ASan.

Enable in `room/Cargo.toml`:

```toml
[features]
dhat-heap = ["dep:dhat"]

[dependencies]
dhat = { version = "0.3", optional = true }
```

Run:

```bash
cargo run --features dhat-heap
cargo test --test demo_playthrough --features dhat-heap
```

A `dhat-heap.json` file is produced; view it with the [dhat viewer](https://valgrind.org/docs/manual/dh-manual.html).

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **wdoom** (74493 symbols, 88542 relationships, 632 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact before editing.** Use `impact({target: "symbolName", direction: "upstream"})` or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .`; report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- MUST warn on HIGH/CRITICAL `risk` pre-edit; never use `riskSharedAxes` to waive a HIGH/CRITICAL `risk` warning. Compare File/symbol: MCP File omits axes; Graph-RAG expands File.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- **MUST use `query({search_query: "concept"})` for concepts/flows, `context({name: "symbolName"})` for a named symbol, or `impact` for blast radius, on read-only callers, dependencies, imports, or execution flow.** Graph first; text search only for empty/`UNKNOWN`/literals.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/wdoom/context` | Codebase overview, check index freshness |
| `gitnexus://repo/wdoom/clusters` | All functional areas |
| `gitnexus://repo/wdoom/processes` | All execution flows |
| `gitnexus://repo/wdoom/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
