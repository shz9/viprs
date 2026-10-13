# VIPRS in the browser

[Launch the VIPRS browser application](webapp/index.html)

The application runs the actual `VIPRS` and `VIPRSMix` interfaces in a Pyodide Web Worker.
Each build compiles VIPRS from the checked-out source and uses the latest published
magenpy release, plus pinned Zarr storage dependencies. A compatible magenpy WASM
release asset is reused; if absent, its released source is compiled. Exact versions,
commits, and the magenpy wheel's origin are recorded in the generated build manifest.
The Cython/C++ inference and LD kernels execute in WebAssembly on the user's device.

## Run it

The published documentation includes a tested WASM bundle. Source checkouts do
**not** contain generated wheels or manifests. Before serving a fresh checkout,
build the bundle below, or download `viprs-webapp-wasm.zip` and
`viprs-webapp-wasm.sha256` from a [GitHub Release](https://github.com/shz9/viprs/releases)
that contains these assets. For example, replace `vX.Y.Z` with that release's tag:

```sh
gh release download vX.Y.Z --repo shz9/viprs --pattern 'viprs-webapp-wasm.*' --dir dist/wasm
(cd dist/wasm && shasum -a 256 -c viprs-webapp-wasm.sha256)
python -m zipfile -e dist/wasm/viprs-webapp-wasm.zip docs
```

This installs the complete app and wheels from that release together. Build locally
instead when testing changes to the current source. Then, from the repository root:

```sh
python -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open **http://localhost:8000/webapp/**. The runtime and packages initialize in the
background immediately, while you configure inputs. Click **Run inference** to fit
the selected model; if initialization is still in progress, the run waits for it.
Completed runs reuse the same initialized worker. Visitors to the deployed site
do not need a compiler or Python installed for inference.
The local Python command above only serves static files. `mkdocs serve` also works,
and `mkdocs build` copies the application and wheels into the published site.

An internet connection is required for Pyodide's pinned scientific runtime packages
and the default data downloads. Use HTTP(S); opening `index.html` with `file://` does
not support the worker and asset requests. The application documentation links point
to the built MkDocs site.

## Default test data

The default example uses the complete supplied chromosome 22 datasets:

- LD: `hf://datasets/shz9/ukb-ld/EUR/chr_22.zip`, fetched from
  [Hugging Face](https://huggingface.co/datasets/shz9/ukb-ld/resolve/main/EUR/chr_22.zip).
- Summary statistics: [UK Biobank height fastGWA](https://github.com/shz9/magenpy/blob/master/magenpy/data/ukb_height_chr22.fastGWA.gz),
  fetched as the [raw gzip file](https://raw.githubusercontent.com/shz9/magenpy/master/magenpy/data/ukb_height_chr22.fastGWA.gz).

The compressed downloads total about 4.6 MB. The browser resolves `hf://datasets/…`
and GitHub `blob` links into direct HTTPS downloads. It also accepts ordinary HTTP(S)
URLs, relative web paths, or locally selected files.

In prototype verification with VIPRS 0.1.5, seed zero, 200 maximum iterations,
tolerance `1e-6`, and `fastGWA` format, the full browser test harmonized **14,518 variants** and converged
in **77 iterations**. Exported posterior means and variances matched the native
VIPRS run at the exported precision; the maximum PIP difference was approximately
`7e-15`. These are verification results for the supplied data, rather than a general
accuracy or performance guarantee.

## Inputs and results

Select your summary-statistics format and supply a plain-text or gzipped summary
statistics file, or select several files with distinct chromosomes. Hosted input
fields accept one URL per line. Select one or more LD ZIPs containing complete
**magenpy Zarr v2 stores**, preserving `.zgroup`, `.zattrs`, `matrix/`, metadata, and
all chunks. Both an archive containing a store directory and a store at archive root
are accepted. The default LD archive already has this structure.

Chromosomes retained by GWADataLoader are fitted together with shared model
hyperparameters. Summary-statistics files and extracted LD stores are passed jointly
to its constructor, which handles reading, chromosome matching, and harmonization.
A combined summary-statistics file and ZIP containing multiple chromosome stores
also work. The browser adapter uses the loader's input-handling behavior.

Choose **VIPRS** for a spike-and-slab prior or **VIPRSMix** for a sparse Gaussian
mixture. VIPRSMix exposes the number of non-null components (1–10), in addition to
the point mass at zero. Optimization settings include the maximum iteration count,
convergence tolerance, and random seed. Defaults are 1,000 iterations, as in
`viprs_fit`, and a tolerance of `1e-6` for both parameters and ELBO, as in
`VIPRS.fit`.

The worker stages inputs in its in-memory filesystem, extracts the LD archives,
and makes one uninterrupted `fit()` call. For example, with a combined summary file:

```python
from magenpy import GWADataLoader
from viprs import VIPRS

gdl = GWADataLoader(
    sumstats_files=sumstats_path,
    sumstats_format="fastGWA",
    ld_store_files=extracted_ld_store_paths,
    threads=1,
)
model = VIPRS(gdl, threads=1, float_precision="float64")
model.fit(max_iter=1000, x_abs_tol=1e-6, f_abs_tol=1e-6, disable_pbar=True)
weights = model.to_table()
```

magenpy handles harmonization; VIPRS learns its hyperparameters through variational
EM. Model equations and inference kernels are the repository implementation.
There are no per-iteration UI callbacks: a rotating activity indicator stays visible
until fitting completes, then results and the complete ELBO history become available.
The random seed defaults to zero. Initialization, downloads, data preparation, and
model fitting are timed separately in the report; the displayed run timer excludes
runtime initialization.

Download the complete posterior table as TSV (`BETA`, `PIP`, `VAR_BETA` and variant
metadata), or a JSON run report including convergence status, ELBO history, options,
package versions, loaded extension paths, input locations, and wheel checksums.
The table displays the 20 variants with highest posterior inclusion probability,
with an optional chromosome filter. Downloads include all fitted variants.
An iteration-limited run is labeled separately from a converged run.

The output is the unmodified `VIPRS.to_table()` result. Individual polygenic scores
still require target genotypes and the appropriate genotype scaling; the application
currently fits variant weights only.

## Bundled artifacts

`docs/webapp/runtime.json` defines the exact Pyodide version, its required built-in
packages, and the local wheel URLs with SHA-256 hashes. `docs/webapp/build-info.json`
records the source commit and hashes, the resolved magenpy release and commit, compiled modules,
and build tools.
The application installs these explicit wheel files without resolving packages from
PyPI at run time. Standard scientific packages are loaded from the pinned Pyodide CDN.

The five compiled extension modules are:

- `viprs.model.vi.e_step_cpp`
- `viprs.utils.math_utils`
- `magenpy.stats.ld.c_utils`
- `magenpy.stats.score.score_cpp`
- `magenpy.stats.variant.variant_cpp`

They have the Python 3.13 / `pyodide_2025_0_wasm32` wheel tag and contain WebAssembly
binary modules, despite Python's `.so` extension naming convention. The runtime is
Pyodide **0.28.3**, with Python **3.13.2** and Emscripten **4.0.9**.

## Continuous builds and storage

[`.github/workflows/wasm.yml`](https://github.com/shz9/viprs/blob/master/.github/workflows/wasm.yml)
is the entry point for building, testing, and distributing the application.

| Event | Build and test | Destination |
| --- | --- | --- |
| Push or merge to `main` / `master` | Build VIPRS; reuse magenpy WASM or build its release source | Actions artifact, retained for 30 days |
| Pull request, including updates | Same build and tests | Actions artifact only; no deployment or release write access |
| Published GitHub Release (including prereleases) | Build the release tag | Actions artifact plus assets attached to that release |
| Manual **Run workflow** | Build the selected branch or tag | Actions artifact only |
| Successful default-branch push build | Reuse the tested bundle | Deploy documentation and application to `gh-pages` |

A merge is already a push event; no separate merge trigger is needed. Feature
branches are checked through pull requests or the manual trigger. All main/master
pushes are built, including documentation-only changes, so documentation deployment
always uses an app and bundle from the same commit. A newer push cancels a previous
unfinished build on that branch. Release builds are not cancelled by newer pushes.
Only the pinned Emscripten SDK and pip downloads are cached. VIPRS is rebuilt;
magenpy is downloaded from its own release when a compatible wheel is available. The compiler configuration remains Python 3.13, Pyodide 0.28.3, and
Emscripten 4.0.9.

Before publication, CI checks wheel hashes and actual WASM extension bytes, compares
bundled model source with the checkout, runs native adapter tests, and runs the real
chromosome 22 and two-chromosome browser tests in Chromium. Browser reports and
screenshots are retained for 14 days, including partial reports on failure. Tests
need access to the Pyodide CDN and the public example data hosts.

The distributable files are:

- **`viprs-webapp-wasm.zip`**: the static `webapp/` directory, all manifest-listed
  wheels, runtime and build manifests, license, and serving instructions.
- **`viprs-webapp-wasm.sha256`**: a checksum for the archive. Each wheel also has
  its own checksum in the runtime manifest and build provenance.

`python tools/wasm/package.py` creates these files under `dist/wasm/` after checking
the bundle. It excludes stale, unlisted wheels. Actions artifacts are convenient
for testing and transferring files between jobs, but expire; GitHub Release assets
are the versioned distribution. The app serves wheels alongside its own static
files on Pages, rather than fetching a moving "latest" release during initialization.
Keep `docs/webapp/wheels/`, `runtime.json`, and `build-info.json` out of source control.
The existing local builds are still usable; the ignore rules do not delete them.

The docs workflow is now reusable and called by the WASM workflow after its tests
pass. It verifies the downloaded archive, extracts it into `docs/`, and uses the
existing `mkdocs gh-deploy` publishing mechanism. Keep GitHub Pages configured to
serve the `gh-pages` branch. It is intentionally not deployed from PRs, releases,
or manual builds. A failed build leaves the previous site in place.

The normal PyPI release workflow remains separate. These browser wheels are shipped
as GitHub Release assets, not mixed into native PyPI wheel uploads. Release attachment
uses the repository's built-in `GITHUB_TOKEN` with `contents: write`; no new secret
is needed. Rerunning the release workflow replaces only the two named WASM assets.
Actions must be enabled and repository rules must permit release uploads and the
existing `gh-pages` push. A release tag must include the workflow and WASM sources.

If the repository enables **immutable releases**, assets must be uploaded while the
release is still a draft; GitHub prohibits adding assets after publication. For
that setting, run this workflow manually for the intended tag, download its tested
artifact, and attach both files to the draft before publishing. The published-release job checks that both assets are already attached and leaves
them unchanged for immutable releases. It fails with instructions if either asset
is missing; it cannot repair an already-published immutable release.

See GitHub's [artifact storage documentation](https://docs.github.com/en/actions/tutorials/store-and-share-data),
[release asset upload command](https://cli.github.com/manual/gh_release_upload), and
[immutable release restrictions](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

## Rebuild the WASM bundle

Build wheels locally after changing VIPRS source, or let GitHub Actions build them.
Generated wheels and manifests are ignored by Git; keep the build scripts and app
source in the repository.
The build script uses an isolated copy of the working tree under `build/wasm/`,
preserves existing source edits, and resolves magenpy's latest stable GitHub release
to an exact commit once per build. It looks for a matching
`magenpy-VERSION-cp313-cp313-pyodide_2025_0_wasm32.whl` release asset. Native wheels,
other Python/Pyodide ABIs, and wheels for another package version are not reused.

A reused wheel is verified against GitHub's SHA-256 asset digest, or the adjacent
`.whl.sha256` release asset when a digest is unavailable. Its internal package version,
ABI metadata, and all three actual WASM extensions are also checked. Network,
authentication, checksum, and corrupt-wheel errors fail the build; they do not
silently trigger a source rebuild. Only absence of a compatible wheel triggers that
fallback, using the selected release's commit rather than the development branch.
The source archive checksum is recorded in build provenance.

The selected release is recorded in `build/wasm/magenpy-release.json`. Native
regression tests install the same release source using the generated
`build/wasm/native-test-requirements.txt`, even when its WASM wheel is reused. The
browser still loads its bundled local copy of the wheel, so "latest" is resolved
at build time, never each time a visitor opens the app.

For a repeatable release selection, pass `--magenpy-release v0.2.2` (or another
published tag). The default is `--magenpy-release latest`, which excludes draft and
prerelease releases. `build-info.json` records `magenpy.origin` as `release-asset`
or `source-build`, along with the selected tag, commit, and checksums.

Install [uv](https://docs.astral.sh/uv/) and Git, then from the repository root run:

```sh
uv run --no-project --python 3.13 --with-requirements tools/wasm/requirements-build.txt pyodide xbuildenv install 0.28.3

# First-time compiler setup (skip the clone if this directory already exists):
git clone --depth 1 --branch 4.0.9 https://github.com/emscripten-core/emsdk.git build/wasm/emsdk
build/wasm/emsdk/emsdk install 4.0.9
build/wasm/emsdk/emsdk activate 4.0.9
source build/wasm/emsdk/emsdk_env.sh

uv run --no-project --python 3.13 --with-requirements tools/wasm/requirements-build.txt python tools/wasm/build.py
```

The build was exercised on macOS ARM64. It compiles the original `.pyx` and C++
headers using the existing non-BLAS, single-threaded kernels. It avoids host BLAS
and OpenMP probes by generating an Emscripten-specific setuptools configuration in
the staging directory. The staged VIPRS package defers POSIX shared-memory imports
until multiprocess grid search is requested, and omits the unused `psutil`
requirement. Those features are outside the browser interface. The source tree's
Python and C++ inference implementation is preserved.

Tqdm's background monitor thread is disabled in the worker because this runtime
uses a single thread. The worker sends initialization and run status messages;
VIPRS completes optimization without per-iteration messages.

The script writes the WASM wheels, pinned pure-Python wheels, runtime manifest, and
build provenance into `docs/webapp/`. Use `--package viprs` to rebuild only VIPRS
when a previous full build exists (keeping its already-selected magenpy wheel),
`--package magenpy` to refresh magenpy's release selection, or `--bundle-only` to
regenerate manifests from completed builds without querying releases.
`--prepare-only` stages sources without compiling.

See Pyodide's [cross-build guide](https://pyodide.org/en/0.28.3/development/building-packages.html)
and [worker guide](https://pyodide.org/en/0.28.3/usage/webworker.html).

## Publishing magenpy WASM independently

magenpy owns its standalone `.github/workflows/wasm.yml` and `tools/wasm/` build
helpers in the magenpy repository. Its release workflow attaches three assets:

- `magenpy-VERSION-cp313-cp313-pyodide_2025_0_wasm32.whl`
- The same filename with `.sha256` appended
- `magenpy-wasm.json` with version, source and compiler provenance

It tests real imports and LD matrix operations in Pyodide before publishing. This
workflow does not depend on VIPRS or build a VIPRS application. PR/push builds retain
Actions artifacts; published releases attach the wheel to that release. There is a
manual `release_tag` + `publish` option for adding WASM to an existing release using
its tagged source and the new build helpers from the selected workflow branch.

Once the magenpy workflow has been committed and pushed, an existing release can
be backfilled without changing its tag, for example:

```sh
gh workflow run wasm.yml --repo shz9/magenpy --ref master -f release_tag=v0.2.2 -f publish=true
```

Use the appropriate default branch if it differs. This command publishes assets;
leave `publish=false` for an artifact-only test. A future VIPRS build will discover
the attached wheel automatically. VIPRS does not upload its fallback build into the
magenpy repository, and does not need credentials with cross-repository write access.
If the two projects release simultaneously before the magenpy asset is attached,
VIPRS safely uses the source-build fallback for that same release.

## Verification

```sh
python -m pytest tests/test_webapp.py tests/test_wasm_package.py tests/test_magenpy_release.py
node --test tests/webapp_sources.mjs tests/webapp_worker.mjs
python tests/webapp_fixtures.py
uv run --no-project --with playwright python tests/webapp_browser.py
```

The Python checks validate archive extraction, options, wheel hashes, WebAssembly
magic bytes, the bundled VIPRS/VIPRSMix source, and joint inference on two chromosomes
using separate and combined inputs. The JavaScript tests check HF/GitHub URL
resolution, streamed input integrity, cleanup on failure, and acceptance above the
old size thresholds with a simulated filesystem. The browser test runs the real chromosome 22 inference and asserts that
WASM extension modules were loaded. It verifies eager initialization, runtime reuse,
VIPRS and VIPRSMix with two chromosomes, downloads, cancellation, initialization
retry, and mobile layout. Small generated fixtures are used only in integration tests;
the application default remains the supplied chromosome 22 data. It writes reports, weights, and screenshots under
`build/wasm/browser-test/`. If local copies of the supplied data are present under
`build/wasm/test-data/`, it also checks file uploads against the default run.

The compiled-bundle test skips on an ordinary source checkout without generated
assets. CI sets `VIPRS_REQUIRE_WASM=1` so a missing bundle fails the WASM job.

Install Playwright Chromium first, or set `VIPRS_BROWSER_EXECUTABLE` to an existing
Chromium-based browser executable. The browser test needs network access to the
runtime CDN and the two public data hosts, and starts its own loopback HTTP server.

## Browser limits

There is no application-imposed byte limit on input files or extracted LD data.
The earlier 256 MiB per-file, 512 MiB combined, and 1 GiB extraction caps were
conservative prototype guardrails, not VIPRS requirements. Local files and downloads
are streamed into the virtual filesystem without assembling an extra full-file
JavaScript buffer. Downloads have no fixed time limit; Cancel can stop a stalled transfer.
This is still an in-memory application: staged files, extracted LD, and runtime arrays
must fit in available browser memory. Arrays can occupy substantially more memory
than the archives, especially with many mixture components. Multiple
chromosomes are supported, but whole-genome memory use has not been validated.
Up to 64 files per input type and 50,000 ZIP entries per archive are accepted.
Archive traversal paths, symlinks, and duplicate paths are rejected.
Streaming remote Zarr access, multiple worker threads,
grid search, and genotype-based scoring are not implemented in the interface.

Use file selection for local data: a typed filesystem path cannot grant a website
access to your disk. Remote hosts must allow CORS. Data is fetched without credentials,
and HTTPS pages require HTTPS input URLs. Cancel terminates the worker and releases
its memory, then starts a new runtime in the background. Normal completed runs and
input-validation failures reuse the initialized runtime; per-run staged files are
removed after completion. Download results before leaving.
Selected files are not sent to an inference server. Standard browser network caching
still applies to runtime assets and hosted inputs.
