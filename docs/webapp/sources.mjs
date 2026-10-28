export const EXAMPLE = Object.freeze({
  ld: "hf://datasets/shz9/ukb-ld/EUR/chr_22.zip",
  sumstats: "https://github.com/shz9/magenpy/blob/master/magenpy/data/ukb_height_chr22.fastGWA.gz",
});

const GWAS_CATALOG_ROOT = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/";

export function gwasCatalogDirectory(accession) {
  const match = /^GCST(\d+)$/i.exec(accession.trim());
  if (!match) throw new Error("Enter a GWAS Catalog accession such as GCST005195.");
  const number = Number(match[1]);
  if (!Number.isSafeInteger(number) || number < 1) throw new Error("The GWAS Catalog accession is out of range.");
  const width = Math.max(6, match[1].length);
  const start = Math.floor((number - 1) / 1000) * 1000 + 1;
  const end = start + 999;
  const id = "GCST" + String(number).padStart(width, "0");
  const range = "GCST" + String(start).padStart(width, "0") + "-GCST" + String(end).padStart(width, "0");
  return GWAS_CATALOG_ROOT + range + "/" + id + "/";
}

function listingLinks(html, directory) {
  const links = [];
  for (const match of html.matchAll(/href=["']([^"']+)["']/gi)) {
    const url = new URL(match[1], directory);
    if (url.origin === new URL(directory).origin && url.href.startsWith(directory)) links.push(url.href);
  }
  return links;
}

async function fetchListing(directory, fetcher) {
  let response;
  try { response = await fetcher(directory, {credentials: "omit"}); }
  catch { throw new Error("Could not reach the GWAS Catalog FTP server. Check the connection."); }
  if (!response.ok) {
    if (response.status === 404) throw new Error("No GWAS Catalog summary statistics were found for this accession.");
    throw new Error(`Could not list GWAS Catalog files (HTTP ${response.status}).`);
  }
  return listingLinks(await response.text(), directory);
}

function isBuild37Filename(url) {
  return /(?:grch37|build(?:grch)?37|hg19)/i.test(decodeURIComponent(new URL(url).pathname));
}

function metadataIsBuild37(metadata) {
  const assembly = metadata.match(/(?:^|\n)\s*genome_assembly\s*:\s*["']?([^\s#"']+)/i)?.[1]?.toLowerCase();
  return assembly === "grch37" || assembly === "hg19" || assembly === "ncbi37";
}

async function isBuild37(candidate, metadataLinks, fetcher) {
  if (isBuild37Filename(candidate)) return true;
  const metadataURL = metadataLinks.find(url => url === candidate + "-meta.yaml");
  if (!metadataURL) return false;
  try {
    const response = await fetcher(metadataURL, {credentials: "omit"});
    return response.ok && metadataIsBuild37(await response.text());
  } catch { return false; }
}

export async function resolveGWASCatalog(accession, fetcher = fetch) {
  const directory = gwasCatalogDirectory(accession);
  const rootLinks = await fetchListing(directory, fetcher);
  const candidates = [];
  const harmonisedDirectory = rootLinks.find(url => url === directory + "harmonised/");
  if (harmonisedDirectory) {
    const links = await fetchListing(harmonisedDirectory, fetcher);
    for (const suffix of [".h.tsv.gz", ".f.tsv.gz"]) {
      for (const url of links.filter(url => new URL(url).pathname.toLowerCase().endsWith(suffix)).sort()) {
        candidates.push({url, metadataLinks: links});
      }
    }
  }
  for (const url of rootLinks.filter(url => /\.tsv(?:\.gz)?$/i.test(new URL(url).pathname)).sort()) {
    candidates.push({url, metadataLinks: rootLinks});
  }
  for (const candidate of candidates) {
    if (await isBuild37(candidate.url, candidate.metadataLinks, fetcher)) return [candidate.url];
  }
  throw new Error("This accession has no GRCh37/hg19 harmonised, formatted, or GWAS-SSF summary-statistics file compatible with the default LD reference.");
}

export function globMatcher(pattern) {
  let expression = "";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === "*") {
      if (pattern[i + 1] === "*") {
        i++;
        if (pattern[i + 1] === "/") { i++; expression += "(?:.*/)?"; }
        else expression += ".*";
      } else expression += "[^/]*";
    } else if (c === "?") expression += "[^/]";
    else if (c === "[") {
      const end = pattern.indexOf("]", i + 1);
      if (end < 0) throw new Error("Close the character range in the glob pattern.");
      const range = pattern.slice(i + 1, end);
      if (!/^[!a-zA-Z0-9_-]+$/.test(range)) throw new Error("Use simple character ranges such as [1-9] in glob patterns.");
      expression += "[" + range.replace(/^!/, "^") + "]";
      i = end;
    } else expression += c.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
  }
  try { return new RegExp("^" + expression + "$"); }
  catch { throw new Error("Invalid character range in glob pattern."); }
}

export function selectLocalFiles(files, patterns) {
  const entries = patterns.split(/\r?\n/).map(p => p.trim()).filter(Boolean);
  if (!entries.length) throw new Error("Enter at least one file pattern.");
  const selected = new Map();
  for (const pattern of entries) {
    if (/^(\/|[a-z]:|file:)/i.test(pattern) || pattern.split("/").includes("..")) {
      throw new Error("Use paths relative to the selected folder, such as EUR/chr_*.zip.");
    }
    const match = globMatcher(pattern.replace(/^\.\//, ""));
    const found = Array.from(files).filter(file => match.test(
      file.webkitRelativePath ? file.webkitRelativePath.split("/").slice(1).join("/") : file.name));
    if (!found.length) throw new Error("No files match: " + pattern);
    for (const file of found) selected.set(file.webkitRelativePath || file.name, file);
  }
  if (selected.size > 64) throw new Error("The patterns match more than 64 files. Narrow the selection.");
  return [...selected].sort(([a], [b]) => a.localeCompare(b, undefined, {numeric: true})).map(([, file]) => file);
}

// Glob expansion requires a repository listing API, not a wildcard HTTP request.
export async function expandDataURL(input, baseURL, fetcher = fetch) {
  const url = new URL(resolveDataURL(input, baseURL));
  const parts = url.pathname.split("/").filter(Boolean);
  const decodedPath = decodeURIComponent(url.pathname);
  if (!/[*[]/.test(decodedPath)) return [url.href];
  let api, prefix, pattern, provider;
  if (url.hostname === "huggingface.co" && parts[0] === "datasets" && parts[3] === "resolve" && parts.length >= 6) {
    const [, owner, repo, , revision, ...path] = parts;
    prefix = "https://huggingface.co/datasets/" + owner + "/" + repo + "/resolve/" + revision + "/";
    pattern = decodeURIComponent(path.join("/"));
    const directory = pattern.slice(0, pattern.search(/[*[]/)).split("/").slice(0, -1).join("/");
    api = "https://huggingface.co/api/datasets/" + owner + "/" + repo + "/tree/" + revision +
      (directory ? "/" + directory.split("/").map(encodeURIComponent).join("/") : "") + "?recursive=true&limit=1000";
    provider = "hf";
  } else if (url.hostname === "raw.githubusercontent.com" && parts.length >= 4) {
    const [owner, repo, revision, ...path] = parts;
    prefix = "https://raw.githubusercontent.com/" + owner + "/" + repo + "/" + revision + "/";
    pattern = decodeURIComponent(path.join("/"));
    api = "https://api.github.com/repos/" + owner + "/" + repo + "/git/trees/" + revision + "?recursive=1";
    provider = "github";
  } else {
    throw new Error("Hosted globs require a Hugging Face dataset or GitHub file URL. For other hosts, enter explicit URLs; for local paths, choose Local folder.");
  }
  if (/[*[\]]/.test(prefix)) throw new Error("Use a fixed repository and revision; glob only the file path.");
  const match = globMatcher(pattern), matches = new Set(), visited = new Set();
  while (api) {
    if (visited.has(api)) throw new Error("Repository listing repeated a page.");
    visited.add(api);
    let response;
    try { response = await fetcher(api, {credentials: "omit"}); }
    catch { throw new Error("Could not list matching files. Check the connection and repository access."); }
    if (!response.ok) throw new Error("Could not list matching files (HTTP " + response.status + "). Check repository access or API rate limits.");
    const body = await response.json();
    if (body.truncated) throw new Error("GitHub returned an incomplete file listing. Use explicit file URLs.");
    for (const entry of provider === "hf" ? body : body.tree) {
      if (entry.type === (provider === "hf" ? "file" : "blob") && match.test(entry.path)) {
        matches.add(prefix + entry.path.split("/").map(encodeURIComponent).join("/"));
        if (matches.size > 64) throw new Error("The glob matches more than 64 files. Narrow the pattern.");
      }
    }
    const next = response.headers.get("link")?.match(/<([^>]+)>;\s*rel="next"/)?.[1];
    api = next ? new URL(next, api).href : null;
    if (api && new URL(api).origin !== "https://huggingface.co") throw new Error("Unexpected repository pagination URL.");
  }
  if (!matches.size) throw new Error("No files match: " + input);
  return [...matches].sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
}

export function resolveDataURL(input, baseURL) {
  const raw = input.trim();
  if (!raw || /^(file:|[a-z]:[\\/])/i.test(raw)) {
    throw new Error("Enter a hosted URL, or select a local file using the file picker.");
  }
  const url = new URL(raw, baseURL);
  if (url.protocol === "hf:") {
    const parts = url.pathname.split("/").filter(Boolean);
    if (url.hostname !== "datasets" || parts.length < 3) {
      throw new Error("Use hf://datasets/owner/repository/path/to/ld.zip.");
    }
    const [owner, repository, ...path] = parts;
    const [repo, revision = "main"] = repository.split("@");
    return `https://huggingface.co/datasets/${owner}/${repo}/resolve/${revision}/${path.join("/")}`;
  }
  if (!["https:", "http:"].includes(url.protocol) || url.username || url.password) {
    throw new Error("Use an HTTP(S) or Hugging Face dataset URL without embedded credentials.");
  }
  if (url.hostname === "github.com") {
    const parts = url.pathname.split("/");
    if (parts[3] === "blob") {
      url.hostname = "raw.githubusercontent.com";
      parts.splice(3, 1);
      url.pathname = parts.join("/");
      url.search = "";
    }
  }
  return url.href;
}
