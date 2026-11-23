const ROOT = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/";
export function catalogDirectory(value) {
  const match = /^GCST(\d+)$/i.exec(value.trim());
  if (!match) throw new Error("Enter a GWAS Catalog accession such as GCST005195.");
  const n = Number(match[1]), width = Math.max(6, match[1].length);
  if (!Number.isSafeInteger(n) || n < 1) throw new Error("The GWAS Catalog accession is out of range.");
  const id = n => `GCST${String(n).padStart(width, "0")}`;
  const start = Math.floor((n - 1) / 1000) * 1000 + 1;
  return `${ROOT}${id(start)}-${id(start + 999)}/${id(n)}/`;
}
function links(text, directory) {
  return [...new Set([...text.matchAll(/href=["']([^"']+)["']/gi)].flatMap(match => {
    try {
      const url = new URL(match[1].replaceAll("&amp;", "&"), directory);
      return url.href.startsWith(directory) && !url.search && !url.hash ? [url.href] : [];
    } catch { return []; }
  }))];
}
function normalizeBuild(value) {
  if (/^(?:grch37|hg19|ncbi37|37|19)$/i.test(value)) return "GRCh37";
  if (/^(?:grch38|hg38|ncbi38|38)$/i.test(value)) return "GRCh38";
  return null;
}
export async function validateCatalog(accession, {fetcher = fetch, signal, timeout = 15000} = {}) {
  const directory = catalogDirectory(accession);
  async function read(url, head = false) {
    const controller = new AbortController(), abort = () => controller.abort();
    signal?.addEventListener("abort", abort, {once: true});
    if (signal?.aborted) controller.abort();
    const timer = setTimeout(abort, timeout);
    try {
      const response = await fetcher(url, {credentials: "omit", signal: controller.signal, method: head ? "HEAD" : "GET"});
      if (response.status === 404) throw new Error("No summary-statistics files were found at this FTP address. Check the GCST ID or choose another study.");
      if (!response.ok) throw new Error(`GWAS Catalog returned HTTP ${response.status}. Retry validation shortly.`);
      if (head) {
        const length = response.headers?.get("content-length");
        const bytes = length && /^\d+$/.test(length) ? Number(length) : null;
        return Number.isSafeInteger(bytes) && bytes > 0 ? bytes : null;
      }
      return await response.text();
    } catch (error) {
      if (signal?.aborted) throw error;
      if (controller.signal.aborted) throw new Error("GWAS Catalog request timed out. Retry validation or check your connection.");
      if (error instanceof TypeError) throw new Error("Could not reach the GWAS Catalog FTP server. Check your connection and retry validation.");
      throw error;
    } finally {
      clearTimeout(timer); signal?.removeEventListener("abort", abort);
    }
  }
  const all = links(await read(directory), directory), warnings = [];
  let complete = true;
  if (all.includes(directory + "harmonised/")) {
    try { all.push(...links(await read(directory + "harmonised/"), directory + "harmonised/")); }
    catch (error) { if (signal?.aborted) throw error; complete = false; warnings.push(error.message); }
  }
  const rank = url => url.startsWith(directory + "harmonised/") ? (url.endsWith(".h.tsv.gz") ? 0 : url.endsWith(".f.tsv.gz") ? 1 : 2) : 3;
  const candidates = [...new Set(all.filter(url => /\.tsv(?:\.gz)?$/i.test(url)))].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  if (!candidates.length && complete) throw new Error("This accession exists on the FTP server, but has no supported TSV summary-statistics files. Choose another study or provide a supported file directly.");
  const files = {};
  for (const url of candidates) {
    const match = /(?:grch|build(?:grch)?|hg|ncbi)(37|38|19)(?![0-9])/i.exec(decodeURIComponent(new URL(url).pathname.split("/").at(-1)));
    let build = match ? normalizeBuild(match[1]) : null;
    if (!build && all.includes(url + "-meta.yaml")) {
      try {
        const metadata = await read(url + "-meta.yaml");
        const assembly = /(?:^|\n)\s*genome_assembly\s*:\s*["']?([^\s#"']+)/i.exec(metadata);
        build = assembly ? normalizeBuild(assembly[1]) : null;
      } catch (error) { if (signal?.aborted) throw error; warnings.push(error.message); }
    }
    if (build) files[build] ||= url;
    else complete = false;
  }
  // HEAD avoids downloading summary statistics just to measure their size.
  // Unavailable size information must not invalidate a verified study/build.
  const file_sizes = {};
  await Promise.all(Object.entries(files).map(async ([build, url]) => {
    try { file_sizes[build] = await read(url, true); }
    catch (error) { if (signal?.aborted) throw error; file_sizes[build] = null; }
  }));
  return {accession: accession.trim().toUpperCase(), files, complete, warnings, file_sizes};
}
