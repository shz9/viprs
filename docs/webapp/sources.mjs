export const EXAMPLE = Object.freeze({
  ld: "hf://datasets/shz9/ukb-ld/EUR/chr_22.zip",
  sumstats: "https://github.com/shz9/magenpy/blob/master/magenpy/data/ukb_height_chr22.fastGWA.gz",
});

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
