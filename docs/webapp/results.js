import {getLDBrowserURL} from "./sources.mjs";
import {formatDuration} from "./time.js";
import {getScoringGuide, genotypeFormats, recommendedTool, platformGuidance} from "./scoring.js";

const $ = id => document.getElementById(id);
const modes = ["summary", "scoring", "table", "plots"];
const finite = value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
const number = value => finite(value) ? Number(value) : null;
const fmt = value => finite(value) ? Number(value).toLocaleString(undefined, {maximumSignificantDigits: 5}) : "Not recorded";
const heap = value => finite(value) ? `${(value / 1024 ** 2).toFixed(1)} MiB` : "Not recorded";
const el = (tag, text, cls) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; };
let result = null, mode = "summary", generation = 0, prepared = false, makeDownload = null;
let plotRows = [], screenPoints = [], selected = null;
let exports = new Map(), outputChosen = false;

function activate(next, focus = false) {
  mode = next;
  for (const name of modes) {
    const tab = $(`tab-${name}`), active = next === name;
    tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1;
    $(`view-${name}`).hidden = !active;
  }
  if (focus) $(`tab-${next}`).focus();
  if (result && next === "plots") drawWeights();
  if (result && next === "scoring") {
    if ($("download-format").value === "native" && !outputChosen) recommendScoring();
    const token = generation;
    prepareScoring().catch(() => {
      if (token !== generation) return;
      prepared = false;
      exports.delete("dosage"); exports.delete("pgs"); updateDownload();
      $("scoring-availability").textContent = "Scoring files could not be prepared. The complete inferred weights are still available. Reopen this view to retry.";
    });
  }
}
for (const name of modes) {
  $(`tab-${name}`).addEventListener("click", () => activate(name));
  $(`tab-${name}`).addEventListener("keydown", event => {
    const keys = ["ArrowRight", "ArrowLeft", "Home", "End"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const i = modes.indexOf(name);
    activate(event.key === "Home" ? modes[0] : event.key === "End" ? modes.at(-1) : modes[(i + (event.key === "ArrowRight" ? 1 : 3)) % 4], true);
  });
}
function build() { return ["GRCh37", "GRCh38"].includes(result?.report.genome_build) ? result.report.genome_build : null; }
function diagnostics() { return result?.report.chromosome_diagnostics || {}; }
function renderSummary() {
  const r = result.report, d = diagnostics();
  $("fit-overview").replaceChildren();
  for (const [label, value, note] of [
    ["Fitting time", formatDuration(r.timings.fit_seconds), "Across all fitted chromosomes"],
    ["Run time", formatDuration(r.timings.run_seconds), "Includes inputs; excludes runtime startup"],
    ["Iterations", fmt(r.iterations), "Sum across independent chromosome fits"],
    ["Convergence", r.converged ? "Converged" : "Review fit", `${r.chromosomes.length} chromosome(s)`],
  ]) {
    const stat = el("div", undefined, "fit-stat"); stat.append(el("span", label), el("strong", value), el("small", note)); $("fit-overview").append(stat);
  }
  const message = $("fit-message");
  message.textContent = r.converged ? "All chromosomes converged successfully." : "Some chromosome fits did not converge. Review the diagnostics before using the weights.";
  if (!r.converged) {
    const issues = r.chromosomes.filter(c => d[c]?.converged === false);
    if (issues.length) {
      const list = el("ul", undefined, "instruction-checklist");
      for (const c of issues) list.append(el("li", `Chromosome ${c}: ${d[c].message || "Did not converge."}`));
      message.append(list);
    }
  }
  $("fit-message").classList.toggle("attention", !r.converged);
  $("diagnostic-chromosome").replaceChildren(...r.chromosomes.map(c => new Option(`Chromosome ${c}`, c)));
  renderELBO();
  const measured = r.chromosomes.filter(c => finite(d[c]?.fit_seconds));
  $("runtime-chart").replaceChildren();
  if (!measured.length) $("runtime-chart").append(el("p", "Per-chromosome timing was not recorded for this run.", "hint"));
  else {
    const bars = el("div", undefined, "runtime-bars");
    const max = Math.max(...measured.map(c => d[c].fit_seconds), 0.001);
    for (const c of measured) {
      const row = el("div", undefined, "runtime-bar"), track = el("div", undefined, "runtime-bar-track"), fill = el("div", undefined, "runtime-bar-fill");
      fill.style.width = `${100 * d[c].fit_seconds / max}%`; track.append(fill);
      row.append(el("span", `Chr ${c}`), track, el("span", formatDuration(d[c].fit_seconds))); bars.append(row);
    }
    $("runtime-chart").append(bars);
  }
  $("diagnostic-rows").replaceChildren(...r.chromosomes.map(c => {
    const row = el("tr"), v = d[c] || {};
    for (const val of [c, fmt(r.variants_per_chromosome[c]), fmt(v.iterations), formatDuration(v.fit_seconds), heap(v.wasm_heap_bytes_at_fit_end), v.converged === undefined ? "Not recorded" : v.converged ? "Converged" : "Review fit"]) row.append(el("td", val));
    return row;
  }));
}
function renderELBO() {
  const c = $("diagnostic-chromosome").value;
  const history = diagnostics()[c]?.elbo || (result.report.chromosomes.length === 1 ? result.report.elbo : []);
  const points = (history || []).slice(1).map((y, i) => ({x: i + 1, y: number(y)})).filter(p => p.y !== null);
  const root = $("elbo-chart"); root.replaceChildren();
  if (!points.length) { root.append(el("p", history?.length === 1 ? "No fitted iterations were recorded after initialization." : "An ELBO trajectory was not recorded for this chromosome.", "hint")); return; }
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 460 245"); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Chromosome ${c} ELBO across ${points.length} fitted observations; first ${fmt(points[0].y)}, final ${fmt(points.at(-1).y)}.`);
  function item(tag, attrs, text) { const n = document.createElementNS(ns, tag); for (const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); if (text !== undefined) n.textContent = text; svg.append(n); return n; }
  const lo = Math.min(...points.map(p => p.y)), hi = Math.max(...points.map(p => p.y));
  const y = val => 197 - (hi === lo ? .5 : (val-lo)/(hi-lo)) * 168;
  for (let i=0; i<3; i++) { const v = lo + (hi-lo)*i/2, py=y(v); item("path", {d:`M85 ${py}H440`,stroke:"#dcded2",fill:"none"}); item("text", {x:78,y:py+4,"text-anchor":"end"}, fmt(v)); }
  item("path", {d:points.map((p,i) => `${i?"L":"M"}${85+355*(p.x-points[0].x)/Math.max(points.at(-1).x-points[0].x,1)} ${y(p.y)}`).join(" "),stroke:"#205540","stroke-width":2,fill:"none"});
  if (points.length === 1) item("circle",{cx:85,cy:y(points[0].y),r:3,fill:"#205540"});
  item("text",{x:85,y:217},String(points[0].x)); item("text",{x:440,y:217,"text-anchor":"end"},String(history.length-1)); item("text",{x:260,y:239,"text-anchor":"middle"},"Iteration");
  root.append(svg);
}
$("diagnostic-chromosome").addEventListener("change", renderELBO);

function renderGuide(target, recipe) {
  const root = $(target); root.replaceChildren();
  recipe.steps.forEach(([title, text, command, links = []], i) => {
    const step = el("section", undefined, "guide-step");
    step.append(el("span", String(i+1).padStart(2,"0"), "step-number"), el("h4", title));
    const checklist = el("ul", undefined, "instruction-checklist");
    for (const item of text) checklist.append(el("li", item));
    step.append(checklist);
    if (command) {
      command = command.replaceAll("BUILD", build() || "REPLACE_WITH_TARGET_BUILD");
      const box = el("div", undefined, "command-block"), pre = el("pre"), code = el("code",command), copy = el("button","Copy", "copy-command"), feedback = el("span", "", "copy-feedback");
      pre.tabIndex = 0; pre.append(code); copy.type = "button"; copy.setAttribute("aria-label", `Copy ${title.toLowerCase()}`); feedback.setAttribute("role", "status");
      copy.addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(command); feedback.textContent="Copied to clipboard."; copy.textContent="Copied ✓"; }
        catch { const selection=window.getSelection(), range=document.createRange(); range.selectNodeContents(code); selection.removeAllRanges(); selection.addRange(range); feedback.textContent="Command selected. Press Ctrl+C or ⌘C to copy."; }
      });
      box.append(pre,copy,feedback); step.append(box);
    }
    for (const [label, url] of links) {
      const link = el("a", label + " ↗", "instruction-link");
      link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer";
      step.append(link);
    }
    root.append(step);
  });
}
function renderScoringGuide() {
  const tool = $("scoring-tool").value, environment = $("biobank-platform").value;
  $("execution-options").hidden = !["plink", "pgsc"].includes(tool) || (tool === "plink" && environment === "ukb");
  $("execution-runtime").options[0].textContent = tool === "pgsc" ? "Conda · no containers" : "Native PLINK · install if needed";
  renderGuide("scoring-guide", getScoringGuide(tool, environment, $("genotype-format").value, $("execution-runtime").value, $("genotype-layout").value, result?.report.chromosomes));
  const advice = platformGuidance(environment, $("genotype-format").value, tool), root = $("platform-guidance");
  root.hidden = !advice; root.replaceChildren();
  if (advice) {
    root.append(el("strong", advice.title));
    const list = el("ul", undefined, "instruction-checklist");
    for (const item of advice.items) list.append(el("li", item));
    root.append(list);
    for (const [label, url] of advice.links) {
      const link = el("a", label + " ↗", "instruction-link");
      link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; root.append(link);
    }
  }
}
function recommendScoring(resetTool = false) {
  const genotype = $("genotype-format").value, format = genotypeFormats[genotype];
  const recommended = recommendedTool(genotype, $("biobank-platform").value);
  for (const option of $("scoring-tool").options) {
    option.disabled = !format.tools.includes(option.value);
    option.textContent = {plink: genotype === "bed" ? "PLINK 2 / PLINK 1.9" : "PLINK 2", viprs: "VIPRS", pgsc: "PGS Catalog Calculator · pgsc_calc", score: "BCFtools Score plugin · VCF / BCF"}[option.value] + (option.value === recommended ? " · Recommended" : "");
  }
  if (resetTool || !format.tools.includes($("scoring-tool").value)) $("scoring-tool").value = recommended;
  $("scoring-recommendation").textContent = format.hint;
  $("download-format").value = ["pgsc", "score"].includes($("scoring-tool").value) ? "pgs" : "dosage";
  updateDownload(); renderScoringGuide();
}
$("genotype-format").addEventListener("change", () => recommendScoring(true));
$("scoring-tool").addEventListener("change", () => recommendScoring());
$("biobank-platform").addEventListener("change", () => recommendScoring(true));
$("execution-runtime").addEventListener("change", renderScoringGuide);
$("genotype-layout").addEventListener("change", renderScoringGuide);
const exportFiles = {native: "viprs-weights.tsv", dosage: "viprs-scoring.tsv", pgs: "viprs-pgs.txt", report: "viprs-run.json"};
function updateDownload() {
  const format = $("download-format").value, link = $("download"), url = exports.get(format);
  link.setAttribute("aria-disabled", String(!url)); link.tabIndex = url ? 0 : -1;
  link.download = exportFiles[format];
  if (url) link.href = url; else link.removeAttribute("href");
  $("download-status").textContent = url ? "" : format === "pgs" ? "PGS export requires valid positions, reference frequencies and a known genome build." : "Dosage export requires valid effect estimates and LD reference frequencies.";
}
function saveExport(format, data, type) {
  exports.set(format, makeDownload(format, data, type)); updateDownload();
}
$("download-format").addEventListener("change", () => {
  outputChosen = true;
  const token = generation;
  updateDownload();
  if (result && ["dosage", "pgs"].includes($("download-format").value)) {
    prepareScoring().catch(() => {
      if (token !== generation) return;
      prepared = false; exports.delete("dosage"); exports.delete("pgs"); updateDownload();
      $("scoring-availability").textContent = "Scoring files could not be prepared. Select another output format and retry.";
    });
  }
});
$("download").addEventListener("click", event => { if (!exports.has($("download-format").value)) event.preventDefault(); });
async function prepareScoring() {
  if (prepared || !result) return;
  prepared=true;
  const token=generation, tsv=result.tsv, knownBuild=build();
  $("scoring-availability").textContent="Preparing dosage scoring downloads…";
  // Process lines without constructing an additional array of all input records.
  const lines=tsv.matchAll(/[^\r\n]+/g), first=lines.next().value;
  const columns=first?.[0].split("\t") || [], idx=Object.fromEntries(columns.map((k,i)=>[k,i]));
  if (!["CHR","SNP","A1","A2","BETA","MAF"].every(k=>k in idx)) { $("scoring-availability").textContent="Dosage export unavailable: this run did not record the LD reference frequencies and required weight columns. The inferred-weight download is available; use a new fit to obtain scoring exports."; return; }
  const dosageColumns=["CHR","SNP","POS","A1","A2","PIP","BETA","VAR_BETA", ...columns.filter(k=>!["CHR","SNP","POS","A1","A2","PIP","BETA","VAR_BETA"].includes(k))];
  const dosage=[dosageColumns.join("\t")+"\n"], pgs=[`#pgs_id=VIPRS\n#pgs_name=VIPRS\n#trait_reported=User_provided_trait\n#genome_build=${knownBuild}\nchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n`];
  let count=0, positions=true, invalid=false;
  for(const line of lines) {
    const cells=line[0].split("\t"), maf=number(cells[idx.MAF]), beta=number(cells[idx.BETA]);
    if(maf===null || maf<=0 || maf>=1 || beta===null) { invalid=true; break; }
    const scale=Math.sqrt(2*maf*(1-maf)); cells[idx.BETA]=String(beta/scale);
    if("VAR_BETA" in idx && finite(cells[idx.VAR_BETA])) cells[idx.VAR_BETA]=String(Number(cells[idx.VAR_BETA])/(scale*scale));
    dosage.push(dosageColumns.map(k=>cells[idx[k]] ?? "").join("\t")+"\n");
    const pos=cells[idx.POS];
    if(!finite(pos) || Number(pos)<1 || !Number.isInteger(Number(pos))) positions=false;
    pgs.push([cells[idx.CHR],pos,cells[idx.A1],cells[idx.A2],cells[idx.BETA]].join("\t")+"\n");
    if(++count%10000===0) { await new Promise(resolve=>setTimeout(resolve,0)); if(token!==generation) return; }
  }
  if(token!==generation) return;
  if(invalid || !count) { $("scoring-availability").textContent="Dosage export unavailable: at least one coefficient or LD frequency is missing or invalid. No incomplete scoring file was created. Download the inferred weights and inspect the run."; return; }
  saveExport("dosage",new Blob(dosage,{type:"text/tab-separated-values"}),"text/tab-separated-values");
  if(positions && knownBuild) { saveExport("pgs",new Blob(pgs,{type:"text/plain"}),"text/plain"); }
  $("scoring-availability").textContent=`Dosage file: all ${count.toLocaleString()} fitted variants. BETA = native BETA / √(2p(1−p)); VAR_BETA is converted by 2p(1−p), using the fitted LD reference frequencies. ${positions && knownBuild ? "PGS format is ready for pgsc_calc and bcftools +score." : "PGS export requires valid positions and a known GRCh37/GRCh38 build; use the native dosage file after checking the build."}`;
}

function inspect(row) {
  selected=row;
  const dl=el("dl");
  for(const [label,v] of [["Variant",row.SNP],["Position",`${row.CHR}:${row.POS ?? "unknown"}`],["A1 / A2",`${row.A1} / ${row.A2}`],["PIP",fmt(row.PIP)],["Posterior β (standardized)",fmt(row.BETA)], ["Posterior β / allele",finite(row.MAF)&&Number(row.MAF)>0&&Number(row.MAF)<1&&finite(row.BETA) ? fmt(Number(row.BETA)/Math.sqrt(2*Number(row.MAF)*(1-Number(row.MAF)))) : "Not recorded"],["Variance",fmt(row.VAR_BETA)],["LD MAF",fmt(row.MAF)],["GWAS Z",fmt(row.GWAS_Z)]]) {
    const block=el("div"); block.append(el("dt",label),el("dd",v)); dl.append(block);
  }
  $("variant-detail").replaceChildren(dl);
  if(finite(row.POS)) {
    const button=el("button","Inspect this locus →"); button.type="button";
    button.addEventListener("click",()=>{ $("weight-plot-kind").value="locus"; $("plot-chromosome").value=String(row.CHR); $("locus-center").value=row.POS; drawWeights(); });
    $("variant-detail").append(button);
  }
}
$("variant-search").addEventListener("submit", e => {
  e.preventDefault(); const id=$("variant-id").value.trim(); const row=plotRows.find(row=>String(row.SNP)===id);
  if(row) inspect(row); else $("variant-detail").textContent="Variant not found in the plotted data. If this run was sampled, inspect the complete weight download.";
});
for(const id of ["weight-plot-kind","plot-chromosome","miami-metric","trumpet-scale","locus-window"]) $(id).addEventListener("change",()=>{
  if(id==="plot-chromosome") $("locus-center").value="";
  drawWeights();
});
$("locus-controls").addEventListener("submit",e=>{e.preventDefault();drawWeights();});
$("weight-canvas").addEventListener("click",e=>{
  const rect=e.currentTarget.getBoundingClientRect(), x=e.clientX-rect.left,y=e.clientY-rect.top;
  let best=null, distance=144;
  for(const p of screenPoints) { const d=(p.x-x)**2+(p.y-y)**2; if(d<distance){best=p.row;distance=d;} }
  if(best) inspect(best);
});
$("save-plot").addEventListener("click",()=>{
  $("weight-canvas").toBlob(blob=>{if(!blob)return; const url=URL.createObjectURL(blob), a=el("a");a.href=url;a.download=`viprs-${$("weight-plot-kind").value}.png`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
});
new ResizeObserver(()=>{if(result && mode==="plots") drawWeights();}).observe($("weight-canvas").parentElement);

function drawWeights() {
  if(!result || mode!=="plots") return;
  const kind=$("weight-plot-kind").value, chromosome=$("plot-chromosome").value;
  $("locus-controls").hidden=kind!=="locus"; $("miami-options").hidden=kind!=="miami"; $("trumpet-options").hidden=kind!=="trumpet";
  const total=result.plot_data?.total ?? result.report.variants, sampled=result.plot_data?.sampled ?? total>plotRows.length;
  $("plot-coverage").textContent=sampled ? `Plotting ${plotRows.length.toLocaleString()} of ${total.toLocaleString()} fitted variants: an evenly spaced sample plus top posterior and GWAS signals per chromosome. Local plots are also sampled. Downloads include every fitted variant.` : `All ${plotRows.length.toLocaleString()} fitted variants are available to plot. Filters and missing fields can reduce the displayed count.`;
  const canvas=$("weight-canvas"), wrap=canvas.parentElement;
  // The results section can be hidden when it is initially populated.
  const w=Math.max(280,wrap.clientWidth), h=w<500?360:440, ratio=Math.min(window.devicePixelRatio||1,2);
  canvas.width=w*ratio;canvas.height=h*ratio;canvas.style.height=`${h}px`;
  const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);ctx.fillStyle="#fffdf7";ctx.fillRect(0,0,w,h);
  ctx.font="11px ui-monospace,monospace";
  const left=w<500?60:85, right=w-22, top=40, bottom=h-(kind==="miami"?88:55), mid=(top+bottom)/2;
  const text=(str,x,y,align="left",color="#626e66")=>{ctx.fillStyle=color;ctx.textAlign=align;ctx.fillText(str,x,y);};
  const line=(x1,y1,x2,y2,color="#dcded2")=>{ctx.beginPath();ctx.strokeStyle=color;ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();};
  let rows=plotRows.filter(r=>!chromosome || String(r.CHR)===chromosome), xValue, xMin=0, xMax=1, yValue, yMax=1, lowerValue=null, lowerMax=1;
  let description="", xLabel="", upperLabel="", lowerLabel="", centers=[];
  $("locus-browser").hidden=true; $("locus-ld-browser").hidden=true;
  if(kind==="trumpet") {
    rows=rows.filter(r=>finite(r.MAF)&&Number(r.MAF)>0&&Number(r.MAF)<=.5&&finite(r.BETA));
    const dosage = $("trumpet-scale").value === "dosage";
    xValue=r=>Number(r.MAF);xMax=.5;
    yValue=r=>dosage ? Number(r.BETA)/Math.sqrt(2*Number(r.MAF)*(1-Number(r.MAF))) : Number(r.BETA);
    yMax=rows.reduce((m,r)=>Math.max(m,Math.abs(yValue(r))),0)||1;
    xLabel="LD reference minor allele frequency";upperLabel=dosage ? "Posterior β / allele" : "Posterior β (standardized)";
    description=`Trumpet plot: LD-reference minor allele frequency versus ${dosage ? "signed posterior effect per allele, β / √(2p(1−p)), matching the dosage scoring download" : "signed posterior effect on standardized genotypes, matching the native weight download"}. The phenotype scale remains standardized. Green points have PIP < 0.5; coral points have PIP ≥ 0.5. Posterior shrinkage and the LD panel’s frequency coverage affect the shape; these are fitted posterior weights, not a selection of significant marginal GWAS hits.`;
  } else if(kind==="miami") {
    const useP=rows.some(r=>finite(r.GWAS_P)&&Number(r.GWAS_P)>=0&&Number(r.GWAS_P)<=1);
    yValue=useP ? r=>finite(r.GWAS_P)&&Number(r.GWAS_P)>=0&&Number(r.GWAS_P)<=1 ? -Math.log10(Math.max(Number(r.GWAS_P),1e-300)) : null : r=>finite(r.GWAS_Z) ? Number(r.GWAS_Z)**2 : null;
    lowerValue=$("miami-metric").value==="PIP" ? r=>number(r.PIP) : r=>finite(r.BETA) ? Number(r.BETA)**2 : null;
    rows=rows.filter(r=>finite(r.POS)&&finite(yValue(r))&&finite(lowerValue(r)));
    const chromosomes=result.report.chromosomes.filter(c=>rows.some(r=>String(r.CHR)===c));
    const ranges=new Map(); let offset=0;
    const span=rows.reduce((m,r)=>Math.max(m,Number(r.POS)),1), gap=span*.025;
    for(const c of chromosomes){const end=rows.reduce((m,r)=>String(r.CHR)===c?Math.max(m,Number(r.POS)):m,1);ranges.set(c,offset);centers.push([c,offset+end/2]);offset+=end+gap;}
    xValue=r=>ranges.get(String(r.CHR))+Number(r.POS);xMax=Math.max(1,offset-gap);
    yMax=rows.reduce((m,r)=>Math.max(m,yValue(r)),0)||1;
    lowerMax=$("miami-metric").value==="PIP" ? 1 : rows.reduce((m,r)=>Math.max(m,lowerValue(r)),0)||1;
    upperLabel=useP?"GWAS −log₁₀(P)":"GWAS Z²";lowerLabel=$("miami-metric").value==="PIP"?"PIP":"Posterior β²";xLabel="Chromosome";
    [yValue, lowerValue] = [lowerValue, yValue];
    [yMax, lowerMax] = [lowerMax, yMax];
    [upperLabel, lowerLabel] = [lowerLabel, upperLabel];
    description=`Miami plot: ${upperLabel} above, ${lowerLabel} below. The panels use independent scales and show matched, fitted variants only. GWAS P=0 is capped at −log₁₀(P)=300. β² is the square of the posterior mean, not the posterior second moment. Chromosome colors alternate.`;
  } else {
    let c=chromosome || selected?.CHR?.toString() || result.report.chromosomes[0];
    if(!chromosome) {$("plot-chromosome").value=c;rows=plotRows.filter(r=>String(r.CHR)===c);}
    const valid=rows.filter(r=>finite(r.POS)&&finite(r.PIP));
    if(!$("locus-center").value){const lead=valid.reduce((a,b)=>!a||b.PIP>a.PIP?b:a,null);if(lead)$("locus-center").value=lead.POS;}
    const center=number($("locus-center").value), window=Number($("locus-window").value);
    xMin=Math.max(1,(center||1)-window);xMax=(center||1)+window;
    rows=valid.filter(r=>r.POS>=xMin&&r.POS<=xMax);xValue=r=>Number(r.POS);yValue=r=>Number(r.PIP);yMax=1;
    xLabel=`Chromosome ${c} position (Mb)`;upperLabel="PIP";
    description=`Local posterior inclusion probabilities in a ±${window/1000} kb window. Genomic positions give context among neighboring fitted variants. Green points have PIP < 0.5; coral points have PIP ≥ 0.5. Gene models are available in the linked UCSC genome browser when the genome build is known. The LD browser shows regional LD for supported reference panels.`;
    if(build()) {
      const chromosome = String(c).replace(/^chr/i, ""), start = Math.floor(xMin), end = Math.ceil(xMax);
      const database = build() === "GRCh37" ? "hg19" : "hg38";
      $("locus-browser").href = "https://genome.ucsc.edu/cgi-bin/hgTracks?" + new URLSearchParams({db: database, position: `chr${chromosome}:${start}-${end}`});
      $("locus-browser").hidden = false;
      const ldURL = getLDBrowserURL(result.report, chromosome, start, end);
      if (ldURL) {
        $("locus-ld-browser").href = ldURL;
        $("locus-ld-browser").hidden = false;
      }
    }
  }
  $("weight-plot-description").textContent=description;
  screenPoints=[];
  $("plot-empty").hidden=rows.length>0;canvas.hidden=!rows.length;$("save-plot").disabled=!rows.length;
  if(!rows.length){$("plot-empty").textContent=`No variants with the required ${kind==="trumpet"?"LD frequencies and effect estimates":kind==="miami"?"positions, GWAS statistics and posterior estimates":"positions and PIPs"} in this selection. Try another chromosome or locus, or inspect the complete download.`;return;}
  canvas.setAttribute("aria-label",`${kind} plot of ${rows.length.toLocaleString()} variants. ${description}`);
  const xx=v=>left+(v-xMin)/(xMax-xMin)*(right-left);
  const yy=v=>kind==="trumpet"?mid-v/yMax*(bottom-top)/2:kind==="miami"?mid-10-v/yMax*(mid-top-20):bottom-v/yMax*(bottom-top);
  const lowerTop=mid+10;
  const yl=v=>lowerTop+v/lowerMax*(bottom-lowerTop);
  line(left,top,left,bottom);line(left,bottom,right,bottom);
  for(let i=0;i<3;i++) {
    const val=kind==="trumpet" ? -yMax+yMax*i : yMax*i/2;
    const py=yy(val);line(left,py,right,py);text(fmt(val),left-8,py+4,"right");
    if(kind==="miami"){const low=lowerMax*i/2,ly=yl(low);line(left,ly,right,ly);text(fmt(low),left-8,ly+4,"right");}
  }
  text(upperLabel,left,19,"left",kind==="miami"?"#a85b49":"#205540");
  if(kind==="miami") text(lowerLabel,left,h-12,"left","#205540");
  if(kind==="trumpet")line(left,mid,right,mid,"#95a795");
  for(const row of rows) {
    const x=xx(xValue(row)), y=yy(yValue(row)), ci=result.report.chromosomes.indexOf(String(row.CHR));
    ctx.fillStyle=kind==="miami"?(ci%2?"#d9aa98":"#b86a55"):row.PIP>=.5?"#cd745e":"#29664e";
    ctx.globalAlpha=.65;ctx.beginPath();ctx.arc(x,y,2.4,0,Math.PI*2);ctx.fill();screenPoints.push({x,y,row});
    if(lowerValue){const by=yl(lowerValue(row));ctx.fillStyle=ci%2?"#8eae9a":"#29664e";ctx.beginPath();ctx.arc(x,by,2.4,0,Math.PI*2);ctx.fill();screenPoints.push({x,y:by,row});}
  }
  ctx.globalAlpha=1;
  if(centers.length) {const step=Math.ceil(centers.length/Math.max(3,Math.floor((right-left)/23)));centers.forEach(([c,v],i)=>{if(i%step===0)text(c,xx(v),bottom+20,"center");});}
  else for(let i=0;i<3;i++){const v=xMin+(xMax-xMin)*i/2;text(kind==="locus"?(v/1e6).toFixed(2):v.toFixed(2),xx(v),bottom+20,"center");}
  text(xLabel,(left+right)/2,h-(kind==="miami"?38:12),"center");
}

export const resultsPanel = {
  clear() {
    generation++;result=null;plotRows=[];screenPoints=[];selected=null;prepared=false;makeDownload=null;
    $("variant-detail").textContent="Select a variant to see its position, alleles and posterior estimates.";
    $("variant-id").value="";$("locus-center").value="";
    exports.clear(); outputChosen = false; updateDownload();
    $("locus-browser").hidden=true; $("locus-ld-browser").hidden=true;activate("summary");
  },
  show(next, download) {
    this.clear();result=next;makeDownload=download;
    plotRows=next.plot_data?.rows || next.rows || [];
    renderSummary();
    $("scoring-build").textContent=`Weight build: ${build() || "not recorded — verify before scoring"}. Native BETA uses standardized genotypes. The dosage download divides by √(2p(1−p)) using LD-reference frequencies. Score = Σ dosage(A1) × dosage weight, on a standardized phenotype scale; an absolute risk or phenotype prediction requires separate calibration. Match genome build, alleles and variant IDs before scoring.`;
    $("scoring-tool").value="plink";$("biobank-platform").value="generic";
    $("genotype-format").value="pgen";$("execution-runtime").value="native";$("genotype-layout").value="combined";
    saveExport("native", next.tsv, "text/tab-separated-values");
    saveExport("report", JSON.stringify(next.report, null, 2), "application/json");
    recommendScoring();
    $("download-format").value = "native"; updateDownload();
    $("plot-chromosome").replaceChildren(new Option("All chromosomes",""),...next.report.chromosomes.map(c=>new Option(`Chromosome ${c}`,c)));
    $("weight-plot-kind").value="miami";$("miami-metric").value="PIP";$("trumpet-scale").value="dosage";
    activate("summary");
  }
};
