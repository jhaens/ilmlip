// The command builder: a form per task, the command or script it gives, and what that does.
const builder = (() => {
  const sized = (pre, post = "", sizes = ["XS", "S", "M", "L", "XL", "XXL"]) => sizes.map((z) => `${pre}-${z}${post}`);
  const GROUPS = [
    ["Inorganic materials", [...sized("OAM", "-omat"), ...sized("OAM", "-mpa"), "OAM-L-mpa-WiSE10", "OAM-L-omat-WiSE10", "OAM-XXL-mpa-WiSE10", "OAM-S30-mpa",
      ...sized("MATPES-PBE"), ...sized("MATPES-r2SCAN")]],
    ["Molecules", sized("OMOL", "", ["XS", "S", "M", "L", "XL"])],
    ["Solid acids", sized("SOLIDACID", "", ["S", "M", "L"])],
  ];
  const COMMITTEES = ["OAM-L-mpa", "OAM-L-omat"];
  const noStress = (m) => m.startsWith("OMOL") || m.startsWith("SOLIDACID");
  const takesChargeSpin = (s) => s.model.startsWith("OMOL") || s.model === "file";
  const ASE = "https://docs.ase-lib.org/ase/";
  const link = (url, text) => `<a href="${url}" target="_blank" rel="noopener">${text}</a>`;
  // BibTeX keys of references.bib: the data each model family was trained on
  const DATA = [["OAM", ["omat24", "alexandria", "mptrj", "mp_1", "mp_2"], "OMat24, sAlex (Alexandria) and MPtrj (Materials Project)"],
    ["MATPES", ["matpes"], "MatPES"], ["OMOL", ["omol25"], "OMol25"],
    ["SOLIDACID", ["solidacid0", "haenseroth2026htscreening"], "SOLID-ACID-0"]];
  const data = (name) => DATA.find(([p]) => name.startsWith(p)) || [null, [], null];

  function about(name) {
    const size = { XS: "XS is the fastest.", S: "S: fast, a good start.", M: "M: S with a wider readout.", L: "L: richer features than M.",
      XL: "XL: more accurate, slower.", XXL: "XXL is the largest and most accurate." }[name.match(/-(XS|S|M|L|XL|XXL)\b/)?.[1]] ?? "";
    const data = name.startsWith("OAM") && name.endsWith("-mpa")
      ? "Inorganic materials. OMat24, sAlex and MPtrj, last stage MPtrj and sAlex: Materials Project energies, as used by Matbench Discovery."
      : name.startsWith("OAM") ? "Inorganic materials. OMat24, sAlex and MPtrj, last stage OMat24: OMat24 energies (PBE and PBE+U)."
      : name.startsWith("MATPES-PBE") ? "Inorganic materials: MatPES at the PBE level (no +U)."
      : name.startsWith("MATPES") ? "Inorganic materials: MatPES at the r2SCAN level (meta-GGA)."
      : name.startsWith("OMOL") ? "Molecules: OMol25 (ωB97M-V/def2-TZVPD), with the total charge and spin multiplicity as inputs."
      : "Solid acids: SOLID-ACID-0 (CP2K, PBE-D3).";
    return `${data} ${size}${name.includes("WiSE") ? " A weight average of ten runs of the last stage (WiSE), as fast as one model." : ""}`;
  }

  // fields: {type, key, label, def, help, show(s), more, options, place, min, max, step}
  const field = (type) => (key, label, def, help, extra = {}) => ({ type, key, label, def, help, ...extra });
  const seg = field("seg"), num = field("number"), txt = field("text"), tog = field("toggle"), sel = field("select");
  const row = (fields, extra = {}) => ({ type: "row", fields, ...extra });
  const section = (title) => ({ type: "section", title });
  const modelOptions = [...GROUPS.map(([g, names]) => [g, names.map((n) => [n, n])]), ["Your own", [["file", "Checkpoint file…"]]]];
  const job = [
    section("Run on"),
    seg("job", "Where", "shell", (s) => ({
      shell: "A terminal on a machine with a GPU.",
      slurm: "A job script for SLURM: submit it with <code>sbatch</code>.",
      lsf: "A job script for LSF: submit it with <code>bsub &lt;</code>.",
    })[s.job], { options: [["shell", "Terminal"], ["slurm", "SLURM"], ["lsf", "LSF"]] }),
    row([num("hours", "Hours", 24, "", { min: 1, max: 720, step: 1 }), txt("queue", "Queue", "", "", { place: "default" })],
      { show: (s) => s.job !== "shell", help: "Time limit and, if your cluster wants one, the partition or queue." }),
  ];
  const parallelWise = (s) => s.mode === "naive" && Number(s.wise) > 1 && Number(s.wisep) > 1;
  const tracking = [
    seg("track", "Live tracking", "none", `Log the training curves to ${link("https://wandb.ai/site", "Weights & Biases")} or ${link("https://mlflow.org/", "MLflow")} (installed separately). If the tracker fails, training goes on.`,
      { options: [["none", "none"], ["wandb", "wandb"], ["mlflow", "mlflow"]] }),
    txt("project", "Project", "ilmlip", "The wandb project or MLflow experiment.", { show: (s) => s.track !== "none" }),
  ];

  const TABS = {
    md: {
      title: "Run MD",
      fields: [
        section("Model"),
        sel("model", "Model", "OAM-S-omat", (s) => s.model === "file"
          ? "A checkpoint of your own: a fine-tune's <code>model.pt</code> or a training run's <code>best.pt</code>."
          : about(s.model), { options: modelOptions }),
        txt("file", "Checkpoint file", "model.pt", "", { show: (s) => s.model === "file" }),
        tog("committee", "Committee of five", false,
          "The model and four more trained with other seeds: their mean, and the spread as an uncertainty (<code>energy_std</code>, <code>force_dev</code>, <code>max_force_dev</code> in <code>calc.results</code>). Five times the cost.",
          { show: (s) => COMMITTEES.includes(s.model) }),
        row([num("charge", "Charge", 0, "", { min: -10, max: 10, step: 1 }), num("spin", "Spin multiplicity", 1, "", { min: 1, max: 11, step: 1 })],
          { show: takesChargeSpin, help: (s) => "Total charge (−10 … 10) and multiplicity 2S + 1 (1 … 11) of the whole system."
            + (s.model === "file" ? " Only for models with charge and spin inputs." : "") }),
        section("Hardware"),
        seg("device", "Device", "auto", (s) => ({
          auto: "The GPU when JAX can use one, else the CPU.",
          cuda: "The GPU, in JAX; with the fused kernels when cuEquivariance is installed.",
          cpu: "The CPU, in the C++ engine; compiled for this CPU on first use (needs <code>g++</code>).",
        })[s.device], { options: [["auto", "auto"], ["cuda", "GPU"], ["cpu", "CPU"]] }),
        seg("engine", "Engine", "auto", (s) => ({
          auto: "JAX on the GPU, the C++ engine on the CPU. What cannot run here falls back with a warning.",
          jax: "A neighbour list on the device, rebuilt in the background; one compiled call per step.",
          cpp: "The hand-written C++ engine: OpenMP over the atoms, SIMD; float32 unless the precision is float64.",
          torch: "The training model evaluated directly: slow, for checks.",
        })[s.engine], {
          options: [["auto", "auto"], ["jax", "JAX"], ["cpp", "C++"], ["torch", "PyTorch"]],
          disabled: (s, v) => v === "cpp" && s.device === "cuda",
        }),
        seg("dtype", "Precision", "auto", (s) => ({
          auto: "The precision the model was trained in: tf32 for the released models, float64 for one trained in double precision.",
          tf32: "float32, with the network's matrix products in TensorFloat-32 on NVIDIA GPUs of the Ampere generation or newer; positions, cell and strain stay in float32. On CPUs the same as float32.",
          float32: "float32 throughout.",
          float64: "Double precision: slow on most GPUs that are not data-centre ones.",
        })[s.dtype], { options: [["auto", "auto"], ["tf32", "tf32"], ["float32", "float32"], ["float64", "float64"]] }),
        num("threads", "Threads", "", "Threads of the C++ engine. Empty: OpenMP's default count (OMP_NUM_THREADS if set); 1 for one MD per core.",
          { place: "all", min: 1, max: 1024, step: 1, show: (s) => s.device !== "cuda" && ["auto", "cpp"].includes(s.engine) }),
        section("Script"),
        seg("script", "What to write", "md", (s) => s.script === "md"
          ? `A complete MD run with ASE 3.29 or newer: read a structure, set velocities (${link(ASE + "md.html#ase.md.velocitydistribution.thermalize_momenta", "thermalize_momenta")}), integrate, write a trajectory and a log (${link(ASE + "md.html", "MD in ASE")}).`
          : "Only the calculator, for an <code>atoms</code> you already have.",
          { options: [["md", "MD script"], ["calc", "Calculator only"]] }),
        txt("structure", "Structure", "structure.xyz", `Any file ${link(ASE + "io/io.html", "ASE reads")}: extxyz, POSCAR, CIF, …`, { show: (s) => s.script === "md" }),
        seg("ensemble", "Ensemble", "nvt", (s) => ({
          nve: `${link(ASE + "md.html#ase.md.verlet.VelocityVerlet", "Velocity Verlet")} at constant energy.`,
          nvt: `Nosé–Hoover chain thermostat (${link(ASE + "md.html#ase.md.nose_hoover_chain.NoseHooverChainNVT", "NoseHooverChainNVT")}), time constant 100 time steps. Other thermostats: ${link(ASE + "md.html#constant-nvt-simulations-the-canonical-ensemble", "NVT in ASE")}.`,
          npt: `Martyna–Tobias–Klein barostat with Nosé–Hoover chains (${link(ASE + "md.html#ase.md.nose_hoover_chain.IsotropicMTKNPT", "IsotropicMTKNPT")}), time constants 100 and 1000 time steps; needs a periodic cell. Others: ${link(ASE + "md.html#constant-npt-simulations-the-isothermal-isobaric-ensemble", "NPT in ASE")}.`,
        })[s.ensemble] + (noStress(s.model) ? " This model was trained without stresses: no NPT." : ""), {
          options: [["nve", "NVE"], ["nvt", "NVT"], ["npt", "NPT"]], show: (s) => s.script === "md",
          disabled: (s, v) => v === "npt" && noStress(s.model),
        }),
        row([num("temperature", "Temperature (K)", 300, "", { min: 1, max: 10000, step: 1 }),
          num("pressure", "Pressure (bar)", 1, "", { min: 0, max: 1e6, show: (s) => s.ensemble === "npt" })],
          { show: (s) => s.script === "md" }),
        row([num("timestep", "Time step (fs)", 0.5, "", { min: 0.5, max: 5, step: 0.5 }), num("steps", "Steps", 20000, "", { min: 1, max: 1e9, step: 1 }),
          num("interval", "Write every", 100, "", { min: 1, max: 1e9, step: 1 })],
          { show: (s) => s.script === "md", help: `Time step in steps of 0.5 fs; 0.5 fs is safe with hydrogen (${link(ASE + "md.html#choosing-the-time-step", "choosing the time step")}). Write every: steps between trajectory and log entries.` }),
        txt("prefix", "Output name", "md", (s) => `Writes <code>${esc(s.prefix || "md")}.traj</code> (${link(ASE + "io/trajectory.html", "trajectory")}) and <code>${esc(s.prefix || "md")}.log</code> (${link(ASE + "md.html#ase.md.MDLogger", "MD log")}).`,
          { show: (s) => s.script === "md" }),
        seg("kernels", "GPU kernels", "auto", "fused: cuEquivariance's kernels; xla: plain JAX. auto takes fused when it is installed.",
          { more: true, options: [["auto", "auto"], ["fused", "fused"], ["xla", "xla"]], show: (s) => s.device !== "cpu" && ["auto", "jax"].includes(s.engine) }),
        num("skin", "Verlet skin (Å)", 1.5, "The neighbour list holds pairs up to the cutoff plus this; larger means fewer rebuilds, more pairs.",
          { more: true, min: 0, max: 5, step: 0.1, show: (s) => s.engine !== "torch" }),
      ],
      build: buildMd,
    },
    finetune: {
      title: "Fine-tune",
      fields: [
        section("Start from"),
        sel("foundation", "Foundation", "OAM-S-omat", (s) => s.foundation === "file"
          ? "An ILMlip checkpoint: a pretrained model file or an earlier fine-tune's <code>model.pt</code>."
          : `${about(s.foundation)} Given by name; downloaded once if it is not on this machine.`, { options: modelOptions }),
        txt("ffile", "Checkpoint file", "foundation.pt", "", { show: (s) => s.foundation === "file" }),
        section("Data"),
        txt("train", "Training structures", "data.xyz", (s) => s.mode === "multihead"
          ? "One or more files; each gets its own head (readout and reference energies) on the shared descriptor."
          : "One or more files with energies and forces (extxyz or anything ASE reads); stress where present."),
        seg("valid", "Validation", "split", (s) => s.valid === "split"
          ? "A fraction of the training structures is held out; the best epoch on it is kept."
          : "Files of your own" + (s.mode === "multihead" ? ", one per training file, in the same order." : "."),
          { options: [["split", "Split off"], ["files", "Own files"]] }),
        num("fraction", "Fraction held out", 0.1, "", { min: 0.01, max: 0.5, step: 0.01, show: (s) => s.valid === "split" }),
        txt("vfiles", "Validation structures", "valid.xyz", "", { show: (s) => s.valid === "files" }),
        txt("test", "Test structures", "", (s) => "Optional; evaluated before and after training, not used to pick the epoch"
          + (s.mode === "multihead" ? "; one per training file, in the same order." : "."), { place: "none" }),
        seg("e0s", "Reference energies", "estimated", (s) => ({
          estimated: s.mode === "scratch" ? "From scratch, composition is always used: a least-squares fit to your energies alone."
            : "The foundation's, plus a per-element correction fitted to your energies: works across codes and settings.",
          composition: "A least-squares fit to your energies alone.",
          foundation: "The foundation's own, for data at its level of theory.",
          custom: "Your own per element, as JSON or a .json file.",
        })[s.e0s], {
          options: [["estimated", "estimated"], ["composition", "composition"], ["foundation", "foundation"], ["custom", "custom"]],
          disabled: (s, v) => v === "foundation" && s.mode === "scratch",
        }),
        txt("e0custom", "Energies (eV)", '{"H": -13.6, "O": -432.1}', "", { show: (s) => s.e0s === "custom" }),
        section("Method"),
        seg("mode", "Mode", "naive", (s) => ({
          naive: "Every weight trains. The usual choice.",
          lora: "The foundation stays frozen; every weight matrix trains a low-rank update, merged into it on saving.",
          multihead: "Every weight trains. Each training file gets its own head; one more head trains on the foundation's own predictions for replay structures, so that the shared descriptor keeps reproducing the foundation.",
          scratch: "The foundation's architecture with random weights: the baseline to compare with.",
        })[s.mode], { options: [["naive", "All weights"], ["lora", "LoRA"], ["multihead", "Multi-head"], ["scratch", "From scratch"]] }),
        row([num("rank", "LoRA rank", 8, "", { min: 1, max: 256, step: 1 }), num("alpha", "LoRA alpha", "", "", { place: "= rank", min: 0.001, max: 10000 })],
          { show: (s) => s.mode === "lora", help: "Each weight matrix trains W + (α/r)·B·A with rank r; α empty = the rank, so the scale α/r is 1." }),
        txt("replay", "Replay structures", "", (s) => (s.mode === "multihead"
          ? "Structures labelled by the foundation: the replay head trains on them, and they measure the drift. "
          : "Optional: structures labelled by the foundation, to measure the drift (how far the fine-tune moves away from it). ")
          + "A folder from <code>python -m ilmlip.finetune prepare-replay</code>, <code>mptrj</code> (download), a URL, a file, or <code>none</code>. "
          + "Empty: <code>replay_set.extxyz</code> and <code>drift_test_set.extxyz</code> in the current folder"
          + (s.mode === "multihead" ? "; without them, MPtrj frames are downloaded." : ", if present."),
          { place: "replay files here", show: (s) => s.mode !== "scratch" }),
        num("replayw", "Replay weight", 1, "Weight of the replay head's loss; 0 trains no replay head.", { min: 0, max: 100, show: (s) => s.mode === "multihead", more: true }),
        num("replayn", "Replay structures to draw", 1000, "How many structures are drawn from mptrj, a URL or a file, among those of the elements the foundation was trained on.", { min: 1, max: 1e6, step: 1, show: (s) => s.mode !== "scratch", more: true }),
        row([num("wise", "WiSE runs", 1, "", { min: 1, max: 64, step: 1 }), num("wisep", "At once", 1, "", { min: 1, max: 64, step: 1 })], {
          show: (s) => s.mode === "naive",
          help: "N fine-tunes that differ only in the order of the batches; model.pt is their weight average. They run one after the other, or P at once on the visible GPUs.",
        }),
        txt("init", "Continue from", "", "An earlier fine-tune of the same foundation (its <code>model.pt</code>): another cycle of epochs with a new warm-up. Use the same data and seed.",
          { place: "none", more: true }),
        section("Training"),
        row([num("epochs", "Epochs", 200, "", { min: 1, max: 1e5, step: 1 }), num("batch", "Structures per batch", 4, "", { min: 1, max: 4096, step: 1 })]),
        num("lr", "Peak learning rate", "", (s) => s.mode === "scratch" ? "Empty: the foundation's."
          : `Empty: ${s.mode === "lora" ? "1e-2" : s.optimizer === "muon" ? "1.5e-3" : "3e-3"}; higher values drift further from the foundation.`,
          { place: "default", min: 1e-7, max: 1 }),
        seg("optimizer", "Optimizer", "adam", (s) => s.optimizer === "muon"
          ? "Muon on the network's weight matrices, Adam on the rest. Muon fine-tunes move the weights about four times farther and were less stable in MD."
          : "Adam alone.", { more: true, options: [["adam", "Adam"], ["muon", "Muon"]], show: (s) => s.mode !== "lora" }),
        row([num("we", "Energy weight", 1, "", { min: 0, max: 1e4 }), num("wf", "Forces", 10, "", { min: 0, max: 1e4 }), num("ws", "Stress", 10, "", { min: 0, max: 1e4 })],
          { help: "Loss weights." }),
        num("huber", "Huber threshold", "", "A Huber loss with this threshold instead of squared errors.", { place: "none", min: 1e-6, max: 100, more: true }),
        row([num("ema", "EMA", 0.99, "", { min: 0, max: 0.9999, step: 0.0001 }), num("clip", "Gradient clip", 10, "", { min: 0.001, max: 1e4 }),
          num("seed", "Seed", 0, "", { min: 0, max: 2147483647, step: 1 })], {
          more: true, help: "Moving average of the weights; clip of the gradient norm; seed of the split, the shuffling and LoRA.",
        }),
        row([txt("ekey", "Energy key", "", "", { place: "auto" }), txt("fkey", "Forces key", "", "", { place: "auto" }), txt("skey", "Stress key", "", "", { place: "auto" })], {
          more: true, help: "Found automatically under the usual names (REF_energy, energy, free_energy, …); a key with 'virial' is read as a virial.",
        }),
        seg("fdevice", "Device", "auto", "auto: the GPU when PyTorch sees one.", { more: true, options: [["auto", "auto"], ["cuda", "GPU"], ["cpu", "CPU"]] }),
        seg("fdtype", "Precision", "tf32", "float64 for data from a code whose energies need it; float32 without TF32 matrix products.", { more: true, options: [["tf32", "tf32"], ["float32", "float32"], ["float64", "float64"]] }),
        seg("compile", "Compiled step", "auto", "auto: CUDA graphs on a GPU (PyTorch 2.11 or newer), eager on a CPU.",
          { more: true, options: [["auto", "auto"], ["on", "on"], ["off", "off"]] }),
        seg("resident", "Structures", "device", "On the GPU, or copied there for every batch (for very large sets).",
          { more: true, options: [["device", "on the GPU"], ["host", "copied per batch"]] }),
        section("Output"),
        txt("out", "Output folder", "", (s) => `Empty: <code>${esc(defaultOut(s))}</code>.`, { place: "automatic" }),
        ...tracking,
        ...job,
        num("gpus", "GPUs", 1, "One GPU per run at once: as many as WiSE runs at once. Training and MD use one.",
          { min: 1, max: 64, step: 1, show: (s) => s.job !== "shell" && parallelWise(s) }),
      ],
      build: buildFinetune,
    },
    train: {
      title: "Train",
      fields: [
        section("Data"),
        txt("train", "Training structures", "train.xyz", "Files with energies and forces (extxyz or anything ASE reads); stress, charge and spin where present."),
        txt("val", "Validation structures", "val.xyz", "Evaluated during training; the checkpoint that is best on them becomes best.pt."),
        txt("dataset", "Dataset folder", "dataset", "Packed once; batches and neighbour lists are then built on the GPU."),
        seg("scale", "Energy scale", "energy", (s) => s.scale === "forces"
          ? "Set from the force RMS: for total energies (e.g. from CP2K), whose spread per atom is large."
          : "Set from the spread of the energy per atom.",
          { options: [["energy", "energy"], ["forces", "forces"]] }),
        tog("chargespin", "Charge and spin as inputs", false, "From each structure's <code>info['charge']</code> and <code>info['spin']</code>, as for OMol25."),
        section("Model"),
        seg("size", "Size", "S", (s) => ({
          XS: "140,456 parameters: the fastest; cutoffs per element, 4–6 Å.",
          S: "300,612 parameters: XS with more channels and a 3 × 256 readout; cutoffs per element, 4–6 Å.",
          M: "849,732 parameters: S with a readout of width 512; cutoffs per element, 4–6 Å.",
          L: "963,587 parameters: M with richer features (l ≤ 2); cutoffs per element, 4–6 Å.",
          XL: "6,953,980 parameters: l ≤ 3 in the pair products, readout of width 1024; one 6 Å cutoff for all elements.",
          XXL: "27,822,308 parameters: l ≤ 3 throughout, readout of width 2048; one 6 Å cutoff for all elements.",
          json: "Your own architecture and peak learning rate, as in <code>ilmlip/configs/S.json</code>.",
        })[s.size], { options: ["XS", "S", "M", "L", "XL", "XXL"].map((z) => [z, z]).concat([["json", "JSON file"]]) }),
        txt("hjson", "Configuration file", "config.json", "", { show: (s) => s.size === "json" }),
        tog("zbl", "ZBL repulsion", true, "The shipped sizes have it already.", { show: (s) => s.size === "json" }),
        section("Training"),
        seg("length", "Length", "epochs", "", { options: [["epochs", "epochs"], ["steps", "steps"]] }),
        num("epochs", "Epochs", 100, "", { min: 1, max: 1e5, step: 1, show: (s) => s.length === "epochs" }),
        num("steps", "Steps", 1000000, "", { min: 1, max: 1e9, step: 1, show: (s) => s.length === "steps" }),
        row([num("atoms", "Atoms per batch", 8192, "", { min: 1, max: 1e7, step: 1 }), num("structs", "Structures per batch", 512, "", { min: 1, max: 1e5, step: 1 })],
          { help: "A batch ends at whichever limit comes first." }),
        seg("loss", "Loss", "mae", "Absolute errors are robust to the few outliers that large datasets have.", { options: [["mae", "absolute"], ["huber", "Huber"], ["mse", "squared"]] }),
        row([num("we", "Energy weight", 1, "", { min: 0, max: 1e4 }), num("wf", "Forces", 1, "", { min: 0, max: 1e4 }), num("ws", "Stress", 3, "", { min: 0, max: 1e4 })], {
          help: "Stress counts only where the data have it.",
        }),
        num("lr", "Peak learning rate", "", "Empty: the configuration's.", { place: "default", min: 1e-7, max: 1 }),
        seg("precision", "Precision", "tf32", "tf32: float32 with TF32 matrix products, faster on A100/H100-class GPUs and slightly less precise; float64 for data that need it (e.g. CP2K total energies of large cells).",
          { options: [["tf32", "tf32"], ["float32", "float32"], ["float64", "float64"]] }),
        tog("compile", "Compile the step", true, "torch.compile of energy and forces (PyTorch 2.11 or newer)."),
        seg("cmode", "Compile mode", "default", "reduce-overhead replays the step as CUDA graphs.",
          { more: true, options: [["default", "default"], ["reduce-overhead", "reduce-overhead"]], show: (s) => s.compile }),
        row([num("ema", "EMA", 0.999, "", { min: 0, max: 0.9999, step: 0.0001 }), num("warmup", "Warm-up steps", 2000, "", { min: 0, max: 1e7, step: 1 }),
          num("seed", "Seed", 0, "", { min: 0, max: 2147483647, step: 1 })], { more: true }),
        row([num("valevery", "Validate every", 2000, "", { min: 1, max: 1e9, step: 1 }), num("ckptevery", "Checkpoint every", 2000, "", { min: 1, max: 1e9, step: 1 }),
          num("logevery", "Log every", 50, "", { min: 1, max: 1e9, step: 1 })], { more: true, help: "In steps." }),
        txt("selecton", "Best checkpoint by", "f_rmse", "The validation metric that picks best.pt.", { more: true }),
        txt("init", "Start from", "", "A checkpoint whose weights start the run (fresh optimiser), e.g. for a second stage on other data.", { more: true, place: "none" }),
        row([num("refit", "Refit reference energies (batches)", 0, "", { min: 0, max: 1e6, step: 1 }), tog("freeze", "Freeze reference energies", false)], {
          more: true, help: "With a start checkpoint: refit the reference energies on this many batches first. Freeze keeps them fixed.",
        }),
        seg("tdevice", "Device", "auto", "auto: the GPU when PyTorch sees one.", { more: true, options: [["auto", "auto"], ["cuda", "GPU"], ["cpu", "CPU"]] }),
        seg("resident", "Data", "auto", "auto: on the GPU when it takes less than half of its free memory, else memory-mapped on the host.",
          { more: true, options: [["auto", "auto"], ["device", "GPU"], ["host", "host"]] }),
        section("Output"),
        txt("out", "Run folder", "", (s) => `Empty: <code>runs/${esc(sizeName(s))}</code>. Also the resume point.`, { place: "automatic" }),
        tog("evaluate", "Evaluate at the end", true, "Errors of best.pt on the validation structures."),
        ...tracking,
        ...job,
      ],
      build: buildTrain,
    },
  };

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const q = (s) => (/^[\w@%+=:,./~-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`);
  const words = (s) => String(s).trim().split(/\s+/).filter(Boolean);
  const set = (v) => v !== "" && v !== undefined && v !== null;
  const differs = (v, def) => set(v) && Number(v) !== Number(def);
  const stem = (p) => p.split("/").pop().replace(/\.[^.]*$/, "");
  const sizeName = (s) => (s.size === "json" ? stem(s.hjson || "config") : s.size);

  function defaultOut(s) {
    const found = s.foundation === "file" ? stem(s.ffile || "foundation") : s.foundation;
    const wise = s.mode === "naive" ? Math.max(1, Number(s.wise) || 1) : 1;
    return `${stem(words(s.train)[0] || "data")}_${found}_${s.mode}${wise > 1 ? `_wise${wise}` : ""}`;
  }

  function command(head, args) {
    const line = [head, ...args].join(" ");
    return line.length <= 84 ? line : [head, ...args].join(" \\\n    ");
  }

  function wrapJob(s, name, body) {
    if (s.job === "shell") return body;
    const h = Math.max(1, Number(s.hours) || 24), g = Math.max(1, Number(s.gpus) || 1);
    const head = s.job === "slurm"
      ? [`#SBATCH --job-name=${name}`, `#SBATCH --gres=gpu:${g}`, `#SBATCH --cpus-per-task=${8 * g}`, `#SBATCH --time=${h}:00:00`,
        `#SBATCH --output=${name}_%j.log`, ...(s.queue ? [`#SBATCH --partition=${s.queue}`] : [])]
      : [`#BSUB -J ${name}`, `#BSUB -gpu "num=${g}"`, `#BSUB -n ${8 * g}`, `#BSUB -W ${h}:00`, `#BSUB -o ${name}_%J.log`,
        ...(s.queue ? [`#BSUB -q ${s.queue}`] : [])];
    return ["#!/bin/bash", ...head, "", body].join("\n");
  }

  function trackArgs(s) {
    if (s.track === "none") return [];
    return [`--track ${s.track}`, ...(s.project && s.project !== "ilmlip" ? [`--track-project ${q(s.project)}`] : [])];
  }

  function trackNote(s, out, wise = 1) {
    const o = esc(wise > 1 ? `${out}/member_00` : out), all = esc(wise > 1 ? `${out}/member_*` : out);
    return s.track === "wandb"
      ? `Curves go to W&amp;B (<code>pip install wandb</code>): <code>wandb leet ${o}/wandb</code> shows them in the terminal (${link("https://docs.wandb.ai/models/ref/cli/wandb-leet", "LEET")}${wise > 1 ? "; each WiSE run has its own" : ""}); without internet set <code>WANDB_MODE=offline</code> and run <code>wandb sync ${all}/wandb/offline-run-*</code> later.`
      : s.track === "mlflow"
        ? `Curves go to MLflow (<code>pip install mlflow</code>), into <code>mlflow.db</code> in the current folder unless <code>MLFLOW_TRACKING_URI</code> says otherwise: <code>mlflow ui --backend-store-uri sqlite:///mlflow.db</code> shows them at localhost:5000 (${link("https://mlflow.org/docs/latest/ml/tracking/", "MLflow tracking")}).`
        : null;
  }

  function submitNote(s, file) {
    return s.job === "slurm" ? `Submit with <code>sbatch ${file}</code>.` : s.job === "lsf" ? `Submit with <code>bsub &lt; ${file}</code>.` : null;
  }

  function foundationArg(name, file) {
    return name === "file" ? q(file || "foundation.pt") : name;
  }

  // a number field's value within its bounds, on its step
  function clamp(f, v) {
    if (!f || f.type !== "number" || !set(v)) return v;
    let x = Number(v);
    if (!Number.isFinite(x)) return String(f.def);
    const lo = f.min ?? -Infinity, hi = f.max ?? Infinity;
    x = Math.min(hi, Math.max(lo, x));
    if (f.step) x = Math.min(hi, (Number.isFinite(lo) ? lo : 0) + Math.round((x - (Number.isFinite(lo) ? lo : 0)) / f.step) * f.step);
    return String(+x.toFixed(10));
  }

  function buildMd(s) {
    const name = s.model === "file" ? (s.file || "model.pt") : s.model + (s.committee && COMMITTEES.includes(s.model) ? "-committee" : "");
    const kw = [];
    if (s.device !== "auto") kw.push(`device="${s.device}"`);
    if (s.engine !== "auto") kw.push(`engine="${s.engine}"`);
    if (s.kernels !== "auto" && s.device !== "cpu" && ["auto", "jax"].includes(s.engine)) kw.push(`kernels="${s.kernels}"`);
    if (s.dtype !== "auto") kw.push(`dtype="${s.dtype}"`);
    if (takesChargeSpin(s)) {
      if (differs(s.charge, 0)) kw.push(`charge=${Number(s.charge)}`);
      if (differs(s.spin, 1)) kw.push(`spin=${Number(s.spin)}`);
    }
    if (set(s.threads) && s.device !== "cuda" && ["auto", "cpp"].includes(s.engine)) kw.push(`threads=${Number(s.threads)}`);
    if (differs(s.skin, 1.5) && s.engine !== "torch") kw.push(`skin=${Number(s.skin)}`);
    let calc = `load_calculator(${[JSON.stringify(name), ...kw].join(", ")})`;
    if (calc.length > 56) calc = `load_calculator(\n    ${[JSON.stringify(name), ...kw].join(",\n    ")},\n)`;

    const committee = s.committee && COMMITTEES.includes(s.model);
    const cite = ["ilmlip", ...data(s.model)[1], "ase"];
    const notes = [
      s.model === "file" ? `Loads <code>${esc(name)}</code>.`
        : `${committee ? "The five model files are" : `<code>${esc(name)}</code> is`} taken from this machine, or else downloaded once into <code>~/.cache/ilmlip/models</code>.`,
      committee ? "<code>calc.results[\"force_dev\"]</code>: the spread of the forces per atom, a measure of uncertainty; <code>calc.members</code> are the five calculators."
        : "<code>calc.engine</code> and <code>calc.device</code> say what runs; anything that cannot run here falls back with a warning.",
    ];
    if (s.script === "calc") {
      return { lang: "python", file: "calculator.py", notes, cite,
        code: `from ilmlip import load_calculator\n\natoms.calc = ${calc}\n` };
    }
    const T = Number(s.temperature) || 300, dt = Number(s.timestep) || 0.5, P = set(s.pressure) ? Number(s.pressure) : 1;
    const every = Math.max(1, Number(s.interval) || 100), steps = Math.max(0, Number(s.steps) || 0), out = s.prefix || "md";
    const files = `trajectory=${JSON.stringify(`${out}.traj`)}, logfile=${JSON.stringify(`${out}.log`)}, loginterval=${every}`;
    const call = (name, groups) => `${name}(\n${groups.map((g) => `    ${g},`).join("\n")}\n)`;
    const step = `atoms, ${dt} * units.fs`, damp = (n) => `${+(n * dt).toFixed(6)} * units.fs`;
    const imports = ["import numpy as np", "from ase import units", "from ase.io import read"];
    let dyn;
    if (s.ensemble === "nve") {
      imports.push("from ase.md.verlet import VelocityVerlet");
      dyn = call("VelocityVerlet", [step, files]);
    } else if (s.ensemble === "nvt") {
      imports.push("from ase.md.nose_hoover_chain import NoseHooverChainNVT");
      dyn = call("NoseHooverChainNVT", [step, `temperature_K=${T}, tdamp=${damp(100)}`, files]);
    } else {
      imports.push("from ase.md.nose_hoover_chain import IsotropicMTKNPT");
      dyn = call("IsotropicMTKNPT", [step, `temperature_K=${T}, pressure_au=${P} * units.bar`,
        `tdamp=${damp(100)}, pdamp=${damp(1000)}`, files]);
    }
    imports.push("from ase.md.velocitydistribution import thermalize_momenta");
    const code = [...imports, "", "from ilmlip import load_calculator", "",
      `atoms = read(${JSON.stringify(s.structure || "structure.xyz")})`, `atoms.calc = ${calc}`, "",
      `thermalize_momenta(atoms, temperature_K=${T}, rng=np.random.default_rng())`, `dyn = ${dyn}`, `dyn.run(${steps})`, ""].join("\n");
    notes.push(`${steps.toLocaleString("en")} steps of ${dt} fs = ${+(steps * dt / 1000).toFixed(3)} ps; <code>${esc(out)}.traj</code> and <code>${esc(out)}.log</code> every ${every} steps.`);
    notes.push(`Run it with <code>python ${esc(out)}.py</code>.`);
    return { lang: "python", file: `${out}.py`, code, notes, cite };
  }

  function buildFinetune(s) {
    const a = [`--foundation ${foundationArg(s.foundation, s.ffile)}`, `--train ${words(s.train || "data.xyz").map(q).join(" ")}`];
    const n = words(s.train || "data.xyz").length;
    if (s.mode !== "naive") a.push(`--mode ${s.mode}`);
    if (s.valid === "files") a.push(`--valid ${words(s.vfiles || "valid.xyz").map(q).join(" ")}`);
    else if (differs(s.fraction, 0.1)) a.push(`--valid-fraction ${Number(s.fraction)}`);
    if (set(s.test)) a.push(`--test ${words(s.test).map(q).join(" ")}`);
    if (s.e0s === "custom") a.push(`--e0s ${q(s.e0custom || "{}")}`);
    else if (s.e0s !== "estimated") a.push(`--e0s ${s.e0s}`);
    if (s.mode === "lora") {
      if (differs(s.rank, 8)) a.push(`--lora-rank ${Number(s.rank)}`);
      if (set(s.alpha)) a.push(`--lora-alpha ${Number(s.alpha)}`);
    }
    if (s.mode !== "scratch") {
      if (set(s.replay)) a.push(`--replay ${q(s.replay)}`);
      if (differs(s.replayn, 1000)) a.push(`--replay-n ${Number(s.replayn)}`);
    }
    if (s.mode === "multihead" && differs(s.replayw, 1)) a.push(`--replay-weight ${Number(s.replayw)}`);
    const wise = s.mode === "naive" ? Math.max(1, Number(s.wise) || 1) : 1;
    if (wise > 1) {
      a.push(`--wise ${wise}`);
      if (differs(s.wisep, 1)) a.push(`--wise-parallel ${Math.min(wise, Number(s.wisep))}`);
    }
    if (set(s.init)) a.push(`--init ${q(s.init)}`);
    if (differs(s.epochs, 200)) a.push(`--epochs ${Number(s.epochs)}`);
    if (differs(s.batch, 4)) a.push(`--batch-size ${Number(s.batch)}`);
    if (set(s.lr)) a.push(`--lr ${Number(s.lr)}`);
    if (s.optimizer === "muon" && s.mode !== "lora") a.push("--optimizer muon");
    if (differs(s.we, 1)) a.push(`--w-energy ${Number(s.we)}`);
    if (differs(s.wf, 10)) a.push(`--w-forces ${Number(s.wf)}`);
    if (differs(s.ws, 10)) a.push(`--w-stress ${Number(s.ws)}`);
    if (set(s.huber)) a.push(`--huber ${Number(s.huber)}`);
    if (differs(s.ema, 0.99)) a.push(`--ema ${Number(s.ema)}`);
    if (differs(s.clip, 10)) a.push(`--clip ${Number(s.clip)}`);
    if (differs(s.seed, 0)) a.push(`--seed ${Number(s.seed)}`);
    for (const [k, flag] of [["ekey", "--energy-key"], ["fkey", "--forces-key"], ["skey", "--stress-key"]]) if (set(s[k])) a.push(`${flag} ${q(s[k])}`);
    if (s.fdevice !== "auto") a.push(`--device ${s.fdevice}`);
    if (s.fdtype !== "tf32") a.push(`--dtype ${s.fdtype}`);
    if (s.compile !== "auto") a.push(s.compile === "on" ? "--compile" : "--no-compile");
    if (s.resident !== "device") a.push(`--resident ${s.resident}`);
    const out = s.out || defaultOut(s);
    if (set(s.out)) a.push(`--out ${q(s.out)}`);
    a.push(...trackArgs(s));

    const notes = [];
    if (s.foundation !== "file") notes.push(`<code>${s.foundation}</code> is taken from this machine, or else downloaded once into <code>~/.cache/ilmlip/models</code>.`);
    notes.push(s.valid === "split" ? `${+(100 * (Number(s.fraction) || 0.1)).toFixed(1)} % of the structures are held out for validation; the best epoch is kept.`
      : "The best epoch on the validation files is kept.");
    if (s.mode === "multihead") {
      const replayHead = s.replay !== "none" && !(set(s.replayw) && Number(s.replayw) === 0);
      notes.push(`<code>${esc(out)}/model.pt</code> is the first file's head${n > 1 ? ", <code>model_&lt;stem&gt;.pt</code> the head of each further file" : ""}${replayHead ? ", <code>model_pt_head.pt</code> the replay head" : ""}.`);
      if (replayHead && !set(s.replay)) notes.push("Without <code>replay_set.extxyz</code> and <code>drift_test_set.extxyz</code> in the current folder, MPtrj frames are downloaded for the replay head.");
      if (n < 2 && !replayHead) notes.push("<strong>Multi-head needs a second head:</strong> replay structures or a second training file.");
      const per = [s.valid === "files" && words(s.vfiles).length !== n && "validation", set(s.test) && words(s.test).length !== n && "test"].filter(Boolean);
      if (n > 1 && per.length) notes.push(`<strong>Give one ${per.join(" and one ")} file per training file.</strong>`);
    } else if (wise > 1) {
      notes.push(`${wise} runs in <code>${esc(out)}/member_00</code> … and their weight average <code>${esc(out)}/model.pt</code>; running the command again finishes the runs that did not finish.`);
    } else {
      notes.push(`Writes <code>${esc(out)}/model.pt</code> (best epoch), <code>log.txt</code>, <code>history.csv</code> and <code>summary.json</code>.`);
    }
    if (s.foundation.startsWith("OMOL")) notes.push("An OMOL foundation takes the total charge and spin multiplicity from <code>info['charge']</code> and <code>info['spin']</code> of the training structures (0 and 1 where they are missing).");
    notes.push(`Use it: <code>load_calculator("${esc(out)}/model.pt")</code>.`);
    if (s.compile !== "off" && s.fdevice !== "cpu") notes.push("On a GPU the first epoch includes compiling the step; every new batch shape compiles once.");
    const tn = trackNote(s, out, wise), sn = submitNote(s, "finetune.sh");
    if (tn) notes.push(tn);
    if (sn) notes.push(sn);
    const cite = ["ilmlip", ...(s.foundation === "file" ? [] : data(s.foundation)[1])];
    return { lang: "sh", file: "finetune.sh", notes, cite, code: wrapJob(parallelWise(s) ? s : { ...s, gpus: 1 }, "finetune", command("python -m ilmlip.finetune", a)) + "\n" };
  }

  function buildTrain(s) {
    const data = s.dataset || "dataset", out = s.out || `runs/${sizeName(s)}`;
    const pack = command(`python -m ilmlip.data ${q(data)} ${words(s.train || "train.xyz").map(q).join(" ")}`,
      [`--val ${words(s.val || "val.xyz").map(q).join(" ")}`]);
    const a = [`--packed ${q(data)}`, `--hparams ${s.size === "json" ? q(s.hjson || "config.json") : s.size}`];
    if (s.chargespin) a.push("--charge-spin");
    if (s.size === "json" && s.zbl) a.push("--zbl");
    if (s.length === "epochs") a.push(`--epochs ${Math.max(1, Number(s.epochs) || 100)}`);
    else if (differs(s.steps, 1000000)) a.push(`--steps ${Number(s.steps)}`);
    if (differs(s.atoms, 8192)) a.push(`--batch-atoms ${Number(s.atoms)}`);
    if (differs(s.structs, 512)) a.push(`--batch-structs ${Number(s.structs)}`);
    if (s.loss !== "mae") a.push(`--loss ${s.loss}`);
    if (differs(s.we, 1)) a.push(`--w-energy ${Number(s.we)}`);
    if (differs(s.wf, 1)) a.push(`--w-forces ${Number(s.wf)}`);
    if (differs(s.ws, 3)) a.push(`--w-stress ${Number(s.ws)}`);
    if (set(s.lr)) a.push(`--lr ${Number(s.lr)}`);
    if (s.scale !== "energy") a.push(`--scale-from ${s.scale}`);
    if (s.precision !== "tf32") a.push(`--dtype ${s.precision}`);
    if (s.compile) {
      a.push("--compile");
      if (s.cmode !== "default") a.push(`--compile-mode ${s.cmode}`);
    }
    if (differs(s.ema, 0.999)) a.push(`--ema ${Number(s.ema)}`);
    if (differs(s.warmup, 2000)) a.push(`--warmup ${Number(s.warmup)}`);
    if (differs(s.seed, 0)) a.push(`--seed ${Number(s.seed)}`);
    if (differs(s.valevery, 2000)) a.push(`--val-every ${Number(s.valevery)}`);
    if (differs(s.ckptevery, 2000)) a.push(`--ckpt-every ${Number(s.ckptevery)}`);
    if (differs(s.logevery, 50)) a.push(`--log-every ${Number(s.logevery)}`);
    if (set(s.selecton) && s.selecton !== "f_rmse") a.push(`--select-on ${q(s.selecton)}`);
    if (set(s.init)) {
      a.push(`--init ${q(s.init)}`);
      if (differs(s.refit, 0)) a.push(`--refit-e0 ${Number(s.refit)}`);
    }
    if (s.freeze) a.push("--freeze-e0");
    if (s.tdevice !== "auto") a.push(`--device ${s.tdevice}`);
    if (s.resident !== "auto") a.push(`--resident ${s.resident}`);
    a.push(...trackArgs(s), `--out ${q(out)}`);
    const train = command("python -m ilmlip.train", a);
    const steps = [s.job === "shell" ? pack : `[ -f ${q(data)}/meta.json ] || ${pack}`, train];
    if (s.evaluate) steps.push(`python -m ilmlip.evaluate ${q(out)}/best.pt ${q(data)}`);

    const notes = [
      `<code>ilmlip.data</code> reads energies, forces and stress under the usual names (REF_energy, energy, free_energy, …; REF_forces, forces; REF_stress, stress) and packs them into <code>${esc(data)}/</code>.`,
      `Running the same train command again resumes from <code>${esc(out)}/checkpoint.pt</code>, mid-epoch; SIGTERM writes a checkpoint first.`,
      `<code>${esc(out)}/best.pt</code> has the lowest validation ${esc(s.selecton || "f_rmse")}: <code>load_calculator("${esc(out)}/best.pt")</code>.`,
    ];
    if (s.job !== "shell") notes.push("The job packs the data only once, so submitting it again continues the training.");
    const tn = trackNote(s, out), sn = submitNote(s, "train.sh");
    if (tn) notes.push(tn);
    if (sn) notes.push(sn);
    return { lang: "sh", file: "train.sh", notes, cite: ["ilmlip"], code: wrapJob(s, "train", steps.join("\n\n")) + "\n" };
  }

  // the page
  const state = {}, SPEC = {};
  for (const [id, tab] of Object.entries(TABS)) {
    state[id] = {};
    SPEC[id] = {};
    const init = (fs) => fs.forEach((f) => (f.type === "row" ? init(f.fields) : f.key && (SPEC[id][f.key] = f, state[id][f.key] = f.def)));
    init(tab.fields);
  }
  const clamped = (id, s) => Object.fromEntries(Object.entries(s).map(([k, v]) => [k, clamp(SPEC[id][k], v)]));

  function control(f, s) {
    const id = `f-${f.key}`;
    if (f.type === "seg") {
      return `<div class="label">${f.label}</div><div class="seg" role="group" aria-label="${esc(f.label)}">`
        + f.options.map(([v, l]) => `<button type="button" data-key="${f.key}" data-v="${esc(v)}" aria-pressed="${s[f.key] === v}">${esc(l)}</button>`).join("")
        + "</div>";
    }
    if (f.type === "toggle") {
      return `<label class="switch"><input type="checkbox" id="${id}" data-key="${f.key}"${s[f.key] ? " checked" : ""}> ${f.label}</label>`;
    }
    if (f.type === "select") {
      const opts = f.options.map(([g, items]) => `<optgroup label="${esc(g)}">`
        + items.map(([v, l]) => `<option value="${esc(v)}"${s[f.key] === v ? " selected" : ""}>${esc(l)}</option>`).join("") + "</optgroup>").join("");
      return `<label for="${id}">${f.label}</label><select id="${id}" data-key="${f.key}">${opts}</select>`;
    }
    const type = f.type !== "number" ? "text"
      : `number" step="${f.step ?? "any"}"${f.min !== undefined ? ` min="${f.min}"` : ""}${f.max !== undefined ? ` max="${f.max}"` : ""} inputmode="decimal`;
    return `<label for="${id}">${f.label}</label><input type="${type}" id="${id}" data-key="${f.key}" value="${esc(s[f.key])}"`
      + `${f.place ? ` placeholder="${esc(f.place)}"` : ""} spellcheck="false">`;
  }

  function block(f, s) {
    if (f.type === "row") {
      return `<div class="pair" style="grid-template-columns: repeat(${f.fields.length}, minmax(0, 1fr))">`
        + f.fields.map((g) => `<div data-field="${g.key}">${control(g, s)}</div>`).join("") + "</div>";
    }
    return control(f, s);
  }

  const $ = (sel) => document.querySelector(sel);
  let current = "md";
  const flat = [];

  function draw() {
    const tab = TABS[current], s = state[current];
    const form = $(".form");
    let html = "", more = "";
    flat.length = 0;
    tab.fields.forEach((f, i) => {
      if (f.type === "section") {
        html += `<h3>${f.title}</h3>`;
        return;
      }
      flat.push([i, f]);
      const box = `<div class="field" data-i="${i}">${block(f, s)}<div class="help"></div></div>`;
      if (f.more) more += box;
      else html += box;
    });
    form.innerHTML = html + (more ? `<details class="more"><summary>More options</summary>${more}</details>` : "");
    refresh();
  }

  function refresh() {
    const tab = TABS[current], s = state[current];
    for (const [i, f] of flat) {
      const box = $(`.field[data-i="${i}"]`);
      box.hidden = f.show ? !f.show(s) : false;
      const help = typeof f.help === "function" ? f.help(s) : f.help || "";
      box.querySelector(":scope > .help").innerHTML = help;
      box.querySelector(":scope > .help").hidden = !help;
      for (const g of f.type === "row" ? f.fields : []) {
        const cell = box.querySelector(`[data-field="${g.key}"]`);
        if (cell) cell.hidden = g.show ? !g.show(s) : false;
      }
      if (f.type === "seg") {
        for (const b of box.querySelectorAll("button[data-v]")) {
          b.disabled = f.disabled ? f.disabled(s, b.dataset.v) : false;
          if (b.disabled && s[f.key] === b.dataset.v) s[f.key] = f.def;
        }
        for (const b of box.querySelectorAll("button[data-v]")) b.setAttribute("aria-pressed", String(s[f.key] === b.dataset.v));
      }
    }
    const r = tab.build(clamped(current, s));
    const code = $(".out code");
    code.className = `language-${r.lang}`;
    code.textContent = r.code;
    site.highlight(code, r.lang);
    $(".notes").innerHTML = r.notes.map((n) => `<li>${n}</li>`).join("");
    $(".out").dataset.file = r.file;
    cite(r.cite);
  }

  // who to cite: ILMlip, the data of the model, ASE for MD; the (i) shows the BibTeX
  let bib = null, keys = [];
  function cite(k) {
    keys = k;
    const [, , names] = data(current === "md" ? state.md.model : state.finetune.foundation);
    const what = current === "train" ? "the data you trained on"
      : current === "md" && state.md.model === "file" || current === "finetune" && state.finetune.foundation === "file"
        ? "the data of your model" : names;
    $(".cite span").innerHTML = `Publishing with it? Please cite ILMlip${what.startsWith("the ") ? " and " : ", "}${what}${keys.includes("ase") ? ", and ASE" : ""}.`;
    if (!$(".bib").hidden) showBib();
  }
  async function showBib() {
    if (!bib) {
      try {
        const text = await (await fetch("assets/references.bib")).text();
        bib = Object.fromEntries(text.split(/\n(?=@)/).map((e) => [e.match(/^@\w+\{([^,]+),/)?.[1], e.trim()]));
      } catch {
        bib = {};
      }
    }
    const code = $(".bib code");
    code.textContent = keys.map((k) => bib[k]).filter(Boolean).join("\n\n") || "references.bib could not be loaded.";
    if (!$(".bib .copy")) site.copyButton($(".bib pre"));
  }

  function select(id) {
    current = TABS[id] ? id : "md";
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === current)));
    if (location.hash.slice(1) !== current) history.replaceState(null, "", `#${current}`);
    draw();
  }

  function start() {
    $(".tabs").innerHTML = Object.entries(TABS)
      .map(([id, t]) => `<button type="button" role="tab" data-tab="${id}" aria-selected="false">${t.title}</button>`).join("");
    $(".tabs").addEventListener("click", (e) => e.target.dataset.tab && select(e.target.dataset.tab));
    const form = $(".form");
    form.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-v]");
      if (!b || b.disabled) return;
      state[current][b.dataset.key] = b.dataset.v;
      refresh();
    });
    form.addEventListener("input", (e) => {
      const el = e.target;
      if (!el.dataset.key) return;
      state[current][el.dataset.key] = el.type === "checkbox" ? el.checked : el.value;
      refresh();
    });
    form.addEventListener("submit", (e) => e.preventDefault());
    form.addEventListener("change", (e) => {
      const el = e.target, f = SPEC[current][el.dataset.key];
      if (!f || f.type !== "number") return;
      const v = clamp(f, el.value);
      if (v !== el.value) {
        el.value = v;
        state[current][f.key] = v;
        refresh();
      }
    });
    $(".cite .info").addEventListener("click", (e) => {
      const open = $(".bib").hidden;
      $(".bib").hidden = !open;
      e.currentTarget.setAttribute("aria-expanded", String(open));
      if (open) showBib();
    });
    $(".actions").addEventListener("click", async (e) => {
      const act = e.target.dataset.act, text = $(".out code").textContent;
      if (act === "copy") {
        try {
          await navigator.clipboard.writeText(text);
          e.target.textContent = "Copied";
        } catch {
          e.target.textContent = "Select and copy";
        }
        setTimeout(() => (e.target.textContent = "Copy"), 1500);
      } else if (act === "save") {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
        a.download = $(".out").dataset.file;
        a.click();
        URL.revokeObjectURL(a.href);
      }
    });
    window.addEventListener("hashchange", () => select(location.hash.slice(1)));
    select(location.hash.slice(1));
  }

  if (typeof document !== "undefined" && document.querySelector(".builder")) start();
  return { TABS, state, clamped, buildMd, buildFinetune, buildTrain };
})();
if (typeof module !== "undefined") module.exports = builder;
