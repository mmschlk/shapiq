// The GraphSHAP-IQ demo (demos/graphshapiq/index.rst): which atoms of a molecule a graph neural
// network (GNN) sees together when it answers a question about it, explained by the exact Shapley
// interactions of its atoms, of any order, as in GraphSHAP-IQ (Muschalik et al., ICLR 2025).
//
// The players are the molecule's heavy atoms; a coalition of them is what the GNN gets to know:
// the other atoms keep their bonds, but their element is masked (replaced by the average atom of
// the GNN's training molecules). Every task (a question, such as "does it contain an indole?") has
// its own GNN, B-XAIC's GIN of three layers. The explanations are precomputed per molecule and task
// in data.js (by the scripts in ../compute): the k-SII of orders 1 (the Shapley values) to 4 and the
// Möbius interactions of all orders, exact. The demo shows:
//
// - no explanation: the molecule and the GNN's prediction; a click masks an atom, and the GNN
//   itself (in data.js, a few thousand weights) predicts anew, in the page;
// - the interactions: the SI-graph of the paper's figures on the molecule, of the order picked
//   with − and +: every atom tinted by what it adds alone, groups of atoms joined by what they add
//   together, red if they raise the odds of yes, blue if they lower them;
// - the receptive fields: what the GNN's layers let an atom see, the reason why GraphSHAP-IQ is
//   exact with far fewer maskings than all of them.
//
// Pointing at an atom (hover, keyboard focus, or a tap that pins it) shows its interactions or its
// receptive field. Without JavaScript, the static figure in the <noscript> shows; when the data
// does not load or does not fit, the script shows that figure itself.
(() => {
  const root = document.querySelector(".graphshapiq-demo");
  if (!root) return;
  // with scripts on, a <noscript>'s content is its HTML as text
  const fallback = root.querySelector("noscript")?.textContent ?? "";
  const showFallback = () => {
    root.classList.remove("graphshapiq-demo-ready");
    root.classList.add("graphshapiq-demo-failed");
    root.innerHTML = fallback;
  };

  const SVG = "http://www.w3.org/2000/svg";
  const RED = [255, 13, 87];
  const BLUE = [30, 136, 229];
  const TOP = 5; // the strongest interactions listed
  const DRAWN = 14; // the most interactions drawn on the molecule
  const PAD = 1.3; // the room around a molecule, in bond lengths
  const ROOM = 7.4; // the least width of a drawing, in bond lengths: small molecules stay small
  const VIEWS = [
    // the view, its label, and the label on small screens
    ["none", "No explanation", "None"],
    ["interactions", "Interactions", "Interactions"],
    ["fields", "Receptive fields", "Fields"],
  ];

  const loader = document.createElement("script");
  loader.src = `${root.dataset.root}data.js`;
  loader.onload = () => {
    try {
      start(window.SHAPIQ_GRAPHSHAPIQ);
    } catch (error) {
      console.error("The interactive explanation could not start:", error);
      showFallback();
    }
  };
  loader.onerror = showFallback;
  document.head.appendChild(loader);

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const svg = (tag, attributes = {}, parent = null) => {
    const node = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (parent) parent.append(node);
    return node;
  };
  const signed = (value) => {
    const size = Math.abs(value);
    if (size < 0.005) return "0.00";
    return `${value >= 0 ? "+" : "−"}${size.toFixed(2)}`;
  };
  const number = (value) => value.toLocaleString("en-US");
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const percent = (p) => {
    if (p > 0.995) return ">99%";
    if (p < 0.005) return "<1%";
    return `${Math.round(100 * p)}%`;
  };
  const solid = (value) => `rgb(${(value >= 0 ? RED : BLUE).join(", ")})`;
  // a value in a text: red or blue by its sign, in shades that read well as text in either theme
  const sign = (value) => {
    if (Math.abs(value) < 0.005) return "graphshapiq-demo-zero";
    return value >= 0 ? "graphshapiq-demo-up" : "graphshapiq-demo-down";
  };
  const valued = (value) => el("strong", sign(value), signed(value));
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // the atoms of a coalition, written as a bitmask of their indices
  const atomsOf = (mask) => {
    const atoms = [];
    for (let k = 0; mask; k += 1, mask >>>= 1) if (mask & 1) atoms.push(k);
    return atoms;
  };
  const ELEMENTS = {
    C: "carbon", N: "nitrogen", O: "oxygen", S: "sulfur", P: "phosphorus", F: "fluorine",
    Cl: "chlorine", Br: "bromine", I: "iodine", B: "boron",
  };
  const GROUPS = ["", "single atoms", "pairs", "up to triples", "up to four atoms"]; // by order

  function start(data) {
    const has = (molecule, task) => `${molecule}/${task}` in data.explanations;
    const first = data.molecules.find((m) => data.tasks.some((t) => has(m.id, t.id)));
    if (!first) throw new Error("no molecule has an explanation");
    const state = {
      molecule: first.id,
      task: first.task && has(first.id, first.task) ? first.task : data.tasks.find((t) => has(first.id, t.id)).id,
      view: "interactions",
      order: Math.min(2, data.orders), // 1 to data.orders: k-SII; data.orders + 1: Möbius
      layers: data.model.layers, // in the receptive fields view
      focus: null,
      pinned: false,
      masked: new Set(), // the atoms masked without an explanation
    };
    const MOEBIUS = data.orders + 1;

    // -- the frame: molecules, views, tasks, the stage with the molecule, and the panel -----------
    root.replaceChildren();
    const bar = el("div", "graphshapiq-demo-bar");
    const picker = el("div", "graphshapiq-demo-molecules");
    picker.setAttribute("role", "group");
    picker.setAttribute("aria-label", "Molecule");
    const views = el("div", "graphshapiq-demo-switch");
    views.setAttribute("role", "group");
    views.setAttribute("aria-label", "Explanation");
    const tasks = el("div", "graphshapiq-demo-switch");
    tasks.setAttribute("role", "group");
    tasks.setAttribute("aria-label", "Question");
    const switches = el("div", "graphshapiq-demo-switches");
    const info = el("button", "graphshapiq-demo-info", "i");
    info.type = "button";
    info.setAttribute("aria-label", "What is a graph neural network, and what does it see?");
    info.setAttribute("aria-expanded", "false");
    info.setAttribute("aria-controls", "graphshapiq-demo-about");
    const about = el("div", "graphshapiq-demo-about");
    about.id = "graphshapiq-demo-about";
    about.hidden = true;
    about.innerHTML =
      '<p class="graphshapiq-demo-about-title">What does a graph neural network see?</p>' +
      "<p>A molecule as a <strong>graph</strong>: its atoms, linked by its bonds. Here, the GNN " +
      "knows of an atom only its element, and of a bond only that it links two atoms, not whether " +
      "it is single, double or aromatic.</p>" +
      `<p>In each of its ${data.model.layers} <strong>layers</strong>, every atom gathers what ` +
      "its bonded neighbors know. After them, an atom knows the atoms within " +
      `${data.model.layers} bonds, its <strong>receptive field</strong>. The GNN adds up what ` +
      "all atoms know and reads a yes or a no off the sum.</p>" +
      "<p>To explain an answer, the atoms are <strong>players</strong>: masking an atom's " +
      "element tells what it adds on its own, and what groups of atoms add together, their " +
      "<strong>interactions</strong>. Usually, that takes all maskings: " +
      "2<sup><var>n</var></sup> for <var>n</var> atoms. <strong>GraphSHAP-IQ</strong> needs only " +
      "those within receptive fields, as atoms that never share one cannot interact, and still " +
      "computes every interaction exactly.</p>";
    const close = el("button", "graphshapiq-demo-about-close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Close");
    about.prepend(close);
    const setAbout = (open) => {
      about.hidden = !open;
      info.setAttribute("aria-expanded", String(open));
    };
    info.addEventListener("click", () => setAbout(about.hidden));
    about.addEventListener("focusout", (event) => {
      if (!about.contains(event.relatedTarget) && event.relatedTarget !== info) setAbout(false);
    });
    close.addEventListener("click", () => {
      setAbout(false);
      info.focus();
    });
    // the molecules and the "i" in a row, the views and the questions in a row under them
    switches.append(views, tasks);
    bar.append(picker, info, switches, about);

    const body = el("div", "graphshapiq-demo-body");
    const stage = el("div", "graphshapiq-demo-stage");
    const frame = el("div", "graphshapiq-demo-frame");
    const drawing = svg("svg", { class: "graphshapiq-demo-drawing", "aria-hidden": "true" });
    const defs = svg("defs", {}, drawing);
    // the outline of the ground truth: the shape grown by a hair, without the shape
    const outline = svg("filter", { id: "graphshapiq-demo-outline", x: "-20%", y: "-20%", width: "140%", height: "140%" }, defs);
    svg("feMorphology", { in: "SourceAlpha", operator: "dilate", radius: "0.07", result: "grown" }, outline);
    svg("feComposite", { in: "grown", in2: "SourceAlpha", operator: "out", result: "ring" }, outline);
    svg("feFlood", { class: "graphshapiq-demo-truth-color", result: "color" }, outline);
    svg("feComposite", { in: "color", in2: "ring", operator: "in" }, outline);
    const layers = {};
    // the interactions over the bonds, as a highlighter, and under the atoms' labels
    for (const name of ["fields", "truth", "bonds", "knockouts", "edges", "nodes", "labels"])
      layers[name] = svg("g", { class: `graphshapiq-demo-${name}` }, drawing);
    layers.truth.setAttribute("filter", "url(#graphshapiq-demo-outline)");
    const atomsLayer = el("div", "graphshapiq-demo-atoms");
    frame.append(drawing, atomsLayer);
    const caption = el("div", "graphshapiq-demo-caption");
    const captionName = el("p", "graphshapiq-demo-name");
    const captionBlurb = el("p", "graphshapiq-demo-blurb");
    caption.append(captionName, captionBlurb);
    stage.append(frame, caption);

    const panel = el("div", "graphshapiq-demo-panel");
    const title = el("p", "graphshapiq-demo-title");
    const detail = el("p", "graphshapiq-demo-detail");
    // one slot of a fixed height for all views, so that the panel keeps its size
    const slot = el("div", "graphshapiq-demo-slot");
    const stepper = el("div", "graphshapiq-demo-step-group");
    const lower = el("button", "graphshapiq-demo-step", "−");
    const higher = el("button", "graphshapiq-demo-step", "+");
    const stepLabel = el("span", "graphshapiq-demo-step-label");
    lower.type = higher.type = "button";
    stepper.append(lower, stepLabel, higher);
    const list = el("ol", "graphshapiq-demo-list");
    const chart = el("div", "graphshapiq-demo-chart");
    slot.append(stepper, list, chart);
    const legend = el("p", "graphshapiq-demo-legend");
    const meta = el("p", "graphshapiq-demo-meta");
    const live = el("p", "graphshapiq-demo-live");
    live.setAttribute("aria-live", "polite");
    // without an explanation: the GNN's prediction, and what masking atoms does to it
    const prediction = el("div", "graphshapiq-demo-prediction");
    const predictionModel = el("p", "graphshapiq-demo-prediction-model");
    const predictionNumber = el("p", "graphshapiq-demo-prediction-number");
    const predictionValue = el("span", "graphshapiq-demo-prediction-value");
    const predictionVerdict = el("span", "graphshapiq-demo-verdict");
    predictionNumber.append(predictionValue, predictionVerdict);
    const predictionLabel = el("p", "graphshapiq-demo-prediction-label");
    const gauge = el("div", "graphshapiq-demo-gauge");
    gauge.setAttribute("aria-hidden", "true"); // the number says the same
    const gaugeFill = el("span", "graphshapiq-demo-gauge-fill");
    const gaugeWas = el("span", "graphshapiq-demo-gauge-was");
    gauge.append(gaugeFill, gaugeWas);
    const predictionTruth = el("p", "graphshapiq-demo-prediction-truth");
    const predictionMask = el("p", "graphshapiq-demo-prediction-mask");
    const predictionWhy = el("p", "graphshapiq-demo-prediction-why");
    const toView = (view, label) => {
      const button = el("button", "graphshapiq-demo-link", label);
      button.type = "button";
      button.addEventListener("click", () => {
        setView(view);
        viewButtons[VIEWS.findIndex(([v]) => v === view)].focus(); // the link itself goes away
      });
      return button;
    };
    predictionWhy.append("But why? See the ", toView("interactions", "interactions"), " of its atoms.");
    prediction.append(
      predictionModel,
      predictionNumber,
      predictionLabel,
      gauge,
      predictionTruth,
      predictionMask,
      predictionWhy,
    );
    panel.append(title, detail, slot, legend, meta, prediction, live);
    body.append(stage, panel);
    root.append(bar, body);

    const moleculeButtons = data.molecules.map((molecule) => {
      const button = el("button", "graphshapiq-demo-molecule");
      button.type = "button";
      button.setAttribute("aria-label", molecule.name);
      button.title = molecule.name;
      button.append(thumbnail(molecule), el("span", "graphshapiq-demo-molecule-name", molecule.short ?? molecule.name));
      button.addEventListener("click", () =>
        crossFade(() => {
          state.molecule = molecule.id;
          if (molecule.task && has(molecule.id, molecule.task)) state.task = molecule.task;
          state.masked = new Set();
          reset();
        }),
      );
      picker.append(button);
      return button;
    });
    const viewButtons = VIEWS.map(([view, label, short]) => {
      const button = el("button", "graphshapiq-demo-option");
      button.append(el("span", "graphshapiq-demo-long", label), el("span", "graphshapiq-demo-short", short));
      button.setAttribute("aria-label", label);
      button.type = "button";
      button.addEventListener("click", () => setView(view));
      views.append(button);
      return button;
    });
    const taskButtons = data.tasks.map((task) => {
      const button = el("button", "graphshapiq-demo-option", task.label);
      button.type = "button";
      button.title = task.question;
      button.addEventListener("click", () => {
        state.task = task.id;
        reset(true);
      });
      tasks.append(button);
      return button;
    });
    const step = (button, by) =>
      button.addEventListener("click", () => {
        if (button.getAttribute("aria-disabled") === "true") return;
        if (state.view === "fields") state.layers = Math.max(1, Math.min(4, state.layers + by));
        else state.order = Math.max(1, Math.min(MOEBIUS, state.order + by));
        show(null, true);
      });
    step(lower, -1);
    step(higher, 1);

    // -- the molecule and its explanation for the current task -------------------------------------
    let mol = null; // the molecule: its atoms, bonds and rings, as pack.py draws them
    let ex = null; // the explanation: log-odds, Möbius interactions, k-SII, ground truth
    let task = null;
    let atomButtons = [];
    let neighbors = []; // the atoms bonded to every atom
    let distances = []; // the number of bonds between every two atoms
    let roving = 0; // the atom in the tab order; the arrow keys move between atoms
    let started = false; // nothing is announced while the demo builds itself

    const element = (k) => ELEMENTS[mol.atoms[k][0]] ?? mol.atoms[k][0];
    const name = (k) => `${element(k)} ${k + 1}`; // for screen readers
    const group = (atoms) => {
      // the elements of a group of atoms, as a formula: "N O O" becomes "NO₂"
      const counts = {};
      for (const k of atoms) counts[mol.atoms[k][0]] = (counts[mol.atoms[k][0]] ?? 0) + 1;
      const order = Object.keys(counts).sort((a, b) => (a === "C" ? -1 : b === "C" ? 1 : a.localeCompare(b)));
      const sub = (n) => (n > 1 ? String(n).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[d]) : "");
      return order.map((symbol) => symbol + sub(counts[symbol])).join("");
    };
    const inTruth = (atoms) => atoms.every((k) => ex.truth.includes(k));

    function reset(keepFocus = false) {
      // a molecule without an explanation for the task switches to a task that has one
      if (!has(state.molecule, state.task)) state.task = data.tasks.find((t) => has(state.molecule, t.id)).id;
      mol = data.molecules.find((m) => m.id === state.molecule);
      task = data.tasks.find((t) => t.id === state.task);
      ex = data.explanations[`${state.molecule}/${state.task}`];
      prepare(ex);
      prepareGNN(task);
      moleculeButtons.forEach((b, k) => {
        b.setAttribute("aria-pressed", String(data.molecules[k].id === state.molecule));
        b.disabled = !data.tasks.some((t) => has(data.molecules[k].id, t.id));
      });
      taskButtons.forEach((b, k) => {
        b.setAttribute("aria-pressed", String(data.tasks[k].id === state.task));
        b.disabled = !has(state.molecule, data.tasks[k].id);
      });
      if (!keepFocus || !atomButtons.length || atomButtons.length !== mol.atoms.length) {
        drawMolecule();
        state.focus = null;
        state.pinned = false;
        roving = 0;
      }
      if (Math.abs(logOdds(new Set()) - ex.logOdds) > 0.01)
        console.warn(`The GNN in the page predicts other log-odds for ${mol.name} than its explanation.`);
      const formulaText = el("span", "graphshapiq-demo-formula");
      formulaText.append(formula(mol.formula));
      captionName.replaceChildren(mol.name, formulaText);
      legend.replaceChildren(...legendItems(task, ex.truth.length > 0));
      captionBlurb.textContent = mol.blurb ?? "";
      describeAtoms();
      meta.replaceChildren(...metaText());
      reserveRoom();
      setInteractive();
      show(state.pinned ? state.focus : null, true);
    }

    // per explanation: its interactions by order, and the largest value of every order (for the
    // scale of the drawing)
    function prepare(e) {
      if (e.prepared) return;
      e.byOrder = {};
      for (let order = 1; order <= MOEBIUS; order += 1) {
        const terms = (order === MOEBIUS ? e.moebius : e.kSII[order] ?? []).map(([m, value]) => ({ atoms: atomsOf(m >>> 0), value }));
        e.byOrder[order] = { terms, largest: Math.max(1e-9, ...terms.map((t) => Math.abs(t.value))) };
      }
      e.prepared = true;
    }

    // the task's GNN, as pack.py writes it: affine maps (float32, base64) with ReLUs, the first of
    // every layer after each atom gathered its neighbors' features, the last after the atoms were
    // summed up
    const decode = (text) => new Float32Array(Uint8Array.from(atob(text), (c) => c.charCodeAt(0)).buffer);
    function prepareGNN(t) {
      if (t.steps) return;
      t.steps = t.gnn.steps.map((step) => ({ ...step, w: decode(step.w), b: decode(step.b), rows: step.shape[0], columns: step.shape[1] }));
    }

    // the GNN's log-odds for the molecule with the masked atoms' elements masked (replaced by the
    // average atom of the GNN's training molecules), as GraphSHAP-IQ's game values them
    function logOdds(masked) {
      let h = mol.atoms.map(([symbol], k) => {
        if (masked.has(k)) return Float64Array.from(task.baseline);
        const row = new Float64Array(data.symbols.length);
        row[data.symbols.includes(symbol) ? data.symbols.indexOf(symbol) : data.symbols.length - 1] = 1;
        return row;
      });
      for (const step of task.steps) {
        if (step.gather)
          h = h.map((row, k) => {
            const gathered = Float64Array.from(row);
            for (const n of neighbors[k]) for (let c = 0; c < row.length; c += 1) gathered[c] += h[n][c];
            return gathered;
          });
        if (step.sum) {
          const total = new Float64Array(h[0].length);
          for (const row of h) for (let c = 0; c < row.length; c += 1) total[c] += row[c];
          h = [total];
        }
        h = h.map((row) => {
          const out = new Float64Array(step.rows);
          for (let r = 0; r < step.rows; r += 1) {
            let sum = step.b[r];
            for (let c = 0; c < step.columns; c += 1) sum += step.w[r * step.columns + c] * row[c];
            out[r] = step.relu ? Math.max(0, sum) : sum;
          }
          return out;
        });
      }
      return h[0][0];
    }

    const formula = (text) => {
      const span = el("span");
      for (const part of text.match(/[A-Z][a-z]?|\d+|[+-]/g) ?? []) {
        if (/\d/.test(part)) span.append(el("sub", "", part));
        else span.append(part);
      }
      return span;
    };

    // the legend: red and blue, and the outline of the ground truth if there is one
    const legendItems = (t, truth) => {
      const key = (kind, text) => {
        const item = el("span", "graphshapiq-demo-legend-item");
        item.append(el("span", `graphshapiq-demo-key graphshapiq-demo-key-${kind}`), text);
        return item;
      };
      return [key("red", `for “${t.yes}”`), key("blue", "against"), ...(truth ? [key("truth", `the ${t.truthLabel}`)] : [])];
    };

    const SPLITS = { train: "training", valid: "validation", test: "test" };
    const metaText = (m = mol, t = task, e = ex) => {
      const scores = `GIN trained on ${number(t.trained)} molecules of ${t.dataset}, F1 ${t.f1.toFixed(2)} on held-out ones.`;
      const seen = e.training ? ` ${m.name} is in its ${SPLITS[e.training.split]} data, labeled ${e.training.label ? t.yes : t.no}.` : "";
      return [scores + seen];
    };

    // -- drawing ---------------------------------------------------------------------------------
    // the molecule's box: its atoms with some room around, at least ROOM bond lengths wide, square
    function box(m, room = ROOM, pad = PAD) {
      const xs = m.atoms.map((a) => a[1]);
      const ys = m.atoms.map((a) => a[2]);
      const size = Math.max(room, Math.max(...xs) - Math.min(...xs) + 2 * pad, Math.max(...ys) - Math.min(...ys) + 2 * pad);
      const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
      const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
      return { x: cx - size / 2, y: cy - size / 2, size };
    }

    // the bonds of a molecule as lines: double bonds of a ring inside it, other double bonds on
    // the side of the other bonds, or centered at a terminal atom
    function bondLines(m) {
      const at = (k) => [m.atoms[k][1], m.atoms[k][2]];
      const near = m.atoms.map(() => []);
      for (const [i, j] of m.bonds) {
        near[i].push(j);
        near[j].push(i);
      }
      const labeled = (k) => m.atoms[k][0] !== "C" || m.atoms[k][4] !== 0;
      const lines = [];
      for (const [i, j, order] of m.bonds) {
        const p = at(i);
        const q = at(j);
        const length = Math.hypot(q[0] - p[0], q[1] - p[1]);
        const u = [(q[0] - p[0]) / length, (q[1] - p[1]) / length];
        const normal = [-u[1], u[0]];
        const shift = (point, by, along = 0) => [point[0] + normal[0] * by + u[0] * along, point[1] + normal[1] * by + u[1] * along];
        if (order === 1) {
          lines.push([p, q, i, j]);
          continue;
        }
        if (order === 3) {
          for (const by of [-0.16, 0, 0.16]) lines.push([shift(p, by), shift(q, by), i, j]);
          continue;
        }
        const ring = m.rings.filter((r) => r.includes(i) && r.includes(j)).sort((a, b) => a.length - b.length)[0];
        let side = 0;
        if (ring) {
          const center = ring.map(at).reduce(([a, b], [c, d]) => [a + c / ring.length, b + d / ring.length], [0, 0]);
          side = Math.sign((center[0] - p[0]) * normal[0] + (center[1] - p[1]) * normal[1]);
        } else if (!((near[i].length === 1 && labeled(i)) || (near[j].length === 1 && labeled(j)))) {
          const others = [...near[i].filter((k) => k !== j), ...near[j].filter((k) => k !== i)];
          side = Math.sign(others.reduce((s, k) => s + Math.sign((at(k)[0] - p[0]) * normal[0] + (at(k)[1] - p[1]) * normal[1]), 0)) || 1;
        }
        if (side === 0) {
          lines.push([shift(p, -0.09), shift(q, -0.09), i, j], [shift(p, 0.09), shift(q, 0.09), i, j]);
        } else {
          const inset = (k) => (labeled(k) ? 0 : 0.16);
          lines.push([p, q, i, j], [shift(p, 0.19 * side, inset(i)), shift(q, 0.19 * side, -inset(j)), i, j]);
        }
      }
      return lines;
    }

    // a small drawing of a molecule, for its button: its bonds and its heteroatoms as dots
    function thumbnail(m) {
      const node = svg("svg", { class: "graphshapiq-demo-thumbnail", "aria-hidden": "true" });
      const b = box(m, 0, 0.5);
      node.setAttribute("viewBox", `${b.x} ${b.y} ${b.size} ${b.size}`);
      for (const [p, q] of bondLines(m)) svg("line", { x1: p[0], y1: p[1], x2: q[0], y2: q[1] }, node);
      m.atoms.forEach(([symbol, x, y]) => {
        if (symbol !== "C") svg("circle", { cx: x, cy: y, r: 0.32 }, node);
      });
      return node;
    }

    // a molecule's atom as a label: its element, its hydrogens on the side away from its bonds,
    // its charge
    function label(k, parent) {
      const [symbol, x, y, hydrogens, charge] = mol.atoms[k];
      const g = svg("g", { class: "graphshapiq-demo-label" }, parent);
      const text = svg("text", { x, y }, g);
      text.textContent = symbol;
      const width = 0.34 * symbol.length + 0.04;
      if (hydrogens) {
        // the side with the fewest bonds: right, left, below or above
        const dx = neighbors[k].reduce((s, n) => s + mol.atoms[n][1] - x, 0);
        const dy = neighbors[k].reduce((s, n) => s + mol.atoms[n][2] - y, 0);
        const h = svg("text", {}, g);
        h.textContent = "H";
        if (hydrogens > 1) {
          const count = svg("tspan", { class: "graphshapiq-demo-sub", dy: 0.16 }, h);
          count.textContent = hydrogens;
        }
        // above or below an atom between bonds left and right, else beside it (left of a bond
        // to the right)
        const along = neighbors[k].length > 1 && Math.abs(dx) < 0.35 && Math.abs(dy) > 0.5;
        if (along) {
          h.setAttribute("x", x + (hydrogens > 1 ? 0.1 : 0));
          h.setAttribute("y", y + (dy > 0 ? -0.62 : 0.62));
        } else if (dx > 0.05) {
          h.setAttribute("x", x - width / 2 - 0.02);
          h.setAttribute("y", y);
          h.style.textAnchor = "end";
        } else {
          h.setAttribute("x", x + width / 2 + 0.02);
          h.setAttribute("y", y);
          h.style.textAnchor = "start";
        }
      }
      if (charge) {
        const c = svg("text", { x: x + width / 2 + 0.08, y: y - 0.3, class: "graphshapiq-demo-charge" }, g);
        c.textContent = charge > 0 ? "+" : "−";
      }
      return g;
    }

    function drawMolecule() {
      const b = box(mol);
      drawing.setAttribute("viewBox", `${b.x} ${b.y} ${b.size} ${b.size}`);
      for (const layer of Object.values(layers)) layer.replaceChildren();
      neighbors = mol.atoms.map(() => []);
      for (const [i, j] of mol.bonds) {
        neighbors[i].push(j);
        neighbors[j].push(i);
      }
      // the number of bonds between every two atoms, by a breadth-first search from each
      distances = mol.atoms.map((_, a) => {
        const d = mol.atoms.map(() => Infinity);
        d[a] = 0;
        const queue = [a];
        while (queue.length) {
          const k = queue.shift();
          for (const n of neighbors[k]) {
            if (d[n] < Infinity) continue;
            d[n] = d[k] + 1;
            queue.push(n);
          }
        }
        return d;
      });
      for (const [p, q, i, j] of bondLines(mol))
        svg("line", { x1: p[0], y1: p[1], x2: q[0], y2: q[1], "data-i": i, "data-j": j }, layers.bonds);
      mol.atoms.forEach(([symbol, x, y, , charge], k) => {
        const labeled = symbol !== "C" || charge !== 0;
        if (labeled) svg("circle", { cx: x, cy: y, r: 0.36, "data-k": k }, layers.knockouts);
        svg("circle", { cx: x, cy: y, r: 0, "data-k": k }, layers.nodes);
        if (labeled) label(k, layers.labels).dataset.k = k;
      });
      // the atoms as buttons over the drawing
      atomsLayer.replaceChildren();
      atomButtons = mol.atoms.map(([, x, y], k) => {
        const button = el("button", "graphshapiq-demo-atom");
        button.type = "button";
        button.style.left = `${(100 * (x - b.x)) / b.size}%`;
        button.style.top = `${(100 * (y - b.y)) / b.size}%`;
        button.style.width = button.style.height = `${(100 * 0.78) / b.size}%`;
        button.addEventListener("pointerenter", (event) => {
          if (event.pointerType !== "touch" && !state.pinned && state.view !== "none") show(k);
        });
        button.addEventListener("focus", () => {
          setRoving(k);
          if (!state.pinned && state.view !== "none") show(k, true);
        });
        button.addEventListener("click", () => {
          if (state.view === "none") {
            toggleMask(k);
            return;
          }
          const again = state.pinned && state.focus === k;
          state.pinned = !again;
          show(again ? null : k, true);
        });
        atomsLayer.append(button);
        return button;
      });
      describeAtoms();
    }

    function describeAtoms() {
      atomButtons.forEach((button, k) => {
        const notes = [];
        if (ex.truth.includes(k)) notes.push(`part of the ${task.truthLabel ?? "pattern"}`);
        if (state.view === "none") notes.push(state.masked.has(k) ? "masked; click to unmask" : "click to mask");
        button.setAttribute("aria-label", `${name(k)}${notes.length ? `, ${notes.join(", ")}` : ""}`);
        button.setAttribute("aria-pressed", state.view === "none" ? String(state.masked.has(k)) : "false");
      });
    }

    // a group of atoms as in the SI-graph: a disc at every atom, joined to their center (to each
    // other for two atoms) by bands that narrow in the middle; all of one color, as wide and as
    // opaque as the interaction is strong (after shapiq's si_graph_plot)
    function hyperedge(atoms, value, strength, parent, scale = 1) {
      const points = atoms.map((k) => [mol.atoms[k][1], mol.atoms[k][2]]);
      // the half width of the bands, narrower for large groups, whose bands overlap
      const r = scale * (0.09 + 0.25 * strength) * (atoms.length > 3 ? 0.7 : 1);
      const g = svg("g", { fill: solid(value), opacity: (0.22 + 0.58 * strength).toFixed(2) }, parent);
      const center = atoms.length === 2 ? points[1] : points.reduce(([a, b], [c, d]) => [a + c / points.length, b + d / points.length], [0, 0]);
      const band = (p) => {
        const d = [center[0] - p[0], center[1] - p[1]];
        const length = Math.hypot(...d) || 1;
        const n = [-d[1] / length, d[0] / length];
        const middle = [p[0] + d[0] / 2, p[1] + d[1] / 2];
        const at = (q, by) => `${(q[0] + n[0] * by).toFixed(3)} ${(q[1] + n[1] * by).toFixed(3)}`;
        return `M ${at(p, -r)} Q ${middle[0].toFixed(3)} ${middle[1].toFixed(3)} ${at(center, -r)} L ${at(center, r)} Q ${middle[0].toFixed(3)} ${middle[1].toFixed(3)} ${at(p, r)} Z`;
      };
      const d = (atoms.length === 2 ? [points[0]] : points).map(band).join(" ");
      svg("path", { d }, g);
      for (const p of atoms.length === 2 ? points : [...points, center])
        svg("circle", { cx: p[0], cy: p[1], r: r * 1.15 }, g);
      return g;
    }

    // the outline of a group of atoms (the ground truth): discs at its atoms, bands along its bonds
    function blob(atoms, parent, radius = 0.5) {
      const inside = new Set(atoms);
      for (const k of atoms) svg("circle", { cx: mol.atoms[k][1], cy: mol.atoms[k][2], r: radius }, parent);
      for (const [i, j] of mol.bonds)
        if (inside.has(i) && inside.has(j))
          svg("line", { x1: mol.atoms[i][1], y1: mol.atoms[i][2], x2: mol.atoms[j][1], y2: mol.atoms[j][2], "stroke-width": 2 * radius * 0.9 }, parent);
    }

    // every atom's disc: tinted by a value (its main effect, or its interaction with the atom
    // pointed at), as large and as opaque as the value is strong
    function tintAtoms(values, largest) {
      for (const disc of layers.nodes.children) {
        const k = Number(disc.dataset.k);
        const value = values ? values[k] : 0;
        const strength = largest > 0 ? Math.min(Math.abs(value) / largest, 1) : 0;
        disc.setAttribute("r", strength > 0.02 ? (0.16 + 0.26 * Math.sqrt(strength)).toFixed(3) : 0);
        // an opaque tint, so that a disc keeps its color over the interactions it is in
        disc.style.fill = `color-mix(in srgb, ${solid(value)} ${Math.round(100 * (0.3 + 0.6 * strength))}%, var(--shapiq-surface-raised))`;
        disc.classList.remove("graphshapiq-demo-masked");
      }
    }

    // -- the views ---------------------------------------------------------------------------------
    function setView(view) {
      state.view = view;
      state.pinned = false;
      viewButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(VIEWS[k][0] === view)));
      for (const [v] of VIEWS) root.classList.toggle(`graphshapiq-demo-${v}`, v === view);
      setInteractive();
      describeAtoms();
      show(null, true);
    }

    function setInteractive() {
      atomButtons.forEach((button, k) => {
        button.tabIndex = k === roving ? 0 : -1;
      });
    }

    function setRoving(k) {
      if (atomButtons[roving]) atomButtons[roving].tabIndex = -1;
      roving = k;
      atomButtons[k].tabIndex = 0;
    }

    // screen readers hear what changed when it was asked for (a click, the keyboard, a switch),
    // not on every hover
    function show(atom, announce = false) {
      state.focus = atom;
      atomButtons.forEach((b, k) => b.classList.toggle("graphshapiq-demo-focus", k === atom));
      let message;
      if (state.view === "none") message = showNone();
      else if (state.view === "fields") message = showFields(atom ?? defaultAtom());
      else if (atom !== null) message = showAtom(atom);
      else message = showInteractions();
      if (announce && started) live.textContent = message;
    }

    // the atom whose receptive field shows at rest: the first of the ground truth, or the most
    // central atom
    function defaultAtom() {
      if (ex.truth.length) return ex.truth[0];
      const eccentricity = distances.map((d) => Math.max(...d));
      return eccentricity.indexOf(Math.min(...eccentricity));
    }

    function clearDrawing({ truth = false } = {}) {
      highlight(null);
      layers.fields.replaceChildren();
      layers.edges.replaceChildren();
      layers.truth.replaceChildren();
      if (truth && ex.truth.length) blob(ex.truth, layers.truth, 0.5);
      for (const node of [...layers.bonds.children, ...layers.labels.children, ...layers.knockouts.children])
        node.classList.remove("graphshapiq-demo-far");
    }

    // no explanation: the molecule and the GNN's prediction; a click masks or unmasks an atom
    function showNone() {
      clearDrawing();
      tintAtoms(null, 0);
      for (const disc of layers.nodes.children) {
        const masked = state.masked.has(Number(disc.dataset.k));
        disc.classList.toggle("graphshapiq-demo-masked", masked);
        if (masked) disc.setAttribute("r", 0.42);
      }
      for (const g of layers.labels.children) g.classList.toggle("graphshapiq-demo-far", state.masked.has(Number(g.dataset.k)));
      const full = sigmoid(ex.logOdds);
      const now = state.masked.size ? sigmoid(logOdds(state.masked)) : full;
      predictionModel.textContent = `The GNN for “${task.question}” predicts`;
      countTo(now);
      predictionVerdict.textContent = now >= 0.5 ? task.yes : task.no;
      predictionVerdict.dataset.verdict = now >= 0.5 ? "yes" : "no";
      predictionLabel.textContent = state.masked.size
        ? `with ${state.masked.size} ${state.masked.size === 1 ? "atom" : "atoms"} masked (${percent(full)} with none)`
        : task.probability;
      gaugeFill.style.width = `${(100 * now).toFixed(1)}%`;
      gaugeWas.style.left = `${(100 * full).toFixed(1)}%`;
      gaugeWas.hidden = !state.masked.size;
      predictionTruth.replaceChildren(truthText());
      predictionMask.replaceChildren(
        state.masked.size
          ? "A masked atom keeps its bonds, but the GNN sees an average atom instead of its element. "
          : "Click atoms to mask them, and the GNN predicts anew, here in your browser. ",
      );
      if (state.masked.size) {
        const again = el("button", "graphshapiq-demo-link", "Unmask all");
        again.type = "button";
        again.addEventListener("click", () => {
          state.masked = new Set();
          describeAtoms();
          show(null, true);
        });
        predictionMask.append(again, ".");
      }
      const masked = state.masked.size ? ` With ${state.masked.size} atoms masked: ${percent(now)}.` : "";
      return `No explanation. The GNN predicts ${percent(full)} ${task.yes}.${masked}`;
    }

    // the probability counts to its new value, as the gauge moves
    let shown = null;
    let counting = 0;
    function countTo(p) {
      cancelAnimationFrame(counting);
      const from = shown ?? p;
      shown = p;
      if (reducedMotion() || Math.abs(from - p) < 0.005) {
        predictionValue.textContent = percent(p);
        return;
      }
      const start = performance.now();
      const frame = (time) => {
        const t = Math.min(1, (time - start) / 400);
        const eased = 1 - (1 - t) ** 3;
        predictionValue.textContent = percent(from + (p - from) * eased);
        if (t < 1) counting = requestAnimationFrame(frame);
      };
      counting = requestAnimationFrame(frame);
    }

    const truthText = () => {
      const yes = ex.truth.length > 0;
      const node = el("span");
      node.append(el("span", `graphshapiq-demo-truth-mark graphshapiq-demo-truth-${yes ? "yes" : "no"}`, yes ? "✓" : "✗"));
      node.append(yes ? task.truthYes : task.truthNo);
      return node;
    };

    function toggleMask(k) {
      if (state.masked.has(k)) state.masked.delete(k);
      else state.masked.add(k);
      describeAtoms();
      show(null, true);
    }

    const orderName = (order) => {
      if (order === 1) return "Order 1 · Shapley values";
      if (order === MOEBIUS) return "All orders · Möbius";
      return `Order ${order} · ${order}-SII`;
    };

    // the interactions of the current order: every atom tinted by its own value, the strongest
    // groups of atoms joined
    function showInteractions() {
      const { terms, largest } = ex.byOrder[state.order];
      clearDrawing({ truth: true });
      const values = mol.atoms.map(() => 0);
      for (const t of terms) if (t.atoms.length === 1) values[t.atoms[0]] = t.value;
      tintAtoms(values, largest);
      const groups = terms.filter((t) => t.atoms.length > 1);
      const drawn = groups.slice(0, state.order === MOEBIUS ? 8 : DRAWN).filter((t) => Math.abs(t.value) >= 0.06 * largest);
      for (const t of [...drawn].reverse()) hyperedge(t.atoms, t.value, Math.abs(t.value) / largest, layers.edges);
      title.textContent = orderTitle(state.order);
      detail.replaceChildren(...orderDetail(state.order));
      setStepper(state.order, 1, MOEBIUS, orderName(state.order), "Lower order", "Higher order");
      const listed = (state.order === 1 ? terms : groups).slice(0, TOP);
      fillList(listed, largest);
      chart.hidden = true;
      legend.style.visibility = "";
      const [best] = listed;
      return best
        ? `${orderName(state.order)}. Strongest: ${best.atoms.map(name).join(", ")}, ${signed(best.value)}.`
        : `${orderName(state.order)}.`;
    }

    const orderTitle = (order) => {
      if (order === 1) return "What every atom adds: its Shapley value";
      return `Which atoms act together: ${order === MOEBIUS ? "groups of any size" : GROUPS[order] ?? `up to ${order} atoms`}`;
    };

    function orderDetail(order) {
      const sum = ["With the masked molecule's ", valued(ex.empty)];
      if (order === MOEBIUS)
        return [
          "GraphSHAP-IQ's exact Möbius interactions: what every group adds beyond its subgroups. ",
          ...sum,
          `, all ${number(ex.moebiusCount)} add up to `,
          valued(ex.logOdds),
          ", large ones offsetting each other.",
        ];
      const what =
        order === 1
          ? `What every atom adds to the log-odds of “${task.yes}”, sharing fairly what it adds with others. `
          : `The ${order}-SII: what every atom and every group of up to ${order} atoms adds, sharing larger groups. `;
      return [what, ...sum, ", they add up to the GNN's ", valued(ex.logOdds), "."];
    }

    // the interactions of an atom pointed at: the groups it is in
    function showAtom(atom) {
      const { terms, largest } = ex.byOrder[state.order];
      clearDrawing({ truth: true });
      const own = terms.find((t) => t.atoms.length === 1 && t.atoms[0] === atom);
      const groups = terms.filter((t) => t.atoms.length > 1 && t.atoms.includes(atom));
      // the other atoms tinted by the groups they share with it
      const shared = mol.atoms.map(() => 0);
      for (const t of groups) for (const k of t.atoms) if (k !== atom) shared[k] += t.value / (t.atoms.length - 1);
      shared[atom] = own?.value ?? 0;
      tintAtoms(shared, Math.max(...shared.map(Math.abs), 1e-9));
      for (const t of groups.slice(0, DRAWN).reverse()) hyperedge(t.atoms, t.value, Math.abs(t.value) / largest, layers.edges);
      title.textContent = `Interactions of this ${element(atom)}`;
      detail.replaceChildren(
        state.order === MOEBIUS ? "On its own (its Möbius interaction), it adds " : state.order === 1 ? "Its Shapley value: it adds " : "On its own, it adds ",
        valued(own?.value ?? 0),
        state.order === 1 ? " to the log-odds." : `. ${groups.length ? "With other atoms:" : "It acts with no other atom at this order."}`,
      );
      setStepper(state.order, 1, MOEBIUS, orderName(state.order), "Lower order", "Higher order");
      fillList((state.order === 1 ? terms : groups).slice(0, TOP), largest, atom);
      chart.hidden = true;
      legend.style.visibility = "";
      const [best] = groups;
      return `${name(atom)}: on its own ${signed(own?.value ?? 0)}${best ? `; most with ${best.atoms.filter((k) => k !== atom).map(name).join(", ")}, ${signed(best.value)}` : ""}.`;
    }

    function fillList(terms, largest, atom = null) {
      list.replaceChildren();
      list.hidden = false;
      for (const t of terms) {
        const item = el("li");
        const thumb = groupThumbnail(t.atoms, t.value, atom);
        const label = el("span", "graphshapiq-demo-group", t.atoms.length === 1 ? element(t.atoms[0]) : group(t.atoms));
        if (inTruth(t.atoms)) {
          const mark = el("span", "graphshapiq-demo-in-truth");
          mark.title = `In the ${task.truthLabel}`;
          mark.setAttribute("aria-label", `in the ${task.truthLabel}`);
          label.append(mark);
        }
        const valueBar = el("span", "graphshapiq-demo-bar-value");
        valueBar.style.setProperty("--graphshapiq-width", `${(100 * Math.abs(t.value)) / largest}%`);
        valueBar.style.setProperty("--graphshapiq-color", solid(t.value));
        item.append(thumb, label, valueBar, el("span", "graphshapiq-demo-number", signed(t.value)));
        item.addEventListener("pointerenter", () => highlight(t, largest));
        item.addEventListener("pointerleave", () => highlight(null));
        list.append(item);
      }
    }

    // a group of atoms on the stage, while its row of the list is pointed at
    let highlighted = null;
    function highlight(t, largest) {
      highlighted?.remove();
      highlighted = null;
      layers.edges.classList.toggle("graphshapiq-demo-dimmed", t !== null);
      if (t && t.atoms.length > 1) {
        highlighted = hyperedge(t.atoms, t.value, Math.max(0.6, Math.abs(t.value) / largest), layers.edges);
        highlighted.classList.add("graphshapiq-demo-highlight");
      }
      atomButtons.forEach((b, k) => b.classList.toggle("graphshapiq-demo-marked", t !== null && t.atoms.includes(k)));
    }

    // a group of atoms in the list: the molecule, small, with the group on it
    function groupThumbnail(atoms, value, atom) {
      const node = svg("svg", { class: "graphshapiq-demo-group-thumb", "aria-hidden": "true" });
      const b = box(mol, 0, 0.7);
      node.setAttribute("viewBox", `${b.x} ${b.y} ${b.size} ${b.size}`);
      for (const [p, q] of bondLines(mol)) svg("line", { x1: p[0], y1: p[1], x2: q[0], y2: q[1] }, node);
      const scale = Math.max(1, b.size / 5);
      if (atoms.length > 1) hyperedge(atoms, value, 1, node, scale);
      for (const k of atoms)
        svg("circle", { cx: mol.atoms[k][1], cy: mol.atoms[k][2], r: 0.3 * scale, fill: solid(value), class: k === atom ? "graphshapiq-demo-thumb-focus" : "" }, node);
      return node;
    }

    function setStepper(value, min, max, text, lowerLabel, higherLabel) {
      stepper.hidden = false;
      stepLabel.textContent = text;
      lower.setAttribute("aria-label", lowerLabel);
      higher.setAttribute("aria-label", higherLabel);
      lower.setAttribute("aria-disabled", String(value <= min));
      higher.setAttribute("aria-disabled", String(value >= max));
    }

    // the receptive field of an atom: the atoms within as many bonds as the GNN has layers, ring
    // by ring; and how many maskings GraphSHAP-IQ evaluates for GNNs of 1 to 4 layers
    function showFields(atom) {
      clearDrawing();
      tintAtoms(null, 0);
      const d = distances[atom];
      const reach = state.layers;
      for (let hop = reach; hop >= 0; hop -= 1) {
        const ring = svg("g", { class: `graphshapiq-demo-hop graphshapiq-demo-hop-${hop}` }, layers.fields);
        ring.style.setProperty("--graphshapiq-hop", hop);
        blob(mol.atoms.map((_, k) => k).filter((k) => d[k] <= hop), ring, 0.42 + 0.06 * (reach - hop));
      }
      for (const line of layers.bonds.children)
        line.classList.toggle("graphshapiq-demo-far", Math.max(d[line.dataset.i], d[line.dataset.j]) > reach);
      for (const node of [...layers.labels.children, ...layers.knockouts.children])
        node.classList.toggle("graphshapiq-demo-far", d[node.dataset.k] > reach);
      const field = d.filter((x) => x <= reach).length;
      const n = mol.atoms.length;
      title.textContent = `The receptive field of this ${element(atom)}`;
      // at rest, the atom whose field shows is outlined as if pointed at
      atomButtons.forEach((b, k) => b.classList.toggle("graphshapiq-demo-focus", k === atom));
      detail.replaceChildren(fieldDetail(atom, reach));
      setStepper(reach, 1, 4, `${reach} ${reach === 1 ? "layer" : "layers"}${reach === data.model.layers ? " (this GNN)" : ""}`, "Fewer layers", "More layers");
      list.hidden = true;
      legend.style.visibility = "hidden"; // (it keeps its room)
      drawChart(reach);
      return `Receptive field of ${name(atom)} after ${reach} layers: ${field} of ${n} atoms. GraphSHAP-IQ evaluates ${number(mol.budgets[reach])} of ${number(2 ** n)} maskings.`;
    }

    const fieldDetail = (atom, reach) => {
      const field = distances[atom].filter((x) => x <= reach).length;
      const bonds = `${reach} ${reach === 1 ? "bond" : "bonds"}`;
      return (
        `After ${reach} ${reach === 1 ? "layer" : "layers"}, what the GNN knows of this atom depends only on the ` +
        `${field} of ${mol.atoms.length} atoms within ${bonds}. Atoms that never share such a field cannot interact.`
      );
    };

    // the title and the description keep the room of their longest texts for the molecule, so
    // that changing the order, the view, or the atom pointed at moves nothing
    function reserveRoom() {
      const longest = mol.atoms.map((_, k) => k).sort((a, b) => element(b).length - element(a).length)[0];
      const titles = [...Array(MOEBIUS).keys()].map((k) => [orderTitle(k + 1)]);
      titles.push([`The receptive field of this ${element(longest)}`], [`Interactions of this ${element(longest)}`]);
      reserve(title, titles);
      const details = [...Array(MOEBIUS).keys()].map((k) => orderDetail(k + 1));
      const widest = distances.map((d) => d.filter((x) => x <= 4).length).indexOf(Math.max(...distances.map((d) => d.filter((x) => x <= 4).length)));
      for (let reach = 1; reach <= 4; reach += 1) details.push([fieldDetail(widest, reach)]);
      reserve(detail, details);
    }

    // the molecule's description and the model's line keep the room of the longest of all
    // molecules and questions, so that switching between them moves nothing
    function reserveAll() {
      reserve(captionBlurb, data.molecules.map((m) => [m.blurb ?? ""]));
      const lines = [];
      for (const m of data.molecules)
        for (const t of data.tasks) if (has(m.id, t.id)) lines.push(metaText(m, t, data.explanations[`${m.id}/${t.id}`]));
      reserve(meta, lines);
      reserve(legend, data.tasks.map((t) => legendItems(t, true)));
    }

    // the least height of a node: that of the tallest of the contents, in a hidden copy of it
    function reserve(node, contents) {
      const width = node.getBoundingClientRect().width;
      if (!width) return; // not laid out: keep what it has
      const probe = node.cloneNode(false);
      Object.assign(probe.style, { position: "absolute", visibility: "hidden", width: `${width}px`, minHeight: "0" });
      node.after(probe);
      let tallest = 0;
      for (const content of contents) {
        probe.replaceChildren(...content);
        tallest = Math.max(tallest, probe.getBoundingClientRect().height);
      }
      probe.remove();
      node.style.minHeight = `${Math.ceil(tallest)}px`;
    }

    // the maskings GraphSHAP-IQ evaluates for GNNs of 1 to 4 layers, against all maskings, on a
    // logarithmic scale
    function drawChart(reach) {
      chart.hidden = false;
      chart.replaceChildren();
      const n = mol.atoms.length;
      const all = 2 ** n;
      const top = Math.log10(all);
      const row = (labelText, count, className) => {
        const r = el("div", `graphshapiq-demo-chart-row ${className}`);
        const fill = el("span", "graphshapiq-demo-chart-fill");
        fill.style.width = `${Math.max(1.5, (100 * Math.log10(Math.max(count, 1))) / top)}%`;
        const track = el("span", "graphshapiq-demo-chart-track");
        track.append(fill);
        r.append(el("span", "graphshapiq-demo-chart-label", labelText), track, el("span", "graphshapiq-demo-chart-count", number(count)));
        return r;
      };
      const head = el("p", "graphshapiq-demo-chart-head", "Maskings the GNN is evaluated on (log scale)");
      chart.append(head);
      for (let layer = 1; layer <= 4; layer += 1) {
        const className = layer === reach ? "graphshapiq-demo-chart-now" : "";
        chart.append(row(`${layer} ${layer === 1 ? "layer" : "layers"}`, mol.budgets[layer], className));
      }
      chart.append(row(`all: 2${superscript(n)}`, all, "graphshapiq-demo-chart-all"));
      const ratio = all / mol.budgets[reach];
      const depth = `${reach} ${reach === 1 ? "layer" : "layers"}`;
      chart.append(
        el(
          "p",
          "graphshapiq-demo-chart-note",
          mol.budgets[reach] < all
            ? `GraphSHAP-IQ: ${ratio < 10 ? ratio.toFixed(1) : number(Math.round(ratio))} times fewer maskings for ${depth}, and exact.`
            : `For ${depth}, every receptive field holds the whole molecule: GraphSHAP-IQ needs all maskings.`,
        ),
      );
    }
    const superscript = (n) => String(n).replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);

    // a new molecule fades in over the old one, which fades out
    function crossFade(change) {
      for (const ghost of stage.querySelectorAll(".graphshapiq-demo-ghost")) ghost.remove();
      if (reducedMotion()) {
        change();
        return;
      }
      const ghost = drawing.cloneNode(true);
      ghost.classList.add("graphshapiq-demo-ghost");
      ghost.querySelector("filter")?.removeAttribute("id"); // the original keeps its filter
      frame.append(ghost);
      change();
      drawing.classList.add("graphshapiq-demo-fresh");
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          drawing.classList.remove("graphshapiq-demo-fresh");
          ghost.classList.add("graphshapiq-demo-gone");
          setTimeout(() => ghost.remove(), 500);
        }),
      );
    }

    // the arrow keys move to the nearest atom in their direction
    atomsLayer.addEventListener("keydown", (event) => {
      const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
      const k = atomButtons.indexOf(event.target);
      if (k < 0) return;
      if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        atomButtons[event.key === "Home" ? 0 : atomButtons.length - 1].focus();
        return;
      }
      if (!(event.key in directions)) return;
      event.preventDefault();
      const [dx, dy] = directions[event.key];
      const [, x, y] = mol.atoms[k];
      let best = -1;
      let bestScore = Infinity;
      mol.atoms.forEach(([, x2, y2], j) => {
        const along = (x2 - x) * dx + (y2 - y) * dy;
        if (j === k || along <= 0.2) return;
        const across = Math.abs((x2 - x) * dy - (y2 - y) * dx);
        const score = along + 2 * across;
        if (score < bestScore) [best, bestScore] = [j, score];
      });
      if (best >= 0) atomButtons[best].focus();
    });

    stage.addEventListener("pointerleave", () => {
      if (!state.pinned && state.focus !== null) show(null);
    });
    stage.addEventListener("focusout", (event) => {
      if (!state.pinned && !stage.contains(event.relatedTarget) && state.focus !== null) show(null);
    });
    document.addEventListener("pointerdown", (event) => {
      if (!about.hidden && !about.contains(event.target) && !info.contains(event.target)) setAbout(false);
      if (state.pinned && !stage.contains(event.target) && !slot.contains(event.target)) {
        state.pinned = false;
        show(null);
      }
    });
    root.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !about.hidden) {
        setAbout(false);
        info.focus();
      } else if (event.key === "Escape" && state.focus !== null) {
        state.pinned = false;
        show(null);
      }
    });

    // a new width lays the texts out again (in the next frame, outside the observer's callback)
    let width = -1;
    new ResizeObserver(([entry]) => {
      const now = Math.round(entry.contentRect.width);
      if (now === width) return;
      width = now;
      requestAnimationFrame(() => {
        reserveAll();
        reserveRoom();
      });
    }).observe(root);

    root.classList.add("graphshapiq-demo-ready");
    reset();
    setView(state.view);
    started = true;
  }
})();
