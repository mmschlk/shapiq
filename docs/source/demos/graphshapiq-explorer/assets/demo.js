// The GraphSHAP-IQ explorer (demos/graphshapiq-explorer/index.rst): eight molecules, three GNNs
// (one per question), and every molecule explained for every question, exactly, with GraphSHAP-IQ.
//
// Three views: the prediction (click atoms to hide them; the GNN runs in the page), the atoms alone
// (their Shapley values, a ring around each atom), and the atoms together (k-SII of orders 2 to 4
// and the Möbius interactions: bands between two atoms, shapes around more). The explanations are
// precomputed in data.js (../../graphshapiq-common/compute); the shared library
// (../../graphshapiq-common/assets/demo.js, `ShapiqGraph`) draws and animates the molecule.
(() => {
  const G = window.ShapiqGraph;
  if (!G) return console.error("The GraphSHAP-IQ explorer needs ../graphshapiq-common/demo.js first.");
  const { h, svg, sig, bits, signed, percent, value, legend, formula } = G;
  const VIEWS = [["predict", "Prediction"], ["alone", "Atoms alone"], ["together", "Atoms together"]];
  const ORDERS = { 2: ["Pairs", "2-SII"], 3: ["Groups of 3", "3-SII"], 4: ["Groups of 4", "4-SII"], M: ["All orders", "Möbius"] };

  G.mount("graphshapiq-explorer-demo", "SHAPIQ_GRAPHSHAPIQ_EXPLORER", (root, data) => {
    const taskById = (id) => data.tasks.find((t) => t.id === id);
    const first = data.molecules[0];
    const state = { mol: first, task: taskById(first.task), view: "together", order: 2, masked: new Set(), focus: null, pinned: false, row: null, hinted: false };
    const ex = () => data.explanations[`${state.mol.id}/${state.task.id}`];
    const rowsOf = (e, order) => (order === "M" ? e.moebius : e.kSII[String(order)] || []);
    const singles = (e, order) => { const o = {}; for (const [mask, v] of rowsOf(e, order)) { const a = bits(mask); if (a.length === 1) o[a[0]] = v; } return o; };
    const groupsOf = (e, order) => rowsOf(e, order).map(([mask, v]) => ({ atoms: bits(mask), v })).filter((g) => g.atoms.length > 1).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const orderSeq = () => { const s = []; for (let k = 2; k <= Math.min(4, Object.keys(ex().kSII).length); k++) s.push(k); s.push("M"); return s; };

    // -- the frame --------------------------------------------------------------------------------
    const bar = h("div", "bar"), mols = h("div", "mols"), switches = h("div", "switches");
    mols.setAttribute("role", "group"); mols.setAttribute("aria-label", "Molecule");
    const viewSwitch = h("div", "switch"), taskSwitch = h("div", "switch");
    viewSwitch.setAttribute("role", "group"); viewSwitch.setAttribute("aria-label", "Explanation");
    taskSwitch.setAttribute("role", "group"); taskSwitch.setAttribute("aria-label", "Question");
    const info = h("button", "info", "i");
    info.type = "button"; info.setAttribute("aria-label", "What does the demo show?"); info.setAttribute("aria-expanded", "false");
    const about = h("div", "about"); about.hidden = true;
    about.innerHTML =
      '<button class="about-close" type="button" aria-label="Close">×</button>' +
      "<p><strong>What does the demo show?</strong></p>" +
      "<p>A <strong>graph neural network</strong> (GNN) reads a molecule as a graph: atoms, linked by bonds. In each of its three layers, every atom gathers what its neighbours know. At the end, the GNN sums its atoms and answers one yes/no question, each question with a GNN of its own.</p>" +
      "<p>To explain an answer, <code>shapiq</code> hides atoms in many combinations and watches the answer change. The atoms are the <strong>players</strong>: what each adds alone, and what groups add only together, their <strong>interactions</strong>.</p>" +
      "<p>For a GNN these are computed <strong>exactly</strong>: atoms more than three bonds apart cannot interact, so a fraction of all combinations suffices.</p>";
    switches.append(viewSwitch, taskSwitch, info);
    bar.append(mols, switches, about);
    const setAbout = (open) => { about.hidden = !open; info.setAttribute("aria-expanded", String(open)); };
    info.addEventListener("click", () => setAbout(about.hidden));
    about.querySelector(".about-close").addEventListener("click", () => { setAbout(false); info.focus(); });
    about.addEventListener("focusout", (e) => { if (!about.contains(e.relatedTarget) && e.relatedTarget !== info) setAbout(false); });

    const body = h("div", "body"), stage = h("div", "stage"), frame = h("div", "frame"), panel = h("div", "panel");
    const plot = svg("svg", { class: "plot", viewBox: "0 0 100 100", role: "group", "aria-label": "The molecule; its atoms are the players" }, frame);
    const hint = h("span", "hint"); hint.setAttribute("aria-hidden", "true"); frame.append(hint);
    const caption = h("p", "caption");
    stage.append(frame, caption);
    const live = h("p", "sr"); live.setAttribute("aria-live", "polite");
    body.append(stage, panel);
    root.append(bar, body, live);
    const scene = G.createScene(plot);

    const molButtons = data.molecules.map((m) => {
      const b = h("button", "mol"); b.type = "button"; b.title = m.name;
      b.append(G.tile(m), h("span", "", m.short || m.name));
      b.addEventListener("click", () => { state.mol = m; state.task = taskById(m.task); reset(true); });
      mols.append(b);
      return b;
    });
    const viewButtons = VIEWS.map(([id, label]) => { const b = h("button", "opt", label); b.type = "button"; b.addEventListener("click", () => setView(id)); viewSwitch.append(b); return b; });
    const taskButtons = data.tasks.map((t) => { const b = h("button", "opt", t.label); b.type = "button"; b.addEventListener("click", () => { state.task = t; reset(false); }); taskSwitch.append(b); return b; });
    function setView(id) { state.view = id; state.focus = null; state.pinned = false; state.row = null; update(); viewButtons[VIEWS.findIndex((v) => v[0] === id)].focus({ preventScroll: true }); }
    function reset(newMolecule) {
      state.masked = new Set(); state.focus = null; state.pinned = false; state.row = null;
      if (!orderSeq().includes(state.order)) state.order = 2;
      if (newMolecule) scene.setMolecule(state.mol);
      update();
    }

    // -- pointing at atoms, and the keyboard ------------------------------------------------------
    function toggle(k) { if (state.masked.has(k)) state.masked.delete(k); else state.masked.add(k); state.hinted = true; draw(); renderPanel(); }
    function pin(k) { state.hinted = true; if (k < 0 || (state.pinned && state.focus === k)) { state.pinned = false; state.focus = null; } else { state.pinned = true; state.focus = k; state.row = null; } draw(); renderPanel(); }
    plot.addEventListener("mousedown", (e) => e.preventDefault()); // a click points; it does not move the keyboard's focus
    plot.addEventListener("pointermove", (e) => {
      if (e.pointerType === "touch") return;
      const k = scene.atomAt(e);
      plot.style.cursor = k >= 0 && state.view !== "alone" ? "pointer" : "";
      if (state.view === "together" && !state.pinned && state.focus !== (k < 0 ? null : k)) { state.focus = k < 0 ? null : k; state.row = null; if (k >= 0) state.hinted = true; draw(); renderPanel(); }
    });
    plot.addEventListener("pointerleave", () => { if (state.view === "together" && !state.pinned && state.focus != null) { state.focus = null; draw(); renderPanel(); } });
    plot.addEventListener("click", (e) => { const k = scene.atomAt(e); if (state.view === "predict") { if (k >= 0) toggle(k); } else if (state.view === "together") pin(k); });
    plot.addEventListener("keydown", (e) => {
      const k = Number(e.target.dataset?.k);
      if (Number.isNaN(k) || (e.key !== "Enter" && e.key !== " ")) return;
      e.preventDefault();
      if (state.view === "predict") toggle(k); else if (state.view === "together") pin(k);
    });
    plot.addEventListener("focusin", (e) => { const k = Number(e.target.dataset?.k); if (!Number.isNaN(k)) scene.keyFocus(k); });
    plot.addEventListener("focusout", (e) => { if (!plot.contains(e.relatedTarget)) scene.keyFocus(null); });

    // -- the molecule: the state as the scene's rings, bands and shapes ---------------------------
    function draw() {
      const m = state.mol, e = ex();
      for (const hit of scene.hits()) {
        const k = Number(hit.dataset.k), name = G.atomName(m, k);
        if (state.view === "alone") { hit.removeAttribute("tabindex"); hit.removeAttribute("role"); hit.removeAttribute("aria-label"); hit.removeAttribute("aria-pressed"); continue; }
        hit.setAttribute("tabindex", "0"); hit.setAttribute("role", "button");
        if (state.view === "predict") { hit.setAttribute("aria-label", `${name} ${k + 1}, ${state.masked.has(k) ? "hidden" : "shown"}`); hit.setAttribute("aria-pressed", String(state.masked.has(k))); }
        else { hit.setAttribute("aria-label", `Interactions of ${name.toLowerCase()} ${k + 1}`); hit.removeAttribute("aria-pressed"); }
      }
      const explained = state.view !== "predict";
      const own = state.view === "alone" ? singles(e, 1) : state.view === "together" ? singles(e, state.order) : {};
      let groups = [], groupLargest = 1;
      if (state.view === "together") {
        const all = groupsOf(e, state.order);
        groupLargest = Math.max(1e-9, Math.abs(all[0]?.v || 0));
        const limit = state.order === "M" ? 4 : state.order === 2 ? 12 : 7;
        groups = state.row ? [state.row] : state.focus != null ? all.filter((g) => g.atoms.includes(state.focus)).slice(0, 8) : all.slice(0, limit);
      }
      scene.update({
        rings: own, ringLargest: Math.max(1e-9, ...Object.values(own).map(Math.abs)), groups, groupLargest,
        masked: explained ? new Set() : state.masked, focus: state.view === "together" ? state.focus : null,
        truth: e.truth, truthKey: `${m.id}/${state.task.id}`, truthShown: explained, dim: state.row ? new Set(state.row.atoms) : null,
      });
    }

    // -- the panel --------------------------------------------------------------------------------
    function meta(m, t, e) {
      const p = h("p", "meta"), a = h("span"), b = h("span");
      a.append("Explained with ", h("strong", "", "GraphSHAP-IQ"), `, exact: ${e.budget.toLocaleString("en")} of ${(2 ** e.nAtoms).toLocaleString("en")} maskings`);
      b.append(`GIN, 3 layers: ${percent(e.p)} “${t.yes}”, a score of `, value(e.logOdds));
      p.append(a, b);
      return p;
    }
    function renderPanel() {
      const m = state.mol, t = state.task, e = ex(), content = h("div", "panel-in");
      if (state.view === "predict") predictionPanel(content, m, t, e); else explanationPanel(content, m, t, e);
      panel.replaceChildren(content);
    }
    function predictionPanel(box, m, t, e) {
      const n = state.masked.size, p = sig(n ? G.predict(data.symbols, m, t, state.masked) : e.logOdds);
      const number = h("p", "pred-number", h("span", "pred-value", percent(p)), h("span", "verdict", p >= 0.5 ? t.yes : t.no));
      const scale = h("div", "scale"), fill = h("span", "scale-fill"), mark = h("span", "scale-mark");
      fill.style.width = `${p * 100}%`; mark.style.left = `${p * 100}%`; scale.setAttribute("aria-hidden", "true");
      scale.append(fill, h("span", "scale-half"), mark, h("span", "scale-label", t.no), h("span", "scale-label r", t.yes));
      const compare = h("p", "pred-compare");
      if (n) {
        const d = Math.round(p * 100) - Math.round(e.p * 100);
        compare.append(`With ${n} ${n === 1 ? "atom" : "atoms"} hidden: `, h("strong", d >= 0 ? "up" : "down", `${d >= 0 ? "+" : "−"}${Math.abs(d)} points`), ` from the whole molecule’s ${percent(e.p)}. A hidden atom keeps its bonds, but the GNN no longer knows its element.`);
      } else compare.append("Click atoms to hide them from the GNN, and see what it predicts without them: it runs in your browser.");
      const actions = h("div", "actions");
      const act = (label, fn, hidden) => { const b = h("button", "act", label); b.type = "button"; b.hidden = !!hidden; b.addEventListener("click", () => { fn(); state.hinted = true; draw(); renderPanel(); }); actions.append(b); };
      if (e.truth.length) act(`Hide the ${t.truthLabel}`, () => { state.masked = new Set(e.truth); });
      act("Hide all", () => { state.masked = new Set(m.atoms.map((_, k) => k)); });
      act("Show all", () => { state.masked = new Set(); }, !n);
      const link = (v, label) => { const b = h("button", "link", label); b.type = "button"; b.addEventListener("click", () => setView(v)); return b; };
      box.append(
        h("p", "pred-model", `${t.question[0].toUpperCase()}${t.question.slice(1)} The GNN predicts`), number,
        h("p", "", `the probability of “${t.yes}”`), scale, compare, actions,
        h("p", "pred-why", "But why? See the ", link("alone", "atoms alone"), " or the ", link("together", "atoms together"), "."),
      );
      live.textContent = `The GNN predicts ${percent(p)}, ${p >= 0.5 ? t.yes : t.no}.`;
    }
    function explanationPanel(box, m, t, e) {
      const title = h("p", "title"), detail = h("p", "detail"), slot = h("div", "slot"), list = h("ol", "rows");
      if (state.view === "alone") {
        title.append(h("strong", "", "Atoms alone"), "the Shapley value of every atom");
        detail.append("What each atom adds to the GNN’s score, on average over all maskings. With the masked molecule’s ", value(e.empty), ", the atoms add up to the GNN’s ", value(e.logOdds), ".");
        const own = singles(e, 1), rows = Object.keys(own).map((k) => ({ atoms: [Number(k)], v: own[k] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
        const largest = Math.max(1e-9, Math.abs(rows[0]?.v || 0));
        for (const g of rows.slice(0, 5)) list.append(G.row(m, g.atoms, g.v, largest, G.atomName(m, g.atoms[0])));
      } else {
        const groups = groupsOf(e, state.order), seq = orderSeq(), at = seq.indexOf(state.order), [label, index] = ORDERS[state.order];
        if (state.focus != null) {
          title.append(h("strong", "", G.atomName(m, state.focus)), "its strongest interactions");
          detail.append("On its own, this atom adds ", value(singles(e, state.order)[state.focus] || 0), ". The rest of what it adds, it adds together with the atoms it is joined to here.");
        } else {
          title.append(h("strong", "", "Atoms together"), state.order === "M" ? "groups of any size" : state.order === 2 ? "pairs of atoms" : `groups of up to ${state.order} atoms`);
          if (state.order === "M") detail.append(`The Möbius interactions: the share of every group of atoms that no smaller group explains, ${e.moebiusCount.toLocaleString("en")} of them here. Exact, but not simple: large groups add and subtract a lot.`);
          else detail.append(`The ${index}: what an atom adds alone (the ring around it) and what ${state.order === 2 ? "two atoms" : "groups of atoms"} add only together (${state.order === 2 ? "a band" : "a band or a shape"}). They still add up to the GNN’s `, value(e.logOdds), ".");
        }
        const stepper = h("div", "stepper"), less = h("button", "step", "−"), more = h("button", "step", "+");
        less.type = more.type = "button";
        less.setAttribute("aria-label", "Smaller groups"); more.setAttribute("aria-label", "Larger groups");
        less.setAttribute("aria-disabled", String(at <= 0)); more.setAttribute("aria-disabled", String(at >= seq.length - 1));
        const go = (by) => { const to = seq[at + by]; if (to === undefined) return; state.order = to; state.row = null; update(); panel.querySelectorAll(".step")[by < 0 ? 0 : 1]?.focus({ preventScroll: true }); };
        less.addEventListener("click", () => go(-1)); more.addEventListener("click", () => go(1));
        stepper.append(less, h("span", "step-label", label, h("small", "", index)), more);
        slot.append(stepper);
        const rows = state.focus != null ? groups.filter((g) => g.atoms.includes(state.focus)) : groups;
        const largest = Math.max(1e-9, Math.abs(groups[0]?.v || 0));
        for (const g of rows.slice(0, 5)) {
          const li = G.row(m, g.atoms, g.v, largest, G.groupName(m, g.atoms));
          li.tabIndex = 0; li.setAttribute("aria-label", `${G.groupName(m, g.atoms)}: ${signed(g.v)}`);
          const on = () => { state.row = g; draw(); for (const o of list.children) o.classList.toggle("dim", o !== li); };
          const off = () => { state.row = null; draw(); for (const o of list.children) o.classList.remove("dim"); };
          li.addEventListener("pointerenter", (ev) => { if (ev.pointerType !== "touch") on(); });
          li.addEventListener("pointerleave", (ev) => { if (ev.pointerType !== "touch") off(); });
          li.addEventListener("focus", on); li.addEventListener("blur", off);
          list.append(li);
        }
        if (!rows.length) list.append(h("li", "none", "No interactions of this atom at this order."));
      }
      slot.append(list);
      box.append(title, detail, slot, legend(t, e.truth));
      if (t.id === m.task && m.takeaway && state.focus == null) box.append(h("p", "takeaway", m.takeaway));
      box.append(meta(m, t, e));
    }

    function update() {
      const m = state.mol, t = state.task;
      molButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(data.molecules[k] === m)));
      viewButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(VIEWS[k][0] === state.view)));
      taskButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(data.tasks[k] === t)));
      caption.replaceChildren(h("strong", "", m.name), formula(m.formula), h("span", "blurb", m.blurb));
      hint.textContent = state.view === "predict" ? "Click an atom to hide it" : "Point at an atom";
      hint.hidden = state.view === "alone" || state.hinted || state.focus != null;
      draw();
      renderPanel();
    }
    scene.setMolecule(state.mol);
    update();
  });
})();
