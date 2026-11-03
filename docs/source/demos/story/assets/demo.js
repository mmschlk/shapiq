// The story demo (demos/story): two steps up to the orders demo, with a game small enough to see
// whole. A house of California housing, valued by an XGBoost model, plays with three of its
// features, its latitude, its longitude and its income (the other five unknown): the value of a
// coalition of them is the house's predicted price with only those features known, relative to
// knowing none. The game is the house's full decomposition, restricted to these three (computed in
// ../compute/pack.py into data.js). Its eight coalitions lie in a lattice, from none known at the
// top to all three at the bottom, every step down one feature more, as the orders demo's layers grow
// more complex downwards.
//
// The demo has two parts, each a container of its own (data-part):
//
// - "credit", who gets the credit? Tap a feature: the steps down the lattice where it joins light
//   up, each with what it adds there, and its Shapley value, its credit, is their weighted mean
//   (the share of the orders in which the features could join that pass that step). Tap two: the
//   square of their coalitions lights up, and what they do only together, their interaction.
//   Latitude alone adds $68k, longitude alone $3k, and both together only $3k: their interaction,
//   -$68k, which the Shapley values split between them (+$34k and -$32k), and so hide.
// - "fit", how good is an explanation? An explanation of order k (FSII) predicts every coalition:
//   the sum of the values of its groups of up to k features in it. On the lattice, the model's
//   value of every coalition (orange, as the orders demo's mechanism) and the explanation's
//   (green, as an explanation applied to the input there), and how far apart; its terms, and its
//   R^2 beyond size, the orders demo's faithfulness. Order 1, the Shapley values, misses what
//   latitude and longitude do together; order 2 takes it; order 3, the full decomposition, is the
//   model itself.
(() => {
  const roots = [...document.querySelectorAll(".story-demo")];
  if (!roots.length) return;

  const SVG = "http://www.w3.org/2000/svg";
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const add = (parent, tag, attributes = {}) => {
    const node = document.createElementNS(SVG, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    parent.appendChild(node);
    return node;
  };
  const round = (value, digits) => Math.round(value * 10 ** digits) / 10 ** digits;
  // a value in thousands of dollars: +$68k, -$32k, +$0.1k
  const money = (value) => {
    const size = Math.abs(value);
    const shown = size < 9.95 ? size.toFixed(1).replace(/\.0$/, "") : String(Math.round(size));
    return `${value < -0.005 ? "−" : "+"}$${shown}k`;
  };
  const tone = (value) => (Math.abs(value) < 0.5 ? "story-zero" : value > 0 ? "story-up" : "story-down");
  // R^2 as the orders demo shows it: two decimals, three close to 1, and 1.00 only where exact
  const fit2 = (r2) => {
    if (r2 >= 1 - 5e-7) return "1.00";
    if (r2 < 0.995) return r2.toFixed(2);
    return r2 >= 0.9995 ? "> 0.999" : r2.toFixed(3);
  };
  const SHORT = ["Lat", "Long", "Inc"]; // the features, short, in the lattice's coalitions
  // the lattice, row by row, from none known to all three
  const ROWS = [[[]], [[0], [1], [2]], [[0, 1], [0, 2], [1, 2]], [[0, 1, 2]]];
  const keyOf = (group) => [...group].sort((a, b) => a - b).join(",");

  function start(root, data) {
    const part = root.dataset.part === "fit" ? "fit" : "credit";
    const n = data.players.length;
    const values = new Map(data.values.map(([group, value]) => [keyOf(group), value]));
    const v = (group) => values.get(keyOf(group));
    const names = data.players.map((player) => player.name);
    const nameOf = (group) => (group.length ? group.map((p) => names[p]).join(" and ") : "none");

    root.replaceChildren();
    root.classList.add("story-demo-ready", `story-demo-${part}`);

    // -- the head: the features, with their values (in the credit part, the buttons that pick them)
    const head = el("div", "story-head");
    const players = el("div", "story-players");
    const chips = data.players.map((player, p) => {
      const chip = el(part === "credit" ? "button" : "span", "story-chip");
      if (part === "credit") chip.type = "button";
      chip.append(el("span", "story-chip-name", player.name), el("span", "story-chip-value", player.value));
      players.append(chip);
      return chip;
    });
    head.append(players);
    const lead = el("p", "story-lead");
    head.append(lead);
    root.append(head);

    // -- the lattice: its coalitions, as cards, and the steps between them, a feature joining
    const lattice = el("div", "story-lattice");
    const edges = document.createElementNS(SVG, "svg");
    edges.setAttribute("class", "story-edges");
    edges.setAttribute("aria-hidden", "true");
    lattice.append(edges);
    const nodes = ROWS.flat().map((group) => {
      const card = el("div", "story-node");
      card.setAttribute("role", "img");
      const pills = el("span", "story-pills");
      for (let p = 0; p < n; p++) pills.append(el("span", group.includes(p) ? "story-pill story-pill-on" : "story-pill", SHORT[p]));
      card.append(pills);
      const body = el("span", "story-node-body");
      card.append(body);
      lattice.append(card);
      return { group, card, body };
    });
    const nodeOf = (group) => nodes.find((node) => keyOf(node.group) === keyOf(group));
    const steps = [];
    for (const from of nodes) {
      for (const to of nodes) {
        if (to.group.length === from.group.length + 1 && from.group.every((p) => to.group.includes(p))) {
          const player = to.group.find((p) => !from.group.includes(p));
          steps.push({ from, to, player, line: add(edges, "line", { class: "story-step" }), tag: el("span", "story-step-tag") });
        }
      }
    }
    for (const step of steps) lattice.append(step.tag);
    root.append(lattice);

    const panel = el("div", "story-panel");
    root.append(panel);

    // -- the lattice's layout, for its width: three columns, a row for every size of coalition
    let place = { width: 0 };
    function layout() {
      const width = lattice.clientWidth;
      if (!width || width === place.width) return;
      const narrow = width < 480;
      const cardWidth = Math.min(narrow ? 112 : 150, width / 3 - (narrow ? 8 : 18));
      const cardHeight = part === "fit" ? (narrow ? 66 : 70) : narrow ? 52 : 56;
      const gap = narrow ? 30 : 40;
      const columns = [width / 6, width / 2, (5 * width) / 6];
      const rowY = (row) => cardHeight / 2 + 4 + row * (cardHeight + gap);
      place = { width, cardWidth, cardHeight, at: new Map() };
      ROWS.forEach((row, r) => {
        row.forEach((group, k) => {
          const x = row.length === 1 ? width / 2 : columns[k];
          place.at.set(keyOf(group), [x, rowY(r)]);
          const card = nodeOf(group).card;
          Object.assign(card.style, { left: `${round(x - cardWidth / 2, 2)}px`, top: `${round(rowY(r) - cardHeight / 2, 2)}px`, width: `${round(cardWidth, 2)}px`, height: `${cardHeight}px` });
        });
      });
      const height = rowY(ROWS.length - 1) + cardHeight / 2 + 4;
      lattice.style.height = `${height}px`;
      edges.setAttribute("width", width);
      edges.setAttribute("height", height);
      for (const step of steps) {
        const [[x1, y1], [x2, y2]] = [place.at.get(keyOf(step.from.group)), place.at.get(keyOf(step.to.group))];
        const [a, b] = [y1 + cardHeight / 2, y2 - cardHeight / 2];
        Object.entries({ x1: round(x1, 1), y1: round(a, 1), x2: round(x2, 1), y2: round(b, 1) }).forEach(([name, value]) => step.line.setAttribute(name, value));
        // its tag halfway down the step
        Object.assign(step.tag.style, { left: `${round((x1 + x2) / 2, 1)}px`, top: `${round((a + b) / 2, 1)}px` });
      }
    }

    // -- the parts --------------------------------------------------------------------------------
    if (part === "credit") credit();
    else fit();
    layout();
    new ResizeObserver(layout).observe(lattice);

    // who gets the credit? A feature's steps, what it adds on each, and their weighted mean
    function credit() {
      lead.append("The house's predicted price with only some of these features known, against knowing none. Tap a feature for its credit, or two for what they do only together.");
      for (const node of nodes) {
        const value = v(node.group);
        node.body.replaceChildren(el("span", `story-value ${tone(value)}`, node.group.length ? money(value) : "$0"));
        node.card.setAttribute("aria-label", `${node.group.length ? nameOf(node.group) : "No feature"} known: ${money(value)}`);
      }
      // a feature joins a coalition of s others in s! (n - s - 1)! of the n! orders of arrival
      const factorial = (k) => (k <= 1 ? 1 : k * factorial(k - 1));
      const weight = (s) => (factorial(s) * factorial(n - s - 1)) / factorial(n);
      const fraction = (s) => ({ 2: "½", 3: "⅓", 6: "⅙" })[factorial(n) / (factorial(s) * factorial(n - s - 1))] ?? String(round(weight(s), 3));
      let picked = [0];
      chips.forEach((chip, p) => {
        chip.addEventListener("click", () => {
          if (picked.includes(p)) picked = picked.filter((other) => other !== p);
          else picked = [...picked, p].slice(-2);
          show();
        });
      });
      function show() {
        chips.forEach((chip, p) => chip.setAttribute("aria-pressed", String(picked.includes(p))));
        const lit = new Set();
        const litNodes = new Set();
        for (const step of steps) {
          step.tag.replaceChildren();
          step.tag.className = "story-step-tag";
        }
        panel.replaceChildren();
        if (picked.length === 1) {
          // its steps: wherever it joins, what it adds there; and their mean, weighted
          const [p] = picked;
          const terms = el("div", "story-terms");
          const mine = steps.filter((step) => step.player === p).sort((a, b) => a.from.group.length - b.from.group.length);
          let sum = 0;
          for (const step of mine) {
            const added = v(step.to.group) - v(step.from.group);
            sum += weight(step.from.group.length) * added;
            lit.add(step);
            litNodes.add(step.from).add(step.to);
            step.tag.className = `story-step-tag story-step-tag-on ${tone(added)}`;
            step.tag.textContent = money(added);
            const term = el("span", "story-term");
            const others = step.from.group.length ? `with ${step.from.group.map((q) => names[q]).join(" and ")}` : "alone";
            term.append(el("span", "story-term-weight", fraction(step.from.group.length)), el("span", "story-term-times", "×"), el("span", `story-term-value ${tone(added)}`, money(added)), el("span", "story-term-when", others));
            terms.append(term);
          }
          const total = el("span", "story-term story-term-total");
          total.append(el("span", "story-term-times", "="), el("span", `story-term-value ${tone(sum)}`, money(sum)), el("span", "story-term-when", "its credit"));
          terms.append(total);
          panel.append(el("p", "story-panel-title", `${names[p]}'s credit, its Shapley value`), terms);
          const alone = v([p]);
          panel.append(
            el(
              "p",
              "story-panel-note",
              `Alone, ${names[p].toLowerCase()} adds ${money(alone)}. Its credit is the mean of what it adds wherever it joins, weighted by the share of the ${factorial(n)} orders in which the features could join that take that step: ${money(sum)}.`,
            ),
          );
        } else if (picked.length === 2) {
          // what they do only together: the square of their coalitions
          const [p, q] = [...picked].sort((a, b) => a - b);
          const corners = [[], [p], [q], [p, q]];
          for (const step of steps) {
            const inside = corners.some((group) => keyOf(group) === keyOf(step.from.group)) && corners.some((group) => keyOf(group) === keyOf(step.to.group));
            if (inside) {
              lit.add(step);
              litNodes.add(step.from).add(step.to);
            }
          }
          const together = v([p, q]) - v([p]) - v([q]) + v([]);
          const terms = el("div", "story-terms");
          const term = (sign, group) => {
            const node = el("span", "story-term");
            node.append(el("span", "story-term-times", sign), el("span", `story-term-value ${tone(v(group))}`, group.length ? money(v(group)) : "$0"), el("span", "story-term-when", group.length ? group.map((r) => names[r]).join(" and ") : "none"));
            return node;
          };
          terms.append(term("", [p, q]), term("−", [p]), term("−", [q]), term("+", []));
          const total = el("span", "story-term story-term-total");
          total.append(el("span", "story-term-times", "="), el("span", `story-term-value ${tone(together)}`, money(together)), el("span", "story-term-when", "only together"));
          terms.append(total);
          panel.append(el("p", "story-panel-title", `${names[p]} × ${names[q]}: what they do only together`), terms);
          const [sp, sq] = [data.shapley[p], data.shapley[q]];
          panel.append(
            el(
              "p",
              "story-panel-note",
              `Together they add ${money(v([p, q]))}, not ${money(v([p]) + v([q]))}: their interaction, ${money(together)}, is what they do only together. Their credits split it between them (${names[p]} ${money(sp)}, ${names[q]} ${money(sq)}), and so hide it.`,
            ),
          );
        } else {
          panel.append(el("p", "story-panel-note", "Tap a feature for its credit, or two for what they do only together."));
        }
        for (const step of steps) step.line.classList.toggle("story-step-on", lit.has(step));
        for (const node of nodes) node.card.classList.toggle("story-node-on", litNodes.has(node));
        root.classList.toggle("story-picked", lit.size > 0);
        // the credits of all three, which add up to the value of all three known
        const credits = el("div", "story-credits");
        credits.append(el("span", "story-credits-title", "Credits (Shapley values)"));
        data.shapley.forEach((value, p) => {
          const item = el("span", picked.includes(p) ? "story-credit story-credit-on" : "story-credit");
          item.append(el("span", "story-credit-name", names[p]), el("span", `story-credit-value ${tone(value)}`, money(value)));
          credits.append(item);
        });
        const all = el("span", "story-credit story-credit-sum");
        all.append(el("span", "story-credit-name", "together"), el("span", `story-credit-value ${tone(v([0, 1, 2]))}`, money(v([0, 1, 2]))));
        credits.append(all);
        panel.append(credits);
      }
      show();
    }

    // how good is an explanation? Each order's prediction of every coalition, against the model's
    function fit() {
      lead.append("An explanation predicts every coalition: the sum of its values of the features, and, from order 2, of their groups, in it. Next to the model's price (orange), the explanation's (green): where they part, the explanation misses what the features do together.");
      const orders = data.orders.map((order) => ({ ...order, values: new Map(order.values.map(([group, value]) => [keyOf(group), value])) }));
      const predict = (order, group) => [...order.values].reduce((sum, [key, value]) => (key.split(",").every((p) => group.includes(Number(p))) ? sum + value : sum), 0);
      const largest = Math.max(...data.values.map(([, value]) => Math.abs(value)), ...orders.flatMap((order) => nodes.map((node) => Math.abs(predict(order, node.group)))));
      // the order to show: buttons, with what each order is, and its R^2
      const picker = el("div", "story-orders");
      picker.setAttribute("role", "group");
      picker.setAttribute("aria-label", "Order of the explanation");
      const LABELS = ["Shapley values", "pairs", "full decomposition"];
      const buttons = orders.map((order, k) => {
        const button = el("button", "story-order");
        button.type = "button";
        button.append(el("strong", "", `Order ${order.order}`), el("span", "story-order-what", LABELS[k] ?? ""));
        button.addEventListener("click", () => {
          shown = k;
          show();
        });
        picker.append(button);
        return button;
      });
      head.append(picker);
      // a card's bars: the model's (an orange tick) and the explanation's (a green bar), from 0
      for (const node of nodes) {
        const track = el("span", "story-track");
        const bar = el("span", "story-bar");
        const tick = el("span", "story-tick");
        track.append(el("span", "story-track-zero"), bar, tick);
        const numbers = el("span", "story-numbers");
        node.parts = { track, bar, tick, numbers };
        node.body.append(track, numbers);
      }
      let shown = 0;
      function show() {
        const order = orders[shown];
        buttons.forEach((button, k) => button.setAttribute("aria-pressed", String(k === shown)));
        for (const node of nodes) {
          const [model, guess] = [v(node.group), predict(order, node.group)];
          const at = (value) => `${round(50 + (value / largest) * 50, 2)}%`;
          const [lo, hi] = guess >= 0 ? [50, 50 + (guess / largest) * 50] : [50 + (guess / largest) * 50, 50];
          Object.assign(node.parts.bar.style, { left: `${round(lo, 2)}%`, width: `${round(hi - lo, 2)}%` });
          node.parts.tick.style.left = at(model);
          const off = guess - model;
          node.parts.numbers.replaceChildren(el("span", "story-number-model", node.group.length ? money(model) : "$0"), el("span", "story-number-guess", node.group.length ? money(guess) : "$0"));
          node.card.classList.toggle("story-node-off", Math.abs(off) >= 1);
          node.card.setAttribute("aria-label", `${node.group.length ? nameOf(node.group) : "No feature"} known: the model ${money(model)}, order ${order.order} ${money(guess)}`);
        }
        // its terms, and its fit
        panel.replaceChildren();
        const terms = el("div", "story-terms story-terms-wrap");
        const sorted = [...order.values].sort(([a], [b]) => a.split(",").length - b.split(",").length || a.localeCompare(b));
        for (const [key, value] of sorted) {
          const group = key.split(",").map(Number);
          const term = el("span", "story-term story-term-chip");
          term.append(el("span", "story-term-when", group.map((p) => SHORT[p]).join(" × ")), el("span", `story-term-value ${tone(value)}`, money(value)));
          terms.append(term);
        }
        const meter = el("div", "story-fit");
        const track = el("span", "story-fit-track");
        const fill = el("span", "story-fit-fill");
        fill.style.width = `${round(Math.max(0, Math.min(1, order.r2)) * 100, 2)}%`;
        track.append(fill);
        meter.append(el("span", "story-fit-name", "faithfulness"), track, el("span", "story-fit-r2", `R² ${fit2(order.r2)}`));
        const NOTES = [
          `The Shapley values predict a coalition by adding up the credits of its features: latitude alone ${money(predict(order, [0]))}, though the model gives it ${money(v([0]))}. They miss what latitude and longitude do only together.`,
          `With the pairs, latitude alone is ${money(predict(order, [0]))} again, and the pair of latitude and longitude takes ${money(order.values.get("0,1"))} back: what they do only together.`,
          "With every group of features, the explanation is the model itself: the full decomposition, exact on every coalition.",
        ];
        panel.append(el("p", "story-panel-title", `Order ${order.order}: ${LABELS[shown]}`), terms, meter, el("p", "story-panel-note", NOTES[shown] ?? ""));
      }
      show();
    }
  }

  // -- loading ----------------------------------------------------------------------------------------
  const begin = () => {
    for (const root of roots) {
      try {
        start(root, window.SHAPIQ_STORY);
      } catch (error) {
        console.error("The story demo could not start:", error);
      }
    }
  };
  if (window.SHAPIQ_STORY) {
    begin();
  } else {
    const loader = document.createElement("script");
    loader.src = `${roots[0].dataset.root}data.js?v=${roots[0].dataset.version ?? ""}`;
    loader.onload = begin;
    document.head.appendChild(loader);
  }
})();
