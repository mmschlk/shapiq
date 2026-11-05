// The shared part of the user guide's illustrations (_ext/shapiq_illustrations.py). An
// illustration's script defines it with shapiqIllustrations.define(name, start); start(stage, kit)
// builds it in the stage of every figure of that name on the page. The kit has the URL of the
// illustration's assets (root), the checksum of its data (version: the page loads data.js before the
// script; where it is missing, the script loads it with ?v=version), and the helpers below: elements, data loading, tweens that respect
// reduced motion, visibility, and the segmented control (pills with a sliding thumb) that the
// illustrations share.
(() => {
  const SVG = "http://www.w3.org/2000/svg";
  const reducedQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const reducedMotion = () => reducedQuery.matches;

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const svg = (tag, attributes = {}, parent = null) => {
    const node = document.createElementNS(SVG, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    if (parent) parent.appendChild(node);
    return node;
  };

  // a script of the assets, loaded once per page: data.js works also for docs opened from disk
  const scripts = new Map();
  const loadScript = (url) => {
    if (!scripts.has(url)) {
      scripts.set(
        url,
        new Promise((resolve, reject) => {
          const script = el("script");
          script.src = url;
          script.onload = resolve;
          script.onerror = () => reject(new Error(`could not load ${url}`));
          document.head.appendChild(script);
        }),
      );
    }
    return scripts.get(url);
  };

  const ease = {
    out: (t) => 1 - (1 - t) ** 3,
    inOut: (t) => (t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2),
    // overshoots a little before it settles, like a spring
    spring: (t) => 1 + 2.2 * (t - 1) ** 3 + 1.2 * (t - 1) ** 2,
  };

  // calls update(p) with the eased progress from 0 to 1 on every frame; with reduced motion, it
  // jumps to the end. Returns a function that stops it.
  const tween = (duration, update, { easing = ease.out, delay = 0, done } = {}) => {
    if (reducedMotion() || duration <= 0) {
      update(1);
      done?.();
      return () => {};
    }
    let frame = 0;
    let start = null;
    const step = (now) => {
      start ??= now + delay;
      const t = Math.min(1, Math.max(0, (now - start) / duration));
      update(easing(t));
      if (t < 1) frame = requestAnimationFrame(step);
      else done?.();
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  };

  // calls back with true when at least `threshold` of the node is visible, or half of the window
  // shows it (a node taller than the window, as on a phone or at a high zoom), and with false when
  // not; returns a function that stops watching
  const watchVisibility = (node, callback, threshold = 0.4) => {
    if (!("IntersectionObserver" in window)) {
      callback(true);
      return () => {};
    }
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[entries.length - 1];
        const view = entry.rootBounds?.height || window.innerHeight;
        callback(
          entry.isIntersecting &&
            (entry.intersectionRatio >= threshold || entry.intersectionRect.height >= 0.5 * view),
        );
      },
      { threshold: Array.from({ length: 21 }, (_, k) => k / 20) },
    );
    observer.observe(node);
    return () => observer.disconnect();
  };

  // a radio group of options, as the demos' switches (demos/orders): the one chosen sits in a black
  // pill, a copy of the options in the chosen colors cut to it, which slides to the next one chosen,
  // so that every label turns exactly as the pill passes over it. Arrow keys move between them.
  // options: [[value, label], ...]; onChange(value, byUser) runs when the value changes.
  const segmented = (label, options, onChange) => {
    const group = el("div", "shapiq-ill-segmented");
    group.setAttribute("role", "radiogroup");
    group.setAttribute("aria-label", label);
    const pill = el("div", "shapiq-ill-segmented-pill");
    pill.setAttribute("aria-hidden", "true");
    const buttons = new Map();
    let value = null;
    let placed = false;
    const place = () => {
      const button = buttons.get(value);
      if (!button || !group.clientWidth) return;
      const top = button.offsetTop;
      const left = button.offsetLeft;
      const right = group.clientWidth - left - button.offsetWidth;
      const bottom = group.clientHeight - top - button.offsetHeight;
      // the first time without sliding there
      if (!placed) pill.style.transition = "none";
      pill.style.clipPath = `inset(${top}px ${right}px ${bottom}px ${left}px round 999px)`;
      if (!placed) {
        placed = true;
        requestAnimationFrame(() => requestAnimationFrame(() => pill.style.removeProperty("transition")));
      }
    };
    const set = (next, byUser = false) => {
      if (next === value) return;
      value = next;
      for (const [key, button] of buttons) {
        button.setAttribute("aria-checked", String(key === value));
        button.tabIndex = key === value ? 0 : -1;
      }
      place();
      onChange?.(value, byUser);
    };
    options.forEach(([key, text], index) => {
      const button = el("button", "shapiq-ill-segment", text);
      button.type = "button";
      button.setAttribute("role", "radio");
      button.addEventListener("click", () => set(key, true));
      button.addEventListener("keydown", (event) => {
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
        if (!step) return;
        event.preventDefault();
        const [nextKey] = options[(index + step + options.length) % options.length];
        set(nextKey, true);
        buttons.get(nextKey).focus();
      });
      buttons.set(key, button);
      group.appendChild(button);
      pill.appendChild(el("span", "shapiq-ill-segment", text));
    });
    group.appendChild(pill);
    new ResizeObserver(place).observe(group);
    return { node: group, set, get value() { return value; } };
  };

  const kit = { el, svg, SVG, ease, tween, reducedMotion, watchVisibility, loadScript, segmented };

  const define = (name, start) => {
    const run = () => {
      const figures = document.querySelectorAll(
        `figure.shapiq-illustration[data-illustration="${name}"]`,
      );
      for (const figure of figures) {
        const stage = figure.querySelector(".shapiq-illustration-stage");
        if (!stage || stage.dataset.started) continue;
        stage.dataset.started = "true";
        const description = stage.getAttribute("aria-label") ?? "";
        stage.replaceChildren();
        Promise.resolve()
          .then(() =>
            start(stage, { ...kit, root: figure.dataset.root, version: figure.dataset.version ?? "" }),
          )
          .then(() => figure.classList.add("shapiq-illustration-ready"))
          .catch((error) => {
            console.error(`The illustration "${name}" could not start:`, error);
            figure.classList.add("shapiq-illustration-failed");
            stage.replaceChildren(el("p", "shapiq-illustration-noscript", description));
          });
      }
    };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", run);
    else run();
  };

  window.shapiqIllustrations = { define, ...kit };
})();
