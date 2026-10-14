// The players of the logo are springs: hovering the logo makes them wiggle, and a player can be
// pulled and let go. Each player is held in place by a spring, the lines between players are
// springs too, and the lines and their caps follow the players they end at. The sidebar carries
// the logo inline (_templates/sidebar/brand.html); the landing page's logo is replaced by a copy.
(() => {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const ANCHOR = 140; // stiffness of the spring that holds a player in place (1/s²)
  const LINK = 45; // stiffness of the lines between players (1/s²)
  const DAMPING = 6; // a few visible swings, at rest after about a second (1/s)
  const KICK = 95; // speed of the wiggle on hover (logo units/s; the logo is 280 units wide)
  const MAX_PULL = 70; // beyond this distance, a pulled player follows the pointer reluctantly
  const PLAYER_RADIUS = 15; // larger circles are players, smaller ones are the caps of lines

  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  // the center of a circle in its group's coordinates (some circles carry rotate(-90))
  function center(circle) {
    const transforms = circle.transform.baseVal;
    const x = circle.cx.baseVal.value;
    const y = circle.cy.baseVal.value;
    if (!transforms.numberOfItems) return { x, y };
    const m = transforms.consolidate().matrix;
    return { x: m.a * x + m.c * y + m.e, y: m.b * x + m.d * y + m.f };
  }

  // the two ends of a straight line drawn as "M x1,y1 x2,y2" (or relative "m")
  function ends(path) {
    const d = path.getAttribute("d").trim();
    const numbers = d.match(/-?\d*\.?\d+(?:e-?\d+)?/gi)?.map(Number) ?? [];
    if (!/^[Mm]/.test(d) || numbers.length !== 4) return null;
    const [x1, y1, x2, y2] = numbers;
    return d[0] === "m" ? [{ x: x1, y: y1 }, { x: x1 + x2, y: y1 + y2 }] : [{ x: x1, y: y1 }, { x: x2, y: y2 }];
  }

  function model(svg) {
    const group = svg.querySelector("circle")?.parentNode;
    if (!group) return null;
    const circles = [...group.querySelectorAll(":scope > circle")];
    const players = circles
      .filter((circle) => circle.r.baseVal.value > PLAYER_RADIUS)
      .map((circle) => {
        const rest = center(circle); // consolidates the transform: read the attribute after it
        circle.classList.add("shapiq-logo-player");
        const base = circle.getAttribute("transform") ?? "";
        return { el: circle, base, rest, r: circle.r.baseVal.value, x: 0, y: 0, vx: 0, vy: 0 };
      });
    if (!players.length) return null;
    const nearest = (point) =>
      players.reduce((best, p) => (distance(point, p.rest) < distance(point, best.rest) ? p : best));
    const caps = circles
      .filter((circle) => circle.r.baseVal.value <= PLAYER_RADIUS)
      .map((circle) => {
        const player = nearest(center(circle));
        return { el: circle, base: circle.getAttribute("transform") ?? "", player };
      });
    const lines = [...group.querySelectorAll(":scope > path")]
      .filter((path) => path.getAttribute("fill") === "none")
      .map((path) => ({ el: path, ends: ends(path) }))
      .filter((line) => line.ends)
      .map((line) => ({ ...line, players: line.ends.map(nearest) }));
    const links = [];
    for (const { players: [a, b] } of lines) {
      if (a !== b && !links.some((l) => (l.a === a && l.b === b) || (l.a === b && l.b === a))) {
        links.push({ a, b, length: distance(a.rest, b.rest) });
      }
    }
    return { svg, group, players, caps, lines, links, held: null, running: false };
  }

  function render(m) {
    for (const p of m.players) p.el.setAttribute("transform", `translate(${p.x} ${p.y}) ${p.base}`);
    for (const c of m.caps) c.el.setAttribute("transform", `translate(${c.player.x} ${c.player.y}) ${c.base}`);
    for (const line of m.lines) {
      const [a, b] = line.ends;
      const [pa, pb] = line.players;
      line.el.setAttribute("d", `M ${a.x + pa.x},${a.y + pa.y} L ${b.x + pb.x},${b.y + pb.y}`);
    }
  }

  function step(m, dt) {
    for (const p of m.players) {
      p.ax = -ANCHOR * p.x - DAMPING * p.vx;
      p.ay = -ANCHOR * p.y - DAMPING * p.vy;
    }
    for (const { a, b, length } of m.links) {
      const dx = b.rest.x + b.x - (a.rest.x + a.x);
      const dy = b.rest.y + b.y - (a.rest.y + a.y);
      const now = Math.hypot(dx, dy) || 1;
      const force = (LINK * (now - length)) / now;
      a.ax += force * dx;
      a.ay += force * dy;
      b.ax -= force * dx;
      b.ay -= force * dy;
    }
    for (const p of m.players) {
      if (p === m.held) continue;
      p.vx += p.ax * dt;
      p.vy += p.ay * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
    }
  }

  function animate(m) {
    if (m.running) return;
    m.running = true;
    let last = performance.now();
    const frame = (now) => {
      const dt = Math.min((now - last) / 1000, 1 / 30);
      last = now;
      for (let i = 0; i < 4; i += 1) step(m, dt / 4);
      // a tenth of a unit is a fraction of a pixel: stop the loop there and snap to rest
      const moving = m.players.some((p) => Math.hypot(p.x, p.y) > 0.15 || Math.hypot(p.vx, p.vy) > 1.5);
      if (!moving && !m.held) {
        for (const p of m.players) Object.assign(p, { x: 0, y: 0, vx: 0, vy: 0 });
        render(m);
        m.running = false;
        return;
      }
      render(m);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  function pointerIn(m, event) {
    const matrix = m.group.getScreenCTM();
    return matrix ? new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()) : null;
  }

  function attach(svg) {
    let m = null;
    const ready = () => (m ??= model(svg));
    let lastKick = 0;
    svg.addEventListener("pointerenter", () => {
      if (!ready() || m.held || performance.now() - lastKick < 400) return;
      lastKick = performance.now();
      for (const p of m.players) {
        const angle = Math.random() * 2 * Math.PI;
        const speed = KICK * (0.6 + 0.4 * Math.random());
        p.vx += speed * Math.cos(angle);
        p.vy += speed * Math.sin(angle);
      }
      animate(m);
    });
    let pulled = false;
    svg.addEventListener("pointerdown", (event) => {
      const point = ready() && pointerIn(m, event);
      if (!point) return;
      const player = m.players.find((p) => distance(point, { x: p.rest.x + p.x, y: p.rest.y + p.y }) < p.r + 6);
      if (!player) return;
      event.preventDefault(); // no text selection, no dragging of the link
      m.held = player;
      m.grab = { x: point.x - player.rest.x - player.x, y: point.y - player.rest.y - player.y };
      pulled = false;
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("is-pulling");
      animate(m);
    });
    svg.addEventListener("pointermove", (event) => {
      const p = m?.held;
      const point = p && pointerIn(m, event);
      if (!point) return;
      let dx = point.x - m.grab.x - p.rest.x;
      let dy = point.y - m.grab.y - p.rest.y;
      const far = Math.hypot(dx, dy);
      if (far > MAX_PULL) {
        const scale = (MAX_PULL + (far - MAX_PULL) * 0.25) / far; // rubber band
        dx *= scale;
        dy *= scale;
      }
      if (Math.hypot(dx - p.x, dy - p.y) > 0.5) pulled = true;
      Object.assign(p, { x: dx, y: dy, vx: 0, vy: 0 });
    });
    const release = () => {
      if (!m?.held) return;
      m.held = null;
      svg.classList.remove("is-pulling");
      animate(m);
    };
    svg.addEventListener("pointerup", release);
    svg.addEventListener("pointercancel", release);
    // the sidebar logo is a link home: a pull is not a click
    svg.closest("a")?.addEventListener(
      "click",
      (event) => {
        if (pulled) {
          event.preventDefault();
          pulled = false;
        }
      },
      true,
    );
  }

  // the landing page shows the logo as an image: replace it by a copy of the sidebar's logo
  function inlineHeroLogos() {
    for (const image of document.querySelectorAll("img.shapiq-hero-logo")) {
      const theme = image.classList.contains("only-dark") ? "dark" : "light";
      const source = document.querySelector(`.sidebar-brand svg.shapiq-logo.only-${theme}`);
      if (!source) continue;
      const copy = source.cloneNode(true);
      copy.setAttribute("class", `shapiq-logo shapiq-hero-logo only-${theme}`);
      copy.setAttribute("aria-label", image.alt || "shapiq");
      for (const element of copy.querySelectorAll("[id]")) {
        const id = element.id;
        element.id = `${id}-hero`;
        for (const user of copy.querySelectorAll(`[fill="url(#${id})"]`)) {
          user.setAttribute("fill", `url(#${id}-hero)`);
        }
      }
      image.replaceWith(copy);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    inlineHeroLogos();
    document.querySelectorAll("svg.shapiq-logo").forEach(attach);
  });
})();
