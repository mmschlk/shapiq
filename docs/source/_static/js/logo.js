// The players of the landing page's logo are springs: hovering the logo makes them wiggle, and a
// player can be pulled and let go. Each player is held in place by a spring, the lines between
// players are springs too, the stiffer the thicker the line (a stronger interaction): the
// thickest line holds its two players together almost like a rod, the thinnest barely pulls. The
// lines and their caps follow the players they end at. When the logo is first seen, a random
// player wiggles and the wiggle runs through the logo along the lines; when it is scrolled into
// view quickly, it gives a little bump in the direction of travel. The landing page shows the logo as an image; it is replaced by a copy of the
// sidebar's inline logo (_templates/sidebar/brand.html). The sidebar's logo itself stays still:
// it is on every page and a link home, where a wiggle on every pass of the mouse would distract.
//
// With reduced motion (macOS "Reduce motion", Windows with "Animation effects" off), the logo
// does not move by itself: no wiggle on hover, no ripple, no bump. The players follow a pulled
// one without swinging, and when it is let go, all of them glide back to their places.
(() => {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const ANCHOR = 140; // stiffness of the spring that holds a player in place (1/s²)
  const LINK = 2.5; // stiffness of a line per squared logo unit of its width (1/s²): 39 to 1180
  const LINK_DAMPING = 0.08; // damping of a line per unit of its stiffness (s): no buzz in stiff pairs
  const DAMPING = 6; // a few visible swings, at rest after about a second (1/s)
  const GLIDE = 0.25; // time constant of the return to rest with reduced motion (s)
  const KICK = 95; // speed of the wiggle on hover (logo units/s; the logo is 280 units wide)
  const RIPPLE = 220; // speed of the wiggle that starts the ripple (logo units/s), halved per hop
  const RIPPLE_HOP = 80; // time the ripple takes along a line (ms)
  const BUMP = 100; // speed of the bump when the logo is scrolled into view (logo units/s)
  const FAST_SCROLL = 1000; // scrolling faster than this bumps the logo (px/s)
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
      .map((path) => ({ el: path, ends: ends(path), width: parseFloat(getComputedStyle(path).strokeWidth) || 1 }))
      .filter((line) => line.ends)
      .map((line) => ({ ...line, players: line.ends.map(nearest) }));
    const links = [];
    for (const { players: [a, b], width } of lines) {
      if (a === b) continue;
      const link = links.find((l) => (l.a === a && l.b === b) || (l.a === b && l.b === a));
      if (link) link.stiffness += LINK * width ** 2;
      else links.push({ a, b, length: distance(a.rest, b.rest), stiffness: LINK * width ** 2 });
    }
    // damping that lets no player swing while one is held (with reduced motion): twice the square
    // root of the stiffest player's springs, a bound on the stiffness of the fastest swing
    // (Gershgorin)
    const springs = Math.max(
      ...players.map((p) => links.reduce((sum, l) => sum + (l.a === p || l.b === p ? l.stiffness : 0), 0)),
    );
    const stiffest = ANCHOR + 2 * springs;
    const calmDamping = 2 * Math.sqrt(stiffest);
    // a bound on how fast a player responds (1/s): shorter simulation steps than that stay stable
    const fastest = Math.max(Math.sqrt(stiffest), calmDamping + 2 * LINK_DAMPING * springs);
    return { svg, group, players, caps, lines, links, calmDamping, fastest, held: null, running: false };
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
    if (reducedMotion.matches && !m.held) {
      const keep = Math.exp(-dt / GLIDE);
      for (const p of m.players) Object.assign(p, { x: p.x * keep, y: p.y * keep, vx: 0, vy: 0 });
      return;
    }
    const damping = reducedMotion.matches ? m.calmDamping : DAMPING;
    for (const p of m.players) {
      p.ax = -ANCHOR * p.x - damping * p.vx;
      p.ay = -ANCHOR * p.y - damping * p.vy;
    }
    for (const { a, b, length, stiffness } of m.links) {
      const dx = b.rest.x + b.x - (a.rest.x + a.x);
      const dy = b.rest.y + b.y - (a.rest.y + a.y);
      const now = Math.hypot(dx, dy) || 1;
      const stretching = ((b.vx - a.vx) * dx + (b.vy - a.vy) * dy) / now; // speed along the line
      const force = (stiffness * (now - length) + LINK_DAMPING * stiffness * stretching) / now;
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
      const steps = Math.max(4, Math.ceil(dt * m.fastest));
      for (let i = 0; i < steps; i += 1) step(m, dt / steps);
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

  // a ripple from a random player: it wiggles, then the players a line away from it are pushed
  // away from it, then the ones two lines away, each half as hard as the one before
  function ripple(m) {
    const start = m.players[Math.floor(Math.random() * m.players.length)];
    const hops = new Map([[start, 0]]);
    for (const queue = [start]; queue.length; ) {
      const p = queue.shift();
      for (const { a, b } of m.links) {
        const next = a === p ? b : b === p ? a : null;
        if (next && !hops.has(next)) {
          hops.set(next, hops.get(p) + 1);
          queue.push(next);
        }
      }
    }
    const push = (p, speed, angle) => {
      if (m.held) return;
      p.vx += speed * Math.cos(angle);
      p.vy += speed * Math.sin(angle);
      animate(m);
    };
    const angle = Math.random() * 2 * Math.PI;
    push(start, RIPPLE, angle);
    setTimeout(() => push(start, 0.8 * RIPPLE, angle + Math.PI), 70); // and back: a wiggle
    for (const p of m.players) {
      const hop = hops.get(p);
      if (!hop) continue; // the start, or a player no line reaches
      const away = Math.atan2(p.rest.y - start.rest.y, p.rest.x - start.rest.x);
      const speed = RIPPLE * 0.5 ** hop * (0.85 + 0.3 * Math.random());
      setTimeout(() => push(p, speed, away), hop * RIPPLE_HOP);
    }
  }

  // the speed of the page's scrolling (px/s, positive downwards), 0 once it has stopped
  const scrolling = { y: window.scrollY, time: performance.now(), speed: 0 };
  window.addEventListener(
    "scroll",
    () => {
      const time = performance.now();
      const y = window.scrollY;
      const speed = ((y - scrolling.y) / Math.max(time - scrolling.time, 1)) * 1000;
      scrolling.speed = time - scrolling.time > 100 ? speed : 0.5 * scrolling.speed + 0.5 * speed;
      Object.assign(scrolling, { y, time });
    },
    { passive: true },
  );
  const scrollSpeed = () => (performance.now() - scrolling.time < 150 ? scrolling.speed : 0);

  // a bump in the direction the logo travels on the screen, as if it had been carried there; a
  // clear one or none, as a faint twitch looks like a glitch
  function bump(m, speed) {
    const along = -Math.sign(speed); // scrolling up carries the logo down
    for (const p of m.players) p.vy += along * BUMP * (0.7 + 0.6 * Math.random());
    animate(m);
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
      if (reducedMotion.matches || !ready() || m.held || performance.now() - lastKick < 400) return;
      lastKick = performance.now();
      for (const p of m.players) {
        const angle = Math.random() * 2 * Math.PI;
        const speed = KICK * (0.6 + 0.4 * Math.random());
        p.vx += speed * Math.cos(angle);
        p.vy += speed * Math.sin(angle);
      }
      animate(m);
    });
    svg.addEventListener("pointerdown", (event) => {
      const point = ready() && pointerIn(m, event);
      if (!point) return;
      const player = m.players.find((p) => distance(point, { x: p.rest.x + p.x, y: p.rest.y + p.y }) < p.r + 6);
      if (!player) return;
      event.preventDefault(); // no text selection, no dragging of the link
      m.held = player;
      m.grab = { x: point.x - player.rest.x - player.x, y: point.y - player.rest.y - player.y };
      svg.setPointerCapture(event.pointerId);
      document.documentElement.classList.add("shapiq-logo-pulling"); // the closed hand everywhere
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
      Object.assign(p, { x: dx, y: dy, vx: 0, vy: 0 });
    });
    const release = () => {
      if (!m?.held) return;
      m.held = null;
      document.documentElement.classList.remove("shapiq-logo-pulling");
      animate(m);
    };
    svg.addEventListener("pointerup", release);
    svg.addEventListener("pointercancel", release);
    svg.addEventListener("lostpointercapture", release); // e.g. the window lost the focus

    // the ripple the first time the logo is seen, the bump when it is scrolled into view quickly
    let seen = false;
    let visible = false;
    new IntersectionObserver(
      ([entry]) => {
        const appears = entry.isIntersecting && !visible;
        visible = entry.isIntersecting;
        if (!appears) return;
        const still = reducedMotion.matches || !ready() || m.held;
        if (!seen) {
          seen = true;
          if (!still) ripple(m);
        } else if (!still && Math.abs(scrollSpeed()) > FAST_SCROLL) {
          bump(m, scrollSpeed());
        }
      },
      { threshold: 0.3 },
    ).observe(svg);
  }

  // replace the landing page's logo images by copies of the sidebar's inline logo
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
    document.querySelectorAll("svg.shapiq-hero-logo").forEach(attach);
  });
})();
