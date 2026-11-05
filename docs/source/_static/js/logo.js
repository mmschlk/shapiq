// The landing page's logo hangs together like a model of rings and rubber bands: hovering it makes
// it wiggle, and a player can be pulled and let go. Each player is a ring, held in place by a
// spring. Each interaction is a rubber band whose two ends are beads that slide along the rings of
// the players it links. Mounted a little stretched, a band pulls its ends onto the line through the
// centers of its players, and the thicker the band (a stronger interaction), the stiffer it is: the
// thickest holds its players together almost like a rod, the thinnest barely pulls. Each bead grips
// its place on the ring with a spring, so a pulled player drags the others along, while its beads
// turn a little towards the players they link. Beads on a ring push each other aside when they come
// closer than they would touch with a little space between them, and through their bands move
// further players. The logo as drawn is this model at rest (the logo kit places the ends so, with
// the same constants); if a logo is not, its ends stay where they are drawn on the rings. When the
// logo is first seen, a random player wiggles and the wiggle runs through the logo along the
// interactions; when it is scrolled into view quickly, it gives a little bump in the direction of
// travel. The landing page shows the logo as an image; it is replaced by a copy of the sidebar's
// inline logo (_templates/sidebar/brand.html). The sidebar's logo itself stays still: it is on every
// page and a link home, where a wiggle on every pass of the mouse would distract.
//
// An interaction is one shape, a metaball: its two end dots flow into a neck between them. The logo
// file says how it was drawn (data-metaball on the group; data-ends, data-r and data-w on each
// interaction), and the shapes are redrawn from that formula while the players move: the end dots
// are the beads, and a stretched neck narrows as if it kept its volume. The colors come from
// one attribution scale (data-scale on the group): red is positive, blue negative (data-sign), and
// the step is the strength (data-step, 1 to 5). Tension moves an element a little up its scale, so
// a stretched interaction and a pulled player look a little stronger until they are back at rest.
// The color of an interaction is spread like pigment over its shape: while its end dots deepen, a
// neck stretched thinner pales towards white and a squeezed one deepens, shaded along the
// interaction by a gradient that exists only while it moves.
// At rest the logo shows its own paths and colors, exactly as drawn. If its paths no longer follow
// the formula (a new logo kit), it stays still; if its colors no longer follow the scale, it moves
// without changing color. Either way it says why in the console.
//
// With reduced motion (macOS "Reduce motion", Windows with "Animation effects" off), the logo
// does not move by itself: no wiggle on hover, no ripple, no bump. The players follow a pulled
// one without swinging, and when it is let go, all of them glide back to their places.
//
// On a touch screen, a finger on a player pulls it; anywhere else on the logo, it scrolls the page.
(() => {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const ANCHOR = 200; // stiffness of the spring that holds a player in place (1/s²)
  const LINK = 2.5; // stiffness of an interaction per squared logo unit of its width (1/s²): 39 to 1180
  const LINK_DAMPING = 0.08; // damping of an interaction per unit of its stiffness (s): no buzz in stiff pairs
  const DAMPING = 10; // a swing or two, at rest within a second (1/s)
  const GLIDE = 0.25; // time constant of the return to rest with reduced motion (s)
  const KICK = 95; // speed of the wiggle on hover (logo units/s; the logo is 280 units wide)
  const RIPPLE = 250; // speed of the wiggle that starts the ripple (logo units/s), halved per hop
  const RIPPLE_HOP = 80; // time the ripple takes along an interaction (ms)
  const BUMP = 130; // speed of the bump when the logo is scrolled into view (logo units/s)
  const FAST_SCROLL = 1000; // scrolling faster than this bumps the logo (px/s)
  const MAX_PULL = 55; // beyond this distance, a pulled player follows the pointer reluctantly
  const FINGER = 22; // a finger reaches this far from a player's center at least (px): 44 px targets
  const PLAYER_RADIUS = 15; // larger circles are players
  const PRETENSION = 0.1; // the bands are mounted stretched by a tenth of their length
  const FRICTION = 2; // a bead slides along its ring at the force along the ring over this (units/s)
  const CONTACT = 400; // stiffness with which two beads on a ring push each other apart (1/s²)
  const SPACE = 3; // two beads on a ring push when the space between them gets smaller than this
  // how firmly a bead holds its place on its ring (a spring to the angle it is drawn at, so the logo
  // as drawn stays at rest), relative to its band's stiffness: sliding it along the ring costs GRIP
  // times as much as stretching its band as far. 0: the beads slide freely; Infinity: they stay put.
  // At 2 a pulled player drags the others almost as far as with fixed beads (97%), while its beads
  // still turn a little towards the players they link
  const GRIP = 2;

  const NECK = 0.5; // a neck stretched by a factor s narrows by s^-NECK: as if it kept its volume
  const NECK_LIMITS = [0.55, 1.15]; // but never by more than this, nor widens by more than this
  const SHADE_LINK = 1; // how far tension moves an interaction up its scale, at most (steps)
  const SHADE_PLAYER = 0.75; // the same for a pulled player
  const STRAIN = 0.3; // an interaction stretched or squeezed by this much has moved 3/4 of the way
  const PULL = 30; // a player this far from its place (logo units) has moved 3/4 of the way
  const DEEPEST = 6; // past its strongest step (5), the scale goes on a little deeper
  // steps a neck pales per unit it narrows: stretched by half, it keeps its color while its dots
  // deepen; at its narrowest it is 2.25 steps paler than its dots
  const THINNING = 5;

  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const numbers = (text) => (text.match(/-?\d*\.?\d+(?:e-?\d+)?/gi) ?? []).map(Number);
  const settings = (text) =>
    Object.fromEntries((text ?? "").split(/\s+/).filter(Boolean).map((pair) => pair.split("=")));

  // the attribution scale (as the logo kit builds it): per side, lightness runs evenly from step 1 to
  // the side's anchor (step 5), chroma grows with the distance from the neutral color, and the hue
  // drifts a little towards the light end; in OKLCH, fitted into sRGB by lowering the chroma
  const linear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const gamma = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

  function oklch(hex) {
    const [r, g, b] = [1, 3, 5].map((i) => linear(parseInt(hex.slice(i, i + 2), 16) / 255));
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    const a = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
    const bb = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
    return { L: 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, C: Math.hypot(a, bb), H: Math.atan2(bb, a) };
  }

  function srgb(L, C, H) {
    let rgb;
    for (let i = 0; i < 60; i += 1, C *= 0.97) {
      const [a, b] = [C * Math.cos(H), C * Math.sin(H)];
      const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
      const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
      const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
      rgb = [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
      ];
      if (rgb.every((v) => v >= -1e-6 && v <= 1 + 1e-6)) break;
    }
    return rgb.map((v) => Math.round(Math.min(1, Math.max(0, gamma(Math.max(0, v)))) * 255));
  }

  // the scale as a function of sign (+1 red, -1 blue) and step (any number from 0, the neutral
  // color, to DEEPEST); below step 1 it fades evenly into the neutral color
  function scale(spec) {
    const { red, blue, neutral, L1, p, dr, db } = settings(spec);
    if (![red, blue, neutral].every((hex) => /^#[0-9a-f]{6}$/i.test(hex ?? ""))) return null;
    const lightest = Number(L1);
    const white = oklch(neutral).L;
    const side = (anchor, drift) => {
      const { L: La, C: Ca, H: Ha } = oklch(anchor);
      const hue = (k) => Ha + (((Number(drift) || 0) * (5 - k)) / 4) * (Math.PI / 180);
      const chroma = (L) => Ca * Math.max(0, (white - L) / (white - La)) ** Number(p);
      return (k) => {
        if (k < 1) return srgb(white + (lightest - white) * k, chroma(lightest) * k, hue(1));
        const L = lightest + ((La - lightest) * (k - 1)) / 4;
        return srgb(L, chroma(L), hue(k));
      };
    };
    const sides = { 1: side(red, dr), [-1]: side(blue, db) };
    return (sign, k) => sides[sign](Math.min(DEEPEST, Math.max(0, k)));
  }

  // the metaball between end dots c1 and c2 (radius r, half-width w), as the logo kit draws it: a
  // cubic leaves each dot along its rim at the angle phi and runs into a waist, halfway along (a
  // curved neck) or at the length D (a straight one). The waist is scaled by `neck`. Dots no wider
  // than the interaction (a capsule) leave at a right angle.
  function metaball(c1, c2, r, w, shape, neck = 1) {
    const length = distance(c1, c2);
    const u = { x: (c2.x - c1.x) / length, y: (c2.y - c1.y) / length };
    const capsule = r <= w * 1.02;
    if (capsule && shape.mid >= 1) return null; // drawn with fillets, which are not redrawn here
    const phi = ((capsule ? 90 : shape.phi) * Math.PI) / 180;
    const [sin, cos] = [Math.sin(phi), Math.cos(phi)];
    const waist = Math.min(w, Math.max(shape.mid * w, shape.floor)) * neck;
    const straight = shape.mid >= 1 || shape.straight > 0;
    const at = (c, along, across, a, b) => ({
      x: c.x + along.x * a + across.x * b,
      y: c.y + along.y * a + across.y * b,
    });
    // one of the four curves: from the rim of dot c (pointing at the other dot along `along`) on the
    // side `s` of the axis to the waist
    const curve = (c, along, s) => {
      const across = { x: -u.y * s, y: u.x * s };
      const rim = at(c, along, across, r * cos, r * sin);
      const tangent = at({ x: 0, y: 0 }, along, across, sin, -cos);
      const reach = straight ? Math.min(shape.D * r, length / 2) : length / 2;
      const span = Math.max(reach - r * cos, 0.5);
      const waistPoint = at(c, along, across, reach, waist);
      const drop = r * sin - waist;
      const a = Math.max(0, cos > 1e-6 ? Math.min(shape.a * span, (0.9 * drop) / cos) : shape.acap * span);
      return [rim, at(rim, tangent, { x: 0, y: 0 }, a, 0), at(waistPoint, along, across, -shape.b * span, 0), waistPoint];
    };
    const back = { x: -u.x, y: -u.y };
    const [T1p, A1p, B1p, E1p] = curve(c1, u, 1);
    const [T2p, A2p, B2p, E2p] = curve(c2, back, 1);
    const [T1m, A1m, B1m, E1m] = curve(c1, u, -1);
    const [T2m, A2m, B2m, E2m] = curve(c2, back, -1);
    const pt = (q) => `${q.x.toFixed(3)},${q.y.toFixed(3)}`;
    // the arc around the far side of dot c, from p to q
    const arc = (p, q, c, along) => {
      const far = at(c, along, { x: 0, y: 0 }, -r, 0);
      const sweep = (p.x - c.x) * (far.y - c.y) - (p.y - c.y) * (far.x - c.x) > 0 ? 1 : 0;
      return `A ${r.toFixed(3)} ${r.toFixed(3)} 0 ${phi < Math.PI / 2 ? 1 : 0} ${sweep} ${pt(q)}`;
    };
    return (
      `M ${pt(T1p)} C ${pt(A1p)} ${pt(B1p)} ${pt(E1p)} ${straight ? `L ${pt(E2p)} ` : ""}` +
      `C ${pt(B2p)} ${pt(A2p)} ${pt(T2p)} ${arc(T2p, T2m, c2, back)} ` +
      `C ${pt(A2m)} ${pt(B2m)} ${pt(E2m)} ${straight ? `L ${pt(E1m)} ` : ""}` +
      `C ${pt(B1m)} ${pt(A1m)} ${pt(T1m)} ${arc(T1m, T1p, c1, u)} Z`
    );
  }

  // the center of a circle in its group's coordinates (some circles carry rotate(-90))
  function center(circle) {
    const transforms = circle.transform.baseVal;
    const x = circle.cx.baseVal.value;
    const y = circle.cy.baseVal.value;
    if (!transforms.numberOfItems) return { x, y };
    const m = transforms.consolidate().matrix;
    return { x: m.a * x + m.c * y + m.e, y: m.b * x + m.d * y + m.f };
  }

  // where a bead is: on the ring of its player, at the angle it was drawn at, turned along the ring
  function beadAt(e) {
    const angle = e.home + e.turn;
    const p = e.player;
    return { x: p.rest.x + p.x + e.radius * Math.cos(angle), y: p.rest.y + p.y + e.radius * Math.sin(angle) };
  }

  // the colorable parts of an element: its sign and strength, and the color it is drawn with
  const colorOf = (el) => ({ sign: Number(el.dataset.sign), step: Number(el.dataset.step), fill: el.getAttribute("fill") });

  function model(svg) {
    const group = svg.querySelector("circle")?.parentNode;
    if (!group) return null;
    const players = [...group.querySelectorAll(":scope > circle")]
      .filter((circle) => circle.r.baseVal.value > PLAYER_RADIUS)
      .map((circle) => {
        const rest = center(circle); // consolidates the transform: read the attribute after it
        circle.classList.add("shapiq-logo-player");
        const base = circle.getAttribute("transform") ?? "";
        return { el: circle, base, rest, r: circle.r.baseVal.value, ...colorOf(circle), x: 0, y: 0, vx: 0, vy: 0 };
      });
    if (!players.length) return null;
    const nearest = (point) =>
      players.reduce((best, p) => (distance(point, p.rest) < distance(point, best.rest) ? p : best));
    const shape = Object.fromEntries(Object.entries(settings(group.dataset.metaball)).map(([k, v]) => [k, Number(v)]));
    const interactions = [...group.querySelectorAll(":scope > path[data-ends]")].map((path) => {
      const [x1, y1, x2, y2] = numbers(path.dataset.ends);
      const ends = [{ x: x1, y: y1 }, { x: x2, y: y2 }];
      return {
        el: path,
        d: path.getAttribute("d"),
        ends,
        length: distance(ends[0], ends[1]),
        r: Number(path.dataset.r),
        w: Number(path.dataset.w),
        players: ends.map(nearest),
        ...colorOf(path),
      };
    });

    // the logo moves only if it is drawn by the formula, and changes color only if its colors are
    // on the scale: at rest it shows its own paths and colors, which then match what moves
    const drawn = interactions.every((i) => {
      const redrawn = numbers(metaball(i.ends[0], i.ends[1], i.r, i.w, shape) ?? "");
      const original = numbers(i.d);
      return redrawn.length === original.length && redrawn.every((v, k) => Math.abs(v - original[k]) < 0.002);
    });
    if (!interactions.length || !drawn) {
      console.warn("shapiq logo: its interactions are not drawn by the metaball formula of logo.js, so it stays still");
      return null;
    }
    let shade = scale(group.dataset.scale);
    const elements = [...players, ...interactions];
    const onScale = (e) => {
      const [r, g, b] = shade(e.sign, e.step);
      const hex = [1, 3, 5].map((k) => parseInt(e.fill.slice(k, k + 2), 16));
      return [r, g, b].every((v, k) => Math.abs(v - hex[k]) <= 1);
    };
    if (shade && !elements.every((e) => (e.sign === 1 || e.sign === -1) && e.step >= 1 && onScale(e))) shade = null;
    if (!shade) console.warn("shapiq logo: its colors are not on its attribution scale, so they stay as drawn");

    // the bands and their beads; at rest a band pulls with its tension, which the springs that hold
    // the players in place take up
    const beads = [];
    for (const p of players) p.hold = { x: 0, y: 0 };
    for (const i of interactions) {
      i.stiffness = LINK * (2 * i.w) ** 2;
      i.tension = PRETENSION * i.stiffness * i.length;
      i.beads = i.ends.map((end, k) => {
        const p = i.players[k];
        const home = Math.atan2(end.y - p.rest.y, end.x - p.rest.x);
        return { player: p, interaction: i, radius: distance(end, p.rest), home, turn: 0, size: i.r };
      });
      beads.push(...i.beads);
      const along = { x: (i.ends[1].x - i.ends[0].x) / i.length, y: (i.ends[1].y - i.ends[0].y) / i.length };
      for (const [k, sign] of [[0, -1], [1, 1]]) {
        i.players[k].hold.x += sign * i.tension * along.x;
        i.players[k].hold.y += sign * i.tension * along.y;
      }
    }
    const contacts = [];
    for (const [k, a] of beads.entries()) {
      for (const b of beads.slice(k + 1)) if (a.player === b.player) contacts.push({ a, b, reach: a.size + b.size + SPACE });
    }
    // the beads slide only if the logo as drawn is at rest: no bead pushed along its ring
    const sliding = beads.length > 0 && Math.max(...beadForces(beads, interactions, contacts).map(Math.abs)) < 0.01;
    if (!sliding) console.warn("shapiq logo: its interactions' ends are not at rest on the rings of logo.js, so they stay where they are drawn");
    const links = interactions.map((i) => ({ a: i.players[0], b: i.players[1] })); // the ripple's paths

    // damping that lets no player swing while one is held (with reduced motion): twice the square
    // root of the stiffest player's springs, a bound on the stiffness of the fastest swing
    // (Gershgorin); a band adds its stiffness along it and its tension over its length across it
    const springs = Math.max(
      ...players.map((p) =>
        interactions.reduce((sum, i) => sum + (i.players.includes(p) ? i.stiffness + i.tension / i.length : 0), 0),
      ),
    );
    const stiffest = ANCHOR + 2 * springs;
    const calmDamping = 2 * Math.sqrt(stiffest);
    // what holds a bead along its ring, for a bound on how fast it responds: its band's stiffness,
    // the band's tension over the ring's radius (twice, for the pull of the other end), the beads it
    // may touch (and its grip, see fastest)
    for (const e of beads) {
      const touching = contacts.filter((c) => c.a === e || c.b === e).length;
      e.firmness = e.interaction.stiffness + (2 * e.interaction.tension) / e.radius + touching * CONTACT;
    }
    const swinging = Math.max(Math.sqrt(stiffest), calmDamping + 2 * LINK_DAMPING * springs);
    return { svg, group, players, interactions, beads, contacts, links, shape, shade, sliding, calmDamping, swinging, held: null, running: false };
  }

  // the grip of the beads: GRIP, or the logo's data-grip ("fixed" or a number)
  const grip = (m) => {
    const value = m.svg.dataset.grip;
    if (value === "fixed") return Infinity;
    return value !== undefined && Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : GRIP;
  };
  // a bound on how fast a player or a bead responds (1/s), with the beads' grip as it is now
  const fastest = (m) => {
    const g = grip(m);
    if (!m.sliding || !Number.isFinite(g)) return m.swinging;
    return Math.max(m.swinging, ...m.beads.map((e) => (e.firmness + g * e.interaction.stiffness) / FRICTION));
  };

  const css = ([r, g, b]) => `rgb(${r}, ${g}, ${b})`;
  const response = (m) => {
    const strength = Number(m.svg.dataset.colorResponse ?? 1);
    return Number.isFinite(strength) ? strength : 1;
  };

  // the element's color moved `lift` steps up its scale; its own color when it hardly moved
  function paint(m, e, lift) {
    lift *= response(m);
    if (!m.shade || Math.abs(lift) < 0.01) e.el.setAttribute("fill", e.fill);
    else e.el.setAttribute("fill", css(m.shade(e.sign, e.step + lift)));
  }

  let gradients = 0; // for ids that stay unique with both logos on a page

  // an interaction whose end dots moved `lift` steps up its scale and whose neck was scaled by
  // `neck`: a gradient from dot a to dot b, the dots' color from their centers to their rims, and
  // from there eased to the neck's color halfway
  function paintInteraction(m, i, a, b, lift, neck) {
    const pale = THINNING * (1 - neck) * response(m);
    if (!m.shade || Math.abs(pale) < 0.01) {
      paint(m, i, lift);
      return;
    }
    if (!i.gradient) {
      const svgNS = "http://www.w3.org/2000/svg";
      const defs = m.svg.querySelector(":scope > defs") ?? m.svg.insertBefore(document.createElementNS(svgNS, "defs"), m.svg.firstChild);
      i.gradient = defs.appendChild(document.createElementNS(svgNS, "linearGradient"));
      i.gradient.id = `shapiq-logo-neck-${(gradients += 1)}`;
      i.gradient.setAttribute("gradientUnits", "userSpaceOnUse");
      i.stops = Array.from({ length: 11 }, () => i.gradient.appendChild(document.createElementNS(svgNS, "stop")));
    }
    for (const [key, value] of Object.entries({ x1: a.x, y1: a.y, x2: b.x, y2: b.y })) i.gradient.setAttribute(key, value);
    const end = i.step + lift * response(m);
    const waist = end - pale;
    const rim = Math.min(0.45, i.r / distance(a, b));
    const ease = (f) => f * f * (3 - 2 * f);
    // offsets and steps of one half, from the center of a dot to the middle
    const half = [[0, end], ...[0, 0.25, 0.5, 0.75, 1].map((f) => [rim + (0.5 - rim) * f, end + (waist - end) * ease(f)])];
    const stops = [...half, ...half.slice(0, -1).reverse().map(([offset, k]) => [1 - offset, k])];
    stops.forEach(([offset, k], n) => {
      i.stops[n].setAttribute("offset", offset);
      i.stops[n].setAttribute("stop-color", css(m.shade(i.sign, k)));
    });
    i.el.setAttribute("fill", `url(#${i.gradient.id})`);
  }

  function render(m) {
    for (const p of m.players) {
      p.el.setAttribute("transform", `translate(${p.x} ${p.y}) ${p.base}`);
      paint(m, p, SHADE_PLAYER * Math.tanh(Math.hypot(p.x, p.y) / PULL));
    }
    for (const i of m.interactions) {
      const [a, b] = i.beads.map(beadAt);
      const stretch = distance(a, b) / i.length;
      const neck = Math.min(NECK_LIMITS[1], Math.max(NECK_LIMITS[0], stretch ** -NECK));
      i.el.setAttribute("d", metaball(a, b, i.r, i.w, m.shape, neck));
      paintInteraction(m, i, a, b, SHADE_LINK * Math.tanh(Math.abs(stretch - 1) / STRAIN), neck);
    }
  }

  // back to the logo as drawn: its own paths and colors
  function rest(m) {
    for (const p of m.players) {
      Object.assign(p, { x: 0, y: 0, vx: 0, vy: 0 });
      p.el.setAttribute("transform", p.base ? `translate(0 0) ${p.base}` : "translate(0 0)");
      p.el.setAttribute("fill", p.fill);
    }
    for (const e of m.beads) e.turn = 0;
    for (const i of m.interactions) {
      i.el.setAttribute("d", i.d);
      i.el.setAttribute("fill", i.fill);
    }
  }

  // the forces on the beads (fx, fy; the bands' damping from their players' speeds), and for each
  // bead the part along its ring
  function beadForces(beads, interactions, contacts) {
    for (const e of beads) Object.assign(e, { at: beadAt(e), fx: 0, fy: 0 });
    for (const i of interactions) {
      const [a, b] = i.beads;
      const [p, q] = i.players;
      const dx = b.at.x - a.at.x;
      const dy = b.at.y - a.at.y;
      const now = Math.hypot(dx, dy) || 1;
      const stretching = ((q.vx - p.vx) * dx + (q.vy - p.vy) * dy) / now; // how fast its players pull it longer
      const pull = (i.tension + i.stiffness * (now - i.length) + LINK_DAMPING * i.stiffness * stretching) / now;
      a.fx += pull * dx;
      a.fy += pull * dy;
      b.fx -= pull * dx;
      b.fy -= pull * dy;
    }
    for (const { a, b, reach } of contacts) {
      const dx = b.at.x - a.at.x;
      const dy = b.at.y - a.at.y;
      const gap = Math.hypot(dx, dy);
      if (gap >= reach || gap < 1e-9) continue;
      const push = (CONTACT * (reach - gap)) / gap;
      a.fx -= push * dx;
      a.fy -= push * dy;
      b.fx += push * dx;
      b.fy += push * dy;
    }
    return beads.map((e) => -Math.sin(e.home + e.turn) * e.fx + Math.cos(e.home + e.turn) * e.fy);
  }

  function step(m, dt) {
    if (reducedMotion.matches && !m.held) {
      const keep = Math.exp(-dt / GLIDE);
      for (const p of m.players) Object.assign(p, { x: p.x * keep, y: p.y * keep, vx: 0, vy: 0 });
      for (const e of m.beads) e.turn *= keep;
      return;
    }
    const damping = reducedMotion.matches ? m.calmDamping : DAMPING;
    for (const p of m.players) {
      p.ax = -ANCHOR * p.x - damping * p.vx + p.hold.x;
      p.ay = -ANCHOR * p.y - damping * p.vy + p.hold.y;
    }
    const along = beadForces(m.beads, m.interactions, m.sliding ? m.contacts : []);
    // a bead slides as far along its ring as the force along the ring, less its grip, pushes it
    // against the friction; the ring, and with it the player, takes the whole force
    const g = grip(m);
    m.beads.forEach((e, k) => {
      if (!m.sliding || !Number.isFinite(g)) e.turn = 0;
      else e.turn += ((along[k] - g * e.interaction.stiffness * e.turn * e.radius) / FRICTION / e.radius) * dt;
      e.player.ax += e.fx;
      e.player.ay += e.fy;
    });
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
      const steps = Math.max(4, Math.ceil(dt * fastest(m)));
      for (let i = 0; i < steps; i += 1) step(m, dt / steps);
      // a tenth of a unit is a fraction of a pixel: stop the loop there and snap to rest
      const moving =
        m.players.some((p) => Math.hypot(p.x, p.y) > 0.15 || Math.hypot(p.vx, p.vy) > 1.5) ||
        m.beads.some((e) => Math.abs(e.turn) * e.radius > 0.15);
      if (!moving && !m.held) {
        rest(m);
        m.running = false;
        return;
      }
      render(m);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // a ripple from a random player: it wiggles, then the players an interaction away from it are
  // pushed away from it, then the ones two away, each half as hard as the one before
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
      if (!hop) continue; // the start, or a player no interaction reaches
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

  // the player within reach of a point, the one with the nearest edge: a pointer reaches 6 units
  // past a player's edge, a finger at least FINGER px from its center
  function playerAt(m, point, finger) {
    const least = finger ? FINGER / (m.group.getScreenCTM()?.a || 1) : 0;
    let hit = null;
    for (const p of m.players) {
      const off = distance(point, { x: p.rest.x + p.x, y: p.rest.y + p.y }) - p.r;
      if (off < Math.max(6, least - p.r) && (!hit || off < hit.off)) hit = { p, off };
    }
    return hit?.p ?? null;
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
      const player = point && playerAt(m, point, event.pointerType === "touch");
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

    // a finger on a player pulls it and does not scroll the page; elsewhere it scrolls. The browser
    // decides that at the touch's start, from the element under the finger, but interactions lie
    // over the players' edges and a finger reaches past them, so the decision is taken here
    svg.addEventListener(
      "touchstart",
      (event) => {
        const point = ready() && pointerIn(m, event.changedTouches[0]);
        if (point && playerAt(m, point, true)) event.preventDefault();
      },
      { passive: false },
    );

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
