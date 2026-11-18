/* The AI involvement of a page: one thin line beside the text. Its place comes from two inputs of
   each part, what a person did and what AI did (0 to 3 each): the more AI did and the less a
   person did, the further right the line runs, towards the text. A part that a person wrote alone
   sits on the far left, one that AI wrote without a person on the far right, and one where both
   did as much sits in the middle. The line curves where a part changes. Pointing at it shows a dot on it and tints the part on the page, a click
   shows what is known about the part, and the tag at the top names the level of the part being
   read. A reader can switch the line off. A page with ``mixer = true`` gets a panel above its
   title to try colors and views. The data is the JSON that _ext/shapiq_ai_involvement.py writes. */
(function () {
  "use strict";

  var source = document.getElementById("shapiq-ai-involvement");
  var article = document.querySelector("article[role='main']") || document.querySelector("article");
  if (!source || !article) return;
  var data;
  try {
    data = JSON.parse(source.textContent);
  } catch (error) {
    return;
  }
  var levels = data.levels;
  var top = levels[levels.length - 1].level;
  var byLevel = {};
  levels.forEach(function (level) { byLevel[level.level] = level; });
  var humanSteps = data.human || [];
  var aiSteps = data.ai || [];
  var axisTop = Math.max(1, humanSteps.length - 1, aiSteps.length - 1);

  var SVG = "http://www.w3.org/2000/svg";
  var GAP = 4; // the distance of two lanes, in px (--shapiq-ai-gap of the stylesheet)
  var PAD = 1.5; // the space left of lane 0
  var HEAD = 18; // where the line starts
  var CLOSE = 64; // a gap between two parts below this height belongs to the part above it
  var SPECTRUM = [160, 105, 70, 45, 25, 8]; // the hue of each level in the palette "spectrum"

  // what the mixer sets, and what a page without the mixer shows
  var DEFAULTS = {
    palette: "spectrum", // "spectrum" of hues, or "mono", one hue in shades
    hue: 265, // the hue of "mono"
    saturation: 0.85,
    lightness: 0.6,
    glow: 3, // the blur of the halo, in px
    width: 1.25, // the line, in px
    shade: 0.28, // the strongest shade under the line
    curve: 90, // half the height of a curve, at most, in px
  };
  var STORE = "shapiq-ai-mixer";
  var VISIBLE = "shapiq-ai-visible";

  function read(key) {
    try { return window.localStorage.getItem(key); } catch (error) { return null; }
  }
  function write(key, value) {
    try { window.localStorage.setItem(key, value); } catch (error) { /* private window */ }
  }

  var page = Object.assign({ name: "The page as a whole", whole: true }, data.page);
  var settings = Object.assign({}, DEFAULTS);
  if (page.mixer) {
    try { Object.assign(settings, JSON.parse(read(STORE) || "{}")); } catch (error) { /* keep */ }
  }
  var visible = read(VISIBLE) !== "off";

  function el(name, attributes, text) {
    var node = name.indexOf("svg:") === 0
      ? document.createElementNS(SVG, name.slice(4))
      : document.createElement(name);
    Object.keys(attributes || {}).forEach(function (key) { node.setAttribute(key, attributes[key]); });
    if (text != null) node.textContent = text;
    return node;
  }

  function date(text) {
    if (!text) return "";
    var parsed = new Date(text + "T00:00:00Z");
    if (isNaN(parsed)) return text;
    return parsed.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  }

  // the part's name: its own, or the heading or the caption number of its element
  function nameOf(part, node) {
    if (part.name) return part.name;
    var heading = node && node.querySelector(".shapiq-statement-title, h1, h2, h3, h4, h5, h6");
    if (heading) return heading.textContent.replace(/[\s#¶ ​]+$/, "").trim();
    var number = node && node.querySelector(".caption-number");
    if (number) return number.textContent.replace(/[.\s]+$/, "");
    return part.target;
  }

  // the two axes of a part: its own, or those of its level. A quoted part reports no level of its
  // own and keeps the place of the part around it (its host).
  function humanOf(part) {
    if (part.quoted) return part.host ? humanOf(part.host) : 0;
    return part.human_level != null ? part.human_level : (byLevel[part.level].human || 0);
  }
  function aiOf(part) {
    if (part.quoted) return part.host ? aiOf(part.host) : 0;
    return part.ai_level != null ? part.ai_level : (byLevel[part.level].ai || 0);
  }

  // what a part's level is called and means
  var QUOTED = { name: "Quoted", summary: "Taken from the work of others, as the sources say. How that work was written is not known, so it has no level." };
  function levelOf(part) { return part.quoted ? QUOTED : byLevel[part.level]; }
  function keyOf(part) { return part.quoted ? "quoted" : String(part.level); }

  // -- colors ------------------------------------------------------------------------------------

  function dark() {
    var theme = document.body.dataset.theme;
    if (theme === "dark") return true;
    if (theme === "light") return false;
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function hsl(h, s, l) {
    return "hsl(" + Math.round(h) + " " + Math.round(100 * Math.max(0, Math.min(1, s))) + "% "
      + Math.round(100 * Math.max(0, Math.min(1, l))) + "%)";
  }

  // the color at a place t of the line, from 0 (a person alone) to 1 (AI alone)
  function colorAt(t) {
    var l = settings.lightness + (dark() ? 0.08 : 0);
    if (settings.palette === "mono") {
      return hsl(settings.hue, settings.saturation * (0.3 + 0.7 * t), l + (1 - t) * 0.22 * (dark() ? -0.6 : 1));
    }
    return hsl(SPECTRUM[0] + (SPECTRUM[SPECTRUM.length - 1] - SPECTRUM[0]) * t, settings.saturation, l);
  }

  // the place of a part, from its two inputs
  function placeOf(part) { return (aiOf(part) - humanOf(part) + axisTop) / (2 * axisTop); }
  function colorOf(level) { return colorAt(placeOf({ level: level })); }
  function partColor(part) {
    if (part.quoted) return dark() ? "hsl(220 8% 62%)" : "hsl(220 8% 66%)";
    return colorAt(placeOf(part));
  }

  // -- the widget --------------------------------------------------------------------------------

  var root = el("div", { class: "shapiq-ai", "aria-label": "AI involvement of this page, part by part" });
  root.setAttribute("role", "group");
  // the tag that sticks to the top of the line: the level of the part being read
  var label = el("button", { class: "shapiq-ai-label", type: "button", "aria-expanded": "false" });
  var labelText = el("span", { class: "shapiq-ai-label-text" });
  label.appendChild(labelText);
  var svg = el("svg:svg", { class: "shapiq-ai-svg", "aria-hidden": "true" });
  var defs = el("svg:defs");
  var lanes = el("svg:g", { class: "shapiq-ai-lanes" });
  var tracksGroup = el("svg:g");
  var dots = el("svg:g");
  svg.appendChild(defs);
  svg.appendChild(lanes);
  svg.appendChild(tracksGroup);
  svg.appendChild(dots);
  var hits = el("div", { class: "shapiq-ai-hits" });
  var card = el("div", { class: "shapiq-ai-card", role: "dialog", hidden: "" });
  var glow = el("div", { class: "shapiq-ai-glow", "aria-hidden": "true" }); // the part pointed at, on the page
  root.appendChild(svg);
  root.appendChild(label);
  root.appendChild(hits);
  article.appendChild(root);
  article.appendChild(glow);
  article.appendChild(card);
  article.classList.add("shapiq-ai-page");

  var segments = []; // the stretches of the page, one per part
  var tracks = []; // the lines: their stretches with places, and their colors
  var open = null; // the segment (or the page) whose card is shown
  var pointed = null;
  var pointedY = 0;

  // Every part covers its element from top to bottom. Where parts overlap, the innermost one
  // counts. What no part covers is the page's.
  function measure() {
    var box = article.getBoundingClientRect();
    var height = article.offsetHeight;
    var items = [{ part: page, top: 0, bottom: height, depth: -1, order: -1 }];
    (data.parts || []).forEach(function (part, order) {
      var found = [];
      try { found = article.querySelectorAll(part.target); } catch (error) { found = []; }
      Array.prototype.forEach.call(found, function (node) {
        var rect = node.getBoundingClientRect();
        if (!rect.height) return;
        var depth = 0;
        for (var up = node; up && up !== article; up = up.parentElement) depth++;
        items.push({
          part: part, node: node, name: nameOf(part, node), order: order, depth: depth,
          top: Math.max(0, rect.top - box.top), bottom: Math.min(height, rect.bottom - box.top),
        });
      });
    });
    // a quoted part keeps the place of the innermost part around it that has a level
    items.forEach(function (item) {
      if (!item.part.quoted) return;
      var host = items[0];
      items.forEach(function (other) {
        if (other.part.quoted || other.depth >= item.depth) return;
        if (other.top <= item.top + 0.5 && item.bottom <= other.bottom + 0.5 && other.depth > host.depth) host = other;
      });
      item.part = Object.assign({}, item.part, { host: host.part });
    });
    var cuts = [];
    items.forEach(function (item) { cuts.push(item.top, item.bottom); });
    cuts = cuts.sort(function (a, b) { return a - b; }).filter(function (y, i, all) {
      return i === 0 || y - all[i - 1] > 0.5;
    });
    var found = [];
    for (var i = 0; i + 1 < cuts.length; i++) {
      var middle = (cuts[i] + cuts[i + 1]) / 2;
      var best = items[0];
      items.forEach(function (item) {
        if (item.top <= middle && middle < item.bottom
          && (item.depth > best.depth || (item.depth === best.depth && item.order > best.order))) best = item;
      });
      var last = found[found.length - 1];
      if (last && last.item === best) last.bottom = cuts[i + 1];
      else found.push({ item: best, top: cuts[i], bottom: cuts[i + 1] });
    }
    // a short stretch of the page's between two parts (the margin of two sections) is the upper part's
    for (var j = found.length - 2; j > 0; j--) {
      var gap = found[j];
      if (gap.item === items[0] && gap.bottom - gap.top < CLOSE) {
        found[j - 1].bottom = gap.bottom;
        found.splice(j, 1);
        if (found[j] && found[j].item === found[j - 1].item) {
          found[j - 1].bottom = found[j].bottom;
          found.splice(j, 1);
        }
      }
    }
    if (found.length) found[0].top = Math.max(found[0].top, HEAD);
    return { height: height, segments: found };
  }

  // The line is straight along a part and curves gently into the place of the next part. The
  // curve takes up to settings.curve px of either part, and at most 45 % of it.
  function shape(list, xOf) {
    var out = list.map(function (segment) {
      return { segment: segment, x: xOf(segment.item.part), y0: segment.top, y1: segment.bottom };
    });
    out.forEach(function (stretch, i) {
      var next = out[i + 1];
      if (!next || next.x === stretch.x) return;
      var a = stretch.segment, b = next.segment;
      var half = Math.min(settings.curve, (a.bottom - a.top) * 0.45, (b.bottom - b.top) * 0.45);
      stretch.y1 = a.bottom - half;
      next.y0 = b.top + half;
    });
    return out;
  }

  function path(list) {
    var d = "";
    list.forEach(function (stretch, i) {
      var previous = list[i - 1];
      if (!previous) d += "M" + stretch.x.toFixed(2) + " " + stretch.y0.toFixed(1);
      else {
        var bend = (stretch.y0 - previous.y1) * 0.55;
        d += "C" + previous.x.toFixed(2) + " " + (previous.y1 + bend).toFixed(1) + " " + stretch.x.toFixed(2) + " "
          + (stretch.y0 - bend).toFixed(1) + " " + stretch.x.toFixed(2) + " " + stretch.y0.toFixed(1);
      }
      d += "L" + stretch.x.toFixed(2) + " " + stretch.y1.toFixed(1);
    });
    return d;
  }

  // where a line is at a height y: in a part's place, or on the curve between two parts
  function at(list, y) {
    for (var i = 0; i < list.length; i++) {
      var stretch = list[i];
      if (y <= stretch.y1) {
        if (y >= stretch.y0 || i === 0) return stretch.x;
        var previous = list[i - 1];
        var bend = (stretch.y0 - previous.y1) * 0.55;
        var ys = [previous.y1, previous.y1 + bend, stretch.y0 - bend, stretch.y0];
        var xs = [previous.x, previous.x, stretch.x, stretch.x];
        var bez = function (p, t) {
          var u = 1 - t;
          return u * u * u * p[0] + 3 * u * u * t * p[1] + 3 * u * t * t * p[2] + t * t * t * p[3];
        };
        var lo = 0, hi = 1;
        for (var k = 0; k < 24; k++) {
          var mid = (lo + hi) / 2;
          if (bez(ys, mid) < y) lo = mid; else hi = mid;
        }
        return bez(xs, (lo + hi) / 2);
      }
    }
    return list.length ? list[list.length - 1].x : 0;
  }

  function segmentAt(y) {
    return segments.filter(function (segment) { return segment.top <= y && y < segment.bottom; })[0]
      || segments[segments.length - 1];
  }

  // the line: where each part sits, its color, and where its shade starts
  function views(width) {
    return [{ name: "line", base: 0, x: function (part) { return PAD + placeOf(part) * (width - 2 * PAD); },
      color: partColor }];
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function draw() {
    var gap = parseFloat(getComputedStyle(root).getPropertyValue("--shapiq-ai-gap"));
    if (gap > 0) GAP = gap;
    var result = measure();
    segments = result.segments;
    var height = result.height;
    var width = 2 * PAD + top * GAP;
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    root.style.height = height + "px";
    root.style.setProperty("--shapiq-ai-width", settings.width + "px");
    clear(lanes);
    clear(defs);
    clear(tracksGroup);
    clear(dots);
    // a lane for each balance of the two inputs, the middle one where both did as much
    for (var k = 0; k <= 2 * axisTop; k++) {
      var x = PAD + k * (width - 2 * PAD) / (2 * axisTop);
      lanes.appendChild(el("svg:line", { class: k === axisTop ? "center" : "", x1: x, x2: x, y1: HEAD, y2: height }));
    }
    tracks = views(width).map(function (view, index) {
      var list = shape(segments, view.x);
      var d = path(list);
      var id = "shapiq-ai-" + view.name;
      // the color along the page
      var color = el("svg:linearGradient", { id: id + "-color", gradientUnits: "userSpaceOnUse", x1: 0, y1: 0, x2: 0, y2: height });
      list.forEach(function (stretch) {
        var c = view.color(stretch.segment.item.part);
        [stretch.y0, stretch.y1].forEach(function (y) {
          color.appendChild(el("svg:stop", { offset: (y / height).toFixed(5), "stop-color": c }));
        });
      });
      // the shade, stronger away from where the line starts
      var far = width;
      var across = el("svg:linearGradient", { id: id + "-across", gradientUnits: "userSpaceOnUse", x1: view.base, y1: 0, x2: far, y2: 0 });
      across.appendChild(el("svg:stop", { offset: 0, "stop-color": "#fff", "stop-opacity": 0.02 }));
      across.appendChild(el("svg:stop", { offset: 1, "stop-color": "#fff", "stop-opacity": settings.shade }));
      var mask = el("svg:mask", { id: id + "-fade", maskUnits: "userSpaceOnUse", x: 0, y: 0, width: width, height: height });
      mask.appendChild(el("svg:rect", { x: 0, y: 0, width: width, height: height, fill: "url(#" + id + "-across)" }));
      defs.appendChild(color);
      defs.appendChild(across);
      defs.appendChild(mask);
      var group = el("svg:g", { class: "shapiq-ai-track" });
      if (list.length && settings.shade > 0) {
        var first = list[0], last = list[list.length - 1];
        group.appendChild(el("svg:path", {
          class: "shapiq-ai-shade", fill: "url(#" + id + "-color)", mask: "url(#" + id + "-fade)",
          d: d + "L" + view.base + " " + last.y1.toFixed(1) + "L" + view.base + " " + first.y0.toFixed(1) + "Z",
        }));
      }
      if (settings.glow > 0) {
        var halo = el("svg:path", { class: "shapiq-ai-halo", stroke: "url(#" + id + "-color)", d: d });
        halo.style.strokeWidth = (settings.width + 2 * settings.glow) + "px";
        halo.style.filter = "blur(" + settings.glow + "px)";
        group.appendChild(halo);
      }
      var line = el("svg:path", { class: "shapiq-ai-line", stroke: "url(#" + id + "-color)", d: d });
      line.style.strokeWidth = settings.width + "px";
      group.appendChild(line);
      tracksGroup.appendChild(group);
      var dot = el("svg:circle", { class: "shapiq-ai-dot", r: 3, cx: -20, cy: -20 });
      dots.appendChild(dot);
      return { view: view, list: list, dot: dot, index: index };
    });
    clear(hits);
    segments.forEach(function (segment) {
      var part = segment.item.part;
      var name = segment.item.name || part.name;
      var button = el("button", { type: "button", class: "shapiq-ai-hit", "aria-expanded": "false" });
      button.style.top = segment.top + "px";
      button.style.height = Math.max(8, segment.bottom - segment.top) + "px";
      button.setAttribute("aria-label", name + ": " + levelOf(part).name + ". Show details.");
      button.title = name + " · " + levelOf(part).name;
      button.addEventListener("focus", function () { if (!button.matches(":hover")) point(segment); });
      button.addEventListener("blur", function () { point(null); });
      button.addEventListener("click", function (event) {
        if (open && open.item === segment.item) close();
        else show(segment, button, event);
      });
      segment.button = button;
      hits.appendChild(button);
    });
    if (open && !open.whole) {
      var again = segments.filter(function (segment) { return segment.item === open.item; })[0];
      if (again) { again.y = open.y; open = again; again.button.setAttribute("aria-expanded", "true"); }
    }
    point(pointed && segments.filter(function (s) { return s.item === pointed.item; })[0], pointed && pointedY);
    current = null;
    reading();
  }

  // The part pointed at, or the one whose card is open: a dot on each line where the pointer is,
  // and a soft tint of the part's color over the part on the page.
  function point(segment, y) {
    pointed = segment || null;
    if (pointed) pointedY = y == null ? (segment.top + segment.bottom) / 2 : y;
    var show = pointed || (open && !open.whole ? open : null);
    if (!show || !visible) {
      tracks.forEach(function (track) { track.dot.classList.remove("on"); });
      glow.classList.remove("on");
      return;
    }
    var yy = pointed ? pointedY : (open.y != null ? open.y : (show.top + show.bottom) / 2);
    tracks.forEach(function (track) {
      var c = track.view.color(show.item.part);
      track.dot.setAttribute("cx", at(track.list, yy).toFixed(2));
      track.dot.setAttribute("cy", yy.toFixed(1));
      track.dot.style.fill = c;
      track.dot.style.color = c;
      track.dot.classList.add("on");
    });
    glow.style.setProperty("--shapiq-ai-glow", partColor(show.item.part));
    glow.style.left = root.offsetLeft + "px";
    glow.style.top = show.top + "px";
    glow.style.height = (show.bottom - show.top) + "px";
    glow.classList.add("on");
  }

  hits.addEventListener("mousemove", function (event) {
    var y = event.clientY - article.getBoundingClientRect().top;
    point(segmentAt(y), y);
  });
  hits.addEventListener("mouseleave", function () { point(null); });

  // the tag names the level of the part a third down the window
  var current = null;
  function reading() {
    if (!segments.length) return;
    if (!visible) {
      current = null;
      labelText.textContent = "off";
      label.style.removeProperty("--shapiq-ai-tag");
      label.setAttribute("aria-label", "The AI involvement line is off. Show it.");
      label.title = "Show the AI involvement of this page";
      return;
    }
    var y = window.innerHeight * 0.33 - article.getBoundingClientRect().top;
    var segment = segmentAt(Math.max(HEAD, y));
    var part = segment.item.part;
    var key = keyOf(part) + "|" + partColor(part);
    if (current === key) return;
    current = key;
    labelText.textContent = levelOf(part).name;
    label.style.setProperty("--shapiq-ai-tag", partColor(part));
    label.setAttribute("aria-label", "AI involvement here: " + levelOf(part).name
      + ". Show the page's AI involvement.");
    label.title = levelOf(part).summary;
  }

  // -- the card ----------------------------------------------------------------------------------

  function row(list, term, value) {
    if (!value || (Array.isArray(value) && !value.length)) return;
    list.appendChild(el("dt", null, term));
    list.appendChild(el("dd", null, Array.isArray(value) ? value.join(", ") : value));
  }

  function scale(level) {
    var drawing = el("svg:svg", { class: "shapiq-ai-scale", width: 6 * (top + 1), height: 10, "aria-hidden": "true" });
    for (var i = 0; i <= top; i++) {
      drawing.appendChild(el("svg:circle", { cx: 3 + 6 * i, cy: 5, r: i === level ? 2.8 : 1.5, fill: i <= level ? colorOf(i) : "", class: i === level ? "on" : "" }));
    }
    return drawing;
  }

  // the two axes of a part, as two small bars from the center
  function axes(part) {
    var h = humanOf(part), a = aiOf(part);
    var box = el("div", { class: "shapiq-ai-axes" });
    [["Person", h, humanSteps, "human"], ["AI", a, aiSteps, "ai"]].forEach(function (axis) {
      var line = el("div", { class: "shapiq-ai-axis" });
      line.appendChild(el("span", { class: "who" }, axis[0]));
      var bar = el("span", { class: "track" });
      var fillBar = el("span", { class: "bar" });
      fillBar.style.width = (100 * axis[1] / axisTop) + "%";
      fillBar.style.background = partColor(part);
      bar.appendChild(fillBar);
      line.appendChild(bar);
      var step = axis[2][axis[1]];
      line.appendChild(el("span", { class: "what" }, step ? step.name : String(axis[1])));
      if (step) line.title = step.summary;
      box.appendChild(line);
    });
    return box;
  }

  // the share of the page's height at each level
  function shares() {
    var total = 0;
    var by = {};
    segments.forEach(function (segment) {
      var size = segment.bottom - segment.top;
      total += size;
      var key = keyOf(segment.item.part);
      by[key] = (by[key] || 0) + size;
    });
    var list = el("ul", { class: "shapiq-ai-shares" });
    levels.map(function (level) { return { level: level.level }; }).concat([{ quoted: true }]).forEach(function (part) {
      var key = keyOf(part);
      if (!by[key]) return;
      var item = el("li");
      var bar = el("span", { class: "bar" });
      bar.style.width = Math.max(2, Math.round(100 * by[key] / total)) + "%";
      bar.style.background = partColor(part);
      item.appendChild(el("span", { class: "name" }, levelOf(part).name));
      var track = el("span", { class: "track" });
      track.appendChild(bar);
      item.appendChild(track);
      item.appendChild(el("span", { class: "share" }, Math.round(100 * by[key] / total) + "%"));
      list.appendChild(item);
    });
    return list;
  }

  function fill(part, name) {
    clear(card);
    var level = levelOf(part);
    card.style.setProperty("--shapiq-ai-card", partColor(part));
    var head = el("div", { class: "shapiq-ai-card-head" });
    if (!part.quoted) head.appendChild(scale(part.level));
    head.appendChild(el("span", { class: "shapiq-ai-card-level" }, level.name));
    var shut = el("button", { type: "button", class: "shapiq-ai-card-close", "aria-label": "Close" }, "×");
    shut.addEventListener("click", function () { close(true); });
    head.appendChild(shut);
    card.appendChild(head);
    card.appendChild(el("p", { class: "shapiq-ai-card-name" }, name));
    card.appendChild(el("p", { class: "shapiq-ai-card-meaning" }, level.summary));
    if (humanSteps.length && aiSteps.length && !part.quoted) card.appendChild(axes(part));
    if (part.whole && part.summary) card.appendChild(el("p", { class: "shapiq-ai-card-summary" }, part.summary));
    var list = el("dl");
    row(list, "Written by", part.written);
    row(list, "Models", part.models);
    row(list, "Person", part.human);
    row(list, "Sources", part.sources);
    row(list, "Last checked", date(part.checked));
    if (list.firstChild) card.appendChild(list);
    if (part.note) card.appendChild(el("p", { class: "shapiq-ai-card-note" }, part.note));
    if (part.whole) {
      card.appendChild(el("p", { class: "shapiq-ai-card-caption" }, "Share of the page at each level"));
      card.appendChild(shares());
    }
    var foot = el("div", { class: "shapiq-ai-card-foot" });
    foot.appendChild(el("a", { class: "shapiq-ai-card-more", href: data.about }, "How the levels work"));
    var hide = el("button", { type: "button", class: "shapiq-ai-card-hide" }, "Hide the line");
    hide.addEventListener("click", function () { close(); setVisible(false); label.focus(); });
    foot.appendChild(hide);
    card.appendChild(foot);
    card.setAttribute("aria-label", "AI involvement of " + name);
  }

  function place(y) {
    var gutter = root.getBoundingClientRect();
    var box = article.getBoundingClientRect();
    card.style.left = Math.max(0, gutter.right - box.left + 6) + "px";
    card.hidden = false;
    var height = card.offsetHeight;
    var bottom = article.offsetHeight;
    card.style.top = Math.max(0, Math.min(y - 12, bottom - height)) + "px";
  }

  function show(segment, button, event) {
    close();
    open = segment;
    button.setAttribute("aria-expanded", "true");
    fill(segment.item.part, segment.item.name || segment.item.part.name);
    var box = article.getBoundingClientRect();
    var y = event && event.clientY ? event.clientY - box.top : segment.top + 8;
    segment.y = y;
    place(Math.max(segment.top, Math.min(segment.bottom - 8, y)));
    point(pointed, pointedY);
  }

  function showPage() {
    close();
    open = { whole: true, item: { part: page } };
    label.setAttribute("aria-expanded", "true");
    fill(page, page.name);
    place(label.getBoundingClientRect().top - article.getBoundingClientRect().top);
  }

  function close(focus) {
    if (!open) return;
    var button = open.whole ? label : open.button;
    if (button) button.setAttribute("aria-expanded", "false");
    open = null;
    card.hidden = true;
    point(pointed, pointedY);
    if (focus && button) button.focus();
  }

  // -- on and off --------------------------------------------------------------------------------

  function setVisible(on) {
    visible = on;
    write(VISIBLE, on ? "on" : "off");
    root.classList.toggle("off", !on);
    if (mixerToggle) mixerToggle.checked = on;
    point(null);
    current = null;
    reading();
  }

  label.addEventListener("click", function () {
    if (!visible) { setVisible(true); return; }
    if (open && open.whole) close();
    else showPage();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && open) close(true);
  });
  document.addEventListener("click", function (event) {
    if (open && !card.contains(event.target) && !root.contains(event.target)) close();
  });

  // -- the mixer ---------------------------------------------------------------------------------

  var mixerToggle = null;
  var waiting = false;
  function later() {
    if (waiting) return;
    waiting = true;
    requestAnimationFrame(function () { waiting = false; draw(); });
  }

  function mixer() {
    var panel = el("details", { class: "shapiq-ai-mixer", open: "" });
    panel.appendChild(el("summary", null, "AI line mixer"));
    var grid = el("div", { class: "shapiq-ai-mixer-grid" });
    var inputs = {};
    function save() { write(STORE, JSON.stringify(settings)); later(); }
    function choice(key, title, options) {
      var field = el("label", { class: "shapiq-ai-mixer-field" });
      field.appendChild(el("span", { class: "name" }, title));
      var select = el("select");
      options.forEach(function (option) {
        var node = el("option", { value: option[0] }, option[1]);
        if (settings[key] === option[0]) node.selected = true;
        select.appendChild(node);
      });
      select.addEventListener("change", function () { settings[key] = select.value; sync(); save(); });
      field.appendChild(select);
      grid.appendChild(field);
      inputs[key] = { node: select, field: field };
    }
    function slider(key, title, min, max, step, format) {
      var field = el("label", { class: "shapiq-ai-mixer-field" });
      field.appendChild(el("span", { class: "name" }, title));
      var input = el("input", { type: "range", min: min, max: max, step: step });
      input.value = settings[key];
      var value = el("span", { class: "value" }, format(settings[key]));
      input.addEventListener("input", function () {
        settings[key] = parseFloat(input.value);
        value.textContent = format(settings[key]);
        sync();
        save();
      });
      field.appendChild(input);
      field.appendChild(value);
      grid.appendChild(field);
      inputs[key] = { node: input, field: field, value: value, format: format };
    }
    var hue = function (v) { return Math.round(v) + "°"; };
    var px = function (v) { return String(Math.round(v * 100) / 100) + " px"; };
    var pct = function (v) { return Math.round(100 * v) + "%"; };
    choice("palette", "Palette", [["spectrum", "Spectrum"], ["mono", "One hue"]]);
    slider("hue", "Hue", 0, 360, 1, hue);
    slider("saturation", "Intensity", 0, 1, 0.01, pct);
    slider("lightness", "Lightness", 0.3, 0.85, 0.01, pct);
    slider("glow", "Glow", 0, 10, 0.5, px);
    slider("width", "Line", 0.5, 4, 0.05, px);
    slider("shade", "Shade", 0, 0.8, 0.01, pct);
    slider("curve", "Curve", 0, 300, 5, px);
    function sync() {
      inputs.hue.field.hidden = settings.palette !== "mono";
      ["hue"].forEach(function (key) {
        inputs[key].node.style.setProperty("--swatch", hsl(settings[key], settings.saturation, settings.lightness));
      });
    }
    panel.appendChild(grid);
    var actions = el("div", { class: "shapiq-ai-mixer-actions" });
    var toggleLabel = el("label", { class: "shapiq-ai-mixer-toggle" });
    mixerToggle = el("input", { type: "checkbox" });
    mixerToggle.checked = visible;
    mixerToggle.addEventListener("change", function () { setVisible(mixerToggle.checked); });
    toggleLabel.appendChild(mixerToggle);
    toggleLabel.appendChild(el("span", null, "Line on this page"));
    actions.appendChild(toggleLabel);
    var reset = el("button", { type: "button" }, "Reset");
    reset.addEventListener("click", function () {
      settings = Object.assign({}, DEFAULTS);
      Object.keys(inputs).forEach(function (key) {
        inputs[key].node.value = settings[key];
        if (inputs[key].value) inputs[key].value.textContent = inputs[key].format(settings[key]);
      });
      sync();
      save();
    });
    actions.appendChild(reset);
    var copy = el("button", { type: "button" }, "Copy settings");
    var out = el("code", { class: "shapiq-ai-mixer-out" });
    copy.addEventListener("click", function () {
      var text = JSON.stringify(settings);
      out.textContent = text;
      try { navigator.clipboard.writeText(text); copy.textContent = "Copied"; } catch (error) { /* shown below */ }
      setTimeout(function () { copy.textContent = "Copy settings"; }, 1500);
    });
    actions.appendChild(copy);
    panel.appendChild(actions);
    panel.appendChild(out);
    sync();
    return panel;
  }

  if (page.mixer) article.insertBefore(mixer(), article.firstChild);

  root.classList.toggle("off", !visible);
  draw();
  if (window.ResizeObserver) new ResizeObserver(later).observe(article);
  window.addEventListener("load", later);
  window.addEventListener("resize", later);
  var scrolling = false;
  window.addEventListener("scroll", function () {
    if (scrolling) return;
    scrolling = true;
    requestAnimationFrame(function () { scrolling = false; reading(); });
  }, { passive: true });
  // the colors follow the theme
  new MutationObserver(later).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
  if (window.matchMedia) {
    var scheme = window.matchMedia("(prefers-color-scheme: dark)");
    if (scheme.addEventListener) scheme.addEventListener("change", later);
  }
})();
