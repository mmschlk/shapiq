/* The AI involvement of a page: a thin line beside the text, from level 0 (a person wrote it
   without AI) on the left to level 5 (AI wrote it and no person checked it) on the right. The line
   steps where the level of a part changes. A click on the line shows what is known about the part.
   The data is the JSON that the directive of _ext/shapiq_ai_involvement.py writes into the page. */
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

  var SVG = "http://www.w3.org/2000/svg";
  var GAP = 2.5; // the distance of two lanes, in px (--shapiq-ai-gap of the stylesheet)
  var PAD = 1.5; // the space left of lane 0
  var HEAD = 18; // the height of the label above the line
  var STEP = 16; // half the height of the curve from one lane to the next, at most
  var CLOSE = 64; // a gap between two parts below this height belongs to the part above it

  function el(name, attributes, text) {
    var node = name.indexOf("svg:") === 0
      ? document.createElementNS(SVG, name.slice(4))
      : document.createElement(name);
    Object.keys(attributes || {}).forEach(function (key) { node.setAttribute(key, attributes[key]); });
    if (text != null) node.textContent = text;
    return node;
  }

  function lane(level) { return PAD + level * GAP; }

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
    if (heading) return heading.textContent.replace(/[\s#¶\u00a0\u200b]+$/, "").trim();
    var number = node && node.querySelector(".caption-number");
    if (number) return number.textContent.replace(/[.\s]+$/, "");
    return part.target;
  }

  var page = Object.assign({ name: "The page as a whole", whole: true }, data.page);

  // the widget: the lanes, the line, the part pointed at, a button per stretch of the line, the card
  var root = el("div", { class: "shapiq-ai", "aria-label": "AI involvement of this page, part by part" });
  root.setAttribute("role", "group");
  // the tag that sticks to the top of the line: the level of the part being read
  var label = el("button", { class: "shapiq-ai-label", type: "button", "aria-expanded": "false" });
  var labelText = el("span", { class: "shapiq-ai-label-text" });
  label.appendChild(el("span", { class: "shapiq-ai-label-ai" }, "AI"));
  label.appendChild(labelText);
  var svg = el("svg:svg", { class: "shapiq-ai-svg", "aria-hidden": "true" });
  // the color of each level along the page, and a shade that grows stronger towards the text
  var defs = el("svg:defs");
  var color = el("svg:linearGradient", { id: "shapiq-ai-color", gradientUnits: "userSpaceOnUse", x1: 0, y1: 0, x2: 0 });
  var across = el("svg:linearGradient", { id: "shapiq-ai-across", gradientUnits: "userSpaceOnUse", x1: 0, y1: 0, y2: 0 });
  across.appendChild(el("svg:stop", { offset: 0, "stop-color": "#fff", "stop-opacity": 0.04 }));
  across.appendChild(el("svg:stop", { offset: 1, "stop-color": "#fff", "stop-opacity": 0.42 }));
  var fade = el("svg:mask", { id: "shapiq-ai-fade", maskUnits: "userSpaceOnUse", x: 0, y: 0 });
  var fadeRect = el("svg:rect", { x: 0, y: 0, fill: "url(#shapiq-ai-across)" });
  fade.appendChild(fadeRect);
  defs.appendChild(color);
  defs.appendChild(across);
  defs.appendChild(fade);
  var lanes = el("svg:g", { class: "shapiq-ai-lanes" });
  var shade = el("svg:path", { class: "shapiq-ai-shade", fill: "url(#shapiq-ai-color)", mask: "url(#shapiq-ai-fade)" });
  var line = el("svg:path", { class: "shapiq-ai-line", stroke: "url(#shapiq-ai-color)" });
  var dot = el("svg:circle", { class: "shapiq-ai-dot", r: 4, cx: -20, cy: -20 });
  svg.appendChild(defs);
  svg.appendChild(lanes);
  svg.appendChild(shade);
  svg.appendChild(line);
  svg.appendChild(dot);
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

  var segments = [];
  var open = null; // the segment (or the page) whose card is shown
  var pointed = null;

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

  // The line is straight along a part and curves smoothly into the lane of the next part. The
  // curve takes at most a third of either part.
  function shape(list) {
    list.forEach(function (segment, i) {
      segment.x = lane(segment.item.part.level);
      segment.y0 = segment.top;
      segment.y1 = segment.bottom;
    });
    list.forEach(function (segment, i) {
      var next = list[i + 1];
      if (!next || next.x === segment.x) return;
      var half = Math.min(STEP, (segment.bottom - segment.top) / 3, (next.bottom - next.top) / 3);
      segment.y1 = segment.bottom - half;
      next.y0 = next.top + half;
    });
  }

  function path(list) {
    var d = "";
    list.forEach(function (segment, i) {
      var previous = list[i - 1];
      if (!previous) d += "M" + segment.x + " " + segment.y0.toFixed(1);
      else {
        var bend = (segment.y0 - previous.y1) * 0.55;
        d += "C" + previous.x + " " + (previous.y1 + bend).toFixed(1) + " " + segment.x + " "
          + (segment.y0 - bend).toFixed(1) + " " + segment.x + " " + segment.y0.toFixed(1);
      }
      d += "L" + segment.x + " " + segment.y1.toFixed(1);
    });
    return d;
  }

  // where the line is at a height y: in a part's lane, or on the curve between two parts
  function at(y) {
    for (var i = 0; i < segments.length; i++) {
      var segment = segments[i];
      if (y <= segment.y1) {
        if (y >= segment.y0 || i === 0) return segment.x;
        var previous = segments[i - 1];
        var bend = (segment.y0 - previous.y1) * 0.55;
        var ys = [previous.y1, previous.y1 + bend, segment.y0 - bend, segment.y0];
        var xs = [previous.x, previous.x, segment.x, segment.x];
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
    return segments.length ? segments[segments.length - 1].x : 0;
  }

  function segmentAt(y) {
    return segments.filter(function (segment) { return segment.top <= y && y < segment.bottom; })[0]
      || segments[segments.length - 1];
  }

  // the tag names the level of the part a third down the window, and opens the page's card
  var current = null;
  function reading() {
    if (!segments.length) return;
    var y = window.innerHeight * 0.33 - article.getBoundingClientRect().top;
    var segment = segmentAt(Math.max(HEAD, y));
    var level = segment.item.part.level;
    if (current === level) return;
    current = level;
    labelText.textContent = byLevel[level].name;
    label.style.setProperty("--shapiq-ai-tag", colorOf(level));
    label.setAttribute("aria-label", "AI involvement here: level " + level + ", " + byLevel[level].name
      + ". Show the page's AI involvement.");
    label.title = "Level " + level + " \u00b7 " + byLevel[level].summary;
  }

  function colorOf(level) {
    return getComputedStyle(root).getPropertyValue("--shapiq-ai-" + level).trim() || "currentColor";
  }

  function draw() {
    var gap = parseFloat(getComputedStyle(root).getPropertyValue("--shapiq-ai-gap"));
    if (gap > 0) GAP = gap;
    var result = measure();
    segments = result.segments;
    var width = lane(top) + PAD;
    svg.setAttribute("width", width);
    svg.setAttribute("height", result.height);
    svg.setAttribute("viewBox", "0 0 " + width + " " + result.height);
    root.style.height = result.height + "px";
    while (lanes.firstChild) lanes.removeChild(lanes.firstChild);
    for (var level = 0; level <= top; level++) {
      lanes.appendChild(el("svg:line", { x1: lane(level), x2: lane(level), y1: HEAD, y2: result.height }));
    }
    shape(segments);
    var d = path(segments);
    line.setAttribute("d", d);
    if (segments.length) {
      var first = segments[0];
      var last = segments[segments.length - 1];
      shade.setAttribute("d", d + "L0 " + last.y1.toFixed(1) + "L0 " + first.y0.toFixed(1) + "Z");
    }
    color.setAttribute("y2", result.height);
    while (color.firstChild) color.removeChild(color.firstChild);
    segments.forEach(function (segment) {
      var c = colorOf(segment.item.part.level);
      [segment.y0, segment.y1].forEach(function (y) {
        color.appendChild(el("svg:stop", { offset: (y / result.height).toFixed(5), "stop-color": c }));
      });
    });
    across.setAttribute("x2", width);
    fade.setAttribute("width", width);
    fade.setAttribute("height", result.height);
    fadeRect.setAttribute("width", width);
    fadeRect.setAttribute("height", result.height);
    while (hits.firstChild) hits.removeChild(hits.firstChild);
    segments.forEach(function (segment) {
      var part = segment.item.part;
      var name = segment.item.name || part.name;
      var button = el("button", { type: "button", class: "shapiq-ai-hit", "aria-expanded": "false" });
      button.style.top = segment.top + "px";
      button.style.height = Math.max(8, segment.bottom - segment.top) + "px";
      button.setAttribute("aria-label", name + ": level " + part.level + ", " + byLevel[part.level].name + ". Show details.");
      button.title = name + " · " + part.level + " " + byLevel[part.level].name;
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
      if (again) { open = again; again.button.setAttribute("aria-expanded", "true"); }
    }
    point(pointed && segments.filter(function (s) { return s.item === pointed.item; })[0], pointed && pointedY);
    current = null;
    reading();
  }

  // The part pointed at, or the one whose card is open: a dot on the line where the pointer is, and
  // a soft tint of the part's color over the part on the page.
  var pointedY = 0;
  function point(segment, y) {
    pointed = segment || null;
    if (pointed) pointedY = y == null ? (segment.top + segment.bottom) / 2 : y;
    var show = pointed || (open && !open.whole ? open : null);
    if (!show) {
      dot.classList.remove("on");
      glow.classList.remove("on");
      return;
    }
    var level = show.item.part.level;
    var c = colorOf(level);
    var yy = pointed ? pointedY : (open.y != null ? open.y : (show.top + show.bottom) / 2);
    dot.setAttribute("cx", at(yy).toFixed(2));
    dot.setAttribute("cy", yy.toFixed(1));
    dot.style.fill = c;
    dot.classList.add("on");
    glow.style.setProperty("--shapiq-ai-glow", c);
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

  // the share of the page's height at each level
  function shares() {
    var total = 0;
    var by = {};
    segments.forEach(function (segment) {
      var size = segment.bottom - segment.top;
      total += size;
      by[segment.item.part.level] = (by[segment.item.part.level] || 0) + size;
    });
    var list = el("ul", { class: "shapiq-ai-shares" });
    levels.forEach(function (level) {
      if (!by[level.level]) return;
      var item = el("li");
      var bar = el("span", { class: "bar" });
      bar.style.width = Math.max(2, Math.round(100 * by[level.level] / total)) + "%";
      bar.style.background = colorOf(level.level);
      item.appendChild(el("span", { class: "name" }, level.level + " · " + level.name));
      var track = el("span", { class: "track" });
      track.appendChild(bar);
      item.appendChild(track);
      item.appendChild(el("span", { class: "share" }, Math.round(100 * by[level.level] / total) + "%"));
      list.appendChild(item);
    });
    return list;
  }

  function fill(part, name) {
    while (card.firstChild) card.removeChild(card.firstChild);
    var level = byLevel[part.level];
    card.style.setProperty("--shapiq-ai-card", colorOf(part.level));
    var head = el("div", { class: "shapiq-ai-card-head" });
    head.appendChild(scale(part.level));
    head.appendChild(el("span", { class: "shapiq-ai-card-level" }, "Level " + part.level + " · " + level.name));
    var shut = el("button", { type: "button", class: "shapiq-ai-card-close", "aria-label": "Close" }, "×");
    shut.addEventListener("click", function () { close(true); });
    head.appendChild(shut);
    card.appendChild(head);
    card.appendChild(el("p", { class: "shapiq-ai-card-name" }, name));
    card.appendChild(el("p", { class: "shapiq-ai-card-meaning" }, level.summary));
    if (part.whole && part.summary) card.appendChild(el("p", { class: "shapiq-ai-card-summary" }, part.summary));
    var list = el("dl");
    row(list, "Written by", part.written);
    row(list, "Models", part.models);
    row(list, "Human role", part.human);
    row(list, "Sources", part.sources);
    row(list, "Last checked", date(part.checked));
    if (list.firstChild) card.appendChild(list);
    if (part.note) card.appendChild(el("p", { class: "shapiq-ai-card-note" }, part.note));
    if (part.whole) {
      card.appendChild(el("p", { class: "shapiq-ai-card-caption" }, "Share of the page at each level"));
      card.appendChild(shares());
    }
    var more = el("a", { class: "shapiq-ai-card-more", href: data.about }, "How the levels work");
    card.appendChild(more);
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

  label.addEventListener("click", function () {
    if (open && open.whole) close();
    else showPage();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && open) close(true);
  });
  document.addEventListener("click", function (event) {
    if (open && !card.contains(event.target) && !root.contains(event.target)) close();
  });

  var waiting = false;
  function later() {
    if (waiting) return;
    waiting = true;
    requestAnimationFrame(function () { waiting = false; draw(); });
  }
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
