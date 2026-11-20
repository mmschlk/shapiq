/* Talks (talks/, _ext/shapiq_talks.py): a page of slides, read as a handout or presented as a deck.

   On the page, every slide shows all its steps, with its notes below it. "Present" (or the key P)
   moves the slides into a deck that fills the window and shows one slide at a time:

     →, Space, Page Down    the next step, or the next slide      (a presenter's clicker sends Page Down)
     ←, Backspace, Page Up  the step before, or the slide before
     Home, End              the first and the last slide
     N                      the notes of the slide, on this screen
     F                      full screen
     Esc                    back to the page

   On a phone (a window up to 720 px wide), the deck scrolls instead: the slides follow each other,
   each at least one screen tall, and the scroll snaps softly to the start of each. A slide that is
   taller than the screen, as most demos are, scrolls through. Every step of a slide shows at once,
   and a thin line at the top tells how far the talk is. Turning the phone switches between both.

   The address follows the slide (#slide-3). A slide that builds up in steps tells its parts and
   figures with the class shapiq-later (a part of a later step), its data-talk-step, and the event
   "shapiq-talk-step" (detail: step, steps, presenting), which bubbles. The document hears
   "shapiq-talk-slide" (detail: slide, index) whenever another slide is shown.

   "PDF" (on the page, and in the corner of the deck) lays the slides out for print, one slide a page
   of 16:9, and opens the browser's print dialog, where "Save as PDF" makes the slides a PDF. Every
   slide is laid out at 1600 x 900 px, the size of its page, as the deck shows it in a window of that
   size, at its last step, in the light theme, without notes, the corner or
   anything that moves. A slide with a demo or a figure says where it lives, as a link at its foot.
   The address ?print lays the slides out so without the dialog (to look at, or for a test), and
   html[data-talk-print="ready"] tells when all of it has settled. While laid out for print, the root
   has the class shapiq-talk-print and the slides are in .shapiq-deck.shapiq-deck--print, as in a
   deck shown one slide at a time. The document hears "shapiq-talk-print" (detail: printing), for the
   demos and figures to show what reads best on paper. Ctrl+P on the page prints it as it is. */
(function () {
  "use strict";

  const slides = Array.from(document.querySelectorAll("section.shapiq-slide"));
  if (!slides.length) return;
  const root = document.documentElement;
  root.classList.add("shapiq-talk");

  const store = {
    get(key) {
      try { return window.sessionStorage.getItem(key); } catch (error) { return null; }
    },
    set(key, value) {
      try {
        if (value == null) window.sessionStorage.removeItem(key);
        else window.sessionStorage.setItem(key, value);
      } catch (error) { /* a private window: the deck still works, it is only not remembered */ }
    },
  };
  const PRESENTING = "shapiq-talk-presenting:" + location.pathname;
  // a phone scrolls through the slides, a larger screen shows one at a time
  const phone = window.matchMedia("(max-width: 720px)");
  const scrolling = () => Boolean(deck) && deck.classList.contains("shapiq-deck--scroll");

  const ICONS = {
    play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>',
    prev: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 18l-6-6 6-6"/></svg>',
    next: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18l6-6-6-6"/></svg>',
    notes: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16M4 10h16M4 15h10M4 20h7"/></svg>',
    theme: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor"/></svg>',
    full: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    pdf: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 13h6M9 17h4"/></svg>',
  };

  const make = (tag, className, html) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (html != null) element.innerHTML = html;
    return element;
  };

  // ---- steps -----------------------------------------------------------------------------------

  const fromStep = (element) => {
    const match = /(?:^|\s)shapiq-from-(\d+)(?:\s|$)/.exec(element.className);
    return match ? Number(match[1]) : 0;
  };
  const parts = (slide) => slide.querySelectorAll('[class*="shapiq-from-"]');

  for (const slide of slides) {
    slide.dataset.total = String(slides.length);
    let last = 0;
    for (const part of parts(slide)) last = Math.max(last, fromStep(part));
    slide.shapiqSteps = Math.max(Number(slide.dataset.steps) || 1, last + 1);
  }

  function setStep(slide, step, presenting) {
    step = Math.max(0, Math.min(slide.shapiqSteps - 1, step));
    slide.shapiqStep = step;
    slide.dataset.talkStep = String(step);
    for (const part of parts(slide)) part.classList.toggle("shapiq-later", fromStep(part) > step);
    slide.dispatchEvent(new CustomEvent("shapiq-talk-step", {
      bubbles: true,
      detail: { step, steps: slide.shapiqSteps, presenting },
    }));
  }

  // on the page, every slide shows all of it
  for (const slide of slides) setStep(slide, slide.shapiqSteps - 1, false);

  // ---- on the page -----------------------------------------------------------------------------

  // the talk's event and recording (the shapiq-talk directive), if it names them
  const info = document.querySelector(".shapiq-talk-info");
  const recording = info ? info.dataset.recording : "";

  const bar = make("div", "shapiq-talk-bar");
  const start = make("button", "shapiq-talk-present", ICONS.play + "<span>Present</span>");
  start.type = "button";
  start.addEventListener("click", () => present(0));
  const pdf = make("button", "shapiq-talk-pdf", ICONS.pdf + "<span>PDF</span>");
  pdf.type = "button";
  pdf.title = "The slides as a PDF, one slide a page";
  pdf.addEventListener("click", () => printSlides());
  const hint = make("p");
  const hints = () => {
    hint.innerHTML = phone.matches
      ? "One slide per screen. Scroll from slide to slide."
      : 'One slide at a time. Press <kbd class="shapiq-talk-key">P</kbd> to start, ' +
        'the arrow keys to move, and <kbd class="shapiq-talk-key">Esc</kbd> to come back here.';
  };
  hints();
  bar.append(start, pdf, hint);
  if (info && (info.dataset.event || recording)) {
    const about = make("p", "shapiq-talk-about");
    if (info.dataset.event) about.append(info.dataset.event);
    if (recording) {
      if (info.dataset.event) about.append(" · ");
      const link = make("a", "", "Recording");
      link.href = recording;
      about.append(link);
    }
    bar.append(about);
  }
  slides[0].before(bar);

  slides.forEach((slide, index) => {
    const here = make("button", "shapiq-slide-here", "Present from here");
    here.type = "button";
    here.addEventListener("click", () => present(index));
    slide.prepend(here);
  });

  // ---- for readers: BibTeX to copy, and the recording --------------------------------------------

  // a reference's BibTeX goes to the clipboard, or is selected where the clipboard is refused
  for (const copy of document.querySelectorAll(".shapiq-copy-bibtex")) {
    copy.addEventListener("click", () => {
      const source = copy.nextElementSibling;
      const done = (text) => {
        copy.textContent = text;
        setTimeout(() => { copy.textContent = "Copy BibTeX"; }, 1600);
      };
      const select = () => {
        source.hidden = false;
        const range = document.createRange();
        range.selectNodeContents(source);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        done("Selected, copy it");
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(source.textContent).then(() => done("Copied"), select);
      } else select();
    });
  }

  // where a slide starts in the recording: a link once the talk names its recording
  if (recording) {
    for (const slot of document.querySelectorAll(".shapiq-slide-recording")) {
      const time = slot.querySelector(".shapiq-slide-time");
      const link = make("a", "shapiq-slide-time", time.textContent);
      link.href = recording + (recording.includes("?") ? "&" : "?") + "t=" + slot.dataset.seconds;
      time.replaceWith(link);
    }
  } else {
    for (const slot of document.querySelectorAll(".shapiq-slide-recording")) slot.hidden = true;
  }

  // ---- the deck --------------------------------------------------------------------------------

  let deck = null;
  let current = 0;
  let marks = [];
  let count, progress, notes, notesButton, fullButton;
  let pointerTimer = 0;

  function button(icon, label, action) {
    const element = make("button", "", ICONS[icon]);
    element.type = "button";
    element.dataset.action = icon;
    element.title = label;
    element.setAttribute("aria-label", label);
    element.addEventListener("click", (event) => {
      event.stopPropagation();
      action();
    });
    return element;
  }

  function present(index, step) {
    if (deck) {
      go(index, step || 0);
      return;
    }
    deck = make("div", "shapiq-deck");
    deck.tabIndex = -1;
    deck.setAttribute("role", "region");
    deck.setAttribute("aria-roledescription", "slide deck");
    deck.setAttribute("aria-label", document.title);
    // the slides move into the deck, and a mark keeps the place of each on the page
    marks = slides.map((slide) => {
      const mark = document.createComment(" slide ");
      slide.before(mark);
      deck.append(slide);
      return mark;
    });

    const corner = make("div", "shapiq-deck-bar");
    count = make("span", "shapiq-deck-count");
    count.setAttribute("aria-live", "polite");
    notesButton = button("notes", "Notes (N)", toggleNotes);
    notesButton.setAttribute("aria-pressed", "false");
    fullButton = button("full", "Full screen (F)", toggleFull);
    fullButton.setAttribute("aria-pressed", "false");
    corner.append(
      button("prev", "Back (←)", back),
      count,
      button("next", "Next (→)", forward),
      notesButton,
      button("theme", "Light or dark", toggleTheme),
      fullButton,
      button("pdf", "The slides as a PDF", printSlides),
      button("close", "Back to the page (Esc)", stop),
    );
    progress = make("div", "shapiq-deck-progress", "<span></span>");
    notes = make("aside", "shapiq-deck-notes");
    notes.hidden = true;
    deck.append(corner, progress, notes);
    deck.addEventListener("pointerup", keepKeys);
    deck.addEventListener("pointerdown", swipeStart);
    deck.addEventListener("pointerup", swipeEnd);
    deck.addEventListener("pointermove", showCorner);
    deck.addEventListener("scroll", followScroll, { passive: true });

    document.body.append(deck);
    root.classList.add("shapiq-presenting");
    store.set(PRESENTING, "1");
    current = index;
    layout(step || 0);
    deck.focus({ preventScroll: true });
    // the demos measure their room again
    window.dispatchEvent(new Event("resize"));
  }

  function stop() {
    if (!deck) return;
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    slides.forEach((slide, index) => {
      marks[index].replaceWith(slide);
      slide.classList.remove("shapiq-current");
      slide.removeAttribute("aria-hidden");
      setStep(slide, slide.shapiqSteps - 1, false);
    });
    deck.remove();
    deck = null;
    marks = [];
    root.classList.remove("shapiq-presenting");
    store.set(PRESENTING, null);
    window.dispatchEvent(new Event("resize"));
    slides[current].scrollIntoView({ block: "start" });
  }

  // one slide at a time, or all of them to scroll through, by the width of the window
  function layout(step) {
    const scroll = phone.matches;
    deck.classList.toggle("shapiq-deck--scroll", scroll);
    const index = current;
    if (scroll) {
      slides.forEach((slide) => {
        slide.classList.remove("shapiq-current");
        slide.removeAttribute("aria-hidden");
        setStep(slide, slide.shapiqSteps - 1, true);
      });
      slides[index].scrollIntoView({ block: "start" });
      show(index);
    } else {
      slides.forEach((slide, other) => setStep(slide, other < index ? slide.shapiqSteps - 1 : 0, true));
      slides[index].classList.remove("shapiq-current");
      go(index, step);
    }
    window.dispatchEvent(new Event("resize"));
  }

  phone.addEventListener("change", () => {
    hints();
    if (deck) layout(0);
  });
  window.addEventListener("resize", () => {
    if (deck || paper) fit();
  });

  // what the corner, the line and the notes say about the slide that is shown
  function show(index) {
    const slide = slides[index];
    const changed = index !== current || !count.textContent;
    current = index;
    count.textContent = `${index + 1} / ${slides.length}`;
    if (!changed && scrolling()) return;
    const source = slide.querySelector(":scope > .shapiq-slide-notes");
    notes.innerHTML = source ? source.innerHTML : '<p class="shapiq-deck-notes-empty">No notes for this slide.</p>';
    try {
      history.replaceState(null, "", "#" + slide.id);
    } catch (error) { /* a page in a frame may not change its address */ }
    document.dispatchEvent(new CustomEvent("shapiq-talk-slide", { detail: { slide, index } }));
  }

  // scrolled: the slide shown is the last one whose top has passed the upper third of the screen,
  // and the line at the top grows with the scroll
  let scrollFrame = 0;
  function followScroll() {
    if (!scrolling() || scrollFrame) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = 0;
      if (!deck) return;
      const line = deck.clientHeight / 3;
      let index = 0;
      slides.forEach((slide, other) => {
        if (slide.getBoundingClientRect().top <= line) index = other;
      });
      const room = deck.scrollHeight - deck.clientHeight;
      progress.firstChild.style.transform = `scaleX(${room > 0 ? deck.scrollTop / room : 1})`;
      if (index !== current) show(index);
    });
  }

  function go(index, step) {
    index = Math.max(0, Math.min(slides.length - 1, index));
    if (scrolling()) {
      slides[index].scrollIntoView({ block: "start", behavior: "smooth" });
      return;
    }
    const slide = slides[index];
    const moved = index !== current || !slide.classList.contains("shapiq-current");
    if (moved) {
      slides.forEach((other) => {
        const on = other === slide;
        other.classList.toggle("shapiq-current", on);
        if (on) other.removeAttribute("aria-hidden");
        else other.setAttribute("aria-hidden", "true");
      });
      slide.scrollTop = 0;
    }
    setStep(slide, step, true);
    const done = (index + (slide.shapiqStep + 1) / slide.shapiqSteps) / slides.length;
    progress.firstChild.style.transform = `scaleX(${done})`;
    if (moved) {
      show(index);
      watch(slide);
    }
  }

  // The safe area of a slide is its padding: what it shows stays above the line and the corner.
  // A body taller than its room (a demo) is scaled down about its top, to two thirds at most, and
  // scrolls only beyond that. Its size is watched, as a demo can grow.
  const MIN_SCALE = 0.66;
  const sizes = typeof ResizeObserver === "function" ? new ResizeObserver(() => fit()) : null;
  function watch(slide) {
    slides.forEach((other) => {
      if (other !== slide) unfit(other);
    });
    if (sizes) {
      sizes.disconnect();
      const body = slide.querySelector(":scope > .shapiq-slide-body");
      if (body) sizes.observe(body);
    }
    fit();
  }

  function unfit(slide) {
    const body = slide.querySelector(":scope > .shapiq-slide-body");
    if (!body) return;
    body.style.transform = "";
    body.style.marginBottom = "";
    delete body.dataset.fit;
  }

  function fit() {
    if (paper) {
      // on paper, every slide, and as far down as it takes: nothing is cut off
      slides.forEach((slide) => scaleDown(slide, 0));
      return;
    }
    if (!deck || scrolling()) {
      slides.forEach(unfit);
      return;
    }
    scaleDown(slides[current], MIN_SCALE);
  }

  function scaleDown(slide, least) {
    const body = slide.querySelector(":scope > .shapiq-slide-body");
    if (!body) return;
    const style = getComputedStyle(slide);
    const title = slide.querySelector(":scope > .shapiq-slide-title");
    const above = title ? title.offsetHeight + parseFloat(getComputedStyle(title).marginBottom) : 0;
    const room = slide.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom) - above;
    // (measured unscaled: the body grows into the room its negative margin leaves, so that, measured
    // scaled, a body once scaled down stays as small, even when what it holds has shrunk since)
    const [transform, margin] = [body.style.transform, body.style.marginBottom];
    body.style.transform = "";
    body.style.marginBottom = "";
    const height = body.offsetHeight;
    body.style.transform = transform;
    body.style.marginBottom = margin;
    const scale = height > room + 1 ? Math.max(least, room / height) : 1;
    const value = scale < 1 ? `scale(${scale.toFixed(4)})` : "";
    if (body.style.transform === value) return;
    body.style.transform = value;
    body.style.marginBottom = scale < 1 ? `${-(height * (1 - scale)).toFixed(1)}px` : "";
    if (scale < 1) body.dataset.fit = scale.toFixed(2);
    else delete body.dataset.fit;
  }

  function forward() {
    if (scrolling()) return go(current + 1);
    const slide = slides[current];
    if (slide.shapiqStep < slide.shapiqSteps - 1) go(current, slide.shapiqStep + 1);
    else if (current < slides.length - 1) {
      setStep(slides[current + 1], 0, true);
      go(current + 1, 0);
    }
  }

  function back() {
    if (scrolling()) return go(current - 1);
    const slide = slides[current];
    if (slide.shapiqStep > 0) go(current, slide.shapiqStep - 1);
    else if (current > 0) go(current - 1, slides[current - 1].shapiqSteps - 1);
  }

  function toggleNotes() {
    notes.hidden = !notes.hidden;
    notesButton.setAttribute("aria-pressed", String(!notes.hidden));
  }

  function toggleFull() {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (root.requestFullscreen) root.requestFullscreen().catch(() => {});
  }

  document.addEventListener("fullscreenchange", () => {
    if (fullButton) fullButton.setAttribute("aria-pressed", String(Boolean(document.fullscreenElement)));
  });

  function toggleTheme() {
    // Furo's own button, which cycles through auto, light and dark and remembers the choice
    const furo = document.querySelector(".theme-toggle");
    if (furo) furo.click();
  }

  function showCorner() {
    deck.classList.add("shapiq-pointer-moved");
    clearTimeout(pointerTimer);
    pointerTimer = setTimeout(() => deck && deck.classList.remove("shapiq-pointer-moved"), 1800);
  }

  // A mouse click never moves through the talk: the demos are clicked all the time. After a click on
  // a button or a slider of a demo, the keys go back to the deck, so that → and Space move on.
  function keepKeys(event) {
    if (event.pointerType === "touch") return;
    const focused = document.activeElement;
    if (focused && focused !== deck && deck.contains(focused) && control(focused) && !typing(focused)) {
      deck.focus({ preventScroll: true });
    }
  }

  // a swipe on the slide itself moves on (a tablet), not one on its text, a demo or a figure
  const onBackground = (target) => target.classList &&
    (target.classList.contains("shapiq-slide") || target.classList.contains("shapiq-slide-body"));

  let swipe = null;
  function swipeStart(event) {
    swipe = event.pointerType === "touch" && !scrolling() && onBackground(event.target) ? { x: event.clientX, y: event.clientY } : null;
  }
  function swipeEnd(event) {
    if (!swipe) return;
    const dx = event.clientX - swipe.x;
    const dy = event.clientY - swipe.y;
    swipe = null;
    if (Math.abs(dx) > 50 && Math.abs(dx) > 1.5 * Math.abs(dy)) {
      event.preventDefault();
      if (dx < 0) forward();
      else back();
    }
  }

  // ---- on paper --------------------------------------------------------------------------------

  // the size of a page, and of every slide laid out for print: as the deck at 1440 x 900 to 1920 x 1080
  const PAGE = { width: 1600, height: 900 };
  let paper = null;
  let paperMarks = [];
  let before = null; // what the page was before: its theme, the slide presented, where it was scrolled
  let printing = false;
  const pages = typeof ResizeObserver === "function" ? new ResizeObserver(() => fit()) : null;

  // The deck's sizes follow the window (vw, vh). On paper, the window is the page: every declaration
  // that uses them is copied, the page's sizes in their place, into a sheet that is on while the
  // slides are laid out for print. So they are laid out on the screen as they print, whatever the
  // window, and the demos measure on the screen the room they get on paper.
  const VIEWPORT = /(-?(?:\d+\.?\d*|\.\d+))(s|l|d)?(vw|vh|vmin|vmax)\b/g;
  function pageSheet() {
    const units = { vw: PAGE.width / 100, vh: PAGE.height / 100 };
    units.vmin = Math.min(units.vw, units.vh);
    units.vmax = Math.max(units.vw, units.vh);
    const swap = (text) => text.replace(VIEWPORT, (_, value, kind, unit) => `${Number((value * units[unit]).toFixed(3))}px`);
    const copy = (rules) => {
      let css = "";
      for (const rule of rules) {
        if (rule.selectorText != null && rule.style) {
          // (only what uses them: the rest of the rule may be overruled by a later one, as it is)
          let text = "";
          for (const name of rule.style) {
            const value = rule.style.getPropertyValue(name);
            VIEWPORT.lastIndex = 0;
            if (!VIEWPORT.test(value)) continue;
            const priority = rule.style.getPropertyPriority(name);
            text += `${name}: ${swap(value)}${priority ? ` !${priority}` : ""}; `;
          }
          if (text) css += `${rule.selectorText}{${text}}\n`;
        } else if (rule.cssRules && /^@(media|supports|container|layer)\b/.test(rule.cssText)) {
          const inner = copy(rule.cssRules);
          if (inner) css += `${rule.cssText.slice(0, rule.cssText.indexOf("{"))}{\n${inner}}\n`;
        }
      }
      return css;
    };
    let css = "";
    for (const sheet of document.styleSheets) {
      let rules = null;
      try {
        rules = sheet.cssRules;
      } catch (error) { /* a sheet of another origin (fonts): nothing of the deck */ }
      if (!rules || sheet.disabled) continue;
      const inner = copy(rules);
      const media = sheet.media && sheet.media.mediaText;
      if (inner) css += media ? `@media ${media}{\n${inner}}\n` : inner;
    }
    const style = make("style", "shapiq-talk-page-units");
    style.textContent = `@page { size: ${PAGE.width}px ${PAGE.height}px; margin: 0; }\n${css}`;
    return style;
  }

  // the foot of a slide on paper: its number, and where its demo or figure lives, as links
  const canonical = document.querySelector('link[rel="canonical"]');
  const address = new URL(canonical ? canonical.href : location.href);
  address.search = "";
  address.hash = "";
  if (/^https?:$/.test(address.protocol)) address.pathname = address.pathname.replace(/\/index\.html$/, "/");
  const shown = (url) => url.replace(/^https?:\/\//, "");
  function footOf(slide, index) {
    const foot = make("div", "shapiq-slide-foot");
    const live = slide.querySelector(":scope > .shapiq-slide-body [data-root]");
    if (live) {
      const links = make("p", "shapiq-slide-live");
      const link = (label, url) => {
        const anchor = make("a");
        anchor.href = url;
        anchor.textContent = shown(url);
        links.append(label + " ", anchor);
      };
      link("Live", `${address.href}?present#${slide.id || `slide-${index + 1}`}`);
      const seen = new Set();
      for (const demo of slide.querySelectorAll("[data-page]")) {
        const url = new URL(demo.dataset.page, location.href);
        url.search = "";
        url.hash = "";
        if (/^https?:$/.test(url.protocol)) url.pathname = url.pathname.replace(/\/index\.html$/, "/");
        if (seen.has(url.href)) continue;
        seen.add(url.href);
        links.append(" · ");
        link("Demo", url.href);
      }
      foot.append(links);
    }
    foot.append(make("span", "shapiq-slide-page", `${index + 1} / ${slides.length}`));
    return foot;
  }

  // the slides, laid out for print: each a page, at its last step, in the light theme
  function layOut() {
    if (paper) return;
    const presented = deck ? current : -1;
    if (deck) stop();
    before = { theme: document.body.dataset.theme, presented, scroll: window.scrollY };
    document.body.dataset.theme = "light";
    root.classList.add("shapiq-talk-print");
    delete root.dataset.talkPrint;
    document.head.append(pageSheet());
    paper = make("div", "shapiq-deck shapiq-deck--print");
    paper.setAttribute("aria-label", document.title);
    document.body.append(paper);
    paperMarks = slides.map((slide, index) => {
      const mark = document.createComment(" slide ");
      slide.before(mark);
      paper.append(slide);
      slide.append(footOf(slide, index));
      return mark;
    });
    for (const image of paper.querySelectorAll('img[loading="lazy"]')) image.loading = "eager";
    for (const slide of slides) setStep(slide, slide.shapiqSteps - 1, true);
    document.dispatchEvent(new CustomEvent("shapiq-talk-print", { detail: { printing: true } }));
    window.dispatchEvent(new Event("resize"));
    if (pages) {
      for (const slide of slides) {
        const body = slide.querySelector(":scope > .shapiq-slide-body");
        if (body) pages.observe(body);
      }
    }
    fit();
  }

  // back to the page (or the deck) as it was
  function unprint() {
    if (!paper) return;
    if (pages) pages.disconnect();
    slides.forEach((slide, index) => {
      const foot = slide.querySelector(":scope > .shapiq-slide-foot");
      if (foot) foot.remove();
      paperMarks[index].replaceWith(slide);
      unfit(slide);
      setStep(slide, slide.shapiqSteps - 1, false);
    });
    paper.remove();
    paper = null;
    paperMarks = [];
    for (const style of document.querySelectorAll("style.shapiq-talk-page-units")) style.remove();
    root.classList.remove("shapiq-talk-print");
    delete root.dataset.talkPrint;
    if (before.theme == null) delete document.body.dataset.theme;
    else document.body.dataset.theme = before.theme;
    document.dispatchEvent(new CustomEvent("shapiq-talk-print", { detail: { printing: false } }));
    window.dispatchEvent(new Event("resize"));
    if (before.presented >= 0) present(before.presented);
    else window.scrollTo(0, before.scroll);
  }

  const later = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const frame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

  // until the formulas are typeset, the fonts and the pictures loaded, and the demos started and
  // laid out anew (10 s at most)
  async function settle() {
    const until = performance.now() + 10000;
    const left = () => Math.max(0, until - performance.now());
    if (paper.querySelector(".math, mjx-container")) {
      while (!(window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) && left()) await later(100);
      // (every typesetting is chained to the promise, which it replaces)
      for (let last = null; window.MathJax && window.MathJax.startup && window.MathJax.startup.promise !== last && left(); ) {
        last = window.MathJax.startup.promise;
        await Promise.race([last.catch(() => {}), later(left())]);
      }
    }
    if (document.fonts) await Promise.race([document.fonts.ready, later(left())]);
    await Promise.race([
      Promise.all(Array.from(paper.querySelectorAll("img"), (image) => (image.decode ? image.decode().catch(() => {}) : null))),
      later(left()),
    ]);
    // (a demo's container: <name>-demo, which says <name>-demo-ready or <name>-demo-failed once started)
    const starting = () => Array.from(paper.querySelectorAll("[data-root]")).some((demo) => {
      const names = Array.from(demo.classList);
      return names.some((name) => /-demo$/.test(name)) && !names.some((name) => /-demo-(ready|failed)$/.test(name));
    });
    while (paper && starting() && left()) await later(100);
    await later(800);
    await frame();
    await frame();
    if (!paper) return;
    fit();
    await frame();
    root.dataset.talkPrint = "ready";
  }

  // "PDF": the slides laid out for print, and the browser's print dialog once they have settled
  async function printSlides() {
    if (printing) return;
    printing = true;
    layOut();
    await settle();
    if (!paper) {
      printing = false;
      return;
    }
    window.addEventListener("afterprint", () => {
      printing = false;
      unprint();
    }, { once: true });
    window.print();
  }

  // ---- keys ------------------------------------------------------------------------------------

  const typing = (target) => target.closest &&
    target.closest('textarea, select, [contenteditable=""], [contenteditable="true"], input:not([type]), ' +
      'input[type="text"], input[type="search"], input[type="number"]');
  // controls that use the arrow keys or Space themselves: sliders, tabs, buttons
  const control = (target) => target.closest &&
    target.closest('input, button, [role="slider"], [role="tab"], [role="radio"], [role="option"], [role="button"]');

  document.addEventListener("keydown", (event) => {
    // in the deck, the browser's print (Ctrl+P, ⌘P) prints the slides, as "PDF" does
    if (deck && !scrolling() && (event.ctrlKey || event.metaKey) && !event.altKey && (event.key === "p" || event.key === "P")) {
      event.preventDefault();
      printSlides();
      return;
    }
    if (paper) {
      if (event.key === "Escape" && !printing) unprint();
      return;
    }
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
    if (typing(event.target)) return;
    const key = event.key;
    if (!deck) {
      if (key === "p" || key === "P") {
        event.preventDefault();
        present(Math.max(0, slides.findIndex((slide) => slide.getBoundingClientRect().bottom > 80)));
      }
      return;
    }
    const inControl = control(event.target);
    let action = null;
    if (key === "PageDown") action = forward;
    else if (key === "PageUp") action = back;
    else if ((key === "ArrowRight" || key === " ") && !inControl) action = forward;
    else if ((key === "ArrowLeft" || key === "Backspace") && !inControl) action = back;
    else if (key === "Home" && !inControl) action = () => go(0, 0);
    else if (key === "End" && !inControl) action = () => go(slides.length - 1, slides[slides.length - 1].shapiqSteps - 1);
    else if (key === "n" || key === "N") action = toggleNotes;
    else if (key === "f" || key === "F") action = toggleFull;
    else if (key === "Escape" && !document.fullscreenElement) action = stop;
    if (action) {
      event.preventDefault();
      action();
      // a clicker's key leaves a control that kept the focus
      if (inControl && (key === "PageDown" || key === "PageUp")) deck && deck.focus({ preventScroll: true });
    }
  });

  // ---- where to start --------------------------------------------------------------------------

  const fromHash = () => {
    const match = /^#slide-(\d+)$/.exec(location.hash);
    return match ? Math.min(slides.length, Number(match[1])) - 1 : -1;
  };
  const asked = (name) => new RegExp(`(?:^|[?&])${name}(?:[=&]|$)`).test(location.search.slice(1));
  if (asked("print")) {
    layOut();
    settle();
  } else if (asked("present") || store.get(PRESENTING) === "1") present(Math.max(0, fromHash()));
  window.addEventListener("hashchange", () => {
    const index = fromHash();
    if (deck && index >= 0 && index !== current) go(index, 0);
  });
})();
