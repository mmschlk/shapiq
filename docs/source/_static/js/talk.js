/* Talks (talks/, _ext/shapiq_talks.py): a page of slides, read as a handout or presented as a deck.

   On the page, every slide shows all its steps, with its notes below it. "Present" (or the key P)
   moves the slides into a deck that fills the window and shows one slide at a time:

     →, Space, Page Down    the next step, or the next slide      (a presenter's clicker sends Page Down)
     ←, Page Up             the step before, or the slide before
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
   "shapiq-talk-slide" (detail: slide, index) whenever another slide is shown. */
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

  const bar = make("div", "shapiq-talk-bar");
  const start = make("button", "shapiq-talk-present", ICONS.play + "<span>Present</span>");
  start.type = "button";
  start.addEventListener("click", () => present(0));
  const hint = make("p");
  const hints = () => {
    hint.innerHTML = phone.matches
      ? "One slide per screen. Scroll from slide to slide."
      : 'One slide at a time. Press <kbd class="shapiq-talk-key">P</kbd> to start, ' +
        'the arrow keys to move, and <kbd class="shapiq-talk-key">Esc</kbd> to come back here.';
  };
  hints();
  bar.append(start, hint);
  slides[0].before(bar);

  slides.forEach((slide, index) => {
    const here = make("button", "shapiq-slide-here", "Present from here");
    here.type = "button";
    here.addEventListener("click", () => present(index));
    slide.prepend(here);
  });

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
      button("close", "Back to the page (Esc)", stop),
    );
    progress = make("div", "shapiq-deck-progress", "<span></span>");
    notes = make("aside", "shapiq-deck-notes");
    notes.hidden = true;
    deck.append(corner, progress, notes);
    deck.addEventListener("click", clickToAdvance);
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
    if (moved) show(index);
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

  // a click or a swipe on the slide itself moves on, not one on its text, a demo or a figure
  const onBackground = (target) => target.classList &&
    (target.classList.contains("shapiq-slide") || target.classList.contains("shapiq-slide-body"));

  function clickToAdvance(event) {
    if (event.button !== 0 || scrolling() || !onBackground(event.target)) return;
    if (String(window.getSelection && window.getSelection()).length) return;
    forward();
  }

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

  // ---- keys ------------------------------------------------------------------------------------

  const typing = (target) => target.closest &&
    target.closest('textarea, select, [contenteditable=""], [contenteditable="true"], input:not([type]), ' +
      'input[type="text"], input[type="search"], input[type="number"]');
  // controls that use the arrow keys or Space themselves: sliders, tabs, buttons
  const control = (target) => target.closest &&
    target.closest('input, button, [role="slider"], [role="tab"], [role="radio"], [role="option"], [role="button"]');

  document.addEventListener("keydown", (event) => {
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
    else if (key === "ArrowLeft" && !inControl) action = back;
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
  const wanted = /(?:^|[?&])present(?:[=&]|$)/.test(location.search.slice(1)) || store.get(PRESENTING) === "1";
  if (wanted) present(Math.max(0, fromHash()));
  window.addEventListener("hashchange", () => {
    const index = fromHash();
    if (deck && index >= 0 && index !== current) go(index, 0);
  });
})();
