// The symbols of the user guide's notation link to their row of the notation page
// (user_guide/notation.html#notation-<macro>). Their MathJax macros (_NOTATION in conf.py) mark them
// with the class shapiq-notation-<macro>; once MathJax has typeset the page, a click on a marked
// symbol opens its row. The link is built from the page's content root, so it also works for docs
// opened from disk.
(() => {
  const PREFIX = "shapiq-notation-";
  const root = document.documentElement.dataset.content_root ?? "";

  const link = () => {
    for (const symbol of document.querySelectorAll(`[class*="${PREFIX}"]`)) {
      const name = [...symbol.classList].find((c) => c.startsWith(PREFIX))?.slice(PREFIX.length);
      if (!name || symbol.classList.contains("shapiq-notation")) continue;
      symbol.classList.add("shapiq-notation");
      symbol.title = "See the notation";
      symbol.addEventListener("click", () => {
        window.location.href = `${root}user_guide/notation.html#notation-${name}`;
      });
    }
  };

  // MathJax loads after this script; wait for it, then for its first typesetting of the page
  const start = Date.now();
  const wait = () => {
    const ready = window.MathJax?.startup?.promise;
    if (ready) ready.then(link);
    else if (Date.now() - start < 20000) setTimeout(wait, 100);
  };
  if (document.querySelector('script[src*="mathjax"]')) wait();
})();
