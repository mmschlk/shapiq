// Keep the place in the sidebar when moving between pages. Every page is its own HTML file, so
// furo's sidebar would start at the top again: the link you clicked stays at the height where you
// clicked it, and a page reached in another way scrolls its own entry into view.
(() => {
  const KEY = "shapiq-sidebar-position";

  const sidebar = () => document.querySelector(".sidebar-scroll");
  const heightIn = (element, box) =>
    element.getBoundingClientRect().top - box.getBoundingClientRect().top;

  document.addEventListener("click", (event) => {
    const link = event.target.closest(".sidebar-tree a.reference");
    const box = sidebar();
    if (!link || !box) return;
    const position = { page: new URL(link.href).pathname, height: heightIn(link, box) };
    try {
      sessionStorage.setItem(KEY, JSON.stringify(position));
    } catch {
      // storage can be unavailable (private windows, blocked cookies): just start at the top
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    const box = sidebar();
    const current = document.querySelector(".sidebar-tree .current-page > a");
    if (!box || !current) return;
    let position = null;
    try {
      position = JSON.parse(sessionStorage.getItem(KEY));
      sessionStorage.removeItem(KEY);
    } catch {
      position = null;
    }
    const height = heightIn(current, box);
    // furo scrolls the sidebar smoothly; jump instead, or every page would open with an animation
    const scrollBy = (delta) => box.scrollTo({ top: box.scrollTop + delta, behavior: "instant" });
    if (position && position.page === window.location.pathname) {
      scrollBy(height - position.height); // the clicked link stays where it was
    } else if (height < 0 || height > box.clientHeight - current.offsetHeight) {
      scrollBy(height - box.clientHeight / 3); // bring this page's entry into view
    }
  });
})();
