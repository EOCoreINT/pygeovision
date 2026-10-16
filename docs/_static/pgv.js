/* PyGeoVision docs behaviour.
   1. Sidebar accordion: each caption is a collapsible group; only the group holding the current page starts open.
   2. Subtle motion: figures, cards, tables, admonitions and code blocks fade in as they scroll into view.
   3. Click-to-enlarge for figures.
   4. Reading-progress bar and back-to-top button.
   5. Remember the chosen language. Respects prefers-reduced-motion. */
document.addEventListener("DOMContentLoaded", function () {
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* 1 -- sidebar accordion */
  var tree = document.querySelector(".sidebar-tree");
  if (tree) {
    var caps = Array.prototype.slice.call(tree.querySelectorAll(":scope > p.caption"));
    caps.forEach(function (cap) {
      var list = cap.nextElementSibling;
      if (!list || list.tagName !== "UL") return;
      var group = document.createElement("div");
      group.className = "pgv-group";
      cap.parentNode.insertBefore(group, cap);
      group.appendChild(cap);
      group.appendChild(list);
      var hasCurrent = !!list.querySelector("li.current, li.current-page, a.current");
      group.classList.toggle("pgv-open", hasCurrent);
      cap.setAttribute("tabindex", "0");
      cap.setAttribute("role", "button");
      cap.setAttribute("aria-expanded", hasCurrent ? "true" : "false");
      function toggle() {
        var open = !group.classList.contains("pgv-open");
        group.classList.toggle("pgv-open", open);
        cap.setAttribute("aria-expanded", open ? "true" : "false");
      }
      cap.addEventListener("click", toggle);
      cap.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      });
    });
    var cur = tree.querySelector("li.current-page > a, a.current");
    if (cur && cur.scrollIntoView) { cur.scrollIntoView({ block: "center" }); }
  }

  /* 2 -- reveal on scroll */
  var targets = document.querySelectorAll("article figure, article .sd-card, article .pgv-stat, article table.docutils, article .admonition, article div[class*='highlight']");
  if (!reduce && "IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("pgv-in"); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    targets.forEach(function (t) { t.classList.add("pgv-reveal"); io.observe(t); });
  }

  /* 3 -- lightbox */
  var box;
  function close() { if (box) { box.remove(); box = null; } }
  document.querySelectorAll("article figure img").forEach(function (img) {
    img.style.cursor = "zoom-in";
    img.addEventListener("click", function () {
      close();
      box = document.createElement("div"); box.className = "pgv-lightbox";
      var big = document.createElement("img"); big.src = img.currentSrc || img.src; big.alt = img.alt;
      box.appendChild(big); box.addEventListener("click", close); document.body.appendChild(box);
    });
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") close(); });

  /* 4 -- progress bar and back to top */
  var bar = document.createElement("div"); bar.className = "pgv-progress"; bar.setAttribute("aria-hidden", "true");
  var top = document.createElement("button"); top.className = "pgv-top"; top.type = "button"; top.setAttribute("aria-label", "Back to top");
  top.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M6 14l6-6 6 6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  document.body.appendChild(bar); document.body.appendChild(top);
  top.addEventListener("click", function () { window.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" }); });
  var ticking = false;
  function onScroll() {
    if (ticking) return; ticking = true;
    window.requestAnimationFrame(function () {
      var h = document.documentElement, max = h.scrollHeight - h.clientHeight;
      bar.style.width = (max > 0 ? Math.min(100, (h.scrollTop / max) * 100) : 0) + "%";
      top.classList.toggle("pgv-show", h.scrollTop > 600);
      ticking = false;
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
});

/* 5 -- remember the language chosen in the switcher */
document.addEventListener("click", function (e) {
  var a = e.target.closest && e.target.closest(".pgv-lang-menu a");
  if (a) { try { localStorage.setItem("pgv-lang", a.dataset.lang); } catch (_) {} }
});
