/* RF-LEGO project page — progressive enhancement only.
   Nothing here is required for the content to be readable. */
(function () {
  'use strict';

  /* ---------- KaTeX ---------- */
  function renderMath() {
    if (typeof renderMathInElement !== 'function') return;
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ],
      throwOnError: false,
      strict: false
    });
  }
  if (document.readyState !== 'loading') renderMath();
  else document.addEventListener('DOMContentLoaded', renderMath);

  /* ---------- theme ---------- */
  var root = document.documentElement;  /* stored theme is applied inline in <head> */

  var toggle = document.getElementById('themeToggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var current = root.getAttribute('data-theme');
      if (current !== 'light' && current !== 'dark') {
        current = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      }
      var next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('rflego-theme', next); } catch (e) { /* ignore */ }
    });
  }

  /* ---------- reveal on scroll ---------- */
  /* Rect-based rather than IntersectionObserver: a fast scroll or a jump to an
     anchor must never leave a section stuck at opacity 0. */
  var pending = Array.prototype.slice.call(document.querySelectorAll('.reveal'));
  function sweep() {
    var limit = window.innerHeight * 0.94;
    for (var i = pending.length - 1; i >= 0; i--) {
      if (pending[i].getBoundingClientRect().top < limit) {
        pending[i].classList.add('in');
        pending.splice(i, 1);
      }
    }
  }

  /* ---------- scroll progress + active section ---------- */
  var bar = document.getElementById('progress');
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.topbar nav a'));
  var sections = navLinks
    .map(function (a) { return document.querySelector(a.getAttribute('href')); })
    .filter(Boolean);

  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(function () {
      var h = document.documentElement.scrollHeight - window.innerHeight;
      if (bar) bar.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0) + '%';

      var idx = -1;
      for (var i = 0; i < sections.length; i++) {
        if (sections[i].getBoundingClientRect().top <= 120) idx = i;
      }
      navLinks.forEach(function (a, i) { a.classList.toggle('is-active', i === idx); });
      sweep();
      ticking = false;
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  onScroll();

  /* if anything is still pending after load, show it — content is never hidden */
  window.addEventListener('load', function () {
    setTimeout(sweep, 60);
  });

  /* ---------- copy BibTeX ---------- */
  var btn = document.getElementById('copyBib');
  var bib = document.getElementById('bibtex');
  if (btn && bib) {
    btn.addEventListener('click', function () {
      var text = bib.textContent;
      var done = function () {
        btn.textContent = 'Copied';
        btn.classList.add('done');
        setTimeout(function () { btn.textContent = 'Copy'; btn.classList.remove('done'); }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else { fallback(); }

      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.cssText = 'position:absolute;left:-9999px';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ }
        document.body.removeChild(ta);
      }
    });
  }
})();
