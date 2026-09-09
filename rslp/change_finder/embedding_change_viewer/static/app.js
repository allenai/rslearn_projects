(function () {
  "use strict";

  // --- State ---
  let examples = [];
  let currentIndex = 0;
  let selectedYear = 0;
  let showOverlay = false;

  // --- DOM refs ---
  const btnOverlay = document.getElementById("btn-overlay");
  const exCount = document.getElementById("ex-count");
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");
  const navInfo = document.getElementById("nav-info");
  const coordText = document.getElementById("coord-text");
  const windowInfo = document.getElementById("window-info");
  const yearSelector = document.getElementById("year-selector");
  const sentinelGrid = document.getElementById("sentinel-grid");
  const maskRow = document.getElementById("mask-row");

  // --- Helpers ---
  function fetchJSON(url) {
    return fetch(url).then((r) => r.json());
  }

  function updateHash() {
    const url = new URL(window.location.href);
    if (showOverlay) {
      url.searchParams.set("overlay", "1");
    } else {
      url.searchParams.delete("overlay");
    }
    url.hash = `#/${currentIndex}`;
    window.history.replaceState(null, "", url);
  }

  function setNavButtons() {
    btnPrev.disabled = currentIndex <= 0;
    btnNext.disabled = currentIndex >= examples.length - 1;
    if (examples.length > 0) {
      navInfo.textContent = `Example ${currentIndex + 1} / ${examples.length}`;
    } else {
      navInfo.textContent = "No examples";
    }
  }

  function getOverlaySrc(ex) {
    return `/image/change_overlay/${encodeURIComponent(ex.window_group)}/${encodeURIComponent(ex.window_name)}`;
  }

  function getMaskSrc(ex) {
    return `/image/change_mask/${encodeURIComponent(ex.window_group)}/${encodeURIComponent(ex.window_name)}`;
  }

  function buildImageStack(baseSrc, altText, overlaySrc) {
    const stack = document.createElement("div");
    stack.className = "image-stack";

    const baseImg = document.createElement("img");
    baseImg.src = baseSrc;
    baseImg.alt = altText;
    baseImg.loading = "lazy";
    stack.appendChild(baseImg);

    if (showOverlay && overlaySrc) {
      const overlayImg = document.createElement("img");
      overlayImg.src = overlaySrc;
      overlayImg.alt = "";
      overlayImg.loading = "lazy";
      overlayImg.className = "overlay-image";
      stack.appendChild(overlayImg);
    }

    return stack;
  }

  function updateOverlayButton() {
    btnOverlay.textContent = showOverlay ? "Overlay: on (a)" : "Overlay: off (a)";
    btnOverlay.classList.toggle("active", showOverlay);
  }

  // --- Render functions ---
  function renderYearButtons() {
    yearSelector.innerHTML = "";
    for (let y = 0; y < 10; y++) {
      const btn = document.createElement("button");
      btn.textContent = `${2016 + y} (y${y})`;
      btn.className = y === selectedYear ? "active" : "";
      btn.addEventListener("click", function () {
        selectedYear = y;
        renderYearButtons();
        renderSentinel();
      });
      yearSelector.appendChild(btn);
    }
  }

  function renderSentinel() {
    sentinelGrid.innerHTML = "";
    if (!examples.length) return;

    const ex = examples[currentIndex];
    const yearData = ex.years[String(selectedYear)] || [];
    const overlaySrc = getOverlaySrc(ex);

    if (yearData.length === 0) {
      sentinelGrid.innerHTML = '<div class="empty-msg">No images for this year</div>';
      return;
    }

    for (const grp of yearData) {
      const col = document.createElement("div");
      col.className = "s2-col";
      col.appendChild(
        buildImageStack(
          `/image/sentinel2/${encodeURIComponent(ex.window_group)}/${encodeURIComponent(ex.window_name)}/${selectedYear}/${grp.group_idx}`,
          `S2 y${selectedYear} g${grp.group_idx}`,
          overlaySrc,
        )
      );

      const dateLabel = document.createElement("div");
      dateLabel.className = "s2-date";
      dateLabel.textContent = grp.date || "?";
      col.appendChild(dateLabel);

      sentinelGrid.appendChild(col);
    }
  }

  function renderMask() {
    maskRow.innerHTML = "";
    if (!examples.length) return;
    const ex = examples[currentIndex];
    maskRow.appendChild(
      buildImageStack(getMaskSrc(ex), "change mask", getOverlaySrc(ex))
    );
  }

  function renderExample() {
    if (!examples.length) {
      coordText.textContent = "\u2014";
      windowInfo.textContent = "\u2014";
      yearSelector.innerHTML = "";
      sentinelGrid.innerHTML = '<div class="empty-msg">No examples</div>';
      maskRow.innerHTML = "";
      setNavButtons();
      return;
    }

    selectedYear = 0;
    const ex = examples[currentIndex];
    coordText.textContent = `${ex.lat.toFixed(4)}, ${ex.lon.toFixed(4)}`;
    windowInfo.textContent = `${ex.window_group} / ${ex.window_name}`;

    setNavButtons();
    updateHash();
    renderYearButtons();
    renderSentinel();
    renderMask();
  }

  function changeExample(delta) {
    const nextIndex = currentIndex + delta;
    if (nextIndex < 0 || nextIndex >= examples.length) {
      return;
    }
    currentIndex = nextIndex;
    renderExample();
  }

  function changeYear(delta) {
    const totalYears = 10;
    selectedYear = (selectedYear + delta + totalYears) % totalYears;
    renderYearButtons();
    renderSentinel();
  }

  function toggleOverlay() {
    showOverlay = !showOverlay;
    updateOverlayButton();
    renderSentinel();
    renderMask();
    updateHash();
  }

  // --- Events ---
  btnOverlay.addEventListener("click", function () {
    toggleOverlay();
  });
  btnPrev.addEventListener("click", function () {
    changeExample(-1);
  });
  btnNext.addEventListener("click", function () {
    changeExample(1);
  });

  document.addEventListener("keydown", function (e) {
    if (
      e.target.tagName === "SELECT" ||
      e.target.tagName === "INPUT" ||
      e.target.tagName === "TEXTAREA"
    ) {
      return;
    }

    if (e.key === "ArrowLeft") {
      changeYear(-1);
      e.preventDefault();
    } else if (e.key === "ArrowRight") {
      changeYear(1);
      e.preventDefault();
    } else if (e.key === "p" || e.key === "P") {
      changeExample(-1);
      e.preventDefault();
    } else if (e.key === "n" || e.key === "N") {
      changeExample(1);
      e.preventDefault();
    } else if (e.key === "a" || e.key === "A") {
      toggleOverlay();
      e.preventDefault();
    }
  });

  // --- Init ---
  function init() {
    showOverlay = new URLSearchParams(window.location.search).get("overlay") === "1";

    let initialIndex = 0;
    const hash = window.location.hash;
    if (hash && hash.startsWith("#/")) {
      const parts = hash.slice(2).split("/");
      if (parts.length >= 1) {
        initialIndex = parseInt(parts[0], 10) || 0;
      }
    }

    updateOverlayButton();
    navInfo.textContent = "Loading...";
    exCount.textContent = "loading...";

    fetchJSON("/api/examples").then(function (data) {
      examples = data;
      currentIndex = Math.min(Math.max(initialIndex, 0), Math.max(0, examples.length - 1));
      exCount.textContent = `${examples.length} examples`;
      renderExample();
    });
  }

  init();
})();
