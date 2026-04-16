(function () {
  "use strict";

  const ALL = "__all__";
  const ALL_LABEL = "All";
  const MONTH_RE = /^\d{4}-(0[1-9]|1[0-2])$/;

  // --- State ---
  let categories = [];
  let examples = [];
  let currentIndex = 0;
  let selectedYear = 0;
  let legend = [];
  let showOverlay = false;
  let srcClassName = ALL;
  let dstClassName = ALL;

  // --- DOM refs ---
  const selSrc = document.getElementById("sel-src");
  const selDst = document.getElementById("sel-dst");
  const btnGo = document.getElementById("btn-go");
  const btnOverlay = document.getElementById("btn-overlay");
  const catCount = document.getElementById("cat-count");
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");
  const navInfo = document.getElementById("nav-info");
  const coordText = document.getElementById("coord-text");
  const windowInfo = document.getElementById("window-info");
  const transitionInfo = document.getElementById("transition-info");
  const yearSelector = document.getElementById("year-selector");
  const sentinelGrid = document.getElementById("sentinel-grid");
  const landcoverRow = document.getElementById("landcover-row");
  const legendDiv = document.getElementById("legend");
  const polygonRow = document.getElementById("polygon-row");
  const annotForm = document.getElementById("annot-form");
  const annotStart = document.getElementById("annot-start");
  const annotEnd = document.getElementById("annot-end");
  const annotStatus = document.getElementById("annot-status");

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
    url.hash = `#/${encodeURIComponent(srcClassName)}/${encodeURIComponent(dstClassName)}/${currentIndex}`;
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

  function getOverlaySrc() {
    const qs = new URLSearchParams({ src: srcClassName, dst: dstClassName }).toString();
    return `/image/change_polygon/${currentIndex}?${qs}`;
  }

  function buildImageStack(baseSrc, altText, overlaySrc) {
    const stack = document.createElement("div");
    stack.className = "image-stack";

    const baseImg = document.createElement("img");
    baseImg.src = baseSrc;
    baseImg.alt = altText;
    baseImg.loading = "lazy";
    stack.appendChild(baseImg);

    if (showOverlay) {
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

  function clearAnnotStatus() {
    annotStatus.textContent = "";
    annotStatus.className = "";
  }

  function setAnnotStatus(msg, kind) {
    annotStatus.textContent = msg;
    annotStatus.className = kind || "";
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
    const overlaySrc = getOverlaySrc();

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

  function renderLandcover() {
    landcoverRow.innerHTML = "";
    if (!examples.length) return;

    const ex = examples[currentIndex];
    const overlaySrc = getOverlaySrc();

    for (const period of ["early", "late"]) {
      const col = document.createElement("div");
      col.className = "lc-col";
      col.appendChild(
        buildImageStack(
          `/image/landcover/${encodeURIComponent(ex.window_group)}/${encodeURIComponent(ex.window_name)}/${period}`,
          `${period} land cover`,
          overlaySrc,
        )
      );

      const lbl = document.createElement("div");
      lbl.className = "lc-label";
      lbl.textContent = period === "early" ? "Early (2016-2018)" : "Late (2023-2025)";
      col.appendChild(lbl);

      landcoverRow.appendChild(col);
    }
  }

  function renderLegend() {
    legendDiv.innerHTML = "";
    for (const item of legend) {
      const el = document.createElement("span");
      el.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.backgroundColor = `rgb(${item.r},${item.g},${item.b})`;
      el.appendChild(swatch);
      el.appendChild(document.createTextNode(item.class_name));
      legendDiv.appendChild(el);
    }
  }

  function renderPolygon() {
    polygonRow.innerHTML = "";
    if (!examples.length) return;

    const img = document.createElement("img");
    img.src = getOverlaySrc();
    img.alt = "change polygon";
    img.loading = "lazy";
    polygonRow.appendChild(img);
  }

  function renderAnnotation() {
    if (!examples.length) {
      annotStart.value = "";
      annotEnd.value = "";
      clearAnnotStatus();
      return;
    }
    const ex = examples[currentIndex];
    annotStart.value = ex.change_start_month || "";
    annotEnd.value = ex.change_end_month || "";
    clearAnnotStatus();
  }

  function renderExample() {
    if (!examples.length) {
      coordText.textContent = "—";
      windowInfo.textContent = "—";
      transitionInfo.textContent = "—";
      yearSelector.innerHTML = "";
      sentinelGrid.innerHTML = '<div class="empty-msg">No examples in this category</div>';
      landcoverRow.innerHTML = "";
      polygonRow.innerHTML = "";
      renderAnnotation();
      setNavButtons();
      return;
    }

    selectedYear = 0;
    const ex = examples[currentIndex];
    coordText.textContent = `${ex.lat.toFixed(4)}, ${ex.lon.toFixed(4)}`;
    windowInfo.textContent = `${ex.window_group} / ${ex.window_name}`;
    transitionInfo.textContent = `${ex.src_class_name} -> ${ex.dst_class_name}`;

    setNavButtons();
    updateHash();
    renderYearButtons();
    renderSentinel();
    renderLandcover();
    renderPolygon();
    renderAnnotation();
  }

  // --- Category dropdowns ---
  function populateDropdowns() {
    const srcSet = new Set(categories.map((c) => c.src_class_name));
    const srcs = [ALL, ...Array.from(srcSet).sort()];
    selSrc.innerHTML = "";
    for (const s of srcs) {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s === ALL ? ALL_LABEL : s;
      selSrc.appendChild(opt);
    }
    selSrc.value = srcClassName;
    updateDstDropdown();
    selDst.value = dstClassName;
  }

  function updateDstDropdown() {
    const src = selSrc.value;
    const dstSet = new Set();
    for (const c of categories) {
      if (src === ALL || c.src_class_name === src) {
        dstSet.add(c.dst_class_name);
      }
    }
    const dsts = [ALL, ...Array.from(dstSet).sort()];
    selDst.innerHTML = "";
    for (const d of dsts) {
      const opt = document.createElement("option");
      opt.value = d;
      opt.textContent = d === ALL ? ALL_LABEL : d;
      selDst.appendChild(opt);
    }
    // If the previous value no longer exists, fall back to All.
    if (![...selDst.options].some((o) => o.value === dstClassName)) {
      dstClassName = ALL;
    }
    selDst.value = dstClassName;
  }

  function loadCategory(src, dst, index) {
    srcClassName = src;
    dstClassName = dst;
    catCount.textContent = "loading...";
    const qs = new URLSearchParams({ src, dst }).toString();
    fetchJSON(`/api/examples?${qs}`)
      .then(function (data) {
        examples = data;
        currentIndex = Math.min(Math.max(index || 0, 0), Math.max(0, examples.length - 1));
        catCount.textContent = `${examples.length} examples`;
        renderExample();
      });
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
    renderLandcover();
    updateHash();
  }

  function saveAnnotation(evt) {
    evt.preventDefault();
    if (!examples.length) return;
    const start = annotStart.value.trim();
    const end = annotEnd.value.trim();
    for (const v of [start, end]) {
      if (v !== "" && !MONTH_RE.test(v)) {
        setAnnotStatus("Months must be YYYY-MM", "err");
        return;
      }
    }
    const ex = examples[currentIndex];
    setAnnotStatus("Saving...", "");
    fetch("/api/annotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        window_group: ex.window_group,
        window_name: ex.window_name,
        src_class_id: ex.src_class_id,
        dst_class_id: ex.dst_class_id,
        change_start_month: start,
        change_end_month: end,
      }),
    })
      .then(function (r) {
        return r.json().then(function (j) {
          return { ok: r.ok, body: j };
        });
      })
      .then(function (res) {
        if (!res.ok || !res.body.ok) {
          setAnnotStatus(`Error: ${res.body.error || "save failed"}`, "err");
          return;
        }
        examples[currentIndex] = res.body.feature;
        annotStart.value = res.body.feature.change_start_month || "";
        annotEnd.value = res.body.feature.change_end_month || "";
        setAnnotStatus("Saved", "ok");
      })
      .catch(function (err) {
        setAnnotStatus(`Error: ${err}`, "err");
      });
  }

  // --- Events ---
  selSrc.addEventListener("change", function () {
    srcClassName = selSrc.value;
    updateDstDropdown();
    dstClassName = selDst.value;
  });
  selDst.addEventListener("change", function () {
    dstClassName = selDst.value;
  });

  btnGo.addEventListener("click", function () {
    loadCategory(selSrc.value, selDst.value, 0);
  });

  btnOverlay.addEventListener("click", function () {
    toggleOverlay();
  });

  btnPrev.addEventListener("click", function () {
    changeExample(-1);
  });

  btnNext.addEventListener("click", function () {
    changeExample(1);
  });

  annotForm.addEventListener("submit", saveAnnotation);

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
    Promise.all([fetchJSON("/api/categories"), fetchJSON("/api/legend")])
      .then(function (results) {
        categories = results[0];
        legend = results[1];
        showOverlay = new URLSearchParams(window.location.search).get("overlay") === "1";

        // Restore from hash if present.
        const hash = window.location.hash;
        if (hash && hash.startsWith("#/")) {
          const parts = hash.slice(2).split("/");
          if (parts.length >= 3) {
            srcClassName = decodeURIComponent(parts[0]) || ALL;
            dstClassName = decodeURIComponent(parts[1]) || ALL;
            const idx = parseInt(parts[2], 10) || 0;
            populateDropdowns();
            updateOverlayButton();
            renderLegend();
            loadCategory(srcClassName, dstClassName, idx);
            return;
          }
        }

        populateDropdowns();
        updateOverlayButton();
        renderLegend();
        loadCategory(ALL, ALL, 0);
      });
  }

  init();
})();
