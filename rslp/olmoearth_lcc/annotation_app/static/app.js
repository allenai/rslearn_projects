(function () {
  "use strict";

  // Land cover categories for the pre/post change dropdowns. These mirror the
  // Land cover class names (see data/olmoearth_lcc/land_cover/config_per_pixel.yaml).
  const CATEGORIES = [
    "bare",
    "burnt",
    "crops",
    "fallow/shifting cultivation",
    "grassland",
    "Lichen and moss",
    "shrub",
    "snow and ice",
    "tree",
    "urban/built-up",
    "water",
    "wetland (herbaceous)",
  ];

  const FINE_CHANGE_CATEGORIES = [
    "new_solar_farm",
    "new_wind_turbine",
    "new_power_tower",
    "new_building",
    "new_road",
    "resurfaced_road",
    "repainted_roof",
    "tree_crops_harvested",
    "tree_crops_growth_gradual",
    "wetland_loss",
    "new_crop_field",
    "deforestation",
    "wildfire",
    "mining",
    "removed_building",
    "removed_road",
    "site_clearing",
    "crop_temporary_building_erected",
    "new_aquafarm",
    "new_offshore_infrastructure",
  ];

  // New three-group change categories. Pick one pre and/or one post, OR one same.
  const PRE_CHANGE_CATEGORIES = [
    "deforestation",
    "urban_erosion",
    "wetland_loss",
    "water_contract",
    "removed_crop_structure",
  ];

  const POST_CHANGE_CATEGORIES = [
    "vegetation_growth",
    "new_building",
    "new_road",
    "new_infrastructure",
    "new_crop_field",
    "new_aquafarm",
    "site_clearing",
    "water_expand",
    "mining",
    "new_crop_structure",
    "selective_logging",
    "landslide",
    "settlement",
  ];

  const SAME_CHANGE_CATEGORIES = [
    "agricultural_activity",
    "wildfire",
    "ice_motion",
    "flooding",
  ];

  // --- State ---
  let entriesList = [];
  let currentEntry = null;
  let currentIndex = 0;
  let selectedYear = null;
  let selectedPointIdx = 0;
  let showOverlay = false;
  let brighten = false;
  let isSaving = false;
  let overlayVersion = Date.now();

  // --- DOM refs ---
  const btnOverlay = document.getElementById("btn-overlay");
  const btnBright = document.getElementById("btn-bright");
  const entryCount = document.getElementById("entry-count");
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");
  const navInfo = document.getElementById("nav-info");
  const btnPtPrev = document.getElementById("btn-pt-prev");
  const btnPtNext = document.getElementById("btn-pt-next");
  const btnPtMakeNegative = document.getElementById("btn-pt-make-negative");
  const pointInfo = document.getElementById("point-info");
  const coordText = document.getElementById("coord-text");
  const pointCoordText = document.getElementById("point-coord-text");
  const windowInfo = document.getElementById("window-info");
  const yearSelector = document.getElementById("year-selector");
  const sentinelGrid = document.getElementById("sentinel-grid");
  const annotForm = document.getElementById("annot-form");
  const annotStatus = document.getElementById("annot-status");
  const btnSave = document.getElementById("btn-save");
  const btnApplyAll = document.getElementById("btn-apply-all");
  const annotInputs = {
    pre_change: document.getElementById("annot-pre-change"),
    first_date_change_noticeable: document.getElementById("annot-first-noticeable"),
    post_change: document.getElementById("annot-post-change"),
    stop_date: document.getElementById("annot-stop-date"),
    pre_category: document.getElementById("annot-pre-category"),
    post_category: document.getElementById("annot-post-category"),
    pre_change_category: document.getElementById("annot-pre-change-category"),
    post_change_category: document.getElementById("annot-post-change-category"),
    same_change_category: document.getElementById("annot-same-change-category"),
    fine_change_category: document.getElementById("annot-fine-change-category"),
  };
  const sampleForm = document.getElementById("sample-form");
  const sampleInputs = {
    anchor_date: document.getElementById("sample-anchor-date"),
    description: document.getElementById("sample-description"),
  };

  function populateCategorySelect(select, categories) {
    select.innerHTML = "";
    var blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "\u2014";
    select.appendChild(blank);
    for (var i = 0; i < categories.length; i++) {
      var opt = document.createElement("option");
      opt.value = categories[i];
      opt.textContent = categories[i];
      select.appendChild(opt);
    }
  }

  function setCategorySelect(select, value) {
    value = value || "";
    // Preserve values not in the predefined list so they aren't silently lost.
    if (value && !Array.prototype.some.call(select.options, function (o) { return o.value === value; })) {
      var opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value + " (legacy)";
      select.appendChild(opt);
    }
    select.value = value;
  }

  // --- Helpers ---
  function updateHash() {
    window.history.replaceState(null, "", "#/" + currentIndex);
  }

  function fetchJSON(url) {
    return fetch(url).then(function (r) { return r.json(); });
  }

  function postJSON(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(function (r) { return r.json(); });
  }

  function setNavButtons() {
    btnPrev.disabled = isSaving || currentIndex <= 0;
    btnNext.disabled = isSaving || currentIndex >= entriesList.length - 1;
    navInfo.textContent = entriesList.length > 0
      ? "Entry " + (currentIndex + 1) + " / " + entriesList.length
      : "No entries";
  }

  function setPointButtons() {
    if (!currentEntry) {
      pointInfo.textContent = "0 / 0";
      btnPtPrev.disabled = true;
      btnPtNext.disabled = true;
      btnPtMakeNegative.disabled = true;
      return;
    }
    var pts = currentEntry.entry.positive_points || [];
    var n = pts.length;
    if (n === 0) {
      pointInfo.textContent = "0 / 0";
      selectedPointIdx = 0;
      btnPtPrev.disabled = true;
      btnPtNext.disabled = true;
      btnPtMakeNegative.disabled = true;
    } else {
      if (selectedPointIdx >= n) selectedPointIdx = n - 1;
      if (selectedPointIdx < 0) selectedPointIdx = 0;
      pointInfo.textContent = (selectedPointIdx + 1) + " / " + n;
      btnPtPrev.disabled = selectedPointIdx <= 0;
      btnPtNext.disabled = selectedPointIdx >= n - 1;
      btnPtMakeNegative.disabled = false;
    }
  }

  function updatePointCoord() {
    if (!currentEntry) {
      pointCoordText.textContent = "\u2014";
      return;
    }
    var pts = currentEntry.entry.positive_points || [];
    if (pts.length === 0 || selectedPointIdx >= pts.length) {
      pointCoordText.textContent = "\u2014";
      return;
    }
    var pt = pts[selectedPointIdx];
    pointCoordText.textContent = pt.lat.toFixed(5) + ", " + pt.lon.toFixed(5);
  }

  function updateOverlayButton() {
    btnOverlay.textContent = showOverlay ? "Overlay: on (a)" : "Overlay: off (a)";
    btnOverlay.classList.toggle("active", showOverlay);
  }

  function updateBrightButton() {
    btnBright.textContent = brighten ? "Brightness \u00f715: on (b)" : "Brightness \u00f715: off (b)";
    btnBright.classList.toggle("active", brighten);
  }

  function getOverlaySrc() {
    var url = "/image/points_overlay/" + currentIndex;
    var pts = (currentEntry && currentEntry.entry.positive_points) || [];
    if (pts.length > 0) {
      url += "?selected=" + selectedPointIdx + "&v=" + overlayVersion;
    } else {
      url += "?v=" + overlayVersion;
    }
    return url;
  }

  // --- Year detection ---
  function getAvailableYears() {
    if (!currentEntry || !currentEntry.meta || !currentEntry.meta.years) return [];
    var years = Object.keys(currentEntry.meta.years).map(Number).sort();
    return years;
  }

  // --- Render ---
  function renderYearButtons() {
    yearSelector.innerHTML = "";
    var years = getAvailableYears();
    if (years.length === 0) return;
    if (selectedYear === null || years.indexOf(selectedYear) === -1) {
      selectedYear = years[0];
    }
    for (var i = 0; i < years.length; i++) {
      var y = years[i];
      var btn = document.createElement("button");
      btn.textContent = String(y);
      btn.className = y === selectedYear ? "active" : "";
      btn.dataset.year = y;
      btn.addEventListener("click", function () {
        selectedYear = Number(this.dataset.year);
        renderYearButtons();
        renderSentinel();
      });
      yearSelector.appendChild(btn);
    }
  }

  function buildImageStack(baseSrc, altText) {
    var stack = document.createElement("div");
    stack.className = "image-stack";

    var baseImg = document.createElement("img");
    baseImg.src = baseSrc;
    baseImg.alt = altText;
    baseImg.loading = "lazy";
    stack.appendChild(baseImg);

    if (showOverlay) {
      var overlayImg = document.createElement("img");
      overlayImg.src = getOverlaySrc();
      overlayImg.alt = "";
      overlayImg.loading = "lazy";
      overlayImg.className = "overlay-image";
      overlayImg.addEventListener("click", handleOverlayClick);
      overlayImg.addEventListener("contextmenu", handleOverlayRightClick);
      stack.appendChild(overlayImg);
    }

    return stack;
  }

  function renderSentinel() {
    sentinelGrid.innerHTML = "";
    if (!currentEntry) return;

    var years = currentEntry.meta.years || {};
    var yearData = years[String(selectedYear)] || [];

    if (yearData.length === 0) {
      sentinelGrid.innerHTML = '<div class="empty-msg">No images for this year</div>';
      return;
    }

    var entry = currentEntry.entry;
    for (var i = 0; i < yearData.length; i++) {
      var grp = yearData[i];
      var col = document.createElement("div");
      col.className = "s2-col";
      col.appendChild(
        buildImageStack(
          "/image/sentinel2/" + encodeURIComponent(entry.group) + "/" + encodeURIComponent(entry.window_name) + "/" + grp.group_idx + (brighten ? "?bright=1" : ""),
          "S2 " + selectedYear + " g" + grp.group_idx
        )
      );
      var dateLabel = document.createElement("div");
      dateLabel.className = "s2-date";
      dateLabel.textContent = grp.date || "?";
      dateLabel.style.cursor = "pointer";
      dateLabel.title = "Click to copy";
      dateLabel.addEventListener("click", (function (date) {
        return function () {
          if (date) navigator.clipboard.writeText(date);
        };
      })(grp.date));
      col.appendChild(dateLabel);
      sentinelGrid.appendChild(col);
    }
  }

  function renderAnnotation() {
    if (!currentEntry) {
      for (var f in annotInputs) annotInputs[f].value = "";
      annotStatus.textContent = "";
      return;
    }
    var pts = currentEntry.entry.positive_points || [];
    if (pts.length === 0 || selectedPointIdx >= pts.length) {
      for (var f2 in annotInputs) annotInputs[f2].value = "";
      annotStatus.textContent = "";
      return;
    }
    var pt = pts[selectedPointIdx];
    annotInputs.pre_change.value = pt.pre_change || "";
    annotInputs.first_date_change_noticeable.value = pt.first_date_change_noticeable || "";
    annotInputs.post_change.value = pt.post_change || "";
    annotInputs.stop_date.value = pt.stop_date || "";
    setCategorySelect(annotInputs.pre_category, pt.pre_category);
    setCategorySelect(annotInputs.post_category, pt.post_category);
    setCategorySelect(annotInputs.pre_change_category, pt.pre_change_category);
    setCategorySelect(annotInputs.post_change_category, pt.post_change_category);
    setCategorySelect(annotInputs.same_change_category, pt.same_change_category);
    setCategorySelect(annotInputs.fine_change_category, pt.fine_change_category);
    // Only show the legacy fine-category dropdown when the point already has a value.
    var fineField = document.getElementById("fine-change-category-field");
    if (fineField) {
      fineField.style.display = pt.fine_change_category ? "" : "none";
    }
    annotStatus.textContent = "";
  }

  function renderSample() {
    var entry = currentEntry ? currentEntry.entry : {};
    sampleInputs.anchor_date.value = entry.anchor_date || "";
    sampleInputs.description.value = entry.description || "";
  }

  function renderEntry() {
    if (!currentEntry) {
      coordText.textContent = "\u2014";
      windowInfo.textContent = "\u2014";
      yearSelector.innerHTML = "";
      sentinelGrid.innerHTML = '<div class="empty-msg">No entry loaded</div>';
      setNavButtons();
      setPointButtons();
      renderSample();
      renderAnnotation();
      return;
    }

    var meta = currentEntry.meta;
    coordText.textContent = meta.lat != null
      ? meta.lat.toFixed(5) + ", " + meta.lon.toFixed(5)
      : "\u2014";
    windowInfo.textContent = currentEntry.entry.group + " / " + currentEntry.entry.window_name;

    setNavButtons();
    setPointButtons();
    updatePointCoord();
    updateHash();
    renderYearButtons();
    renderSentinel();
    renderSample();
    renderAnnotation();
  }

  // --- Navigation ---
  function loadEntry(idx, preserveState) {
    if (idx < 0 || idx >= entriesList.length) return Promise.resolve();
    currentIndex = idx;
    var prevYear = selectedYear;
    var prevPointIdx = selectedPointIdx;
    if (preserveState) overlayVersion++;
    return fetchJSON("/api/entry/" + idx).then(function (data) {
      currentEntry = data;
      if (preserveState) {
        selectedYear = prevYear;
        selectedPointIdx = prevPointIdx;
        var pts = (currentEntry.entry.positive_points || []);
        if (selectedPointIdx >= pts.length) {
          selectedPointIdx = Math.max(0, pts.length - 1);
        }
      } else {
        selectedPointIdx = 0;
        selectedYear = null;
      }
      renderEntry();
    });
  }

  function changeEntry(delta) {
    if (isSaving) return;
    var next = currentIndex + delta;
    if (next < 0 || next >= entriesList.length) return;
    loadEntry(next);
  }

  function changeYear(delta) {
    var years = getAvailableYears();
    if (years.length === 0) return;
    var curIdx = years.indexOf(selectedYear);
    if (curIdx === -1) curIdx = 0;
    var nextIdx = (curIdx + delta + years.length) % years.length;
    selectedYear = years[nextIdx];
    renderYearButtons();
    renderSentinel();
  }

  function changePoint(delta) {
    if (!currentEntry) return;
    var pts = currentEntry.entry.positive_points || [];
    if (pts.length === 0) return;
    var next = selectedPointIdx + delta;
    if (next < 0 || next >= pts.length) return;
    // Keep typed-but-unsaved values for the point being left; they are written
    // to disk with the next save.
    commitPointForm();
    selectedPointIdx = next;
    overlayVersion++;
    setPointButtons();
    updatePointCoord();
    renderAnnotation();
    if (showOverlay) {
      renderSentinel();
    }
  }

  // --- Click handlers for adding/removing points ---
  function handleOverlayClick(evt) {
    evt.preventDefault();
    if (!currentEntry) return;
    var rect = evt.target.getBoundingClientRect();
    var col = (evt.clientX - rect.left) / rect.width * (currentEntry.entry.bounds[2] - currentEntry.entry.bounds[0]);
    var row = (evt.clientY - rect.top) / rect.height * (currentEntry.entry.bounds[3] - currentEntry.entry.bounds[1]);

    // Shift+click always adds, even if close to an existing point.
    if (!evt.shiftKey) {
      var threshold = 6;

      // Check positive points.
      var posPixels = currentEntry.positive_pixels || [];
      for (var i = 0; i < posPixels.length; i++) {
        if (Math.abs(posPixels[i].col - col) < threshold && Math.abs(posPixels[i].row - row) < threshold) {
          removePoint("remove_positive", i);
          return;
        }
      }
      // Check negative points.
      var negPixels = currentEntry.negative_pixels || [];
      for (var j = 0; j < negPixels.length; j++) {
        if (Math.abs(negPixels[j].col - col) < threshold && Math.abs(negPixels[j].row - row) < threshold) {
          removePoint("remove_negative", j);
          return;
        }
      }
    }

    // Otherwise add a positive point.
    addPoint("add_positive", col, row);
  }

  function handleOverlayRightClick(evt) {
    evt.preventDefault();
    if (!currentEntry) return;
    var rect = evt.target.getBoundingClientRect();
    var col = (evt.clientX - rect.left) / rect.width * (currentEntry.entry.bounds[2] - currentEntry.entry.bounds[0]);
    var row = (evt.clientY - rect.top) / rect.height * (currentEntry.entry.bounds[3] - currentEntry.entry.bounds[1]);
    addPoint("add_negative", col, row);
  }

  // --- Editing the working copy ---
  // currentEntry.entry is a working copy of the entry. Every action edits it and
  // then saves the whole entry via saveEntry(). Form values are copied into the
  // working copy before each edit, so point actions also save typed values.

  function setOrDelete(obj, field, value) {
    if (value) {
      obj[field] = value;
    } else {
      delete obj[field];
    }
  }

  function commitPointForm() {
    if (!currentEntry) return;
    var pts = currentEntry.entry.positive_points || [];
    if (pts.length === 0 || selectedPointIdx >= pts.length) return;
    var pt = pts[selectedPointIdx];
    for (var field in annotInputs) {
      setOrDelete(pt, field, annotInputs[field].value.trim());
    }
  }

  function commitForms() {
    if (!currentEntry) return;
    for (var field in sampleInputs) {
      setOrDelete(currentEntry.entry, field, sampleInputs[field].value.trim());
    }
    commitPointForm();
  }

  function addPoint(action, col, row) {
    if (!currentEntry || isSaving) return;
    setSaving(true);
    postJSON("/api/pixel_to_lonlat", {
      entry_idx: currentIndex,
      col: col,
      row: row,
    }).then(function (res) {
      commitForms();
      var entry = currentEntry.entry;
      var point = { lon: res.lon, lat: res.lat };
      if (action === "add_positive") {
        // New positive points start with the first positive point's annotation.
        var existing = entry.positive_points || [];
        if (existing.length > 0) {
          for (var field in annotInputs) {
            if (existing[0][field]) point[field] = existing[0][field];
          }
        }
        entry.positive_points = existing.concat([point]);
      } else {
        entry.negative_points = (entry.negative_points || []).concat([point]);
      }
      return saveEntry("Saved");
    }).catch(function (err) {
      setSaving(false);
      showStatus("Error: " + (err.message || err), "err");
    });
  }

  function removePoint(action, pointIdx) {
    if (!currentEntry || isSaving) return;
    commitForms();
    var key = action === "remove_positive" ? "positive_points" : "negative_points";
    var pts = currentEntry.entry[key] || [];
    if (pointIdx < 0 || pointIdx >= pts.length) return;
    pts.splice(pointIdx, 1);
    if (key === "positive_points" && pointIdx < selectedPointIdx) {
      selectedPointIdx--;
    }
    saveEntry("Saved");
  }

  function makeSelectedNegative() {
    if (!currentEntry || isSaving) return;
    var pts = currentEntry.entry.positive_points || [];
    if (pts.length === 0 || selectedPointIdx >= pts.length) return;
    commitForms();
    var pt = pts.splice(selectedPointIdx, 1)[0];
    var negatives = currentEntry.entry.negative_points || [];
    negatives.push({ lon: pt.lon, lat: pt.lat });
    currentEntry.entry.negative_points = negatives;
    saveEntry("Saved");
  }

  // --- Saving ---
  function setSaving(saving) {
    isSaving = saving;
    btnSave.disabled = saving;
    btnApplyAll.disabled = saving;
    setNavButtons();
  }

  function showStatus(text, cls) {
    annotStatus.textContent = text;
    annotStatus.className = cls;
  }

  // Send the whole working copy to the server. Callers must commit the forms
  // before editing the working copy, not after, since an edit can change which
  // point the form belongs to.
  function saveEntry(okMessage) {
    setSaving(true);
    showStatus("Saving...", "");
    return postJSON("/api/save_entry", {
      entry_idx: currentIndex,
      entry: currentEntry.entry,
    }).then(function (res) {
      if (!res.ok) throw new Error(res.error || "save failed");
      currentEntry = res;
      var pts = currentEntry.entry.positive_points || [];
      if (selectedPointIdx >= pts.length) {
        selectedPointIdx = Math.max(0, pts.length - 1);
      }
      overlayVersion++;
      renderEntry();
      showStatus(okMessage, "ok");
    }).catch(function (err) {
      // Resync with what the server has, so the overlay and hit-testing match.
      var message = "Error: " + (err.message || err);
      return loadEntry(currentIndex, true).then(function () {
        showStatus(message, "err");
      });
    }).finally(function () {
      setSaving(false);
    });
  }

  function saveAll(evt) {
    if (evt) evt.preventDefault();
    if (!currentEntry || isSaving) return;
    commitForms();
    saveEntry("Saved");
  }

  function applyAnnotationToAll() {
    if (!currentEntry || isSaving) return;
    var pts = currentEntry.entry.positive_points || [];
    if (pts.length === 0 || selectedPointIdx >= pts.length) return;
    if (pts.length > 1) {
      if (!window.confirm("Apply this metadata to all " + pts.length + " positive points?")) {
        return;
      }
    }
    commitForms();
    var source = pts[selectedPointIdx];
    for (var i = 0; i < pts.length; i++) {
      for (var field in annotInputs) {
        setOrDelete(pts[i], field, source[field]);
      }
    }
    saveEntry("Applied to all");
  }

  // --- Toggle overlay ---
  function toggleOverlay() {
    showOverlay = !showOverlay;
    updateOverlayButton();
    renderSentinel();
  }

  // --- Toggle brightness ---
  function toggleBright() {
    brighten = !brighten;
    updateBrightButton();
    renderSentinel();
  }

  // --- Events ---
  btnOverlay.addEventListener("click", toggleOverlay);
  btnBright.addEventListener("click", toggleBright);
  btnPrev.addEventListener("click", function () { changeEntry(-1); });
  btnNext.addEventListener("click", function () { changeEntry(1); });
  btnPtPrev.addEventListener("click", function () { changePoint(-1); });
  btnPtNext.addEventListener("click", function () { changePoint(1); });
  btnPtMakeNegative.addEventListener("click", makeSelectedNegative);
  annotForm.addEventListener("submit", saveAll);
  btnApplyAll.addEventListener("click", applyAnnotationToAll);
  sampleForm.addEventListener("submit", saveAll);

  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
    if (e.key === "p" || e.key === "P") {
      changeEntry(-1);
      e.preventDefault();
    } else if (e.key === "n" || e.key === "N") {
      changeEntry(1);
      e.preventDefault();
    } else if (e.key === "ArrowLeft") {
      changeYear(-1);
      e.preventDefault();
    } else if (e.key === "ArrowRight") {
      changeYear(1);
      e.preventDefault();
    } else if (e.key === "a" || e.key === "A") {
      toggleOverlay();
      e.preventDefault();
    } else if (e.key === "b" || e.key === "B") {
      toggleBright();
      e.preventDefault();
    }
  });

  // --- Init ---
  function init() {
    populateCategorySelect(annotInputs.pre_category, CATEGORIES);
    populateCategorySelect(annotInputs.post_category, CATEGORIES);
    populateCategorySelect(annotInputs.pre_change_category, PRE_CHANGE_CATEGORIES);
    populateCategorySelect(annotInputs.post_change_category, POST_CHANGE_CATEGORIES);
    populateCategorySelect(annotInputs.same_change_category, SAME_CHANGE_CATEGORIES);
    populateCategorySelect(annotInputs.fine_change_category, FINE_CHANGE_CATEGORIES);
    fetchJSON("/api/entries").then(function (data) {
      entriesList = data;
      entryCount.textContent = data.length + " entries";
      if (data.length > 0) {
        var startIdx = 0;
        var hash = window.location.hash;
        if (hash && hash.startsWith("#/")) {
          var parsed = parseInt(hash.slice(2), 10);
          if (!isNaN(parsed) && parsed >= 0 && parsed < data.length) {
            startIdx = parsed;
          }
        }
        loadEntry(startIdx);
      } else {
        setNavButtons();
      }
    });
  }

  init();
})();
