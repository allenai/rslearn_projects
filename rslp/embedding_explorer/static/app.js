(function () {
    "use strict";

    // State
    let currentWindow = null;
    let points = [];
    let similarityImg = null;
    let similarityBounds = null;
    let similarityMode = null;
    let overlayLayer = null;
    let activeOverlay = null;
    let activeOverlayKey = null;

    // Map setup
    const map = L.map("map", { zoomControl: true }).setView([0, 0], 3);

    // Bing Maps tile layer (uses quadkey URLs).
    const BingTileLayer = L.TileLayer.extend({
        getTileUrl: function (coords) {
            const x = coords.x;
            const y = coords.y;
            const z = coords.z;
            let quadkey = "";
            for (let i = z; i > 0; i--) {
                let digit = 0;
                const mask = 1 << (i - 1);
                if ((x & mask) !== 0) digit += 1;
                if ((y & mask) !== 0) digit += 2;
                quadkey += digit.toString();
            }
            const sub = (x + y) % 4;
            return `https://ecn.t${sub}.tiles.virtualearth.net/tiles/a${quadkey}.jpeg?g=761`;
        },
    });

    function makeOsmLayer() {
        return L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "&copy; OSM contributors",
            maxZoom: 19,
        });
    }

    function makeBingLayer() {
        return new BingTileLayer("", {
            attribution: "&copy; Microsoft Bing Maps",
            maxZoom: 19,
        });
    }

    // DOM refs
    const windowSelect = document.getElementById("window-select");
    const layerToggles = document.getElementById("layer-toggles");
    const pointsList = document.getElementById("points-list");
    const clearBtn = document.getElementById("clear-points");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdValue = document.getElementById("threshold-value");
    const kInput = document.getElementById("k-input");
    const applyProbeBtn = document.getElementById("apply-probe");
    const loadingSpinner = document.getElementById("loading-spinner");
    const linearProbeGroup = document.getElementById("linear-probe-group");
    const kGroup = document.getElementById("k-group");
    const opacitySlider = document.getElementById("opacity-slider");
    const opacityValue = document.getElementById("opacity-value");
    const thresholdColor = document.getElementById("threshold-color");

    // Populate window dropdown
    Object.keys(WINDOWS_DATA).forEach((key) => {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = key;
        windowSelect.appendChild(opt);
    });

    windowSelect.addEventListener("change", () => {
        selectWindow(windowSelect.value);
    });

    function mercatorBoundsToLatLng(mb) {
        const sw = L.CRS.EPSG3857.unproject(L.point(mb[0], mb[1]));
        const ne = L.CRS.EPSG3857.unproject(L.point(mb[2], mb[3]));
        return L.latLngBounds(sw, ne);
    }

    const layerRadios = {};

    function selectWindow(key) {
        currentWindow = key;
        const win = WINDOWS_DATA[key];
        points = [];
        clearSimilarityOverlay();
        renderPoints();

        const bounds = mercatorBoundsToLatLng(win.mercator_bounds);
        map.flyToBounds(bounds, { padding: [20, 20] });

        const previousKey = activeOverlayKey;
        removeActiveOverlay();
        layerToggles.innerHTML = "";
        Object.keys(layerRadios).forEach((k) => delete layerRadios[k]);

        addLayerRadio("OpenStreetMap", { type: "tile", source: "osm" });
        addLayerRadio("Bing Maps", { type: "tile", source: "bing" });

        addLayerRadio(EMBEDDING_LAYER + " (RGB)", {
            type: "image",
            url: `/api/image/${key}/${EMBEDDING_LAYER}?bands=0,1,2`,
            mercatorBounds: win.mercator_bounds,
        });

        Object.entries(win.image_layers).forEach(([baseName, groups]) => {
            groups.forEach((layerEntry) => {
                const bands = baseName.includes("sentinel2") ? "3,2,1" : "0,1,2";
                addLayerRadio(layerEntry.name, {
                    type: "image",
                    url: `/api/image/${key}/${layerEntry.name}?bands=${bands}`,
                    mercatorBounds: layerEntry.mercator_bounds,
                });
            });
        });

        // Restore previous selection if still available, otherwise default to OSM.
        const targetKey =
            previousKey && layerRadios[previousKey]
                ? previousKey
                : "OpenStreetMap";
        selectLayerRadio(targetKey);
    }

    function selectLayerRadio(name) {
        const entry = layerRadios[name];
        if (!entry) return;
        entry.input.checked = true;
        applyOverlay(name, entry.spec);
    }

    function addLayerRadio(name, spec) {
        const div = document.createElement("div");
        div.className = "layer-toggle";
        const r = document.createElement("input");
        r.type = "radio";
        r.name = "layer-radio";
        r.id = `layer-${name}`;
        r.addEventListener("change", () => {
            if (r.checked) applyOverlay(name, spec);
        });
        const label = document.createElement("label");
        label.htmlFor = r.id;
        label.textContent = name;
        div.appendChild(r);
        div.appendChild(label);
        layerToggles.appendChild(div);
        layerRadios[name] = { input: r, spec: spec };
    }

    function applyOverlay(name, spec) {
        removeActiveOverlay();
        if (spec.type === "tile") {
            activeOverlay = (spec.source === "bing" ? makeBingLayer() : makeOsmLayer()).addTo(map);
        } else if (spec.type === "image") {
            const bounds = mercatorBoundsToLatLng(spec.mercatorBounds);
            activeOverlay = L.imageOverlay(spec.url, bounds, { opacity: 1.0 }).addTo(map);
        }
        activeOverlayKey = name;
        // Keep similarity overlay above the new image overlay.
        if (overlayLayer && overlayLayer.bringToFront) {
            overlayLayer.bringToFront();
        }
    }

    function removeActiveOverlay() {
        if (activeOverlay) {
            map.removeLayer(activeOverlay);
            activeOverlay = null;
        }
        activeOverlayKey = null;
    }

    // Map click -> add point
    map.on("click", (e) => {
        if (!currentWindow) return;
        points.push({ lat: e.latlng.lat, lon: e.latlng.lng, label: "positive" });
        renderPoints();
        maybeAutoRefresh();
    });

    map.on("contextmenu", (e) => {
        if (!currentWindow) return;
        e.originalEvent.preventDefault();
        points.push({ lat: e.latlng.lat, lon: e.latlng.lng, label: "negative" });
        renderPoints();
        maybeAutoRefresh();
    });

    // Points rendering
    let pointMarkers = [];

    function renderPoints() {
        pointMarkers.forEach((m) => map.removeLayer(m));
        pointMarkers = [];
        pointsList.innerHTML = "";

        points.forEach((p, i) => {
            const color = p.label === "positive" ? "#22c55e" : "#ef4444";
            const marker = L.circleMarker([p.lat, p.lon], {
                radius: 8,
                fillColor: color,
                color: "#fff",
                weight: 2,
                fillOpacity: 0.9,
            }).addTo(map);
            pointMarkers.push(marker);

            const div = document.createElement("div");
            div.className = "point-item";
            div.innerHTML = `
                <span class="point-label ${p.label}">${p.label === "positive" ? "+" : "-"}</span>
                <span class="point-coords">${p.lat.toFixed(4)}, ${p.lon.toFixed(4)}</span>
                <button class="point-toggle" data-idx="${i}">flip</button>
                <button class="point-delete" data-idx="${i}">&times;</button>
            `;
            pointsList.appendChild(div);
        });

        pointsList.querySelectorAll(".point-toggle").forEach((btn) => {
            btn.addEventListener("click", () => {
                const idx = parseInt(btn.dataset.idx);
                points[idx].label = points[idx].label === "positive" ? "negative" : "positive";
                renderPoints();
                maybeAutoRefresh();
            });
        });
        pointsList.querySelectorAll(".point-delete").forEach((btn) => {
            btn.addEventListener("click", () => {
                const idx = parseInt(btn.dataset.idx);
                points.splice(idx, 1);
                renderPoints();
                if (points.length > 0) {
                    maybeAutoRefresh();
                } else {
                    clearSimilarityOverlay();
                }
            });
        });

        updateApplyButton();
    }

    clearBtn.addEventListener("click", () => {
        points = [];
        renderPoints();
        clearSimilarityOverlay();
    });

    applyProbeBtn.addEventListener("click", () => {
        requestSimilarity();
    });

    // Similarity
    function getMode() {
        return document.querySelector('input[name="mode"]:checked').value;
    }

    function getDisplay() {
        return document.querySelector('input[name="display"]:checked').value;
    }

    function maybeAutoRefresh() {
        // Linear probe is only refreshed via the Apply button.
        if (getMode() === "linear_probe") return;
        if (points.length === 0) {
            clearSimilarityOverlay();
            return;
        }
        requestSimilarity();
    }

    function requestSimilarity() {
        if (!currentWindow || points.length === 0) return;

        const mode = getMode();
        const k = parseInt(kInput.value) || 3;

        if (mode === "cosine" && !points.some((p) => p.label === "positive")) return;
        if (mode === "linear_probe") {
            const hasPos = points.some((p) => p.label === "positive");
            const hasNeg = points.some((p) => p.label === "negative");
            if (!hasPos || !hasNeg) return;
        }

        const win = WINDOWS_DATA[currentWindow];
        similarityBounds = mercatorBoundsToLatLng(win.mercator_bounds);
        similarityMode = mode;

        const isProbe = mode === "linear_probe";
        if (isProbe) {
            loadingSpinner.style.display = "flex";
            applyProbeBtn.disabled = true;
        }

        fetch("/api/similarity", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                window: currentWindow,
                mode: mode,
                points: points,
                k: k,
            }),
        })
            .then((resp) => {
                if (!resp.ok) throw new Error(resp.statusText);
                return resp.blob();
            })
            .then((blob) => {
                const url = URL.createObjectURL(blob);
                const img = new window.Image();
                img.onload = () => {
                    similarityImg = img;
                    renderSimilarityOverlay();
                    if (isProbe) {
                        loadingSpinner.style.display = "none";
                        updateApplyButton();
                    }
                };
                img.src = url;
            })
            .catch((err) => {
                console.error("Similarity request failed:", err);
                if (isProbe) {
                    loadingSpinner.style.display = "none";
                    updateApplyButton();
                }
            });
    }

    function updateApplyButton() {
        const hasPos = points.some((p) => p.label === "positive");
        const hasNeg = points.some((p) => p.label === "negative");
        applyProbeBtn.disabled = !(hasPos && hasNeg);
    }

    function renderSimilarityOverlay() {
        if (!similarityImg || !similarityBounds) return;

        const canvas = document.createElement("canvas");
        canvas.width = similarityImg.width;
        canvas.height = similarityImg.height;
        const ctx = canvas.getContext("2d");

        ctx.drawImage(similarityImg, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const pixels = imageData.data;

        const display = getDisplay();
        const thresholdRaw = parseInt(thresholdSlider.value);
        const opacity = parseInt(opacitySlider.value) / 100.0;
        const [tr, tg, tb] = thresholdColor.value.split(",").map((v) => parseInt(v));

        const output = ctx.createImageData(canvas.width, canvas.height);
        const out = output.data;

        for (let i = 0; i < pixels.length; i += 4) {
            const gray = pixels[i];

            if (display === "threshold") {
                if (gray >= thresholdRaw) {
                    out[i] = tr;
                    out[i + 1] = tg;
                    out[i + 2] = tb;
                    out[i + 3] = Math.round(255 * opacity);
                } else {
                    out[i + 3] = 0;
                }
            } else {
                // Gradient: blue -> yellow -> red
                const t = gray / 255.0;
                let baseAlpha;
                if (t < 0.5) {
                    const s = t * 2;
                    out[i] = Math.round(s * 255);
                    out[i + 1] = Math.round(s * 255);
                    out[i + 2] = Math.round((1 - s) * 255);
                    baseAlpha = 50 + t * 300;
                } else {
                    const s = (t - 0.5) * 2;
                    out[i] = 255;
                    out[i + 1] = Math.round((1 - s) * 255);
                    out[i + 2] = 0;
                    baseAlpha = 100 + s * 155;
                }
                out[i + 3] = Math.min(255, Math.round(baseAlpha * opacity));
            }
        }

        ctx.putImageData(output, 0, 0);
        const dataUrl = canvas.toDataURL("image/png");

        if (overlayLayer) {
            map.removeLayer(overlayLayer);
        }
        overlayLayer = L.imageOverlay(dataUrl, similarityBounds, {
            opacity: 1.0,
            interactive: false,
        });
        if (showOverlayCheckbox.checked) {
            overlayLayer.addTo(map);
        }
    }

    function clearSimilarityOverlay() {
        similarityImg = null;
        similarityBounds = null;
        if (overlayLayer) {
            map.removeLayer(overlayLayer);
            overlayLayer = null;
        }
    }

    // Show/hide overlay toggle
    const showOverlayCheckbox = document.getElementById("show-overlay");
    showOverlayCheckbox.addEventListener("change", () => {
        if (overlayLayer) {
            if (showOverlayCheckbox.checked) {
                overlayLayer.addTo(map);
            } else {
                map.removeLayer(overlayLayer);
            }
        }
    });

    // Re-render on display setting changes (no network request)
    thresholdSlider.addEventListener("input", () => {
        const raw = parseInt(thresholdSlider.value);
        let displayVal;
        if (similarityMode === "cosine") {
            displayVal = ((raw / 255.0) * 2.0 - 1.0).toFixed(3);
        } else {
            displayVal = (raw / 255.0).toFixed(3);
        }
        thresholdValue.textContent = displayVal;
        renderSimilarityOverlay();
    });

    function syncDisplayUi() {
        const isThreshold = getDisplay() === "threshold";
        document.getElementById("threshold-color-group").style.display = isThreshold
            ? "block"
            : "none";
    }

    document.querySelectorAll('input[name="display"]').forEach((radio) => {
        radio.addEventListener("change", () => {
            syncDisplayUi();
            renderSimilarityOverlay();
        });
    });

    thresholdColor.addEventListener("change", renderSimilarityOverlay);

    opacitySlider.addEventListener("input", () => {
        opacityValue.textContent = `${opacitySlider.value}%`;
        renderSimilarityOverlay();
    });

    function syncModeUi() {
        const mode = getMode();
        kGroup.style.display = mode === "knn" ? "block" : "none";
        linearProbeGroup.style.display = mode === "linear_probe" ? "block" : "none";
        if (mode !== "linear_probe") {
            loadingSpinner.style.display = "none";
        }
    }

    document.querySelectorAll('input[name="mode"]').forEach((radio) => {
        radio.addEventListener("change", () => {
            syncModeUi();
            const mode = getMode();
            if (mode === "linear_probe") {
                // Switching into linear probe should not auto-refresh; clear any
                // stale overlay from the previous mode so the user re-applies.
                clearSimilarityOverlay();
                updateApplyButton();
            } else if (points.length > 0) {
                requestSimilarity();
            }
        });
    });

    kInput.addEventListener("change", () => {
        if (getMode() === "knn" && points.length > 0) requestSimilarity();
    });

    // Initial state
    syncModeUi();
    syncDisplayUi();
    loadingSpinner.style.display = "none";
    updateApplyButton();

    if (Object.keys(WINDOWS_DATA).length > 0) {
        windowSelect.value = Object.keys(WINDOWS_DATA)[0];
        selectWindow(windowSelect.value);
    }
})();
