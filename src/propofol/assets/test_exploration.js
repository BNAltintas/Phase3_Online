/*
 * Test Exploration page - fake-logic-only, browser-side computations.
 *
 * Everything in this file is deliberately disconnected from the real
 * model: no fetch/XHR calls, no imports from any Python module, no PK/PD
 * simulation. Every number here is a hand-picked, clinically-flavored
 * formula whose only job is to react smoothly and plausibly to whatever
 * the user changes on the Test Exploration page, so that page can be used
 * to prototype interaction/workflow ideas before they are ever built into
 * the real Scenario Exploration pipeline.
 *
 * The story this page tells: fill in a patient (opioid defaults to
 * "None") - all pure form state, nothing computes yet, and only the left
 * column is visible - then press "Run Scenario" (the page's single
 * trigger) to reveal the results area: the baseline propofol
 * recommendation (opioid at its own fixed anchor rate, blue) compared
 * against the explored recommendation (your chosen rate, orange - a
 * manual/what-if accent, not another baseline output), and the
 * physiological consequences on the right-hand prediction graphs. More
 * opioid always pulls propofol (and BIS/MAP) down; less opioid pushes
 * them back up. Editing the left column again immediately hides the
 * results area until "Run Scenario" is pressed again.
 */

(function () {
  "use strict";

  var GREEN = "#1f9254";
  var BLUE = "#2f6fed";
  var RED = "#d9362e";
  // The "explored/what-if" accent - matches the app-wide manual-override
  // orange used elsewhere in DosePilot, so the Explored Scenario column
  // and the dose-response graph's "Explored" line read as a manual/
  // what-if result, not another baseline/recommended output.
  var ORANGE = "#c2680f";
  var GRID = "#eef0f5";
  var AXIS_LINE = "#e2e5ec";
  var TEXT = "#1f2330";
  var TEXT_SECONDARY = "#6b7280";

  // Per-opioid slider bounds/units - mirrors TEST_EXPLORATION_OPIOID_DEFAULTS
  // in dashboard_layout.py (that Python copy only seeds the initial page
  // render for the default "remifentanil" selection; this is the source
  // of truth for every subsequent opioid switch).
  var TE_OPIOID_CONFIG = {
    remifentanil: { min: 0.02, max: 0.20, step: 0.01, baseline: 0.10, unit: "µg/kg/min", decimals: 2 },
    sufentanil: { min: 2, max: 15, step: 0.5, baseline: 6, unit: "µg/h", decimals: 1 },
    fentanyl: { min: 30, max: 150, step: 5, baseline: 80, unit: "µg/h", decimals: 0 },
  };

  function teOpioidConfig(opioid) {
    return TE_OPIOID_CONFIG[opioid] || TE_OPIOID_CONFIG.remifentanil;
  }

  function teClamp(value, lo, hi) {
    return Math.max(lo, Math.min(hi, value));
  }

  function teLinspace(start, end, n) {
    var arr = [];
    var step = (end - start) / (n - 1);
    for (var i = 0; i < n; i++) {
      arr.push(start + step * i);
    }
    return arr;
  }

  function teParseFloat(value, fallback) {
    var parsed = parseFloat(value);
    return isFinite(parsed) ? parsed : fallback;
  }

  function teCapitalize(word) {
    word = word || "";
    return word.charAt(0).toUpperCase() + word.slice(1);
  }

  function teDeltaText(pct) {
    if (Math.abs(pct) < 0.5) {
      return "→ 0%";
    }
    var arrow = pct < 0 ? "↓" : "↑";
    return arrow + " " + Math.abs(pct).toFixed(0) + "%";
  }

  function teDeltaClass(pct) {
    if (Math.abs(pct) < 0.5) {
      return "te-result-cell te-result-change-value";
    }
    return "te-result-cell te-result-change-value " + (pct < 0 ? "te-delta-good" : "te-delta-bad");
  }

  /* ---------- shared fake-logic state ---------- */

  // `rate` is the opioid infusion rate (in that opioid's own unit) this
  // particular computation should reflect - the fixed baseline anchor,
  // the committed "explored" rate, or one point along the dose-response
  // sweep, depending on the caller.
  function teOpioidReductionFactor(opioid, rate) {
    if (opioid === "none" || rate === null || rate === undefined) {
      return 0;
    }
    var cfg = teOpioidConfig(opioid);
    var t = teClamp((rate - cfg.min) / (cfg.max - cfg.min), 0, 1);
    return 0.05 + 0.3 * t;
  }

  function teComputeState(age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue, rate) {
    age = teParseFloat(age, 35);
    weight = teParseFloat(weight, 70);
    height = teParseFloat(height, 170);
    sbp = teParseFloat(sbp, 120);
    dbp = teParseFloat(dbp, 70);
    bisLow = teParseFloat(bisLow, 40);
    bisHigh = teParseFloat(bisHigh, 60);
    mapValue = teParseFloat(mapValue, 65);
    opioid = opioid || "none";

    var map = dbp + (sbp - dbp) / 3.0;

    // ---- fake propofol induction dose (mg/kg) ----
    var inductionBase = 2.0;
    inductionBase += -0.01 * (age - 40);
    inductionBase += 0.005 * (map - 80);
    inductionBase += (50 - bisLow) * 0.02;
    if (mapMode === "rel") {
      inductionBase += (mapValue - 70) * 0.02;
    } else {
      inductionBase += (mapValue - 65) * 0.01;
    }
    var reduction = teOpioidReductionFactor(opioid, rate);
    var induction = teClamp(inductionBase * (1 - reduction), 0.8, 3.0);
    var inductionTotalMg = induction * weight;

    // ---- fake propofol maintenance (0-1 pause, 1-8 rate1, 8-15 rate2) ----
    var PROP_CONC_MG_ML = 10.0;
    var propRate1MlH = 16.0 * induction + 6.0;
    var propRate2MlH = propRate1MlH * 0.94;
    var propRate1McgKgMin = (propRate1MlH * PROP_CONC_MG_ML * 1000.0) / 60.0 / weight;
    var propRate2McgKgMin = (propRate2MlH * PROP_CONC_MG_ML * 1000.0) / 60.0 / weight;

    // ---- opioid maintenance display (just formats the given rate) ----
    var opioidRateDisplay = null;
    if (opioid === "remifentanil") {
      var REMI_CONC_MCG_ML = 50.0;
      var mlH = (rate * weight * 60.0) / REMI_CONC_MCG_ML;
      opioidRateDisplay = mlH.toFixed(1) + " mL/h (" + (rate * 1000.0).toFixed(0) + " ng/kg/min)";
    } else if (opioid === "sufentanil") {
      opioidRateDisplay = rate.toFixed(1) + " µg/h";
    } else if (opioid === "fentanyl") {
      opioidRateDisplay = rate.toFixed(0) + " µg/h";
    }

    return {
      age: age, sex: sex, weight: weight, height: height, sbp: sbp, dbp: dbp, map: map,
      opioid: opioid, rate: rate, bisLow: bisLow, bisHigh: bisHigh,
      induction: induction, inductionTotalMg: inductionTotalMg,
      propRate1MlH: propRate1MlH, propRate2MlH: propRate2MlH,
      propRate1McgKgMin: propRate1McgKgMin, propRate2McgKgMin: propRate2McgKgMin,
      opioidRateDisplay: opioidRateDisplay,
    };
  }

  /* ---------- shared plot styling ---------- */

  function teAxis(title, extra) {
    var axis = {
      title: title,
      gridcolor: GRID,
      zeroline: false,
      showline: true,
      linecolor: AXIS_LINE,
    };
    if (extra) {
      for (var key in extra) {
        if (Object.prototype.hasOwnProperty.call(extra, key)) {
          axis[key] = extra[key];
        }
      }
    }
    return axis;
  }

  function teFont() {
    return { family: "Segoe UI, Arial, sans-serif", size: 12, color: TEXT };
  }

  function teLegend() {
    return { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0 };
  }

  function teBand(x, yLow, yHigh, name, color, showInLegend) {
    var xRev = x.slice().reverse();
    var yLowRev = yLow.slice().reverse();
    return {
      x: x.concat(xRev),
      y: yHigh.concat(yLowRev),
      fill: "toself",
      mode: "lines",
      line: { width: 0 },
      fillcolor: color,
      opacity: 0.18,
      name: name,
      showlegend: showInLegend !== false,
      hoverinfo: "skip",
    };
  }

  function teHLineShape(y, xref) {
    return {
      type: "line",
      xref: xref || "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: y,
      y1: y,
      line: { color: TEXT_SECONDARY, width: 1.5, dash: "dash" },
    };
  }

  function teHLineAnnotation(y, text) {
    return {
      xref: "paper",
      x: 1,
      xanchor: "left",
      yref: "y",
      y: y,
      text: text,
      showarrow: false,
      font: { size: 11, color: TEXT_SECONDARY },
    };
  }

  /* ---------- dose-response interaction (fake) - swept across the
     selected opioid's own infusion rate, not propofol dose. `cfg` is
     null when opioid === "none": there is no rate to sweep, so this
     falls back to flat MAP/BIS reference lines at the no-opioid
     baseline instead of hiding the graph entirely. ---------- */

  function teDoseResponseFigure(s, cfg, baselineRate, exploredRate) {
    var hasOpioid = !!cfg;
    var xs, mapCurve, bisCurve, xAxisTitle;

    if (hasOpioid) {
      var mapFloor = 55.0;
      var bisFloor = 25.0;
      var bisCeil = 85.0;
      xs = teLinspace(cfg.min, cfg.max, 61);
      mapCurve = xs.map(function (rate) {
        var t = (rate - cfg.min) / (cfg.max - cfg.min);
        return mapFloor + (s.map - mapFloor) * (1 - 0.55 * t);
      });
      bisCurve = xs.map(function (rate) {
        var t = (rate - cfg.min) / (cfg.max - cfg.min);
        return bisFloor + (bisCeil - bisFloor) * (1 - 0.6 * t);
      });
      xAxisTitle = teCapitalize(s.opioid) + " infusion rate (" + cfg.unit + ")";
    } else {
      // No opioid at all: nothing to sweep, so show the patient's own
      // no-opioid baseline as flat reference lines instead.
      xs = teLinspace(0, 1, 21);
      mapCurve = xs.map(function () {
        return s.map;
      });
      bisCurve = xs.map(function () {
        return 85.0;
      });
      xAxisTitle = "No opioid administered (propofol-only baseline)";
    }

    var xMin = xs[0];
    var xMax = xs[xs.length - 1];

    var targetBand = {
      x: [xMin, xMax, xMax, xMin],
      y: [s.bisHigh, s.bisHigh, s.bisLow, s.bisLow],
      fill: "toself",
      mode: "lines",
      line: { width: 0 },
      fillcolor: "rgba(31, 146, 84, 0.14)",
      name: "Target",
      yaxis: "y2",
      showlegend: true,
      hoverinfo: "skip",
    };

    var mapLine = { x: xs, y: mapCurve, mode: "lines", name: "MAP (mmHg)", line: { color: RED, width: 3 } };
    var bisLine = {
      x: xs, y: bisCurve, mode: "lines", name: "BIS", line: { color: BLUE, width: 3 }, yaxis: "y2",
    };

    var data = [targetBand, mapLine, bisLine];

    if (hasOpioid) {
      data.push({
        x: [baselineRate, baselineRate], y: [0, 100], mode: "lines",
        name: "Baseline (" + baselineRate.toFixed(cfg.decimals) + " " + cfg.unit + ")",
        line: { color: GREEN, width: 2, dash: "dash" }, yaxis: "y2", hoverinfo: "skip",
      });
      // exploredRate is null until "Run Scenario" has actually been
      // clicked - omit the "Explored" line entirely until then, rather
      // than drawing it on top of (and hiding) the baseline line.
      if (exploredRate !== null && exploredRate !== undefined) {
        data.push({
          x: [exploredRate, exploredRate], y: [0, 100], mode: "lines",
          name: "Explored (" + exploredRate.toFixed(cfg.decimals) + " " + cfg.unit + ")",
          line: { color: ORANGE, width: 2.5 }, yaxis: "y2", hoverinfo: "skip",
        });
      }
    }

    var layout = {
      xaxis: teAxis(xAxisTitle, hasOpioid ? null : { showticklabels: false }),
      yaxis: teAxis("MAP (mmHg)", { range: [40, 110] }),
      yaxis2: teAxis("BIS", { overlaying: "y", side: "right", range: [0, 100], gridcolor: "transparent" }),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 40, r: 50, b: 50, l: 55 },
    };

    return { data: data, layout: layout };
  }

  /* ---------- predictions (fake) - reflect the explored scenario ---------- */

  function teBisFigure(s) {
    var t = teLinspace(0, 15, 61);
    var floor = Math.max(20, s.bisLow - 8);
    var y = t.map(function (tt) {
      return floor + (92 - floor) * Math.exp(-tt / 1.3);
    });
    var yLow = y.map(function (v) {
      return Math.max(0, v - 6);
    });
    var yHigh = y.map(function (v) {
      return Math.min(100, v + 6);
    });

    var layout = {
      xaxis: teAxis("Time (min)"),
      yaxis: teAxis("BIS", { range: [0, 100] }),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 40, r: 85, b: 45, l: 55 },
      shapes: [teHLineShape(s.bisLow), teHLineShape(s.bisHigh)],
      annotations: [teHLineAnnotation(s.bisLow, "BIS " + s.bisLow.toFixed(0)), teHLineAnnotation(s.bisHigh, "BIS " + s.bisHigh.toFixed(0))],
    };

    return {
      data: [
        teBand(t, yLow, yHigh, "90% Prediction Interval", "rgba(47, 111, 237, 0.35)"),
        { x: t, y: y, mode: "lines", name: "Predicted BIS", line: { color: GREEN, width: 2.5 } },
      ],
      layout: layout,
    };
  }

  function teMapFigure(s) {
    var t = teLinspace(0, 15, 61);
    var floor = s.map - 22;
    var y = t.map(function (tt) {
      return floor + (s.map - floor) * Math.exp(-tt / 3.0);
    });
    var yLow = y.map(function (v) {
      return v - 8;
    });
    var yHigh = y.map(function (v) {
      return v + 8;
    });
    var lowerBound = s.map * 0.72;

    var layout = {
      xaxis: teAxis("Time (min)"),
      yaxis: teAxis("MAP (mmHg)", { range: [0, Math.max(160, s.map + 30)] }),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 40, r: 100, b: 45, l: 55 },
      shapes: [teHLineShape(lowerBound)],
      annotations: [teHLineAnnotation(lowerBound, "MAP lower bound")],
    };

    return {
      data: [
        teBand(t, yLow, yHigh, "90% Prediction Interval", "rgba(47, 111, 237, 0.35)"),
        { x: t, y: y, mode: "lines", name: "Predicted MAP", line: { color: GREEN, width: 2.5 } },
      ],
      layout: layout,
    };
  }

  function tePropofolPkFigure(s) {
    var t = teLinspace(0, 15, 61);
    var peakCp = 3.0 + s.induction * 1.5;
    var plateauCp = 1.0 + s.propRate1McgKgMin / 60.0;
    var cp = t.map(function (tt) {
      return plateauCp + (peakCp - plateauCp) * Math.exp(-tt / 1.2);
    });
    var ce = t.map(function (tt) {
      return plateauCp + (peakCp - plateauCp) * 0.75 * (Math.exp(-tt / 3.5) - Math.exp(-tt / 0.7));
    });
    var cpLow = cp.map(function (v) {
      return Math.max(0, v * 0.85);
    });
    var cpHigh = cp.map(function (v) {
      return v * 1.15;
    });

    var layout = {
      xaxis: teAxis("Time (min)"),
      yaxis: teAxis("Propofol concentration (mcg/mL)"),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 40, r: 20, b: 45, l: 55 },
    };

    return {
      data: [
        teBand(t, cpLow, cpHigh, "90% Prediction Interval", "rgba(47, 111, 237, 0.35)"),
        { x: t, y: cp, mode: "lines", name: "Predicted Cp", line: { color: GREEN, width: 2.5 } },
        { x: t, y: ce, mode: "lines", name: "Predicted Ce", line: { color: GREEN, width: 2.5, dash: "dash" } },
      ],
      layout: layout,
    };
  }

  // For opioid === "none" this renders a flat zero-concentration line
  // rather than an empty/hidden graph - "no opioid administered" is
  // itself a valid, displayable result of the scenario that was run.
  function teOpioidPkFigure(s) {
    var t = teLinspace(0, 15, 61);
    var hasOpioid = s.opioid !== "none";
    var plateau = hasOpioid ? { remifentanil: 2.4, sufentanil: 0.4, fentanyl: 3.2 }[s.opioid] || 1.0 : 0;
    var y = t.map(function (tt) {
      return hasOpioid ? plateau * (1 - Math.exp(-tt / 3.5)) : 0;
    });
    var drugLabel = hasOpioid ? teCapitalize(s.opioid) : "Opioid";

    var layout = {
      xaxis: teAxis("Time (min)"),
      yaxis: teAxis(drugLabel + " concentration (ng/mL)", hasOpioid ? null : { range: [0, 1] }),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 40, r: 20, b: 45, l: 55 },
    };

    if (!hasOpioid) {
      return {
        data: [{ x: t, y: y, mode: "lines", name: "Predicted Cp (no opioid)", line: { color: GREEN, width: 2.5 } }],
        layout: layout,
      };
    }

    var yLow = y.map(function (v) {
      return Math.max(0, v * 0.85);
    });
    var yHigh = y.map(function (v) {
      return v * 1.15;
    });

    return {
      data: [
        teBand(t, yLow, yHigh, "90% Prediction Interval", "rgba(47, 111, 237, 0.35)"),
        { x: t, y: y, mode: "lines", name: "Predicted Cp", line: { color: GREEN, width: 2.5 } },
      ],
      layout: layout,
    };
  }

  /* ---------- Dash clientside entry points ---------- */

  // Builds a plain {age, sex, ..., mapValue} snapshot object from raw
  // form values, coercing/defaulting each field the same way
  // teComputeState itself would - shared by runScenario (which writes
  // this into te-committed-scenario-store) and every reader of that
  // store (which can then trust its shape without re-parsing).
  function teBuildSnapshot(age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue) {
    return {
      age: teParseFloat(age, 35), sex: sex || "male",
      height: teParseFloat(height, 170), weight: teParseFloat(weight, 70),
      sbp: teParseFloat(sbp, 120), dbp: teParseFloat(dbp, 70),
      opioid: opioid || "none",
      bisLow: teParseFloat(bisLow, 40), bisHigh: teParseFloat(bisHigh, 60),
      mapMode: mapMode || "abs", mapValue: teParseFloat(mapValue, 65),
    };
  }

  window.dash_clientside = window.dash_clientside || {};
  window.dash_clientside.testExploration = {
    // "Run Scenario" click: commits the left column's current values into
    // te-committed-scenario-store, resets the Explore-strategy slider to
    // the newly-committed opioid's own baseline rate/bounds/unit, and
    // switches the page into State B (te-scenario-active-store = true,
    // revealing te-results-area). Computes nothing else directly -
    // committing the store triggers computeBaseline, and resetting the
    // slider's value triggers updateExploredLive, so the rest of the
    // card fills in through that natural chain reaction.
    runScenario: function (_nClicks, age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue) {
      var noUpdate = window.dash_clientside.no_update;
      var snapshot = teBuildSnapshot(age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue);
      var visible = snapshot.opioid !== "none";

      if (!visible) {
        return [snapshot, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, true];
      }

      var cfg = teOpioidConfig(snapshot.opioid);
      var marks = {};
      marks[cfg.min] = cfg.min.toFixed(cfg.decimals);
      marks[cfg.max] = cfg.max.toFixed(cfg.decimals);
      var label = teCapitalize(snapshot.opioid) + " infusion rate";
      var valueText = cfg.baseline.toFixed(cfg.decimals) + " " + cfg.unit;

      return [snapshot, cfg.min, cfg.max, cfg.step, cfg.baseline, marks, label, valueText, true];
    },

    // Baseline column + section visibility - driven solely by
    // te-committed-scenario-store (fires on page load with this page's
    // default snapshot, and again every "Run Scenario" click). Never
    // reacts to the live left-column fields directly.
    computeBaseline: function (committed) {
      var c = committed || {};
      var opioid = c.opioid || "none";
      var visible = opioid !== "none";
      var rate = visible ? teOpioidConfig(opioid).baseline : 0;
      var s = teComputeState(c.age, c.sex, c.height, c.weight, c.sbp, c.dbp, opioid, c.bisLow, c.bisHigh, c.mapMode, c.mapValue, rate);

      var inductionText = s.induction.toFixed(2) + " mg/kg";
      var totalText = "(" + s.inductionTotalMg.toFixed(1) + " mg total)";
      var propRate1Text = s.propRate1MlH.toFixed(0) + " mL/h (" + s.propRate1McgKgMin.toFixed(0) + " µg/kg/min)";
      var propRate2Text = s.propRate2MlH.toFixed(0) + " mL/h (" + s.propRate2McgKgMin.toFixed(0) + " µg/kg/min)";
      var opioidTitle = visible ? teCapitalize(opioid) + " – Early maintenance (0–15 min)" : "-";
      var remiRateText = visible ? s.opioidRateDisplay : "-";
      var exploreCardStyle = visible ? null : { display: "none" };
      var opioidSectionStyle = visible ? null : { display: "none" };

      return [inductionText, totalText, propRate1Text, propRate2Text, opioidTitle, remiRateText, exploreCardStyle, opioidSectionStyle];
    },

    // The Explore-strategy slider's live-update path - fires on every
    // drag tick, reading te-committed-scenario-store (never the live
    // left-column fields) so it always reflects the last committed
    // patient/opioid, even while an unrelated left-column edit is
    // pending. No "Run Scenario" click needed for this to update.
    updateExploredLive: function (rateValue, committed) {
      var c = committed || {};
      var opioid = c.opioid || "none";
      var visible = opioid !== "none";
      var cfg = visible ? teOpioidConfig(opioid) : null;
      var baselineRate = visible ? cfg.baseline : 0;
      var exploredRate = visible ? teParseFloat(rateValue, cfg.baseline) : 0;

      var sBase = teComputeState(c.age, c.sex, c.height, c.weight, c.sbp, c.dbp, opioid, c.bisLow, c.bisHigh, c.mapMode, c.mapValue, baselineRate);
      var sExp = teComputeState(c.age, c.sex, c.height, c.weight, c.sbp, c.dbp, opioid, c.bisLow, c.bisHigh, c.mapMode, c.mapValue, exploredRate);

      var expInductionText = sExp.induction.toFixed(2) + " mg/kg";
      var expTotalText = "(" + sExp.inductionTotalMg.toFixed(1) + " mg total)";
      var expPropRate1Text = sExp.propRate1MlH.toFixed(0) + " mL/h (" + sExp.propRate1McgKgMin.toFixed(0) + " µg/kg/min)";
      var expPropRate2Text = sExp.propRate2MlH.toFixed(0) + " mL/h (" + sExp.propRate2McgKgMin.toFixed(0) + " µg/kg/min)";
      var expRemiRateText = visible ? sExp.opioidRateDisplay : "-";

      var propofolDeltaPct = ((sExp.induction - sBase.induction) / sBase.induction) * 100;
      var opioidDeltaPct = !visible || baselineRate === 0 ? 0 : ((exploredRate - baselineRate) / baselineRate) * 100;

      var subtitle = visible
        ? "How " + opioid + " infusion rate affects MAP and BIS"
        : "Propofol-only baseline (no opioid administered)";
      var doseResponseFig = teDoseResponseFigure(sExp, cfg, baselineRate, visible ? exploredRate : null);
      var opioidPkFig = teOpioidPkFigure(sExp);

      return [
        expInductionText, "te-result-explored-induction-value", expTotalText,
        "Pause", expPropRate1Text, expPropRate2Text, "Pause", expRemiRateText,
        teDeltaText(propofolDeltaPct), teDeltaClass(propofolDeltaPct),
        teDeltaText(opioidDeltaPct), teDeltaClass(opioidDeltaPct),
        doseResponseFig, subtitle,
        teBisFigure(sExp), teMapFigure(sExp), tePropofolPkFigure(sExp), opioidPkFig,
      ];
    },

    updateExploreLiveLabel: function (rateValue, committed) {
      var opioid = (committed && committed.opioid) || "remifentanil";
      var cfg = teOpioidConfig(opioid);
      var rate = teParseFloat(rateValue, cfg.baseline);
      return rate.toFixed(cfg.decimals) + " " + cfg.unit;
    },

    // Editing ANY Patient/Medication/Targets field while a scenario is
    // active immediately marks it stale - the page's two-state switch
    // (te-scenario-active-store) flips back to false, which
    // renderTeResultsArea below turns into hiding the whole results area
    // (Scenario Result, Explore Opioid Strategy, Dose-Response, and the
    // 4 prediction graphs). No computation happens here at all.
    markScenarioStale: function () {
      return false;
    },

    // The page's two-state switch: shows/hides te-results-area based on
    // te-scenario-active-store alone. The left column lives outside this
    // container entirely, so it is never affected by this toggle.
    renderTeResultsArea: function (active) {
      return active ? null : { display: "none" };
    },
  };
})();
