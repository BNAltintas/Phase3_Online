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
 * trigger) to reveal the results area: the Baseline Recommendation
 * (whatever opioid the left column held at that moment, green) plus an
 * Explore Opioid Strategy card with its own, fully independent opioid
 * dropdown - choosing an opioid there and moving its rate slider away
 * from the recommended value it starts at reveals the Explored Scenario
 * (orange - a manual/what-if accent, not another baseline output) side
 * by side with the baseline, on every graph and in the comparison table.
 * The baseline and explored opioid never have to match - you can explore
 * "what if we gave remifentanil instead" even when the baseline itself
 * used no opioid at all. More opioid always pulls propofol (and BIS/MAP)
 * down; less opioid pushes them back up. Editing the left column again
 * immediately hides the results area until "Run Scenario" is pressed
 * again.
 */

(function () {
  "use strict";

  // Green = Baseline Recommendation, Orange = Explored Scenario -
  // applied consistently across every Test Exploration graph and the
  // Scenario Result card alike. RED/BLUE are a separate axis (MAP vs
  // BIS variable identity) used only by the Dose-Response sweep curves,
  // where the green/orange distinction is carried by the two vertical
  // reference lines instead.
  var GREEN = "#1f9254";
  var ORANGE = "#c2680f";
  var ORANGE_BAND = "rgba(194, 104, 15, 0.22)";
  var BLUE = "#2f6fed";
  var RED = "#d9362e";
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
      // Both baselineRate and exploredRate are null whenever the vertical
      // line they'd represent doesn't apply to this sweep - e.g. the
      // swept opioid isn't the baseline's own opioid, or no exploration
      // is active yet - so each line is omitted independently rather
      // than drawn on top of (and hiding) the other.
      if (baselineRate !== null && baselineRate !== undefined) {
        data.push({
          x: [baselineRate, baselineRate], y: [0, 100], mode: "lines",
          name: "Baseline Recommendation (" + baselineRate.toFixed(cfg.decimals) + " " + cfg.unit + ")",
          line: { color: GREEN, width: 2, dash: "dash" }, yaxis: "y2", hoverinfo: "skip",
        });
      }
      if (exploredRate !== null && exploredRate !== undefined) {
        data.push({
          x: [exploredRate, exploredRate], y: [0, 100], mode: "lines",
          name: "Explored Scenario (" + exploredRate.toFixed(cfg.decimals) + " " + cfg.unit + ")",
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

  /* ---------- predictions (fake) - each graph always shows the Baseline
     Recommendation (green). The Explored Scenario (orange) - and its
     confidence band - is added on top only when `sExp` is non-null,
     i.e. an opioid has been chosen to explore AND its rate has actually
     been moved away from the recommended value (see updateExploredLive):
     before that, these graphs show baseline-only, no orange trace, no
     "Explored Scenario" legend entry at all. ---------- */

  // More propofol (higher induction) -> deeper/lower steady-state BIS;
  // less propofol -> lighter/higher BIS. Still anchored to this
  // scenario's own BIS targets, same as before.
  function teBisFloor(s) {
    var base = Math.max(20, s.bisLow - 8);
    var shift = (s.induction - 2.0) * 12;
    return teClamp(base - shift, 15, 90);
  }

  function teBisCurve(s, t) {
    var floor = teBisFloor(s);
    return t.map(function (tt) {
      return floor + (92 - floor) * Math.exp(-tt / 1.3);
    });
  }

  function teBisFigure(sBase, sExp) {
    var t = teLinspace(0, 15, 61);
    var yBase = teBisCurve(sBase, t);

    var layout = {
      xaxis: teAxis("Time (min)"),
      yaxis: teAxis("BIS", { range: [0, 100] }),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 26, r: 65, b: 34, l: 46 },
      shapes: [teHLineShape(sBase.bisLow), teHLineShape(sBase.bisHigh)],
      annotations: [teHLineAnnotation(sBase.bisLow, "BIS " + sBase.bisLow.toFixed(0)), teHLineAnnotation(sBase.bisHigh, "BIS " + sBase.bisHigh.toFixed(0))],
    };

    var data = [{ x: t, y: yBase, mode: "lines", name: "Baseline Recommendation", line: { color: GREEN, width: 2 } }];

    if (sExp) {
      var yExp = teBisCurve(sExp, t);
      var yLow = yExp.map(function (v) {
        return Math.max(0, v - 6);
      });
      var yHigh = yExp.map(function (v) {
        return Math.min(100, v + 6);
      });
      data = [
        teBand(t, yLow, yHigh, "90% Prediction Interval", ORANGE_BAND),
        data[0],
        { x: t, y: yExp, mode: "lines", name: "Explored Scenario", line: { color: ORANGE, width: 2.5 } },
      ];
    }

    return { data: data, layout: layout };
  }

  // The opioid infusion rate pulls MAP down, same relationship the
  // Dose-Response sweep uses - so Baseline and Explored genuinely
  // diverge here whenever their rates differ.
  function teMapStartValue(s) {
    if (s.opioid === "none" || s.rate === null || s.rate === undefined) {
      return s.map;
    }
    var cfg = teOpioidConfig(s.opioid);
    var t = teClamp((s.rate - cfg.min) / (cfg.max - cfg.min), 0, 1);
    var mapFloorRate = 55.0;
    return mapFloorRate + (s.map - mapFloorRate) * (1 - 0.55 * t);
  }

  function teMapCurve(s, t) {
    var start = teMapStartValue(s);
    var floor = start - 22;
    return t.map(function (tt) {
      return floor + (start - floor) * Math.exp(-tt / 3.0);
    });
  }

  function teMapFigure(sBase, sExp) {
    var t = teLinspace(0, 15, 61);
    var yBase = teMapCurve(sBase, t);
    var lowerBound = sBase.map * 0.72;

    var layout = {
      xaxis: teAxis("Time (min)"),
      yaxis: teAxis("MAP (mmHg)", { range: [0, Math.max(160, sBase.map + 30)] }),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 26, r: 96, b: 34, l: 46 },
      shapes: [teHLineShape(lowerBound)],
      annotations: [teHLineAnnotation(lowerBound, "MAP lower bound")],
    };

    var data = [{ x: t, y: yBase, mode: "lines", name: "Baseline Recommendation", line: { color: GREEN, width: 2 } }];

    if (sExp) {
      var yExp = teMapCurve(sExp, t);
      var yLow = yExp.map(function (v) {
        return v - 8;
      });
      var yHigh = yExp.map(function (v) {
        return v + 8;
      });
      data = [
        teBand(t, yLow, yHigh, "90% Prediction Interval", ORANGE_BAND),
        data[0],
        { x: t, y: yExp, mode: "lines", name: "Explored Scenario", line: { color: ORANGE, width: 2.5 } },
      ];
    }

    return { data: data, layout: layout };
  }

  function tePropofolCurves(s, t) {
    var peakCp = 3.0 + s.induction * 1.5;
    var plateauCp = 1.0 + s.propRate1McgKgMin / 60.0;
    var cp = t.map(function (tt) {
      return plateauCp + (peakCp - plateauCp) * Math.exp(-tt / 1.2);
    });
    var ce = t.map(function (tt) {
      return plateauCp + (peakCp - plateauCp) * 0.75 * (Math.exp(-tt / 3.5) - Math.exp(-tt / 0.7));
    });
    return { cp: cp, ce: ce };
  }

  function tePropofolPkFigure(sBase, sExp) {
    var t = teLinspace(0, 15, 61);
    var base = tePropofolCurves(sBase, t);

    var layout = {
      xaxis: teAxis("Time (min)"),
      yaxis: teAxis("Propofol concentration (mcg/mL)"),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 26, r: 16, b: 34, l: 46 },
    };

    // Cp and Ce each get their own line (solid vs dashed) so both curves
    // stay visually distinguishable, but share one legend entry per
    // scenario (legendgroup + showlegend:false on the Ce trace) - 3
    // legend entries total, same count/pattern as every other prediction
    // graph on this page. 5 separately-named entries wrapped onto 2 lines
    // here and Plotly fell back to its own scrollable-legend widget
    // instead of just showing both lines, however much top margin was
    // given - merging them is the reliable fix, not a margin tweak.
    var data = [
      { x: t, y: base.cp, mode: "lines", name: "Baseline Recommendation", legendgroup: "te-baseline", line: { color: GREEN, width: 2 } },
      { x: t, y: base.ce, mode: "lines", name: "Baseline Recommendation", legendgroup: "te-baseline", showlegend: false, line: { color: GREEN, width: 2, dash: "dash" } },
    ];

    if (sExp) {
      var exp = tePropofolCurves(sExp, t);
      var cpLow = exp.cp.map(function (v) {
        return Math.max(0, v * 0.85);
      });
      var cpHigh = exp.cp.map(function (v) {
        return v * 1.15;
      });
      data = [
        teBand(t, cpLow, cpHigh, "90% Prediction Interval", ORANGE_BAND),
        data[0],
        data[1],
        { x: t, y: exp.cp, mode: "lines", name: "Explored Scenario", legendgroup: "te-explored", line: { color: ORANGE, width: 2.5 } },
        { x: t, y: exp.ce, mode: "lines", name: "Explored Scenario", legendgroup: "te-explored", showlegend: false, line: { color: ORANGE, width: 2.5, dash: "dash" } },
      ];
    }

    return { data: data, layout: layout };
  }

  // More opioid rate -> higher steady-state concentration (scaled off
  // this opioid's own configured rate range), so Baseline and Explored
  // diverge whenever their rates differ. For opioid === "none" both are
  // flat zero - "no opioid administered" is itself a valid, displayable
  // result of the scenario that was run.
  function teOpioidPkPlateau(s) {
    if (s.opioid === "none" || s.rate === null || s.rate === undefined) {
      return 0;
    }
    var cfg = teOpioidConfig(s.opioid);
    var t = teClamp((s.rate - cfg.min) / (cfg.max - cfg.min), 0, 1);
    var basePlateau = { remifentanil: 1.2, sufentanil: 0.2, fentanyl: 1.6 }[s.opioid] || 0.5;
    return basePlateau + basePlateau * 2.0 * t;
  }

  function teOpioidPkCurve(s, t) {
    var plateau = teOpioidPkPlateau(s);
    return t.map(function (tt) {
      return plateau * (1 - Math.exp(-tt / 3.5));
    });
  }

  function teOpioidPkFigure(sBase, sExp) {
    var t = teLinspace(0, 15, 61);
    // Whichever opioid is actually driving a visible curve names the
    // axis - the explored one when active, else the baseline's own
    // (which may itself be "none", giving a flat zero-concentration
    // line - "no opioid administered" is a valid, displayable result).
    var primaryOpioid = sExp && sExp.opioid !== "none" ? sExp.opioid : sBase.opioid;
    var hasOpioid = primaryOpioid !== "none";
    var drugLabel = hasOpioid ? teCapitalize(primaryOpioid) : "Opioid";
    var yBase = teOpioidPkCurve(sBase, t);

    var layout = {
      xaxis: teAxis("Time (min)"),
      yaxis: teAxis(drugLabel + " concentration (ng/mL)", hasOpioid ? null : { range: [0, 1] }),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 26, r: 16, b: 34, l: 46 },
    };

    var data = [{ x: t, y: yBase, mode: "lines", name: "Baseline Recommendation", line: { color: GREEN, width: 2 } }];

    if (sExp) {
      var yExp = teOpioidPkCurve(sExp, t);
      var yLow = yExp.map(function (v) {
        return Math.max(0, v * 0.85);
      });
      var yHigh = yExp.map(function (v) {
        return v * 1.15;
      });
      data = [
        teBand(t, yLow, yHigh, "90% Prediction Interval", ORANGE_BAND),
        data[0],
        { x: t, y: yExp, mode: "lines", name: "Explored Scenario", line: { color: ORANGE, width: 2.5 } },
      ];
    }

    return { data: data, layout: layout };
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
    // te-committed-scenario-store (this is the Baseline Recommendation
    // from now on - whatever opioid the left column held at click time,
    // including "none"), resets the independent Explore-opioid dropdown
    // back to "none" (a fresh scenario always starts with no exploration
    // active - see onExploreOpioidChange below, which reacts to that
    // reset by hiding the rate slider again), and switches the page into
    // State B (te-scenario-active-store = true, revealing
    // te-results-area). Computes nothing else directly - committing the
    // store triggers computeBaseline, and resetting the dropdown triggers
    // onExploreOpioidChange -> updateExploredLive, so the rest of the
    // card fills in through that natural chain reaction.
    runScenario: function (_nClicks, age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue) {
      var snapshot = teBuildSnapshot(age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue);
      return [snapshot, "none", true];
    },

    // Baseline column - driven solely by te-committed-scenario-store
    // (fires on page load with this page's default snapshot, and again
    // every "Run Scenario" click). Never reacts to the live left-column
    // fields, and never reacts to the (fully independent) Explore-opioid
    // dropdown - this is the Baseline Recommendation, unaffected by
    // whatever is being explored.
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
      var remiRateText = visible ? s.opioidRateDisplay : "-";

      return [inductionText, totalText, propRate1Text, propRate2Text, remiRateText];
    },

    // Fires whenever the Explore Opioid Strategy card's own opioid
    // dropdown changes (including runScenario resetting it to "none").
    // "None" hides the rate slider entirely (nothing to explore).
    // Any real opioid reconfigures + reveals the slider, always
    // reinitialized exactly at that opioid's own recommended rate - the
    // starting point updateExploredLive treats as "not explored yet".
    onExploreOpioidChange: function (exploreOpioidValue) {
      var noUpdate = window.dash_clientside.no_update;
      var opioid = exploreOpioidValue || "none";

      if (opioid === "none") {
        return [noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, "", "", { display: "none" }];
      }

      var cfg = teOpioidConfig(opioid);
      var marks = {};
      marks[cfg.min] = cfg.min.toFixed(cfg.decimals);
      marks[cfg.max] = cfg.max.toFixed(cfg.decimals);
      var title = teCapitalize(opioid) + " infusion rate";
      var label = cfg.baseline.toFixed(cfg.decimals) + " " + cfg.unit;

      return [cfg.min, cfg.max, cfg.step, cfg.baseline, marks, title, label, null];
    },

    // The Explore-strategy live-update path - fires on every slider drag
    // tick, every Explore-opioid dropdown change, and every "Run
    // Scenario" commit. Reads te-committed-scenario-store for the
    // Baseline Recommendation (never the live left-column fields) and
    // the live Explore-opioid dropdown + slider for the Explored
    // Scenario - two fully independent opioid choices. "Explored" only
    // becomes active once an opioid is chosen here AND its rate has
    // actually been moved away from the recommended value it was reset
    // to; until then every Explored cell/graph shows baseline-only.
    updateExploredLive: function (rateValue, committed, exploreOpioidValue) {
      var c = committed || {};
      var baseOpioid = c.opioid || "none";
      var baseVisible = baseOpioid !== "none";
      var baseCfg = baseVisible ? teOpioidConfig(baseOpioid) : null;
      var baselineRate = baseVisible ? baseCfg.baseline : 0;
      var sBase = teComputeState(c.age, c.sex, c.height, c.weight, c.sbp, c.dbp, baseOpioid, c.bisLow, c.bisHigh, c.mapMode, c.mapValue, baselineRate);

      var exploreOpioid = exploreOpioidValue || "none";
      var exploreVisible = exploreOpioid !== "none";
      var exploreCfg = exploreVisible ? teOpioidConfig(exploreOpioid) : null;
      var recommendedExploreRate = exploreVisible ? exploreCfg.baseline : 0;
      var exploredRateRaw = exploreVisible ? teParseFloat(rateValue, exploreCfg.baseline) : 0;
      var active = exploreVisible && Math.abs(exploredRateRaw - recommendedExploreRate) > 1e-9;

      var sExp = active
        ? teComputeState(c.age, c.sex, c.height, c.weight, c.sbp, c.dbp, exploreOpioid, c.bisLow, c.bisHigh, c.mapMode, c.mapValue, exploredRateRaw)
        : null;

      var expInductionText, expInductionClass, expTotalText, expPropRate1Text, expPropRate2Text, expRemiRateText;
      var propofolDeltaText, propofolDeltaClass, opioidDeltaText, opioidDeltaClass;
      var propRate1DeltaText, propRate1DeltaClass, propRate2DeltaText, propRate2DeltaClass;

      if (sExp) {
        expInductionText = sExp.induction.toFixed(2) + " mg/kg";
        expInductionClass = "te-result-explored-induction-value";
        expTotalText = "(" + sExp.inductionTotalMg.toFixed(1) + " mg total)";
        expPropRate1Text = sExp.propRate1MlH.toFixed(0) + " mL/h (" + sExp.propRate1McgKgMin.toFixed(0) + " µg/kg/min)";
        expPropRate2Text = sExp.propRate2MlH.toFixed(0) + " mL/h (" + sExp.propRate2McgKgMin.toFixed(0) + " µg/kg/min)";
        expRemiRateText = sExp.opioidRateDisplay;

        var propofolDeltaPct = ((sExp.induction - sBase.induction) / sBase.induction) * 100;
        propofolDeltaText = teDeltaText(propofolDeltaPct);
        propofolDeltaClass = teDeltaClass(propofolDeltaPct);

        // Propofol early-maintenance rows track the same induction-driven
        // rate (see teComputeState) - both are real numbers whenever
        // sExp exists, so these always get a percent change, never "–".
        var propRate1DeltaPct = ((sExp.propRate1MlH - sBase.propRate1MlH) / sBase.propRate1MlH) * 100;
        propRate1DeltaText = teDeltaText(propRate1DeltaPct);
        propRate1DeltaClass = teDeltaClass(propRate1DeltaPct);
        var propRate2DeltaPct = ((sExp.propRate2MlH - sBase.propRate2MlH) / sBase.propRate2MlH) * 100;
        propRate2DeltaText = teDeltaText(propRate2DeltaPct);
        propRate2DeltaClass = teDeltaClass(propRate2DeltaPct);

        if (baseVisible && baseOpioid === exploreOpioid) {
          var opioidDeltaPct = baselineRate === 0 ? 0 : ((exploredRateRaw - baselineRate) / baselineRate) * 100;
          opioidDeltaText = teDeltaText(opioidDeltaPct);
          opioidDeltaClass = teDeltaClass(opioidDeltaPct);
        } else {
          // Baseline and explored are different drugs (or there's no
          // baseline opioid at all) - a percentage change between two
          // different opioids' rates isn't a meaningful number, so show
          // a flat dash instead of a misleading one.
          opioidDeltaText = "–";
          opioidDeltaClass = "te-result-cell te-result-change-value";
        }
      } else {
        expInductionText = !exploreVisible
          ? "Select an opioid to begin exploring alternative strategies."
          : "Move the slider to explore an alternative strategy.";
        expInductionClass = "te-result-explored-induction-value te-result-explored-value--placeholder";
        expTotalText = "";
        expPropRate1Text = "–";
        expPropRate2Text = "–";
        expRemiRateText = "–";
        propofolDeltaText = "–";
        propofolDeltaClass = "te-result-cell te-result-change-value";
        propRate1DeltaText = "–";
        propRate1DeltaClass = "te-result-cell te-result-change-value";
        propRate2DeltaText = "–";
        propRate2DeltaClass = "te-result-cell te-result-change-value";
        opioidDeltaText = "–";
        opioidDeltaClass = "te-result-cell te-result-change-value";
      }

      // Dose-Response sweep: across the explored opioid's own rate range
      // whenever one is chosen (that's the drug actually being
      // explored), else the baseline's own opioid, else the "no opioid
      // administered" flat fallback - same graph structure/shapes/axes
      // throughout (teDoseResponseFigure is unchanged), only which drug
      // is swept and which vertical reference lines are drawn changes.
      var sweepOpioid = exploreVisible ? exploreOpioid : baseOpioid;
      var sweepCfg = sweepOpioid !== "none" ? teOpioidConfig(sweepOpioid) : null;
      var sSweep = teComputeState(c.age, c.sex, c.height, c.weight, c.sbp, c.dbp, sweepOpioid, c.bisLow, c.bisHigh, c.mapMode, c.mapValue, 0);
      var doseBaselineRate = sweepCfg && baseVisible && baseOpioid === sweepOpioid ? baselineRate : null;
      var doseExploredRate = sweepCfg && active && exploreOpioid === sweepOpioid ? exploredRateRaw : null;
      var doseResponseFig = teDoseResponseFigure(sSweep, sweepCfg, doseBaselineRate, doseExploredRate);

      var strategyBaselineText = baseVisible
        ? teCapitalize(baseOpioid) + " " + baselineRate.toFixed(baseCfg.decimals) + " " + baseCfg.unit
        : "No opioid";
      var strategyExploredText = !exploreVisible
        ? "Not selected"
        : !active
        ? "Move slider to explore an alternative strategy"
        : teCapitalize(exploreOpioid) + " " + exploredRateRaw.toFixed(exploreCfg.decimals) + " " + exploreCfg.unit;

      return [
        expInductionText, expInductionClass, expTotalText,
        "Pause", expPropRate1Text, expPropRate2Text, "Pause", expRemiRateText,
        propofolDeltaText, propofolDeltaClass,
        propRate1DeltaText, propRate1DeltaClass,
        propRate2DeltaText, propRate2DeltaClass,
        opioidDeltaText, opioidDeltaClass,
        doseResponseFig,
        teBisFigure(sBase, sExp), teMapFigure(sBase, sExp), tePropofolPkFigure(sBase, sExp), teOpioidPkFigure(sBase, sExp),
        strategyBaselineText, strategyExploredText,
      ];
    },

    // Live label above the slider - fires on every drag tick (cheap
    // string formatting only), reading the live Explore-opioid dropdown
    // (not the committed baseline) so its unit/decimals always match
    // whatever the slider is actually configured for.
    updateExploreLiveLabel: function (rateValue, exploreOpioidValue) {
      var opioid = exploreOpioidValue || "none";
      if (opioid === "none") {
        return window.dash_clientside.no_update;
      }
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
