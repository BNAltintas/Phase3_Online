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
 * The story this page tells: pick a patient -> see a baseline propofol
 * recommendation (opioid at its own fixed anchor rate) -> drag the
 * "Explore opioid strategy" slider (a local, uncommitted preview) ->
 * press "Run Scenario" to commit that rate -> compare the explored
 * recommendation against the baseline -> see the physiological
 * consequences on the right-hand prediction graphs. More opioid always
 * pulls propofol (and BIS/MAP) down; less opioid pushes them back up.
 */

(function () {
  "use strict";

  var GREEN = "#1f9254";
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
      return "te-delta-value te-delta-neutral";
    }
    return "te-delta-value " + (pct < 0 ? "te-delta-good" : "te-delta-bad");
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

  function teMaintRow(time, text) {
    return (
      '<div class="te-maint-row"><span class="te-maint-time">' +
      time +
      '</span><span class="te-maint-value">' +
      text +
      "</span></div>"
    );
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

  /* ---------- dose-response interaction (fake) - now swept across the
     selected opioid's own infusion rate, not propofol dose ---------- */

  function teDoseResponseFigure(s, cfg, baselineRate, exploredRate) {
    var xs = teLinspace(cfg.min, cfg.max, 61);
    var mapFloor = 55.0;
    var bisFloor = 25.0;
    var bisCeil = 85.0;

    var mapCurve = xs.map(function (rate) {
      var t = (rate - cfg.min) / (cfg.max - cfg.min);
      return mapFloor + (s.map - mapFloor) * (1 - 0.55 * t);
    });
    var bisCurve = xs.map(function (rate) {
      var t = (rate - cfg.min) / (cfg.max - cfg.min);
      return bisFloor + (bisCeil - bisFloor) * (1 - 0.6 * t);
    });

    var targetBand = {
      x: [cfg.min, cfg.max, cfg.max, cfg.min],
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
    var baselineLine = {
      x: [baselineRate, baselineRate], y: [0, 100], mode: "lines",
      name: "Baseline (" + baselineRate.toFixed(cfg.decimals) + " " + cfg.unit + ")",
      line: { color: GREEN, width: 2, dash: "dash" }, yaxis: "y2", hoverinfo: "skip",
    };
    var exploredLine = {
      x: [exploredRate, exploredRate], y: [0, 100], mode: "lines",
      name: "Explored (" + exploredRate.toFixed(cfg.decimals) + " " + cfg.unit + ")",
      line: { color: BLUE, width: 2.5 }, yaxis: "y2", hoverinfo: "skip",
    };

    var layout = {
      xaxis: teAxis(teCapitalize(s.opioid) + " infusion rate (" + cfg.unit + ")"),
      yaxis: teAxis("MAP (mmHg)", { range: [40, 110] }),
      yaxis2: teAxis("BIS", { overlaying: "y", side: "right", range: [0, 100], gridcolor: "transparent" }),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 40, r: 50, b: 50, l: 55 },
    };

    return { data: [targetBand, mapLine, bisLine, baselineLine, exploredLine], layout: layout };
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

  function teOpioidPkFigure(s) {
    var t = teLinspace(0, 15, 61);
    var plateau = { remifentanil: 2.4, sufentanil: 0.4, fentanyl: 3.2 }[s.opioid] || 1.0;
    var y = t.map(function (tt) {
      return plateau * (1 - Math.exp(-tt / 3.5));
    });
    var yLow = y.map(function (v) {
      return Math.max(0, v * 0.85);
    });
    var yHigh = y.map(function (v) {
      return v * 1.15;
    });
    var drugLabel = teCapitalize(s.opioid || "opioid");

    var layout = {
      xaxis: teAxis("Time (min)"),
      yaxis: teAxis(drugLabel + " concentration (ng/mL)"),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      margin: { t: 40, r: 20, b: 45, l: 55 },
    };

    return {
      data: [
        teBand(t, yLow, yHigh, "90% Prediction Interval", "rgba(47, 111, 237, 0.35)"),
        { x: t, y: y, mode: "lines", name: "Predicted Cp", line: { color: GREEN, width: 2.5 } },
      ],
      layout: layout,
    };
  }

  function teEmptyFigure() {
    return {
      data: [],
      layout: { plot_bgcolor: "#ffffff", paper_bgcolor: "#ffffff", font: teFont() },
    };
  }

  /* ---------- Dash clientside entry points ---------- */

  window.dash_clientside = window.dash_clientside || {};
  window.dash_clientside.testExploration = {
    updateBaseline: function (age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue) {
      var cfg = teOpioidConfig(opioid);
      var rate = opioid === "none" ? 0 : cfg.baseline;
      var s = teComputeState(age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue, rate);

      var inductionText = s.induction.toFixed(2) + " mg/kg";
      var totalText = "(" + s.inductionTotalMg.toFixed(1) + " mg total)";

      var propofolTable =
        teMaintRow("0–1 min", "Pause") +
        teMaintRow("1–8 min", s.propRate1MlH.toFixed(0) + " mL/h (" + s.propRate1McgKgMin.toFixed(0) + " µg/kg/min)") +
        teMaintRow("8–15 min", s.propRate2MlH.toFixed(0) + " mL/h (" + s.propRate2McgKgMin.toFixed(0) + " µg/kg/min)");

      var opioidVisible = s.opioid !== "none";
      var opioidSectionStyle = opioidVisible ? null : { display: "none" };
      var opioidTitle = opioidVisible ? teCapitalize(s.opioid) + " – Early maintenance (0–15 min)" : "-";
      var opioidTable = opioidVisible
        ? teMaintRow("0–2 min", "Pause") + teMaintRow("2–15 min", s.opioidRateDisplay)
        : "";

      return [inductionText, totalText, propofolTable, opioidSectionStyle, opioidTitle, opioidTable];
    },

    updateExploreControls: function (opioid) {
      var noUpdate = window.dash_clientside.no_update;
      var visible = opioid !== "none";
      var contentStyle = visible ? null : { display: "none" };
      var placeholderStyle = visible ? { display: "none" } : null;

      if (!visible) {
        return [noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, contentStyle, placeholderStyle];
      }

      var cfg = teOpioidConfig(opioid);
      var marks = {};
      marks[cfg.min] = cfg.min.toFixed(cfg.decimals);
      marks[cfg.max] = cfg.max.toFixed(cfg.decimals);
      var label = teCapitalize(opioid) + " infusion rate";
      var valueText = cfg.baseline.toFixed(cfg.decimals) + " " + cfg.unit;

      return [cfg.min, cfg.max, cfg.step, cfg.baseline, marks, label, valueText, cfg.baseline, contentStyle, placeholderStyle];
    },

    updateExploreLiveLabel: function (rateValue, opioid) {
      var cfg = teOpioidConfig(opioid);
      var rate = teParseFloat(rateValue, cfg.baseline);
      return rate.toFixed(cfg.decimals) + " " + cfg.unit;
    },

    runScenario: function (_nClicks, sliderValue) {
      return teParseFloat(sliderValue, 0);
    },

    updateScenarioResult: function (age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue, exploredRate) {
      var cfg = teOpioidConfig(opioid);
      var baselineRate = opioid === "none" ? 0 : cfg.baseline;
      var explored = opioid === "none" ? 0 : teParseFloat(exploredRate, cfg.baseline);

      var sBase = teComputeState(age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue, baselineRate);
      var sExp = teComputeState(age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue, explored);

      var opioidLabel = opioid === "none" ? "Opioid" : teCapitalize(opioid);
      var baseOpioid = opioid === "none" ? "–" : sBase.opioidRateDisplay;
      var expOpioid = opioid === "none" ? "–" : sExp.opioidRateDisplay;

      var baseInduction = sBase.induction.toFixed(2) + " mg/kg";
      var baseMaint = sBase.propRate1MlH.toFixed(0) + " → " + sBase.propRate2MlH.toFixed(0) + " mL/h";
      var expInduction = sExp.induction.toFixed(2) + " mg/kg";
      var expMaint = sExp.propRate1MlH.toFixed(0) + " → " + sExp.propRate2MlH.toFixed(0) + " mL/h";

      var propofolDeltaPct = ((sExp.induction - sBase.induction) / sBase.induction) * 100;
      var opioidDeltaPct = opioid === "none" || baselineRate === 0 ? 0 : ((explored - baselineRate) / baselineRate) * 100;

      return [
        baseInduction, baseMaint, opioidLabel, baseOpioid,
        expInduction, expMaint, opioidLabel, expOpioid,
        teDeltaText(propofolDeltaPct), teDeltaClass(propofolDeltaPct),
        opioidLabel, teDeltaText(opioidDeltaPct), teDeltaClass(opioidDeltaPct),
      ];
    },

    updateGraphs: function (age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue, exploredRate) {
      var cfg = teOpioidConfig(opioid);
      var baselineRate = opioid === "none" ? 0 : cfg.baseline;
      var explored = opioid === "none" ? 0 : teParseFloat(exploredRate, cfg.baseline);

      // Predictions reflect the explored scenario (the "physiological
      // consequences" step of the workflow) - equal to the baseline until
      // Run Scenario is first pressed.
      var s = teComputeState(age, sex, height, weight, sbp, dbp, opioid, bisLow, bisHigh, mapMode, mapValue, explored);

      var subtitle =
        opioid === "none"
          ? "Select an opioid to see its dose-response interaction"
          : "How " + opioid + " infusion rate affects MAP and BIS";

      var opioidPkVisible = opioid !== "none";
      var opioidPkStyle = opioidPkVisible ? null : { display: "none" };
      var opioidPkFig = opioidPkVisible ? teOpioidPkFigure(s) : teEmptyFigure();
      var doseResponseFig = opioid === "none" ? teEmptyFigure() : teDoseResponseFigure(s, cfg, baselineRate, explored);

      return [doseResponseFig, subtitle, teBisFigure(s), teMapFigure(s), tePropofolPkFigure(s), opioidPkFig, opioidPkStyle];
    },
  };
})();
