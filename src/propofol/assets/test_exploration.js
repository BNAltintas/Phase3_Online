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

  // The clamp bounds for the fake propofol induction dose (mg/kg) -
  // named here (not just inline in teComputeState below) because
  // teDoseResponseFigure's own dose sweep needs to cover exactly this
  // same range, so its curves are guaranteed to contain both the
  // baseline and explored dose markers.
  var TE_DOSE_MIN_MGKG = 0.8;
  var TE_DOSE_MAX_MGKG = 3.0;

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
    var induction = teClamp(inductionBase * (1 - reduction), TE_DOSE_MIN_MGKG, TE_DOSE_MAX_MGKG);
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

  /* ---------- dose-response interaction (fake) - mimics the
     Recommendation page's own "Induction-dose rationale" graph
     (make_induction_dose_rationale_figure in app.py), but entirely
     fake/local: swept across propofol INDUCTION DOSE (mg/kg, same
     TE_DOSE_MIN_MGKG..TE_DOSE_MAX_MGKG range teComputeState's induction
     is clamped to - never the opioid infusion rate. The curve shape
     itself depends only on the committed patient/targets (sBase.map,
     sBase.bisLow/bisHigh - always identical between sBase/sExp, since
     only opioid+rate differ between them), so it stays one stable
     reference curve; exploring a different opioid/rate only ever moves
     the two vertical markers left/right along it (sBase.induction /
     sExp.induction, already computed by teComputeState's existing
     opioid-reduction formula - no new dose logic needed for that part).
     No real PK/PD simulation, no Su2023PropofolRemifentanilRecommender -
     just smooth saturating exp() curves standing in for it. ---------- */

  // Saturating (not linear) MAP/BIS-vs-dose curves - steep drop at low
  // dose, flattening at high dose, same qualitative shape a real
  // resimulated sweep would have. floor values are anchored to this
  // patient's own baseline MAP / BIS targets so the curve stays
  // clinically plausible per-patient, not a fixed constant for everyone.
  function teDoseCurveT(dose) {
    return teClamp((dose - TE_DOSE_MIN_MGKG) / (TE_DOSE_MAX_MGKG - TE_DOSE_MIN_MGKG), 0, 1);
  }

  function teDoseMapValue(dose, s) {
    var floor = Math.max(35, s.map * 0.55);
    return floor + (s.map - floor) * Math.exp(-3.2 * teDoseCurveT(dose));
  }

  function teDoseBisValue(dose, s) {
    var ceil = 88.0;
    var floor = teClamp(s.bisLow - 10, 15, 40);
    return floor + (ceil - floor) * Math.exp(-2.6 * teDoseCurveT(dose));
  }

  // One green diamond + one orange diamond (each on both curves) per
  // active scenario - color encodes WHICH scenario (baseline/explored),
  // consistent with every other trace/marker on this page, rather than
  // which variable (that's already carried by the red/blue curves
  // themselves).
  function teDoseMarkerTrace(dose, value, yaxis, color) {
    return {
      x: [dose], y: [value], mode: "markers", yaxis: yaxis, showlegend: false,
      marker: { symbol: "diamond", size: 12, color: color, line: { color: "#ffffff", width: 1.5 } },
      hoverinfo: "skip",
    };
  }

  function teDoseResponseFigure(sBase, sExp) {
    var xs = teLinspace(TE_DOSE_MIN_MGKG, TE_DOSE_MAX_MGKG, 61);
    var mapCurve = xs.map(function (dose) {
      return teDoseMapValue(dose, sBase);
    });
    var bisCurve = xs.map(function (dose) {
      return teDoseBisValue(dose, sBase);
    });

    var mapTargetLow = sBase.map * 0.72;
    var mapTargetHigh = sBase.map;
    var bisTargetLow = sBase.bisLow;
    var bisTargetHigh = sBase.bisHigh;

    var shapes = [
      {
        type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y",
        y0: mapTargetLow, y1: mapTargetHigh,
        fillcolor: "rgba(217, 54, 46, 0.10)", line: { width: 0 }, layer: "below",
      },
      {
        type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y2",
        y0: bisTargetLow, y1: bisTargetHigh,
        fillcolor: "rgba(47, 111, 237, 0.10)", line: { width: 0 }, layer: "below",
      },
      {
        type: "line", xref: "x", yref: "paper",
        x0: sBase.induction, x1: sBase.induction, y0: 0, y1: 1,
        line: { color: GREEN, width: 2.5, dash: "dash" }, layer: "above",
      },
    ];

    var data = [
      { x: xs, y: mapCurve, mode: "lines", name: "MAP", line: { color: RED, width: 3 } },
      { x: xs, y: bisCurve, mode: "lines", name: "BIS", line: { color: BLUE, width: 3 }, yaxis: "y2" },
      // Shapes (target bands, vertical dose lines) never generate their
      // own legend entry, so - same trick the Recommendation page's own
      // rationale graph uses - these zero-data dummy traces exist purely
      // to give them one.
      {
        x: [null], y: [null], mode: "markers", showlegend: true, name: "Target",
        marker: { symbol: "square", size: 10, color: "rgba(0, 150, 0, 0.25)" },
      },
      {
        x: [null], y: [null], mode: "lines", showlegend: true, name: "Baseline Recommendation",
        line: { color: GREEN, width: 2.5, dash: "dash" },
      },
    ];

    data.push(teDoseMarkerTrace(sBase.induction, teDoseMapValue(sBase.induction, sBase), "y", GREEN));
    data.push(teDoseMarkerTrace(sBase.induction, teDoseBisValue(sBase.induction, sBase), "y2", GREEN));

    if (sExp) {
      shapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: sExp.induction, x1: sExp.induction, y0: 0, y1: 1,
        line: { color: ORANGE, width: 2.5, dash: "dash" }, layer: "above",
      });
      data.push({
        x: [null], y: [null], mode: "lines", showlegend: true, name: "Explored Scenario",
        line: { color: ORANGE, width: 2.5, dash: "dash" },
      });
      data.push(teDoseMarkerTrace(sExp.induction, teDoseMapValue(sExp.induction, sBase), "y", ORANGE));
      data.push(teDoseMarkerTrace(sExp.induction, teDoseBisValue(sExp.induction, sBase), "y2", ORANGE));
    }

    var layout = {
      xaxis: teAxis("Propofol induction dose (mg/kg)", { range: [TE_DOSE_MIN_MGKG, TE_DOSE_MAX_MGKG] }),
      yaxis: teAxis("MAP (mmHg)", { range: [30, Math.max(120, sBase.map + 20)] }),
      // Reversed, same as the Recommendation page's own rationale graph -
      // "up" on the chart still reads as "better/lighter", even though
      // the BIS number itself gets smaller as dose increases.
      yaxis2: teAxis("BIS", { overlaying: "y", side: "right", range: [100, 0], gridcolor: "transparent" }),
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      font: teFont(),
      legend: teLegend(),
      shapes: shapes,
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

      // Dose-Response sweep is now over propofol induction dose (mg/kg,
      // same axis as the Recommendation page's own rationale graph) -
      // sBase/sExp already carry their own .induction, so no separate
      // "which opioid to sweep" logic is needed any more; the sweep
      // curve itself only ever depends on sBase (patient/targets, always
      // identical between sBase/sExp), while sExp being null/non-null is
      // exactly the "explored active" flag that decides whether the
      // orange marker is drawn at all.
      var doseResponseFig = teDoseResponseFigure(sBase, sExp);

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
