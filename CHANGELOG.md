# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-04-02

### Added
- `recommend_regimen.py`: code to predict optimal induction dose + minute-wise maintenance rates for 15 minutes based on BIS and MAP thresholds.
- `dashboard_layout.py`: build lay-out for dashboard.
- `app.py`: dashboard with input parameters, recommended dose and figures.

### Changed
- `patient.py`: option added to calculate MAP and PP from SAP and DAP.

## [0.3.0] - 2026-03-25

### Added
- Su 2022 haemodynamic model
- Class to convert dosing strategy or cumulative dose to dosing rates
- Tests script

### Changed
- Updated ruff specs in `pyproject.toml`

## [0.2.0] - 2025-11-19

### Added
- Eleveld 2018 PK/PD model
- GitHub workflows for linting, unit tests and large files

## [0.1.0] - 2025-11-19

### Added
- Created project and repository
- Added technical documentation as submodule in `docs/`
