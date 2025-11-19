# PROPOFOL-model

**Aim:**  
Development of the model for the dose decision-support tool of propofol.

---

## Background

Despite the frequent long-term use of propofol, the anesthesia induction dose targeted to an individual and resulting in sufficient anesthetic depth to perform highly stimulating airway manipulation without significant cardiovascular side effects or awareness is insufficiently known. Moreover, the state-of-the-art propofol dosing schemes based on PK–PD modeling do not take into account important patient-related factors, nor hemodynamic side effects. As a result of attempts to avoid awareness, significant hemodynamic side effects occur every day in the operating room. In particular, hypotension has repeatedly been associated with adverse outcomes, such as acute kidney injury and myocardial injury.

To prevent post-induction hypotension while achieving sufficient anesthetic depth, we aim to design an easy-to-use dosing-support tool for propofol, designed for application in daily care. This tool will assist anesthesiologists in optimizing propofol dosing by incorporating key patient-related factors and hemodynamic considerations.

---

## Model objectives

The PROPOFOL-model repository contains code and documentation to:

- **Predict the optimal induction dose for general anesthesia**, incorporating patient-related factors and hemodynamic risk.
- **(Planned) Prediction manual dose**:  
  - Either a pure **bolus induction dose**, or  
  - **Bolus + early infusions** within 15 minutes or until surgical incision.
- **For TCI (target-controlled infusion)**:
  - Predict the **optimal target effect-site concentration** (Eleveld model) to induce general anesthesia.
- **Incorporate clinical endpoints**:
  - Anesthetic depth target: **BIS < 60**
  - Hemodynamic safety targets:
    - **MAP drop < 20% from baseline**, and/or  
    - **Absolute MAP ≥ 65 mmHg**

The final model is intended to be integrated into a clinical decision-support tool used at the bedside.

---

## Repository scope

This repository focuses on **model development**:

- PK–PD and/or machine learning
- Model estimation, validation, and internal performance checks
- Simulation of dosing strategies and hemodynamic responses
- Export of model parameters and logic for use in the interface tool

Other parts of the project (data cleaning, UI) live in separate repositories and are **not** included here.

---

## Development workflow

We use a **branch-based workflow with code review**. No one edits `main` directly.

### Branching model

- **`main`**
  - Always stable and reproducible.
  - Contains only reviewed and approved changes.
  - Protected: no direct pushes.

- **Feature branches**
  - Created from `main` for all work:
    - `feature/add-hemodynamic-submodel`
    - `feature/tci-effect-site-target`
    - `bugfix/fix-baseline-map-calculation`
  - Short-lived; removed after merge.

### How to make a change

1. **Update local `main`**
   ```
   git checkout main
   git pull origin main
   ```
2. **Create a feature branch**
   ```
   git checkout -b feature/short-description
   ```
3. **Implement and test**
   * Add or modify code in src/
   * Add or update tests in tests/
   * Ensure model scripts run on test/example data
4. **Commit with a clear message**
   ```
   git add .
   git commit -m "Add MAP-constrained induction dose objective"
   ```
5. **Push the branch**
   ```
   git push -u origin feature/short-description
   ```
6. **Open a Pull Request (PR)**
   * Base branch: `main`
   * Link any related issue(s), e.g. `Closes #12`
   * Briefly describe:
     * What has changed
     * New model assumptions / endpoints
     * How the change was tested
7. **Review and approval**
   * At least one reviewer who did not write the code
   * Reviewer checks:
     * Clinical/model logic
     * Code clarity and maintainability
     * Tests and documentation
8. **Merge**
   * After approval (and passing automated tests, if configured), the reviewer merges the PR into main (preferably using “Squash and merge”).
   * The feature branch is then deleted.




