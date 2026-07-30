# =============================================================================
# MetaSpatial — clean up the public repo and publish.
#
# Makes the GitHub repo a professional, usable TOOL: removes the per-figure
# scripts, the manuscript, and scratch/output folders from version control
# (they STAY on your disk — this only untracks them), then adds the new
# README, overview image, save/load API, and quickstart, and pushes.
#
# Run from INSIDE the repo folder:
#     cd "C:\Users\pc\Desktop\spatial metabolism\MetaSpatial"
#     powershell -ExecutionPolicy Bypass -File .\CLEANUP_AND_PUBLISH.ps1
# =============================================================================

Write-Host "== 1) untrack per-figure scripts, manuscript, and scratch (files stay on disk) =="
git rm -r --cached --ignore-unmatch `
    "reproduce/figures" `
    "manuscript" `
    "results" `
    "metabolic" `
    "pathways" `
    "desium_model.pkl" `
    "metaspatial_names.py" `
    "__pycache__" `
    "metaspatial/__pycache__"

Write-Host "== 2) keep them out of the repo going forward (.gitignore) =="
$ignore = @(
    "", "# --- excluded from the public tool repo ---",
    "reproduce/figures/", "manuscript/", "results/", "metabolic/", "pathways/",
    "desium_model.pkl", "metaspatial_names.py", "__pycache__/", "*.pyc"
)
Add-Content -Path ".gitignore" -Value ($ignore -join "`n")

Write-Host "== 3) stage the new / updated tool files =="
git add README.md docs/overview.png metaspatial/metaspatial.py examples/quickstart.py .gitignore

Write-Host "== 4) commit + push =="
git commit -m "Make MetaSpatial a usable tool: professional README + overview image, MetaSpatial.load/save API, quickstart; remove per-figure scripts and manuscript from the public repo"
git push

Write-Host ""
Write-Host "Done. Verify on GitHub that reproduce/figures/ and manuscript/ are gone and the README renders with docs/overview.png."
Write-Host "Note: the pre-trained model 'metaspatial_model.pkl' remains tracked so users can predict out of the box."
