param([switch]$RefreshResults)
$ErrorActionPreference = "Stop"
$thesisDirectory = Split-Path -Parent $PSScriptRoot
$thesisBuildDirectory = Join-Path $thesisDirectory "build"
New-Item -ItemType Directory -Path $thesisBuildDirectory -Force | Out-Null
Push-Location $thesisDirectory
try {
    if ($RefreshResults) {
        & python (Join-Path $PSScriptRoot "generate_results.py")
        if ($LASTEXITCODE -ne 0) { throw "Result-asset generation failed." }
    }
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -output-directory=build thesis.tex
    if ($LASTEXITCODE -ne 0) { throw "First pdfLaTeX pass failed. See build/thesis.log." }
    & biber --input-directory=build --output-directory=build thesis
    if ($LASTEXITCODE -ne 0) { throw "Biber failed. See build/thesis.blg." }
    for ($thesisPass = 0; $thesisPass -lt 2; $thesisPass++) {
        & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -output-directory=build thesis.tex
        if ($LASTEXITCODE -ne 0) { throw "pdfLaTeX reference pass failed." }
    }
    Write-Output "Built: $(Join-Path $thesisBuildDirectory 'thesis.pdf')"
} finally {
    Pop-Location
}
