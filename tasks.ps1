<#
Windows task runner — PowerShell equivalent of the Makefile.
Usage:  .\tasks.ps1 <task>
Tasks:  install | dev | train | app | test | lint | format | clean
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "dev", "train", "app", "test", "lint", "format", "clean", "help")]
    [string]$Task = "help"
)

switch ($Task) {
    "install" { pip install -e . }
    "dev" {
        pip install -e ".[dev]"
        pre-commit install
    }
    "train" { python -m hotel_cancel.train }
    "app" { streamlit run app/streamlit_app.py }
    "test" { pytest }
    "lint" { ruff check . }
    "format" {
        ruff check --fix .
        black .
    }
    "clean" {
        Get-ChildItem -Recurse -Directory -Include __pycache__, *.egg-info, .pytest_cache, .ruff_cache |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item models\*.pkl, reports\figures\*.png -ErrorAction SilentlyContinue
    }
    default {
        Write-Host "Tasks: install | dev | train | app | test | lint | format | clean"
    }
}
