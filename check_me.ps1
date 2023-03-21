param (
    [string]$task = "test"
)

$env:PYTHONPATH = "$env:PYTHONPATH;$pwd"
.\venv\Scripts\activate
if ($task -eq "test") {
    # run commands for test task
    poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0
    poetry run coverage report -m
    poetry run coverage-badge -o coverage.svg -f
} elseif ($task -eq "lint") {
    # run commands for lint task
    poetry run flake8 .
    poetry run mypy .
} else {
    # invalid task argument
    Write-Host "Only test or lint are allowed as tasks"
}
