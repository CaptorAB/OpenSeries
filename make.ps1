param (
    [string]$task = "active"
)

$ErrorActionPreference = 'Stop'

# Function to get the latest Python 3.10 version from pyenv
function Get-LatestPython310Version {
    $versions = pyenv versions --bare 3.10.*
    $latestVersion = $versions | Where-Object { $_ -match '^3\.10\.\d+$' } | Sort-Object -Descending | Select-Object -First 1
    return $latestVersion
}

if ($task -eq "active")
{
    .\venv\Scripts\activate
    if ($null -ne $env:PYTHONPATH)
    {
        if (-not ($env:PYTHONPATH -match [regex]::Escape($PWD)))
        {
            $env:PYTHONPATH = "$PWD" + ";$env:PYTHONPATH"
            Write-Output "`nPYTHONPATH changed. It is '$( $env:PYTHONPATH )'"
        }
        else
        {
            Write-Output "`nPYTHONPATH not changed. It is '$( $env:PYTHONPATH )'"
        }
    }
    else
    {
        $env:PYTHONPATH = $PWD
        Write-Output "`nPYTHONPATH set to: $( $env:PYTHONPATH )"
    }
    Write-Output "`nThe Python used in the '$(Split-Path -Leaf $env:VIRTUAL_ENV)' environment is:"
    Get-Command python
}
elseif ($task -eq "make")
{
    Remove-Item -Path ".\venv" -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path $env:USERPROFILE\.pyenv) {
        $latestVersion = Get-LatestPython310Version
        if ($latestVersion) {
            pyenv global $latestVersion
            pyenv local $latestVersion
            Remove-Item -Path '.python-version' -Force -ErrorAction SilentlyContinue
            Write-Output "Python $latestVersion set as both local and global version using pyenv."
        } else {
            Write-Warning "No Python 3.10 versions found with pyenv."
        }
    } else {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -like "*3.10*") {
            Write-Output "Python 3.10 is identified as the system's Python version."
        } else {
            Write-Warning "Python 3.10 is not installed or configured. Please install Python 3.10 or pyenv."
        }
    }
    python -m venv ./venv
    if ($?) {
        Write-Host "Virtual environment 'venv' created successfully." -ForegroundColor Green
    } else {
        Write-Host "Failed to create virtual environment 'venv'." -ForegroundColor Red
        exit 1
    }
    .\venv\Scripts\activate
    if ($null -ne $env:PYTHONPATH)
    {
        if (-not ($env:PYTHONPATH -match [regex]::Escape($PWD)))
        {
            $env:PYTHONPATH = "$PWD" + ";$env:PYTHONPATH"
            Write-Output "`nPYTHONPATH changed. It is '$( $env:PYTHONPATH )'"
        }
        else
        {
            Write-Output "`nPYTHONPATH not changed. It is '$( $env:PYTHONPATH )'"
        }
    }
    else
    {
        $env:PYTHONPATH = $PWD
        Write-Output "`nPYTHONPATH set to: $( $env:PYTHONPATH )"
    }
    Write-Output "`nThe Python used in the '$(Split-Path -Leaf $env:VIRTUAL_ENV)' environment is:"
    Get-Command python
    python.exe -m pip install --upgrade pip
    pip install poetry==1.8.3
    Remove-Item -Path 'poetry.lock' -Force -ErrorAction SilentlyContinue
    poetry install --with dev
    poetry run pre-commit install
}
elseif ($task -eq "test")
{
    poetry run coverage run -m pytest --verbose --capture=no
    poetry run coverage report -m
    poetry run coverage-badge -o coverage.svg -f
}
elseif ($task -eq "lint")
{
    poetry run ruff check . --fix --exit-non-zero-on-fix
    poetry run mypy .
}
elseif ($task -eq "clean")
{
    pre-commit uninstall
    deactivate
    Remove-Item -Path ".\venv" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path 'poetry.lock' -Force -ErrorAction SilentlyContinue
}
else
{
    Write-Output "Only active, make, test, clean or lint are allowed as tasks"
}
