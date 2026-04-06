# Install Miniconda on Windows 11

## Installer

- **File:** `Miniconda3-py313_26.1.1-1-Windows-x86_64.exe`
- **Python:** 3.13
- **Conda:** 26.1.1

## Silent Install (PowerShell)

The default install failed with _"Failed to extract packages"_ due to TEMP directory issues. The fix was to use a clean temporary directory:

```powershell
# Create a clean temp directory
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\conda_temp" | Out-Null
$env:TEMP = "$env:USERPROFILE\conda_temp"
$env:TMP = "$env:USERPROFILE\conda_temp"

# Run silent install
Start-Process -FilePath "$env:USERPROFILE\Downloads\Miniconda3-py313_26.1.1-1-Windows-x86_64.exe" `
  -ArgumentList "/S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /D=$env:USERPROFILE\miniconda3" `
  -Wait

# Clean up temp directory
Remove-Item -Recurse -Force "$env:USERPROFILE\conda_temp"
```

### Installer Flags

| Flag | Purpose |
|------|---------|
| `/S` | Silent mode (no GUI) |
| `/InstallationType=JustMe` | Per-user install, no admin required |
| `/RegisterPython=0` | Don't register as system default Python |
| `/AddToPath=0` | Don't add to system PATH (use `conda init` instead) |
| `/D=...` | Installation directory |

## Post-Install Configuration

```powershell
# Initialize conda for PowerShell
conda init powershell

# Disable automatic base environment activation
conda config --set auto_activate_base false
```

After running `conda init`, restart your terminal for changes to take effect.

## Install Location

`$env:USERPROFILE\miniconda3`
