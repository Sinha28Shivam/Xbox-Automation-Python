# flash-leonardo.ps1
# Waits for an Arduino Leonardo (atmega32u4) bootloader COM port to appear,
# then immediately flashes the selected GIMX firmware with avrdude.
#
# The Leonardo bootloader only stays active ~8 seconds after a RESET, so this
# script polls rapidly and fires avrdude the moment the port shows up.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\flash-leonardo.ps1
#   powershell -ExecutionPolicy Bypass -File .\flash-leonardo.ps1 -Firmware EMUPS4 -TimeoutSeconds 120

param(
    [string]$Firmware       = "EMUXONE",
    [string]$GimxDir        = "C:\Program Files\GIMX",
    [int]$TimeoutSeconds    = 90,
    [string]$ForcePort      = ""    # e.g. "COM9" to skip auto-detection
)

$ErrorActionPreference = "Stop"

$avrdude  = Join-Path $GimxDir "avrdude.exe"
$avrconf  = Join-Path $GimxDir "avrdude.conf"
$hexFile  = Join-Path $GimxDir ("firmware\{0}.hex" -f $Firmware)

# ---------- Pre-flight checks ----------
foreach ($f in @($avrdude, $avrconf, $hexFile)) {
    if (-not (Test-Path $f)) { Write-Host "ERROR: missing required file: $f" -ForegroundColor Red; exit 1 }
}

Write-Host "=== GIMX Leonardo Flasher ===" -ForegroundColor Cyan
Write-Host "avrdude  : $avrdude"
Write-Host "firmware : $hexFile"
Write-Host ""

# Arduino Leonardo bootloader = VID_2341 PID_0036 (also 2A03:0036 for .org boards)
function Get-BootloaderPort {
    $dev = Get-PnpDevice -PresentOnly -Class Ports -ErrorAction SilentlyContinue |
           Where-Object { $_.InstanceId -match 'VID_(2341|2A03)&PID_0036' -or
                          $_.FriendlyName -match 'bootloader' }
    if ($dev) {
        foreach ($d in $dev) {
            if ($d.FriendlyName -match '\((COM\d+)\)') { return $Matches[1] }
        }
    }
    return $null
}

# Any present Arduino/AVR port (sketch mode or bootloader)
function Get-AnyArduinoPort {
    $dev = Get-PnpDevice -PresentOnly -Class Ports -ErrorAction SilentlyContinue |
           Where-Object { $_.InstanceId -match 'VID_(2341|2A03|03EB|1B4F)' }
    foreach ($d in $dev) {
        if ($d.FriendlyName -match '\((COM\d+)\)') {
            return [pscustomobject]@{ Port = $Matches[1]; Name = $d.FriendlyName }
        }
    }
    return $null
}

$port = $ForcePort

if (-not $port) {
    # If the board is present in normal (sketch) mode, we can trigger the
    # bootloader by opening its port at 1200 baud (the Leonardo "magic" reset).
    $existing = Get-AnyArduinoPort
    if ($existing -and $existing.Name -notmatch 'bootloader') {
        Write-Host "Board found in sketch mode on $($existing.Port) ($($existing.Name))." -ForegroundColor Yellow
        Write-Host "Triggering bootloader via 1200-baud touch..." -ForegroundColor Yellow
        try {
            $sp = New-Object System.IO.Ports.SerialPort $existing.Port, 1200, 'None', 8, 'One'
            $sp.Open(); Start-Sleep -Milliseconds 250; $sp.Close()
        } catch {
            Write-Host "  (1200-baud touch failed: $($_.Exception.Message)) - press RESET manually." -ForegroundColor DarkYellow
        }
        Start-Sleep -Milliseconds 800
    } else {
        Write-Host "Plug in the Arduino Leonardo's USB port now." -ForegroundColor Yellow
        Write-Host "If nothing happens, tap the RESET button once (bootloader lasts ~8s)." -ForegroundColor Yellow
    }

    Write-Host "Waiting up to $TimeoutSeconds s for a bootloader COM port..." -ForegroundColor Cyan
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $port = Get-BootloaderPort
        if ($port) { break }
        Start-Sleep -Milliseconds 200
    }
}

if (-not $port) {
    Write-Host ""
    Write-Host "TIMED OUT: no Leonardo bootloader port detected." -ForegroundColor Red
    Write-Host "Checks: USB data cable (not charge-only), board plugged into the PC," -ForegroundColor Red
    Write-Host "and press RESET right as the script starts waiting." -ForegroundColor Red
    Write-Host ("Present COM ports: " + ([System.IO.Ports.SerialPort]::GetPortNames() -join ', '))
    exit 2
}

Write-Host ""
Write-Host ">>> Bootloader detected on $port - flashing NOW <<<" -ForegroundColor Green

# avrdude writes its progress output to stderr, which PowerShell would treat as
# a fatal NativeCommandError while $ErrorActionPreference is "Stop". Relax it and
# run via cmd.exe so stderr is merged harmlessly into stdout.
$ErrorActionPreference = "Continue"

$devPort = "\\.\" + $port
$outLog  = Join-Path $env:TEMP "avrdude-out.txt"
$errLog  = Join-Path $env:TEMP "avrdude-err.txt"

$avrArgs = @(
    "-C", "`"$avrconf`"",
    "-p", "atmega32u4",
    "-c", "avr109",
    "-P", $devPort,
    "-b", "57600",
    "-D",
    "-U", "`"flash:w:$hexFile`:i`""
)

$proc = Start-Process -FilePath $avrdude -ArgumentList $avrArgs -NoNewWindow -Wait -PassThru `
                      -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$code = $proc.ExitCode

foreach ($lf in @($outLog, $errLog)) {
    if (Test-Path $lf) {
        Get-Content $lf | Where-Object { $_ -ne "" } | ForEach-Object { Write-Host $_ }
    }
}

Write-Host ""
if ($code -eq 0) {
    Write-Host "SUCCESS: $Firmware flashed to the Leonardo." -ForegroundColor Green
    Write-Host "Unplug/replug the board; it should now appear as a GIMX USB adapter" -ForegroundColor Green
    Write-Host "(no longer as an Arduino serial port)." -ForegroundColor Green
} else {
    Write-Host "FAILED: avrdude exited with code $code." -ForegroundColor Red
    Write-Host "Most common cause: the ~8s bootloader window closed. Press RESET and rerun." -ForegroundColor Red
}

Write-Host ""
Write-Host "--- Post-flash device state ---" -ForegroundColor Cyan
Start-Sleep -Seconds 2
Get-PnpDevice -PresentOnly |
    Where-Object { $_.InstanceId -match 'VID_(2341|2A03|03EB|1B4F|0403)' -or $_.Class -eq 'Ports' } |
    Select-Object Status, Class, FriendlyName | Format-Table -AutoSize | Out-String -Width 160 | Write-Host
Write-Host ("COM ports: " + ([System.IO.Ports.SerialPort]::GetPortNames() -join ', '))

exit $code
