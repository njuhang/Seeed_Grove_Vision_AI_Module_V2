param(
    [string]$Port = "COM5",
    [int]$Baudrate = 921600,
    [int]$ListenPort = 57005,
    [switch]$ResetOnClientConnect,
    [int]$ResetPulseMs = 150,
    [int]$ResetSettleMs = 300
)

$ErrorActionPreference = "Stop"

$serial = [System.IO.Ports.SerialPort]::new($Port, $Baudrate, [System.IO.Ports.Parity]::None, 8, [System.IO.Ports.StopBits]::One)
$serial.ReadTimeout = 50
$serial.WriteTimeout = 1000
$serial.Open()

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $ListenPort)
$listener.Start()
Write-Host "Serial TCP bridge listening on 127.0.0.1:$ListenPort -> $Port@$Baudrate"

try {
    $client = $listener.AcceptTcpClient()
    $client.NoDelay = $true
    $stream = $client.GetStream()

    if ($ResetOnClientConnect) {
        Write-Host "Pulsing DTR/RTS reset on $Port"
        $serial.DtrEnable = $false
        $serial.RtsEnable = $false
        Start-Sleep -Milliseconds $ResetPulseMs
        $serial.DtrEnable = $true
        $serial.RtsEnable = $true
        Start-Sleep -Milliseconds $ResetPulseMs
        $serial.DtrEnable = $false
        $serial.RtsEnable = $false
        Start-Sleep -Milliseconds $ResetSettleMs
    }

    $serialBuffer = New-Object byte[] 4096
    $socketBuffer = New-Object byte[] 4096

    while ($client.Connected -and $serial.IsOpen) {
        while ($serial.BytesToRead -gt 0) {
            $count = $serial.Read($serialBuffer, 0, [Math]::Min($serialBuffer.Length, $serial.BytesToRead))
            if ($count -gt 0) {
                $stream.Write($serialBuffer, 0, $count)
                $stream.Flush()
            }
        }

        while ($client.Available -gt 0) {
            $count = $stream.Read($socketBuffer, 0, [Math]::Min($socketBuffer.Length, $client.Available))
            if ($count -gt 0) {
                $serial.Write($socketBuffer, 0, $count)
            }
        }

        Start-Sleep -Milliseconds 1
    }
}
finally {
    if ($client) {
        $client.Close()
    }
    $listener.Stop()
    if ($serial.IsOpen) {
        $serial.Close()
    }
}
