param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Pdf,
    [Parameter(Mandatory = $true)][string]$Audit
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$pdfPath = [System.IO.Path]::GetFullPath($Pdf)
$auditPath = [System.IO.Path]::GetFullPath($Audit)
if (Test-Path -LiteralPath $pdfPath) { throw "Refusing to overwrite PDF: $pdfPath" }
if ([System.IO.Path]::GetExtension($sourcePath) -ne '.docx') { throw 'Source must be DOCX' }
$sourceHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
$null = [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($pdfPath))
$null = [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($auditPath))
$word = $null
$ownsWordInstance = $false
$document = $null
$started = [DateTime]::UtcNow
try {
    $word = New-Object -ComObject Word.Application
    if ($word.Documents.Count -ne 0) { throw 'Word instance already contains documents; refusing to alter its state' }
    $ownsWordInstance = $true
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($sourcePath, $false, $true)
    $document.Repaginate()
    $pageCount = $document.ComputeStatistics(2)
    $document.ExportAsFixedFormat($pdfPath, 17)
    if (-not (Test-Path -LiteralPath $pdfPath)) { throw 'Word did not create PDF' }
    if ((Get-Item -LiteralPath $pdfPath).Length -lt 1000) { throw 'Word PDF is implausibly small' }
    $result = [ordered]@{
        status = 'RENDERED_VISUAL_INSPECTION_PENDING'
        source = $sourcePath
        source_sha256 = $sourceHash
        pdf = $pdfPath
        pdf_sha256 = (Get-FileHash -LiteralPath $pdfPath -Algorithm SHA256).Hash
        word_version = $word.Version
        word_page_count = $pageCount
        source_opened_read_only = $true
        started_utc = $started.ToString('o')
        finished_utc = [DateTime]::UtcNow.ToString('o')
    }
    if ((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash -ne $sourceHash) { throw 'Source DOCX hash changed during render' }
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $auditPath -Encoding utf8
    Write-Output "Rendered $pageCount pages: $pdfPath"
}
finally {
    if ($null -ne $document) {
        $document.Close(0)
        $null = [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        if ($ownsWordInstance) { $word.Quit(0) }
        $null = [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
