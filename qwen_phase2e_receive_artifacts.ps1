param(
    [string]$Repo = 'C:\Users\behno\Repos\Thesis Build Up',
    [string]$Remote = 'root@212.13.234.23',
    [int]$Port = 13177,
    [string]$Key = 'C:\Users\behno\vast-ai-instance',
    [string]$RemoteXfer = '/dev/shm/phase2e_transfer_v1',
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$ReceiveId = 'v1',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$Outputs = Join-Path $Repo 'outputs'
$StudyName = 'qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle'
$StudyLocal = Join-Path $Outputs $StudyName
$ReceiveRoot = Join-Path $Outputs ".qwen_phase2e_receive_${ReceiveId}"
$Stage = Join-Path $ReceiveRoot 'transfer'
$ExtractRoot = Join-Path $ReceiveRoot 'extracted_outputs'
$StudyIncoming = Join-Path $ExtractRoot $StudyName
$Integrity = Join-Path $StudyIncoming 'integrity\transfer_manifests'

if (Test-Path -LiteralPath $StudyLocal) {
    throw "Refusing merge into pre-existing $StudyLocal"
}
if ((Test-Path -LiteralPath $ReceiveRoot) -and -not $Resume) {
    throw "Refusing reuse of prior receive transaction: $ReceiveRoot"
}
$repoRoot = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Repo))
$repoDriveName = $repoRoot.TrimEnd([char[]]@(':', '\'))
$repoDrive = Get-PSDrive -Name $repoDriveName
$existingTransactionBytes = 0
if ($Resume -and (Test-Path -LiteralPath $ReceiveRoot)) {
    $existingTransactionBytes = [int64]((
        Get-ChildItem -LiteralPath $ReceiveRoot -File -Recurse |
            Measure-Object Length -Sum
    ).Sum)
}
if (($repoDrive.Free + $existingTransactionBytes) -lt 22GB) {
    throw "Need at least 22 GiB free on $repoRoot before Phase 2E transfer"
}
New-Item -ItemType Directory -Path $Stage -Force | Out-Null
New-Item -ItemType Directory -Path $ExtractRoot -Force | Out-Null

function Assert-Sha256Manifest([string]$Manifest, [string]$Base) {
    $baseFull = [IO.Path]::GetFullPath($Base).TrimEnd('\') + '\'
    $count = 0
    foreach ($line in Get-Content -LiteralPath $Manifest) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') {
            throw "Bad manifest line: $line"
        }
        $expected = $Matches[1].ToLowerInvariant()
        $relative = $Matches[2].Replace('/', '\')
        if ([IO.Path]::IsPathRooted($relative)) {
            throw "Absolute manifest path: $relative"
        }
        $full = [IO.Path]::GetFullPath((Join-Path $Base $relative))
        if (-not $full.StartsWith($baseFull, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Escaping manifest path: $relative"
        }
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            throw "Missing: $full"
        }
        $actual = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) { throw "SHA mismatch: $full" }
        $count++
    }
    if ($count -eq 0) { throw "Empty manifest: $Manifest" }
    Write-Host "Verified $count files from $Manifest"
}

function Join-Parts([string]$ChunkDir, [string]$Archive) {
    if (Test-Path -LiteralPath $Archive) { throw "Refusing overwrite: $Archive" }
    $parts = @(Get-ChildItem -LiteralPath $ChunkDir -File | Sort-Object Name)
    if ($parts.Count -eq 0) { throw "No chunks: $ChunkDir" }
    $out = [IO.File]::Open(
        $Archive,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None
    )
    try {
        foreach ($part in $parts) {
            $input = [IO.File]::OpenRead($part.FullName)
            try { $input.CopyTo($out) } finally { $input.Dispose() }
        }
    } finally {
        $out.Dispose()
    }
}

function Get-ManifestPaths([string]$Manifest) {
    $paths = @()
    foreach ($line in Get-Content -LiteralPath $Manifest) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') {
            throw "Bad manifest line: $line"
        }
        $paths += $Matches[2]
    }
    if ($paths.Count -eq 0 -or @($paths | Sort-Object -Unique).Count -ne $paths.Count) {
        throw "Empty or duplicate-path manifest: $Manifest"
    }
    return $paths
}

function Get-SingleManifestEntry([string]$Manifest) {
    $lines = @(Get-Content -LiteralPath $Manifest | Where-Object { $_.Trim() })
    if ($lines.Count -ne 1 -or $lines[0] -notmatch '^([0-9a-fA-F]{64})  (.+)$') {
        throw "Expected one SHA-256 entry in $Manifest"
    }
    return [pscustomobject]@{
        Hash = $Matches[1].ToLowerInvariant()
        Path = $Matches[2]
    }
}

function Assert-SafeTar(
    [string]$Archive,
    [string]$ExpectedPrefix,
    [string]$FileManifest
) {
    $entries = @(& tar -tf $Archive)
    if ($LASTEXITCODE -ne 0 -or $entries.Count -eq 0) {
        throw "Unreadable or empty tar: $Archive"
    }
    $verbose = @(& tar -tvf $Archive)
    if ($LASTEXITCODE -ne 0 -or $verbose.Count -ne $entries.Count) {
        throw "Could not verify tar member types: $Archive"
    }
    $regularFiles = @()
    foreach ($entry in $entries) {
        if (
            $entry.StartsWith('/') -or
            $entry -match '[\\\x00-\x1f\x7f]' -or
            $entry -match '(^|/)\.\.(/|$)' -or
            -not $entry.StartsWith($ExpectedPrefix)
        ) {
            throw "Unsafe tar entry: $entry"
        }
    }
    for ($index = 0; $index -lt $entries.Count; $index++) {
        $memberType = $verbose[$index].Substring(0, 1)
        if ($memberType -notin @('-', 'd')) {
            throw "Link or unsupported tar member: $($entries[$index])"
        }
        if ($memberType -eq '-') { $regularFiles += $entries[$index] }
    }
    if (@($entries | Sort-Object -Unique).Count -ne $entries.Count) {
        throw "Duplicate tar member: $Archive"
    }
    if (@($entries | ForEach-Object { $_.ToLowerInvariant() } |
            Sort-Object -Unique).Count -ne $entries.Count) {
        throw "Case-colliding tar members are unsafe on Windows: $Archive"
    }
    $expectedFiles = @(Get-ManifestPaths $FileManifest | Sort-Object)
    $actualFiles = @($regularFiles | Sort-Object)
    if (Compare-Object -ReferenceObject $expectedFiles -DifferenceObject $actualFiles) {
        throw "Tar regular-file inventory differs from $FileManifest"
    }
}

function Remove-VerifiedStageItems([string[]]$Paths) {
    $stageFull = [IO.Path]::GetFullPath($Stage).TrimEnd('\') + '\'
    foreach ($path in $Paths) {
        $full = [IO.Path]::GetFullPath($path)
        if (-not $full.StartsWith($stageFull, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Unsafe cleanup target: $full"
        }
    }
    foreach ($path in $Paths) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

function Receive-ManifestFiles(
    [string]$Manifest,
    [string]$AllowedPrefix
) {
    $stageFull = [IO.Path]::GetFullPath($Stage).TrimEnd('\') + '\'
    foreach ($line in Get-Content -LiteralPath $Manifest) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') {
            throw "Bad manifest line: $line"
        }
        $expected = $Matches[1].ToLowerInvariant()
        $relative = $Matches[2]
        if (
            $relative -match '[\\\x00-\x1f\x7f]' -or
            $relative.StartsWith('/') -or
            $relative -match '(^|/)\.\.(/|$)' -or
            -not $relative.StartsWith($AllowedPrefix)
        ) {
            throw "Unsafe transfer-manifest path: $relative"
        }
        $full = [IO.Path]::GetFullPath((
            Join-Path $Stage $relative.Replace('/', '\')
        ))
        if (-not $full.StartsWith($stageFull, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Escaping transfer path: $relative"
        }
        $valid = $false
        if (Test-Path -LiteralPath $full -PathType Leaf) {
            $actual = (
                Get-FileHash -LiteralPath $full -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            $valid = $actual -eq $expected
        }
        if ($valid) {
            Write-Host "Retained verified transfer file: $relative"
            continue
        }
        if (Test-Path -LiteralPath $full) {
            Remove-Item -LiteralPath $full -Force
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $full) -Force |
            Out-Null
        for ($attempt = 1; $attempt -le 4; $attempt++) {
            & scp -o BatchMode=yes -o IdentitiesOnly=yes `
                -o ServerAliveInterval=15 -o ServerAliveCountMax=4 `
                -o ConnectTimeout=20 -i $Key -P $Port `
                "${Remote}:${RemoteXfer}/${relative}" $full
            if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $full -PathType Leaf)) {
                $actual = (
                    Get-FileHash -LiteralPath $full -Algorithm SHA256
                ).Hash.ToLowerInvariant()
                if ($actual -eq $expected) {
                    $valid = $true
                    break
                }
            }
            if (Test-Path -LiteralPath $full) {
                Remove-Item -LiteralPath $full -Force
            }
            if ($attempt -lt 4) { Start-Sleep -Seconds (2 * $attempt) }
        }
        if (-not $valid) {
            throw "Failed four verified transfer attempts: $relative"
        }
    }
    Assert-Sha256Manifest $Manifest $Stage
}

if (-not $Resume) {
    & scp -o BatchMode=yes -o IdentitiesOnly=yes -i $Key -P $Port -r `
        "${Remote}:${RemoteXfer}/manifests" $Stage
    if ($LASTEXITCODE -ne 0) { throw 'Manifest transfer failed' }
    & scp -o BatchMode=yes -o IdentitiesOnly=yes -i $Key -P $Port `
        "${Remote}:${RemoteXfer}/manifest_bundle.sha256" $Stage
    if ($LASTEXITCODE -ne 0) { throw 'Manifest-bundle transfer failed' }
} elseif (
    -not (Test-Path -LiteralPath (Join-Path $Stage 'manifests')) -or
    -not (Test-Path -LiteralPath (Join-Path $Stage 'manifest_bundle.sha256'))
) {
    throw 'Resume requires the previously verified manifest transaction'
}
Assert-Sha256Manifest (Join-Path $Stage 'manifest_bundle.sha256') $Stage
$remoteHashLines = @(
    & ssh -o BatchMode=yes -o IdentitiesOnly=yes -i $Key -p $Port $Remote `
        "sha256sum ${RemoteXfer}/manifest_bundle.sha256"
)
$remoteHashLine = @(
    $remoteHashLines | Where-Object { $_ -match '^[0-9a-fA-F]{64}\s' }
)
if ($remoteHashLine.Count -ne 1) {
    throw 'Could not obtain one remote manifest-bundle hash'
}
$remoteBundle = ($remoteHashLine[0] -split '\s+')[0].ToLowerInvariant()
$localBundle = (
    Get-FileHash -LiteralPath (Join-Path $Stage 'manifest_bundle.sha256') `
        -Algorithm SHA256
).Hash.ToLowerInvariant()
if ($remoteBundle -ne $localBundle) {
    throw 'Remote/local manifest-bundle SHA mismatch'
}

$metadataFilesManifest = Join-Path $Stage 'manifests\metadata_files.sha256'
$metadataAlreadyVerified = $Resume -and (Test-Path -LiteralPath $StudyIncoming)
if ($metadataAlreadyVerified) {
    Assert-Sha256Manifest $metadataFilesManifest $ExtractRoot
    Write-Host 'Retained previously verified extracted metadata'
} else {
    Receive-ManifestFiles `
        (Join-Path $Stage 'manifests\metadata_chunks.sha256') `
        'metadata_chunks/'
    New-Item -ItemType Directory -Path (Join-Path $Stage 'archives') -Force |
        Out-Null
    $metadataArchive = Join-Path $Stage 'archives\phase2e-metadata.tar.zst'
    if (Test-Path -LiteralPath $metadataArchive) {
        Remove-VerifiedStageItems @($metadataArchive)
    }
    Join-Parts (Join-Path $Stage 'metadata_chunks') $metadataArchive
    Assert-Sha256Manifest `
        (Join-Path $Stage 'manifests\metadata_archive.sha256') $Stage
    Assert-SafeTar $metadataArchive "$StudyName/" $metadataFilesManifest
    & tar -xf $metadataArchive -C $ExtractRoot
    if ($LASTEXITCODE -ne 0) { throw 'Metadata extraction failed' }
    Assert-Sha256Manifest $metadataFilesManifest $ExtractRoot
    Remove-VerifiedStageItems @(
        (Join-Path $Stage 'metadata_chunks'),
        $metadataArchive
    )
}

$inventory = Import-Csv `
    -LiteralPath (Join-Path $Stage 'manifests\transfer_inventory.tsv') `
    -Delimiter "`t"
$expectedRuns = @{
    'lr5e-6' = 'qwen-phase2e-base-sequence-classifier-token-count-prompt-lr5e-6-5epochs-full-parameter-20260808-seed42-v1'
    'lr1e-5' = 'qwen-phase2e-base-sequence-classifier-token-count-prompt-lr1e-5-5epochs-full-parameter-20260808-seed42-v1'
    'lr2e-5' = 'qwen-phase2e-base-sequence-classifier-token-count-prompt-lr2e-5-5epochs-full-parameter-20260808-seed42-v1'
}
if (
    @($inventory).Count -ne $expectedRuns.Count -or
    @($inventory | Where-Object { -not $expectedRuns.ContainsKey($_.variant) }).Count -ne 0
) {
    throw 'Transfer inventory must contain exactly the three declared Phase 2E variants'
}
foreach ($variant in @('lr5e-6', 'lr1e-5', 'lr2e-5')) {
    $row = @($inventory | Where-Object variant -eq $variant)
    if ($row.Count -ne 1) { throw "Inventory row mismatch for $variant" }
    if ($row[0].checkpoint_id -notmatch '^step-[0-9]{6}$') {
        throw "Unsafe checkpoint ID for ${variant}: $($row[0].checkpoint_id)"
    }
    $expectedRelative = "$StudyName/trials/$variant/runs/$($expectedRuns[$variant])/checkpoints/$($row[0].checkpoint_id)"
    if (
        $row[0].run_id -ne $expectedRuns[$variant] -or
        $row[0].relative_path -ne $expectedRelative -or
        [int]$row[0].chunk_count -lt 1
    ) {
        throw "Inventory identity mismatch for $variant"
    }
    $checkpointManifest = Join-Path $Stage `
        "manifests\${variant}_selected_checkpoint_files.sha256"
    $archiveManifest = Join-Path $Stage "manifests\${variant}_archive.sha256"
    $archiveEntry = Get-SingleManifestEntry $archiveManifest
    $expectedArchivePath = "archives/phase2e-${variant}-$($row[0].checkpoint_id).tar.zst"
    if (
        $archiveEntry.Hash -ne $row[0].archive_sha256 -or
        $archiveEntry.Path -ne $expectedArchivePath
    ) {
        throw "Inventory/archive manifest mismatch: $variant"
    }
    $checkpointPath = Join-Path $ExtractRoot $expectedRelative.Replace('/', '\')
    if ($Resume -and (Test-Path -LiteralPath $checkpointPath)) {
        Assert-Sha256Manifest $checkpointManifest $ExtractRoot
        Write-Host "Retained previously verified extracted checkpoint: $variant"
        continue
    }
    Receive-ManifestFiles `
        (Join-Path $Stage "manifests\${variant}_chunks.sha256") `
        "${variant}_chunks/"
    $chunkDir = Join-Path $Stage "${variant}_chunks"
    $actualChunkCount = @(Get-ChildItem -LiteralPath $chunkDir -File).Count
    if ($actualChunkCount -ne [int]$row[0].chunk_count) {
        throw "Chunk-count mismatch: $variant"
    }
    $archive = Join-Path $Stage `
        "archives\phase2e-${variant}-$($row[0].checkpoint_id).tar.zst"
    if (Test-Path -LiteralPath $archive) {
        Remove-VerifiedStageItems @($archive)
    }
    Join-Parts $chunkDir $archive
    Assert-Sha256Manifest $archiveManifest $Stage
    if ((Get-Item -LiteralPath $archive).Length -ne [int64]$row[0].archive_bytes) {
        throw "Archive-size mismatch: $variant"
    }
    Assert-SafeTar $archive "$expectedRelative/" `
        $checkpointManifest
    & tar -xf $archive -C $ExtractRoot
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $variant" }
    Assert-Sha256Manifest $checkpointManifest $ExtractRoot
    Remove-VerifiedStageItems @($chunkDir, $archive)
}

Assert-Sha256Manifest `
    (Join-Path $Stage 'manifests\metadata_files.sha256') $ExtractRoot
foreach ($variant in @('lr5e-6', 'lr1e-5', 'lr2e-5')) {
    Assert-Sha256Manifest `
        (Join-Path $Stage "manifests\${variant}_selected_checkpoint_files.sha256") `
        $ExtractRoot
}

New-Item -ItemType Directory -Path $Integrity -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Integrity 'manifests') -Force |
    Out-Null
Copy-Item -Path (Join-Path $Stage 'manifests\*') `
    -Destination (Join-Path $Integrity 'manifests') -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Stage 'manifest_bundle.sha256') `
    -Destination $Integrity -Force
Assert-Sha256Manifest `
    (Join-Path $Integrity 'manifest_bundle.sha256') $Integrity
$bytes = (
    Get-ChildItem -LiteralPath $StudyIncoming -File -Recurse |
        Measure-Object Length -Sum
).Sum
$verification = [ordered]@{
    status = 'passed'
    instance_id = 46617164
    remote = $Remote
    remote_transfer_root = $RemoteXfer
    study = $StudyName
    variants = @($inventory)
    metadata_and_all_three_selected_checkpoints_verified = $true
    all_remote_and_local_hashes_match = $true
    remote_originals_retained = $true
    resumed_transaction = [bool]$Resume
    local_study_bytes = [int64]$bytes
    local_study_gib = [math]::Round($bytes / 1GB, 6)
    verified_at = [DateTimeOffset]::UtcNow.ToString('o')
}
$verification | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath (Join-Path $StudyIncoming 'integrity\selected_checkpoints_transfer_verification.json') `
    -Encoding utf8

if (Test-Path -LiteralPath $StudyLocal) {
    throw "Final study path appeared during receive transaction: $StudyLocal"
}
Move-Item -LiteralPath $StudyIncoming -Destination $StudyLocal

[pscustomobject]@{
    Study = $StudyLocal
    Bytes = $bytes
    GiB = [math]::Round($bytes / 1GB, 3)
    FreeGiB = [math]::Round($repoDrive.Free / 1GB, 3)
    ReceiveTransaction = $ReceiveRoot
} | Format-List
