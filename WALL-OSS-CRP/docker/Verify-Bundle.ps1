param([string]$Bundle = $PSScriptRoot, [switch]$AssetsOnly)
$ErrorActionPreference = 'Stop'
function Assert-RelativePath([string]$Name) {
    if ([string]::IsNullOrEmpty($Name) -or $Name -match '[\\:\x00-\x1f\x7f]') {
        throw "Unsafe or nonportable bundle path: $Name"
    }
    foreach ($part in $Name.Split('/')) {
        if ($part -in @('', '.', '..') -or $part -match '[. ]$' -or
            $part -match '^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$') {
            throw "Unsafe or nonportable bundle path: $Name"
        }
    }
    if ($Name -match '\.partial$') { throw "Incomplete temporary file remains: $Name" }
}
function Assert-SameNames($Expected, $Actual, [string]$Message) {
    if (@($Expected).Count -ne @($Actual).Count -or
        @(Compare-Object -ReferenceObject @($Expected) -DifferenceObject @($Actual) -CaseSensitive).Count) {
        throw $Message
    }
}
$root = Get-Item -LiteralPath $Bundle -Force
if (-not $root.PSIsContainer -or ($root.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'Bundle must be a real directory'
}
$Bundle = $root.FullName.TrimEnd([IO.Path]::DirectorySeparatorChar)
$kind = if ($AssetsOnly) { 'assets' } else { 'transfer' }
$required = @('models.tar.gz', 'smoke-data.tar.gz', 'model_files.manifest.json',
              'sources.lock.json', 'verify_bundle.py')
if ($AssetsOnly) {
    $required += @('source-assets.sha256', 'dataset-root.txt', 'source-check.log',
                   'migration-preflight.json', 'STATUS.txt', 'Verify-Bundle.ps1')
} else {
    $required += @('environment-image.tar.gz', 'image-id.txt', 'image-tag.txt',
                   'image-identity.json', 'image_archive_identity.py',
                   'verify_native_architectures.py', 'verify_h100_bundle.sh', 'OFFLINE-DEPLOY.md')
    $required += @('build.log', 'image-inspect.json', 'runtime.log', 'setup-verification.json',
                   'dataset-verification.json', 'dependencies.json', 'validated-dataset-root.txt',
                   'validated-assets.sha256', 'validated-image-id.txt', 'native-architectures.json',
                   'native-architectures.log') | ForEach-Object { 'reports/' + $_ }
}
$marker = Get-Content -LiteralPath (Join-Path $Bundle 'BUNDLE_READY.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($marker.schema -cne 'walloss-offline-bundle' -or $marker.version -ne 1 -or
    $marker.status -cne 'READY' -or $marker.package_kind -cne $kind) {
    throw 'Missing or unsupported complete-bundle marker'
}
Assert-SameNames $required @($marker.required_files) 'Completion marker has a different required file set'
$actualFiles = @{}
$allNames = @{}
Get-ChildItem -LiteralPath $Bundle -Recurse -Force | ForEach-Object {
    $relative = $_.FullName.Substring($Bundle.Length + 1).Replace('\', '/')
    Assert-RelativePath $relative
    if ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point is not allowed: $relative" }
    if ($allNames.ContainsKey($relative)) { throw "Case-insensitive duplicate bundle path: $relative" }
    $allNames[$relative] = $true
    if (-not $_.PSIsContainer -and $relative -cnotin @('SHA256SUMS', 'BUNDLE_READY.json')) {
        $actualFiles[$relative] = $_.FullName
    }
}
$manifestPath = Join-Path $Bundle 'SHA256SUMS'
$manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
if ($marker.manifest.filename -cne 'SHA256SUMS' -or $marker.manifest.sha256 -cne $manifestHash) {
    throw 'Manifest digest differs from completion marker'
}
$records = @{}
Get-Content -LiteralPath $manifestPath -Encoding UTF8 | ForEach-Object {
    if ($_ -cnotmatch '^([0-9a-f]{64})  (.+)$') { throw 'Invalid SHA256SUMS line' }
    $expected = $Matches[1]
    $relative = $Matches[2]
    Assert-RelativePath $relative
    if ($relative -in @('SHA256SUMS', 'BUNDLE_READY.json') -or $records.ContainsKey($relative)) {
        throw "Duplicate or reserved manifest path: $relative"
    }
    $records[$relative] = $expected
}
if ($records.Count -eq 0) { throw 'SHA256SUMS must not be empty' }
if ($marker.manifest.file_count -ne $records.Count) { throw 'Manifest file count differs from completion marker' }
Assert-SameNames @($actualFiles.Keys) @($records.Keys) 'Manifest does not cover the exact directory file set'
foreach ($name in $required) {
    if (@($records.Keys) -cnotcontains $name) { throw "Missing required file: $name" }
}
$archives = @('models.tar.gz', 'smoke-data.tar.gz')
if (-not $AssetsOnly) { $archives += 'environment-image.tar.gz' }
Assert-SameNames $archives @($marker.archives.PSObject.Properties.Name) 'Completion marker has a different archive set'
foreach ($name in $archives) {
    if ($marker.archives.$name -cne $records[$name]) { throw "Archive checksum differs from completion marker: $name" }
}
if ($AssetsOnly) {
    if ($null -ne $marker.image_identity) { throw 'Assets-only package must not claim an image identity' }
} else {
    $identity = Get-Content -LiteralPath (Join-Path $Bundle 'image-identity.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($identity.source_image_id -cnotmatch '^sha256:[0-9a-f]{64}$' -or
        [string]::IsNullOrEmpty($identity.transfer_tag) -or
        $marker.image_identity.filename -cne 'image-identity.json' -or
        $marker.image_identity.sha256 -cne $records['image-identity.json'] -or
        $marker.image_identity.source_image_id -cne $identity.source_image_id -or
        $marker.image_identity.transfer_tag -cne $identity.transfer_tag) {
        throw 'Image identity differs from completion marker'
    }
    foreach ($name in @('image-id.txt', 'reports/validated-image-id.txt')) {
        if ((Get-Content -LiteralPath (Join-Path $Bundle $name) -Raw -Encoding UTF8).Trim() -cne $identity.source_image_id) {
            throw "Source image identity files disagree: $name"
        }
    }
    if ((Get-Content -LiteralPath (Join-Path $Bundle 'image-tag.txt') -Raw -Encoding UTF8).Trim() -cne $identity.transfer_tag) {
        throw 'Transfer tag files disagree'
    }
}
foreach ($name in $records.Keys) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $actualFiles[$name]).Hash.ToLowerInvariant()
    if ($actual -cne $records[$name]) { throw "Checksum mismatch: $name" }
}
Write-Output "Complete $kind bundle verified: $($records.Count) files. Copy the entire directory unchanged."
