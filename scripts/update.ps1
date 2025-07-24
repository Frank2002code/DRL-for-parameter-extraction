<#
.PARAMETER Message
  Commit 訊息，若沒給就用現在時間
#>
param(
  [string]$Message = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
)

git pull
git status
git add -A
git commit -m $Message -q 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "No changes to commit"
}
git push
