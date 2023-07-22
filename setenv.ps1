$mypath = $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $mypath + $env:PYTHONPATH
