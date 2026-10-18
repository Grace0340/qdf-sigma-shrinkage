# Generates every number, table row and figure data file used in paper/main.tex
# from the frozen summary files. Run from any directory:
#   powershell -ExecutionPolicy Bypass -File paper\scripts\gen_numbers.ps1
# No training or model evaluation is performed; the script only reads JSON.

$ErrorActionPreference = "Stop"
$paper = Split-Path -Parent $PSScriptRoot
$res = Join-Path $paper "..\experiments\q1a_qdf_attribution\results"
$out = Join-Path $paper "generated"
New-Item -ItemType Directory -Force $out | Out-Null
$ic = [System.Globalization.CultureInfo]::InvariantCulture

function F([double]$x, [int]$d) { return $x.ToString("F$d", $ic) }
function Pct([double]$x, [int]$d) { return (100.0 * $x).ToString("F$d", $ic) }
function Load($name) { return Get-Content -Raw (Join-Path $res $name) | ConvertFrom-Json }
function Save($name, $text) {
    if ($name -like "*.tex") { $text = [regex]::Replace($text, '(?<![\w.])-(?=\d)', '\ensuremath{-}') }
    [System.IO.File]::WriteAllText((Join-Path $out $name), $text + "`n", (New-Object System.Text.UTF8Encoding($false)))
}
function Sci([double]$x) {
    $e = [math]::Floor([math]::Log10([math]::Abs($x)))
    return ("\ensuremath{{{0}\times10^{{{1}}}}}" -f ($x / [math]::Pow(10, $e)).ToString("F1", $ic), $e)
}

$v3 = Load "summary_v3_amend1.json"
$v4 = Load "summary_v4.json"
$v5 = Load "summary_v5.json"
$v6 = Load "summary_v6.json"
$v7 = Load "summary_v7.json"
$shape = Load "sigma_shape_v1.json"

# ---- all 17 cells in one list --------------------------------------------
$cells = @()
foreach ($c in $v4.cells) { $cells += [pscustomobject]@{ src = "v4"; bb = "TQNet"; name = $c.cell; nseed = 5; c = $c } }
foreach ($c in $v5.cells) { $cells += [pscustomobject]@{ src = "v5"; bb = "TQNet"; name = $c.cell; nseed = 3; c = $c } }
foreach ($c in $v7.cells) {
    $parts = $c.cell.Split("/")
    $cells += [pscustomobject]@{ src = "v7"; bb = $parts[0]; name = $parts[1]; nseed = 5; c = $c }
}

function PluginArms($c) {
    $arms = @()
    foreach ($p in $c.arms.PSObject.Properties) {
        if ($p.Name -like "PLUGIN_A*" -and $null -ne $p.Value.R) {
            $arms += [pscustomobject]@{ a = [double]$p.Name.Substring(8); R = [double]$p.Value.R; g = [double]$p.Value.g_mean }
        }
    }
    return $arms | Sort-Object a
}

# ---- Table I: analysis cells ----------------------------------------------
$rows = @()
$nVal80 = 0; $nBest80 = 0; $nStruct = 0; $nLabelSeeds = 0; $nLabelMean = 0
$bestMin = 9.0; $srMin = 9.0; $srMax = -9.0
$ana = $cells | Where-Object { $_.c.analysis }
foreach ($x in $ana) {
    $c = $x.c
    $val = $c.arms.PLUGIN_VAL; $cov = $c.arms.COVY_VAL; $sr = $c.arms.SPECTRUM_RANDOM
    $best = (PluginArms $c | Sort-Object R -Descending | Select-Object -First 1)
    $bbShort = @{ "TQNet" = "TQ"; "iTransformer" = "iT"; "DLinear" = "DL" }[$x.bb]
    $data = $x.name -replace "Weather", "Wth"
    if ($val.R -ge 0.8 -and $val.pos -eq $x.nseed) { $nVal80++ }
    if ($best.R -ge 0.8) { $nBest80++ }
    if ($best.R -lt $bestMin) { $bestMin = $best.R }
    if ($c.D_struct.pos -eq $x.nseed) { $nStruct++ }
    if ($c.D_label.mean -gt 0) { $nLabelMean++ }
    $req = if ($x.nseed -eq 5) { 4 } else { 3 }
    if ($c.D_label.mean -gt 0 -and $c.D_label.pos -ge $req) { $nLabelSeeds++ }
    if ($sr.R -lt $srMin) { $srMin = $sr.R }
    if ($sr.R -gt $srMax) { $srMax = $sr.R }
    $bestA = $best.a.ToString("G", $ic)
    $rows += ("{0} & {1} & {2} & {3}\,({4}/{5}) & {6}\,({7}) & {8} & {9} \\" -f `
        $bbShort, $data, (Pct $c.g_QDF_mean 2), (F $val.R 2), $val.pos, $x.nseed, (F $best.R 2), $bestA, (F $cov.R 2), (F $sr.R 2))
}
Save "table1_rows.tex" ($rows -join "`n")

# Table II (all settings, with confidence intervals) is written by stats_checks.py.

# ---- Figure data ----------------------------------------------------------
# (a) R(a) curves for the analysis cells, one file per cell
$i = 0
foreach ($x in $ana) {
    $i++
    $lines = @("a R")
    foreach ($p in (PluginArms $x.c)) { $lines += ("{0} {1}" -f $p.a.ToString("G", $ic), (F $p.R 4)) }
    Save ("dose_{0}.dat" -f $i) ($lines -join "`n")
}
# (b) g_QDF vs g_PLUGIN_VAL over all 17 cells (percent)
$sc = @("gqdf gval cls")
foreach ($x in $cells) {
    $bbs = @{ "TQNet" = "TQ"; "iTransformer" = "iT"; "DLinear" = "DL" }[$x.bb]
    $cls = $bbs + $(if ($x.c.analysis) { "a" } else { "n" })
    $sc += ("{0} {1} {2}" -f (Pct $x.c.g_QDF_mean 4), (Pct $x.c.arms.PLUGIN_VAL.g_mean 4), $cls)
}
Save "scatter.dat" ($sc -join "`n")

# ---- In-text numbers ------------------------------------------------------
$m = [ordered]@{}
$m["NCells"] = $cells.Count
$m["NAnalysis"] = @($ana).Count
$m["NValEighty"] = $nVal80
$m["NBestEighty"] = $nBest80
$m["BestMin"] = F $bestMin 2
$m["NStruct"] = $nStruct
$m["NLabelSeeds"] = $nLabelSeeds
$m["NLabelMean"] = $nLabelMean
$m["SRMin"] = F $srMin 2
$m["SRMax"] = F $srMax 2
$gq = @($ana | ForEach-Object { [double]$_.c.g_QDF_mean })
$m["GQDFMin"] = Pct ($gq | Measure-Object -Minimum).Minimum 1
$m["GQDFMax"] = Pct ($gq | Measure-Object -Maximum).Maximum 1

# cells where QDF hurts (negative mean gain)
$neg = @($cells | Where-Object { $_.c.g_QDF_mean -lt 0 })
$m["NNeg"] = $neg.Count
$m["NegQDFMin"] = Pct (($neg | ForEach-Object { [double]$_.c.g_QDF_mean }) | Measure-Object -Minimum).Minimum 2
$m["NegQDFMax"] = Pct (($neg | ForEach-Object { [double]$_.c.g_QDF_mean }) | Measure-Object -Maximum).Maximum 2
$m["NegValMin"] = Pct (($neg | ForEach-Object { [double]$_.c.arms.PLUGIN_VAL.g_mean }) | Measure-Object -Minimum).Minimum 2
$m["NegValMax"] = Pct (($neg | ForEach-Object { [double]$_.c.arms.PLUGIN_VAL.g_mean }) | Measure-Object -Maximum).Maximum 2

# square-root shape (I+S0n)^(1/2)-I versus S0n+Diag S0n (sigma_shape_v1, one free scalar each)
$sq = @($shape | Where-Object { $_.fit_sqrt_shape.rel_resid -lt $_.fit_S0n_plus_diag.rel_resid })
$m["NSqrtBetter"] = $sq.Count
$sqTags = @($sq | ForEach-Object { ($_.tag -split "_")[0..1] -join "-" } | Sort-Object -Unique)
$m["SqrtCells"] = ($sqTags -join ",")
$m["SqrtResMin"] = F (($sq | ForEach-Object { [double]$_.fit_sqrt_shape.rel_resid }) | Measure-Object -Minimum).Minimum 2
$m["SqrtResMax"] = F (($sq | ForEach-Object { [double]$_.fit_sqrt_shape.rel_resid }) | Measure-Object -Maximum).Maximum 2
$m["SqrtDiagResMin"] = F (($sq | ForEach-Object { [double]$_.fit_S0n_plus_diag.rel_resid }) | Measure-Object -Minimum).Minimum 2
$m["SqrtDiagResMax"] = F (($sq | ForEach-Object { [double]$_.fit_S0n_plus_diag.rel_resid }) | Measure-Object -Maximum).Maximum 2
$m["SqrtDevMin"] = F (($sq | ForEach-Object { [double]$_.rel_dev_from_I }) | Measure-Object -Minimum).Minimum 2
$others = @($shape | Where-Object { $_.fit_sqrt_shape.rel_resid -ge $_.fit_S0n_plus_diag.rel_resid })
$m["OtherDevMax"] = F (($others | ForEach-Object { [double]$_.rel_dev_from_I }) | Measure-Object -Maximum).Maximum 2

# ETTh1-720 (TQNet): learned Sigma close to I, yet per-seed QDF gains spread widely
$h1 = @($shape | Where-Object { $_.tag -like "ETTh1_720_*" })
$m["EhoneDevMax"] = F (($h1 | ForEach-Object { [double]$_.rel_dev_from_I }) | Measure-Object -Maximum).Maximum 2
$h1q = @(($v4.cells | Where-Object { $_.cell -eq "ETTh1-720" }).arms.QDF.g_per_seed | ForEach-Object { [double]$_ })
$m["EhoneQMin"] = Pct ($h1q | Measure-Object -Minimum).Minimum 1
$m["EhoneQMax"] = Pct ($h1q | Measure-Object -Maximum).Maximum 1

# run-to-run variation of exact repeats (GPU cells only)
$nus = @($cells | Where-Object { $null -ne $_.c.nu } | ForEach-Object { [double]$_.c.nu })
$m["NuMax"] = Pct ($nus | Measure-Object -Maximum).Maximum 2
$m["NGPUCells"] = $nus.Count

# sign agreement between QDF and the validated surrogate, all cells
$m["NSameSign"] = @($cells | Where-Object { [math]::Sign([double]$_.c.g_QDF_mean) -eq [math]::Sign([double]$_.c.arms.PLUGIN_VAL.g_mean) }).Count

# best grid intensity per analysis cell, split by horizon
$best96 = @(); $best720 = @()
foreach ($x in $ana) {
    $b = (PluginArms $x.c | Sort-Object R -Descending | Select-Object -First 1).a
    if ($x.name -like "*-96") { $best96 += $b } else { $best720 += $b }
}
$m["BestAShortMin"] = ($best96 | Measure-Object -Minimum).Minimum.ToString("G", $ic)
$m["BestAShortMax"] = ($best96 | Measure-Object -Maximum).Maximum.ToString("G", $ic)
$m["BestALongMin"] = ($best720 | Measure-Object -Minimum).Minimum.ToString("G", $ic)
$m["BestALongMax"] = ($best720 | Measure-Object -Maximum).Maximum.ToString("G", $ic)

# ETTm2-96 (TQNet): example of a validation miss
$e = ($cells | Where-Object { $_.src -eq "v5" -and $_.name -eq "ETTm2-96" }).c
$sel = @($e.arms.PLUGIN_VAL.selected_a | Sort-Object -Unique)
$m["MtwoSelA"] = ($sel -join ",")
$m["MtwoRVal"] = F $e.arms.PLUGIN_VAL.R 2
$eb = (PluginArms $e | Sort-Object R -Descending | Select-Object -First 1)
$m["MtwoBestA"] = $eb.a.ToString("G", $ic)
$m["MtwoRBest"] = F $eb.R 2

# DLinear cells
$dl = @($cells | Where-Object { $_.bb -eq "DLinear" })
$m["DLMin"] = Pct (($dl | ForEach-Object { [double]$_.c.g_QDF_mean }) | Measure-Object -Minimum).Minimum 2
$m["DLMax"] = Pct (($dl | ForEach-Object { [double]$_.c.g_QDF_mean }) | Measure-Object -Maximum).Maximum 2

# iTransformer/ECL-96 cosine between S0 and label covariance; range over analysis cells
$cosv = @($ana | ForEach-Object { [double]$_.c.descriptive.cos_S0_Cy })
$m["CosCyMin"] = F ($cosv | Measure-Object -Minimum).Minimum 2
$m["CosCyMax"] = F ($cosv | Measure-Object -Maximum).Maximum 2

# Ledoit-Wolf (v4 analysis cells only)
$lw = @($v4.cells | Where-Object { $_.analysis })
$m["LWRMin"] = F (($lw | ForEach-Object { [double]$_.arms.LW.R }) | Measure-Object -Minimum).Minimum 2
$m["LWRMax"] = F (($lw | ForEach-Object { [double]$_.arms.LW.R }) | Measure-Object -Maximum).Maximum 2
$deltas = @($lw | ForEach-Object { $_.descriptive.lw_delta } | ForEach-Object { [double]$_ })
$m["LWDeltaMin"] = Sci ($deltas | Measure-Object -Minimum).Minimum
$m["LWDeltaMax"] = Sci ($deltas | Measure-Object -Maximum).Maximum

# full whitening (v3 amend1 analysis cells)
$fg = @($v3.cells | Where-Object { $_.analysis })
$m["FGLSRMin"] = F (($fg | ForEach-Object { [double]$_.arms.FGLS_FULL.R }) | Measure-Object -Minimum).Minimum 2
$m["FGLSRMax"] = F (($fg | ForEach-Object { [double]$_.arms.FGLS_FULL.R }) | Measure-Object -Maximum).Maximum 2

# cost (v3 amend1, AutoDL T=96 cells)
$cst = @($v3.cells | Where-Object { $_.platform -eq "autodl" })
$m["SZeroSecMax"] = F (($cst | ForEach-Object { [double]$_.cost.S0_s }) | Measure-Object -Maximum).Maximum 1
$m["MetaSecMin"] = F (($cst | ForEach-Object { [double]$_.cost.QDF_meta_phase_s }) | Measure-Object -Minimum).Minimum 0
$m["MetaSecMax"] = F (($cst | ForEach-Object { [double]$_.cost.QDF_meta_phase_s }) | Measure-Object -Maximum).Maximum 0
$ratios = @($cst | ForEach-Object { [double]$_.cost.meta_free_procedure_s / [double]$_.cost.QDF_procedure_s })
$m["CostRatioMin"] = F ($ratios | Measure-Object -Minimum).Minimum 1
$m["CostRatioMax"] = F ($ratios | Measure-Object -Maximum).Maximum 1

# learned-Sigma shape: v6 (4 cells x 3 seeds) and sigma_shape_v1 (ETTh, 20 runs)
$cs0 = @(); $csd = @()
foreach ($c in $v6.cells) { foreach ($s in $c.shape) { $cs0 += [double]$s.cos_S0; $csd += [double]$s.cos_S0diag } }
$m["CosPlainMin"] = F ($cs0 | Measure-Object -Minimum).Minimum 2
$m["CosPlainMax"] = F ($cs0 | Measure-Object -Maximum).Maximum 2
$m["CosDiagMin"] = F ($csd | Measure-Object -Minimum).Minimum 2
$m["CosDiagMax"] = F ($csd | Measure-Object -Maximum).Maximum 2
$m["RDiagMin"] = F (($v6.cells | ForEach-Object { [double]$_.arms.PLUGIN_DIAG_MATCHED.R }) | Measure-Object -Minimum).Minimum 2
$m["RDiagMax"] = F (($v6.cells | ForEach-Object { [double]$_.arms.PLUGIN_DIAG_MATCHED.R }) | Measure-Object -Maximum).Maximum 2
$m["RPlainMin"] = F (($v6.cells | ForEach-Object { [double]$_.arms.PLUGIN_MATCHED.R }) | Measure-Object -Minimum).Minimum 2
$m["RPlainMax"] = F (($v6.cells | ForEach-Object { [double]$_.arms.PLUGIN_MATCHED.R }) | Measure-Object -Maximum).Maximum 2
$m["RelLearnedMax"] = Pct ([double]$v6.max_rel_learned) 3
$ratio = @($shape | ForEach-Object { [double]$_.diag_over_off })
$m["DiagOffMin"] = F ($ratio | Measure-Object -Minimum).Minimum 2
$m["DiagOffMax"] = F ($ratio | Measure-Object -Maximum).Maximum 2
$m["NShapeRuns"] = $shape.Count
$better = @($shape | Where-Object { $_.fit_S0n_plus_diag.rel_resid -lt $_.fit_S0n.rel_resid }).Count
$m["NShapeBetter"] = $better

$tex = @("% Auto-generated by paper/scripts/gen_numbers.ps1 -- do not edit by hand.")
foreach ($k in $m.Keys) { $tex += ("\newcommand{{\num{0}}}{{{1}}}" -f $k, $m[$k]) }
Save "numbers.tex" ($tex -join "`n")
Save "numbers.json" ($m | ConvertTo-Json)
"wrote $out"
$m.GetEnumerator() | ForEach-Object { "{0} = {1}" -f $_.Key, $_.Value }
""
$rows
