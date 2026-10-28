$docPath = 'E:\contion simulation\李佳明_简历_Agent开发岗位_精排版.docx'
$pdfPath = 'E:\contion simulation\resume_layout_qa\preview.pdf'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {
    $doc = $word.Documents.Open($docPath, $false, $true)
    $doc.Repaginate()
    $doc.ExportAsFixedFormat($pdfPath, 17)
    $doc.Close($false)
} finally {
    if ($doc -ne $null) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
$poppler = 'C:\Users\27013\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
& (Join-Path $poppler 'pdfinfo.exe') $pdfPath | Select-String 'Pages|Page size'
& (Join-Path $poppler 'pdftoppm.exe') -png -r 150 $pdfPath 'E:\contion simulation\resume_layout_qa\page'
