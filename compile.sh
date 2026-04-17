set -euo pipefail
echo "*****"
mkdir -p pages

process_file() {
  local fp="$1"
  local fp_no_extension="${fp%.*}"
  local filename=$(basename "$fp_no_extension")
  quarto render "$fp" --to html --output-dir "./pages/"
  mv "./pages/$fp_no_extension.html" "./pages/$filename.html"
  echo "processed $filename"
}

if [ $# -eq 0 ]; then
  for fp in md/*.md nbs/*.ipynb; do
    process_file "$fp"
  done
else
  for fp in "$@"; do
    process_file "$fp"
  done
fi

echo "*****"
