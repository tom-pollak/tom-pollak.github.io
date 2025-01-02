set -euo pipefail
echo "*****"
mkdir -p pages

for fp in md/*.md nbs/*.ipynb; do
  fp_no_extension=$(echo "$fp" | cut -f 1 -d '.')
  filename=$(basename "$fp_no_extension")
  quarto render "$fp" --to html --output-dir "./pages/"
  mv "./pages/$fp_no_extension.html" "./pages/$filename.html"
  echo "processed $filename"
done

echo "*****"
