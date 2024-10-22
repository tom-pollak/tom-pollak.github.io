set -euo pipefail
echo "*****"
mkdir -p pages

for fp in md/*.md nbs/*.ipynb; do
  filename=$(basename "$fp" | cut -f 1 -d '.')
  quarto render "$fp" --to html --output-dir "./pages/" --no-execute

  

  mv "./pages/$(echo "$fp" | cut -f 1 -d '.').html" "./pages/$filename.html"
  echo "processed $filename"
done

echo "*****"
