set -euo pipefail
echo "*****"
mkdir -p pages

for file in md/*.md nbs/*.ipynb; do
  filename=$(basename "$file" | cut -f 1 -d '.')
  quarto render "$file" --to html --output-dir "./pages/" --output "$filename.html" --no-execute
  echo "processed $filename"
done

echo "*****"
